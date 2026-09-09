"""Durable SQLite storage for embedding jobs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sqlite3
import time
from pathlib import Path

from perch_api.storage import S3ObjectRef


@dataclass(frozen=True)
class Job:
  job_id: str
  ref: S3ObjectRef
  model_name: str
  status: str
  attempts: int
  error: str | None


class SQLiteJobQueue:
  """A durable, single-process queue with at-least-once delivery."""

  def __init__(
      self,
      path: str | Path,
      max_attempts: int = 3,
      lease_s: float = 300.0,
      retry_backoff_s: float = 1.0,
  ):
    if max_attempts < 1:
      raise ValueError("max_attempts must be positive")
    self._path = str(path)
    self._max_attempts = max_attempts
    self._lease_s = lease_s
    self._retry_backoff_s = retry_backoff_s
    if self._path != ":memory:":
      Path(self._path).parent.mkdir(parents=True, exist_ok=True)
    self._initialize()

  def _connect(self) -> sqlite3.Connection:
    connection = sqlite3.connect(self._path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection

  def _initialize(self) -> None:
    with self._connect() as connection:
      connection.execute("PRAGMA journal_mode = WAL")
      connection.execute(
          """
          CREATE TABLE IF NOT EXISTS embed_jobs (
            job_id TEXT PRIMARY KEY,
            identity TEXT NOT NULL,
            bucket TEXT NOT NULL,
            object_key TEXT NOT NULL,
            version_id TEXT,
            etag TEXT,
            model_name TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN
              ('PENDING', 'PROCESSING', 'COMPLETED', 'FAILED')),
            attempts INTEGER NOT NULL DEFAULT 0,
            available_at REAL NOT NULL,
            lease_until REAL,
            error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            completed_at REAL,
            cancelled INTEGER NOT NULL DEFAULT 0,
            UNIQUE(identity, model_name)
          )
          """
      )
      connection.execute(
          "CREATE INDEX IF NOT EXISTS embed_jobs_ready_idx "
          "ON embed_jobs(status, available_at)"
      )
      connection.execute(
          """
          CREATE TABLE IF NOT EXISTS embed_tombstones (
            uri TEXT PRIMARY KEY,
            created_at REAL NOT NULL
          )
          """
      )
      columns = {
          row["name"]
          for row in connection.execute("PRAGMA table_info(embed_jobs)")
      }
      if "cancelled" not in columns:
        connection.execute(
            "ALTER TABLE embed_jobs ADD COLUMN cancelled INTEGER NOT NULL DEFAULT 0"
        )

  @staticmethod
  def _job_id(ref: S3ObjectRef, model_name: str) -> str:
    del model_name
    return hashlib.sha256(ref.identity.encode("utf-8")).hexdigest()

  @staticmethod
  def _job_from_row(row: sqlite3.Row) -> Job:
    return Job(
        job_id=row["job_id"],
        ref=S3ObjectRef(
            bucket=row["bucket"],
            key=row["object_key"],
            version_id=row["version_id"],
            etag=row["etag"],
        ),
        model_name=row["model_name"],
        status=row["status"],
        attempts=row["attempts"],
        error=row["error"],
    )

  def enqueue(self, ref: S3ObjectRef, model_name: str) -> str:
    now = time.time()
    job_id = self._job_id(ref, model_name)
    with self._connect() as connection:
      connection.execute("DELETE FROM embed_tombstones WHERE uri = ?", (ref.uri,))
      connection.execute(
          """
          INSERT INTO embed_jobs (
            job_id, identity, bucket, object_key, version_id, etag, model_name,
            status, available_at, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?)
          ON CONFLICT(identity, model_name) DO UPDATE SET
            status = 'PENDING', available_at = excluded.available_at,
            updated_at = excluded.updated_at, error = NULL, cancelled = 0
          WHERE embed_jobs.cancelled = 1
          """,
          (
              job_id,
              ref.identity,
              ref.bucket,
              ref.key,
              ref.version_id,
              ref.etag,
              model_name,
              now,
              now,
              now,
          ),
      )
    return job_id

  def recover_stale(self, now: float | None = None) -> int:
    now = time.time() if now is None else now
    with self._connect() as connection:
      result = connection.execute(
          """
          UPDATE embed_jobs
          SET status = 'PENDING', lease_until = NULL, available_at = ?,
              updated_at = ?, error = 'Recovered after interrupted processing'
          WHERE status = 'PROCESSING' AND lease_until <= ?
          """,
          (now, now, now),
      )
      return result.rowcount

  def retry_failed(self, now: float | None = None) -> int:
    now = time.time() if now is None else now
    with self._connect() as connection:
      result = connection.execute(
          """
          UPDATE embed_jobs
          SET status = 'PENDING', attempts = 0, available_at = ?,
              lease_until = NULL, updated_at = ?, error = NULL
          WHERE status = 'FAILED' AND cancelled = 0
            AND NOT EXISTS (
              SELECT 1 FROM embed_tombstones AS tombstone
              WHERE tombstone.uri =
                's3://' || embed_jobs.bucket || '/' || embed_jobs.object_key
            )
          """,
          (now, now),
      )
      return result.rowcount

  def claim(self, limit: int, now: float | None = None) -> list[Job]:
    if limit < 1:
      return []
    now = time.time() if now is None else now
    claimed: list[Job] = []
    with self._connect() as connection:
      connection.execute("BEGIN IMMEDIATE")
      rows = connection.execute(
          """
          SELECT * FROM embed_jobs
          WHERE status = 'PENDING' AND available_at <= ?
            AND cancelled = 0
            AND NOT EXISTS (
              SELECT 1 FROM embed_tombstones AS tombstone
              WHERE tombstone.uri =
                's3://' || embed_jobs.bucket || '/' || embed_jobs.object_key
            )
          ORDER BY created_at, job_id
          LIMIT ?
          """,
          (now, limit),
      ).fetchall()
      lease_until = now + self._lease_s
      for row in rows:
        connection.execute(
            """
            UPDATE embed_jobs
            SET status = 'PROCESSING', attempts = attempts + 1,
                lease_until = ?, updated_at = ?, error = NULL
            WHERE job_id = ? AND status = 'PENDING'
            """,
            (lease_until, now, row["job_id"]),
        )
        claimed.append(
            Job(
                job_id=row["job_id"],
                ref=S3ObjectRef(
                    bucket=row["bucket"],
                    key=row["object_key"],
                    version_id=row["version_id"],
                    etag=row["etag"],
                ),
                model_name=row["model_name"],
                status="PROCESSING",
                attempts=row["attempts"] + 1,
                error=None,
            )
        )
    return claimed

  def tombstone(self, ref: S3ObjectRef, now: float | None = None) -> int:
    """Prevents queued jobs for an object URI from producing vectors."""
    now = time.time() if now is None else now
    with self._connect() as connection:
      connection.execute(
          "INSERT OR REPLACE INTO embed_tombstones (uri, created_at) VALUES (?, ?)",
          (ref.uri, now),
      )
      result = connection.execute(
          """
          UPDATE embed_jobs
          SET cancelled = 1, updated_at = ?, error = 'Cancelled after object removal'
          WHERE bucket = ? AND object_key = ? AND cancelled = 0
          """,
          (now, ref.bucket, ref.key),
      )
      return result.rowcount

  def is_tombstoned(self, ref: S3ObjectRef) -> bool:
    with self._connect() as connection:
      row = connection.execute(
          "SELECT 1 FROM embed_tombstones WHERE uri = ?", (ref.uri,)
      ).fetchone()
    return row is not None

  def complete(self, job_id: str, now: float | None = None) -> None:
    now = time.time() if now is None else now
    with self._connect() as connection:
      connection.execute(
          """
          UPDATE embed_jobs
          SET status = 'COMPLETED', lease_until = NULL, completed_at = ?,
              updated_at = ?, error = NULL
          WHERE job_id = ? AND status = 'PROCESSING'
          """,
          (now, now, job_id),
      )

  def fail(self, job_id: str, error: str, now: float | None = None) -> str:
    now = time.time() if now is None else now
    with self._connect() as connection:
      row = connection.execute(
          "SELECT attempts FROM embed_jobs WHERE job_id = ?", (job_id,)
      ).fetchone()
      if row is None:
        raise KeyError(job_id)
      if row["attempts"] >= self._max_attempts:
        status = "FAILED"
        available_at = now
      else:
        status = "PENDING"
        available_at = now + self._retry_backoff_s * 2 ** (row["attempts"] - 1)
      connection.execute(
          """
          UPDATE embed_jobs
          SET status = ?, lease_until = NULL, available_at = ?, updated_at = ?,
              error = ?
          WHERE job_id = ? AND status = 'PROCESSING'
          """,
          (status, available_at, now, error, job_id),
      )
      return status

  def get(self, job_id: str) -> Job | None:
    with self._connect() as connection:
      row = connection.execute(
          "SELECT * FROM embed_jobs WHERE job_id = ?", (job_id,)
      ).fetchone()
    return None if row is None else self._job_from_row(row)
