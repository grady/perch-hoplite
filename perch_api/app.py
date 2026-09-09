"""FastAPI webhook application."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import asynccontextmanager
import hashlib
import logging
from threading import Event
from typing import Any, Callable
from urllib.parse import unquote_plus

from fastapi import FastAPI, HTTPException
import numpy as np

from perch_api.config import Settings
from perch_api.job_queue import Job, SQLiteJobQueue
from perch_api.service import EmbeddingService
from perch_api.storage import S3ObjectRef
from perch_api.vector_store import VectorWriter

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.INFO)


def _refs_from_event_type(
    event: dict[str, Any], event_prefix: str
) -> list[S3ObjectRef]:
  refs = []
  for record in event.get("Records", []):
    event_name = record.get("eventName", "")
    if not event_name.startswith(event_prefix):
      continue
    s3 = record.get("s3", {})
    bucket = s3.get("bucket", {}).get("name")
    obj = s3.get("object", {})
    key = obj.get("key")
    if bucket and key:
      refs.append(
          S3ObjectRef(
              bucket=bucket,
              key=unquote_plus(key),
              version_id=obj.get("versionId"),
              etag=obj.get("eTag"),
          )
      )
  return refs


def refs_from_event(event: dict[str, Any]) -> list[S3ObjectRef]:
  return _refs_from_event_type(event, "ObjectCreated:")


def removed_refs_from_event(event: dict[str, Any]) -> list[S3ObjectRef]:
  refs = _refs_from_event_type(event, "ObjectRemoved:")
  for ref in refs:
    _LOG.info("S3 object deletion received; removing vectors: %s", ref.uri)
  return refs


class JobQueue:
  def __init__(
      self,
      service_factory: Callable[[], EmbeddingService],
      workers: int = 1,
      database_path: str = "./perch_api_jobs.sqlite3",
      max_attempts: int = 3,
      lease_s: float = 300.0,
      retry_backoff_s: float = 1.0,
  ):
    self._service_factory = service_factory
    self._database_path = database_path
    self._max_attempts = max_attempts
    self._lease_s = lease_s
    self._retry_backoff_s = retry_backoff_s
    self._queue: SQLiteJobQueue | None = None
    self._stop = Event()
    self._workers = workers
    self._dispatcher: ThreadPoolExecutor | None = None
    self._loader: ThreadPoolExecutor | None = None
    self._service: EmbeddingService | None = None
    self._writer: VectorWriter | None = None

  @staticmethod
  def job_id(ref: S3ObjectRef) -> str:
    return hashlib.sha256(ref.identity.encode("utf-8")).hexdigest()

  def submit(self, ref: S3ObjectRef) -> str:
    if self._dispatcher is None:
      self.start()
    return self._queue.enqueue(ref, self._service.pipeline.model_name)

  def delete(self, ref: S3ObjectRef) -> None:
    if self._dispatcher is None:
      self.start()
    self._queue.tombstone(ref)
    self._writer.delete(ref, self._service.pipeline.model_name)

  def retry_failed(self) -> int:
    if self._dispatcher is None:
      self.start()
    return self._queue.retry_failed()

  def start(self) -> None:
    """Loads the embedding service and starts the worker thread."""
    if self._dispatcher is not None:
      return
    _LOG.info("Loading embedding service at startup")
    self._service = self._service_factory()
    self._queue = SQLiteJobQueue(
        self._database_path,
        max_attempts=self._max_attempts,
        lease_s=self._lease_s,
        retry_backoff_s=self._retry_backoff_s,
    )
    _LOG.info("Embedding service ready: model=%s", self._service.pipeline.model_name)
    sample_rate = self._service.pipeline.model.sample_rate
    _LOG.info("Warming embedding model with a dummy audio window")
    self._service.pipeline.embed_audio(
      np.zeros(sample_rate * 5, dtype=np.float32),
      S3ObjectRef(bucket="startup", key="warmup.wav"),
    )
    _LOG.info("Embedding model warmup complete")
    self._loader = ThreadPoolExecutor(max_workers=self._workers)
    self._writer = VectorWriter(
      self._service.vectors,
      batch_size=self._service.settings.upsert_batch_size,
      on_write=self._on_write,
      on_error=self._on_write_error,
    )
    self._dispatcher = ThreadPoolExecutor(max_workers=1)
    self._queue.recover_stale()
    self._dispatcher.submit(self._run)

  def _run(self) -> None:
    pending: dict[Any, Job] = {}
    while not self._stop.is_set() or pending:
      if not self._stop.is_set():
        for job in self._queue.claim(self._workers - len(pending)):
          pending[self._loader.submit(self._service.load_audio, job.ref)] = job
      if not pending:
        self._stop.wait(0.1)
        continue
      completed, _ = wait(pending, return_when=FIRST_COMPLETED)
      for future in completed:
        job = pending.pop(future)
        try:
          _LOG.info("Starting embedding job for %s", job.ref.identity)
          audio = future.result()
          if self._queue.is_tombstoned(job.ref):
            _LOG.info("Skipping tombstoned embedding job for %s", job.ref.identity)
            continue
          windows = self._service.pipeline.embed_audio(audio, job.ref)
          if not self._queue.is_tombstoned(job.ref):
            self._writer.submit(job.ref, job.model_name, windows)
        except Exception as exc:
          self._fail_job(job.ref, exc)

  def _fail_job(self, ref: S3ObjectRef, error: Exception) -> None:
    status = self._queue.fail(self.job_id(ref), str(error))
    _LOG.error(
        "Embedding job failed for %s; state=%s: %s",
        ref.identity,
        status,
        error,
    )

  def _on_write(self, ref: S3ObjectRef, count: int) -> None:
    self._queue.complete(self.job_id(ref))
    _LOG.info(
        "Embedding job completed for %s: upserted %d vectors", ref.identity, count
    )

  def _on_write_error(self, ref: S3ObjectRef, error: Exception) -> None:
    self._fail_job(ref, error)

  def close(self) -> None:
    self._stop.set()
    if self._dispatcher is not None:
      self._dispatcher.shutdown(wait=True)
    if self._loader is not None:
      self._loader.shutdown(wait=True)
    if self._writer is not None:
      try:
        self._writer.close()
      except Exception:
        _LOG.exception("Vector writer failed during API shutdown")

def create_app(service: EmbeddingService | None = None) -> FastAPI:
  settings = service.settings if service is not None else Settings.from_env()
  queue = JobQueue(
      service_factory=lambda: service or EmbeddingService.from_env(),
      workers=settings.job_workers,
      database_path=settings.job_database_path,
      max_attempts=settings.job_max_attempts,
      lease_s=settings.job_lease_s,
      retry_backoff_s=settings.job_retry_backoff_s,
  )
  @asynccontextmanager
  async def lifespan(_api: FastAPI):
    queue.start()
    try:
      yield
    finally:
      queue.close()

  api = FastAPI(title="Perch Embedding API", lifespan=lifespan)

  @api.get("/healthz")
  def healthz() -> dict[str, str]:
    return {"status": "ok"}

  @api.post("/webhooks/s3", status_code=202)
  def s3_webhook(event: dict[str, Any]) -> dict[str, list[str]]:
    refs = refs_from_event(event)
    removed_refs = removed_refs_from_event(event)
    if not refs and not removed_refs:
      raise HTTPException(status_code=400, detail="No S3 objects found")
    try:
      job_ids = [queue.submit(ref) for ref in refs]
      for ref in removed_refs:
        queue.delete(ref)
    except (OSError, RuntimeError, ValueError) as exc:
      raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"job_ids": job_ids}

  @api.post("/admin/jobs/retry-failed", status_code=202)
  def retry_failed_jobs() -> dict[str, int]:
    try:
      return {"requeued": queue.retry_failed()}
    except (OSError, RuntimeError, ValueError) as exc:
      raise HTTPException(status_code=503, detail=str(exc)) from exc

  return api


app = create_app()
