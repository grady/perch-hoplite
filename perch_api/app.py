"""FastAPI webhook application."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging
from queue import Empty, Full, Queue
from threading import Event
from typing import Any, Callable
from urllib.parse import unquote_plus

from fastapi import FastAPI, HTTPException

from perch_api.service import EmbeddingService
from perch_api.storage import S3ObjectRef

_LOG = logging.getLogger(__name__)


def refs_from_event(event: dict[str, Any]) -> list[S3ObjectRef]:
  refs = []
  for record in event.get("Records", []):
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


class JobQueue:
  def __init__(
      self,
      service_factory: Callable[[], EmbeddingService],
      maxsize: int = 32,
      workers: int = 1,
  ):
    self._service_factory = service_factory
    self._jobs: Queue[S3ObjectRef] = Queue(maxsize=maxsize)
    self._stop = Event()
    self._workers = workers
    self._executor: ThreadPoolExecutor | None = None

  @staticmethod
  def job_id(ref: S3ObjectRef) -> str:
    return hashlib.sha256(ref.identity.encode("utf-8")).hexdigest()

  def submit(self, ref: S3ObjectRef) -> str:
    if self._executor is None:
      self._executor = ThreadPoolExecutor(max_workers=self._workers)
      self._executor.submit(self._run)
    try:
      self._jobs.put_nowait(ref)
    except Full as exc:
      raise RuntimeError("Embedding job queue is full") from exc
    return self.job_id(ref)

  def _run(self) -> None:
    service = None
    while not self._stop.is_set():
      try:
        ref = self._jobs.get(timeout=0.1)
      except Empty:
        continue
      try:
        if service is None:
          service = self._service_factory()
        service.ingest(ref)
      except Exception:
        _LOG.exception("Embedding job failed for %s", ref.identity)
      finally:
        self._jobs.task_done()

  def close(self) -> None:
    self._stop.set()
    if self._executor is not None:
      self._executor.shutdown(wait=False, cancel_futures=True)


def create_app(service: EmbeddingService | None = None) -> FastAPI:
  settings = service.settings if service is not None else None
  queue = JobQueue(
      service_factory=lambda: service or EmbeddingService.from_env(),
      maxsize=settings.job_queue_size if settings else 32,
      workers=settings.job_workers if settings else 1,
  )
  api = FastAPI(title="Perch Embedding API")

  @api.get("/healthz")
  def healthz() -> dict[str, str]:
    return {"status": "ok"}

  @api.post("/webhooks/s3", status_code=202)
  def s3_webhook(event: dict[str, Any]) -> dict[str, list[str]]:
    refs = refs_from_event(event)
    if not refs:
      raise HTTPException(status_code=400, detail="No S3 objects found")
    try:
      job_ids = [queue.submit(ref) for ref in refs]
    except RuntimeError as exc:
      raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"job_ids": job_ids}

  return api


app = create_app()
