"""FastAPI webhook application."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import asynccontextmanager
import hashlib
import logging
from queue import Empty, Full, Queue
from threading import Event
from typing import Any, Callable
from urllib.parse import unquote_plus

from fastapi import FastAPI, HTTPException
import numpy as np

from perch_api.service import EmbeddingService
from perch_api.storage import S3ObjectRef
from perch_api.vector_store import VectorWriter

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.INFO)


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
    try:
      self._jobs.put_nowait(ref)
    except Full as exc:
      raise RuntimeError("Embedding job queue is full") from exc
    return self.job_id(ref)

  def start(self) -> None:
    """Loads the embedding service and starts the worker thread."""
    if self._dispatcher is not None:
      return
    _LOG.info("Loading embedding service at startup")
    self._service = self._service_factory()
    _LOG.info("Embedding service ready: model=%s", self._service.pipeline.model_name)
    sample_rate = self._service.pipeline.model.sample_rate
    _LOG.info("Warming embedding model with a dummy audio window")
    self._service.pipeline.embed_audio(
      np.zeros(sample_rate * 5, dtype=np.float32),
      S3ObjectRef(bucket="startup", key="warmup.wav"),
    )
    _LOG.info("Embedding model warmup complete")
    self._writer = VectorWriter(
        self._service.vectors,
        batch_size=self._service.settings.upsert_batch_size,
        maxsize=self._service.settings.job_queue_size,
        on_write=lambda ref, count: _LOG.info(
            "Embedding job completed for %s: upserted %d vectors",
            ref.identity,
            count,
        ),
    )
    self._loader = ThreadPoolExecutor(max_workers=self._workers)
    self._dispatcher = ThreadPoolExecutor(max_workers=1)
    self._dispatcher.submit(self._run)

  def _run(self) -> None:
    pending = {}
    while not self._stop.is_set() or not self._jobs.empty() or pending:
      while len(pending) < self._workers:
        try:
          ref = self._jobs.get_nowait()
        except Empty:
          break
        pending[self._loader.submit(self._service.load_audio, ref)] = ref
      if not pending:
        self._stop.wait(0.1)
        continue
      completed, _ = wait(pending, return_when=FIRST_COMPLETED)
      for future in completed:
        ref = pending.pop(future)
        try:
          _LOG.info("Starting embedding job for %s", ref.identity)
          windows = self._service.pipeline.embed_audio(future.result(), ref)
          self._writer.submit(ref, self._service.pipeline.model_name, windows)
        except Exception:
          _LOG.exception("Embedding job failed for %s", ref.identity)
        finally:
          self._jobs.task_done()

  def close(self) -> None:
    self._stop.set()
    if self._dispatcher is not None:
      self._dispatcher.shutdown(wait=True)
    if self._loader is not None:
      self._loader.shutdown(wait=True)
    if self._writer is not None:
      self._writer.close()


def create_app(service: EmbeddingService | None = None) -> FastAPI:
  settings = service.settings if service is not None else None
  queue = JobQueue(
      service_factory=lambda: service or EmbeddingService.from_env(),
      maxsize=settings.job_queue_size if settings else 32,
      workers=settings.job_workers if settings else 1,
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
    if not refs:
      raise HTTPException(status_code=400, detail="No S3 objects found")
    try:
      job_ids = [queue.submit(ref) for ref in refs]
    except RuntimeError as exc:
      raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"job_ids": job_ids}

  return api


app = create_app()
