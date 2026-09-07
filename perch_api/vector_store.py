"""Qdrant persistence for embedding windows."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock
from concurrent.futures import ThreadPoolExecutor
import uuid
from typing import Callable, Iterable

import httpx
from qdrant_client import QdrantClient, models

from perch_api.embedding import EmbeddedWindow
from perch_api.storage import S3ObjectRef


@dataclass(frozen=True)
class VectorWrite:
  ref: S3ObjectRef
  model_name: str
  windows: tuple[EmbeddedWindow, ...]


class QdrantStore:
  def __init__(
      self,
      url: str,
      collection: str,
      api_key: str | None = None,
      timeout: float = 30.0,
      retries: int = 3,
      client=None,
  ):
    self.client = client or QdrantClient(url=url, api_key=api_key, timeout=timeout)
    self.collection = collection
    self._retries = retries
    self._collection_exists: bool | None = None
    self._collection_lock = Lock()

  def _request(self, operation):
    for attempt in range(self._retries + 1):
      try:
        return operation()
      except Exception as exc:
        if not self._is_transport_error(exc):
          raise
        if attempt >= self._retries:
          raise
        time.sleep(0.5 * (2**attempt))

  @staticmethod
  def _is_transport_error(error: Exception) -> bool:
    seen = set()
    current: Exception | None = error
    while current is not None and id(current) not in seen:
      seen.add(id(current))
      if isinstance(current, httpx.TransportError):
        return True
      current = current.__cause__ or current.__context__
    return False

  def ensure_collection(self, vector_size: int) -> None:
    if self._collection_is_available():
      return
    self._request(
        lambda: self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=vector_size, distance=models.Distance.DOT
            ),
        ),
    )
    self._collection_exists = True

  def _collection_is_available(self) -> bool:
    if self._collection_exists is not None:
      return self._collection_exists
    with self._collection_lock:
      if self._collection_exists is None:
        self._collection_exists = self._request(
            lambda: self.client.collection_exists(self.collection)
        )
    return self._collection_exists

  def has_vectors(self, ref: S3ObjectRef, model_name: str) -> bool:
    """Returns whether Qdrant already contains vectors for this object."""
    if not self._collection_is_available():
      return False
    conditions = self._identity_conditions(ref, model_name)
    conditions.append(
      models.FieldCondition(key="complete", match=models.MatchValue(value=True))
    )
    records, _ = self._request(
      lambda: self.client.scroll(
        collection_name=self.collection,
        scroll_filter=models.Filter(must=conditions),
        limit=1,
        with_payload=False,
        with_vectors=False,
      )
    )
    return bool(records)

  @staticmethod
  def _identity_conditions(ref: S3ObjectRef, model_name: str):
    conditions = [
        models.FieldCondition(
            key="source", match=models.MatchValue(value=ref.uri)
        ),
        models.FieldCondition(
            key="model", match=models.MatchValue(value=model_name)
        ),
    ]
    if ref.version_id is not None:
      conditions.append(
          models.FieldCondition(
              key="version_id", match=models.MatchValue(value=ref.version_id)
          )
      )
    elif ref.etag is not None:
      conditions.append(
          models.FieldCondition(
              key="etag", match=models.MatchValue(value=ref.etag)
          )
      )
    return conditions

  def upsert(
      self,
      ref: S3ObjectRef,
      model_name: str,
      windows: Iterable[EmbeddedWindow],
      batch_size: int = 256,
  ) -> int:
    windows = list(windows)
    if not windows:
      return 0
    self.ensure_collection(windows[0].vector.shape[-1])
    points = [
        models.PointStruct(
            id=self.point_id(ref, model_name, window),
            vector=window.vector.tolist(),
            payload={
                "source": ref.uri,
                "version_id": ref.version_id,
                "etag": ref.etag,
                "model": model_name,
                "complete": False,
                "frame_index": window.frame_index,
                "channel_index": window.channel_index,
                "start_s": window.start_s,
                "end_s": window.end_s,
                "start_time": (
                  window.start_time.isoformat()
                  if window.start_time is not None
                  else None
                ),
                "end_time": (
                  window.end_time.isoformat()
                  if window.end_time is not None
                  else None
                ),
            },
        )
        for window in windows
    ]
    for start in range(0, len(points), batch_size):
        self._request(
          lambda: self.client.upsert(
            collection_name=self.collection,
            points=points[start : start + batch_size],
            wait=True,
          )
      )
        self._request(
          lambda: self.client.set_payload(
            collection_name=self.collection,
            payload={"complete": True},
            points=models.Filter(
              must=self._identity_conditions(ref, model_name)
            ),
            wait=True,
          )
        )
    return len(points)

  @staticmethod
  def point_id(ref: S3ObjectRef, model_name: str, window: EmbeddedWindow) -> str:
    """Returns a deterministic UUID accepted by Qdrant as a point ID."""
    value = (
        f"{ref.identity}|{model_name}|{window.frame_index}|"
        f"{window.channel_index}"
    )
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest[:16]))


class VectorWriter:
  """Asynchronously writes completed embedding batches to Qdrant."""

  def __init__(
      self,
      store: QdrantStore,
      batch_size: int = 256,
      maxsize: int = 32,
      on_write: Callable[[S3ObjectRef, int], None] | None = None,
  ):
    self._store = store
    self._batch_size = batch_size
    self._on_write = on_write
    self._queue: Queue[VectorWrite] = Queue(maxsize=maxsize)
    self._stop = Event()
    self._executor: ThreadPoolExecutor | None = None
    self._error: Exception | None = None

  def submit(
      self,
      ref: S3ObjectRef,
      model_name: str,
      windows: Iterable[EmbeddedWindow],
  ) -> None:
    if self._executor is None:
      self._executor = ThreadPoolExecutor(max_workers=1)
      self._executor.submit(self._run)
    self._queue.put(VectorWrite(ref, model_name, tuple(windows)))

  def _run(self) -> None:
    while not self._stop.is_set() or not self._queue.empty():
      try:
        write = self._queue.get(timeout=0.1)
      except Empty:
        continue
      try:
        count = self._store.upsert(
            write.ref,
            write.model_name,
            write.windows,
            batch_size=self._batch_size,
        )
        if self._on_write is not None:
          self._on_write(write.ref, count)
      except Exception as exc:
        self._error = exc
        while True:
          try:
            self._queue.get_nowait()
          except Empty:
            break
          else:
            self._queue.task_done()
      finally:
        self._queue.task_done()

  def close(self) -> None:
    if self._executor is None:
      return
    self._queue.join()
    self._stop.set()
    self._executor.shutdown(wait=True)
    if self._error is not None:
      raise self._error

  def abort(self) -> None:
    """Stops accepting queued writes without waiting for the queue to drain."""
    if self._executor is None:
      return
    while True:
      try:
        self._queue.get_nowait()
      except Empty:
        break
      else:
        self._queue.task_done()
    self._stop.set()
    self._executor.shutdown(wait=False, cancel_futures=True)
