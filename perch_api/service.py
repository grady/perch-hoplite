"""Application service shared by the webhook and backfill CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from perch_api.config import Settings
from perch_api.embedding import EmbeddingPipeline
from perch_api.storage import S3ObjectRef, S3Storage
from perch_api.vector_store import QdrantStore


@dataclass
class EmbeddingService:
  storage: S3Storage
  pipeline: EmbeddingPipeline
  vectors: QdrantStore
  settings: Settings

  @classmethod
  def from_env(cls) -> "EmbeddingService":
    settings = Settings.from_env()
    return cls(
        storage=S3Storage(),
      pipeline=EmbeddingPipeline(
        settings.model_name, inference_chunk_s=settings.inference_chunk_s
      ),
        vectors=QdrantStore(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=settings.qdrant_collection,
            timeout=settings.qdrant_timeout_s,
        ),
        settings=settings,
    )

  def ingest(self, ref: S3ObjectRef) -> int:
    with self.storage.staged(ref) as path:
      audio = self.pipeline.load_audio(path)
    return self.ingest_loaded(ref, audio)

  def load_audio(self, ref: S3ObjectRef) -> np.ndarray:
    with self.storage.staged(ref) as path:
      return self.pipeline.load_audio(path)

  def ingest_loaded(self, ref: S3ObjectRef, audio: np.ndarray) -> int:
    windows = self.pipeline.embed_audio(audio, ref)
    return self.vectors.upsert(
        ref,
        self.pipeline.model_name,
        windows,
        batch_size=self.settings.upsert_batch_size,
    )
