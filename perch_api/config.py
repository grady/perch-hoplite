"""Environment-backed service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
  model_name: str = "perch_v2"
  qdrant_url: str = "http://localhost:6333"
  qdrant_api_key: str | None = None
  qdrant_collection: str = "perch_embeddings"
  qdrant_timeout_s: float = 30.0
  upsert_batch_size: int = 256
  job_queue_size: int = 32
  job_workers: int = 1

  @classmethod
  def from_env(cls) -> "Settings":
    return cls(
        model_name=os.getenv("PERCH_API_MODEL", cls.model_name),
        qdrant_url=os.getenv("QDRANT_URL", cls.qdrant_url),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        qdrant_collection=os.getenv(
            "QDRANT_COLLECTION", cls.qdrant_collection
        ),
        qdrant_timeout_s=float(
            os.getenv("QDRANT_TIMEOUT_S", str(cls.qdrant_timeout_s))
        ),
        upsert_batch_size=int(
            os.getenv("PERCH_API_UPSERT_BATCH_SIZE", str(cls.upsert_batch_size))
        ),
        job_queue_size=int(
            os.getenv("PERCH_API_JOB_QUEUE_SIZE", str(cls.job_queue_size))
        ),
        job_workers=int(
            os.getenv("PERCH_API_JOB_WORKERS", str(cls.job_workers))
        ),
    )
