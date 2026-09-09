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
  inference_chunk_s: float = 60.0
  job_workers: int = 4
  job_database_path: str = "./perch_api_jobs.sqlite3"
  job_max_attempts: int = 3
  job_lease_s: float = 300.0
  job_retry_backoff_s: float = 1.0

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
        inference_chunk_s=float(
            os.getenv("PERCH_API_INFERENCE_CHUNK_S", str(cls.inference_chunk_s))
        ),
        job_workers=int(
            os.getenv("PERCH_API_JOB_WORKERS", str(cls.job_workers))
        ),
        job_database_path=os.getenv(
            "PERCH_API_JOB_DATABASE", cls.job_database_path
        ),
        job_max_attempts=int(
            os.getenv("PERCH_API_JOB_MAX_ATTEMPTS", str(cls.job_max_attempts))
        ),
        job_lease_s=float(
            os.getenv("PERCH_API_JOB_LEASE_S", str(cls.job_lease_s))
        ),
        job_retry_backoff_s=float(
            os.getenv(
                "PERCH_API_JOB_RETRY_BACKOFF_S", str(cls.job_retry_backoff_s)
            )
        ),
    )
