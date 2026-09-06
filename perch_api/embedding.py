"""Whole-object audio loading and batched model inference."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
from perch_hoplite import audio_io
from perch_hoplite.zoo import model_configs
from perch_hoplite.zoo import zoo_interface

from perch_api.storage import S3ObjectRef


_DATETIME_SUFFIX = re.compile(r"(?P<timestamp>\d{8}_\d{6})$")


@dataclass(frozen=True)
class EmbeddedWindow:
  vector: np.ndarray
  start_s: float
  end_s: float
  frame_index: int
  channel_index: int
  start_time: datetime.datetime | None = None
  end_time: datetime.datetime | None = None


def timestamp_from_filename(path: str | Path) -> datetime.datetime | None:
  """Extracts a recording timestamp using the TZ environment variable."""
  stem = Path(path).stem
  timezone = ZoneInfo(os.getenv("TZ") or "UTC")

  match = _DATETIME_SUFFIX.search(stem)
  if match:
    try:
      return datetime.datetime.strptime(
          match.group("timestamp"), "%Y%m%d_%H%M%S"
        ).replace(tzinfo=timezone)
    except ValueError:
      pass
  return None


class EmbeddingPipeline:
  def __init__(self, model_name: str = "perch_v2", model=None):
    self.model_name = model_name
    self.model = model or model_configs.load_model_by_name(model_name)

  def embed_file(self, path: Path, ref: S3ObjectRef) -> list[EmbeddedWindow]:
    return self.embed_audio(self.load_audio(path), ref)

  def load_audio(self, path: Path) -> np.ndarray:
    return np.asarray(
        audio_io.load_audio_file(
        path, target_sample_rate=self.model.sample_rate, dtype="float32"
        ),
        dtype=np.float32,
    )

  def embed_audio(
      self, audio: np.ndarray, ref: S3ObjectRef
  ) -> list[EmbeddedWindow]:
    outputs = self.model.embed(audio)
    if outputs.embeddings is None:
      raise ValueError(f"Model {self.model_name!r} did not return embeddings")
    embeddings = np.asarray(outputs.embeddings)
    if embeddings.ndim != 3:
      raise ValueError(
          f"Expected [frames, channels, features], got {embeddings.shape}"
      )
    window_size_s = float(getattr(self.model, "window_size_s", 5.0))
    hop_size_s = float(getattr(self.model, "hop_size_s", window_size_s))
    recording_time = timestamp_from_filename(ref.key)
    return [
        EmbeddedWindow(
            vector=np.asarray(vector, dtype=np.float32),
            start_s=frame_index * hop_size_s,
            end_s=frame_index * hop_size_s + window_size_s,
            frame_index=frame_index,
            channel_index=channel_index,
            start_time=(
                recording_time
                + datetime.timedelta(seconds=frame_index * hop_size_s)
                if recording_time is not None
                else None
            ),
            end_time=(
                recording_time
                + datetime.timedelta(
                    seconds=frame_index * hop_size_s + window_size_s
                )
                if recording_time is not None
                else None
            ),
        )
        for frame_index, frame in enumerate(embeddings)
        for channel_index, vector in enumerate(frame)
    ]
