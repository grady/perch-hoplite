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
  def __init__(
      self, model_name: str = "perch_v2", model=None, inference_chunk_s: float = 60.0
  ):
    if inference_chunk_s <= 0:
      raise ValueError("inference_chunk_s must be positive")
    self.model_name = model_name
    self.model = model or model_configs.load_model_by_name(model_name)
    self.inference_chunk_s = inference_chunk_s

  def embed_file(self, path: Path, ref: S3ObjectRef) -> list[EmbeddedWindow]:
    return self.embed_audio(self.load_audio(path), ref)

  def load_audio(self, path: Path) -> np.ndarray:
    audio = np.asarray(
        audio_io.load_audio_file(
        path, target_sample_rate=self.model.sample_rate, dtype="float32"
        ),
        dtype=np.float32,
    )
    if audio.ndim == 2:
      audio = audio[:, 0]
    return audio

  def embed_audio(
      self, audio: np.ndarray, ref: S3ObjectRef
  ) -> list[EmbeddedWindow]:
    window_size_s = float(getattr(self.model, "window_size_s", 5.0))
    hop_size_s = float(getattr(self.model, "hop_size_s", window_size_s))
    sample_rate = self.model.sample_rate
    chunk_samples = max(1, int(self.inference_chunk_s * sample_rate))
    overlap_samples = max(0, int((window_size_s - hop_size_s) * sample_rate))
    step_samples = max(1, chunk_samples - overlap_samples)
    hop_samples = max(1, int(hop_size_s * sample_rate))
    frame_embeddings: dict[int, np.ndarray] = {}
    for start in range(0, len(audio), step_samples):
      chunk = audio[start : start + chunk_samples]
      outputs = self.model.embed(chunk)
      if outputs.embeddings is None:
        raise ValueError(f"Model {self.model_name!r} did not return embeddings")
      embeddings = np.asarray(outputs.embeddings)
      if embeddings.ndim != 3:
        raise ValueError(
            f"Expected [frames, channels, features], got {embeddings.shape}"
        )
      first_frame = round(start / hop_samples)
      for frame_index, frame in enumerate(embeddings):
        frame_embeddings.setdefault(first_frame + frame_index, frame)
    embeddings_by_frame = sorted(frame_embeddings.items())
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
        for frame_index, frame in embeddings_by_frame
        for channel_index, vector in enumerate(frame)
    ]
