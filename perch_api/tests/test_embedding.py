"""Tests for filename timestamp parsing and window timestamps."""

import datetime
import os
import pathlib
import unittest
from unittest import mock
import uuid
from types import SimpleNamespace

import numpy as np

from perch_api.embedding import EmbeddingPipeline
from perch_api.embedding import timestamp_from_filename
from perch_api.embedding import EmbeddedWindow
from perch_api.storage import S3ObjectRef
from perch_api.vector_store import QdrantStore


class TimestampFromFilenameTest(unittest.TestCase):
  def test_parses_formatted_datetime(self):
    with mock.patch.dict(os.environ, {"TZ": "America/Los_Angeles"}):
      self.assertEqual(
          timestamp_from_filename(
              pathlib.Path("recording_20240102_030405.flac")
          ),
          datetime.datetime(
              2024,
              1,
              2,
              3,
              4,
              5,
              tzinfo=datetime.timezone(datetime.timedelta(hours=-8)),
          ),
      )

  def test_requires_datetime_at_end_of_stem(self):
    self.assertIsNone(timestamp_from_filename("20240102_030405-recording.flac"))

  def test_returns_none_without_timestamp(self):
    self.assertIsNone(timestamp_from_filename("bird-call.wav"))


class PointIdTest(unittest.TestCase):
  def test_point_id_is_a_stable_uuid(self):
    ref = S3ObjectRef("audio", "recording.flac", etag="etag")
    window = EmbeddedWindow(
        vector=[], start_s=0.0, end_s=5.0, frame_index=0, channel_index=0
    )

    point_id = QdrantStore.point_id(ref, "perch_v2", window)

    self.assertEqual(point_id, QdrantStore.point_id(ref, "perch_v2", window))
    self.assertIsInstance(uuid.UUID(point_id), uuid.UUID)


class EmbeddingPipelineTest(unittest.TestCase):
  def setUp(self):
    self.model = mock.MagicMock()
    self.model.sample_rate = 16000
    self.model.window_size_s = 4.0
    self.model.hop_size_s = 2.0
    self.pipeline = EmbeddingPipeline("test_model", model=self.model)

  def test_load_audio_returns_float32_audio_at_model_sample_rate(self):
    with mock.patch("perch_api.embedding.audio_io.load_audio_file") as load_audio:
      load_audio.return_value = [1, 2, 3]

      audio = self.pipeline.load_audio(pathlib.Path("bird.wav"))

    load_audio.assert_called_once_with(
        pathlib.Path("bird.wav"), target_sample_rate=16000, dtype="float32"
    )
    self.assertEqual(audio.dtype, np.float32)
    np.testing.assert_array_equal(audio, np.array([1, 2, 3], dtype=np.float32))

  def test_embed_audio_creates_windows_with_timing(self):
    self.model.embed.return_value = SimpleNamespace(
        embeddings=np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
    )
    ref = S3ObjectRef("audio", "recording_20240102_030405.wav")

    windows = self.pipeline.embed_audio(np.zeros(10), ref)

    self.assertEqual(len(windows), 4)
    self.assertEqual(
        [(window.frame_index, window.channel_index) for window in windows],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    )
    self.assertEqual((windows[0].start_s, windows[0].end_s), (0.0, 4.0))
    self.assertEqual((windows[2].start_s, windows[2].end_s), (2.0, 6.0))
    self.assertEqual(windows[1].start_time.hour, 3)
    self.assertEqual(
        windows[1].end_time,
        windows[1].start_time + datetime.timedelta(seconds=4),
    )
    np.testing.assert_array_equal(
        windows[3].vector, np.array([7.0, 8.0], dtype=np.float32)
    )

  def test_embed_audio_rejects_missing_embeddings(self):
    self.model.embed.return_value = SimpleNamespace(embeddings=None)

    with self.assertRaisesRegex(ValueError, "did not return embeddings"):
      self.pipeline.embed_audio(
          np.zeros(10), S3ObjectRef("audio", "bird.wav")
      )

  def test_embed_audio_rejects_non_three_dimensional_embeddings(self):
    self.model.embed.return_value = SimpleNamespace(
        embeddings=np.zeros((2, 3))
    )

    with self.assertRaisesRegex(
        ValueError, "Expected \\[frames, channels, features\\]"
    ):
      self.pipeline.embed_audio(
          np.zeros(10), S3ObjectRef("audio", "bird.wav")
      )


if __name__ == "__main__":
  unittest.main()