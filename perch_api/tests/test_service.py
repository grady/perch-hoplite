"""Tests for the embedding service orchestration."""

from contextlib import contextmanager
import unittest
from unittest import mock

import numpy as np

from perch_api.config import Settings
from perch_api.service import EmbeddingService
from perch_api.storage import S3ObjectRef


class EmbeddingServiceTest(unittest.TestCase):
  def setUp(self):
    self.storage = mock.MagicMock()
    self.pipeline = mock.MagicMock()
    self.vectors = mock.MagicMock()
    self.settings = Settings(upsert_batch_size=7)
    self.service = EmbeddingService(
        storage=self.storage,
        pipeline=self.pipeline,
        vectors=self.vectors,
        settings=self.settings,
    )
    self.ref = S3ObjectRef("audio", "bird.wav")
    self.audio = np.zeros(10, dtype=np.float32)

  def test_load_audio_stages_object_and_loads_path(self):
    staged_path = mock.sentinel.staged_path
    self.pipeline.load_audio.return_value = self.audio

    @contextmanager
    def staged(_ref):
      yield staged_path

    self.storage.staged.side_effect = staged

    result = self.service.load_audio(self.ref)

    self.assertIs(result, self.audio)
    self.storage.staged.assert_called_once_with(self.ref)
    self.pipeline.load_audio.assert_called_once_with(staged_path)

  def test_ingest_loaded_embeds_and_upserts(self):
    windows = [mock.sentinel.window]
    self.pipeline.model_name = "test_model"
    self.pipeline.embed_audio.return_value = windows
    self.vectors.upsert.return_value = 1

    result = self.service.ingest_loaded(self.ref, self.audio)

    self.assertEqual(result, 1)
    self.pipeline.embed_audio.assert_called_once_with(self.audio, self.ref)
    self.vectors.upsert.assert_called_once_with(
        self.ref, "test_model", windows, batch_size=7
    )

  def test_ingest_loads_audio_then_ingests_loaded_audio(self):
    staged_path = mock.sentinel.staged_path

    @contextmanager
    def staged(_ref):
      yield staged_path

    self.storage.staged.side_effect = staged
    self.pipeline.load_audio.return_value = self.audio
    with mock.patch.object(
        self.service, "ingest_loaded", return_value=3
    ) as ingest_loaded:
      result = self.service.ingest(self.ref)

    self.assertEqual(result, 3)
    self.storage.staged.assert_called_once_with(self.ref)
    self.pipeline.load_audio.assert_called_once_with(staged_path)
    ingest_loaded.assert_called_once_with(self.ref, self.audio)


if __name__ == "__main__":
  unittest.main()