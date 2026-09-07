"""Tests for environment-backed API settings."""

import os
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

from perch_api.config import Settings


class SettingsTest(unittest.TestCase):
  def test_defaults_are_used_without_environment(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      self.assertEqual(Settings.from_env(), Settings())
      self.assertEqual(Settings().job_workers, 4)

  def test_environment_overrides_all_settings(self):
    environment = {
        "PERCH_API_MODEL": "test_model",
        "QDRANT_URL": "http://qdrant.test:6333",
        "QDRANT_API_KEY": "secret",
        "QDRANT_COLLECTION": "test_collection",
        "QDRANT_TIMEOUT_S": "12.5",
        "PERCH_API_UPSERT_BATCH_SIZE": "64",
        "PERCH_API_JOB_WORKERS": "3",
        "PERCH_API_JOB_DATABASE": "/tmp/jobs.sqlite3",
        "PERCH_API_JOB_MAX_ATTEMPTS": "5",
        "PERCH_API_JOB_LEASE_S": "60",
        "PERCH_API_JOB_RETRY_BACKOFF_S": "2.5",
    }

    with mock.patch.dict(os.environ, environment, clear=True):
      self.assertEqual(
          Settings.from_env(),
          Settings(
              model_name="test_model",
              qdrant_url="http://qdrant.test:6333",
              qdrant_api_key="secret",
              qdrant_collection="test_collection",
              qdrant_timeout_s=12.5,
              upsert_batch_size=64,
              job_workers=3,
                job_database_path="/tmp/jobs.sqlite3",
                job_max_attempts=5,
                job_lease_s=60.0,
                job_retry_backoff_s=2.5,
          ),
      )

  def test_invalid_numeric_environment_value_raises(self):
    with mock.patch.dict(
        os.environ, {"PERCH_API_UPSERT_BATCH_SIZE": "not-an-int"}, clear=True
    ):
      with self.assertRaises(ValueError):
        Settings.from_env()

  def test_settings_are_frozen(self):
    with self.assertRaises(FrozenInstanceError):
      Settings().model_name = "other_model"


if __name__ == "__main__":
  unittest.main()
