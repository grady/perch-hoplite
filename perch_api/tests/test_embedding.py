"""Tests for filename timestamp parsing and window timestamps."""

import datetime
import os
import pathlib
import unittest
from unittest import mock
import uuid

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


if __name__ == "__main__":
  unittest.main()