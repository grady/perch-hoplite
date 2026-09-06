"""Tests for webhook parsing and deterministic job IDs."""

import unittest

from perch_api.app import JobQueue, refs_from_event
from perch_api.storage import S3ObjectRef


class AppTest(unittest.TestCase):
  def test_refs_from_s3_event_decodes_key(self):
    refs = refs_from_event(
        {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "audio"},
                        "object": {"key": "folder%2Fbird.wav", "eTag": "abc"},
                    }
                }
            ]
        }
    )
    self.assertEqual(refs[0].bucket, "audio")
    self.assertEqual(refs[0].key, "folder/bird.wav")
    self.assertEqual(refs[0].etag, "abc")

  def test_job_id_is_stable_for_object_identity(self):
    first = S3ObjectRef("audio", "bird.wav", etag="abc")
    second = S3ObjectRef("audio", "bird.wav", etag="abc")
    self.assertEqual(JobQueue.job_id(first), JobQueue.job_id(second))
    self.assertNotEqual(
        JobQueue.job_id(first), JobQueue.job_id(S3ObjectRef("audio", "bird.wav"))
    )


if __name__ == "__main__":
  unittest.main()
