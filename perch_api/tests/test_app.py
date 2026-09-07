"""Tests for webhook parsing and deterministic job IDs."""

import unittest
from unittest import mock

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

  def test_refs_from_event_skips_records_without_bucket_or_key(self):
    refs = refs_from_event(
        {
            "Records": [
                {"s3": {"bucket": {"name": "audio"}, "object": {}}},
                {"s3": {"bucket": {}, "object": {"key": "bird.wav"}}},
                {"s3": {"bucket": {"name": "audio"}, "object": {"key": "ok.wav"}}},
            ]
        }
    )

    self.assertEqual(refs, [S3ObjectRef("audio", "ok.wav")])

  def test_refs_from_event_decodes_plus_in_key_and_preserves_version(self):
    refs = refs_from_event(
        {
            "Records": [
                {
                    "s3": {
                        "bucket": {"name": "audio"},
                        "object": {
                            "key": "folder%2Fbird+call.wav",
                            "versionId": "v1",
                        },
                    }
                }
            ]
        }
    )

    self.assertEqual(
        refs, [S3ObjectRef("audio", "folder/bird call.wav", version_id="v1")]
    )

  def test_submit_raises_when_queue_is_full(self):
    queue = JobQueue(lambda: mock.sentinel.service, maxsize=1)
    with mock.patch.object(queue, "start"):
      queue.submit(S3ObjectRef("audio", "one.wav"))
      with self.assertRaisesRegex(RuntimeError, "queue is full"):
        queue.submit(S3ObjectRef("audio", "two.wav"))

    def test_close_logs_and_contains_writer_failure(self):
        queue = JobQueue(lambda: mock.sentinel.service)
        queue._dispatcher = mock.Mock()
        queue._loader = mock.Mock()
        queue._writer = mock.Mock()
        queue._writer.close.side_effect = RuntimeError("qdrant disconnected")

        with self.assertLogs("perch_api.app", level="ERROR") as logs:
            queue.close()

        queue._dispatcher.shutdown.assert_called_once_with(wait=True)
        queue._loader.shutdown.assert_called_once_with(wait=True)
        queue._writer.close.assert_called_once_with()
        self.assertIn("Vector writer failed during API shutdown", logs.output[0])
        self.assertIn("qdrant disconnected", "\n".join(logs.output))


if __name__ == "__main__":
  unittest.main()
