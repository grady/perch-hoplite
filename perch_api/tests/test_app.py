"""Tests for webhook parsing and deterministic job IDs."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from perch_api.app import JobQueue, refs_from_event, removed_refs_from_event
from perch_api.job_queue import SQLiteJobQueue
from perch_api.storage import S3ObjectRef


class AppTest(unittest.TestCase):
  def test_refs_from_s3_event_decodes_key(self):
    refs = refs_from_event(
        {
            "Records": [
                {
                    "eventName": "ObjectCreated:Put",
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
                {
                    "eventName": "ObjectCreated:Put",
                    "s3": {"bucket": {"name": "audio"}, "object": {}},
                },
                {
                    "eventName": "ObjectCreated:Put",
                    "s3": {"bucket": {}, "object": {"key": "bird.wav"}},
                },
                {
                    "eventName": "ObjectCreated:Put",
                    "s3": {
                        "bucket": {"name": "audio"},
                        "object": {"key": "ok.wav"},
                    },
                },
            ]
        }
    )

    self.assertEqual(refs, [S3ObjectRef("audio", "ok.wav")])

  def test_refs_from_event_decodes_plus_in_key_and_preserves_version(self):
    refs = refs_from_event(
        {
            "Records": [
                {
                    "eventName": "ObjectCreated:CompleteMultipartUpload",
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

  def test_removed_refs_from_event_logs_deleted_objects(self):
    with self.assertLogs("perch_api.app", level="INFO") as logs:
      refs = removed_refs_from_event(
          {
              "Records": [
                  {
                      "eventName": "ObjectRemoved:Delete",
                      "s3": {
                          "bucket": {"name": "audio"},
                          "object": {"key": "folder%2Fbird.wav"},
                      },
                  }
              ]
          }
      )

    self.assertEqual(refs, [S3ObjectRef("audio", "folder/bird.wav")])
    self.assertIn("S3 object deletion received; removing vectors", logs.output[0])

  def test_submit_is_not_limited_by_worker_buffer_size(self):
    service = SimpleNamespace(pipeline=SimpleNamespace(model_name="perch_v2"))
    with tempfile.TemporaryDirectory() as tempdir:
      queue = JobQueue(
          lambda: service,
          database_path=Path(tempdir) / "jobs.sqlite3",
      )
      queue._service = service
      queue._queue = SQLiteJobQueue(Path(tempdir) / "jobs.sqlite3")
      queue._dispatcher = mock.Mock()
      first = S3ObjectRef("audio", "one.wav")
      second = S3ObjectRef("audio", "two.wav")
      self.assertEqual(queue.submit(first), JobQueue.job_id(first))
      self.assertEqual(queue.submit(second), JobQueue.job_id(second))

    def test_delete_removes_vectors_for_object(self):
        service = SimpleNamespace(pipeline=SimpleNamespace(model_name="perch_v2"))
        queue = JobQueue(lambda: service)
        queue._service = service
        queue._dispatcher = mock.Mock()
        queue._queue = mock.Mock()
        queue._writer = mock.Mock()
        ref = S3ObjectRef("audio", "bird.wav", version_id="v1")

        queue.delete(ref)

        queue._queue.tombstone.assert_called_once_with(ref)
        queue._writer.delete.assert_called_once_with(ref, "perch_v2")

    def test_retry_failed_requeues_jobs(self):
        service = SimpleNamespace(pipeline=SimpleNamespace(model_name="perch_v2"))
        queue = JobQueue(lambda: service)
        queue._dispatcher = mock.Mock()
        queue._queue = mock.Mock()
        queue._queue.retry_failed.return_value = 7

        self.assertEqual(queue.retry_failed(), 7)
        queue._queue.retry_failed.assert_called_once_with()

  def test_close_waits_for_dispatcher_and_loader(self):
    queue = JobQueue(lambda: mock.sentinel.service)
    queue._dispatcher = mock.Mock()
    queue._loader = mock.Mock()
    queue.close()

    queue._dispatcher.shutdown.assert_called_once_with(wait=True)
    queue._loader.shutdown.assert_called_once_with(wait=True)


if __name__ == "__main__":
  unittest.main()
