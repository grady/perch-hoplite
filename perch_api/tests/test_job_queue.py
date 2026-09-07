"""Tests for durable embedding job state transitions."""

from pathlib import Path
import tempfile
import time
import unittest

from perch_api.job_queue import SQLiteJobQueue
from perch_api.storage import S3ObjectRef


class SQLiteJobQueueTest(unittest.TestCase):
  def setUp(self):
    self.tempdir = tempfile.TemporaryDirectory()
    self.path = Path(self.tempdir.name) / "jobs.sqlite3"
    self.ref = S3ObjectRef("audio", "bird.wav", etag="etag")

  def tearDown(self):
    self.tempdir.cleanup()

  def test_enqueue_is_durable_and_deduplicated(self):
    first = SQLiteJobQueue(self.path)
    job_id = first.enqueue(self.ref, "perch_v2")
    second = SQLiteJobQueue(self.path)

    self.assertEqual(second.enqueue(self.ref, "perch_v2"), job_id)
    job = second.get(job_id)
    self.assertIsNotNone(job)
    self.assertEqual(job.ref, self.ref)
    self.assertEqual(job.status, "PENDING")

  def test_claim_complete_and_retry(self):
    queue = SQLiteJobQueue(self.path, max_attempts=2, retry_backoff_s=0)
    job_id = queue.enqueue(self.ref, "perch_v2")
    now = time.time()

    claimed = queue.claim(1, now=now)
    self.assertEqual([job.job_id for job in claimed], [job_id])
    self.assertEqual(queue.fail(job_id, "temporary", now=now), "PENDING")
    self.assertEqual(queue.claim(1, now=now)[0].attempts, 2)
    self.assertEqual(queue.fail(job_id, "permanent", now=now), "FAILED")
    self.assertEqual(queue.get(job_id).status, "FAILED")

  def test_stale_processing_job_is_recovered_after_restart(self):
    queue = SQLiteJobQueue(self.path, lease_s=10)
    job_id = queue.enqueue(self.ref, "perch_v2")
    now = time.time()
    queue.claim(1, now=now)

    restarted = SQLiteJobQueue(self.path, lease_s=10)
    self.assertEqual(restarted.recover_stale(now=now + 11), 1)
    self.assertEqual(restarted.claim(1, now=now + 12)[0].job_id, job_id)


if __name__ == "__main__":
  unittest.main()