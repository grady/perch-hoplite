"""Tests for CLI webhook submission."""

import unittest
from unittest import mock

import click
import httpx

from perch_api.cli import _s3_event, submit_webhook
from perch_api.storage import S3ObjectRef


class WebhookSubmissionTest(unittest.TestCase):
  def test_s3_event_preserves_object_metadata(self):
    ref = S3ObjectRef(
        "audio", "folder/bird.wav", version_id="version", etag="etag"
    )

    self.assertEqual(
        _s3_event(ref),
        {
            "Records": [{
                "s3": {
                    "bucket": {"name": "audio"},
                    "object": {
                        "key": "folder/bird.wav",
                        "eTag": "etag",
                        "versionId": "version",
                    },
                }
            }]
        },
    )

  @mock.patch("perch_api.cli.time.sleep")
  @mock.patch("perch_api.cli.httpx.post")
  def test_retries_503_then_returns_job_id(self, post, sleep):
    post.side_effect = [
        httpx.Response(503, text="busy"),
        httpx.Response(202, json={"job_ids": ["job-123"]}),
    ]

    job_id = submit_webhook(
        S3ObjectRef("audio", "bird.wav"),
        "http://api/webhooks/s3",
        retries=2,
        backoff=1.0,
    )

    self.assertEqual(job_id, "job-123")
    sleep.assert_called_once_with(1.0)
    self.assertEqual(post.call_count, 2)

  @mock.patch("perch_api.cli.time.sleep")
  @mock.patch("perch_api.cli.httpx.post")
  def test_stops_after_503_retries(self, post, sleep):
    post.return_value = httpx.Response(503, text="busy")

    with self.assertRaises(click.ClickException):
      submit_webhook(
          S3ObjectRef("audio", "bird.wav"),
          "http://api/webhooks/s3",
          retries=2,
          backoff=1.0,
      )

    self.assertEqual(post.call_count, 3)
    self.assertEqual(sleep.call_count, 2)

  @mock.patch("perch_api.cli.httpx.post")
  def test_does_not_retry_non_503_response(self, post):
    post.return_value = httpx.Response(400, text="bad request")

    with self.assertRaises(click.ClickException):
      submit_webhook(
          S3ObjectRef("audio", "bird.wav"),
          "http://api/webhooks/s3",
          retries=5,
          backoff=1.0,
      )

    post.assert_called_once()


if __name__ == "__main__":
  unittest.main()
