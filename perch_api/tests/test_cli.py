"""Tests for CLI webhook submission."""

import unittest
from unittest import mock

import click
import httpx
from click.testing import CliRunner

from perch_api.cli import _retry_delay, _s3_event, ingest, submit_webhook
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
            "eventName": "ObjectCreated:Put",
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

  def test_retry_delay_uses_numeric_retry_after(self):
    response = httpx.Response(503, headers={"Retry-After": "2.5"})

    self.assertEqual(_retry_delay(response, attempt=3, backoff=1.0), 2.5)

  def test_retry_delay_falls_back_to_exponential_backoff(self):
    response = httpx.Response(503, headers={"Retry-After": "invalid"})

    self.assertEqual(_retry_delay(response, attempt=3, backoff=1.5), 12.0)


class IngestCommandTest(unittest.TestCase):
  def setUp(self):
    self.runner = CliRunner()

  def test_requires_object_uri_or_bucket(self):
    result = self.runner.invoke(ingest, [])

    self.assertEqual(result.exit_code, 2)
    self.assertIn("Provide --object-uri or --bucket", result.output)

  def test_prefix_requires_bucket(self):
    result = self.runner.invoke(
        ingest, ["--object-uri", "s3://audio/bird.wav", "--prefix", "folder/"]
    )

    self.assertEqual(result.exit_code, 2)
    self.assertIn("--prefix requires --bucket", result.output)

  @mock.patch("perch_api.cli.S3Storage")
  def test_dry_run_filters_audio_suffixes(self, storage_class):
    storage_class.return_value.list.return_value = [
        S3ObjectRef("audio", "bird.WAV"),
        S3ObjectRef("audio", "notes.txt"),
    ]

    result = self.runner.invoke(
        ingest, ["--bucket", "audio", "--prefix", "recordings/", "--dry-run"]
    )

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(result.output.splitlines(), ["s3://audio/bird.WAV"])
    storage_class.return_value.list.assert_called_once_with("audio", "recordings/")


if __name__ == "__main__":
  unittest.main()
