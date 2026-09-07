"""Tests for S3 references and staged object access."""

from pathlib import Path
import os
import unittest
from unittest import mock

from perch_api.storage import S3ObjectRef, S3Storage


class S3ObjectRefTest(unittest.TestCase):
  def test_from_uri_parses_bucket_and_key(self):
    ref = S3ObjectRef.from_uri("s3://audio/folder/bird.wav")

    self.assertEqual(ref, S3ObjectRef("audio", "folder/bird.wav"))
    self.assertEqual(ref.uri, "s3://audio/folder/bird.wav")

  def test_identity_prefers_version_then_etag_then_latest(self):
    self.assertEqual(
        S3ObjectRef("audio", "bird.wav", version_id="v1", etag="e1").identity,
        "s3://audio/bird.wav@v1",
    )
    self.assertEqual(
        S3ObjectRef("audio", "bird.wav", etag="e1").identity,
        "s3://audio/bird.wav@e1",
    )
    self.assertEqual(
        S3ObjectRef("audio", "bird.wav").identity,
        "s3://audio/bird.wav@latest",
    )

  def test_from_uri_rejects_non_s3_or_empty_keys(self):
    for uri in ("audio/bird.wav", "http://audio/bird.wav", "s3://audio/"):
      with self.subTest(uri=uri):
        with self.assertRaises(ValueError):
          S3ObjectRef.from_uri(uri)


class S3StorageTest(unittest.TestCase):
  def setUp(self):
    self.client = mock.MagicMock()
    self.storage = S3Storage(client=self.client)

  def test_list_returns_objects_from_all_pages(self):
    paginator = self.client.get_paginator.return_value
    paginator.paginate.return_value = [
        {"Contents": [{"Key": "one.wav", "ETag": "etag-1"}]},
        {"Contents": [],},
        {"Contents": [{"Key": "two.flac"}]},
    ]

    refs = list(self.storage.list("audio", "recordings/"))

    self.assertEqual(
        refs,
        [
            S3ObjectRef("audio", "one.wav", etag="etag-1"),
            S3ObjectRef("audio", "two.flac"),
        ],
    )
    paginator.paginate.assert_called_once_with(
        Bucket="audio", Prefix="recordings/"
    )

  def test_staged_downloads_with_suffix_and_cleans_up(self):
    ref = S3ObjectRef("audio", "folder/bird.wav")
    staged_path = None

    def download_file(bucket, key, filename):
      self.assertEqual((bucket, key), ("audio", "folder/bird.wav"))
      Path(filename).write_bytes(b"audio")

    self.client.download_file.side_effect = download_file
    with self.storage.staged(ref) as path:
      staged_path = path
      self.assertEqual(path.suffix, ".wav")
      self.assertEqual(path.read_bytes(), b"audio")
      self.assertTrue(path.exists())

    self.assertFalse(staged_path.exists())

  def test_staged_cleans_up_when_download_fails(self):
    self.client.download_file.side_effect = OSError("download failed")

    with self.assertRaisesRegex(OSError, "download failed"):
      with self.storage.staged(S3ObjectRef("audio", "bird.mp3")):
        self.fail("download failure should prevent yielding")

    filename = self.client.download_file.call_args.args[2]
    self.assertFalse(os.path.exists(filename))

  def test_default_client_uses_s3_endpoint_environment(self):
    with mock.patch.dict(os.environ, {"AWS_ENDPOINT_URL_S3": "http://s3.test"}):
      with mock.patch("perch_api.storage.boto3.client") as client:
        storage = S3Storage()

    client.assert_called_once_with("s3", endpoint_url="http://s3.test")
    self.assertIs(storage.client, client.return_value)


if __name__ == "__main__":
  unittest.main()
