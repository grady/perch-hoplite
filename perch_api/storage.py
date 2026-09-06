"""S3-compatible object access."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Iterator, Generator
from urllib.parse import urlparse

import boto3


@dataclass(frozen=True)
class S3ObjectRef:
  bucket: str
  key: str
  version_id: str | None = None
  etag: str | None = None

  @classmethod
  def from_uri(cls, uri: str) -> "S3ObjectRef":
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path[1:]:
      raise ValueError(f"Expected s3://bucket/key, got {uri!r}")
    return cls(parsed.netloc, parsed.path[1:])

  @property
  def uri(self) -> str:
    return f"s3://{self.bucket}/{self.key}"

  @property
  def identity(self) -> str:
    return f"{self.uri}@{self.version_id or self.etag or 'latest'}"


class S3Storage:
  """Materializes S3 objects using boto3's standard credential chain."""

  def __init__(self, client=None):
    if client is None:
      endpoint_url = os.getenv("AWS_ENDPOINT_URL_S3")
      client = boto3.client("s3", endpoint_url=endpoint_url)
    self.client = client

  def list(self, bucket: str, prefix: str = "") -> Iterator[S3ObjectRef]:
    paginator = self.client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
      for item in page.get("Contents", []):
        yield S3ObjectRef(
            bucket=bucket,
            key=item["Key"],
            etag=item.get("ETag"),
        )

  @contextmanager
  def staged(self, ref: S3ObjectRef) -> Generator[Path, None, None]:
    suffix = Path(ref.key).suffix
    fd, filename = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    path = Path(filename)
    try:
      self.client.download_file(ref.bucket, ref.key, str(path))
      yield path
    finally:
      path.unlink(missing_ok=True)
