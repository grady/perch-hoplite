"""CLI for backfilling existing S3 audio objects."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone

import click
import httpx
from tqdm import tqdm

from perch_api.service import EmbeddingService
from perch_api.config import Settings
from perch_api.storage import S3ObjectRef, S3Storage
from perch_api.vector_store import QdrantStore, VectorWriter


def _s3_event(ref: S3ObjectRef) -> dict:
  return {
      "Records": [{
          "s3": {
              "bucket": {"name": ref.bucket},
              "object": {
                  "key": ref.key,
                  "eTag": ref.etag,
                  "versionId": ref.version_id,
              },
          }
      }]
  }


def _retry_delay(response: httpx.Response, attempt: int, backoff: float) -> float:
  retry_after = response.headers.get("Retry-After")
  if retry_after:
    try:
      return max(0.0, float(retry_after))
    except ValueError:
      try:
        retry_at = parsedate_to_datetime(retry_after)
        if retry_at.tzinfo is None:
          retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
      except (TypeError, ValueError, OverflowError):
        pass
  return backoff * (2**attempt)


def submit_webhook(
    ref: S3ObjectRef,
    webhook_url: str,
    retries: int,
    backoff: float,
) -> str:
  for attempt in range(retries + 1):
    try:
      response = httpx.post(webhook_url, json=_s3_event(ref), timeout=30.0)
    except httpx.HTTPError as exc:
      raise click.ClickException(f"Webhook request failed for {ref.uri}: {exc}") from exc
    if response.status_code == 202:
      try:
        job_ids = response.json()["job_ids"]
        return job_ids[0]
      except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise click.ClickException(
            f"Webhook returned an invalid response for {ref.uri}"
        ) from exc
    if response.status_code != 503 or attempt >= retries:
      detail = response.text.strip()
      raise click.ClickException(
          f"Webhook returned HTTP {response.status_code} for {ref.uri}"
          + (f": {detail}" if detail else "")
      )
    time.sleep(_retry_delay(response, attempt, backoff))
  raise AssertionError("unreachable")


@click.command()
@click.option(
    "--object-uri",
    "object_uris",
    multiple=True,
    help="S3 object URI. May be supplied more than once.",
)
@click.option("--bucket", help="Bucket to scan when --prefix is supplied.")
@click.option("--prefix", default="", show_default=True)
@click.option(
    "--suffix",
    multiple=True,
    default=(".wav", ".flac", ".ogg", ".mp3"),
    show_default=True,
    help="Audio suffix to include. May be supplied more than once.",
)
@click.option("--dry-run", is_flag=True, help="List objects without embedding.")
@click.option(
  "--progress",
  "show_progress",
  is_flag=True,
  help="Show a progress bar instead of per-object upsert messages.",
)
@click.option(
  "--workers",
  type=click.IntRange(min=1),
  default=1,
  show_default=True,
  help="Number of objects to ingest concurrently.",
)
@click.option(
    "--no-skip-existing",
    is_flag=True,
    help="Re-embed objects that already have vectors in Qdrant.",
)
@click.option(
  "--webhook-url",
  envvar="PERCH_API_WEBHOOK_URL",
  help="Submit jobs to the API webhook instead of processing locally.",
)
@click.option(
  "--webhook-retries",
  type=click.IntRange(min=0),
  default=5,
  show_default=True,
  help="Retries after HTTP 503 responses.",
)
@click.option(
  "--webhook-backoff",
  type=click.FloatRange(min=0),
  default=1.0,
  show_default=True,
  help="Initial seconds to wait between HTTP 503 retries.",
)
def ingest(
  object_uris,
  bucket,
  prefix,
  suffix,
  dry_run,
  show_progress,
  workers,
  no_skip_existing,
  webhook_url,
  webhook_retries,
  webhook_backoff,
) -> None:
  """Ingest explicit S3 objects or all audio objects under a prefix."""
  if not object_uris and not bucket:
    raise click.UsageError("Provide --object-uri or --bucket.")
  if prefix and not bucket:
    raise click.UsageError("--prefix requires --bucket.")

  storage = S3Storage()
  refs = [S3ObjectRef.from_uri(uri) for uri in object_uris]
  if bucket:
    refs.extend(storage.list(bucket, prefix))
  refs = [
      ref for ref in refs if ref.key.lower().endswith(tuple(s.lower() for s in suffix))
  ]

  if dry_run:
    for ref in refs:
      click.echo(ref.uri)
    return

  if webhook_url:
    settings = Settings.from_env()
    vectors = None
    if not no_skip_existing:
      vectors = QdrantStore(
          url=settings.qdrant_url,
          api_key=settings.qdrant_api_key,
          collection=settings.qdrant_collection,
          timeout=settings.qdrant_timeout_s,
      )
    progress = tqdm(total=len(refs), desc="Submitting", unit="file") if show_progress else None
    try:
      for ref in refs:
        if vectors is not None and vectors.has_vectors(ref, settings.model_name):
          if progress is None:
            click.echo(f"{ref.uri}: skipped, vectors already exist")
          else:
            progress.update(1)
          continue
        job_id = submit_webhook(
            ref, webhook_url, webhook_retries, webhook_backoff
        )
        if progress is None:
          click.echo(f"{ref.uri}: submitted job {job_id}")
        else:
          progress.update(1)
    finally:
      if progress is not None:
        progress.close()
    return

  service = EmbeddingService.from_env()
  model_name = service.pipeline.model_name

  def load_if_needed(ref):
    if not no_skip_existing and service.vectors.has_vectors(ref, model_name):
      return None
    return service.load_audio(ref)

  on_write = (
      lambda ref, count: click.echo(f"{ref.uri}: upserted {count} vectors")
      if not show_progress
      else None
  )
  writer = VectorWriter(
      service.vectors,
      batch_size=service.settings.upsert_batch_size,
      maxsize=service.settings.job_queue_size,
      on_write=on_write,
  )
  executor = ThreadPoolExecutor(max_workers=workers)
  interrupted = False
  progress = None
  try:
    pending = {
      executor.submit(load_if_needed, ref): ref for ref in refs[:workers]
    }
    remaining_refs = iter(refs[workers:])
    progress = tqdm(total=len(refs), desc="Ingesting", unit="file") if show_progress else None
    while pending:
      completed, _ = wait(pending, return_when=FIRST_COMPLETED)
      for future in completed:
        ref = pending.pop(future)
        audio = future.result()
        if audio is None:
          try:
            next_ref = next(remaining_refs)
          except StopIteration:
            pass
          else:
            pending[executor.submit(load_if_needed, next_ref)] = next_ref
          if progress is not None:
            progress.update(1)
          else:
            click.echo(f"{ref.uri}: skipped, vectors already exist")
          continue
        windows = service.pipeline.embed_audio(audio, ref)
        writer.submit(ref, service.pipeline.model_name, windows)
        try:
          next_ref = next(remaining_refs)
        except StopIteration:
          pass
        else:
          pending[executor.submit(load_if_needed, next_ref)] = next_ref
        if progress is not None:
          progress.update(1)
  except KeyboardInterrupt:
    interrupted = True
    for future in pending:
      future.cancel()
    click.echo("Interrupted; cancelling pending work.", err=True)
    raise click.Abort()
  finally:
    if progress is not None:
      progress.close()
    if interrupted:
      writer.abort()
      executor.shutdown(wait=False, cancel_futures=True)
    else:
      writer.close()
      executor.shutdown(wait=True)


if __name__ == "__main__":
  ingest()
