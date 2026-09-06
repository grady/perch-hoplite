"""CLI for backfilling existing S3 audio objects."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import click
from tqdm import tqdm

from perch_api.service import EmbeddingService
from perch_api.storage import S3ObjectRef, S3Storage
from perch_api.vector_store import VectorWriter


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
def ingest(
  object_uris,
  bucket,
  prefix,
  suffix,
  dry_run,
  show_progress,
  workers,
  no_skip_existing,
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
  with ThreadPoolExecutor(max_workers=workers) as executor:
    pending = {
      executor.submit(load_if_needed, ref): ref for ref in refs[:workers]
    }
    remaining_refs = iter(refs[workers:])
    progress = tqdm(total=len(refs), desc="Ingesting", unit="file") if show_progress else None
    try:
      while pending:
        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
          ref = pending.pop(future)
          audio = future.result()
          if audio is None:
            if progress is not None:
              progress.update(1)
            else:
              click.echo(f"{ref.uri}: skipped, vectors already exist")
            try:
              next_ref = next(remaining_refs)
            except StopIteration:
              pass
            else:
              pending[executor.submit(load_if_needed, next_ref)] = next_ref
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
    finally:
      if progress is not None:
        progress.close()
      writer.close()


if __name__ == "__main__":
  ingest()
