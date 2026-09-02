# coding=utf-8
# Copyright 2026 The Perch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line interface for Perch Hoplite workflows."""

import importlib.metadata
import os

import click
from perch_hoplite.zoo import model_configs


def _package_version() -> str:
  try:
    return importlib.metadata.version('perch-hoplite')
  except importlib.metadata.PackageNotFoundError:
    return 'unknown'


def _load_embed_dependencies(db_backend: str):
  """Load embedding modules after any required native database dependency."""
  if db_backend == 'pg_qdrant':
    import psycopg2  # pylint: disable=unused-import,g-import-not-at-top

  from perch_hoplite.agile import colab_utils  # pylint: disable=g-import-not-at-top
  from perch_hoplite.agile import embed  # pylint: disable=g-import-not-at-top
  from perch_hoplite.agile import source_info  # pylint: disable=g-import-not-at-top

  return colab_utils, embed, source_info


@click.group()
@click.version_option(version=_package_version())
def main() -> None:
  """Run Perch Hoplite workflows."""


@main.command(name='embed')
@click.option('--dataset-name', required=True, help='Project name in the database.')
@click.option(
    '--audio-path',
    required=True,
    type=str,
    help='Directory or s3:// URI containing audio files.',
)
@click.option('--audio-glob', required=True, help='Glob relative to --audio-path.')
@click.option(
    '--model',
    'model_config_key',
    type=click.Choice([model.value for model in model_configs.ModelConfigName]),
    default='perch_v2',
    show_default=True,
    help='Embedding model preset.',
)
@click.option(
    '--db-backend',
    type=click.Choice(('sqlite_usearch', 'pg_qdrant')),
    default='sqlite_usearch',
    show_default=True,
)
@click.option('--db-path', type=str, help='SQLite database directory.')
@click.option('--db-dsn', type=str, help='PostgreSQL DSN for pg_qdrant.')
@click.option('--qdrant-host', type=str, help='Qdrant host for pg_qdrant.')
@click.option('--qdrant-port', type=click.IntRange(min=1), help='Qdrant port.')
@click.option('--qdrant-collection-name', type=str, help='Qdrant collection name.')
@click.option('--batch-size', type=click.IntRange(min=1), default=16, show_default=True)
@click.option(
    '--handle-duplicates',
    type=click.Choice(('error', 'skip', 'overwrite', 'allow')),
    default='skip',
    show_default=True,
)
@click.option(
    '--audio-workers', type=click.IntRange(min=1), default=8, show_default=True
)
@click.option('--cache-local-audio/--no-cache-local-audio', default=True, show_default=True)
@click.option('--min-audio-length', type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--target-sample-rate', type=int, default=-2, show_default=True)
@click.option('--shard-length', type=click.FloatRange(min=0, min_open=True), default=60.0, show_default=True)
@click.option('--no-sharding', is_flag=True, help='Process each audio file as one shard.')
@click.option('--max-shards-per-file', type=click.IntRange(min=1))
@click.option('--timestamp-file-pattern', type=str, help='strptime pattern for recording filenames.')
@click.option('--s3-endpoint', type=str, help='S3-compatible endpoint URL.')
@click.option('--s3-access-key', type=str, help='S3 access key ID.')
@click.option('--s3-secret-key', type=str, help='S3 secret access key.')
@click.option('--s3-session-token', type=str, help='S3 session token.')
@click.option('--s3-region', type=str, help='S3 region.')
@click.option('--s3-use-ssl/--s3-no-use-ssl', default=None, help='Enable or disable S3 TLS.')
@click.option('--s3-verify/--s3-no-verify', default=None, help='Enable or disable S3 certificate verification.')
def embed_audio(
    dataset_name: str,
    audio_path: str,
    audio_glob: str,
    model_config_key: str,
    db_backend: str,
    db_path: str | None,
    db_dsn: str | None,
    qdrant_host: str | None,
    qdrant_port: int | None,
    qdrant_collection_name: str | None,
    batch_size: int,
    handle_duplicates: str,
    audio_workers: int,
    cache_local_audio: bool,
    min_audio_length: float,
    target_sample_rate: int,
    shard_length: float,
    no_sharding: bool,
    max_shards_per_file: int | None,
    timestamp_file_pattern: str | None,
    s3_endpoint: str | None,
    s3_access_key: str | None,
    s3_secret_key: str | None,
    s3_session_token: str | None,
    s3_region: str | None,
    s3_use_ssl: bool | None,
    s3_verify: bool | None,
) -> None:
  """Embed audio files into a Hoplite database."""
  if db_backend == 'sqlite_usearch' and any(
      value is not None
      for value in (db_dsn, qdrant_host, qdrant_port, qdrant_collection_name)
  ):
    raise click.UsageError(
        'PostgreSQL/Qdrant options require --db-backend pg_qdrant.'
    )
  if db_backend == 'pg_qdrant' and not (
      db_dsn or os.environ.get('HOPLITE_PG_DSN')
  ):
    raise click.UsageError(
        '--db-dsn is required for pg_qdrant unless HOPLITE_PG_DSN is set.'
    )

  colab_utils, embed, source_info = _load_embed_dependencies(db_backend)
  audio_sources = source_info.AudioSources(
      (
          source_info.AudioSourceConfig(
              dataset_name=dataset_name,
              base_path=audio_path,
              file_glob=audio_glob,
              min_audio_len_s=min_audio_length,
              target_sample_rate_hz=target_sample_rate,
              shard_len_s=None if no_sharding else shard_length,
              max_shards_per_file=max_shards_per_file,
          ),
      )
  )
  configs = colab_utils.load_configs(
      audio_sources=audio_sources,
      db_path=db_path,
      model_config_key=model_config_key,
      db_key=db_backend,
      db_dsn=db_dsn,
      qdrant_host=qdrant_host,
      qdrant_port=qdrant_port,
      qdrant_collection_name=qdrant_collection_name,
      s3_endpoint=s3_endpoint,
      s3_access_key=s3_access_key,
      s3_secret_key=s3_secret_key,
      s3_session_token=s3_session_token,
      s3_region=s3_region,
      s3_use_ssl=s3_use_ssl,
      s3_verify=s3_verify,
  )
  database = configs.db_config.load_db()
  worker = embed.EmbedWorker(
      audio_sources=configs.audio_sources_config,
      db=database,
      model_config=configs.model_config,
      audio_worker_threads=audio_workers,
      cache_local_audio=cache_local_audio,
      timestamp_file_pattern=timestamp_file_pattern,
  )
  click.echo(f'Embedding dataset as a new db project: {dataset_name}')
  worker.process_all(
      target_dataset_name=dataset_name,
      batch_size=batch_size,
      handle_duplicates=handle_duplicates,
  )
  click.echo(f'Embedding complete, total embeddings: {database.count_embeddings()}')


if __name__ == '__main__':
  main()
