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

"""Tests for the Hoplite Click CLI."""

import os
from types import SimpleNamespace
from unittest import mock

from click.testing import CliRunner
from perch_hoplite.agile import cli
from perch_hoplite.agile import source_info

from absl.testing import absltest


class CliTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.runner = CliRunner()
    self.database = mock.Mock()
    self.database.count_embeddings.return_value = 23
    self.colab_utils = mock.Mock()
    self.embed = mock.Mock()
    self.configs = SimpleNamespace(
        audio_sources_config=mock.sentinel.audio_sources,
        model_config=mock.sentinel.model_config,
        db_config=mock.Mock(),
    )
    self.configs.db_config.load_db.return_value = self.database
    self.colab_utils.load_configs.return_value = self.configs
    dependency_patcher = mock.patch.object(
        cli,
        '_load_embed_dependencies',
        return_value=(self.colab_utils, self.embed, source_info),
    )
    dependency_patcher.start()
    self.addCleanup(dependency_patcher.stop)

  def test_embed_uses_notebook_defaults(self):
    with mock.patch.object(self.embed, 'EmbedWorker') as worker_cls:
      result = self.runner.invoke(
          cli.main,
          [
              'embed',
              '--dataset-name',
              'test-dataset',
              '--audio-path',
              '/audio',
              '--audio-glob',
              '*.wav',
          ],
      )

    self.assertEqual(result.exit_code, 0, result.output)
    audio_config = self.colab_utils.load_configs.call_args.kwargs['audio_sources'].audio_globs[0]
    self.assertEqual(audio_config.dataset_name, 'test-dataset')
    self.assertEqual(audio_config.base_path, '/audio')
    self.assertEqual(audio_config.file_glob, '*.wav')
    self.assertEqual(audio_config.shard_len_s, 60.0)
    self.assertEqual(self.colab_utils.load_configs.call_args.kwargs['model_config_key'], 'perch_v2')
    self.assertEqual(self.colab_utils.load_configs.call_args.kwargs['db_key'], 'sqlite_usearch')
    worker_cls.assert_called_once_with(
        audio_sources=mock.sentinel.audio_sources,
        db=self.database,
        model_config=mock.sentinel.model_config,
        audio_worker_threads=8,
        cache_local_audio=True,
        timestamp_file_pattern=None,
    )
    worker_cls.return_value.process_all.assert_called_once_with(
        target_dataset_name='test-dataset',
        batch_size=16,
        handle_duplicates='skip',
    )
    self.assertIn('total embeddings: 23', result.output)

  def test_embed_forwards_remote_options(self):
    with mock.patch.object(self.embed, 'EmbedWorker'):
      result = self.runner.invoke(
          cli.main,
          [
              'embed',
              '--dataset-name',
              'remote',
              '--audio-path',
              's3://bucket/audio',
              '--audio-glob',
              '**/*.flac',
              '--db-backend',
              'pg_qdrant',
              '--db-dsn',
              'postgresql://localhost/hoplite',
              '--qdrant-url',
              'https://qdrant.example:6333',
              '--qdrant-collection-name',
              'remote-embeddings',
              '--s3-endpoint',
              'storage.example:9000',
              '--s3-access-key',
              'access',
              '--s3-secret-key',
              'secret',
              '--s3-no-use-ssl',
              '--s3-no-verify',
              '--no-sharding',
          ],
      )

    self.assertEqual(result.exit_code, 0, result.output)
    self.assertEqual(self.colab_utils.load_configs.call_args.kwargs['db_dsn'], 'postgresql://localhost/hoplite')
    self.assertEqual(
      self.colab_utils.load_configs.call_args.kwargs['qdrant_url'],
      'https://qdrant.example:6333',
    )
    self.assertEqual(
      self.colab_utils.load_configs.call_args.kwargs['qdrant_collection_name'],
        'remote-embeddings',
    )
    self.assertEqual(self.colab_utils.load_configs.call_args.kwargs['s3_endpoint'], 'storage.example:9000')
    self.assertEqual(self.colab_utils.load_configs.call_args.kwargs['s3_use_ssl'], False)
    self.assertEqual(self.colab_utils.load_configs.call_args.kwargs['s3_verify'], False)
    audio_config = self.colab_utils.load_configs.call_args.kwargs['audio_sources'].audio_globs[0]
    self.assertIsNone(audio_config.shard_len_s)

  def test_pg_qdrant_accepts_environment_dsn(self):
    with mock.patch.dict(os.environ, {'HOPLITE_PG_DSN': 'postgresql://env/db'}), mock.patch.object(
      self.embed, 'EmbedWorker'
    ):
      result = self.runner.invoke(
          cli.main,
          [
              'embed',
              '--dataset-name',
              'remote',
              '--audio-path',
              '/audio',
              '--audio-glob',
              '*.wav',
              '--db-backend',
              'pg_qdrant',
          ],
      )

    self.assertEqual(result.exit_code, 0, result.output)

  def test_rejects_database_options_for_sqlite(self):
    result = self.runner.invoke(
        cli.main,
        [
            'embed',
            '--dataset-name',
            'test',
            '--audio-path',
            '/audio',
            '--audio-glob',
            '*.wav',
            '--db-dsn',
            'postgresql://localhost/hoplite',
            '--qdrant-url',
            'https://qdrant.example:443',
        ],
    )

    self.assertEqual(result.exit_code, 2)
    self.assertIn('PostgreSQL/Qdrant options require', result.output)
    self.colab_utils.load_configs.assert_not_called()

  def test_rejects_pg_qdrant_without_dsn(self):
    with mock.patch.dict(os.environ, {}, clear=True):
      result = self.runner.invoke(
          cli.main,
          [
              'embed',
              '--dataset-name',
              'test',
              '--audio-path',
              '/audio',
              '--audio-glob',
              '*.wav',
              '--db-backend',
              'pg_qdrant',
          ],
      )

    self.assertEqual(result.exit_code, 2)
    self.assertIn('--db-dsn is required', result.output)
    self.colab_utils.load_configs.assert_not_called()


if __name__ == '__main__':
  absltest.main()