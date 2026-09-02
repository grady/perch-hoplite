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

"""Tests for Agile configuration loading."""

import os
from unittest import mock

from perch_hoplite.agile import colab_utils
from perch_hoplite.agile import source_info

from absl.testing import absltest


def _audio_sources() -> source_info.AudioSources:
  return source_info.AudioSources((
      source_info.AudioSourceConfig(
          dataset_name='test', base_path='/audio', file_glob='*.wav'
      ),
  ))


class LoadConfigsTest(absltest.TestCase):

  def test_pg_qdrant_uses_explicit_url(self):
    configs = colab_utils.load_configs(
        audio_sources=_audio_sources(),
        db_key='pg_qdrant',
        db_dsn='postgresql://localhost/hoplite',
        qdrant_url='https://qdrant.example:443',
    )

    self.assertEqual(configs.db_config.db_config.qdrant_cfg.mode, 'remote')
    self.assertEqual(
        configs.db_config.db_config.qdrant_cfg.url,
        'https://qdrant.example:443',
    )
    self.assertNotIn('api_key', configs.db_config.db_config.qdrant_cfg)

  def test_pg_qdrant_uses_url_environment_variable(self):
    with mock.patch.dict(
        os.environ,
        {'HOPLITE_QDRANT_URL': 'http://qdrant.example:6333'},
        clear=True,
    ):
      configs = colab_utils.load_configs(
          audio_sources=_audio_sources(),
          db_key='pg_qdrant',
          db_dsn='postgresql://localhost/hoplite',
      )

    self.assertEqual(
        configs.db_config.db_config.qdrant_cfg.url,
        'http://qdrant.example:6333',
    )

  def test_pg_qdrant_rejects_url_with_credentials(self):
    with self.assertRaisesRegex(ValueError, 'embedded credentials'):
      colab_utils.load_configs(
          audio_sources=_audio_sources(),
          db_key='pg_qdrant',
          db_dsn='postgresql://localhost/hoplite',
          qdrant_url='https://user:secret@qdrant.example:443',
      )


if __name__ == '__main__':
  absltest.main()