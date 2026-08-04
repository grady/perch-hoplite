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

"""Utility functions for Agile modeling notebooks."""

import dataclasses
import os

from etils import epath
from ml_collections import config_dict
from perch_hoplite.agile import embed
from perch_hoplite.agile import source_info
from perch_hoplite.db import db_loader
from perch_hoplite.db import sqlite_usearch_impl
from perch_hoplite.zoo import model_configs


@dataclasses.dataclass
class AgileConfigs:
  """Container for the various configs used in the Agile notebooks."""

  # Config for the raw audio sources.
  audio_sources_config: source_info.AudioSources
  # Database config for the embeddings database.
  db_config: db_loader.DBConfig
  # Config for the embedding model.
  model_config: embed.ModelConfig

  def as_config_dict(self) -> config_dict.ConfigDict:
    """Returns the configs as a ConfigDict."""
    return config_dict.ConfigDict({
        'audio_sources_config': self.audio_sources_config.to_config_dict(),
        'db_config': self.db_config.to_config_dict(),
        'model_config': self.model_config.to_config_dict(),
    })


def load_configs(
    audio_sources: source_info.AudioSources,
    db_path: str | None = None,
    model_config_key: str = 'perch_v2',
    db_key: str = 'sqlite_usearch',
    db_dsn: str | None = None,
    qdrant_host: str | None = None,
    qdrant_port: int | None = None,
    qdrant_collection_name: str | None = None,
) -> AgileConfigs:
  """Load default configs for the notebook and return them as an AgileConfigs.

  Args:
    audio_sources: Mapping from dataset name to pairs of `(root directory, file
      glob)`.
    db_path: Location of the database.  If None, the database will be created in
      the same directory as the audio.
    model_config_key: Name of the embedding model to use.
    db_key: The type of database to use.
    db_dsn: PostgreSQL DSN for the pg_qdrant backend.
    qdrant_host: Qdrant host for the pg_qdrant backend.
    qdrant_port: Qdrant port for the pg_qdrant backend.
    qdrant_collection_name: Qdrant collection name for the pg_qdrant backend.

  Returns:
    AgileConfigs object with the loaded configs.
  """
  if db_path is None:
    if len(audio_sources.audio_globs) > 1:
      raise ValueError(
          'db_path must be specified when embedding multiple datasets.'
      )
    # Put the DB in the same directory as the audio.
    db_path = epath.Path(next(iter(audio_sources.audio_globs)).base_path)  # pyrefly: ignore[bad-assignment]

  preset_info = model_configs.get_preset_model_config(model_config_key)
  db_model_config = embed.ModelConfig(
      model_key=preset_info.model_key,
      embedding_dim=preset_info.embedding_dim,
      model_config=preset_info.model_config,
  )
  db_config = config_dict.ConfigDict({
      'db_path': db_path,
  })
  if db_key == 'sqlite_usearch':
    # A sane default.
    db_config.usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(
        preset_info.embedding_dim
    )
  elif db_key == 'pg_qdrant':
    from perch_hoplite.db import pg_qdrant_impl

    if db_dsn is None:
      db_dsn = os.environ.get('HOPLITE_PG_DSN')
    if not db_dsn:
      raise ValueError(
          'db_dsn must be provided for pg_qdrant, or set HOPLITE_PG_DSN.'
      )
    qdrant_cfg = pg_qdrant_impl.get_default_qdrant_config(
        preset_info.embedding_dim
    )
    if qdrant_collection_name:
      qdrant_cfg.collection_name = qdrant_collection_name
    if qdrant_host is None:
      qdrant_host = os.environ.get('HOPLITE_QDRANT_HOST')
    if qdrant_host:
      qdrant_cfg.mode = 'remote'
      qdrant_cfg.host = qdrant_host
      qdrant_cfg.port = int(
          qdrant_port
          if qdrant_port is not None
          else os.environ.get('HOPLITE_QDRANT_PORT', '6333')
      )
    db_config = config_dict.ConfigDict({
        'db_dsn': db_dsn,
        'qdrant_cfg': qdrant_cfg,
    })
  elif db_key != 'sqlite_usearch':
    raise ValueError(f'Unknown db_key: {db_key}')

  return AgileConfigs(
      audio_sources_config=audio_sources,
      db_config=db_loader.DBConfig(db_key, db_config),
      model_config=db_model_config,
  )
