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

"""Utility functions for testing."""

import os
import unittest

from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import datatypes
from perch_hoplite.db import in_mem_impl
from perch_hoplite.db import interface
from perch_hoplite.db import multi_db_impl
from perch_hoplite.db import pg_qdrant_impl
from perch_hoplite.db import sqlite_usearch_impl

# Set HOPLITE_PG_DSN to enable pg_qdrant tests, e.g.:
#   export HOPLITE_PG_DSN="postgresql://user:pass@localhost:5432/hoplite_test"
_PG_DSN_ENV = 'HOPLITE_PG_DSN'
_QDRANT_HOST_ENV = 'HOPLITE_QDRANT_HOST'
_QDRANT_PORT_ENV = 'HOPLITE_QDRANT_PORT'
_QDRANT_COLLECTION_ENV = 'HOPLITE_QDRANT_COLLECTION'
_DEFAULT_TEST_QDRANT_COLLECTION = 'hoplite_test_embeddings'

# DB types for testing.
DB_TYPES = (
    'in_mem',
    'sqlite_usearch',
    'pg_qdrant',
    'multi_db_in_mem',
    'multi_db_sqlite_usearch',
)
DB_TYPE_NAMED_PAIRS = (
    ('in_mem-sqlite_usearch', 'in_mem', 'sqlite_usearch'),
    (
        'multi_db_in_mem-multi_db_sqlite_usearch',
        'multi_db_in_mem',
        'multi_db_sqlite_usearch',
    ),
)
PERSISTENT_DB_TYPES = ('sqlite_usearch', 'multi_db_sqlite_usearch', 'pg_qdrant')

CLASS_LABELS = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta')


def make_db(
    path: str,
    db_type: str,
    num_embeddings: int,
    rng: np.random.Generator,
    embedding_dim: int = 128,
    fill_random: bool = True,
) -> interface.HopliteDBInterface:
  """Create a test DB of the specified type."""
  if db_type == 'in_mem':
    db = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=embedding_dim)
  elif db_type == 'sqlite_usearch':
    usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(embedding_dim)
    db = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_path=path, usearch_cfg=usearch_cfg
    )
  elif db_type == 'pg_qdrant':
    pg_dsn = os.environ.get(_PG_DSN_ENV)
    if not pg_dsn:
      raise unittest.SkipTest(  # pyrefly: ignore[name-error]
          f'Set {_PG_DSN_ENV} to run pg_qdrant tests.'
      )
    qdrant_cfg = get_qdrant_config(embedding_dim)
    _reset_pg_qdrant_schema(pg_dsn)
    _reset_qdrant_collection(qdrant_cfg)
    db = pg_qdrant_impl.PgQdrantDB.create(
        db_dsn=pg_dsn, qdrant_cfg=qdrant_cfg
    )
  elif db_type == 'multi_db_in_mem':
    db0 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=embedding_dim)
    db1 = in_mem_impl.InMemoryGraphSearchDB.create(embedding_dim=embedding_dim)
    db = multi_db_impl.MultiDBWrapper.create(dbs={'db0': db0, 'db1': db1})
  elif db_type == 'multi_db_sqlite_usearch':
    usearch_cfg = sqlite_usearch_impl.get_default_usearch_config(embedding_dim)
    db0 = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_path=f'{path}_0', usearch_cfg=usearch_cfg
    )
    db1 = sqlite_usearch_impl.SQLiteUSearchDB.create(
        db_path=f'{path}_1', usearch_cfg=usearch_cfg
    )
    db = multi_db_impl.MultiDBWrapper.create(dbs={'db0': db0, 'db1': db1})
  else:
    raise ValueError(f'Unknown db type: {db_type}')
  # Insert a few embeddings...
  if fill_random:
    insert_random_embeddings(db, embedding_dim, num_embeddings, rng)  # pyrefly: ignore[bad-argument-type]
  config = config_dict.ConfigDict()
  config.embedding_dim = embedding_dim
  model_config = config_dict.ConfigDict()
  model_config.embedding_dim = embedding_dim
  model_config.model_name = 'fake_model'
  if isinstance(db, multi_db_impl.MultiDBWrapper):
    db.insert_metadata('db0/db_config', config)
    db.insert_metadata('db0/model_config', model_config)
    db.insert_metadata('db1/db_config', config)
    db.insert_metadata('db1/model_config', model_config)
  else:
    db.insert_metadata('db_config', config)
    db.insert_metadata('model_config', model_config)
  db.commit()
  return db


def _reset_pg_qdrant_schema(pg_dsn: str) -> None:
  """Drop the Hoplite schema so a pg_qdrant test starts clean."""
  conn = pg_qdrant_impl.psycopg2.connect(pg_dsn)
  try:
    cur = conn.cursor()
    cur.execute("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = current_database()
          AND usename = current_user
          AND pid <> pg_backend_pid()
        """)
    conn.commit()
    cur.execute("SET lock_timeout = '5s'")
    cur.execute(
        """
        DROP TABLE IF EXISTS
          annotations, windows, recordings, deployments, hoplite_metadata
        CASCADE
        """
    )
    conn.commit()
  finally:
    conn.close()


def get_qdrant_config(embedding_dim: int) -> config_dict.ConfigDict:
  """Return the Qdrant config for tests.

  If ``HOPLITE_QDRANT_HOST`` is set, use a remote Qdrant server at that host
  and ``HOPLITE_QDRANT_PORT`` (default 6333). Otherwise use in-memory Qdrant.
  """
  qdrant_cfg = pg_qdrant_impl.get_default_qdrant_config(embedding_dim)
  qdrant_cfg.collection_name = os.environ.get(
      _QDRANT_COLLECTION_ENV, _DEFAULT_TEST_QDRANT_COLLECTION
  )
  qdrant_host = os.environ.get(_QDRANT_HOST_ENV)
  if qdrant_host:
    qdrant_cfg.mode = 'remote'
    qdrant_cfg.host = qdrant_host
    qdrant_cfg.port = int(os.environ.get(_QDRANT_PORT_ENV, '6333'))
  return qdrant_cfg


def _reset_qdrant_collection(qdrant_cfg: config_dict.ConfigDict) -> None:
  """Delete the test collection if it already exists."""
  qc = pg_qdrant_impl._make_qdrant_client(qdrant_cfg)
  try:
    existing = {c.name for c in qc.get_collections().collections}
    if qdrant_cfg.collection_name in existing:
      qc.delete_collection(qdrant_cfg.collection_name)
  finally:
    close = getattr(qc, 'close', None)
    if callable(close):
      close()


def insert_random_embeddings(
    db: interface.HopliteDBInterface,
    emb_dim: int = 1280,
    num_embeddings: int = 1000,
    seed: int = 42,
):
  """Insert randomly generated embedding vectors into the DB."""
  rng = np.random.default_rng(seed=seed)
  np_alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

  projects = ('a', 'b', 'c')
  deployment_ids = []
  for project in projects:
    if isinstance(db, multi_db_impl.MultiDBWrapper):
      db.get_dbs()[0].insert_deployment(
          name=f'deployment_{project}', project=project
      )
    deployment_id = db.insert_deployment(
        name=f'deployment_{project}', project=project
    )
    deployment_ids.append(deployment_id)

  window_size_s = 5.0
  for _ in range(num_embeddings):
    deployment_id = rng.choice(deployment_ids).item()
    filename = ''.join(
        [str(a) for a in rng.choice(np_alpha, size=8, replace=False)]
    )
    recording_id = db.insert_recording(
        filename=filename, deployment_id=deployment_id
    )

    embedding = np.float32(rng.normal(size=emb_dim, loc=0, scale=0.1))
    offsets = rng.integers(0, 100, size=[1])
    offsets = [offsets[0], offsets[0] + window_size_s]
    db.insert_window(
        recording_id, offsets, embedding, handle_duplicates='allow'  # pyrefly: ignore[bad-argument-type]
    )
  db.commit()


def clone_embeddings(
    source_db: interface.HopliteDBInterface,
    target_db: interface.HopliteDBInterface,
):
  """Copy all embeddings to target_db and provide an id mapping."""

  # First, clone deployments and keep a map between source and target ids.
  deployment_id_mapping = {None: None}
  for deployment in source_db.get_all_deployments():
    kwargs = deployment.to_kwargs(skip=['id'])
    if isinstance(target_db, multi_db_impl.MultiDBWrapper):
      target_db.get_dbs()[0].insert_deployment(**kwargs)
    target_id = target_db.insert_deployment(**kwargs)
    deployment_id_mapping[deployment.id] = target_id  # pyrefly: ignore[unsupported-operation]

  # Second, clone recordings and keep a map between source and target ids.
  recording_id_mapping = {}
  for recording in source_db.get_all_recordings():
    target_id = target_db.insert_recording(
        deployment_id=deployment_id_mapping[recording.deployment_id],  # pyrefly: ignore[bad-index]
        **recording.to_kwargs(skip=['id', 'deployment_id']),
    )
    recording_id_mapping[recording.id] = target_id

  # Finally, clone windows and keep a map between source and target ids.
  window_id_mapping = {}
  for window in source_db.get_all_windows():
    target_id = target_db.insert_window(
        recording_id=recording_id_mapping[window.recording_id],
        embedding=source_db.get_embedding(window.id),
        handle_duplicates='allow',
        **window.to_kwargs(skip=['id', 'embedding', 'recording_id']),
    )
    window_id_mapping[window.id] = target_id

  # Return the window id mapping.
  return window_id_mapping


def add_random_labels(
    db: interface.HopliteDBInterface,
    rng: np.random.Generator,
    unlabeled_prob: float = 0.5,
    positive_label_prob: float = 0.5,
    provenance: str = 'test',
):
  """Insert random labels for a subset of embeddings."""
  for idx in db.match_window_ids():
    if rng.random() < unlabeled_prob:
      continue
    if rng.random() < positive_label_prob:
      label_type = datatypes.LabelType.POSITIVE
    else:
      label_type = datatypes.LabelType.NEGATIVE
    window = db.get_window(idx)
    db.insert_annotation(
        recording_id=window.recording_id,
        offsets=window.offsets,
        label=str(rng.choice(CLASS_LABELS)),
        label_type=label_type,
        provenance=provenance,
        handle_duplicates='allow',
    )
  db.commit()
