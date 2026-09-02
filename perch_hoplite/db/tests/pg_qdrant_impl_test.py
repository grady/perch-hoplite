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

"""Integration tests for the PostgreSQL + Qdrant database implementation.

These tests require a live PostgreSQL instance. Set the environment variable
``HOPLITE_PG_DSN`` to a valid DSN before running, for example::

    export HOPLITE_PG_DSN="postgresql://user:pass@localhost:5432/hoplite_test"
    python -m pytest perch_hoplite/db/tests/pg_qdrant_impl_test.py

By default the tests use an in-memory Qdrant instance. To point them at an
external Qdrant server, set::

  export HOPLITE_QDRANT_URL=http://localhost:6333

For an authenticated HTTPS endpoint, also set::

  export HOPLITE_QDRANT_URL=https://qdrant.example.com:443
  export HOPLITE_QDRANT_API_KEY=<api-key>

Tests use a dedicated Qdrant collection name by default
(`hoplite_test_embeddings`) so they do not collide with notebook or
application data. Override it with ``HOPLITE_QDRANT_COLLECTION`` if needed.

Tests are automatically skipped when the variable is not set.
"""

import os
import unittest
from unittest import mock

from ml_collections import config_dict
import numpy as np
from perch_hoplite.db import datatypes
from perch_hoplite.db import pg_qdrant_impl
from perch_hoplite.db.tests import test_utils

from absl.testing import absltest
from absl.testing import parameterized

_PG_DSN_ENV = 'HOPLITE_PG_DSN'
EMBEDDING_SIZE = 16  # Small dimension to keep tests fast.


def _get_dsn() -> str:
  dsn = os.environ.get(_PG_DSN_ENV, '')
  if not dsn:
    raise unittest.SkipTest(
        f'Set {_PG_DSN_ENV} to run pg_qdrant integration tests.'
    )
  return dsn


def _make_db(embedding_dim: int = EMBEDDING_SIZE) -> pg_qdrant_impl.PgQdrantDB:
  """Create a fresh PgQdrantDB with the configured Qdrant backend."""
  dsn = _get_dsn()
  qdrant_cfg = test_utils.get_qdrant_config(embedding_dim)
  test_utils._reset_pg_qdrant_schema(dsn)
  test_utils._reset_qdrant_collection(qdrant_cfg)
  return pg_qdrant_impl.PgQdrantDB.create(db_dsn=dsn, qdrant_cfg=qdrant_cfg)


def _reset_db(db: pg_qdrant_impl.PgQdrantDB) -> None:
  """Clean up database state for a finished test run."""
  db.rollback()
  db.db.close()
  test_utils._reset_pg_qdrant_schema(db._db_dsn)
  test_utils._reset_qdrant_collection(db._qdrant_cfg)


class PgQdrantHelperTest(absltest.TestCase):
  """Unit tests for module-level SQL helper functions (no DB required)."""

  def test_remote_qdrant_client_uses_url_and_runtime_api_key(self):
    qdrant_cfg = pg_qdrant_impl.get_default_qdrant_config(16)
    qdrant_cfg.mode = 'remote'
    qdrant_cfg.url = 'https://qdrant.example:443'

    with mock.patch.dict(
        os.environ, {'HOPLITE_QDRANT_API_KEY': 'test-api-key'}, clear=True
    ), mock.patch.object(pg_qdrant_impl, 'QdrantClient') as client_cls:
      pg_qdrant_impl._make_qdrant_client(qdrant_cfg)

    client_cls.assert_called_once_with(
        url='https://qdrant.example:443', api_key='test-api-key'
    )
    self.assertNotIn('api_key', qdrant_cfg)

  def test_remote_qdrant_client_allows_missing_api_key(self):
    qdrant_cfg = pg_qdrant_impl.get_default_qdrant_config(16)
    qdrant_cfg.mode = 'remote'
    qdrant_cfg.url = 'http://localhost:6333'

    with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
        pg_qdrant_impl, 'QdrantClient'
    ) as client_cls:
      pg_qdrant_impl._make_qdrant_client(qdrant_cfg)

    client_cls.assert_called_once_with(
        url='http://localhost:6333', api_key=None
    )

  def test_is_valid_sql_identifier(self):
    self.assertTrue(pg_qdrant_impl.is_valid_sql_identifier('foo'))
    self.assertTrue(pg_qdrant_impl.is_valid_sql_identifier('foo_bar'))
    self.assertTrue(pg_qdrant_impl.is_valid_sql_identifier('_foo'))
    self.assertFalse(pg_qdrant_impl.is_valid_sql_identifier(''))
    self.assertFalse(pg_qdrant_impl.is_valid_sql_identifier('1foo'))
    self.assertFalse(pg_qdrant_impl.is_valid_sql_identifier('foo bar'))
    self.assertFalse(pg_qdrant_impl.is_valid_sql_identifier('foo;drop'))

  def test_format_sql_insert_values_pg(self):
    cols, placeholders, values = pg_qdrant_impl.format_sql_insert_values_pg(
        name='test', latitude=1.0
    )
    self.assertIn('name', cols)
    self.assertIn('latitude', cols)
    self.assertEqual(placeholders.count('%s'), 2)
    self.assertEqual(values, ['test', 1.0])

  def test_format_sql_insert_values_pg_rejects_bad_identifier(self):
    with self.assertRaises(ValueError):
      pg_qdrant_impl.format_sql_insert_values_pg(**{'bad name': 1})

  def test_format_sql_where_conditions_pg_eq(self):
    cond, vals = pg_qdrant_impl.format_sql_where_conditions_pg(
        config_dict.create(eq=dict(project='test'))
    )
    self.assertIn('project = %s', cond)
    self.assertEqual(vals, ['test'])

  def test_format_sql_where_conditions_pg_eq_none(self):
    cond, vals = pg_qdrant_impl.format_sql_where_conditions_pg(
        config_dict.create(eq=dict(deployment_id=None))
    )
    self.assertIn('IS NULL', cond)
    self.assertEmpty(vals)

  def test_format_sql_where_conditions_pg_isin(self):
    cond, vals = pg_qdrant_impl.format_sql_where_conditions_pg(
        config_dict.create(isin=dict(label=['a', 'b', 'c']))
    )
    self.assertIn('IN (%s, %s, %s)', cond)
    self.assertEqual(vals, ['a', 'b', 'c'])

  def test_format_sql_where_conditions_pg_range(self):
    cond, vals = pg_qdrant_impl.format_sql_where_conditions_pg(
        config_dict.create(range=dict(latitude=[10.0, 20.0]))
    )
    self.assertIn('BETWEEN %s AND %s', cond)

  def test_format_sql_where_conditions_pg_approx_offsets(self):
    cond, vals = pg_qdrant_impl.format_sql_where_conditions_pg(
        config_dict.create(approx=dict(offsets=[1.0, 2.0]))
    )
    self.assertIn('approx_float_list', cond)
    self.assertEqual(vals, [[1.0, 2.0]])

  def test_format_sql_where_conditions_pg_unsupported_op(self):
    with self.assertRaises(ValueError):
      pg_qdrant_impl.format_sql_where_conditions_pg(
          config_dict.create(fancy=dict(x=1))
      )


class PgQdrantDBTest(parameterized.TestCase):
  """Integration tests against a live PostgreSQL + in-memory Qdrant."""

  def setUp(self):
    super().setUp()
    self.db = _make_db()

  def tearDown(self):
    super().tearDown()
    _reset_db(self.db)

  # ------------------------------------------------------------------
  # Metadata
  # ------------------------------------------------------------------

  def test_insert_and_get_metadata(self):
    cfg = config_dict.ConfigDict({'foo': 'bar', 'n': 42})
    self.db.insert_metadata('my_key', cfg)
    self.db.commit()
    got = self.db.get_metadata('my_key')
    self.assertEqual(got.foo, 'bar')
    self.assertEqual(got.n, 42)

  def test_get_metadata_all(self):
    self.db.insert_metadata('k1', config_dict.ConfigDict({'x': 1}))
    self.db.insert_metadata('k2', config_dict.ConfigDict({'y': 2}))
    self.db.commit()
    all_meta = self.db.get_metadata(None)
    self.assertIn('k1', all_meta)
    self.assertIn('k2', all_meta)

  def test_get_metadata_missing_key_raises(self):
    with self.assertRaises(KeyError):
      self.db.get_metadata('nonexistent')

  def test_remove_metadata(self):
    self.db.insert_metadata('to_del', config_dict.ConfigDict({'v': 1}))
    self.db.commit()
    self.db.remove_metadata('to_del')
    self.db.commit()
    with self.assertRaises(KeyError):
      self.db.get_metadata('to_del')

  def test_remove_metadata_missing_raises(self):
    with self.assertRaises(KeyError):
      self.db.remove_metadata('nonexistent')

  # ------------------------------------------------------------------
  # Deployments
  # ------------------------------------------------------------------

  def test_insert_and_get_deployment(self):
    dep_id = self.db.insert_deployment(name='site_a', project='proj_x')
    self.db.commit()
    dep = self.db.get_deployment(dep_id)
    self.assertEqual(dep.name, 'site_a')
    self.assertEqual(dep.project, 'proj_x')

  def test_insert_deployment_upsert_returns_same_id(self):
    dep_id1 = self.db.insert_deployment(name='site_a', project='proj_x')
    dep_id2 = self.db.insert_deployment(name='site_a', project='proj_x')
    self.db.commit()
    self.assertEqual(dep_id1, dep_id2)

  def test_get_deployment_missing_raises(self):
    with self.assertRaises(KeyError):
      self.db.get_deployment(999)

  def test_remove_deployment(self):
    dep_id = self.db.insert_deployment(name='site_a', project='proj_x')
    self.db.commit()
    self.db.remove_deployment(dep_id)
    self.db.commit()
    with self.assertRaises(KeyError):
      self.db.get_deployment(dep_id)

  def test_remove_deployment_cascades_to_windows(self):
    rng = np.random.default_rng(0)
    dep_id = self.db.insert_deployment(name='site', project='proj')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    emb = rng.normal(size=EMBEDDING_SIZE).astype(np.float32)
    win_id = self.db.insert_window(rec_id, [0.0, 5.0], embedding=emb)
    self.db.commit()
    self.assertEqual(self.db.count_embeddings(), 1)
    self.db.remove_deployment(dep_id)
    self.db.commit()
    self.assertEqual(self.db.count_embeddings(), 0)
    self.assertEmpty(self.db.match_window_ids())

  # ------------------------------------------------------------------
  # Recordings
  # ------------------------------------------------------------------

  def test_insert_and_get_recording(self):
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='audio.wav', deployment_id=dep_id)
    self.db.commit()
    rec = self.db.get_recording(rec_id)
    self.assertEqual(rec.filename, 'audio.wav')
    self.assertEqual(rec.deployment_id, dep_id)

  def test_insert_recording_bad_deployment_raises(self):
    with self.assertRaises(RuntimeError):
      self.db.insert_recording(filename='x.wav', deployment_id=9999)

  # ------------------------------------------------------------------
  # Windows + embeddings
  # ------------------------------------------------------------------

  def test_insert_window_without_embedding(self):
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    win_id = self.db.insert_window(rec_id, [0.0, 5.0])
    self.db.commit()
    win = self.db.get_window(win_id)
    self.assertEqual(win.recording_id, rec_id)
    self.assertAlmostEqual(win.offsets[0], 0.0)
    self.assertEqual(self.db.count_embeddings(), 0)

  def test_insert_window_with_embedding(self):
    rng = np.random.default_rng(1)
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    emb = rng.normal(size=EMBEDDING_SIZE).astype(np.float32)
    win_id = self.db.insert_window(rec_id, [0.0, 5.0], embedding=emb)
    self.db.commit()
    self.assertEqual(self.db.count_embeddings(), 1)
    got_emb = self.db.get_embedding(win_id)
    np.testing.assert_allclose(got_emb, emb, atol=1e-5)

  def test_get_embedding_missing_raises(self):
    with self.assertRaises(KeyError):
      self.db.get_embedding(999)

  def test_remove_window(self):
    rng = np.random.default_rng(2)
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    emb = rng.normal(size=EMBEDDING_SIZE).astype(np.float32)
    win_id = self.db.insert_window(rec_id, [0.0, 5.0], embedding=emb)
    self.db.commit()
    self.db.remove_window(win_id)
    self.db.commit()
    self.assertEqual(self.db.count_embeddings(), 0)
    with self.assertRaises(KeyError):
      self.db.get_window(win_id)

  def test_insert_window_wrong_dim_raises(self):
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    bad_emb = np.zeros(EMBEDDING_SIZE + 1, dtype=np.float32)
    with self.assertRaises(ValueError):
      self.db.insert_window(rec_id, [0.0, 5.0], embedding=bad_emb)

  def test_insert_windows_batch(self):
    rng = np.random.default_rng(3)
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    n = 5
    windows_batch = [
        {'recording_id': rec_id, 'offsets': [float(i), float(i + 5)]}
        for i in range(n)
    ]
    embeddings_batch = rng.normal(size=(n, EMBEDDING_SIZE)).astype(np.float32)
    ids = self.db.insert_windows_batch(
        windows_batch, embeddings_batch, handle_duplicates='allow'
    )
    self.db.commit()
    self.assertLen(ids, n)
    self.assertEqual(self.db.count_embeddings(), n)
    for index, window_id in enumerate(ids):
      self.assertSequenceEqual(
        self.db.get_window(window_id).offsets,
        windows_batch[index]['offsets'],
      )
      np.testing.assert_allclose(
        self.db.get_embedding(window_id), embeddings_batch[index], atol=1e-5
      )

  def test_insert_windows_batch_allow_extra_column(self):
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    windows_batch = [
        {
            'recording_id': rec_id,
            'offsets': [float(index), float(index + 5)],
            'timestamp': f'2026-09-01T00:00:0{index}+00:00',
        }
        for index in range(2)
    ]

    ids = self.db.insert_windows_batch(
        windows_batch, handle_duplicates='allow'
    )
    self.db.commit()

    for index, window_id in enumerate(ids):
      self.assertEqual(
          self.db.get_window(window_id).timestamp,
          windows_batch[index]['timestamp'],
      )

  def test_insert_windows_batch_allow_rejects_inconsistent_columns(self):
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)

    with self.assertRaisesRegex(ValueError, 'same columns'):
      self.db.insert_windows_batch(
          [
              {'recording_id': rec_id, 'offsets': [0.0, 5.0]},
              {
                  'recording_id': rec_id,
                  'offsets': [5.0, 10.0],
                  'timestamp': '2026-09-01T00:00:00+00:00',
              },
          ],
          handle_duplicates='allow',
      )

  def test_insert_windows_batch_skip_all_duplicates(self):
    rng = np.random.default_rng(33)
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    windows_batch = [
        {'recording_id': rec_id, 'offsets': [float(i), float(i + 5)]}
        for i in range(3)
    ]
    embeddings_batch = rng.normal(size=(3, EMBEDDING_SIZE)).astype(np.float32)
    inserted_ids = self.db.insert_windows_batch(
        windows_batch, embeddings_batch, handle_duplicates='allow'
    )
    self.db.commit()

    skipped_ids = self.db.insert_windows_batch(
        windows_batch, embeddings_batch, handle_duplicates='skip'
    )
    self.db.commit()

    self.assertSequenceEqual(list(skipped_ids), list(inserted_ids))
    self.assertEqual(self.db.count_embeddings(), 3)

  def test_get_embeddings_batch(self):
    rng = np.random.default_rng(4)
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    embs = rng.normal(size=(3, EMBEDDING_SIZE)).astype(np.float32)
    ids = []
    for i, emb in enumerate(embs):
      ids.append(self.db.insert_window(rec_id, [float(i), float(i + 5)], embedding=emb))
    self.db.commit()
    got = self.db.get_embeddings_batch(ids)
    self.assertEqual(got.shape, (3, EMBEDDING_SIZE))
    np.testing.assert_allclose(got, embs, atol=1e-5)

  # ------------------------------------------------------------------
  # Annotations
  # ------------------------------------------------------------------

  def test_insert_and_get_annotation(self):
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    ann_id = self.db.insert_annotation(
        recording_id=rec_id,
        offsets=[0.0, 5.0],
        label='bird',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='human',
    )
    self.db.commit()
    ann = self.db.get_annotation(ann_id)
    self.assertEqual(ann.label, 'bird')
    self.assertEqual(ann.label_type, datatypes.LabelType.POSITIVE)

  def test_get_window_annotations(self):
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    win_id = self.db.insert_window(rec_id, [0.0, 5.0])
    self.db.insert_annotation(
        recording_id=rec_id,
        offsets=[1.0, 4.0],
        label='sparrow',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='human',
    )
    # Non-overlapping annotation — should not appear.
    self.db.insert_annotation(
        recording_id=rec_id,
        offsets=[10.0, 15.0],
        label='crow',
        label_type=datatypes.LabelType.POSITIVE,
        provenance='human',
    )
    self.db.commit()
    anns = self.db.get_window_annotations(win_id)
    self.assertLen(anns, 1)
    self.assertEqual(anns[0].label, 'sparrow')

  # ------------------------------------------------------------------
  # Filtering / queries
  # ------------------------------------------------------------------

  def test_match_window_ids_no_filter(self):
    test_utils.insert_random_embeddings(
        self.db, emb_dim=EMBEDDING_SIZE, num_embeddings=10, seed=0
    )
    ids = self.db.match_window_ids()
    self.assertLen(ids, 10)

  def test_match_window_ids_with_project_filter(self):
    test_utils.insert_random_embeddings(
        self.db, emb_dim=EMBEDDING_SIZE, num_embeddings=30, seed=1
    )
    projects = self.db.get_all_projects()
    ids = self.db.match_window_ids(
        deployments_filter=config_dict.create(
            eq=dict(project=projects[0])
        )
    )
    self.assertGreater(len(ids), 0)

  def test_match_window_ids_limit(self):
    test_utils.insert_random_embeddings(
        self.db, emb_dim=EMBEDDING_SIZE, num_embeddings=20, seed=2
    )
    ids = self.db.match_window_ids(limit=5)
    self.assertLen(ids, 5)

  def test_get_all_labels_and_count(self):
    rng = np.random.default_rng(5)
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(filename='f.wav', deployment_id=dep_id)
    for label in ('owl', 'owl', 'crow'):
      i = rng.integers(0, 100).item()
      self.db.insert_annotation(
          recording_id=rec_id,
          offsets=[float(i), float(i + 5)],
          label=label,
          label_type=datatypes.LabelType.POSITIVE,
          provenance='test',
          handle_duplicates='allow',
      )
    self.db.commit()
    labels = self.db.get_all_labels()
    self.assertIn('owl', labels)
    self.assertIn('crow', labels)
    counts = self.db.count_each_label()
    self.assertGreaterEqual(counts['owl'], 1)

  # ------------------------------------------------------------------
  # Extra columns
  # ------------------------------------------------------------------

  def test_add_extra_column_and_query(self):
    self.db.add_extra_table_column('recordings', 'sample_rate', int)
    dep_id = self.db.insert_deployment(name='d', project='p')
    rec_id = self.db.insert_recording(
        filename='f.wav', deployment_id=dep_id, sample_rate=44100
    )
    self.db.commit()
    rec = self.db.get_recording(rec_id)
    self.assertEqual(rec.sample_rate, 44100)

  def test_add_extra_column_is_idempotent(self):
    self.db.add_extra_table_column('deployments', 'region', str)
    self.db.add_extra_table_column('deployments', 'region', str)  # no-op
    self.db.commit()

  # ------------------------------------------------------------------
  # Search
  # ------------------------------------------------------------------

  def test_search_returns_results(self):
    rng = np.random.default_rng(6)
    test_utils.insert_random_embeddings(
        self.db, emb_dim=EMBEDDING_SIZE, num_embeddings=50, seed=3
    )
    query = rng.normal(size=EMBEDDING_SIZE).astype(np.float32)
    results = self.db.search(query, search_list_size=5)
    result_list = list(results)
    self.assertLen(result_list, 5)
    # Scores should be in descending order.
    scores = [r.sort_score for r in result_list]
    self.assertEqual(scores, sorted(scores, reverse=True))

  def test_search_exact(self):
    rng = np.random.default_rng(7)
    test_utils.insert_random_embeddings(
        self.db, emb_dim=EMBEDDING_SIZE, num_embeddings=20, seed=4
    )
    query = rng.normal(size=EMBEDDING_SIZE).astype(np.float32)
    results = self.db.search(query, search_list_size=3, approximate=False)
    self.assertLen(list(results), 3)

  # ------------------------------------------------------------------
  # Thread split
  # ------------------------------------------------------------------

  def test_thread_split_creates_independent_connection(self):
    dep_id = self.db.insert_deployment(name='d', project='p')
    self.db.commit()
    db2 = self.db.thread_split()
    # Verify the split DB can read what the original committed.
    dep = db2.get_deployment(dep_id)
    self.assertEqual(dep.name, 'd')
    db2.db.close()

  # ------------------------------------------------------------------
  # Rollback
  # ------------------------------------------------------------------

  def test_rollback_discards_uncommitted_changes(self):
    self.db.insert_deployment(name='will_be_rolled_back', project='p')
    self.db.rollback()
    deps = self.db.get_all_deployments()
    names = [d.name for d in deps]
    self.assertNotIn('will_be_rolled_back', names)


if __name__ == '__main__':
  absltest.main()
