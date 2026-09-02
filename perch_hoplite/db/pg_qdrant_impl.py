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

"""PostgreSQL + Qdrant database implementation.

Relational data (deployments, recordings, windows, annotations, metadata) is
stored in PostgreSQL.  Vector storage and approximate nearest-neighbor search
are handled by Qdrant.

Differences from the SQLite/USearch implementation:
  - Placeholders are ``%s`` (psycopg2) instead of ``?``.
  - ``offsets`` is stored as ``float8[]`` (native PG array) instead of a
    custom binary BLOB type.
  - The three SQLite user-defined functions (``APPROX_FLOAT_LIST``,
    ``GET_OFFSET_START``, ``GET_OFFSET_END``) become installed PL/pgSQL
    functions created idempotently during ``_setup_tables()``.
  - Auto-increment columns use ``BIGSERIAL``; the new row id is returned via
    ``RETURNING id`` rather than ``cursor.lastrowid``.
  - Qdrant's ``Dot`` metric returns the raw inner product directly (higher is
    better), so no ``1 - score`` inversion is needed in ``search()``.
  - Qdrant persists automatically; ``commit()`` only commits the PG
    transaction.
"""

import collections
from collections.abc import Sequence
import dataclasses
import datetime as dt
import functools
import itertools
import json
import os
import re
from typing import Any, Literal

from absl import logging
from ml_collections import config_dict
import numpy as np
import psycopg2
import psycopg2.errors
import psycopg2.extensions
from perch_hoplite.db import brutalism
from perch_hoplite.db import datatypes
from perch_hoplite.db import interface
from perch_hoplite.db import score_functions
from perch_hoplite.db import search_results
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

QDRANT_CONFIG_KEY = 'qdrant_config'
DEFAULT_COLLECTION_NAME = 'hoplite_embeddings'

PYTHON_TYPE_TO_PG_TYPE: dict[type, str] = {
    int: 'BIGINT',
    float: 'DOUBLE PRECISION',
    str: 'TEXT',
    bytes: 'BYTEA',
    dt.datetime: 'TEXT',
    list: 'float8[]',
}

# Maps ``information_schema.columns.data_type`` values to Python types.
PG_TYPE_TO_PYTHON_TYPE: dict[str, type] = {
    'bigint': int,
    'integer': int,
    'double precision': float,
    'real': float,
    'numeric': float,
    'text': str,
    'character varying': str,
    'bytea': bytes,
    'timestamp without time zone': dt.datetime,
    'timestamp with time zone': dt.datetime,
    'ARRAY': list,  # covers float8[]
}

_DEFAULT_COLUMNS: dict[str, set[str]] = {
    'deployments': {'id', 'name', 'project', 'latitude', 'longitude'},
    'recordings': {'id', 'filename', 'datetime', 'deployment_id'},
    'windows': {'id', 'recording_id', 'offsets'},
    'annotations': {
        'id',
        'recording_id',
        'offsets',
        'label',
        'label_type',
        'provenance',
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_default_qdrant_config(
    embedding_dim: int,
) -> config_dict.ConfigDict:
  """Return a sensible default Qdrant config for a given embedding dimension."""
  cfg = config_dict.ConfigDict()
  cfg.embedding_dim = embedding_dim
  cfg.dtype = 'float32'
  cfg.metric_name = 'DOT'
  cfg.collection_name = DEFAULT_COLLECTION_NAME
  # 'memory' is convenient for testing; change to 'local' or 'remote' for prod.
  cfg.mode = 'memory'
  return cfg


def is_valid_sql_identifier(name: str) -> bool:
  """Return True if *name* is a safe SQL identifier."""
  if not name or not isinstance(name, str):
    return False
  return re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name) is not None


def normalize_sql_value(value: Any) -> Any:
  """Normalize a Python value to a type accepted by psycopg2."""
  if isinstance(value, (list, tuple, np.ndarray)):
    return [normalize_sql_value(v) for v in value]
  if isinstance(value, datatypes.LabelType):
    return value.value
  if isinstance(value, dt.datetime):
    return value.isoformat()
  if isinstance(value, np.integer):
    return int(value)
  if isinstance(value, np.floating):
    return float(value)
  return value


def format_sql_insert_values_pg(
    **kwargs: Any,
) -> tuple[str, str, list[Any]]:
  """Build column/placeholder/value components for a ``INSERT`` statement.

  Returns:
    Tuple of ``(columns_str, placeholders_str, values)``.
  """
  for key in kwargs:
    if not is_valid_sql_identifier(key):
      raise ValueError(f'`{key}` is not a valid SQL identifier.')

  columns = list(kwargs.keys())
  placeholders = ['%s'] * len(columns)
  values = normalize_sql_value(list(kwargs.values()))

  return f"({', '.join(columns)})", f"({', '.join(placeholders)})", values


def format_sql_update_on_conflict_pg(*args: str) -> str:
  """Build the ``DO UPDATE SET`` / ``DO NOTHING`` part of ON CONFLICT clauses."""
  for key in args:
    if not is_valid_sql_identifier(key):
      raise ValueError(f'`{key}` is not a valid SQL identifier.')

  if not args:
    return 'DO NOTHING'
  update_clauses = ', '.join(f'{key} = excluded.{key}' for key in args)
  return f'DO UPDATE SET {update_clauses}'


def format_sql_where_conditions_pg(
    filter_dict: config_dict.ConfigDict | None = None,
    table_prefix: str | None = None,
) -> tuple[str, list[Any]]:
  r"""Build WHERE conditions from a filter ConfigDict (psycopg2 ``%s`` style).

  Returns:
    Tuple of ``(conditions_str, values)``.
  """
  if table_prefix and not is_valid_sql_identifier(table_prefix):
    raise ValueError(
        f'Table prefix `{table_prefix}` is not a valid SQL identifier.'
    )

  if not filter_dict:
    return '', []

  supported_ops = {
      'eq', 'neq', 'lt', 'lte', 'gt', 'gte',
      'isin', 'notin', 'range', 'approx',
  }

  conditions: list[str] = []
  values: list[Any] = []

  for op_name, op_filters in filter_dict.items():
    if op_name not in supported_ops:
      raise ValueError(
          f'Unsupported operation: `{op_name}`. Supported: {supported_ops}.'
      )
    if not isinstance(op_filters, config_dict.ConfigDict):
      raise ValueError(f'`{op_name}` value must be a ConfigDict.')

    for key, value in op_filters.items():
      column = f'{table_prefix}.{key}' if table_prefix else key

      if not is_valid_sql_identifier(key):
        raise ValueError(
            f'Column `{column}` is not a valid SQL identifier.'
        )

      value = normalize_sql_value(value)

      if op_name == 'eq':
        if key == 'offsets':
          logging.warning(
              "Do not apply `eq` to `offsets` unless you know what you're "
              'doing. Use `approx` instead to avoid floating-point errors.'
          )
        if value is None:
          conditions.append(f'{column} IS NULL')
        else:
          conditions.append(f'{column} = %s')
          values.append(value)
      elif op_name == 'neq':
        if value is None:
          conditions.append(f'{column} IS NOT NULL')
        else:
          conditions.append(f'{column} != %s')
          values.append(value)
      elif op_name == 'lt':
        conditions.append(f'{column} < %s')
        values.append(value)
      elif op_name == 'lte':
        conditions.append(f'{column} <= %s')
        values.append(value)
      elif op_name == 'gt':
        conditions.append(f'{column} > %s')
        values.append(value)
      elif op_name == 'gte':
        conditions.append(f'{column} >= %s')
        values.append(value)
      elif op_name == 'isin':
        if not isinstance(value, list):
          raise ValueError(f'`{op_name}` value must be a list.')
        placeholders = ['%s'] * len(value)
        conditions.append(f"{column} IN ({', '.join(placeholders)})")
        values.extend(value)
      elif op_name == 'notin':
        if not isinstance(value, list):
          raise ValueError(f'`{op_name}` value must be a list.')
        placeholders = ['%s'] * len(value)
        conditions.append(f"{column} NOT IN ({', '.join(placeholders)})")
        values.extend(value)
      elif op_name == 'range':
        if not isinstance(value, list) or len(value) != 2:
          raise ValueError(f'`{op_name}` value must be a list of 2 elements.')
        conditions.append(f'{column} BETWEEN %s AND %s')
        values.extend(value)
      elif op_name == 'approx':
        if key == 'offsets':
          # Use the installed PL/pgSQL helper; pass the Python list and
          # psycopg2 will adapt it to float8[].
          conditions.append(f'approx_float_list({column}, %s) = TRUE')
        else:
          conditions.append(f'ABS({column} - %s) < 1e-6')
        values.append(value)

  return ' AND '.join(conditions), values


def _get_window_query_components_pg(
    deployments_filter: config_dict.ConfigDict | None = None,
    recordings_filter: config_dict.ConfigDict | None = None,
    windows_filter: config_dict.ConfigDict | None = None,
    annotations_filter: config_dict.ConfigDict | None = None,
) -> tuple[str, str, list[Any]]:
  """Construct FROM, WHERE, and values for window SQL queries (PG dialect)."""
  query_tables: set[str] = {'windows'}
  if annotations_filter:
    query_tables.add('annotations')
  if recordings_filter:
    query_tables.add('recordings')
  if deployments_filter:
    query_tables.update({'recordings', 'deployments'})

  from_clause = 'FROM windows'
  if 'annotations' in query_tables:
    from_clause += (
        ' JOIN annotations'
        ' ON windows.recording_id = annotations.recording_id'
        ' AND get_offset_start(annotations.offsets)'
        ' < get_offset_end(windows.offsets)'
        ' AND get_offset_end(annotations.offsets)'
        ' > get_offset_start(windows.offsets)'
    )
  if 'recordings' in query_tables:
    from_clause += ' JOIN recordings ON windows.recording_id = recordings.id'
  if 'deployments' in query_tables:
    from_clause += (
        ' JOIN deployments ON recordings.deployment_id = deployments.id'
    )

  conditions, values = tuple(
      zip(*[
          format_sql_where_conditions_pg(
              deployments_filter, table_prefix='deployments'
          ),
          format_sql_where_conditions_pg(
              recordings_filter, table_prefix='recordings'
          ),
          format_sql_where_conditions_pg(
              windows_filter, table_prefix='windows'
          ),
          format_sql_where_conditions_pg(
              annotations_filter, table_prefix='annotations'
          ),
      ])
  )
  conditions_str = ' AND '.join(c for c in conditions if c)
  values_flat = list(itertools.chain.from_iterable(values))
  where_clause = f'WHERE {conditions_str}' if conditions_str else ''
  return from_clause, where_clause, values_flat


def _make_qdrant_client(qdrant_cfg: config_dict.ConfigDict) -> QdrantClient:
  """Instantiate a QdrantClient from the given config."""
  mode = qdrant_cfg.mode
  if mode == 'memory':
    return QdrantClient(':memory:')
  if mode == 'local':
    return QdrantClient(path=qdrant_cfg.path)
  if mode == 'remote':
    return QdrantClient(
        url=qdrant_cfg.url,
        api_key=os.environ.get('HOPLITE_QDRANT_API_KEY'),
    )
  raise ValueError(
      f"Unknown Qdrant mode: '{mode}'. Expected 'memory', 'local', or"
      " 'remote'."
  )


def _create_qdrant_collection(
    qc: QdrantClient, collection_name: str, embedding_dim: int, metric_name: str
) -> None:
  """Create the Qdrant collection if needed."""
  existing = {c.name for c in qc.get_collections().collections}
  if collection_name in existing:
    return
  metric = getattr(qmodels.Distance, metric_name)
  qc.create_collection(
      collection_name=collection_name,
      vectors_config=qmodels.VectorParams(size=embedding_dim, distance=metric),
  )


def _validate_qdrant_collection(
    qc: QdrantClient, collection_name: str, embedding_dim: int
) -> None:
  """Check that an existing collection matches the expected vector size."""
  collection_info = qc.get_collection(collection_name)
  vectors_config = collection_info.config.params.vectors
  if not isinstance(vectors_config, qmodels.VectorParams):
    raise ValueError(
        'Named or multi-vector Qdrant collections are not supported by'
        f' PgQdrantDB: {collection_name}.'
    )
  if vectors_config.size != embedding_dim:
    raise ValueError(
        'Qdrant collection dimension mismatch for'
        f" '{collection_name}': expected {embedding_dim},"
        f' found {vectors_config.size}. Use a different collection name or'
        ' delete the existing collection.'
    )


def _make_qdrant_point_struct(
    point_id: int, vector: np.ndarray
) -> qmodels.PointStruct:
  """Build a Qdrant point from an id and vector."""
  return qmodels.PointStruct(
      id=int(point_id),
      vector=vector.astype(np.float32).tolist(),
      payload={},
  )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PgQdrantDB(interface.HopliteDBInterface):
  """Hoplite database backed by PostgreSQL (relational) + Qdrant (vectors).

  Attributes:
    _db_dsn: PostgreSQL DSN string (stored for ``thread_split``).
    _qdrant_cfg: Full Qdrant config (stored for ``thread_split``).
    db: Active psycopg2 connection.
    qc: Active Qdrant client.
    _collection_name: Name of the Qdrant vector collection.
    _embedding_dim: Embedding vector dimension.
    _embedding_dtype: NumPy dtype used when casting vectors for storage.
    _cursor: Reused psycopg2 cursor (created lazily).
    _readonly: Whether the database was opened in read-only mode.
  """

  # Stored for thread_split / config access.
  _db_dsn: str
  _qdrant_cfg: config_dict.ConfigDict

  # Active connections.
  db: psycopg2.extensions.connection
  qc: QdrantClient

  # Configuration derived from qdrant_cfg.
  _collection_name: str
  _embedding_dim: int
  _embedding_dtype: type[Any]

  # Dynamic state.
  _cursor: psycopg2.extensions.cursor | None = None
  _readonly: bool = False

  # ------------------------------------------------------------------
  # Class methods
  # ------------------------------------------------------------------

  @classmethod
  def create(  # pyrefly: ignore[bad-override]
      cls,
      db_dsn: str,
      qdrant_cfg: config_dict.ConfigDict | None = None,
      readonly: bool = False,
  ) -> 'PgQdrantDB':
    """Connect to (and optionally initialise) the database.

    Args:
      db_dsn: PostgreSQL DSN, e.g.
        ``'postgresql://user:pass@localhost:5432/mydb'``.
      qdrant_cfg: Qdrant configuration ConfigDict. If *None*, the config is
        loaded from ``hoplite_metadata``.
      readonly: If *True*, skip table creation and operate read-only.

    Raises:
      ValueError: If *qdrant_cfg* is inconsistent with the stored config, or
        if no config exists and none was provided.

    Returns:
      A new ``PgQdrantDB`` instance.
    """
    db = psycopg2.connect(db_dsn)
    cursor = db.cursor()

    if not readonly:
      cls._setup_tables(cursor)
      db.commit()

    # Load or validate qdrant_cfg from metadata.
    cursor.execute(
        'SELECT value FROM hoplite_metadata WHERE key = %s',
        (QDRANT_CONFIG_KEY,),
    )
    row = cursor.fetchone()
    stored_cfg = config_dict.ConfigDict(json.loads(row[0])) if row else None

    if stored_cfg is not None and qdrant_cfg is not None:
      if stored_cfg != qdrant_cfg:
        raise ValueError(
            'A qdrant_cfg was provided, but a different one is already stored'
            ' in the database.'
        )
    if stored_cfg is not None:
      qdrant_cfg = stored_cfg
    elif qdrant_cfg is None:
      raise ValueError(
          'No qdrant_cfg was found in the database and none was provided.'
      )

    collection_name = qdrant_cfg.collection_name
    embedding_dim = int(qdrant_cfg.embedding_dim)

    # Build the Qdrant client and ensure the collection exists.
    qc = _make_qdrant_client(qdrant_cfg)
    if readonly:
      existing = {c.name for c in qc.get_collections().collections}
      if collection_name not in existing:
        raise FileNotFoundError(
            f"Qdrant collection '{collection_name}' not found."
        )
      _validate_qdrant_collection(qc, collection_name, embedding_dim)
    else:
      _create_qdrant_collection(
          qc, collection_name, embedding_dim, qdrant_cfg.metric_name
      )
      _validate_qdrant_collection(qc, collection_name, embedding_dim)

    hoplite_db = cls(
        _db_dsn=db_dsn,
        _qdrant_cfg=qdrant_cfg,
        db=db,
        qc=qc,
        _collection_name=collection_name,
        _embedding_dim=embedding_dim,
        _embedding_dtype=np.float32,
        _readonly=readonly,
    )

    if not readonly and stored_cfg is None:
      hoplite_db.insert_metadata(QDRANT_CONFIG_KEY, qdrant_cfg)
      hoplite_db.commit()

    return hoplite_db

  # ------------------------------------------------------------------
  # Private helpers
  # ------------------------------------------------------------------

  @staticmethod
  def _setup_tables(
      cursor: psycopg2.extensions.cursor,
  ) -> None:
    """Install PL/pgSQL functions and create tables (idempotent)."""

    # Skip if already initialised.
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_name = 'windows' AND table_schema = 'public'
        """)
    if cursor.fetchone() is not None:
      return

    # PL/pgSQL helpers ---------------------------------------------------

    cursor.execute("""
        CREATE OR REPLACE FUNCTION approx_float_list(a float8[], b float8[])
        RETURNS BOOLEAN AS $$
          SELECT bool_and(ABS(a[i] - b[i]) < 1e-6)
          FROM generate_subscripts(a, 1) AS i;
        $$ LANGUAGE sql IMMUTABLE;
        """)

    cursor.execute("""
        CREATE OR REPLACE FUNCTION get_offset_start(o float8[])
        RETURNS float8 AS $$ SELECT o[1]; $$ LANGUAGE sql IMMUTABLE;
        """)

    cursor.execute("""
        CREATE OR REPLACE FUNCTION get_offset_end(o float8[])
        RETURNS float8 AS $$ SELECT o[2]; $$ LANGUAGE sql IMMUTABLE;
        """)

    # Tables -------------------------------------------------------------

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hoplite_metadata (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS deployments (
            id        BIGSERIAL PRIMARY KEY,
            name      TEXT NOT NULL,
            project   TEXT NOT NULL,
            latitude  DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            UNIQUE (name, project)
        )
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recordings (
            id            BIGSERIAL PRIMARY KEY,
            filename      TEXT NOT NULL,
            datetime      TEXT,
            deployment_id BIGINT REFERENCES deployments(id) ON DELETE CASCADE,
            UNIQUE (filename, deployment_id)
        )
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS windows (
            id           BIGSERIAL PRIMARY KEY,
            recording_id BIGINT NOT NULL REFERENCES recordings(id)
                             ON DELETE CASCADE,
            offsets      float8[] NOT NULL
        )
        """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id           BIGSERIAL PRIMARY KEY,
            recording_id BIGINT NOT NULL REFERENCES recordings(id)
                             ON DELETE CASCADE,
            offsets      float8[] NOT NULL,
            label        TEXT NOT NULL,
            label_type   INTEGER NOT NULL,
            provenance   TEXT NOT NULL
        )
        """)

    # Indexes ------------------------------------------------------------

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_annotations
        ON annotations(recording_id, label, label_type, provenance)
        """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_labels
        ON annotations(label, label_type, provenance)
        """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_recordings_deployment_id
        ON recordings(deployment_id)
        """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_windows_recording_id
        ON windows(recording_id)
        """)

  def _get_cursor(self) -> psycopg2.extensions.cursor:
    if self._cursor is None:
      self._cursor = self.db.cursor()
    return self._cursor

  @functools.cached_property
  def _extra_table_columns(self) -> dict[str, dict[str, type[Any]]]:
    """Discover non-default columns in each table via information_schema."""
    tables = list(_DEFAULT_COLUMNS.keys())
    extra: dict[str, dict[str, type[Any]]] = {t: {} for t in tables}
    cursor = self._get_cursor()
    for table in tables:
      cursor.execute(
          """
          SELECT column_name, data_type
          FROM information_schema.columns
          WHERE table_schema = 'public' AND table_name = %s
          ORDER BY ordinal_position
          """,
          (table,),
      )
      for col_name, data_type in cursor.fetchall():
        if col_name in _DEFAULT_COLUMNS[table]:
          continue
        py_type = PG_TYPE_TO_PYTHON_TYPE.get(data_type)
        if py_type is None:
          raise ValueError(
              f'Unsupported column type {data_type!r} for column'
              f' {col_name!r} in table {table!r}.'
          )
        extra[table][col_name] = py_type
    return extra

  def _delete_qdrant_points_for_windows(
      self, window_ids: Sequence[int]
  ) -> None:
    """Remove the given window IDs from the Qdrant collection."""
    if not window_ids:
      return
    self.qc.delete(
        collection_name=self._collection_name,
        points_selector=qmodels.PointIdsList(points=list(window_ids)),
    )

  def _upsert_embeddings(
      self, window_ids: Sequence[int], embeddings: np.ndarray
  ) -> None:
    """Upsert a batch of embeddings into Qdrant."""
    if len(window_ids) == 0:
      return
    self.qc.upsert(
        collection_name=self._collection_name,
        points=[
            _make_qdrant_point_struct(window_id, embedding)
            for window_id, embedding in zip(window_ids, embeddings)
        ],
    )

  # ------------------------------------------------------------------
  # HopliteDBInterface — lifecycle
  # ------------------------------------------------------------------

  def add_extra_table_column(
      self,
      table_name: str,
      column_name: str,
      column_type: type[Any],
  ) -> None:
    """Add an extra column to a table in the database."""
    if table_name not in _DEFAULT_COLUMNS:
      raise ValueError(f'Table `{table_name}` does not exist.')
    if not is_valid_sql_identifier(column_name):
      raise ValueError(
          f'Column `{column_name}` is not a valid SQL identifier.'
      )
    if not isinstance(column_type, type):
      raise ValueError(f'Column type `{column_type}` must be a type.')
    if column_type not in PYTHON_TYPE_TO_PG_TYPE:
      raise ValueError(
          f'Column type `{column_type.__name__}` is not supported. Use one'
          f' of: {", ".join(t.__name__ for t in PYTHON_TYPE_TO_PG_TYPE)}.'
      )

    # Check if it already exists.
    cursor = self._get_cursor()
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = %s
          AND column_name = %s
        """,
        (table_name, column_name),
    )
    if cursor.fetchone() is not None:
      return

    pg_type = PYTHON_TYPE_TO_PG_TYPE[column_type]
    cursor.execute(f'ALTER TABLE {table_name} ADD COLUMN {column_name} {pg_type}')

    # Bust the cache so the next access re-reads the schema.
    self.__dict__.pop('_extra_table_columns', None)

  def get_extra_table_columns(self) -> dict[str, dict[str, type[Any]]]:
    return self._extra_table_columns

  def commit(self) -> None:
    self.db.commit()
    if self._cursor is not None:
      self._cursor.close()
      self._cursor = None

  def rollback(self) -> None:
    self.db.rollback()
    if self._cursor is not None:
      self._cursor.close()
      self._cursor = None

  def thread_split(self) -> 'PgQdrantDB':
    return self.create(
        db_dsn=self._db_dsn,
        qdrant_cfg=self._qdrant_cfg,
        readonly=self._readonly,
    )

  # ------------------------------------------------------------------
  # HopliteDBInterface — metadata
  # ------------------------------------------------------------------

  def insert_metadata(self, key: str, value: config_dict.ConfigDict) -> None:
    cursor = self._get_cursor()
    cursor.execute(
        """
        INSERT INTO hoplite_metadata (key, value)
        VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = excluded.value
        """,
        (key, value.to_json()),
    )

  def get_metadata(self, key: str | None) -> config_dict.ConfigDict:
    cursor = self._get_cursor()
    if key is None:
      cursor.execute('SELECT key, value FROM hoplite_metadata')
      return config_dict.ConfigDict(
          {k: json.loads(v) for k, v in cursor.fetchall()}
      )
    cursor.execute(
        'SELECT value FROM hoplite_metadata WHERE key = %s', (key,)
    )
    row = cursor.fetchone()
    if row is None:
      raise KeyError(f'Metadata key not found: {key}')
    return config_dict.ConfigDict(json.loads(row[0]))

  def remove_metadata(self, key: str | None) -> None:
    cursor = self._get_cursor()
    if key is None:
      cursor.execute('DELETE FROM hoplite_metadata')
      return
    cursor.execute('DELETE FROM hoplite_metadata WHERE key = %s', (key,))
    if cursor.rowcount == 0:
      raise KeyError(f'Metadata key not found: {key}')

  # ------------------------------------------------------------------
  # HopliteDBInterface — deployments
  # ------------------------------------------------------------------

  def insert_deployment(
      self,
      name: str,
      project: str,
      latitude: float | None = None,
      longitude: float | None = None,
      **kwargs: Any,
  ) -> int:
    for key, value in kwargs.items():
      if key not in self._extra_table_columns['deployments']:
        self.add_extra_table_column('deployments', key, type(value))

    cursor = self._get_cursor()
    columns_str, placeholders_str, values = format_sql_insert_values_pg(
        name=name,
        project=project,
        latitude=latitude,
        longitude=longitude,
        **kwargs,
    )
    update_clause = format_sql_update_on_conflict_pg(
        'latitude',
        'longitude',
        *self._extra_table_columns['deployments'].keys(),
    )
    cursor.execute(
        f"""
        INSERT INTO deployments {columns_str}
        VALUES {placeholders_str}
        ON CONFLICT (name, project)
        {update_clause}
        RETURNING id
        """,
        values,
    )
    row = cursor.fetchone()
    if row is None:
      raise RuntimeError('Error inserting the deployment into the database.')
    return row[0]

  def get_deployment(self, deployment_id: int) -> datatypes.Deployment:
    deployment_id = int(deployment_id)
    cursor = self._get_cursor()
    cursor.execute(
        'SELECT * FROM deployments WHERE id = %s', (deployment_id,)
    )
    row = cursor.fetchone()
    if row is None:
      raise KeyError(f'Deployment id not found: {deployment_id}')
    columns = [col.name for col in cursor.description]
    return datatypes.Deployment(**dict(zip(columns, row)))

  def remove_deployment(self, deployment_id: int) -> None:
    deployment_id = int(deployment_id)
    remove_window_ids = self.match_window_ids(
        deployments_filter=config_dict.create(eq=dict(id=deployment_id))
    )
    self._delete_qdrant_points_for_windows(remove_window_ids)
    cursor = self._get_cursor()
    cursor.execute('DELETE FROM deployments WHERE id = %s', (deployment_id,))
    if cursor.rowcount == 0:
      raise KeyError(f'Deployment id not found: {deployment_id}')

  # ------------------------------------------------------------------
  # HopliteDBInterface — recordings
  # ------------------------------------------------------------------

  def insert_recording(
      self,
      filename: str,
      datetime: dt.datetime | None = None,
      deployment_id: int | None = None,
      **kwargs: Any,
  ) -> int:
    for key, value in kwargs.items():
      if key not in self._extra_table_columns['recordings']:
        self.add_extra_table_column('recordings', key, type(value))

    cursor = self._get_cursor()
    columns_str, placeholders_str, values = format_sql_insert_values_pg(
        filename=filename,
        datetime=datetime,
        deployment_id=deployment_id,
        **kwargs,
    )
    update_clause = format_sql_update_on_conflict_pg(
        'datetime',
        *self._extra_table_columns['recordings'].keys(),
    )
    try:
      cursor.execute(
          f"""
          INSERT INTO recordings {columns_str}
          VALUES {placeholders_str}
          ON CONFLICT (filename, deployment_id)
          {update_clause}
          RETURNING id
          """,
          values,
      )
    except psycopg2.errors.ForeignKeyViolation as e:
      raise RuntimeError(
          'Error inserting the recording into the database.'
          ' Check that the deployment_id exists.'
      ) from e
    except psycopg2.Error as e:
      raise RuntimeError(
          'Error inserting the recording into the database.'
      ) from e

    row = cursor.fetchone()
    if row is None:
      raise RuntimeError('Error inserting the recording into the database.')
    return row[0]

  def get_recording(self, recording_id: int) -> datatypes.Recording:
    recording_id = int(recording_id)
    cursor = self._get_cursor()
    cursor.execute('SELECT * FROM recordings WHERE id = %s', (recording_id,))
    row = cursor.fetchone()
    if row is None:
      raise KeyError(f'Recording id not found: {recording_id}')
    columns = [col.name for col in cursor.description]
    recording = datatypes.Recording(**dict(zip(columns, row)))
    if recording.datetime is not None and isinstance(recording.datetime, str):
      recording.datetime = dt.datetime.fromisoformat(recording.datetime)  # pyrefly: ignore[bad-argument-type]
    return recording

  def remove_recording(self, recording_id: int) -> None:
    recording_id = int(recording_id)
    remove_window_ids = self.match_window_ids(
        recordings_filter=config_dict.create(eq=dict(id=recording_id))
    )
    self._delete_qdrant_points_for_windows(remove_window_ids)
    cursor = self._get_cursor()
    cursor.execute('DELETE FROM recordings WHERE id = %s', (recording_id,))
    if cursor.rowcount == 0:
      raise KeyError(f'Recording id not found: {recording_id}')

  # ------------------------------------------------------------------
  # HopliteDBInterface — windows
  # ------------------------------------------------------------------

  def insert_window(
      self,
      recording_id: int,
      offsets: list[float],
      embedding: np.ndarray | None = None,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
      **kwargs: Any,
  ) -> int:
    if embedding is not None and embedding.shape[-1] != self._embedding_dim:
      raise ValueError(
          f'Incorrect embedding dimension. Expected {self._embedding_dim},'
          f' got {embedding.shape[-1]}.'
      )

    duplicate_id = self._handle_window_duplicates(
        recording_id, offsets, handle_duplicates
    )
    if duplicate_id is not None:
      return duplicate_id

    for key, value in kwargs.items():
      if key not in self._extra_table_columns['windows']:
        self.add_extra_table_column('windows', key, type(value))

    cursor = self._get_cursor()
    columns_str, placeholders_str, values = format_sql_insert_values_pg(
        recording_id=recording_id,
        offsets=list(offsets),
        **kwargs,
    )
    try:
      cursor.execute(
          f"""
          INSERT INTO windows {columns_str}
          VALUES {placeholders_str}
          RETURNING id
          """,
          values,
      )
    except psycopg2.errors.ForeignKeyViolation as e:
      raise RuntimeError(
          'Error inserting the window into the database.'
          ' Check that the recording_id exists.'
      ) from e
    except psycopg2.Error as e:
      raise RuntimeError(
          'Error inserting the window into the database.'
      ) from e

    row = cursor.fetchone()
    if row is None:
      raise RuntimeError('Error inserting the window into the database.')
    window_id = row[0]

    if embedding is not None:
      self._upsert_embeddings([window_id], embedding[None, :])
    return window_id

  def insert_windows_batch(
      self,
      windows_batch: Sequence[dict[str, Any]],
      embeddings_batch: np.ndarray | None = None,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
  ) -> Sequence[int]:
    """Insert a batch of windows, uploading all embeddings to Qdrant at once."""
    if (
        embeddings_batch is not None
        and embeddings_batch.shape[-1] != self._embedding_dim
    ):
      raise ValueError(
          f'Incorrect embedding dimension. Expected {self._embedding_dim},'
          f' got {embeddings_batch.shape[-1]}.'
      )

    if handle_duplicates == 'allow':
      if not windows_batch:
        return []

      column_names = list(windows_batch[0])
      for column_name in column_names:
        if not is_valid_sql_identifier(column_name):
          raise ValueError(f'`{column_name}` is not a valid SQL identifier.')
      expected_columns = set(column_names)
      for window_kwargs in windows_batch:
        if set(window_kwargs) != expected_columns:
          raise ValueError(
              'All windows in an allow-mode batch must have the same columns.'
          )

      for column_name in column_names:
        if column_name in _DEFAULT_COLUMNS['windows']:
          continue
        value = windows_batch[0][column_name]
        if column_name not in self._extra_table_columns['windows']:
          self.add_extra_table_column('windows', column_name, type(value))

      row_placeholders = f"({', '.join(['%s'] * len(column_names))})"
      values = []
      for window_kwargs in windows_batch:
        values.extend(
            normalize_sql_value(window_kwargs[column_name])
            for column_name in column_names
        )

      cursor = self._get_cursor()
      try:
        cursor.execute(
            f"""
            INSERT INTO windows ({', '.join(column_names)})
            VALUES {', '.join([row_placeholders] * len(windows_batch))}
            RETURNING id
            """,
            values,
        )
      except psycopg2.errors.ForeignKeyViolation as e:
        raise RuntimeError(
            'Error inserting the window into the database.'
            ' Check that the recording_id exists.'
        ) from e
      except psycopg2.Error as e:
        raise RuntimeError(
            'Error inserting the window into the database.'
        ) from e

      window_ids = [row[0] for row in cursor.fetchall()]
      if len(window_ids) != len(windows_batch):
        raise RuntimeError('Error inserting the windows into the database.')
      if embeddings_batch is not None:
        self._upsert_embeddings(window_ids, embeddings_batch.astype(np.float32))
      return window_ids

    # Check for intra-batch duplicates (mirrors SQLite implementation).
    if handle_duplicates != 'allow':
      for i in range(len(windows_batch)):
        for j in range(i + 1, len(windows_batch)):
          if windows_batch[i]['recording_id'] == windows_batch[j][
              'recording_id'
          ] and np.allclose(
              windows_batch[i]['offsets'],
              windows_batch[j]['offsets'],
              rtol=0.0,
              atol=1e-6,
          ):
            raise RuntimeError(
                'Duplicates found in `windows_batch`, but this is not'
                ' supported unless `handle_duplicates` is "allow"'
                f' (handle_duplicates = "{handle_duplicates}").'
            )

    window_ids: list[int] = [-1] * len(windows_batch)
    if handle_duplicates in ['overwrite', 'skip', 'error']:
      keep_idx: list[int] = []
      remove_window_ids: set[int] = set()

      for idx, window_kwargs in enumerate(windows_batch):
        matches = self.get_all_windows(
            filter=config_dict.create(
                eq=dict(recording_id=window_kwargs['recording_id']),
                approx=dict(offsets=window_kwargs['offsets']),
            )
        )
        if matches:
          if handle_duplicates == 'overwrite':
            for match in matches:
              remove_window_ids.add(match.id)
            keep_idx.append(idx)
          elif handle_duplicates == 'skip':
            window_ids[idx] = matches[0].id
          elif handle_duplicates == 'error':
            raise RuntimeError(
                f'Duplicate window found (id = {matches[0].id}), but'
                ' `handle_duplicates` is set to "error".'
            )
        else:
          keep_idx.append(idx)

      for wid in remove_window_ids:
        self.remove_window(wid)
    else:
      keep_idx = list(range(len(windows_batch)))

    # Insert windows (without embeddings) so we can get their IDs.
    for idx in keep_idx:
      window_ids[idx] = self.insert_window(
          embedding=None,
          handle_duplicates='allow',
          **windows_batch[idx],
      )

    # Batch-upload all embeddings to Qdrant in one call.
    if embeddings_batch is not None:
      ids_to_upload = np.array(window_ids)[keep_idx]
      vecs_to_upload = embeddings_batch[keep_idx].astype(np.float32)
      if len(ids_to_upload) > 0:
        self._upsert_embeddings(ids_to_upload, vecs_to_upload)

    return window_ids

  def get_window(
      self,
      window_id: int,
      include_embedding: bool = False,
  ) -> datatypes.Window:
    window_id = int(window_id)
    cursor = self._get_cursor()
    cursor.execute('SELECT * FROM windows WHERE id = %s', (window_id,))
    row = cursor.fetchone()
    if row is None:
      raise KeyError(f'Window id not found: {window_id}')
    columns = [col.name for col in cursor.description]
    window = datatypes.Window(embedding=None, **dict(zip(columns, row)))
    if include_embedding:
      window.embedding = self.get_embedding(window_id)
    return window

  def get_window_annotations(
      self, window_id: int, label: str | None = None
  ) -> Sequence[datatypes.Annotation]:
    """Get all annotations intersecting the given window."""
    window = self.get_window(window_id)
    w_start, w_end = window.offsets
    cursor = self._get_cursor()
    query = """
        SELECT *
        FROM annotations
        WHERE recording_id = %s
          AND get_offset_start(offsets) < %s
          AND get_offset_end(offsets) > %s
        """
    params: list[Any] = [window.recording_id, w_end, w_start]
    if label is not None:
      query += ' AND label = %s'
      params.append(label)
    cursor.execute(query, params)
    columns = [col.name for col in cursor.description]
    annotations = []
    for row in cursor.fetchall():
      ann = datatypes.Annotation(**dict(zip(columns, row)))
      ann.label_type = datatypes.LabelType(ann.label_type)
      annotations.append(ann)
    return annotations

  def remove_window(self, window_id: int) -> None:
    window_id = int(window_id)
    cursor = self._get_cursor()
    cursor.execute('DELETE FROM windows WHERE id = %s', (window_id,))
    if cursor.rowcount == 0:
      raise KeyError(f'Window id not found: {window_id}')
    self._delete_qdrant_points_for_windows([window_id])

  # ------------------------------------------------------------------
  # HopliteDBInterface — embeddings
  # ------------------------------------------------------------------

  def get_embedding(self, window_id: int) -> np.ndarray:
    window_id = int(window_id)
    results = self.qc.retrieve(
        collection_name=self._collection_name,
        ids=[window_id],
        with_vectors=True,
        with_payload=False,
    )
    if not results:
      raise KeyError(f'Embedding vector not found for window id: {window_id}')
    vector = results[0].vector
    return np.array(vector, dtype=self._embedding_dtype)

  def get_embeddings_batch(
      self,
      window_ids: Sequence[int],
  ) -> np.ndarray:
    int_ids = [int(wid) for wid in window_ids]
    results = self.qc.retrieve(
        collection_name=self._collection_name,
        ids=int_ids,
        with_vectors=True,
        with_payload=False,
    )
    if len(results) != len(int_ids):
      found = {r.id for r in results}
      missing = [wid for wid in int_ids if wid not in found]
      raise KeyError(
          f'Embedding vectors not found for window ids: {missing}'
      )
    id_to_vector = {r.id: r.vector for r in results}
    embeddings = [id_to_vector[wid] for wid in int_ids]
    return np.array(embeddings, dtype=self._embedding_dtype)

  def count_embeddings(self) -> int:
    info = self.qc.get_collection(self._collection_name)
    return info.points_count or 0

  # ------------------------------------------------------------------
  # HopliteDBInterface — annotations
  # ------------------------------------------------------------------

  def insert_annotation(
      self,
      recording_id: int,
      offsets: list[float],
      label: str,
      label_type: datatypes.LabelType,
      provenance: str,
      handle_duplicates: Literal[
          'allow', 'overwrite', 'skip', 'error'
      ] = 'error',
      **kwargs: Any,
  ) -> int:
    duplicate_id = self._handle_annotation_duplicates(
        recording_id, offsets, label, label_type, provenance, handle_duplicates
    )
    if duplicate_id is not None:
      return duplicate_id

    for key, value in kwargs.items():
      if key not in self._extra_table_columns['annotations']:
        self.add_extra_table_column('annotations', key, type(value))

    cursor = self._get_cursor()
    columns_str, placeholders_str, values = format_sql_insert_values_pg(
        recording_id=recording_id,
        offsets=list(offsets),
        label=label,
        label_type=label_type,
        provenance=provenance,
        **kwargs,
    )
    try:
      cursor.execute(
          f"""
          INSERT INTO annotations {columns_str}
          VALUES {placeholders_str}
          RETURNING id
          """,
          values,
      )
    except psycopg2.errors.ForeignKeyViolation as e:
      raise RuntimeError(
          'Error inserting the annotation into the database.'
          ' Check that the recording_id exists.'
      ) from e
    except psycopg2.Error as e:
      raise RuntimeError(
          'Error inserting the annotation into the database.'
      ) from e

    row = cursor.fetchone()
    if row is None:
      raise RuntimeError('Error inserting the annotation into the database.')
    return row[0]

  def get_annotation(self, annotation_id: int) -> datatypes.Annotation:
    annotation_id = int(annotation_id)
    cursor = self._get_cursor()
    cursor.execute(
        'SELECT * FROM annotations WHERE id = %s', (annotation_id,)
    )
    row = cursor.fetchone()
    if row is None:
      raise KeyError(f'Annotation id not found: {annotation_id}')
    columns = [col.name for col in cursor.description]
    ann = datatypes.Annotation(**dict(zip(columns, row)))
    ann.label_type = datatypes.LabelType(ann.label_type)
    return ann

  def remove_annotation(self, annotation_id: int) -> None:
    annotation_id = int(annotation_id)
    cursor = self._get_cursor()
    cursor.execute(
        'DELETE FROM annotations WHERE id = %s', (annotation_id,)
    )
    if cursor.rowcount == 0:
      raise KeyError(f'Annotation id not found: {annotation_id}')

  # ------------------------------------------------------------------
  # HopliteDBInterface — queries
  # ------------------------------------------------------------------

  def match_window_ids(
      self,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      windows_filter: config_dict.ConfigDict | None = None,
      annotations_filter: config_dict.ConfigDict | None = None,
      limit: int | None = None,
  ) -> Sequence[int]:
    cursor = self._get_cursor()
    select_clause = (
        'SELECT DISTINCT windows.id'
        if annotations_filter
        else 'SELECT windows.id'
    )
    from_clause, where_clause, values = _get_window_query_components_pg(
        deployments_filter=deployments_filter,
        recordings_filter=recordings_filter,
        windows_filter=windows_filter,
        annotations_filter=annotations_filter,
    )
    limit_clause = f'LIMIT {int(limit)}' if limit is not None else ''
    cursor.execute(
        f'{select_clause} {from_clause} {where_clause} {limit_clause}',
        values,
    )
    return [row[0] for row in cursor.fetchall()]

  def get_all_projects(self) -> Sequence[str]:
    cursor = self._get_cursor()
    cursor.execute(
        'SELECT DISTINCT project FROM deployments ORDER BY project'
    )
    return [row[0] for row in cursor.fetchall()]

  def get_all_deployments(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Deployment]:
    cursor = self._get_cursor()
    conditions_str, values = format_sql_where_conditions_pg(filter)
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor.execute(f'SELECT * FROM deployments {where_clause}', values)
    columns = [col.name for col in cursor.description]
    return [
        datatypes.Deployment(**dict(zip(columns, row)))
        for row in cursor.fetchall()
    ]

  def get_all_recordings(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Recording]:
    cursor = self._get_cursor()
    conditions_str, values = format_sql_where_conditions_pg(filter)
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor.execute(f'SELECT * FROM recordings {where_clause}', values)
    columns = [col.name for col in cursor.description]
    recordings = []
    for row in cursor.fetchall():
      rec = datatypes.Recording(**dict(zip(columns, row)))
      if rec.datetime is not None and isinstance(rec.datetime, str):
        rec.datetime = dt.datetime.fromisoformat(rec.datetime)  # pyrefly: ignore[bad-argument-type]
      recordings.append(rec)
    return recordings

  def get_all_windows(
      self,
      include_embedding: bool = False,
      deployments_filter: config_dict.ConfigDict | None = None,
      recordings_filter: config_dict.ConfigDict | None = None,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
      annotations_filter: config_dict.ConfigDict | None = None,
  ) -> Sequence[datatypes.Window]:
    cursor = self._get_cursor()
    select_clause = (
        'SELECT DISTINCT windows.*'
        if annotations_filter
        else 'SELECT windows.*'
    )
    from_clause, where_clause, values = _get_window_query_components_pg(
        deployments_filter=deployments_filter,
        recordings_filter=recordings_filter,
        windows_filter=filter,
        annotations_filter=annotations_filter,
    )
    cursor.execute(
        f'{select_clause} {from_clause} {where_clause}', values
    )
    columns = [col.name for col in cursor.description]
    windows = []
    for row in cursor.fetchall():
      window = datatypes.Window(embedding=None, **dict(zip(columns, row)))
      if include_embedding:
        window.embedding = self.get_embedding(window.id)
      windows.append(window)
    return windows

  def get_all_annotations(
      self,
      filter: config_dict.ConfigDict | None = None,  # pylint: disable=redefined-builtin
  ) -> Sequence[datatypes.Annotation]:
    cursor = self._get_cursor()
    conditions_str, values = format_sql_where_conditions_pg(filter)
    where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor.execute(f'SELECT * FROM annotations {where_clause}', values)
    columns = [col.name for col in cursor.description]
    annotations = []
    for row in cursor.fetchall():
      ann = datatypes.Annotation(**dict(zip(columns, row)))
      ann.label_type = datatypes.LabelType(ann.label_type)
      annotations.append(ann)
    return annotations

  def get_all_labels(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> Sequence[str]:
    cursor = self._get_cursor()
    if label_type is None:
      where_clause, values = '', []
    else:
      filter_dict = config_dict.create(eq=dict(label_type=label_type))
      conditions_str, values = format_sql_where_conditions_pg(filter_dict)
      where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor.execute(
        f"""
        SELECT DISTINCT label
        FROM annotations
        {where_clause}
        ORDER BY label
        """,
        values,
    )
    return [row[0] for row in cursor.fetchall()]

  def count_each_label(
      self,
      label_type: datatypes.LabelType | None = None,
  ) -> collections.Counter[str]:
    cursor = self._get_cursor()
    if label_type is None:
      where_clause, values = '', []
    else:
      filter_dict = config_dict.create(eq=dict(label_type=label_type))
      conditions_str, values = format_sql_where_conditions_pg(filter_dict)
      where_clause = f'WHERE {conditions_str}' if conditions_str else ''
    cursor.execute(
        f"""
        SELECT label, COUNT(*)
        FROM (
            SELECT DISTINCT recording_id, offsets, label, label_type
            FROM annotations
            {where_clause}
        ) AS sub
        GROUP BY label
        ORDER BY label
        """,
        values,
    )
    return collections.Counter({row[0]: row[1] for row in cursor.fetchall()})

  # ------------------------------------------------------------------
  # HopliteDBInterface — embedding config
  # ------------------------------------------------------------------

  def get_embedding_dim(self) -> int:
    return self._embedding_dim

  def get_embedding_dtype(self) -> type[Any]:
    return self._embedding_dtype

  # ------------------------------------------------------------------
  # HopliteDBInterface — search
  # ------------------------------------------------------------------

  def search(
      self,
      query_embedding: np.ndarray,
      search_list_size: int,
      approximate: bool = True,
      target_score: float | None = None,
      score_fn_name: str = 'dot',
      **kwargs: Any,
  ) -> search_results.TopKSearchResults:
    """Search for nearest neighbours using Qdrant.

    When *approximate* is False and *target_score* is not None, falls back to
    the brute-force implementation in the base class.  Otherwise Qdrant
    performs exact or approximate ANN search depending on *approximate*.

    Note: Qdrant's ``Dot`` metric returns the raw inner product as ``score``
    (higher = more similar), so no ``1 - score`` inversion is applied.
    """
    if not approximate and target_score is not None:
      return super().search(
          query_embedding=query_embedding,
          search_list_size=search_list_size,
          approximate=approximate,
          target_score=target_score,
          score_fn_name=score_fn_name,
          **kwargs,
      )

    if target_score is not None:
      raise ValueError(
          'Approximate search does not support target_score sampling.'
      )

    qdrant_response = self.qc.query_points(
        collection_name=self._collection_name,
        query=query_embedding.astype(np.float32).tolist(),
        limit=search_list_size,
        with_payload=False,
        search_params=qmodels.SearchParams(exact=not approximate),
    )
    qdrant_hits = qdrant_response.points

    top_k = search_results.TopKSearchResults(top_k=search_list_size)
    for hit in qdrant_hits:
      top_k.update(search_results.SearchResult(hit.id, hit.score))

    score_fn = score_functions.get_score_fn(score_fn_name)
    return brutalism.rerank(query_embedding, top_k, self, score_fn)
