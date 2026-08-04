# PostgreSQL + Qdrant Implementation Plan

## Goal
Create `PgQdrantDB` — a new `HopliteDBInterface` implementation that replaces:
- **SQLite** → **PostgreSQL** (relational data: deployments, recordings, windows, annotations, metadata)
- **USearch** → **Qdrant** (vector storage and ANN search)

---

## Codebase Summary

### Key files
- `perch_hoplite/db/interface.py` — abstract base class (`HopliteDBInterface`, ~30 abstract methods)
- `perch_hoplite/db/sqlite_usearch_impl.py` — reference implementation (1693 lines)
- `perch_hoplite/db/datatypes.py` — `Deployment`, `Recording`, `Window`, `Annotation`, `LabelType`
- `perch_hoplite/db/db_loader.py` — factory / `DBConfig.load_db()`
- `perch_hoplite/db/tests/test_utils.py` — shared test helpers; `DB_TYPES` tuple drives parameterized tests

### Data model (4 relational tables + 1 KV + 1 vector store)
| Table | Notable columns |
|---|---|
| `hoplite_metadata` | `key TEXT PK, value TEXT` (JSON) |
| `deployments` | `id, name, project, latitude, longitude` |
| `recordings` | `id, filename, datetime, deployment_id FK` |
| `windows` | `id, recording_id FK, offsets` |
| `annotations` | `id, recording_id FK, offsets, label, label_type INT, provenance` |

`offsets` is currently a 2-element `float64` list (binary-encoded BLOB in SQLite).

---

## Challenges & Design Decisions

### 1. SQL dialect
| SQLite | PostgreSQL |
|---|---|
| `?` placeholders | `%s` (psycopg2) |
| `INTEGER PRIMARY KEY AUTOINCREMENT` | `BIGSERIAL PRIMARY KEY` |
| `FLOAT_LIST` custom type (binary blob) | `float8[]` native array |
| `PRAGMA foreign_keys = ON` | default ON; use `DEFERRABLE` if needed |
| `PRAGMA journal_mode = WAL` | n/a |
| `PRAGMA table_info(t)` | `information_schema.columns` |
| `ON CONFLICT (col) DO NOTHING/UPDATE` | same syntax ✓ |
| Custom Python functions via `create_function()` | PL/pgSQL functions installed at `_setup_tables()` |

### 2. `offsets` storage
Use **`float8[]`** (PostgreSQL native array). Advantages:
- No custom adapter/converter code needed
- Direct arithmetic: `offsets[1]` = start, `offsets[2]` = end (1-indexed in PG)
- Simplifies the JOIN condition in `_get_window_query_components`

Existing `approx` filter for offsets becomes:
```sql
ABS(offsets[1] - %s) < 1e-6 AND ABS(offsets[2] - %s) < 1e-6
```

### 3. Custom SQL functions
The three SQLite user-defined functions must become installed PL/pgSQL functions:

```sql
CREATE OR REPLACE FUNCTION approx_float_list(a float8[], b float8[])
RETURNS BOOLEAN AS $$
  SELECT bool_and(ABS(a[i] - b[i]) < 1e-6)
  FROM generate_subscripts(a, 1) AS i;
$$ LANGUAGE sql IMMUTABLE;

CREATE OR REPLACE FUNCTION get_offset_start(offsets float8[])
RETURNS float8 AS $$ SELECT offsets[1]; $$ LANGUAGE sql IMMUTABLE;

CREATE OR REPLACE FUNCTION get_offset_end(offsets float8[])
RETURNS float8 AS $$ SELECT offsets[2]; $$ LANGUAGE sql IMMUTABLE;
```

These are installed during `_setup_tables()` and are idempotent (`CREATE OR REPLACE`).

### 4. Connection management
Use **`psycopg2`** (sync, matches existing synchronous patterns). Connection is stored as `self.db`. `thread_split()` creates a new `PgQdrantDB` via `cls.create(...)` with the same DSN/config.

### 5. USearch → Qdrant mapping
| USearch op | Qdrant equivalent |
|---|---|
| `Index(ndim, metric='IP', ...)` | `client.recreate_collection(name, vectors_config=VectorParams(size, distance=Dot))` |
| `ui.add(ids, vectors)` | `client.upsert(collection, [PointStruct(id=id, vector=v)])` |
| `ui.remove(ids)` | `client.delete(collection, PointIdsList(points=ids))` |
| `ui.contains(id)` | `client.retrieve(collection, ids=[id])` → non-empty |
| `ui.get(id)` | `client.retrieve(collection, ids=[id], with_vectors=True)` |
| `ui.search(query, count=k)` | `client.search(collection, query_vector=query, limit=k)` |
| `ui.size` | `client.get_collection(collection).points_count` |
| `ui.save()` | no-op (Qdrant auto-persists) |
| `ui.load()` | no-op |

**Score handling**: USearch IP returns distance `= 1 - dot`, so `score = 1 - d`. Qdrant `Dot` metric returns the raw inner product as `score` directly. The `search()` method must skip the `1.0 - d` inversion.

**Qdrant connection modes** (controlled by config):
- Local/on-disk: `QdrantClient(path="/some/dir")` — no server, for single-process use
- Remote: `QdrantClient(host=..., port=...)` — production
- In-memory: `QdrantClient(":memory:")` — for testing

### 6. Schema introspection for `_extra_table_columns`
Replace `PRAGMA table_info(t)` with:
```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = %s AND table_schema = 'public'
```

Map PostgreSQL data types back to Python types:
```python
PG_TYPE_TO_PYTHON_TYPE = {
    'bigint': int, 'integer': int,
    'double precision': float, 'real': float,
    'text': str, 'character varying': str,
    'bytea': bytes,
    'timestamp without time zone': dt.datetime,
    'ARRAY': list,  # float8[]
}
```

---

## Implementation Plan

### Phase 1 — New file skeleton + config
**File**: `perch_hoplite/db/pg_qdrant_impl.py`

1. Define `QDRANT_CONFIG_KEY = 'qdrant_config'` and default config helper.
2. Define type mapping dicts (`PYTHON_TYPE_TO_PG_TYPE`, `PG_TYPE_TO_PYTHON_TYPE`).
3. Port `is_valid_sql_identifier` (identical).
4. Port `normalize_sql_value` (identical).
5. Write `format_sql_insert_values_pg` — same logic as SQLite version but `%s` placeholders.
6. Write `format_sql_update_on_conflict_pg` — same logic (PostgreSQL uses same `ON CONFLICT` syntax).
7. Write `format_sql_where_conditions_pg` — same logic with `%s`; `approx` for `offsets` uses `ABS(offsets[1]-%s)<1e-6 AND ABS(offsets[2]-%s)<1e-6`.
8. Write `_get_window_query_components_pg` — port from SQLite version, use `%s`, use `get_offset_start`/`get_offset_end` PL/pgSQL functions for the annotation-window overlap JOIN condition.

### Phase 2 — `PgQdrantDB` class structure
```python
@dataclasses.dataclass
class PgQdrantDB(interface.HopliteDBInterface):
    db_dsn: str              # PostgreSQL DSN
    collection_name: str     # Qdrant collection name
    db: psycopg2.connection
    qc: qdrant_client.QdrantClient
    _embedding_dim: int
    _embedding_dtype: str    # e.g. 'float32'
    _cursor: psycopg2.cursor | None = None
    _readonly: bool = False
```

### Phase 3 — `create()` classmethod
1. Accept `db_dsn`, `qdrant_config` (ConfigDict), `readonly=False`.
2. Open psycopg2 connection.
3. Run `_setup_tables()` (DDL + PL/pgSQL functions) if not readonly.
4. Retrieve/validate qdrant_config from metadata table.
5. Connect Qdrant client (local path or remote host from config).
6. Create Qdrant collection if it doesn't exist.
7. Return `PgQdrantDB(...)`.

### Phase 4 — `_setup_tables()`
Install in this order:
1. PL/pgSQL helper functions (`approx_float_list`, `get_offset_start`, `get_offset_end`).
2. `hoplite_metadata` table.
3. `deployments` table.
4. `recordings` table (FK to deployments with `ON DELETE CASCADE`).
5. `windows` table — `offsets float8[] NOT NULL` (FK to recordings).
6. `annotations` table — `offsets float8[] NOT NULL`, `label_type INTEGER` (FK to recordings).
7. Indexes (`idx_annotations`, `idx_labels`, `idx_recordings_deployment_id`, `idx_windows_recording_id`).

Use `IF NOT EXISTS` throughout for idempotency.

### Phase 5 — Implement all abstract methods
Port each method from `SQLiteUSearchDB`, swapping:
- `?` → `%s`
- USearch calls → Qdrant calls
- `PRAGMA table_info` → `information_schema.columns`
- Binary blob offsets → `float8[]` literals (pass as Python `list[float]`)
- `cursor.lastrowid` → `cursor.fetchone()[0]` with `RETURNING id`
- `sqlite3.Error` → `psycopg2.Error`

Key methods:
- `insert_deployment / get_deployment / remove_deployment`
- `insert_recording / get_recording / remove_recording`
- `insert_window / insert_windows_batch / get_window / remove_window` (+ Qdrant upsert/delete)
- `get_embedding / get_embeddings_batch` (Qdrant retrieve with vectors)
- `insert_annotation / get_annotation / remove_annotation`
- `count_embeddings` → `qc.get_collection(...).points_count`
- `match_window_ids / get_all_windows / get_all_deployments / get_all_recordings / get_all_annotations`
- `get_all_projects / get_all_labels / count_each_label`
- `insert_metadata / get_metadata / remove_metadata`
- `add_extra_table_column / get_extra_table_columns`
- `commit / rollback` — `db.commit()` / `db.rollback()`; no Qdrant equivalent needed
- `thread_split` — `cls.create(db_dsn=..., qdrant_config=..., readonly=...)`
- `search` — call `qc.search(...)`, use score directly (no `1 - d` inversion)
- `get_embedding_dim / get_embedding_dtype`

### Phase 6 — Wire up factory + test utilities
1. **`db_loader.py`**: Add `elif self.db_key == 'pg_qdrant': return PgQdrantDB.create(**self.db_config)`.
2. **`test_utils.py`**: Add `'pg_qdrant'` to `DB_TYPES`/`PERSISTENT_DB_TYPES` and a `make_db` branch that creates an in-memory Qdrant + local PostgreSQL connection (or uses a test DSN env var).
3. **`pyproject.toml`**: Add `pg_qdrant` optional extras group:
   ```toml
   [project.optional-dependencies]
   pg_qdrant = ["psycopg2-binary>=2.9,<3.0", "qdrant-client>=1.9,<2.0"]
   ```

### Phase 7 — Tests
1. **`perch_hoplite/db/tests/pg_qdrant_impl_test.py`**: Integration tests against a live PG + Qdrant instance, skipped when env vars are not set.
2. Existing parameterized `hoplite_test.py` will automatically cover `pg_qdrant` once added to `DB_TYPES` in `test_utils.py` (requires running services).

---

## Tricky Edge Cases

| Issue | Resolution |
|---|---|
| `offsets` INSERT — psycopg2 needs a real Python list for `float8[]` | Pass `list(offsets)` directly; psycopg2 adapts Python lists to PG arrays |
| `RETURNING id` for insert → get new row id | Append `RETURNING id` to INSERT and use `cursor.fetchone()[0]` |
| ON CONFLICT upsert — PostgreSQL needs an explicit conflict target | Already have it: `ON CONFLICT (name, project)` etc. |
| `_extra_table_columns` introspection | Use `information_schema.columns`; map PG types to Python types |
| `approx` filter for `offsets` with two `%s` bindings | The helper emits two conditions (one per element); values list gets two entries |
| Qdrant batch upsert order | Collect `PointStruct` objects then call `upsert` once per batch (mirrors USearch batch add) |
| `remove_deployment` / `remove_recording` cascade in PG | `ON DELETE CASCADE` handles relational data; still need to delete Qdrant vectors for affected windows |
| `thread_split` re-creates PG connection | Store DSN + qdrant config on the instance for re-use |
| Qdrant `Dot` score direction | Higher = more similar; do NOT apply `1 - score` in `search()` |
| `get_window_annotations` uses PL/pgSQL functions in JOIN | Reuse `get_offset_start` / `get_offset_end` PG functions |

---

## File Checklist
- [ ] `perch_hoplite/db/pg_qdrant_impl.py` — **new**
- [ ] `perch_hoplite/db/db_loader.py` — add `pg_qdrant` branch
- [ ] `perch_hoplite/db/tests/test_utils.py` — add `pg_qdrant` to `DB_TYPES`
- [ ] `perch_hoplite/db/tests/pg_qdrant_impl_test.py` — **new** (integration tests)
- [ ] `pyproject.toml` — add `pg_qdrant` optional dependency group
