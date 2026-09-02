# Perch Hoplite

![CI](https://github.com/google-research/perch-hoplite/actions/workflows/ci_uv.yml/badge.svg)

Hoplite is a system for storing large volumes of embeddings from machine
perception models. We focus on combining vector search with active learning
workflows, aka [agile modeling](https://arxiv.org/abs/2505.03071).

In brief, agile modeling is a process for rapidly developing classifiers using
embeddings from a pre-trained 'foundation' model. For bioacoustics work, we
find that new classifiers can often be developed for new signals in under
an hour.

**How does it work?**

We first use a bioacoustics model to convert your unlabeled audio data into
embeddings - these are like semantic 'fingerprints' of 5-second audio clips.
Then, you can *search* the embeddings of your data by providing an example of
what you're looking for. You then give feedback on the results - which examples
are and are not what you're looking for. From this feedback, we can quickly
train a classifier. You can then improve on the classifier with
*active learning*: Examine the classifier outputs, provide more feedback, and
re-train the classifier.

A key feature of this workflow is that we pre-compute the embeddings. This
may take a while if you have a large amount of data, but the subsequent search
and classifier training is very efficient.

To get started, load up the following Colab/Jupyter notebooks:

* [`agile/01_embed_audio.ipynb`](perch_hoplite/agile/01_embed_audio.ipynb)
  – Computes embeddings of your audio data.
* [`agile/02_agile_modeling.ipynb`](perch_hoplite/agile/02_agile_modeling.ipynb)
  – Performs search, classification, and active learning.

## Repository Contents

This repository consists of four sub-libraries:

* `db` – The core database functionality for storing embeddings and related
  metadata. The database also handles labels applied to embeddings and vector
  search, both exact and approximate.
* `agile` – Tooling (and example notebooks) for agile modeling on top of the
  Hoplite db layer, combining search and active learning approaches. This
  library includes organizing labeled data and training linear classifiers over
  embeddings, as well as tooling for embedding large datasets.
* `zoo` – A bioacoustics model zoo. A basic wrapper class is provided, and any
  model which can transform windows of audio samples into embeddings can then
  be used in the agile modeling workflow.
* `taxonomy` – A database of taxonomic information, especially for handling
  conversions between the various bird taxonomies.

Each sub-library has its own documentation.

## Installation

We recommend using `uv` or `pip` for installation. `uv` is a fast rust-based
pip-compatible package installer and resolver.

First, install system dependencies for audio processing:
```bash
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg
```

### With `uv`

If you don't have `uv`, you can install it via `pipx install uv` or
`pip install uv`.
If you are developing locally, clone the repository and install in editable
mode:
```bash
git clone https://github.com/google-research/perch-hoplite.git
cd perch-hoplite
# Create and activate a virtual environment
uv venv
source .venv/bin/activate
uv pip install -e .
```

### With `pip`

You can install the latest stable release from PyPI:
```bash
pip install perch-hoplite
```
Or install the latest version from GitHub:
```bash
pip install git+https://github.com/google-research/perch-hoplite.git
```

## Command-Line Embedding

`hoplite embed` runs the embedding stage of
[`agile/01_embed_audio.ipynb`](perch_hoplite/agile/01_embed_audio.ipynb), then
stores the generated embeddings in a Hoplite database. Install the model
dependencies first; `perch_v2` requires the `tf` or `tf-cuda` extra.

Embed a local directory into a SQLite/USearch database:

```bash
hoplite embed \
  --dataset-name field-recordings \
  --audio-path /data/audio \
  --audio-glob '**/*.flac' \
  --db-path /data/hoplite-db
```

To use PostgreSQL with Qdrant, install the `pg_qdrant` extra and supply the
database details:

```bash
hoplite embed \
  --dataset-name field-recordings \
  --audio-path /data/audio \
  --audio-glob '**/*.flac' \
  --db-backend pg_qdrant \
  --db-dsn 'postgresql://user:pass@host:5432/hoplite' \
  --qdrant-url 'https://qdrant.example.org:443' \
  --qdrant-collection-name field-recordings
```

Set `HOPLITE_QDRANT_URL` instead of passing `--qdrant-url` to configure the
endpoint for repeated runs. For endpoints that require authentication, set
`HOPLITE_QDRANT_API_KEY`; it is passed to Qdrant at connection time and is not
stored in PostgreSQL metadata. Use an `https://` URL for TLS with standard
public certificate verification. Do not embed credentials in the URL.

S3-compatible sources use the same command. Prefer environment variables for
credentials rather than command-line flags so credentials do not enter shell
history:

```bash
export HOPLITE_S3_ENDPOINT='https://storage.example.org:9000'
export HOPLITE_S3_ACCESS_KEY='<access-key>'
export HOPLITE_S3_SECRET_KEY='<secret-key>'
hoplite embed \
  --dataset-name field-recordings \
  --audio-path 's3://audio-bucket/recordings' \
  --audio-glob '**/*.flac' \
  --db-path /data/hoplite-db
```

Run `hoplite embed --help` for controls such as sharding, worker count,
duplicate handling, timestamp parsing, and direct S3 or Qdrant overrides. The
command covers embedding only; continue with
[`agile/02_agile_modeling.ipynb`](perch_hoplite/agile/02_agile_modeling.ipynb)
for search and classifier training.

### Running the Tests

After installation, you can run the tests to check that everything is working:
```bash
python -m unittest discover -s perch_hoplite/db/tests -p "*test.py"
python -m unittest discover -s perch_hoplite/taxonomy -p "*test.py"
python -m unittest discover -s perch_hoplite/zoo -p "*test.py"
python -m unittest discover -s perch_hoplite/agile/tests -p "*test.py"
```

The database tests support three backends:
- in-memory / SQLite + USearch tests run by default
- PostgreSQL + Qdrant tests run when `HOPLITE_PG_DSN` is set
- external Qdrant can be selected with `HOPLITE_QDRANT_URL`

Example:
```bash
export HOPLITE_PG_DSN="postgresql://user:pass@host:5432/hoplite_test"
export HOPLITE_QDRANT_URL=http://localhost:6333
python -m unittest perch_hoplite.db.tests.pg_qdrant_impl_test -v
```

### Notes on Dependencies

Tensorflow is required for agile modeling (classifier training) and for using
the [Perch](https://www.kaggle.com/models/google/bird-vocalization-classifier)
or [BirdNET](https://birdnet.cornell.edu/) models, but is not installed by
default. We recommend installing one of the Tensorflow options:

To install with Tensorflow (CPU version):
```bash
pip install 'perch-hoplite[tf]'
```
To install with Tensorflow with CUDA support (for GPU usage):
```bash
pip install 'perch-hoplite[tf-cuda]'
```

The `zoo` library contains wrappers for various bioacoustic models. Some of
these require JAX. To install with JAX dependencies:
```bash
uv pip install -e '.[jax]'
```
or with pip:
```bash
pip install 'perch-hoplite[jax]'
```
If installing with uv in editable mode, you can use
`uv pip install -e '.[tf,jax]'`.

For PostgreSQL + Qdrant support:
```bash
uv pip install -e '.[pg_qdrant]'
```

### S3-Compatible Audio Ingestion

For agile embedding pipelines, audio can be loaded from `s3://...` URIs.
Configure access using environment variables:

```bash
export HOPLITE_S3_ENDPOINT="http://localhost:9000"   # Optional for AWS S3
export HOPLITE_S3_ACCESS_KEY="<access-key>"
export HOPLITE_S3_SECRET_KEY="<secret-key>"
export HOPLITE_S3_REGION="us-east-1"                 # Optional
export HOPLITE_S3_USE_SSL="false"                    # Optional
```

See [perch_hoplite/agile/README.md](perch_hoplite/agile/README.md) for
base-path and file-glob conventions.

## Disclaimer

This is not an officially supported Google product. This project is not eligible
for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
