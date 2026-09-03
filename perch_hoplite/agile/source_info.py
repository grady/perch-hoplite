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

"""Audio source information handling."""

from collections.abc import Iterator, Sequence
import dataclasses
from pathlib import PurePosixPath
from urllib.parse import urlparse

from etils import epath
from ml_collections import config_dict
from perch_hoplite import audio_io
from perch_hoplite.db import datatypes
import tqdm


def _is_s3_path(path: str) -> bool:
  return path.startswith('s3://')


@dataclasses.dataclass(frozen=True)
class _S3Path:
  path: str

  def as_posix(self) -> str:
    return self.path

  def relative_to(self, other: '_S3Path') -> '_S3Path':
    prefix = other.path.rstrip('/') + '/'
    if not self.path.startswith(prefix):
      raise ValueError(f'{self.path} is not under {other.path}')
    return _S3Path(self.path[len(prefix) :])


def _iter_s3_filepaths(base_path: str, file_glob: str) -> tuple[_S3Path, ...]:
  """Lists S3 objects under base_path and filters by file_glob."""
  parsed = urlparse(base_path)
  bucket = parsed.netloc
  prefix = parsed.path.lstrip('/').rstrip('/')
  if not bucket:
    raise ValueError(f'Invalid S3 base path: {base_path}')
  try:
    import boto3
  except ImportError as exc:
    raise ImportError('S3 support requires installing boto3.') from exc

  s3_client = boto3.client('s3', **audio_io._s3_storage_options_from_env())
  paginator = s3_client.get_paginator('list_objects_v2')
  list_kwargs = {'Bucket': bucket}
  if prefix:
    list_kwargs['Prefix'] = f'{prefix}/'

  matches = []
  for page in paginator.paginate(**list_kwargs):
    for obj in page.get('Contents', ()):
      key = obj.get('Key', '')
      if not key:
        continue
      rel_key = key[len(prefix) + 1 :] if prefix else key
      if PurePosixPath(rel_key).match(file_glob):
        matches.append(_S3Path(f's3://{bucket}/{key}'))
  return tuple(matches)


@dataclasses.dataclass
class SourceId:
  """Source information for pairing audio with embeddings."""

  dataset_name: str
  file_id: str
  offset_s: float
  shard_len_s: float
  filepath: str
  sample_rate_hz: int

  def to_id(self):
    return f'{self.dataset_name}:{self.file_id}:{self.offset_s}'

  def deployment_name_from_file_id(self) -> str:
    """Returns the deployment name for this source."""
    if '/' in self.file_id:
      return self.file_id.split('/')[0]
    else:
      return self.dataset_name


@dataclasses.dataclass
class AudioSourceConfig(datatypes.HopliteConfig):
  """Configuration for embedding a collection of audio sources.

  Attributes:
    dataset_name: Name of the dataset. (Must be unique for each set of files.)
    base_path: Root directory of the dataset.
    file_glob: Glob pattern for the audio files.
    min_audio_len_s: Minimum audio length to process.
    target_sample_rate_hz: Target sample rate for audio. If -2, use the
      embedding model's declared sample rate. If -1, use the file's native
      sample rate. If > 0, resample to the specified rate.
    shard_len_s: If not None, shard the audio into segments of this length.
    max_shards_per_file: If not None, maximum number of shards per file.
  """

  dataset_name: str
  base_path: str
  file_glob: str
  min_audio_len_s: float = 1.0
  target_sample_rate_hz: int = -2
  shard_len_s: float | None = 60.0
  max_shards_per_file: int | None = None

  def is_compatible(self, other: 'AudioSourceConfig') -> bool:
    """Returns True if other is expected to produce comparable embeddings."""
    return (
        self.dataset_name == other.dataset_name
        and self.target_sample_rate_hz == other.target_sample_rate_hz
        and self.min_audio_len_s == other.min_audio_len_s
    )

FileEntry = tuple[AudioSourceConfig, str, epath.Path | _S3Path]


@dataclasses.dataclass
class AudioSources(datatypes.HopliteConfig):
  """A collection of AudioSourceConfig, with SourceId iterator."""

  audio_globs: tuple[AudioSourceConfig, ...]
  _file_info_cache: dict[str, tuple[float, int]] = dataclasses.field(
      default_factory=dict, repr=False, hash=False
  )

  def __post_init__(self):
    dataset_names = set(
        audio_glob.dataset_name for audio_glob in self.audio_globs
    )
    if len(dataset_names) < len(self.audio_globs):
      raise ValueError('Dataset names must be unique.')

  def to_config_dict(self) -> config_dict.ConfigDict:
    """Convert to a config dict."""
    globs = tuple(g.to_config_dict() for g in self.audio_globs)
    return config_dict.ConfigDict({'audio_globs': globs})

  @classmethod
  def from_config_dict(cls, config: config_dict.ConfigDict) -> 'AudioSources':
    """Create an AudioSources from a config dict."""
    globs = tuple(
        AudioSourceConfig(**audio_glob) for audio_glob in config.audio_globs
    )
    return cls(audio_globs=globs)

  def merge_update(self, other: 'AudioSources') -> 'AudioSources':
    """Update the audio sources with the new sources.

    Args:
      other: The new audio sources.

    Raises:
      ValueError if any audio globs appear in both and are incompatible.
    Returns:
      A new AudioSources object with the merged audio globs. In case of a
      conflict, the values in the 'other' audio glob takes precedence.
    """
    my_globs = {g.dataset_name: g for g in self.audio_globs}
    other_globs = {g.dataset_name: g for g in other.audio_globs}
    for dataset_name, my_glob in my_globs.items():
      if dataset_name not in other_globs:
        other_globs[dataset_name] = my_glob
      elif not other_globs[dataset_name].is_compatible(my_glob):
        raise ValueError(
            f'Audio glob {other_globs[dataset_name]} '
            f'is incompatible with {my_glob}.'
        )
    return AudioSources(tuple(other_globs.values()))

  def _get_audio_len_s_and_sample_rate_hz(
      self, filepath: epath.Path
  ) -> tuple[float, int]:
    """Returns the audio length and sample rate of the audio file."""
    filepath_posix = filepath.as_posix()
    if filepath_posix in self._file_info_cache:
      return self._file_info_cache[filepath_posix]
    audio_len_s, sample_rate_hz = audio_io.get_file_length_s_and_sample_rate(
        filepath_posix
    )
    self._file_info_cache[filepath_posix] = (audio_len_s, sample_rate_hz)
    return audio_len_s, sample_rate_hz

  def iterate_files(
      self,
      target_dataset_name: str | None = None,
  ) -> Iterator[FileEntry]:
    """Yields each matching file without inspecting audio metadata."""
    for glob in self.audio_globs:
      if (
          target_dataset_name is not None
          and glob.dataset_name != target_dataset_name
      ):
        continue
      # If base_path is a URL, the posix path may not match the original string.
      if _is_s3_path(glob.base_path):
        base_path = _S3Path(glob.base_path.rstrip('/'))
        filepaths = _iter_s3_filepaths(glob.base_path, glob.file_glob)
      else:
        base_path = epath.Path(glob.base_path)
        filepaths = tuple(base_path.glob(glob.file_glob))

      for filepath in filepaths:
        try:
          file_id = filepath.relative_to(base_path).as_posix()
        except ValueError:
          file_id = filepath.as_posix()[len(base_path.as_posix()) + 1 :]
        yield glob, file_id, filepath

  def iterate_all_sources(
      self,
      target_dataset_name: str | None = None,
      files: Sequence[FileEntry] | None = None,
  ) -> Iterator[SourceId]:
    """Yields all sources for all datasets (or just a single dataset).

    Args:
      target_dataset_name: If not None, only yield sources for this dataset.

    Yields:
      SourceId objects.
    """
    if files is None:
      files = tuple(self.iterate_files(target_dataset_name))
    file_iterator = files
    for glob, file_id, filepath in tqdm.tqdm(file_iterator):
      shard_len_s = glob.shard_len_s
      max_shards_per_file = glob.max_shards_per_file

      audio_len_s, sample_rate_hz = self._get_audio_len_s_and_sample_rate_hz(
          filepath
      )
      if shard_len_s is None:
        yield SourceId(
            dataset_name=glob.dataset_name,
            file_id=file_id,
            offset_s=0,
            shard_len_s=-1,
            filepath=filepath.as_posix(),
            sample_rate_hz=sample_rate_hz,
        )
        continue

      # Otherwise, need to emit sharded SourceId's.
      if audio_len_s <= 0:
        continue
      shard_num = 0
      while max_shards_per_file is None or shard_num < max_shards_per_file:
        offset_s = shard_num * shard_len_s
        if offset_s >= audio_len_s:
          break
        # When the new shard extends beyond the end of the audio, and the
        # shard will be shorter than the minimum audio length, we are done.
        if (
            offset_s + shard_len_s > audio_len_s
            and audio_len_s - offset_s < glob.min_audio_len_s
        ):
          break
        yield SourceId(
            dataset_name=glob.dataset_name,
            file_id=file_id,
            offset_s=offset_s,
            shard_len_s=shard_len_s,
            filepath=filepath.as_posix(),
            sample_rate_hz=sample_rate_hz,
        )
        shard_num += 1
