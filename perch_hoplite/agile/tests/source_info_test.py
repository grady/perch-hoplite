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

"""Tests for source_info."""

import shutil
import tempfile
from unittest import mock

from perch_hoplite.agile import source_info
from perch_hoplite.agile.tests import test_utils

from absl.testing import absltest


class _FakeS3Path:

  def __init__(self, path: str, glob_results=()):
    self._path = path
    self._glob_results = tuple(glob_results)

  def as_posix(self) -> str:
    return self._path

  def glob(self, pattern: str):
    del pattern
    return self._glob_results

  def relative_to(self, other: '_FakeS3Path') -> '_FakeS3Path':
    prefix = other.as_posix().rstrip('/') + '/'
    if not self._path.startswith(prefix):
      raise ValueError('Not a subpath')
    return _FakeS3Path(self._path[len(prefix) :])


class _FakeS3PathNoRelative(_FakeS3Path):

  def relative_to(self, other: '_FakeS3Path') -> '_FakeS3Path':
    del other
    raise ValueError('relative_to not supported')


class SourceInfoTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # `self.create_tempdir()` raises an UnparsedFlagAccessError, which is why
    # we use `tempdir` directly.
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    super().tearDown()
    shutil.rmtree(self.tempdir)

  def test_audio_sources_iteration(self):
    classes = ['pos', 'neg']
    filenames = ['foo', 'bar', 'baz']
    test_utils.make_wav_files(self.tempdir, classes, filenames, file_len_s=6.0)
    audio_sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='pos',
                base_path=self.tempdir,
                file_glob='pos/*.wav',
            ),
            source_info.AudioSourceConfig(
                dataset_name='neg',
                base_path=self.tempdir,
                file_glob='neg/*.wav',
            ),
        )
    )

    with self.subTest('no_sharding'):
      shard_ids = tuple(audio_sources.iterate_all_sources())
      self.assertLen(shard_ids, len(classes) * len(filenames))

    audio_sources.audio_globs[0].shard_len_s = 2.0
    audio_sources.audio_globs[1].shard_len_s = 2.0
    with self.subTest('max_shards_less_than_file_len'):
      audio_sources.audio_globs[0].max_shards_per_file = 2
      audio_sources.audio_globs[1].max_shards_per_file = 2
      shard_ids = tuple(audio_sources.iterate_all_sources())
      self.assertLen(shard_ids, 12)

    with self.subTest('max_shards_matches_file_len'):
      audio_sources.audio_globs[0].max_shards_per_file = 3
      audio_sources.audio_globs[1].max_shards_per_file = 3
      shard_ids = tuple(audio_sources.iterate_all_sources())
      self.assertLen(shard_ids, 18)

    with self.subTest('max_shards_larger_than_file_len'):
      audio_sources.audio_globs[0].max_shards_per_file = 100
      audio_sources.audio_globs[1].max_shards_per_file = 100
      shard_ids = tuple(
          audio_sources.iterate_all_sources(
          )
      )
      self.assertLen(shard_ids, 18)

    with self.subTest('sharded_no_max_shards'):
      audio_sources.audio_globs[0].max_shards_per_file = None
      audio_sources.audio_globs[1].max_shards_per_file = None

      shard_ids = tuple(audio_sources.iterate_all_sources())
      self.assertLen(shard_ids, 18)

  def test_iterate_files_skips_audio_metadata(self):
    test_utils.make_wav_files(
        self.tempdir, ['pos', 'neg'], ['foo'], file_len_s=6.0
    )
    audio_sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='pos', base_path=self.tempdir, file_glob='pos/*.wav'
            ),
            source_info.AudioSourceConfig(
                dataset_name='neg', base_path=self.tempdir, file_glob='neg/*.wav'
            ),
        )
    )

    with mock.patch.object(audio_sources, '_get_audio_len_s_and_sample_rate_hz') as get_info:
      files = tuple(audio_sources.iterate_files(target_dataset_name='pos'))

    self.assertLen(files, 1)
    self.assertEqual(files[0][0].dataset_name, 'pos')
    self.assertEqual(files[0][1], 'pos/foo_pos.wav')
    get_info.assert_not_called()

  def test_audio_glob_compatibility(self):
    audio_glob_1 = source_info.AudioSourceConfig(
        dataset_name='pos',
        base_path='/foo',
        file_glob='*.wav',
        target_sample_rate_hz=16000,
    )
    with self.subTest('different_base_path'):
      glob_diff_base_path = source_info.AudioSourceConfig(
          dataset_name='pos',
          base_path='/bar',
          file_glob='*.wav',
          target_sample_rate_hz=16000,
      )
      self.assertTrue(audio_glob_1.is_compatible(glob_diff_base_path))

    with self.subTest('different_dataset_name'):
      glob_diff_dataset_name = source_info.AudioSourceConfig(
          dataset_name='neg',
          base_path='/foo',
          file_glob='*.wav',
          target_sample_rate_hz=16000,
      )
      self.assertFalse(audio_glob_1.is_compatible(glob_diff_dataset_name))

    with self.subTest('different_target_sample_rate_hz'):
      glob_diff_sample_rate = source_info.AudioSourceConfig(
          dataset_name='pos',
          base_path='/foo',
          file_glob='*.wav',
          target_sample_rate_hz=24000,
      )
      self.assertFalse(audio_glob_1.is_compatible(glob_diff_sample_rate))

    with self.subTest('different_min_audio_len_s'):
      glob_diff_min_audio_len_s = source_info.AudioSourceConfig(
          dataset_name='pos',
          base_path='/foo',
          file_glob='*.wav',
          target_sample_rate_hz=16000,
          min_audio_len_s=2.0,
      )
      self.assertFalse(audio_glob_1.is_compatible(glob_diff_min_audio_len_s))

  def test_audio_sources_merge_update(self):
    audio_sources_1 = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='pos',
                base_path=self.tempdir,
                file_glob='pos/*.wav',
            ),
            source_info.AudioSourceConfig(
                dataset_name='neg',
                base_path=self.tempdir,
                file_glob='neg/*.wav',
            ),
        )
    )
    with self.subTest('no_overlap'):
      disjoint = source_info.AudioSources(
          audio_globs=(
              source_info.AudioSourceConfig(
                  dataset_name='qua',
                  base_path=self.tempdir,
                  file_glob='qua/*.wav',
              ),
              source_info.AudioSourceConfig(
                  dataset_name='huh',
                  base_path=self.tempdir,
                  file_glob='huh/*.wav',
              ),
          )
      )
      got = audio_sources_1.merge_update(disjoint)
      self.assertLen(got.audio_globs, 4)

    with self.subTest('update_base_path'):
      overlap = source_info.AudioSources(
          audio_globs=(
              source_info.AudioSourceConfig(
                  dataset_name='pos',
                  base_path='/other/basedir',
                  file_glob='pos/*.wav',
              ),
          )
      )
      got = audio_sources_1.merge_update(overlap)
      self.assertLen(got.audio_globs, 2)
      self.assertEqual(got.audio_globs[0].base_path, '/other/basedir')

    with self.subTest('update_incompatible_audio_glob'):
      incompatible = source_info.AudioSources(
          audio_globs=(
              source_info.AudioSourceConfig(
                  dataset_name='pos',
                  base_path=self.tempdir,
                  file_glob='pos/*.wav',
                  target_sample_rate_hz=24000,
              ),
          )
      )
      with self.assertRaises(ValueError):
        audio_sources_1.merge_update(incompatible)

  def test_s3_audio_sources_iteration(self):
    base_path = 's3://bucket/audio'
    file_a = _FakeS3Path('s3://bucket/audio/deploy_a/file_a.wav')
    file_b = _FakeS3Path('s3://bucket/audio/deploy_b/file_b.wav')

    audio_sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='s3_dataset',
                base_path=base_path,
                file_glob='**/*.wav',
                shard_len_s=2.0,
            ),
        )
    )

    with mock.patch.object(
        source_info,
        '_iter_s3_filepaths',
        return_value=(file_a, file_b),
    ):
      with mock.patch.object(
          audio_sources,
          '_get_audio_len_s_and_sample_rate_hz',
          return_value=(5.0, 16000),
      ):
        got = tuple(audio_sources.iterate_all_sources())

    self.assertLen(got, 6)
    self.assertEqual(got[0].file_id, 'deploy_a/file_a.wav')
    self.assertEqual(got[1].file_id, 'deploy_a/file_a.wav')
    self.assertEqual(got[2].file_id, 'deploy_a/file_a.wav')
    self.assertEqual(got[0].offset_s, 0.0)
    self.assertEqual(got[1].offset_s, 2.0)
    self.assertEqual(got[2].offset_s, 4.0)
    self.assertEqual(got[3].file_id, 'deploy_b/file_b.wav')
    self.assertEqual(got[0].sample_rate_hz, 16000)

  def test_s3_iterate_files_skips_audio_metadata(self):
    base_path = 's3://bucket/audio'
    file_a = _FakeS3Path('s3://bucket/audio/deploy_a/file_a.wav')
    audio_sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='s3_dataset',
                base_path=base_path,
                file_glob='**/*.wav',
            ),
        )
    )

    with mock.patch.object(
        source_info, '_iter_s3_filepaths', return_value=(file_a,)
    ), mock.patch.object(audio_sources, '_get_audio_len_s_and_sample_rate_hz') as get_info:
      files = tuple(audio_sources.iterate_files())

    self.assertEqual(files[0][1], 'deploy_a/file_a.wav')
    get_info.assert_not_called()

  def test_s3_audio_sources_iteration_file_id_fallback(self):
    base_path = 's3://bucket/audio'
    file_a = _FakeS3PathNoRelative('s3://bucket/audio/deploy/file.wav')

    audio_sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='s3_dataset',
                base_path=base_path,
                file_glob='**/*.wav',
                shard_len_s=None,
            ),
        )
    )

    with mock.patch.object(
        source_info,
        '_iter_s3_filepaths',
        return_value=(file_a,),
    ):
      with mock.patch.object(
          audio_sources,
          '_get_audio_len_s_and_sample_rate_hz',
          return_value=(5.0, 16000),
      ):
        got = tuple(audio_sources.iterate_all_sources())

    self.assertLen(got, 1)
    self.assertEqual(got[0].file_id, 'deploy/file.wav')
    self.assertEqual(got[0].filepath, 's3://bucket/audio/deploy/file.wav')


if __name__ == '__main__':
  absltest.main()
