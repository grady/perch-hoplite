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

"""Tests for audio_loader."""

import os
import shutil
import tempfile
from unittest import mock

import numpy as np
from perch_hoplite.agile import audio_loader
from perch_hoplite.agile import source_info
from perch_hoplite.agile.tests import test_utils

from absl.testing import absltest


class _FakeRemotePath:

  def __init__(self, path: str):
    self._path = path

  def __truediv__(self, suffix: str):
    return _FakeRemotePath(self._path.rstrip('/') + '/' + suffix)

  def exists(self):
    raise AssertionError('exists() should not be called for remote paths')

  def as_posix(self) -> str:
    return self._path


class AudioLoaderTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.tempdir)
    super().tearDown()

  def test_make_filepath_loader_local(self):
    test_utils.make_wav_files(
        self.tempdir, classes=['pos'], filenames=['foo'], file_len_s=1.0
    )
    rel_source_id = 'pos/foo_pos.wav'
    sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='local',
                base_path=self.tempdir,
                file_glob='**/*.wav',
            ),
        )
    )
    loader = audio_loader.make_filepath_loader(
        audio_sources=sources, sample_rate_hz=16000, window_size_s=0.25
    )

    got = loader(rel_source_id, 0.0)
    self.assertEqual(got.dtype, np.float32)
    self.assertEqual(got.shape[0], 4000)

  def test_make_filepath_loader_remote_s3_bypasses_exists(self):
    sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='remote',
                base_path='s3://bucket/audio',
                file_glob='**/*.wav',
            ),
        )
    )

    loader = audio_loader.make_filepath_loader(
        audio_sources=sources,
        sample_rate_hz=16000,
        window_size_s=0.5,
    )

    fake_audio = np.ones((8000,), dtype=np.float32)
    expected_path = 's3://bucket/audio/deploy/file.wav'

    with mock.patch.object(
        audio_loader.epath,
        'Path',
        side_effect=lambda p: _FakeRemotePath(p),
    ):
      with mock.patch.object(
          audio_loader.audio_io,
          'load_audio_window',
          return_value=fake_audio,
      ) as load_audio_window_mock:
        got = loader('deploy/file.wav', 1.25)

    self.assertEqual(got.dtype, np.float32)
    self.assertEqual(got.shape[0], 8000)
    load_audio_window_mock.assert_called_once_with(
        expected_path,
        1.25,
        sample_rate=16000,
        window_size_s=0.5,
        cache_local_audio=False,
    )

  def test_make_filepath_loader_missing_path_raises(self):
    sources = source_info.AudioSources(
        audio_globs=(
            source_info.AudioSourceConfig(
                dataset_name='local',
                base_path=self.tempdir,
                file_glob='**/*.wav',
            ),
        )
    )
    loader = audio_loader.make_filepath_loader(audio_sources=sources)

    with self.assertRaisesRegex(ValueError, 'No audio path found for source_id'):
      loader('missing/file.wav', 0.0)


if __name__ == '__main__':
  absltest.main()
