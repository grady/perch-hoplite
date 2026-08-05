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

"""Tests for audio IO."""

import functools
import http.server
import os
import shutil
import tempfile
import threading

from perch_hoplite import audio_io
from perch_hoplite.agile.tests import test_utils

from absl.testing import absltest


class _CountingRequestHandler(http.server.SimpleHTTPRequestHandler):
  request_count = 0

  def log_message(self, format, *args):  # pylint: disable=redefined-builtin
    del format, args

  def do_GET(self):
    type(self).request_count += 1
    super().do_GET()


class AudioIoTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tempdir = tempfile.mkdtemp()
    audio_io.close_url_audio_cache()
    _CountingRequestHandler.request_count = 0

  def tearDown(self):
    audio_io.close_url_audio_cache()
    shutil.rmtree(self.tempdir)
    super().tearDown()

  def _start_server(self):
    handler = functools.partial(
        _CountingRequestHandler, directory=self.tempdir
    )
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def cleanup():
      server.shutdown()
      thread.join()
      server.server_close()

    self.addCleanup(cleanup)
    return server

  def test_url_audio_is_cached_across_reads(self):
    test_utils.make_wav_files(
        self.tempdir, classes=['pos'], filenames=['foo', 'bar'], file_len_s=2.0
    )
    server = self._start_server()
    base_url = f'http://127.0.0.1:{server.server_address[1]}'
    url = f'{base_url}/pos/foo_pos.wav'

    audio = audio_io.load_audio_window(
        url, offset_s=0.0, sample_rate=16000, window_size_s=1.0
    )
    self.assertEqual(audio.shape[0], 16000)

    audio_2 = audio_io.load_audio_window(
        url, offset_s=0.5, sample_rate=16000, window_size_s=1.0
    )
    self.assertEqual(audio_2.shape[0], 16000)

    file_length_s, sample_rate = audio_io.get_file_length_s_and_sample_rate(url)
    self.assertEqual(sample_rate, 16000)
    self.assertAlmostEqual(file_length_s, 2.0, places=3)
    self.assertEqual(_CountingRequestHandler.request_count, 1)

  def test_url_audio_cache_evicts_lru_entry(self):
    test_utils.make_wav_files(
        self.tempdir, classes=['pos'], filenames=['foo', 'bar'], file_len_s=2.0
    )
    audio_io.configure_url_audio_cache(max_entries=1, max_bytes=1024**3)
    server = self._start_server()
    base_url = f'http://127.0.0.1:{server.server_address[1]}'
    url_foo = f'{base_url}/pos/foo_pos.wav'
    url_bar = f'{base_url}/pos/bar_pos.wav'

    audio_io.load_audio_window(
        url_foo, offset_s=0.0, sample_rate=16000, window_size_s=1.0
    )
    cached_path = audio_io.get_url_audio_cache().get_local_path(
        url_foo, sample_rate=16000, resampling_type='polyphase'
    )
    self.assertTrue(os.path.exists(cached_path))

    audio_io.load_audio_window(
        url_bar, offset_s=0.0, sample_rate=16000, window_size_s=1.0
    )
    self.assertFalse(os.path.exists(cached_path))

    audio_io.load_audio_window(
        url_foo, offset_s=0.0, sample_rate=16000, window_size_s=1.0
    )
    self.assertEqual(_CountingRequestHandler.request_count, 3)


if __name__ == '__main__':
  absltest.main()
