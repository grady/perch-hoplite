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
import types
from unittest import mock

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

  def test_local_audio_can_be_cached_optionally(self):
    test_utils.make_wav_files(
        self.tempdir, classes=['pos'], filenames=['foo'], file_len_s=2.0
    )
    filepath = os.path.join(self.tempdir, 'pos', 'foo_pos.wav')

    audio = audio_io.load_audio_window(
        filepath,
        offset_s=0.0,
        sample_rate=16000,
        window_size_s=1.0,
        cache_local_audio=True,
    )
    self.assertEqual(audio.shape[0], 16000)
    file_length_s, sample_rate = audio_io.get_file_length_s_and_sample_rate(
        filepath, cache_local_audio=True
    )
    self.assertEqual(sample_rate, 16000)
    self.assertAlmostEqual(file_length_s, 2.0, places=3)

    os.unlink(filepath)

    audio_2 = audio_io.load_audio_window(
        filepath,
        offset_s=0.5,
        sample_rate=16000,
        window_size_s=1.0,
        cache_local_audio=True,
    )
    self.assertEqual(audio_2.shape[0], 16000)

  def test_s3_audio_is_cached_across_reads(self):
    test_utils.make_wav_files(
        self.tempdir, classes=['pos'], filenames=['foo'], file_len_s=2.0
    )
    filepath = os.path.join(self.tempdir, 'pos', 'foo_pos.wav')
    with open(filepath, 'rb') as f:
      wav_bytes = f.read()

    url = 's3://audio-bucket/pos/foo_pos.wav'
    read_count = {'count': 0}

    def _mock_read_s3(path):
      self.assertEqual(path, url)
      read_count['count'] += 1
      return wav_bytes, 'audio/wav'

    with mock.patch('perch_hoplite.audio_io._read_s3_object_bytes', _mock_read_s3):
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

    self.assertEqual(read_count['count'], 1)

  def test_s3_audio_cache_evicts_lru_entry(self):
    test_utils.make_wav_files(
        self.tempdir, classes=['pos'], filenames=['foo', 'bar'], file_len_s=2.0
    )
    foo_path = os.path.join(self.tempdir, 'pos', 'foo_pos.wav')
    bar_path = os.path.join(self.tempdir, 'pos', 'bar_pos.wav')
    with open(foo_path, 'rb') as f:
      foo_bytes = f.read()
    with open(bar_path, 'rb') as f:
      bar_bytes = f.read()

    url_foo = 's3://audio-bucket/pos/foo_pos.wav'
    url_bar = 's3://audio-bucket/pos/bar_pos.wav'
    objects = {
      url_foo: foo_bytes,
      url_bar: bar_bytes,
    }
    read_count = {'count': 0}

    def _mock_read_s3(path):
      read_count['count'] += 1
      return objects[path], 'audio/wav'

    audio_io.configure_url_audio_cache(max_entries=1, max_bytes=1024**3)
    with mock.patch('perch_hoplite.audio_io._read_s3_object_bytes', _mock_read_s3):
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

    self.assertEqual(read_count['count'], 3)

  def test_read_s3_object_bytes_rejects_invalid_uri(self):
    with self.assertRaisesRegex(ValueError, 'Invalid S3 URI'):
      audio_io._read_s3_object_bytes('s3://audio-bucket')

  def test_s3_storage_options_from_env(self):
    env_patch = {
        'HOPLITE_S3_ENDPOINT': 'http://minio.internal:9000',
        'HOPLITE_S3_ACCESS_KEY': 'ak',
        'HOPLITE_S3_SECRET_KEY': 'sk',
        'HOPLITE_S3_SESSION_TOKEN': 'tk',
        'HOPLITE_S3_REGION': 'us-test-1',
        'HOPLITE_S3_USE_SSL': 'false',
        'HOPLITE_S3_VERIFY': '/tmp/ca.pem',
    }
    with mock.patch.dict(os.environ, env_patch, clear=False):
      got = audio_io._s3_storage_options_from_env()
    self.assertEqual(got['endpoint_url'], 'http://minio.internal:9000')
    self.assertEqual(got['aws_access_key_id'], 'ak')
    self.assertEqual(got['aws_secret_access_key'], 'sk')
    self.assertEqual(got['aws_session_token'], 'tk')
    self.assertEqual(got['region_name'], 'us-test-1')
    self.assertFalse(got['use_ssl'])
    self.assertEqual(got['verify'], '/tmp/ca.pem')

  def test_s3_storage_options_endpoint_scheme_follows_use_ssl(self):
    with self.subTest('https endpoint with use_ssl false coerces to http'):
      env_patch = {
          'HOPLITE_S3_ENDPOINT': 'https://minio.internal:9000',
          'HOPLITE_S3_USE_SSL': 'false',
      }
      with mock.patch.dict(os.environ, env_patch, clear=False):
        got = audio_io._s3_storage_options_from_env()
      self.assertEqual(got['endpoint_url'], 'http://minio.internal:9000')
      self.assertFalse(got['use_ssl'])

    with self.subTest('bare endpoint with use_ssl true gets https scheme'):
      env_patch = {
          'HOPLITE_S3_ENDPOINT': 'minio.internal:9000',
          'HOPLITE_S3_USE_SSL': 'true',
      }
      with mock.patch.dict(os.environ, env_patch, clear=False):
        got = audio_io._s3_storage_options_from_env()
      self.assertEqual(got['endpoint_url'], 'https://minio.internal:9000')
      self.assertTrue(got['use_ssl'])

  def test_read_s3_object_bytes_uses_boto3_client(self):
    class _Body:
      def __init__(self, b):
        self._b = b
        self.closed = False

      def read(self):
        return self._b

      def close(self):
        self.closed = True

    body = _Body(b'abc123')
    client_mock = mock.MagicMock()
    client_mock.get_object.return_value = {
        'Body': body,
        'ContentType': 'audio/wav',
    }
    boto3_stub = types.SimpleNamespace(
        client=mock.MagicMock(return_value=client_mock)
    )

    with mock.patch.dict('sys.modules', {'boto3': boto3_stub}, clear=False):
      with mock.patch(
          'perch_hoplite.audio_io._s3_storage_options_from_env',
          return_value={'endpoint_url': 'http://localhost:9000'},
      ):
        payload, content_type = audio_io._read_s3_object_bytes(
            's3://bucket-name/path/to/audio.wav'
        )

    self.assertEqual(payload, b'abc123')
    self.assertEqual(content_type, 'audio/wav')
    boto3_stub.client.assert_called_once_with(
        's3', endpoint_url='http://localhost:9000'
    )
    client_mock.get_object.assert_called_once_with(
        Bucket='bucket-name', Key='path/to/audio.wav'
    )
    self.assertTrue(body.closed)


if __name__ == '__main__':
  absltest.main()
