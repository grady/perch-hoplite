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

"""Audio loader function helpers."""

from typing import Callable

from etils import epath
import numpy as np
from perch_hoplite import audio_io
from perch_hoplite.agile import source_info


def make_filepath_loader(
    audio_sources: source_info.AudioSources,
    sample_rate_hz: int = 32000,
    window_size_s: float = 5.0,
    dtype: str = 'float32',
    cache_local_audio: bool = False,
) -> Callable[[str, float], np.ndarray]:
  """Create a function for loading audio from a source ID and offset.

  Note that if multiple globs match a given source ID, the first match is used.

  Args:
    audio_sources: Embedding audio sources.
    sample_rate_hz: Sample rate of the audio.
    window_size_s: Window size of the audio.
    dtype: Data type of the audio.

  Returns:
    Function for loading audio from a source ID and offset.

  Raises:
    ValueError if no audio path is found for the given source ID.
  """

  def loader(source_id: str, offset_s: float) -> np.ndarray:
    found_path = None
    found_audio_source = None
    for audio_source in audio_sources.audio_globs:
      path = epath.Path(audio_source.base_path) / source_id
      if path.exists():
        found_path = path
        found_audio_source = audio_source
        break
    if found_path is None:
      raise ValueError('No audio path found for source_id: ', source_id)
    cache_local_audio = cache_local_audio or (
        found_audio_source.cache_local_audio if found_audio_source else False
    )
    return np.array(
        audio_io.load_audio_window(
            found_path.as_posix(),
            offset_s,
            sample_rate=sample_rate_hz,
            window_size_s=window_size_s,
            cache_local_audio=cache_local_audio,
        ),
        dtype=dtype,
    )

  return loader
