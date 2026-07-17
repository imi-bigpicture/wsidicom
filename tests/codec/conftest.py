#    Copyright 2023 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import pytest
from wsidicom_data import EncodedTestData, TestData

from wsidicom.codec import Settings


def max_band_rms(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum per-channel RMS difference between two images, normalized to an
    8-bit (0-255) scale so the same tolerance applies at any bit depth."""
    scale = 255.0 / np.iinfo(a.dtype).max if np.issubdtype(a.dtype, np.integer) else 1.0
    diff = (a.astype(np.float64) - b.astype(np.float64)) * scale
    if diff.ndim == 3:
        return float(np.sqrt(np.mean(diff**2, axis=(0, 1))).max())
    return float(np.sqrt(np.mean(diff**2)))


@pytest.fixture
def image(settings: Settings) -> np.ndarray:
    return np.asarray(TestData.image(settings.bits, settings.samples_per_pixel))


@pytest.fixture
def encoded(settings: Settings):
    file_path = EncodedTestData.get_filepath_for_encoder_settings(settings)
    with open(file_path, "rb") as file:
        return file.read()
