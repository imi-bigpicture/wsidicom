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

import pytest
from wsidicom_data import EncodedTestData, TestData

from wsidicom.codec import Settings


@pytest.fixture
def image(settings: Settings):
    return TestData.image(settings.bits, settings.samples_per_pixel)


@pytest.fixture
def encoded(settings: Settings):
    file_path = EncodedTestData.get_filepath_for_encoder_settings(settings)
    with open(file_path, "rb") as file:
        return file.read()
