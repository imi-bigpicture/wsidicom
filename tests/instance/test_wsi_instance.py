#    Copyright 2026 SECTRA AB
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

from copy import deepcopy

import pytest
from PIL import Image as Pillow

from tests.data_gen import create_main_dataset
from wsidicom.instance import WsiInstance


@pytest.mark.unittest
class TestWsiInstance:
    def test_create_label_leaves_base_dataset_unchanged(self) -> None:
        # Arrange - the base dataset is commonly the dataset of the base level
        base_dataset = create_main_dataset()
        unchanged = deepcopy(base_dataset)
        label = Pillow.new("RGB", (256, 128), (128, 128, 128))

        # Act
        label_instance = WsiInstance.create_label(label, base_dataset)

        # Assert
        assert base_dataset == unchanged
        assert label_instance.dataset.as_dataset().ImageType[2] == "LABEL"
