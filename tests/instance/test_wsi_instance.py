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

import numpy as np
import pytest
from PIL import Image as Pillow

from tests.data_gen import create_main_dataset
from wsidicom.geometry import Point, Region, Size
from wsidicom.instance import WsiInstance
from wsidicom.instance.numpy_image_data import NumpyImageData
from wsidicom.metadata import ImageType
from wsidicom.thread import ReadExecutor


class RegionReadingImageData(NumpyImageData):
    """Image data that reads regions itself, recording the regions it reads."""

    def __init__(self, array: np.ndarray) -> None:
        super().__init__(array)
        self.regions: list[Region] = []

    def get_decoded_region(
        self, region: Region, z: float, path: str, *, executor: ReadExecutor
    ) -> np.ndarray:
        self.regions.append(region)
        return np.full((region.size.height, region.size.width, 3), 7, np.uint8)


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

    def test_create_instance_leaves_base_dataset_unchanged(self) -> None:
        # Arrange - one base dataset given for several instances
        base_dataset = create_main_dataset()
        unchanged = deepcopy(base_dataset)
        image_data = NumpyImageData.from_image(
            Pillow.new("RGB", (256, 128), (128, 128, 128))
        )

        # Act
        overview = WsiInstance.create_instance(
            image_data, base_dataset, ImageType.OVERVIEW
        )
        label = WsiInstance.create_instance(image_data, base_dataset, ImageType.LABEL)

        # Assert
        assert base_dataset == unchanged
        assert overview.dataset.as_dataset().ImageType[2] == "OVERVIEW"
        assert label.dataset.as_dataset().ImageType[2] == "LABEL"

    def test_region_is_read_through_image_data(self) -> None:
        # Arrange - image data that reads regions itself rather than by tiles
        image_data = RegionReadingImageData(np.zeros((128, 256, 3), np.uint8))
        instance = WsiInstance.create_instance(
            image_data, create_main_dataset(), ImageType.OVERVIEW
        )
        region = Region(Point(10, 20), Size(30, 40))

        # Act
        pixels = instance.get_region(
            region,
            image_data.default_z,
            image_data.default_path,
            executor=ReadExecutor(1, None),
        )

        # Assert
        assert image_data.regions == [region]
        assert (pixels == 7).all()

    def test_region_is_assembled_from_tiles_by_default(self) -> None:
        # Arrange
        array = np.arange(128 * 256 * 3, dtype=np.uint32).astype(np.uint8)
        image_data = NumpyImageData(array.reshape(128, 256, 3))
        region = Region(Point(10, 20), Size(30, 40))

        # Act
        pixels = image_data.get_decoded_region(
            region,
            image_data.default_z,
            image_data.default_path,
            executor=ReadExecutor(1, None),
        )

        # Assert
        assert (pixels == array.reshape(128, 256, 3)[20:60, 10:40]).all()
