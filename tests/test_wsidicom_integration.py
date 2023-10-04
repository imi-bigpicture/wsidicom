#    Copyright 2021, 2023 SECTRA AB
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

from hashlib import md5
from pathlib import Path
from typing import Any, Callable, Dict

import pytest
from PIL import Image

from tests.conftest import WsiInputType, WsiTestDefinitions
from wsidicom import WsiDicom


@pytest.mark.integration
@pytest.mark.parametrize(
    "input_type",
    [
        WsiInputType.FILE,
        WsiInputType.STREAM,
        WsiInputType.WEB,
    ],
)
class TestWsiDicomIntegration:
    @pytest.mark.parametrize(["wsi_name", "region"], WsiTestDefinitions.read_region())
    def test_read_region(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
        region: Dict[str, Any],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        im = wsi.read_region(
            (region["location"]["x"], region["location"]["y"]),
            region["level"],
            (region["size"]["width"], region["size"]["height"]),
        )

        # Assert
        assert md5(im.tobytes()).hexdigest() == region["md5"], region

    @pytest.mark.parametrize(
        ["wsi_name", "region"], WsiTestDefinitions.read_region_mm()
    )
    def test_read_region_mm(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
        region: Dict[str, Any],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        im = wsi.read_region_mm(
            (region["location"]["x"], region["location"]["y"]),
            region["level"],
            (region["size"]["width"], region["size"]["height"]),
        )

        # Assert
        assert md5(im.tobytes()).hexdigest() == region["md5"], region

    @pytest.mark.parametrize(
        ["wsi_name", "region"], WsiTestDefinitions.read_region_mpp()
    )
    def test_read_region_mpp(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
        region: Dict[str, Any],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        im = wsi.read_region_mpp(
            (region["location"]["x"], region["location"]["y"]),
            region["mpp"],
            (region["size"]["width"], region["size"]["height"]),
        )

        # Assert
        assert md5(im.tobytes()).hexdigest() == region["md5"], region

    @pytest.mark.parametrize(["wsi_name", "region"], WsiTestDefinitions.read_tile())
    def test_read_tile(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
        region: Dict[str, Any],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        im = wsi.read_tile(
            region["level"],
            (region["location"]["x"], region["location"]["y"]),
        )

        # Assert
        assert md5(im.tobytes()).hexdigest() == region["md5"], region

    @pytest.mark.parametrize(
        ["wsi_name", "region"], WsiTestDefinitions.read_encoded_tile()
    )
    def test_read_encoded_tile(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
        region: Dict[str, Any],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        im = wsi.read_encoded_tile(
            region["level"],
            (region["location"]["x"], region["location"]["y"]),
        )

        # Assert
        assert md5(im).hexdigest() == region["md5"], region

    @pytest.mark.parametrize(
        ["wsi_name", "region"], WsiTestDefinitions.read_thumbnail()
    )
    def test_read_thumbnail(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
        region: Dict[str, Any],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        im = wsi.read_thumbnail((region["size"]["width"], region["size"]["height"]))

        # Assert
        assert md5(im.tobytes()).hexdigest() == region["md5"], region

    @pytest.mark.parametrize("wsi_path", WsiTestDefinitions.folders())
    def test_replace_label(self, wsi_path: Path, input_type: WsiInputType):
        # Arrange
        if not wsi_path.exists():
            pytest.skip(f"Folder {wsi_path} for wsi does not exist.")
        image = Image.new("RGB", (256, 256), (128, 128, 128))

        # Act
        with WsiDicom.open(wsi_path, label=image) as wsi:
            label = wsi.read_label()

        # Assert
        assert image == label

    @pytest.mark.parametrize(
        ["wsi_name", "expected_level_count"], WsiTestDefinitions.levels()
    )
    def test_number_of_levels(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
        expected_level_count: int,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        levels_count = len(wsi.levels)

        # Assert
        assert levels_count == expected_level_count

    @pytest.mark.parametrize(
        ["wsi_name", "expected_has_label"], WsiTestDefinitions.has_label()
    )
    def test_has_label(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
        expected_has_label: bool,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        has_label = wsi.labels is not None

        # Assert
        assert has_label == expected_has_label

    @pytest.mark.parametrize(
        ["wsi_name", "expected_has_overview"], WsiTestDefinitions.has_overview()
    )
    def test_has_overview(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
        expected_has_overview: bool,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        has_overview = wsi.overviews is not None

        # Assert
        assert has_overview == expected_has_overview
