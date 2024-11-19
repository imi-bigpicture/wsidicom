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
from typing import Any, Callable, Dict, Optional

import numpy as np
import pytest
from PIL import Image as Pillow

from tests.conftest import WsiInputType, WsiTestDefinitions
from wsidicom import WsiDicom
from wsidicom.series.pyramid import Pyramid


@pytest.mark.integration
@pytest.mark.parametrize(
    "input_type",
    [
        WsiInputType.FILE,
        # WsiInputType.STREAM,
        # WsiInputType.WEB,
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
        checksum = md5(im.tobytes()).hexdigest()
        assert checksum == region["md5"], (region, checksum)

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
        checksum = md5(im.tobytes()).hexdigest()
        assert checksum == region["md5"], (region, checksum)

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
        checksum = md5(im.tobytes()).hexdigest()
        assert checksum == region["md5"], (region, checksum)

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
        checksum = md5(im.tobytes()).hexdigest()
        assert checksum == region["md5"], (region, checksum)

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
        checksum = md5(im).hexdigest()
        assert checksum == region["md5"], (region, checksum)

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
        checksum = md5(im.tobytes()).hexdigest()
        assert checksum == region["md5"], (region, checksum)

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
        levels_count = len(wsi.pyramids[0])

        # Assert
        assert levels_count == expected_level_count

    @pytest.mark.parametrize(
        ["wsi_name", "expected_label_hash"], WsiTestDefinitions.label_hash()
    )
    def test_has_label(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
        expected_label_hash: Optional[bool],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        if wsi.labels is not None:
            checksum = md5(wsi.read_label().tobytes()).hexdigest()
        else:
            checksum = None

        # Assert
        assert checksum == expected_label_hash

    @pytest.mark.parametrize(
        ["wsi_name", "expected_overview_hash"], WsiTestDefinitions.overview_hash()
    )
    def test_has_overview(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
        expected_overview_hash: Optional[str],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        if wsi.overviews is not None:
            checksum = md5(wsi.read_overview().tobytes()).hexdigest()
        else:
            checksum = None

        # Assert
        assert checksum == expected_overview_hash

    @pytest.mark.parametrize("wsi_path", WsiTestDefinitions.folders())
    def test_save_replace_label(
        self, wsi_path: Path, input_type: WsiInputType, tmp_path: Path
    ):
        # Arrange
        if not wsi_path.exists():
            pytest.skip(f"Folder {wsi_path} for wsi does not exist.")
        label = Pillow.new("RGB", (256, 256), (128, 128, 128))

        # Act
        with WsiDicom.open(wsi_path) as wsi:
            wsi.save(tmp_path, include_levels=[-1], label=label)

        # Assert
        with WsiDicom.open(tmp_path) as wsi:
            read_label = wsi.read_label()
        assert np.array_equal(np.array(read_label), np.array(label))

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_levels_returns_selected_pyramid(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        levels = wsi.levels

        # Assert
        assert isinstance(levels, Pyramid)
        assert levels == wsi.pyramids[wsi.selected_pyramid]

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_pyramid_returns_selected_pyramid(
        self,
        wsi_name: Path,
        input_type: WsiInputType,
        wsi_factory: Callable[[Path, WsiInputType], WsiDicom],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        pyramid = wsi.pyramid

        # Assert
        assert isinstance(pyramid, Pyramid)
        assert pyramid == wsi.pyramids[wsi.selected_pyramid]
