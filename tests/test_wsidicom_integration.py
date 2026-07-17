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

from collections.abc import Callable
from hashlib import md5
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image as Pillow
from pydicom.uid import UID

from tests.conftest import (
    WsiInputType,
    WsiTestDefinitions,
    skip_if_hash_unstable,
    skip_if_shared_pool_unsupported,
)
from wsidicom import WsiDicom
from wsidicom.config import settings
from wsidicom.downsampler import Cv2Downsampler
from wsidicom.geometry import Point
from wsidicom.series.pyramid import Pyramid
from wsidicom.thread import ReadExecutor


@pytest.fixture(
    params=[
        "pillow",
        pytest.param(
            "opencv",
            marks=pytest.mark.skipif(
                not Cv2Downsampler.is_available(), reason="opencv not installed"
            ),
        ),
    ]
)
def downsampler_backend(request: pytest.FixtureRequest) -> Any:
    """Run a read-golden test under each downsampler backend, pinning it for the
    test and restoring the setting afterwards."""
    original = settings.preferred_downsampler
    settings.preferred_downsampler = request.param
    yield request.param
    settings.preferred_downsampler = original


def expected_md5(region: dict[str, Any], backend: str) -> str:
    """The golden md5 for a region: a plain string when backend-independent, or
    the per-downsampler value when the read downsamples and backends differ."""
    md5 = region["md5"]
    return md5 if isinstance(md5, str) else md5[backend]


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
    @pytest.mark.parametrize(
        ["threads", "use_shared"],
        [
            pytest.param(None, False, id="inline"),
            pytest.param(4, False, id="per_call_pool"),
            pytest.param(None, True, id="shared_pool"),
        ],
    )
    @pytest.mark.parametrize(
        ["wsi_name", "transfer_syntax", "region"], WsiTestDefinitions.read_region()
    )
    def test_read_region(
        self,
        downsampler_backend: str,
        wsi_name: str,
        transfer_syntax: UID,
        input_type: WsiInputType,
        wsi_factory: Callable[..., WsiDicom],
        region: dict[str, Any],
        threads: int | None,
        use_shared: bool,
    ):
        # Arrange
        skip_if_hash_unstable(transfer_syntax)
        skip_if_shared_pool_unsupported(input_type, use_shared)
        wsi = wsi_factory(wsi_name, input_type, use_shared)

        # Act
        im = wsi.read_region(
            (region["location"]["x"], region["location"]["y"]),
            region["level"],
            (region["size"]["width"], region["size"]["height"]),
            threads=threads,
        )

        # Assert
        checksum = md5(im.tobytes()).hexdigest()
        assert checksum == expected_md5(region, downsampler_backend), (region, checksum)

    @pytest.mark.parametrize("threads", [None, 4])
    @pytest.mark.parametrize(
        ["wsi_name", "transfer_syntax", "region"], WsiTestDefinitions.read_region_mm()
    )
    def test_read_region_mm(
        self,
        downsampler_backend: str,
        wsi_name: str,
        transfer_syntax: UID,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
        region: dict[str, Any],
        threads: int | None,
    ):
        # Arrange
        skip_if_hash_unstable(transfer_syntax)
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        im = wsi.read_region_mm(
            (region["location"]["x"], region["location"]["y"]),
            region["level"],
            (region["size"]["width"], region["size"]["height"]),
            slide_origin=region.get("slide_origin", False),
            threads=threads,
        )

        # Assert
        checksum = md5(im.tobytes()).hexdigest()
        assert checksum == expected_md5(region, downsampler_backend), (region, checksum)

    @pytest.mark.parametrize("threads", [None, 4])
    @pytest.mark.parametrize(
        ["wsi_name", "transfer_syntax", "region"], WsiTestDefinitions.read_region_mpp()
    )
    def test_read_region_mpp(
        self,
        downsampler_backend: str,
        wsi_name: str,
        transfer_syntax: UID,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
        region: dict[str, Any],
        threads: int | None,
    ):
        # Arrange
        skip_if_hash_unstable(transfer_syntax)
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        im = wsi.read_region_mpp(
            (region["location"]["x"], region["location"]["y"]),
            region["mpp"],
            (region["size"]["width"], region["size"]["height"]),
            slide_origin=region.get("slide_origin", False),
            threads=threads,
        )

        # Assert
        checksum = md5(im.tobytes()).hexdigest()
        assert checksum == expected_md5(region, downsampler_backend), (region, checksum)

    @pytest.mark.parametrize(
        ["wsi_name", "transfer_syntax", "region"], WsiTestDefinitions.read_region()
    )
    def test_read_region_as_array_matches_read_region(
        self,
        wsi_name: str,
        transfer_syntax: UID,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
        region: dict[str, Any],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)
        location = (region["location"]["x"], region["location"]["y"])
        size = (region["size"]["width"], region["size"]["height"])

        # Act — the numpy read and the PIL read of the same region
        array = wsi.read_region(location, region["level"], size, as_array=True)
        image = wsi.read_region(location, region["level"], size)

        # Assert — pixel-for-pixel identical
        assert np.array_equal(array, np.asarray(image))

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_get_scaled_tile_pads_to_tile_size_when_not_cropping(
        self,
        wsi_name: str,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
        read_executor: ReadExecutor,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)
        base_level = wsi.pyramids[0].base_level
        target_level = base_level.level + 1
        tile_size = base_level.tile_size
        # Bottom-right tile of the (generated) target level is an edge tile.
        target_tiled_size = (base_level.size // 2).ceil_div(tile_size)
        edge_tile = Point(target_tiled_size.width - 1, target_tiled_size.height - 1)
        # The background tile is cached and shared; padding must not mutate it.
        blank_before = base_level.default_instance.image_data.blank_tile.tobytes()

        # Act
        cropped = base_level.get_scaled_tile(
            edge_tile, target_level, crop_to_image_boundary=True, executor=read_executor
        )
        padded = base_level.get_scaled_tile(
            edge_tile,
            target_level,
            crop_to_image_boundary=False,
            executor=read_executor,
        )

        # Assert (arrays are (rows, columns, ...) = (height, width, ...))
        assert padded.shape[:2] == (tile_size.height, tile_size.width)
        assert cropped.shape[1] <= tile_size.width
        assert cropped.shape[0] <= tile_size.height
        # The padded tile holds the cropped tile at the origin, padded elsewhere.
        assert (
            padded[: cropped.shape[0], : cropped.shape[1]].tobytes()
            == cropped.tobytes()
        )
        # Padding copied the background tile rather than mutating the shared one.
        assert (
            base_level.default_instance.image_data.blank_tile.tobytes() == blank_before
        )

    @pytest.mark.parametrize(
        ["wsi_name", "transfer_syntax", "region"], WsiTestDefinitions.read_tile()
    )
    def test_read_tile(
        self,
        downsampler_backend: str,
        wsi_name: str,
        transfer_syntax: UID,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
        region: dict[str, Any],
    ):
        # Arrange
        skip_if_hash_unstable(transfer_syntax)
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        im = wsi.read_tile(
            region["level"],
            (region["location"]["x"], region["location"]["y"]),
            crop_to_image_boundary=region.get("crop_to_image_boundary", True),
        )

        # Assert
        checksum = md5(im.tobytes()).hexdigest()
        assert checksum == expected_md5(region, downsampler_backend), (region, checksum)

    @pytest.mark.parametrize(
        ["wsi_name", "transfer_syntax", "region"],
        WsiTestDefinitions.read_encoded_tile(),
    )
    def test_read_encoded_tile(
        self,
        downsampler_backend: str,
        wsi_name: str,
        transfer_syntax: UID,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
        region: dict[str, Any],
    ):
        # Arrange
        skip_if_hash_unstable(transfer_syntax)
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        im = wsi.read_encoded_tile(
            region["level"],
            (region["location"]["x"], region["location"]["y"]),
            crop_to_image_boundary=region.get("crop_to_image_boundary", True),
        )

        # Assert
        checksum = md5(im).hexdigest()
        assert checksum == expected_md5(region, downsampler_backend), (region, checksum)

    @pytest.mark.parametrize(
        ["wsi_name", "transfer_syntax", "region"], WsiTestDefinitions.read_thumbnail()
    )
    def test_read_thumbnail(
        self,
        downsampler_backend: str,
        wsi_name: str,
        transfer_syntax: UID,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
        region: dict[str, Any],
    ):
        # Arrange
        skip_if_hash_unstable(transfer_syntax)
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        im = wsi.read_thumbnail((region["size"]["width"], region["size"]["height"]))

        # Assert
        checksum = md5(im.tobytes()).hexdigest()
        assert checksum == expected_md5(region, downsampler_backend), (region, checksum)
        assert im.size[0] <= region["size"]["width"]
        assert im.size[1] <= region["size"]["height"]

    @pytest.mark.parametrize(
        ["wsi_name", "expected_level_count"], WsiTestDefinitions.levels()
    )
    def test_number_of_levels(
        self,
        wsi_name: str,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
        expected_level_count: int,
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        levels_count = len(wsi.pyramids[0])

        # Assert
        assert levels_count == expected_level_count

    @pytest.mark.parametrize(
        ["wsi_name", "transfer_syntax", "expected_label_hash"],
        WsiTestDefinitions.label_hash(),
    )
    def test_read_label(
        self,
        wsi_name: str,
        transfer_syntax: UID,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
        expected_label_hash: bool | None,
    ):
        # Arrange
        skip_if_hash_unstable(transfer_syntax)
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        if wsi.labels is not None:
            checksum = md5(wsi.read_label().tobytes()).hexdigest()
        else:
            checksum = None

        # Assert
        assert checksum == expected_label_hash

    @pytest.mark.parametrize(
        ["wsi_name", "transfer_syntax", "expected_overview_hash"],
        WsiTestDefinitions.overview_hash(),
    )
    def test_read_overview(
        self,
        wsi_name: str,
        transfer_syntax: UID,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
        expected_overview_hash: str | None,
    ):
        # Arrange
        skip_if_hash_unstable(transfer_syntax)
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
        wsi_name: str,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
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
        wsi_name: str,
        input_type: WsiInputType,
        wsi_factory: Callable[[str, WsiInputType], WsiDicom],
    ):
        # Arrange
        wsi = wsi_factory(wsi_name, input_type)

        # Act
        pyramid = wsi.pyramid

        # Assert
        assert isinstance(pyramid, Pyramid)
