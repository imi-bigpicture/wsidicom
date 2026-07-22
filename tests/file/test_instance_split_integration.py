#    Copyright 2024 SECTRA AB
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

"""Integration tests for splitting instances on focal planes when writing.

No test slide has multiple focal planes, so a real single-plane TILED_FULL
level is wrapped to present two focal planes (the same tile is served for both
planes). The instances are written through the real `PyramidFileWriter` and the
output DICOM files are inspected to confirm the split.
"""

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pydicom import Dataset, dcmread
from pydicom.sequence import Sequence
from pydicom.uid import UID, generate_uid
from upath import UPath

from tests.conftest import WsiTestDefinitions
from wsidicom import WsiDicom
from wsidicom.codec import Encoder
from wsidicom.file import OffsetTableType
from wsidicom.file.file_writer import PyramidFileWriter
from wsidicom.geometry import Point, Size, SizeMm
from wsidicom.group import Level
from wsidicom.instance import ImageData, WsiDataset, WsiInstance
from wsidicom.metadata import ImageCoordinateSystem, LossyCompression
from wsidicom.metadata.uid_generator import UidGenerator
from wsidicom.options import InstanceSplit
from wsidicom.series import Pyramid

FOCAL_PLANES = [0.0, 1.0]
UNEQUAL_FOCAL_PLANES = [0.0, 0.5, 2.0]


class MultiFocalPlaneImageData(ImageData):
    """Wraps an ImageData and presents several focal planes, serving the source
    tile for every plane."""

    def __init__(
        self,
        source: ImageData,
        focal_planes: list[float],
        optical_paths: list[str] | None = None,
    ):
        super().__init__(source.encoder)
        self._source = source
        self._focal_planes = focal_planes
        self._optical_paths = (
            optical_paths if optical_paths is not None else source.optical_paths
        )

    @property
    def focal_planes(self) -> list[float]:
        return self._focal_planes

    @property
    def optical_paths(self) -> list[str]:
        return self._optical_paths

    @property
    def transfer_syntax(self) -> UID:
        return self._source.transfer_syntax

    @property
    def image_size(self) -> Size:
        return self._source.image_size

    @property
    def tile_size(self) -> Size:
        return self._source.tile_size

    @property
    def pixel_spacing(self) -> SizeMm | None:
        return self._source.pixel_spacing

    @property
    def imaged_size(self) -> SizeMm | None:
        return self._source.imaged_size

    @property
    def samples_per_pixel(self) -> int:
        return self._source.samples_per_pixel

    @property
    def photometric_interpretation(self) -> str:
        return self._source.photometric_interpretation

    @property
    def bits(self) -> int:
        return self._source.bits

    @property
    def image_coordinate_system(self) -> ImageCoordinateSystem | None:
        return self._source.image_coordinate_system

    @property
    def thread_safe(self) -> bool:
        return self._source.thread_safe

    @property
    def lossy_compression(self) -> list[LossyCompression] | None:
        return self._source.lossy_compression

    @property
    def transcoder(self) -> Encoder | None:
        return self._source.transcoder

    def get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        return self._source.get_encoded_tile(
            tile, self._source.default_z, self._source.default_path
        )

    def get_decoded_tile(
        self, tile_point: Point, z: float, path: str, cache: bool = True
    ) -> np.ndarray:
        return self._source.get_decoded_tile(
            tile_point, self._source.default_z, self._source.default_path
        )


@pytest.fixture
def focal_planes():
    yield FOCAL_PLANES


@pytest.fixture
def multi_focal_plane_pyramid(wsi: WsiDicom, focal_planes: list[float]) -> Pyramid:
    """Build a single-level pyramid with several focal planes from the smallest
    real level."""
    level = wsi.pyramids[0][-1]
    source_instance = level.default_instance
    image_data = MultiFocalPlaneImageData(source_instance.image_data, focal_planes)

    dataset = deepcopy(source_instance.dataset)
    dataset.TotalPixelMatrixFocalPlanes = len(focal_planes)
    dataset.NumberOfFrames = image_data.tiled_size.area * len(focal_planes)

    instance = WsiInstance(WsiDataset(dataset), image_data)
    return Pyramid([Level([instance], level.pixel_spacing)], [])


def optical_path_sequence(identifier: str) -> Sequence:
    item = Dataset()
    item.OpticalPathIdentifier = identifier
    return Sequence([item])


def sparse_instance(
    source_instance: WsiInstance,
    optical_path: str,
    focal_planes: list[float],
) -> WsiInstance:
    """Build a TILED_FULL instance for one optical path and its focal planes."""
    image_data = MultiFocalPlaneImageData(
        source_instance.image_data, focal_planes, [optical_path]
    )
    dataset = deepcopy(source_instance.dataset)
    dataset.SOPInstanceUID = generate_uid()
    dataset.TotalPixelMatrixFocalPlanes = len(focal_planes)
    dataset.NumberOfOpticalPaths = 1
    dataset.NumberOfFrames = image_data.tiled_size.area * len(focal_planes)
    dataset.OpticalPathSequence = optical_path_sequence(optical_path)
    return WsiInstance(WsiDataset(dataset), image_data)


@pytest.fixture
def sparse_pyramid(wsi: WsiDicom) -> Pyramid:
    """Build a single-level pyramid with a sparse (path x plane) grid: optical
    path "0" has two focal planes, optical path "1" has only one."""
    level = wsi.pyramids[0][-1]
    source_instance = level.default_instance
    instances = [
        sparse_instance(source_instance, "0", [0.0, 1.0]),
        sparse_instance(source_instance, "1", [0.0]),
    ]
    return Pyramid([Level(instances, level.pixel_spacing)], [])


@pytest.fixture
def non_uniform_pyramid(wsi: WsiDicom) -> Pyramid:
    """Build a two-level pyramid where the levels disagree on focal planes: the
    larger level has two, the smaller has one."""
    pyramid = wsi.pyramids[0]
    larger_level = pyramid[-2]
    smaller_level = pyramid[-1]
    base_pixel_spacing = larger_level.pixel_spacing
    levels = [
        Level(
            [sparse_instance(larger_level.default_instance, "0", [0.0, 1.0])],
            base_pixel_spacing,
        ),
        Level(
            [sparse_instance(smaller_level.default_instance, "0", [0.0])],
            base_pixel_spacing,
        ),
    ]
    return Pyramid(levels, [])


@pytest.fixture
def wsi(wsi_name: str, wsi_factory: Callable[[str], WsiDicom]) -> WsiDicom:
    return wsi_factory(wsi_name)


@pytest.fixture
def write_pyramid(
    tmp_path: Path, uid_generator: UidGenerator
) -> Callable[..., list[UPath]]:
    """Write a pyramid through the real `PyramidFileWriter` and return the paths."""

    def write(
        pyramid: Pyramid, instance_split: InstanceSplit = InstanceSplit.NONE
    ) -> list[UPath]:
        writer = PyramidFileWriter(
            pyramid=pyramid,
            output_path=tmp_path,
            uid_generator=uid_generator,
            max_threads=1,
            offset_table=OffsetTableType.BASIC,
            add_missing_levels=False,
            instance_split=instance_split,
        )
        return writer.write()

    return write


@pytest.mark.integration
class TestInstanceSplitIntegration:
    @pytest.mark.parametrize(
        ["instance_split", "expected_files", "expected_planes_per_file"],
        [
            (InstanceSplit.NONE, 1, len(FOCAL_PLANES)),
            (InstanceSplit.FOCAL_PLANE, len(FOCAL_PLANES), 1),
            (InstanceSplit.OPTICAL_PATH, 1, len(FOCAL_PLANES)),
            (
                InstanceSplit.FOCAL_PLANE | InstanceSplit.OPTICAL_PATH,
                len(FOCAL_PLANES),
                1,
            ),
        ],
    )
    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_split_on_focal_planes(
        self,
        multi_focal_plane_pyramid: Pyramid,
        write_pyramid: Callable[..., list[UPath]],
        instance_split: InstanceSplit,
        expected_files: int,
        expected_planes_per_file: int,
    ):
        # Arrange

        # Act
        filepaths = write_pyramid(multi_focal_plane_pyramid, instance_split)

        # Assert — one file per bucket, with the expected focal-plane count, and
        # every focal plane is written exactly once across the files.
        assert len(filepaths) == expected_files
        total_planes = 0
        for filepath in filepaths:
            assert isinstance(filepath, Path)
            dataset = dcmread(filepath)
            assert dataset.TotalPixelMatrixFocalPlanes == expected_planes_per_file
            total_planes += dataset.TotalPixelMatrixFocalPlanes
        assert total_planes == len(FOCAL_PLANES)

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_split_on_focal_planes_reopens_with_all_planes(
        self,
        multi_focal_plane_pyramid: Pyramid,
        write_pyramid: Callable[..., list[UPath]],
        tmp_path: Path,
    ):
        # Arrange

        # Act
        write_pyramid(multi_focal_plane_pyramid, InstanceSplit.FOCAL_PLANE)

        # Assert — the separate instances reassemble into one level exposing
        # both focal planes.
        with WsiDicom.open(tmp_path) as saved_wsi:
            base_level = saved_wsi.pyramids[0].base_level
            assert len(base_level.instances) == len(FOCAL_PLANES)
            assert sorted(base_level.focal_planes) == FOCAL_PLANES

    @pytest.mark.parametrize(
        "instance_split", [InstanceSplit.NONE, InstanceSplit.FOCAL_PLANE]
    )
    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    @pytest.mark.parametrize("focal_planes", [UNEQUAL_FOCAL_PLANES])
    def test_unequally_spaced_focal_planes_are_split(
        self,
        multi_focal_plane_pyramid: Pyramid,
        write_pyramid: Callable[..., list[UPath]],
        tmp_path: Path,
        instance_split: InstanceSplit,
    ):
        # Arrange

        # Act
        write_pyramid(multi_focal_plane_pyramid, instance_split)

        # Assert — one instance per plane, each round-tripping at its own
        # position.
        with WsiDicom.open(tmp_path) as saved_wsi:
            base_level = saved_wsi.pyramids[0].base_level
            assert len(base_level.instances) == len(UNEQUAL_FOCAL_PLANES)
            assert sorted(base_level.focal_planes) == UNEQUAL_FOCAL_PLANES

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_sparse_grid_splits_optical_paths(
        self,
        sparse_pyramid: Pyramid,
        write_pyramid: Callable[..., list[UPath]],
        tmp_path: Path,
    ):
        # Arrange

        # Act
        filepaths = write_pyramid(sparse_pyramid)

        # Assert — one instance per optical path, each preserving its own focal
        # planes on read-back.
        assert len(filepaths) == 2
        with WsiDicom.open(tmp_path) as saved_wsi:
            base_level = saved_wsi.pyramids[0].base_level
            assert len(base_level.instances) == 2
            assert sorted(base_level.optical_paths) == ["0", "1"]
            assert base_level.focal_planes_by_optical_path == {
                "0": [0.0, 1.0],
                "1": [0.0],
            }

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_non_uniform_pyramid_levels_are_rejected(
        self,
        non_uniform_pyramid: Pyramid,
        write_pyramid: Callable[..., list[UPath]],
    ):
        # Arrange

        # Act & Assert
        with pytest.raises(NotImplementedError, match="non-uniform levels"):
            write_pyramid(non_uniform_pyramid)
