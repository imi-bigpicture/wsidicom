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

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image, ImageChops
from pydicom import Dataset, dcmread
from pydicom.encaps import generate_pixel_data_frame

from tests.conftest import WsiTestDefinitions
from wsidicom import ConcatenationByBytes, ConcatenationByFrames, WsiDicom
from wsidicom.geometry import Size
from wsidicom.metadata.uid_generator import CallableUidGenerator

# Item tag (4 bytes) + item length (4 bytes) per encapsulated frame.
_ITEM_HEADER_BYTES = 8

_FULL = WsiTestDefinitions.wsi_names("full")


def _read_strip(wsi: WsiDicom, strip_size: Size) -> Image.Image:
    """Read a base-level strip of `strip_size` from the top-left."""
    return wsi.read_region((0, 0), 0, (strip_size.width, strip_size.height))


@pytest.fixture
def wsi(
    request: pytest.FixtureRequest, wsi_factory: Callable[[str], WsiDicom]
) -> WsiDicom:
    return wsi_factory(request.param)


@pytest.fixture
def strip_size(wsi: WsiDicom) -> Size:
    # Frames are raster-ordered (row-major), so parts are contiguous bands of
    # rows; a full-height strip a few tiles wide touches a tile in every row and
    # thus crosses every concatenation part boundary.
    width, height = wsi.pyramids[0].base_level.size.to_tuple()
    strip_width = min(width, 3 * wsi.pyramids[0].tile_size.width)
    return Size(strip_width, height)


@pytest.fixture
def frame_count(wsi: WsiDicom) -> int:
    """Base-level frame count of the WSI."""
    return wsi.pyramids[0].base_level.default_instance.dataset.frame_count


@pytest.fixture
def unsplit_dir(
    wsi: WsiDicom, tmp_path: Path, uid_generator: CallableUidGenerator
) -> Path:
    """Save `wsi` unsplit once; the baseline every split test compares against."""
    unsplit_dir = tmp_path / "unsplit"
    unsplit_dir.mkdir()
    wsi.save(unsplit_dir, uid_generator=uid_generator)
    return unsplit_dir


@pytest.fixture
def unsplit_region(unsplit_dir: Path, strip_size: Size) -> Image.Image:
    """A base-level strip from the unsplit save."""
    with WsiDicom.open(unsplit_dir) as unsplit_wsi:
        return _read_strip(unsplit_wsi, strip_size)


@pytest.fixture
def largest_encapsulated_bytes(unsplit_dir: Path) -> int:
    """Largest encapsulated pixel-data size across the unsplit save's files."""
    largest = 0
    for path in unsplit_dir.glob("*.dcm"):
        ds = dcmread(path)
        frames = generate_pixel_data_frame(ds.PixelData, int(ds.NumberOfFrames))
        largest = max(largest, sum(len(f) + _ITEM_HEADER_BYTES for f in frames))
    return largest


@pytest.fixture
def split_dir(tmp_path: Path) -> Path:
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    return split_dir


@pytest.mark.integration
@pytest.mark.parametrize("wsi", _FULL, indirect=True)
class TestConcatenation:
    def test_frame_split_matches_unsplit(
        self,
        wsi: WsiDicom,
        uid_generator: CallableUidGenerator,
        frame_count: int,
        unsplit_region: Image.Image,
        strip_size: Size,
        split_dir: Path,
    ):
        # Arrange — a frame budget that forces the base level into several parts.
        concatenation = ConcatenationByFrames(frame_count // 3)

        # Act
        wsi.save(split_dir, uid_generator=uid_generator, concatenation=concatenation)

        # Assert — split output reopens pixel-identical and partitions correctly
        with WsiDicom.open(split_dir) as split_wsi:
            split_region = _read_strip(split_wsi, strip_size)
        self._assert_same_pixels(split_region, unsplit_region)
        groups = self._concatenation_groups(split_dir)
        assert any(len(parts) > 1 for parts in groups.values())
        for parts in groups.values():
            self._assert_partitions(parts)

    def test_byte_split_matches_unsplit(
        self,
        wsi: WsiDicom,
        uid_generator: CallableUidGenerator,
        unsplit_region: Image.Image,
        largest_encapsulated_bytes: int,
        strip_size: Size,
        split_dir: Path,
    ):
        # Arrange — a byte budget ~1/3 of the base level's pixel data.
        max_bytes = largest_encapsulated_bytes // 3
        concatenation = ConcatenationByBytes(max_bytes)

        # Act
        wsi.save(split_dir, uid_generator=uid_generator, concatenation=concatenation)

        # Assert — reopens pixel-identical, partitions, and stays in budget
        with WsiDicom.open(split_dir) as split_wsi:
            split_region = _read_strip(split_wsi, strip_size)
        self._assert_same_pixels(split_region, unsplit_region)
        groups = self._concatenation_groups(split_dir)
        assert any(len(parts) > 1 for parts in groups.values())
        for parts in groups.values():
            self._assert_partitions(parts)
            self._assert_byte_budget(parts, max_bytes)

    @staticmethod
    def _concatenation_groups(directory: Path) -> dict[str, list[Dataset]]:
        """Group written files by concatenation source UID (or own UID)."""
        groups: dict[str, list[Dataset]] = defaultdict(list)
        for path in directory.glob("*.dcm"):
            ds = dcmread(path, stop_before_pixels=True)
            source = getattr(ds, "SOPInstanceUIDOfConcatenationSource", None)
            key = source if source is not None else ds.SOPInstanceUID
            groups[key].append(ds)
        return groups

    @staticmethod
    def _assert_same_pixels(split: Image.Image, unsplit: Image.Image) -> None:
        """Assert the reopened split and unsplit strips are pixel-identical."""
        diff = ImageChops.difference(split.convert("RGB"), unsplit.convert("RGB"))
        assert diff.getbbox() is None

    @staticmethod
    def _assert_partitions(parts: list[Dataset]) -> None:
        """Assert a concatenation group partitions its frames contiguously."""
        if len(parts) == 1:
            # Single instance: no concatenation attributes expected.
            assert "ConcatenationUID" not in parts[0]
            return
        parts = sorted(parts, key=lambda ds: int(ds.InConcatenationNumber))
        concatenation_uid = parts[0].ConcatenationUID
        source_uid = parts[0].SOPInstanceUIDOfConcatenationSource
        running_offset = 0
        for index, ds in enumerate(parts):
            assert ds.ConcatenationUID == concatenation_uid
            assert ds.SOPInstanceUIDOfConcatenationSource == source_uid
            assert int(ds.InConcatenationNumber) == index + 1
            assert int(ds.ConcatenationFrameOffsetNumber) == running_offset
            running_offset += int(ds.NumberOfFrames)

    @staticmethod
    def _assert_byte_budget(parts: list[Dataset], max_bytes: int) -> None:
        """Assert each part's pixel data is within budget (or a single frame)."""
        for ds in parts:
            full = dcmread(ds.filename)
            frames = list(
                generate_pixel_data_frame(full.PixelData, int(full.NumberOfFrames))
            )
            part_bytes = sum(len(frame) + _ITEM_HEADER_BYTES for frame in frames)
            # A part may only exceed the budget when it is a single frame that
            # is itself larger than the budget.
            assert part_bytes <= max_bytes or len(frames) == 1

