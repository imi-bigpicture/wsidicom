#    Copyright 2021, 2022, 2023 SECTRA AB
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

"""Tests for PyramidFileWriter."""

import os
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import pytest
from decoy import Decoy, matchers
from PIL import ImageChops, ImageFilter, ImageStat

from tests.conftest import WsiTestDefinitions
from wsidicom import WsiDicom
from wsidicom.codec import Encoder
from wsidicom.file import OffsetTableType
from wsidicom.file.file_writer import (
    BaseFileWriter,
    PyramidFileWriter,
)
from wsidicom.file.instance_split import InstanceSplit
from wsidicom.metadata.uid_generator import UidGenerator
from wsidicom.series import Pyramid


@pytest.fixture
def decoy() -> Decoy:
    """Create a Decoy instance for mocking."""
    return Decoy()


class SplitOnlyWriter(BaseFileWriter):
    """Minimal concrete writer to exercise `_split_planes_paths` in isolation."""

    def __init__(self, instance_split: InstanceSplit):
        super().__init__(
            output_path=".",
            uid_generator=None,  # type: ignore[arg-type]
            transcoder=None,
            force_transcoding=False,
            offset_table=None,
            file_options=None,
            instance_number_start=0,
            instance_split=instance_split,
        )

    def write(self):  # pragma: no cover - not exercised
        raise NotImplementedError()


@pytest.mark.unittest
class TestPyramidFileWriterHelpers:
    """Tests for PyramidFileWriter static helper methods."""


@pytest.mark.unittest
class TestBaseFileWriterSplit:
    """Tests for `BaseFileWriter._split_planes_paths`."""

    @pytest.mark.parametrize(
        ["split", "expected"],
        [
            (
                InstanceSplit.NONE,
                [([0.0, 1.0], ["0", "1"])],
            ),
            (
                InstanceSplit.FOCAL_PLANE,
                [([0.0], ["0", "1"]), ([1.0], ["0", "1"])],
            ),
            (
                InstanceSplit.OPTICAL_PATH,
                [([0.0, 1.0], ["0"]), ([0.0, 1.0], ["1"])],
            ),
            (
                InstanceSplit.FOCAL_PLANE | InstanceSplit.OPTICAL_PATH,
                [([0.0], ["0"]), ([1.0], ["0"]), ([0.0], ["1"]), ([1.0], ["1"])],
            ),
        ],
    )
    def test_split_planes_paths_buckets(
        self,
        split: InstanceSplit,
        expected: list[tuple[list[float], list[str]]],
    ):
        # Arrange — a complete grid: both optical paths have both focal planes.
        writer = SplitOnlyWriter(split)

        # Act
        buckets = writer._split_planes_paths({"0": [0.0, 1.0], "1": [0.0, 1.0]})

        # Assert
        assert buckets == expected

    def test_unequally_spaced_focal_planes_split_even_without_flag(self):
        # Arrange — unequally spaced focal planes cannot share one TILED_FULL
        # instance, so they are split per plane even with InstanceSplit.NONE.
        writer = SplitOnlyWriter(InstanceSplit.NONE)

        # Act
        buckets = writer._split_planes_paths({"0": [0.0, 0.5, 2.0]})

        # Assert
        assert buckets == [([0.0], ["0"]), ([0.5], ["0"]), ([2.0], ["0"])]

    def test_equally_spaced_focal_planes_stay_combined(self):
        # Arrange
        writer = SplitOnlyWriter(InstanceSplit.NONE)

        # Act
        buckets = writer._split_planes_paths({"0": [0.0, 1.0, 2.0]})

        # Assert
        assert buckets == [([0.0, 1.0, 2.0], ["0"])]

    def test_sparse_grid_splits_optical_paths_even_without_flag(self):
        # Arrange — optical path "1" is missing focal plane 1.0, so the grid is
        # sparse and the optical paths must be split into separate instances.
        writer = SplitOnlyWriter(InstanceSplit.NONE)

        # Act
        buckets = writer._split_planes_paths({"0": [0.0, 1.0], "1": [0.0]})

        # Assert — each bucket is a complete grid.
        assert buckets == [([0.0, 1.0], ["0"]), ([0.0], ["1"])]

    def test_split_covers_every_present_cell_exactly_once(self):
        # Arrange — a sparse grid.
        focal_planes_by_optical_path = {"a": [0.0, 1.0, 2.0], "b": [0.0]}
        writer = SplitOnlyWriter(InstanceSplit.NONE)

        # Act
        buckets = writer._split_planes_paths(focal_planes_by_optical_path)

        # Assert — every present (plane, path) cell is covered exactly once and
        # no absent cell is introduced.
        pairs = [
            (focal_plane, path)
            for planes, paths in buckets
            for focal_plane in planes
            for path in paths
        ]
        expected_pairs = [
            (focal_plane, path)
            for path, planes in focal_planes_by_optical_path.items()
            for focal_plane in planes
        ]
        assert sorted(pairs) == sorted(expected_pairs)
        assert len(pairs) == len(set(pairs))


@pytest.mark.integration
class TestPyramidFileWriterIntegration:
    """Integration tests for PyramidFileWriter using real test DICOM files."""

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_generate_all_levels(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        uid_generator: UidGenerator,
    ):
        """Test generating a full pyramid preserves all existing levels."""
        # Arrange
        wsi = wsi_factory(wsi_name)
        pyramid = wsi.pyramids[0]
        expected_levels_count = len(pyramid)

        # Act
        generator = PyramidFileWriter(
            pyramid=pyramid,
            output_path=tmp_path,
            uid_generator=uid_generator,
            max_threads=4,
            offset_table=OffsetTableType.BASIC,
        )
        generator.write()

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            assert len(saved_wsi.pyramids[0]) == expected_levels_count

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_generate_from_base_level_only(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        uid_generator: UidGenerator,
    ):
        """Test generating a full pyramid from only the base level."""
        # Arrange
        wsi = wsi_factory(wsi_name)
        base_level = wsi.pyramids[0].get(0)
        pyramid_base_only = Pyramid([base_level], [])
        expected_lowest_single_tile = pyramid_base_only.lowest_single_tile_level

        # Act
        generator = PyramidFileWriter(
            pyramid=pyramid_base_only,
            output_path=tmp_path,
            uid_generator=uid_generator,
            max_threads=4,
            offset_table=OffsetTableType.BASIC,
        )
        generator.write()

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            saved_pyramid = saved_wsi.pyramids[0]
            assert len(saved_pyramid) == expected_lowest_single_tile + 1
            assert saved_pyramid[-1].size.all_less_than_or_equal(
                saved_pyramid.tile_size
            )

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_generate_output_is_readable(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        uid_generator: UidGenerator,
    ):
        """Test that generated files produce readable image data."""
        # Arrange
        wsi = wsi_factory(wsi_name)
        pyramid = wsi.pyramids[0]

        # Act
        generator = PyramidFileWriter(
            pyramid=pyramid,
            output_path=tmp_path,
            uid_generator=uid_generator,
            max_threads=4,
            offset_table=OffsetTableType.BASIC,
        )
        generator.write()

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            for level in saved_wsi.pyramids[0]:
                size = level.size.to_tuple()
                region_size = (min(64, size[0]), min(64, size[1]))
                image = saved_wsi.read_region((0, 0), level.level, region_size)
                assert image.size == region_size

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_generate_downsampled_level_quality(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        uid_generator: UidGenerator,
    ):
        """Test that a downsampled level is visually close to the original."""
        # Arrange
        wsi = wsi_factory(wsi_name)
        pyramid = wsi.pyramids[0]
        if len(pyramid) < 2:
            pytest.skip("Need at least 2 levels for quality comparison")

        target_level = pyramid[1]
        base_level = pyramid.get(0)
        pyramid_base_only = Pyramid([base_level], [])

        # Act
        generator = PyramidFileWriter(
            pyramid=pyramid_base_only,
            output_path=tmp_path,
            uid_generator=uid_generator,
            max_threads=4,
            offset_table=OffsetTableType.BASIC,
        )
        generator.write()

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            target_size = target_level.size.to_tuple()
            created_level = saved_wsi.pyramids[0][1]
            created_size = created_level.size.to_tuple()
            assert created_size == target_size

            # Compare a bounded centre crop, not the whole (gigapixel) level —
            # reading an entire level stitches every tile into one in-memory
            # image (hundreds of MB to GBs). The centre is more likely than the
            # corner to contain tissue, keeping the comparison meaningful.
            width, height = created_size
            crop_width = min(1024, width)
            crop_height = min(1024, height)
            location = (width // 2 - crop_width // 2, height // 2 - crop_height // 2)
            crop_size = (crop_width, crop_height)

            created = saved_wsi.read_region(location, created_level.level, crop_size)
            original = wsi.read_region(location, target_level.level, crop_size)

            blur = ImageFilter.GaussianBlur(2)
            diff = ImageChops.difference(created.filter(blur), original.filter(blur))
            for band_rms in ImageStat.Stat(diff).rms:
                assert band_rms < 2


@pytest.mark.integration
class TestPyramidFileWriterBottomUpIntegration:
    """Integration tests for PyramidFileWriter."""

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_generate_all_levels_bottom_up(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        uid_generator: UidGenerator,
    ):
        """Test generating a full pyramid with bottom-up threading model."""
        # Arrange
        wsi = wsi_factory(wsi_name)
        pyramid = wsi.pyramids[0]
        expected_levels_count = len(pyramid)

        # Act
        generator = PyramidFileWriter(
            pyramid=pyramid,
            output_path=tmp_path,
            uid_generator=uid_generator,
            max_threads=4,
            offset_table=OffsetTableType.BASIC,
        )
        generator.write()

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            assert len(saved_wsi.pyramids[0]) == expected_levels_count

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_generate_from_base_level_only_bottom_up(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        uid_generator: UidGenerator,
    ):
        """Test generating from base only with bottom-up threading model."""
        # Arrange
        wsi = wsi_factory(wsi_name)
        base_level = wsi.pyramids[0].get(0)
        pyramid_base_only = Pyramid([base_level], [])
        expected_lowest_single_tile = pyramid_base_only.lowest_single_tile_level

        # Act
        generator = PyramidFileWriter(
            pyramid=pyramid_base_only,
            output_path=tmp_path,
            uid_generator=uid_generator,
            max_threads=4,
            offset_table=OffsetTableType.BASIC,
        )
        generator.write()

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            saved_pyramid = saved_wsi.pyramids[0]
            assert len(saved_pyramid) == expected_lowest_single_tile + 1
            assert saved_pyramid[-1].size.all_less_than_or_equal(
                saved_pyramid.tile_size
            )


@pytest.mark.integration
class TestPyramidFileWriterBottomUpSequentialIntegration:
    """Integration tests for PyramidFileWriter with BOTTOM_UP + source_workers=1."""

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_generate_all_levels_bottom_up_sequential(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        uid_generator: UidGenerator,
    ):
        """Test generating a full pyramid with sequential bottom-up model."""
        # Arrange
        wsi = wsi_factory(wsi_name)
        pyramid = wsi.pyramids[0]
        expected_levels_count = len(pyramid)

        # Act
        generator = PyramidFileWriter(
            pyramid=pyramid,
            output_path=tmp_path,
            uid_generator=uid_generator,
            max_threads=4,
            offset_table=OffsetTableType.BASIC,
            source_workers=1,
        )
        generator.write()

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            assert len(saved_wsi.pyramids[0]) == expected_levels_count

    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_generate_from_base_level_only_bottom_up_sequential(
        self,
        wsi_name: str,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        uid_generator: UidGenerator,
    ):
        """Test generating from base only with sequential bottom-up model."""
        # Arrange
        wsi = wsi_factory(wsi_name)
        base_level = wsi.pyramids[0].get(0)
        pyramid_base_only = Pyramid([base_level], [])
        expected_lowest_single_tile = pyramid_base_only.lowest_single_tile_level

        # Act
        generator = PyramidFileWriter(
            pyramid=pyramid_base_only,
            output_path=tmp_path,
            uid_generator=uid_generator,
            max_threads=4,
            offset_table=OffsetTableType.BASIC,
            source_workers=1,
        )
        generator.write()

        # Assert
        with WsiDicom.open(tmp_path) as saved_wsi:
            saved_pyramid = saved_wsi.pyramids[0]
            assert len(saved_pyramid) == expected_lowest_single_tile + 1
            assert saved_pyramid[-1].size.all_less_than_or_equal(
                saved_pyramid.tile_size
            )


@pytest.mark.integration
class TestPyramidFileWriterFailFast:
    """The pull-based pipeline must fail fast — surface the cause and clean up,
    rather than deadlock — when a stage raises mid-stream."""

    @pytest.mark.parametrize("source_workers", [None, 1])
    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_encoding_failure_fails_fast(
        self,
        wsi_name: str,
        source_workers: int | None,
        uid_generator: UidGenerator,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        decoy: Decoy,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A transcoder that always fails to encode must make write() raise the
        injected error promptly (not hang) and remove its temp directory."""

        # Arrange — a transcoder that mimics the source encoder but fails to encode
        class InjectedEncodeError(Exception):
            pass

        wsi = wsi_factory(wsi_name)
        pyramid = wsi.pyramids[0]
        source_image_data = list(pyramid.base_level.instances.values())[0].image_data

        failing_encoder = decoy.mock(cls=Encoder)
        decoy.when(failing_encoder.transfer_syntax).then_return(
            source_image_data.transfer_syntax
        )
        decoy.when(failing_encoder.bits).then_return(source_image_data.bits)
        decoy.when(failing_encoder.samples_per_pixel).then_return(
            source_image_data.samples_per_pixel
        )
        decoy.when(failing_encoder.photometric_interpretation).then_return(
            source_image_data.photometric_interpretation
        )
        decoy.when(failing_encoder.lossy_method).then_return(None)
        decoy.when(failing_encoder.encode(matchers.Anything())).then_raise(
            InjectedEncodeError("injected encode failure")
        )

        # Capture the pipeline temp dir(s) so we can assert they are cleaned up.
        created_temp_dirs: list = []
        real_mkdtemp = tempfile.mkdtemp

        def capturing_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            created_temp_dirs.append(path)
            return path

        monkeypatch.setattr(tempfile, "mkdtemp", capturing_mkdtemp)

        writer = PyramidFileWriter(
            pyramid=pyramid,
            output_path=tmp_path,
            uid_generator=uid_generator,
            max_threads=4,
            transcoder=failing_encoder,
            force_transcoding=True,
            offset_table=OffsetTableType.BASIC,
            source_workers=source_workers,
        )

        # Act — run write() on a daemon thread so a hang shows up as a join
        # timeout rather than wedging the whole test session.
        outcome: dict[str, Exception] = {}

        def run_write() -> None:
            try:
                writer.write()
            except Exception as error:
                outcome["error"] = error

        worker = threading.Thread(target=run_write, daemon=True)
        worker.start()
        worker.join(timeout=60.0)

        # Assert — returned (no deadlock), surfaced the injected cause, cleaned up
        assert not worker.is_alive(), "write() deadlocked instead of failing fast"
        assert isinstance(outcome.get("error"), InjectedEncodeError)
        assert created_temp_dirs, "pipeline did not create a temp dir"
        for path in created_temp_dirs:
            assert not os.path.exists(path), f"temp dir not cleaned up: {path}"

    @pytest.mark.parametrize("source_workers", [None, 1])
    @pytest.mark.parametrize("wsi_name", WsiTestDefinitions.wsi_names("full"))
    def test_source_read_failure_fails_fast(
        self,
        wsi_name: str,
        source_workers: int | None,
        uid_generator: UidGenerator,
        wsi_factory: Callable[[str], WsiDicom],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """A source whose tile read raises must make write() raise that error
        promptly (not hang) and remove its temp directory."""

        # Arrange — make every read on the base-level source raise
        class InjectedReadError(Exception):
            pass

        wsi = wsi_factory(wsi_name)
        pyramid = wsi.pyramids[0]

        def raising_read(*args, **kwargs):
            raise InjectedReadError("injected source read failure")

        for instance in pyramid.base_level.instances.values():
            for read_method in (
                "get_decoded_tiles",
                "get_encoded_tiles",
                "get_encoded_and_decoded_tiles",
            ):
                monkeypatch.setattr(instance.image_data, read_method, raising_read)

        created_temp_dirs: list = []
        real_mkdtemp = tempfile.mkdtemp

        def capturing_mkdtemp(*args, **kwargs):
            path = real_mkdtemp(*args, **kwargs)
            created_temp_dirs.append(path)
            return path

        monkeypatch.setattr(tempfile, "mkdtemp", capturing_mkdtemp)

        writer = PyramidFileWriter(
            pyramid=pyramid,
            output_path=tmp_path,
            uid_generator=uid_generator,
            max_threads=4,
            offset_table=OffsetTableType.BASIC,
            source_workers=source_workers,
        )

        # Act
        outcome: dict[str, Exception] = {}

        def run_write() -> None:
            try:
                writer.write()
            except Exception as error:
                outcome["error"] = error

        worker = threading.Thread(target=run_write, daemon=True)
        worker.start()
        worker.join(timeout=60.0)

        # Assert
        assert not worker.is_alive(), "write() deadlocked instead of failing fast"
        assert isinstance(outcome.get("error"), InjectedReadError)
        assert created_temp_dirs, "pipeline did not create a temp dir"
        for path in created_temp_dirs:
            assert not os.path.exists(path), f"temp dir not cleaned up: {path}"
