import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterable, Tuple

import pytest
from openslide import OpenSlide
from pydicom.misc import is_dicom
from pytest_benchmark.fixture import BenchmarkFixture

from tests.conftest import SLIDE_FOLDER, WsiTestDefinitions
from wsidicom.config import settings
from wsidicom.geometry import Point, Region, Size
from wsidicom.wsidicom import WsiDicom

# settings.decoded_frame_cache_size = 0
# settings.encoded_frame_cache_size = 0


class SlideModule(Enum):
    OPENSLIDE = "openslide"
    WSIDICOM = "wsidicom"
    WSIDICOM_THREADS = "wsidicom_threads"


class ChunkOrder(Enum):
    ROW = "row"
    COL = "col"
    RANDOM = "random"


@dataclass
class ReadRegion:
    position: Tuple[int, int]
    level: int
    size: Tuple[int, int]
    openslide_position: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        self.openslide_position = (
            self.position[0] * 2**self.level,
            self.position[1] * 2**self.level,
        )


def chunk_region_of_interest(roi: Region, chunk_size: Size, order: ChunkOrder):
    if order == ChunkOrder.ROW or order == ChunkOrder.RANDOM:
        chunks = [
            ReadRegion((x, y), 0, chunk_size.to_tuple())
            for y in range(roi.start.y, roi.end.y, chunk_size.width)
            for x in range(roi.start.x, roi.end.x, chunk_size.height)
            for _ in range(2)
        ]
        if order == ChunkOrder.RANDOM:
            random.seed(42)
            chunks = random.choices(chunks, k=len(chunks))
    elif order == ChunkOrder.COL:
        chunks = [
            ReadRegion((x, y), 0, chunk_size.to_tuple())
            for x in range(roi.start.x, roi.end.x, chunk_size.width)
            for y in range(roi.start.y, roi.end.y, chunk_size.height)
            for _ in range(2)
        ]
    else:
        raise ValueError("order must be 'row' or 'col'")
    return chunks


@pytest.fixture
def full_tiled_slide_folder():
    yield next(
        SLIDE_FOLDER.joinpath(wsi_definition["path"])
        for key, wsi_definition in WsiTestDefinitions.test_definitions.items()
        if key == "Histech^Samantha [1229631]"
    )


@pytest.fixture
def open_and_read_action(
    slide_folder: Path,
    module: SlideModule,
):
    read_region = ReadRegion((0, 0), 0, (256, 256))
    if module == SlideModule.OPENSLIDE:
        slide_file = next(
            file for file in slide_folder.iterdir() if file.is_file() and is_dicom(file)
        )

        def action():
            with OpenSlide(slide_file) as slide:
                slide.read_region(
                    read_region.openslide_position, read_region.level, read_region.size
                ).load()

        yield action
    elif module == SlideModule.WSIDICOM:

        def action():
            with WsiDicom.open(slide_folder) as slide:
                slide.read_region(
                    read_region.position,
                    read_region.level,
                    read_region.size,
                ).load()

        yield action
    elif module == SlideModule.WSIDICOM_THREADS:
        threads = os.cpu_count() or 8

        def action():
            with WsiDicom.open(slide_folder) as slide:
                slide.read_region(
                    read_region.position,
                    read_region.level,
                    read_region.size,
                    threads=threads,
                ).load()

        yield action
    else:
        raise ValueError(module)


@pytest.fixture
def read_action(
    full_tiled_slide_folder: Path,
    module: SlideModule,
):
    if module == SlideModule.OPENSLIDE:
        slide_file = next(
            file
            for file in full_tiled_slide_folder.iterdir()
            if file.is_file() and is_dicom(file)
        )
        slide = OpenSlide(slide_file)

        def action(read_region: ReadRegion):
            slide.read_region(
                read_region.openslide_position, read_region.level, read_region.size
            ).load()

        yield action
        slide.close()
    elif module == SlideModule.WSIDICOM:
        slide = WsiDicom.open(full_tiled_slide_folder)

        def action(read_region: ReadRegion):
            slide.read_region(
                read_region.position, read_region.level, read_region.size, threads=1
            ).load()

        yield action
        slide.close()
    elif module == SlideModule.WSIDICOM_THREADS:
        slide = WsiDicom.open(full_tiled_slide_folder)
        threads = os.cpu_count() or 8

        def action(read_region: ReadRegion):
            slide.read_region(
                read_region.position,
                read_region.level,
                read_region.size,
                threads=threads,
            ).load()

        yield action
        slide.close()
    else:
        raise ValueError(module)


@pytest.fixture
def threaded_read_action(
    full_tiled_slide_folder: Path,
    module: SlideModule,
):
    if module == SlideModule.OPENSLIDE:
        slide_file = next(
            file
            for file in full_tiled_slide_folder.iterdir()
            if file.is_file() and is_dicom(file)
        )
        slide = OpenSlide(slide_file)

        def read_region(read_region: ReadRegion):
            slide.read_region(
                read_region.openslide_position, read_region.level, read_region.size
            ).load()

        def action(read_regions: Iterable[ReadRegion]):
            with ThreadPoolExecutor() as executor:
                executor.map(read_region, read_regions)

        yield action
        slide.close()
    elif module == SlideModule.WSIDICOM:
        slide = WsiDicom.open(full_tiled_slide_folder)

        def read_region(read_region: ReadRegion):
            slide.read_region(
                read_region.position, read_region.level, read_region.size, threads=1
            ).load()

        def action(read_regions: Iterable[ReadRegion]):
            with ThreadPoolExecutor() as executor:
                executor.map(read_region, read_regions)

        yield action
        slide.close()
    elif module == SlideModule.WSIDICOM_THREADS:
        slide = WsiDicom.open(full_tiled_slide_folder)
        threads = os.cpu_count() or 8

        def read_region(read_region: ReadRegion):
            slide.read_region(
                read_region.position,
                read_region.level,
                read_region.size,
                threads=threads,
            ).load()

        def action(read_regions: Iterable[ReadRegion]):
            with ThreadPoolExecutor() as executor:
                executor.map(read_region, read_regions)

        yield action
        slide.close()
    else:
        raise ValueError(module)


@pytest.mark.unittest
class TestWsiDicomBenchmark:
    # @pytest.mark.parametrize("slide_folder", WsiTestDefinitions.folders())
    # @pytest.mark.parametrize("module", [SlideModule.OPENSLIDE, SlideModule.WSIDICOM])
    # @pytest.mark.benchmark(group="Open and read reagion:")
    # def test_open_and_read_region(
    #     self,
    #     open_and_read_action: Callable[[None], None],
    #     benchmark: BenchmarkFixture,
    # ):
    #     benchmark.pedantic(
    #         open_and_read_action, rounds=3, iterations=1, warmup_rounds=1
    #     )

    @pytest.mark.parametrize(
        "module",
        [
            SlideModule.WSIDICOM,
            # SlideModule.WSIDICOM_THREADS,
            SlideModule.OPENSLIDE,
        ],
    )
    @pytest.mark.parametrize("tile_aligned", [True, False])
    @pytest.mark.parametrize(
        "chunk_size",
        [
            # Size(64, 64),
            # Size(128, 128),
            # Size(256, 256),
            Size(512, 512),
            # Size(1024, 1024),
            # Size(4096, 4096),
            # Size(8192, 8192),
        ],
    )
    @pytest.mark.parametrize(
        "order",
        [
            ChunkOrder.ROW,
            ChunkOrder.COL,
            # ChunkOrder.RANDOM
        ],
    )
    def test_read_region(
        self,
        tile_aligned: bool,
        chunk_size: Size,
        order: ChunkOrder,
        read_action: Callable[[ReadRegion], None],
        benchmark: BenchmarkFixture,
    ):
        # Arrange
        benchmark.group = f"Read region tile chunk size: {chunk_size} order: {order} tile_aligned: {tile_aligned}"
        if tile_aligned:
            region_of_interest = Region(Point(8960, 7168), Size(8192, 8192))
        else:
            region_of_interest = Region(Point(9000, 7500), Size(8192, 8192))

        chunks = chunk_region_of_interest(region_of_interest, chunk_size, order)
        chunk_iterator = iter(chunks)
        rounds = len(chunks) // 2

        def setup():
            return (next(chunk_iterator),), {}

        def get_next():
            return next(chunk_iterator)

        # Act
        benchmark.pedantic(
            read_action, setup=setup, rounds=rounds, iterations=1, warmup_rounds=rounds
        )

    # @pytest.mark.parametrize(
    #     "module",
    #     [
    #         SlideModule.WSIDICOM,
    #         # SlideModule.WSIDICOM_THREADS,
    #         SlideModule.OPENSLIDE,
    #     ],
    # )
    # @pytest.mark.parametrize(
    #     "chunk_size",
    #     [
    #         # Size(64, 64),
    #         # Size(128, 128),
    #         # Size(256, 256),
    #         Size(512, 512),
    #         # Size(1024, 1024),
    #         # Size(4096, 4096),
    #         # Size(8192, 8192),
    #     ],
    # )
    # @pytest.mark.parametrize(
    #     "order",
    #     [
    #         ChunkOrder.ROW,
    #         ChunkOrder.COL,
    #         # ChunkOrder.RANDOM
    #     ],
    # )
    # def test_threaded_read_region(
    #     self,
    #     chunk_size: Size,
    #     order: ChunkOrder,
    #     threaded_read_action: Callable[[Iterable[ReadRegion]], None],
    #     benchmark: BenchmarkFixture,
    # ):
    #     # Arrange
    #     benchmark.group = (
    #         f"Threaded read region chunk size: {chunk_size} order: {order}"
    #     )
    #     region_of_interest = Region(Point(8960, 7168), Size(8192, 8192))

    #     def setup():
    #         chunks = chunk_region_of_interest(region_of_interest, chunk_size, order)
    #         return (chunks,), {}

    #     # Act
    #     benchmark.pedantic(
    #         threaded_read_action, setup=setup, rounds=4, iterations=1, warmup_rounds=1
    #     )
