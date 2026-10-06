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

import json
import os
import platform
import random
import sys
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from enum import Enum
from pathlib import Path
from random import Random
from typing import Any

import pytest
from dicomweb_client import DICOMfileClient
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.sequence import Sequence
from pydicom.uid import (
    JPEG2000,
    UID,
    ExplicitVRLittleEndian,
    JPEGBaseline8Bit,
    generate_uid,
)

from tests.file.io.test_wsidicom_writer import WsiDicomTestImageData
from wsidicom import WsiDicom
from wsidicom.config import get_settings
from wsidicom.file import OffsetTableType
from wsidicom.geometry import Size
from wsidicom.instance import TileType
from wsidicom.metadata.uid_generator import CallableUidGenerator
from wsidicom.thread import ReadExecutor
from wsidicom.web.wsidicom_web_client import WsiDicomWebClient

SLIDE_FOLDER = Path(os.environ.get("WSIDICOM_TESTDIR", "tests/testdata/slides"))
REGION_DEFINITIONS_FILE = "tests/testdata/region_definitions.json"
TEST_FRAME_FILE = Path("tests/testdata/test_frame.jpg")
"""An all black 1024x1024 RGB baseline JPEG, of odd length."""


def create_layer_file(
    filename: Path, dataset: Dataset, file_meta: FileMetaDataset
) -> None:
    """Write a dataset with its file meta information to a file."""
    dataset.file_meta = file_meta
    dataset.save_as(
        filename, enforce_file_format=True, implicit_vr=False, little_endian=True
    )


def skip_if_hash_unstable(transfer_syntax: UID) -> None:
    """Skip a pixel-hash test if the syntax does not decode byte-identically here.

    On such a platform the md5 comparison would spuriously fail. Lossy JPEG2000
    decodes differently on macOS.

    Call at the top of a test that asserts an md5 checksum; tests that do not
    hash simply never call it and so are never skipped.
    """
    if transfer_syntax == JPEG2000 and platform.system() == "Darwin":
        pytest.skip("Lossy JPEG2000 does not produce identical output on macOS.")


class SingleThreadDicomFileClient(DICOMfileClient):
    """Test-only shim that runs DICOMfileClient calls on one dedicated thread.

    The calls are those wsidicom uses, and returned iterators are drained there.

    DICOMfileClient creates a SQLite connection bound to the thread that
    constructs it, so reads through a shared multi-thread pool would otherwise
    fail with sqlite3.ProgrammingError. A real DICOMweb HTTP client needs no
    shim. The wrapped client is therefore also constructed on the dedicated
    thread; the base __init__ is deliberately not called, so inherited methods
    without a thread-confined override fail loudly instead of touching SQLite
    from the wrong thread.
    """

    def __init__(self, url: str) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="DicomFileClient"
        )
        self._client = self._executor.submit(DICOMfileClient, url).result()

    def shutdown(self) -> None:
        self._executor.shutdown()

    def search_for_instances(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_on_client_thread("search_for_instances", *args, **kwargs)

    def retrieve_instance_metadata(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_on_client_thread(
            "retrieve_instance_metadata", *args, **kwargs
        )

    def iter_instance_frames(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_on_client_thread("iter_instance_frames", *args, **kwargs)

    def _call_on_client_thread(
        self, method_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        def run() -> Any:
            result = getattr(self._client, method_name)(*args, **kwargs)
            if isinstance(result, Iterator):
                return iter(list(result))
            return result

        return self._executor.submit(run).result()


class WsiInputType(Enum):
    FILE = "file"
    WEB = "web"


class WsiTestDefinitions:
    """Interface for reading test parameters from definition file."""

    with open(REGION_DEFINITIONS_FILE) as json_file:
        test_definitions: dict[str, dict[str, Any]] = json.load(json_file)
    if len(test_definitions) == 0:
        pytest.skip("No test definition found, skipping.")

    @classmethod
    def folders(cls) -> list[Path]:
        return [SLIDE_FOLDER.joinpath(path) for _, path in cls._get_parameter("path")]

    @classmethod
    def folders_and_counts(cls) -> list[tuple[Path, int, bool, bool]]:
        return [
            (
                SLIDE_FOLDER.joinpath(wsi_definition["path"]),
                wsi_definition["levels"],
                wsi_definition["label"],
                wsi_definition["overview"],
            )
            for wsi_definition in cls.test_definitions.values()
        ]

    @classmethod
    def folders_and_instance_counts(cls) -> list[tuple[Path, int]]:
        return [
            (
                SLIDE_FOLDER.joinpath(wsi_definition["path"]),
                wsi_definition["instances"],
            )
            for wsi_definition in cls.test_definitions.values()
        ]

    @classmethod
    def wsi_names(cls, tiling: str | None = None) -> list[str]:
        return [
            key
            for key, value in cls.test_definitions.items()
            if tiling is None or value["tiled"] == tiling
        ]

    @classmethod
    def read_region(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_region", "level_transfer_syntax"
        )

    @classmethod
    def read_region_mm(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_region_mm", "level_transfer_syntax"
        )

    @classmethod
    def read_region_mpp(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_region_mpp", "level_transfer_syntax"
        )

    @classmethod
    def read_tile(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_tile", "level_transfer_syntax"
        )

    @classmethod
    def read_encoded_tile(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_encoded_tile", "level_transfer_syntax"
        )

    @classmethod
    def read_thumbnail(cls) -> list[tuple[str, UID, dict[str, Any]]]:
        return cls._get_test_values_and_transfer_syntax(
            "read_thumbnail", "level_transfer_syntax"
        )

    @classmethod
    def levels(cls) -> list[tuple[str, int]]:
        return cls._get_parameter("levels")

    @classmethod
    def label_hash(cls) -> list[tuple[str, UID, str | None]]:
        return cls._get_parameter_and_transfer_syntax("label", "label_transfer_syntax")

    @classmethod
    def overview_hash(cls) -> list[tuple[str, UID, str | None]]:
        return cls._get_parameter_and_transfer_syntax(
            "overview", "overview_transfer_syntax"
        )

    @classmethod
    def _get_test_values_and_transfer_syntax(
        cls, test_value_name: str, transfer_syntax_name: str
    ) -> list[tuple[str, UID, dict[str, Any]]]:
        return [
            (wsi_name, UID(wsi_definition[transfer_syntax_name]), region)
            for wsi_name, wsi_definition in cls.test_definitions.items()
            for region in wsi_definition[test_value_name]
        ]

    @classmethod
    def _get_parameter_and_transfer_syntax(
        cls, parameter_name: str, transfer_syntax_name: str
    ) -> list[tuple[str, UID, Any]]:
        return [
            (
                wsi_name,
                UID(wsi_definition[transfer_syntax_name]),
                wsi_definition[parameter_name],
            )
            for wsi_name, wsi_definition in cls.test_definitions.items()
        ]

    @classmethod
    def _get_parameter(cls, parameter_name: str) -> list[tuple[str, Any]]:
        return [
            (
                wsi_name,
                wsi_definition[parameter_name],
            )
            for wsi_name, wsi_definition in cls.test_definitions.items()
        ]


@pytest.fixture
def wsi_file(
    tmp_path: Path,
    main_dataset_with_pixel_data: Dataset,
    meta_dataset: FileMetaDataset,
) -> Path:
    test_file_path = tmp_path.joinpath("test_im.dcm")
    create_layer_file(test_file_path, main_dataset_with_pixel_data, meta_dataset)
    return test_file_path


@pytest.fixture
def wsi(wsi_file: Path) -> Iterator[WsiDicom]:
    with WsiDicom.open(wsi_file) as wsi:
        yield wsi


@pytest.fixture(scope="module")
def shared_threadpool_executor() -> Iterator[ThreadPoolExecutor]:
    """Provide a shared thread pool for exercising the supplied-executor read path.

    Module-scoped and requested by `wsi_factory` so it is torn down after the
    wsis that read through it are closed.
    """
    with ThreadPoolExecutor(
        max_workers=4, thread_name_prefix="TestSharedRead"
    ) as executor:
        yield executor


@pytest.fixture(scope="module")
def wsi_factory(
    shared_threadpool_executor: Executor,
) -> Iterator[Callable[..., WsiDicom]]:
    """Provide a callable that opens a WsiDicom from a wsi name and input type.

    Caches opened objects and closes them when the tests using the fixture are done.

    The callable also takes an optional `read_executor` so tests can exercise the
    supplied-executor read path; wsis opened with and without one are cached
    separately.
    """
    clients: list[SingleThreadDicomFileClient] = []
    wsis: dict[tuple[WsiInputType, Path, bool], WsiDicom] = {}

    def open_wsi(
        wsi_name: str,
        input_type: WsiInputType = WsiInputType.FILE,
        use_shared_executor: bool = False,
    ) -> WsiDicom:
        test_definition = WsiTestDefinitions.test_definitions[wsi_name]
        folder = Path(SLIDE_FOLDER).joinpath(test_definition["path"])
        key = (input_type, folder, use_shared_executor)
        if key in wsis:
            return wsis[key]
        read_executor = shared_threadpool_executor if use_shared_executor else None
        if not folder.exists():
            pytest.skip(f"Folder {folder} does not exist.")
        if input_type == WsiInputType.FILE:
            wsi = WsiDicom.open(folder, read_executor=read_executor)
        elif input_type == WsiInputType.WEB:
            file_client = SingleThreadDicomFileClient(
                f"file://{folder.absolute().as_posix()}"
            )
            clients.append(file_client)
            client = WsiDicomWebClient(file_client)
            wsi = WsiDicom.open_web(
                client,
                test_definition["study_instance_uid"],
                test_definition["series_instance_uid"],
                [JPEGBaseline8Bit, JPEG2000, ExplicitVRLittleEndian],
                read_executor=read_executor,
                # The active settings (a test's pinned downsampler, say) with one
                # web thread, not a fresh Settings that would reset them.
                settings=replace(get_settings(), open_web_threads=1),
            )
        else:
            raise NotImplementedError()
        wsis[key] = wsi
        return wsi

    yield open_wsi
    for wsi in wsis.values():
        wsi.close()
    for file_client in clients:
        file_client.shutdown()


@pytest.fixture
def read_executor() -> ReadExecutor:
    """Provide an inline (single-threaded) read executor for read methods under test."""
    return ReadExecutor(None, None)


@pytest.fixture(scope="module")
def tiled_size() -> Size:
    return Size(2, 2)


@pytest.fixture
def frame_count(tiled_size: Size) -> int:
    return tiled_size.area


@pytest.fixture
def rng() -> Random:
    SEED = 0
    return random.Random(SEED)


@pytest.fixture(scope="module")
def bits() -> int:
    return 8


@pytest.fixture(scope="module")
def samples_per_pixel() -> int:
    return 3


@pytest.fixture(scope="module")
def tile_size() -> Size:
    return Size(10, 10)


@pytest.fixture(scope="module")
def transfer_syntax() -> UID:
    return JPEGBaseline8Bit


@pytest.fixture
def frames(
    rng: random.Random,
    transfer_syntax: UID,
    frame_count: int,
    bits: int,
    samples_per_pixel: int,
    tile_size: Size,
) -> list[bytes]:
    if not transfer_syntax.is_encapsulated:
        min_frame_length = bits * tile_size.area * samples_per_pixel // 8
        max_frame_length = min_frame_length
    else:
        min_frame_length = 2
        max_frame_length = 100
    lengths = [
        rng.randint(min_frame_length, max_frame_length) for i in range(frame_count)
    ]
    return [
        rng.getrandbits(length * 8).to_bytes(length, sys.byteorder)
        for length in lengths
    ]


@pytest.fixture
def image_data(frames: list[bytes], tiled_size: Size) -> WsiDicomTestImageData:
    return WsiDicomTestImageData(frames, tiled_size)


@pytest.fixture(scope="module")
def uid_generator() -> CallableUidGenerator:
    return CallableUidGenerator()


@pytest.fixture
def tile_type() -> TileType:
    """Return the tile type of the test volume instance."""
    return TileType.SPARSE


@pytest.fixture
def offset_table() -> OffsetTableType:
    """Return the offset table of the pixel data of the test volume instance."""
    return OffsetTableType.BASIC


@pytest.fixture
def main_dataset(tile_type: TileType) -> Dataset:
    """Return the metadata of a test volume instance, without pixel data."""
    ds = Dataset()
    ds.ImageType = ["DERIVED", "PRIMARY", "VOLUME", "RESAMPLED"]
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.6"
    ds.SOPInstanceUID = generate_uid()
    ds.StudyDate = "20210210"
    ds.ContentDate = "20210210"
    ds.AcquisitionDateTime = ""
    ds.StudyTime = "142341.188035"
    ds.ContentTime = "142341.188035"
    ds.AccessionNumber = "Accession number"
    ds.Modality = "SM"
    ds.Manufacturer = "Manufacturer"
    ds.ReferringPhysicianName = ""
    ds.TimezoneOffsetFromUTC = "+0100"
    ds.ManufacturerModelName = "Model"
    ds.VolumetricProperties = "VOLUME"
    ds.PatientName = "Patient^Name"
    ds.PatientID = "Patient ID"
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    ds.DeviceSerialNumber = "Device serial number"
    ds.SoftwareVersions = ["?"]
    ds.AcquisitionDuration = 50.0
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyID = "Study ID"
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"
    ds.PatientOrientation = ""
    ds.FrameOfReferenceUID = generate_uid()
    ds.PositionReferenceIndicator = "SLIDE_CORNER"

    # Dimension Organization Sequence
    dimension_organization_sequence = Sequence()
    ds.DimensionOrganizationSequence = dimension_organization_sequence

    dimensional_organization_uid = generate_uid()
    # Dimension Organization Sequence: Dimension Organization 1
    dimension_organization1 = Dataset()
    dimension_organization1.DimensionOrganizationUID = dimensional_organization_uid
    dimension_organization_sequence.append(dimension_organization1)

    # Dimension Index Sequence
    dimension_index_sequence = Sequence()
    ds.DimensionIndexSequence = dimension_index_sequence

    # Dimension Index Sequence: Dimension Index 1
    dimension_index1 = Dataset()
    dimension_index1.DimensionOrganizationUID = dimensional_organization_uid
    dimension_index1.DimensionIndexPointer = (0x48, 0x21A)
    dimension_index_sequence.append(dimension_index1)

    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "YBR_FULL_422"
    ds.PlanarConfiguration = 0
    ds.NumberOfFrames = "1"
    ds.Rows = 1024
    ds.Columns = 1024
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.BurnedInAnnotation = "NO"
    ds.LossyImageCompression = "01"
    ds.LossyImageCompressionRatio = "100.0"
    ds.LossyImageCompressionMethod = "ISO_10918_1"
    ds.ContainerIdentifier = "?"

    # Issuer of the Container Identifier Sequence
    issuer_of_the_container_identifier_sequence = Sequence()
    ds.IssuerOfTheContainerIdentifierSequence = (
        issuer_of_the_container_identifier_sequence
    )

    # Container Type Code Sequence
    container_type_code_sequence = Sequence()
    ds.ContainerTypeCodeSequence = container_type_code_sequence

    # Acquisition Context Sequence
    acquisition_context_sequence = Sequence()
    ds.AcquisitionContextSequence = acquisition_context_sequence

    # Specimen Description Sequence
    specimen_description_sequence = Sequence()
    ds.SpecimenDescriptionSequence = specimen_description_sequence

    # Specimen Description Sequence: Specimen Description 1
    specimen_description1 = Dataset()
    specimen_description1.SpecimenIdentifier = "?"
    specimen_description1.SpecimenUID = generate_uid()

    # Issuer of the Specimen Identifier Sequence
    issuer_of_the_specimen_identifier_sequence = Sequence()
    specimen_description1.IssuerOfTheSpecimenIdentifierSequence = (
        issuer_of_the_specimen_identifier_sequence
    )

    # Specimen Preparation Sequence
    specimen_preparation_sequence = Sequence()
    specimen_description1.SpecimenPreparationSequence = specimen_preparation_sequence
    specimen_description_sequence.append(specimen_description1)

    ds.ImagedVolumeWidth = 20.0
    ds.ImagedVolumeHeight = 36.0
    ds.ImagedVolumeDepth = 0.60
    ds.TotalPixelMatrixColumns = 154
    ds.TotalPixelMatrixRows = 290

    # Total Pixel Matrix Origin Sequence
    total_pixel_matrix_origin_sequence = Sequence()
    ds.TotalPixelMatrixOriginSequence = total_pixel_matrix_origin_sequence

    # Total Pixel Matrix Origin Sequence: Total Pixel Matrix Origin 1
    total_pixel_matrix_origin1 = Dataset()
    total_pixel_matrix_origin1.XOffsetInSlideCoordinateSystem = "60.0"
    total_pixel_matrix_origin1.YOffsetInSlideCoordinateSystem = "10.0"
    total_pixel_matrix_origin_sequence.append(total_pixel_matrix_origin1)

    ds.SpecimenLabelInImage = "NO"
    ds.FocusMethod = "AUTO"
    ds.ExtendedDepthOfField = "NO"
    ds.ImageOrientationSlide = [-1, 0, 0, 0, -1, 0]

    # Optical Path Sequence
    optical_path_sequence = Sequence()
    ds.OpticalPathSequence = optical_path_sequence

    # Optical Path Sequence: Optical Path 1
    optical_path1 = Dataset()

    # Illumination Type Code Sequence
    illumination_type_code_sequence = Sequence()
    optical_path1.IlluminationTypeCodeSequence = illumination_type_code_sequence

    # Illumination Type Code Sequence: Illumination Type Code 1
    illumination_type_code1 = Dataset()
    illumination_type_code1.CodeValue = "111744"
    illumination_type_code1.CodingSchemeDesignator = "DCM"
    illumination_type_code1.CodeMeaning = "Brightfield illumination"
    illumination_type_code_sequence.append(illumination_type_code1)

    optical_path1.OpticalPathIdentifier = "0"

    # Illumination Color Code Sequence
    illumination_color_code_sequence = Sequence()
    optical_path1.IlluminationColorCodeSequence = illumination_color_code_sequence

    # Illumination Color Code Sequence: Illumination Color Code 1
    illumination_color_code1 = Dataset()
    illumination_color_code1.CodeValue = "R-102C0"
    illumination_color_code1.CodingSchemeDesignator = "SRT"
    illumination_color_code1.CodeMeaning = "Full Spectrum"
    illumination_color_code_sequence.append(illumination_color_code1)
    optical_path_sequence.append(optical_path1)

    ds.NumberOfOpticalPaths = 1
    ds.TotalPixelMatrixFocalPlanes = 1
    ds.LabelText = ""
    ds.BarcodeValue = ""

    # Per-frame Functional Groups Sequence
    per_frame_functional_groups_sequence = Sequence()
    ds.PerFrameFunctionalGroupsSequence = per_frame_functional_groups_sequence
    # Per-frame Functional Groups Sequence: Per-frame Functional Groups 1
    per_frame_functional_groups1 = Dataset()

    if tile_type == TileType.FULL:
        ds.DimensionOrganizationType = "TILED_FULL"
    else:
        # Per-frame Functional Groups Sequence: Per-frame Functional Groups 1
        per_frame_functional_groups1 = Dataset()

        # Plane Position (Slide) Sequence
        plane_position_slide_sequence = Sequence()
        per_frame_functional_groups1.PlanePositionSlideSequence = (
            plane_position_slide_sequence
        )

        # Plane Position (Slide) Sequence: Plane Position (Slide) 1
        plane_position_slide1 = Dataset()
        plane_position_slide1.XOffsetInSlideCoordinateSystem = "62.0"
        plane_position_slide1.YOffsetInSlideCoordinateSystem = "10.0"
        plane_position_slide1.ZOffsetInSlideCoordinateSystem = "0.0"
        plane_position_slide1.ColumnPositionInTotalImagePixelMatrix = 1
        plane_position_slide1.RowPositionInTotalImagePixelMatrix = 1
        plane_position_slide_sequence.append(plane_position_slide1)

    per_frame_functional_groups_sequence.append(per_frame_functional_groups1)
    # Shared Functional Groups Sequence
    shared_functional_groups_sequence = Sequence()
    ds.SharedFunctionalGroupsSequence = shared_functional_groups_sequence

    # Shared Functional Groups Sequence: Shared Functional Groups 1
    shared_functional_groups1 = Dataset()

    # Pixel Measures Sequence
    pixel_measures_sequence = Sequence()
    shared_functional_groups1.PixelMeasuresSequence = pixel_measures_sequence

    # Pixel Measures Sequence: Pixel Measures 1
    pixel_measures1 = Dataset()
    pixel_measures1.SliceThickness = "0.0006"
    pixel_measures1.SpacingBetweenSlices = "0.0006"
    pixel_measures1.PixelSpacing = [0.1242353, 0.1241778]
    pixel_measures_sequence.append(pixel_measures1)

    # Whole Slide Microscopy Image Frame Type Sequence
    whole_slide_microscopy_image_frame_type_sequence = Sequence()
    shared_functional_groups1.WholeSlideMicroscopyImageFrameTypeSequence = (
        whole_slide_microscopy_image_frame_type_sequence
    )

    # Whole Slide Microscopy Image Frame Type Sequence: Whole Slide Microscopy Image Frame Type 1
    whole_slide_microscopy_image_frame_type1 = Dataset()
    whole_slide_microscopy_image_frame_type1.FrameType = [
        "DERIVED",
        "PRIMARY",
        "VOLUME",
        "RESAMPLED",
    ]
    whole_slide_microscopy_image_frame_type_sequence.append(
        whole_slide_microscopy_image_frame_type1
    )

    # Optical Path Identification Sequence
    optical_path_identification_sequence = Sequence()
    shared_functional_groups1.OpticalPathIdentificationSequence = (
        optical_path_identification_sequence
    )

    # Optical Path Identification Sequence: Optical Path Identification 1
    optical_path_identification1 = Dataset()
    optical_path_identification1.OpticalPathIdentifier = "0"
    optical_path_identification_sequence.append(optical_path_identification1)
    shared_functional_groups_sequence.append(shared_functional_groups1)

    return ds


@pytest.fixture
def test_frame() -> bytes:
    """Return the frame of the test volume instance."""
    return TEST_FRAME_FILE.read_bytes()


@pytest.fixture
def main_dataset_with_pixel_data(
    main_dataset: Dataset, offset_table: OffsetTableType, test_frame: bytes
) -> Dataset:
    """Return the metadata of a test volume instance, with one encapsulated frame."""
    dataset = deepcopy(main_dataset)
    dataset.PixelData = encapsulate(
        [test_frame], has_bot=offset_table == OffsetTableType.BASIC
    )
    return dataset


@pytest.fixture
def meta_dataset() -> FileMetaDataset:
    """Return the file meta information of a test volume instance file."""
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 204
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.77.1.6")
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = UID("1.2.840.10008.1.2.4.50")
    file_meta.ImplementationClassUID = UID("1.2.826.0.1.3680043.8.498.1")
    file_meta.ImplementationVersionName = "PYDICOM 2.2.2"

    return file_meta
