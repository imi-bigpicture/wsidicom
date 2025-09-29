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

import datetime
from typing import Optional, Sequence, Union

import numpy as np
import pytest
from pydicom.uid import UID

from wsidicom.codec import LossyCompressionIsoStandard
from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    Code,
    ContainerTypeCode,
    IlluminationCode,
    IlluminationColorCode,
    ImagePathFilterCode,
    LenseCode,
    LightPathFilterCode,
    SpecimenCollectionProcedureCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenPreparationStepsCode,
    SpecimenSamplingProcedureCode,
    SpecimenStainsCode,
)
from wsidicom.geometry import PointMm, SizeMm
from wsidicom.instance.dataset import ImageType
from wsidicom.metadata import (
    Collection,
    ConstantLutSegment,
    Embedding,
    Equipment,
    ExtendedDepthOfField,
    Fixation,
    FocusMethod,
    Image,
    ImageCoordinateSystem,
    ImagePathFilter,
    Label,
    LightPathFilter,
    LinearLutSegment,
    LossyCompression,
    Lut,
    Objectives,
    OpticalPath,
    Overview,
    Patient,
    PatientDeIdentification,
    PatientSex,
    Processing,
    Pyramid,
    Receiving,
    Sample,
    SampleLocalization,
    Sampling,
    SamplingLocation,
    Series,
    Slide,
    SlideSample,
    Specimen,
    SpecimenIdentifier,
    Staining,
    Storage,
    Study,
    WsiMetadata,
)


@pytest.fixture()
def slide_identifier():
    yield "slide identifier"


@pytest.fixture
def manufacturer():
    yield "manufacturer"


@pytest.fixture
def model_name():
    yield "model name"


@pytest.fixture
def serial_number():
    yield "serial number"


@pytest.fixture
def versions():
    yield ["versions"]


@pytest.fixture()
def equipment(
    manufacturer: Optional[str],
    model_name: Optional[str],
    serial_number: Optional[str],
    versions: Optional[Sequence[str]],
):
    yield Equipment(
        manufacturer,
        model_name,
        serial_number,
        versions,
    )


@pytest.fixture()
def lossy_compressions():
    yield [LossyCompression(LossyCompressionIsoStandard.JPEG_LOSSY, 0.5)]


@pytest.fixture()
def image_comments():
    yield "comments"


@pytest.fixture()
def image_contains_phi():
    yield False


@pytest.fixture()
def image(
    acquisition_datetime: Optional[datetime.datetime],
    focus_method: Optional[FocusMethod],
    extended_depth_of_field: Optional[ExtendedDepthOfField],
    image_coordinate_system: Optional[ImageCoordinateSystem],
    pixel_spacing: Optional[SizeMm],
    focal_plane_spacing: Optional[float],
    depth_of_field: Optional[float],
    lossy_compressions: Optional[Sequence[LossyCompression]],
):
    yield Image(
        acquisition_datetime,
        focus_method,
        extended_depth_of_field,
        image_coordinate_system,
        pixel_spacing,
        focal_plane_spacing,
        depth_of_field,
        lossy_compressions,
    )


@pytest.fixture()
def pyramid_uid():
    yield UID("1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445423")


@pytest.fixture()
def pyramid_description():
    yield "pyramid description"


@pytest.fixture()
def pyramid_label():
    yield "pyramid label"


@pytest.fixture()
def pyramid(
    image: Image,
    optical_path: OpticalPath,
    pyramid_uid: UID,
    pyramid_description: str,
    pyramid_label: str,
    image_contains_phi: bool,
    image_comments: Optional[str],
):
    yield Pyramid(
        image,
        [optical_path],
        pyramid_uid,
        pyramid_description,
        pyramid_label,
        image_contains_phi,
        image_comments,
    )


@pytest.fixture()
def label_text():
    yield "text"


@pytest.fixture()
def label_barcode():
    yield "barcode"


@pytest.fixture()
def label_contains_phi():
    yield True


@pytest.fixture()
def label_pixel_spacing():
    yield None


@pytest.fixture()
def label_lossy_compressions():
    yield [LossyCompression(LossyCompressionIsoStandard.JPEG_LOSSY, 2)]


@pytest.fixture()
def label_image(
    label_image_coordinate_system: Optional[ImageCoordinateSystem],
    label_pixel_spacing: Optional[SizeMm],
    label_lossy_compressions: Optional[Sequence[LossyCompression]],
):
    yield Image(
        None,
        None,
        None,
        label_image_coordinate_system,
        label_pixel_spacing,
        None,
        None,
        label_lossy_compressions,
    )


@pytest.fixture()
def label_optical_path(
    icc_profile: bytes,
):
    yield OpticalPath(
        "identifier",
        "description",
        [IlluminationCode("Brightfield illumination")],
        icc_profile=icc_profile,
    )


@pytest.fixture()
def label_image_comments():
    yield "comments"


@pytest.fixture()
def label(
    label_text: Optional[str],
    label_barcode: Optional[str],
    label_contains_phi: bool,
    label_image: Image,
    label_optical_path: OpticalPath,
    label_image_comments: Optional[str],
):
    yield Label(
        label_text,
        label_barcode,
        label_image,
        [label_optical_path],
        label_contains_phi,
        label_image_comments,
    )


@pytest.fixture()
def overview_contains_phi():
    yield True


@pytest.fixture()
def overview_pixel_spacing():
    yield None


@pytest.fixture()
def overview_lossy_compressions():
    yield [LossyCompression(LossyCompressionIsoStandard.JPEG_LOSSY, 2)]


@pytest.fixture()
def overview_image_contains_slide_label():
    yield True


@pytest.fixture()
def overview_image(
    overview_image_coordinate_system: Optional[ImageCoordinateSystem],
    overview_pixel_spacing: Optional[SizeMm],
    overview_lossy_compressions: Optional[Sequence[LossyCompression]],
):
    yield Image(
        None,
        None,
        None,
        overview_image_coordinate_system,
        overview_pixel_spacing,
        None,
        None,
        overview_lossy_compressions,
    )


@pytest.fixture()
def overview_optical_path(
    icc_profile: bytes,
):
    yield OpticalPath(
        "identifier",
        "description",
        [IlluminationCode("Brightfield illumination")],
        icc_profile=icc_profile,
    )


@pytest.fixture()
def overview_image_comments():
    yield "comments"


@pytest.fixture()
def overview(
    overview_contains_phi: bool,
    overview_image_contains_slide_label: bool,
    overview_image: Image,
    overview_optical_path: OpticalPath,
    overview_image_comments: Optional[str],
):
    yield Overview(
        overview_image,
        [overview_optical_path],
        overview_image_contains_slide_label,
        overview_contains_phi,
        overview_image_comments,
    )


@pytest.fixture()
def light_path_filter():
    yield LightPathFilter(
        [LightPathFilterCode("Green optical filter")],
        500,
        400,
        600,
    )


@pytest.fixture()
def image_path_filter():
    yield ImagePathFilter(
        [
            ImagePathFilterCode("Red optical filter"),
        ],
        500,
        400,
        600,
    )


@pytest.fixture()
def objectives():
    yield Objectives([LenseCode("High power non-immersion lens")], 10.0, 20.0, 0.5)


@pytest.fixture()
def lut():
    yield Lut(
        [ConstantLutSegment(0, 256)],
        [ConstantLutSegment(0, 256)],
        [LinearLutSegment(0, 255, 256)],
        np.uint16,
    )


@pytest.fixture()
def icc_profile():
    yield bytes([0x00, 0x01, 0x02, 0x03])


@pytest.fixture()
def optical_path(
    illumination: Union[IlluminationColorCode, float],
    light_path_filter: LightPathFilter,
    image_path_filter: ImagePathFilter,
    objectives: Objectives,
    lut: Lut,
    icc_profile: bytes,
):
    yield OpticalPath(
        "identifier",
        "description",
        [IlluminationCode("Brightfield illumination")],
        illumination,
        icc_profile,
        lut,
        light_path_filter,
        image_path_filter,
        objectives,
    )


@pytest.fixture
def species_description():
    yield Code("value", "scheme", "meaning")


@pytest.fixture
def patient_deidentification_method():
    yield Code("value", "scheme", "meaning")


@pytest.fixture()
def patient(
    species_description: Union[str, Code],
    patient_deidentification_method: Optional[Union[str, Code]],
):

    yield Patient(
        "name",
        "identifier",
        datetime.date(2023, 8, 5),
        PatientSex.O,
        species_description,
        (
            PatientDeIdentification(True, [patient_deidentification_method])
            if patient_deidentification_method
            else None
        ),
        "comments",
    )


@pytest.fixture()
def date_time():
    yield datetime.datetime(2023, 8, 5)


@pytest.fixture()
def description():
    yield "description"


@pytest.fixture()
def medium():
    yield SpecimenEmbeddingMediaCode("Paraffin wax")


@pytest.fixture()
def fixative():
    yield SpecimenFixativesCode("Neutral Buffered Formalin")


@pytest.fixture()
def collection_method():
    yield SpecimenCollectionProcedureCode("Specimen collection")


@pytest.fixture()
def sampling_method():
    yield SpecimenSamplingProcedureCode("Dissection")


@pytest.fixture()
def processing_method():
    yield SpecimenPreparationStepsCode("Specimen clearing")


@pytest.fixture()
def embedding(
    date_time: datetime.datetime, description: str, medium: SpecimenEmbeddingMediaCode
):
    yield Embedding(
        medium,
        date_time,
        description,
    )


@pytest.fixture()
def fixation(
    date_time: datetime.datetime, description: str, fixative: SpecimenFixativesCode
):
    yield Fixation(
        fixative,
        date_time,
        description,
    )


@pytest.fixture()
def identifier():
    yield "identifier"


@pytest.fixture()
def collection(
    date_time: datetime.datetime,
    description: str,
    collection_method: SpecimenCollectionProcedureCode,
):
    yield Collection(
        collection_method,
        date_time,
        description,
    )


@pytest.fixture()
def sampling(
    extracted_specimen: Specimen,
    date_time: datetime.datetime,
    description: str,
    location: SamplingLocation,
    sampling_method: SpecimenSamplingProcedureCode,
):
    assert extracted_specimen.type is not None
    yield Sampling(
        extracted_specimen,
        sampling_method,
        extracted_specimen.type,
        [],
        date_time=date_time,
        description=description,
        location=location,
    )


@pytest.fixture()
def processing(
    date_time: datetime.datetime,
    description: str,
    processing_method: SpecimenPreparationStepsCode,
):
    yield Processing(
        processing_method,
        date_time,
        description,
    )


@pytest.fixture()
def substances():
    yield [
        SpecimenStainsCode("hematoxylin stain"),
        SpecimenStainsCode("water soluble eosin stain"),
    ]


@pytest.fixture()
def staining(
    substances: Union[str, Sequence[SpecimenStainsCode]],
    date_time: datetime.datetime,
    description: str,
):
    yield Staining(
        substances,
        date_time=date_time,
        description=description,
    )


@pytest.fixture()
def storage(date_time: datetime.datetime, description: str):
    yield Storage(
        date_time=date_time,
        description=description,
    )


@pytest.fixture()
def receiving(date_time: datetime.datetime, description: str):
    yield Receiving(
        date_time=date_time,
        description=description,
    )


@pytest.fixture()
def extracted_specimen(
    collection: Collection, identifier: Union[str, SpecimenIdentifier]
):
    yield Specimen(
        identifier,
        collection,
        AnatomicPathologySpecimenTypesCode("Gross specimen"),
        container=ContainerTypeCode("Specimen container"),
    )


@pytest.fixture()
def sample(extracted_specimen: Specimen):
    processing = Processing(
        SpecimenPreparationStepsCode("Specimen clearing"),
        datetime.datetime(2023, 8, 5),
    )
    yield Sample(
        "sample",
        [
            extracted_specimen.sample(
                SpecimenSamplingProcedureCode("Dissection"),
                datetime.datetime(2023, 8, 5),
                "Sampling to block",
            ),
        ],
        AnatomicPathologySpecimenTypesCode("Tissue section"),
        [processing],
        container=ContainerTypeCode("Tissue cassette"),
    )


@pytest.fixture()
def slide_sample(sample: Sample):
    yield SlideSample(
        "slide sample",
        [Code("value", "scheme", "meaning")],
        sample.sample(
            SpecimenSamplingProcedureCode("Block sectioning"),
            datetime.datetime(2023, 8, 5),
            "Sectioning to slide",
        ),
        uid=UID("1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445423"),
        localization=SampleLocalization(description="left"),
    )


@pytest.fixture()
def slide(slide_identifier: Union[str, SpecimenIdentifier], staining: Staining):
    part_1 = Specimen(
        "part 1",
        Collection(
            SpecimenCollectionProcedureCode("Specimen collection"),
            datetime.datetime(2023, 8, 5),
            "Extracted",
        ),
        AnatomicPathologySpecimenTypesCode("tissue specimen"),
        [
            Fixation(
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                datetime.datetime(2023, 8, 5),
            )
        ],
    )

    part_2 = Specimen(
        "part 2",
        Collection(
            SpecimenCollectionProcedureCode("Specimen collection"),
            datetime.datetime(2023, 8, 5),
            "Extracted",
        ),
        AnatomicPathologySpecimenTypesCode("tissue specimen"),
        [
            Fixation(
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                datetime.datetime(2023, 8, 5),
            )
        ],
    )

    block = Sample(
        "block 1",
        [
            part_1.sample(
                SpecimenSamplingProcedureCode("Dissection"),
                datetime.datetime(2023, 8, 5),
                "Sampling to block",
            ),
            part_2.sample(
                SpecimenSamplingProcedureCode("Dissection"),
                datetime.datetime(2023, 8, 5),
                "Sampling to block",
            ),
        ],
        AnatomicPathologySpecimenTypesCode("tissue specimen"),
        [
            Embedding(
                SpecimenEmbeddingMediaCode("Paraffin wax"),
                datetime.datetime(2023, 8, 5),
            )
        ],
    )

    sample_1 = SlideSample(
        "Sample 1",
        [Code("value", "schema", "meaning")],
        block.sample(
            SpecimenSamplingProcedureCode("Block sectioning"),
            datetime.datetime(2023, 8, 5),
            "Sampling to slide",
            [part_1.samplings[0]],
        ),
        UID("1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445423"),
        SampleLocalization(description="left"),
    )

    sample_2 = SlideSample(
        "Sample 2",
        [Code("value", "schema", "meaning")],
        block.sample(
            SpecimenSamplingProcedureCode("Block sectioning"),
            datetime.datetime(2023, 8, 5),
            "Sampling to slide",
            [part_2.samplings[0]],
        ),
        UID("1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445424"),
        SampleLocalization(description="right"),
    )

    yield Slide(
        identifier=slide_identifier, stainings=[staining], samples=[sample_1, sample_2]
    )


@pytest.fixture()
def series():
    yield Series(
        UID("1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445423"),
        1,
        "Description",
        "SKIN",
    )


@pytest.fixture()
def study():
    yield Study(
        UID("1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445423"),
        "identifier",
        datetime.date(2023, 8, 5),
        datetime.time(12, 3),
        "accession number",
        "referring physician name",
        "description",
    )


@pytest.fixture()
def illumination():
    yield IlluminationColorCode("Full Spectrum")


@pytest.fixture()
def acquisition_datetime():
    yield datetime.datetime(2023, 8, 5)


@pytest.fixture()
def focus_method():
    yield FocusMethod.AUTO


@pytest.fixture()
def extended_depth_of_field():
    yield ExtendedDepthOfField(5, 0.5)


@pytest.fixture()
def image_coordinate_system():
    yield ImageCoordinateSystem(PointMm(20.0, 30.0), 180)


@pytest.fixture()
def label_image_coordinate_system():
    yield ImageCoordinateSystem(PointMm(25, 75), 180)


@pytest.fixture()
def overview_image_coordinate_system():
    yield ImageCoordinateSystem(PointMm(25, 75), 180)


@pytest.fixture()
def pixel_spacing():
    yield None


@pytest.fixture()
def focal_plane_spacing():
    yield None


@pytest.fixture()
def depth_of_field():
    yield None


@pytest.fixture()
def image_type():
    yield ImageType.VOLUME


@pytest.fixture()
def wsi_metadata(
    study: Study,
    series: Series,
    patient: Patient,
    equipment: Equipment,
    slide: Slide,
    pyramid: Pyramid,
    label: Label,
    overview: Overview,
):
    yield WsiMetadata(
        study=study,
        series=series,
        patient=patient,
        equipment=equipment,
        slide=slide,
        pyramid=pyramid,
        label=label,
        overview=overview,
    )
