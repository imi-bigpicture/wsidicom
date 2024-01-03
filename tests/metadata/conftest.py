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
from wsidicom.geometry import SizeMm
from wsidicom.metadata import (
    Equipment,
    ExtendedDepthOfField,
    FocusMethod,
    Image,
    ImageCoordinateSystem,
    ImagePathFilter,
    Label,
    LightPathFilter,
    Lut,
    Objectives,
    OpticalPath,
    Patient,
    PatientDeIdentification,
    PatientSex,
    Series,
    Slide,
    Study,
    WsiMetadata,
)
from wsidicom.metadata.optical_path import ConstantLutSegment, LinearLutSegment
from wsidicom.metadata.sample import (
    Collection,
    Embedding,
    ExtractedSpecimen,
    Fixation,
    Processing,
    Receiving,
    Sample,
    Sampling,
    SlideSample,
    SpecimenLocalization,
    SpecimenIdentifier,
    Staining,
    SamplingLocation,
    Storage,
)


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
def image(
    acquisition_datetime: Optional[datetime.datetime],
    focus_method: Optional[FocusMethod],
    extended_depth_of_field: Optional[ExtendedDepthOfField],
    image_coordinate_system: Optional[ImageCoordinateSystem],
    pixel_spacing: Optional[SizeMm],
    focal_plane_spacing: Optional[float],
    depth_of_field: Optional[float],
):
    yield Image(
        acquisition_datetime,
        focus_method,
        extended_depth_of_field,
        image_coordinate_system,
        pixel_spacing,
        focal_plane_spacing,
        depth_of_field,
    )


@pytest.fixture()
def label():
    yield Label("text", "barcode", True, True, False)


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


@pytest.fixture(
    params=[
        ["specimen description", "method"],
        [Code("value", "scheme", "meaning"), Code("value", "scheme", "meaning")],
    ]
)
def patient(request):
    species_description = request.param[0]
    assert isinstance(species_description, (str, Code))
    method = request.param[1]
    assert isinstance(method, (str, Code))
    patient_deidentification = PatientDeIdentification(True, [method])
    yield Patient(
        "name",
        "identifier",
        datetime.datetime(2023, 8, 5),
        PatientSex.O,
        species_description,
        patient_deidentification,
    )


@pytest.fixture()
def date_time():
    yield datetime.datetime(2023, 8, 5)


@pytest.fixture()
def description():
    yield "description"


@pytest.fixture()
def fixative():
    yield None


@pytest.fixture()
def medium():
    yield None


@pytest.fixture()
def embedding(date_time: datetime.datetime, description: str):
    yield Embedding(
        SpecimenEmbeddingMediaCode("Paraffin wax"),
        date_time,
        description,
    )


@pytest.fixture()
def fixation(date_time: datetime.datetime, description: str):
    yield Fixation(
        SpecimenFixativesCode("Neutral Buffered Formalin"),
        date_time,
        description,
    )


@pytest.fixture()
def processing_method():
    yield None


@pytest.fixture()
def collection(date_time: datetime.datetime, description: str):
    yield Collection(
        SpecimenCollectionProcedureCode("Excision"),
        date_time,
        description,
    )


@pytest.fixture()
def sampling(
    extracted_specimen: ExtractedSpecimen,
    date_time: datetime.datetime,
    description: str,
    location: SamplingLocation,
):
    yield Sampling(
        extracted_specimen,
        SpecimenSamplingProcedureCode("Dissection"),
        [],
        date_time=date_time,
        description=description,
        location=location,
    )


@pytest.fixture()
def processing(date_time: datetime.datetime, description: str):
    yield Processing(
        SpecimenPreparationStepsCode("Specimen clearing"),
        date_time,
        description,
    )


@pytest.fixture()
def staining(date_time: datetime.datetime, description: str):
    yield Staining(
        [
            SpecimenStainsCode("hematoxylin stain"),
            SpecimenStainsCode("water soluble eosin stain"),
        ],
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
def identifier():
    yield "identifier"


@pytest.fixture()
def extracted_specimen(
    collection: Collection, identifier: Union[str, SpecimenIdentifier]
):
    yield ExtractedSpecimen(
        identifier,
        AnatomicPathologySpecimenTypesCode("Gross specimen"),
        collection,
        container=ContainerTypeCode("Specimen container"),
    )


@pytest.fixture()
def sample(extracted_specimen: ExtractedSpecimen):
    processing = Processing(
        SpecimenPreparationStepsCode("Specimen clearing"),
        datetime.datetime(2023, 8, 5),
    )
    yield Sample(
        "sample",
        AnatomicPathologySpecimenTypesCode("Tissue section"),
        [
            extracted_specimen.sample(
                SpecimenSamplingProcedureCode("Dissection"),
                datetime.datetime(2023, 8, 5),
                "Sampling to block",
            ),
        ],
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
        localization=SpecimenLocalization(description="left"),
    )


@pytest.fixture()
def slide():
    part_1 = ExtractedSpecimen(
        "part 1",
        AnatomicPathologySpecimenTypesCode("tissue specimen"),
        Collection(
            SpecimenCollectionProcedureCode("Specimen collection"),
            datetime.datetime(2023, 8, 5),
            "Extracted",
        ),
        [
            Fixation(
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                datetime.datetime(2023, 8, 5),
            )
        ],
    )

    part_2 = ExtractedSpecimen(
        "part 2",
        AnatomicPathologySpecimenTypesCode("tissue specimen"),
        Collection(
            SpecimenCollectionProcedureCode("Specimen collection"),
            datetime.datetime(2023, 8, 5),
            "Extracted",
        ),
        [
            Fixation(
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                datetime.datetime(2023, 8, 5),
            )
        ],
    )

    block = Sample(
        "block 1",
        AnatomicPathologySpecimenTypesCode("tissue specimen"),
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
        SpecimenLocalization(description="left"),
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
        SpecimenLocalization(description="right"),
    )

    stainings = [
        Staining(
            [
                SpecimenStainsCode("hematoxylin stain"),
                SpecimenStainsCode("water soluble eosin stain"),
            ],
            date_time=datetime.datetime(2023, 8, 5),
        ),
    ]

    yield Slide(identifier="Slide 1", stainings=stainings, samples=[sample_1, sample_2])


@pytest.fixture()
def series():
    yield Series(
        UID("1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445423"), 1
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
    )


@pytest.fixture()
def wsi_metadata(
    study: Study,
    series: Series,
    patient: Patient,
    equipment: Equipment,
    optical_path: OpticalPath,
    slide: Slide,
    label: Label,
    image: Image,
):
    yield WsiMetadata(
        study=study,
        series=series,
        patient=patient,
        equipment=equipment,
        optical_paths=[optical_path],
        slide=slide,
        label=label,
        image=image,
    )