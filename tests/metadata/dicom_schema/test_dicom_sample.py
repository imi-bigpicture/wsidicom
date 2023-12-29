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
from typing import Iterator, List, Optional, Sequence, Union
from pydicom import Dataset
from pydicom.valuerep import DSfloat
import pytest

from pydicom.uid import UID
from pydicom.sr.coding import Code
from wsidicom.conceptcode import (
    ConceptCode,
    ContainerTypeCode,
    SpecimenCollectionProcedureCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenPreparationStepsCode,
    SpecimenStainsCode,
    SpecimenSamplingProcedureCode,
    AnatomicPathologySpecimenTypesCode,
    UnitCode,
)
from wsidicom.metadata.dicom_schema.sample.model import (
    CollectionDicomModel,
    SpecimenPreparationStepDicomModel,
    ProcessingDicomModel,
    StainingDicomModel,
    SamplingDicomModel,
    ReceivingDicomModel,
    StorageDicomModel,
)
from wsidicom.metadata.dicom_schema.sample.parser import SpecimenDicomParser
from wsidicom.metadata.dicom_schema.sample.schema import (
    CollectionDicomSchema,
    ProcessingDicomSchema,
    ReceivingDicomSchema,
    SamplingDicomSchema,
    SpecimenDescriptionDicomSchema,
    SpecimenLocalizationDicomSchema,
    StainingDicomSchema,
    SampleCodes,
    StorageDicomSchema,
)

from wsidicom.metadata.sample import (
    Collection,
    Embedding,
    ExtractedSpecimen,
    Fixation,
    LocalIssuerOfIdentifier,
    Measurement,
    Sample,
    SamplingLocation,
    SlideSample,
    SpecimenLocalization,
    SpecimenIdentifier,
    Sampling,
    Processing,
    Staining,
    Receiving,
    Storage,
)


@pytest.fixture()
def specimen_id():
    yield "specimen"


@pytest.fixture()
def slide_sample_id():
    yield "slide sample"


@pytest.fixture()
def slide_sample_uid():
    yield UID("1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445424")


@pytest.fixture()
def specimen_type():
    yield AnatomicPathologySpecimenTypesCode("tissue specimen")


@pytest.fixture()
def collection_method():
    yield SpecimenCollectionProcedureCode("Excision")


@pytest.fixture()
def specimen_sampling_method():
    yield SpecimenSamplingProcedureCode("Dissection")


@pytest.fixture()
def block_id():
    yield "block"


@pytest.fixture()
def block_sampling_method():
    yield SpecimenSamplingProcedureCode("Block sectioning")


@pytest.fixture()
def block_type():
    yield AnatomicPathologySpecimenTypesCode("tissue specimen")


@pytest.fixture()
def specimen_localization():
    yield SpecimenLocalization(description="left")


@pytest.fixture()
def sampling_location():
    yield SamplingLocation(description="left")


@pytest.fixture()
def primary_anatomic_structures():
    yield [Code("value", "schema", "meaning")]


@pytest.fixture()
def container():
    yield ContainerTypeCode("Specimen container")


@pytest.fixture()
def stains():
    yield [
        SpecimenStainsCode("hematoxylin stain"),
        SpecimenStainsCode("water soluble eosin stain"),
    ]


@pytest.fixture()
def collection_dicom(
    collection: Collection,
    identifier: Union[str, SpecimenIdentifier],
    fixative: Optional[SpecimenFixativesCode],
    medium: Optional[SpecimenEmbeddingMediaCode],
    processing_method: Optional[SpecimenPreparationStepsCode],
    container: Optional[ContainerTypeCode],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    yield CollectionDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=collection.date_time,
        description=collection.description,
        fixative=fixative,
        embedding=medium,
        method=collection.method,
        processing=processing_method,
        container=container,
    )


def create_collection_dataset(
    collection_dicom: CollectionDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    if collection_dicom.container is not None:
        items.append(
            create_code_item(SampleCodes.container, collection_dicom.container)
        )
    items.append(create_processing_type_item(collection_dicom))
    if collection_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, collection_dicom.date_time
            ),
        )
    if collection_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, collection_dicom.description
            )
        )
    items.append(
        create_code_item(SampleCodes.specimen_collection, collection_dicom.method)
    )
    if collection_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, collection_dicom.fixative))
    if collection_dicom.embedding is not None:
        items.append(
            create_code_item(SampleCodes.embedding, collection_dicom.embedding)
        )
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


@pytest.fixture()
def collection_dataset(
    collection_dicom: CollectionDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_collection_dataset(collection_dicom, identifier)


@pytest.fixture()
def sampling_dicom(
    sampling: Sampling,
    identifier: Union[str, SpecimenIdentifier],
    fixative: Optional[SpecimenFixativesCode],
    medium: Optional[SpecimenEmbeddingMediaCode],
    processing_method: Optional[SpecimenPreparationStepsCode],
    container: Optional[ContainerTypeCode],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    (
        parent_identifier,
        parent_issuer,
    ) = SpecimenIdentifier.get_string_identifier_and_issuer(
        sampling.specimen.identifier
    )
    yield SamplingDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=sampling.date_time,
        description=sampling.description,
        method=sampling.method,
        fixative=fixative,
        embedding=medium,
        parent_specimen_identifier=parent_identifier,
        issuer_of_parent_specimen_identifier=parent_issuer,
        parent_specimen_type=sampling.specimen.type,
        processing=processing_method,
        location_reference=sampling.location.reference
        if sampling.location is not None
        else None,
        location_description=sampling.location.description
        if sampling.location is not None
        else None,
        location_x=sampling.location.x if sampling.location is not None else None,
        location_y=sampling.location.y if sampling.location is not None else None,
        location_z=sampling.location.z if sampling.location is not None else None,
        container=container,
    )


def create_sampling_dataset(
    sampling_dicom: SamplingDicomModel, identifier: Union[str, SpecimenIdentifier]
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    if sampling_dicom.container is not None:
        items.append(create_code_item(SampleCodes.container, sampling_dicom.container))
    items.append(create_processing_type_item(sampling_dicom))
    if sampling_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, sampling_dicom.date_time
            ),
        )
    if sampling_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, sampling_dicom.description
            )
        )
    items.append(create_code_item(SampleCodes.sampling_method, sampling_dicom.method))
    items.append(
        create_string_item(
            SampleCodes.parent_specimen_identifier,
            sampling_dicom.parent_specimen_identifier,
        )
    )
    if sampling_dicom.issuer_of_parent_specimen_identifier is not None:
        items.append(
            create_string_item(
                SampleCodes.issuer_of_parent_specimen_identifier,
                sampling_dicom.issuer_of_parent_specimen_identifier,
            )
        )
    items.append(
        create_code_item(
            SampleCodes.parent_specimen_type, sampling_dicom.parent_specimen_type
        )
    )
    if sampling_dicom.location_reference is not None:
        items.append(
            create_string_item(
                SampleCodes.location_frame_of_reference,
                sampling_dicom.location_reference,
            )
        )
    if sampling_dicom.location_description is not None:
        items.append(
            create_string_item(
                SampleCodes.location_of_sampling_site,
                sampling_dicom.location_description,
            )
        )
    if sampling_dicom.location_x is not None:
        items.append(
            create_measurement_item(
                SampleCodes.location_of_sampling_site_x, sampling_dicom.location_x
            )
        )
    if sampling_dicom.location_y is not None:
        items.append(
            create_measurement_item(
                SampleCodes.location_of_sampling_site_y, sampling_dicom.location_y
            )
        )
    if sampling_dicom.location_z is not None:
        items.append(
            create_measurement_item(
                SampleCodes.location_of_sampling_site_z, sampling_dicom.location_z
            )
        )
    if sampling_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, sampling_dicom.fixative))
    if sampling_dicom.embedding is not None:
        items.append(create_code_item(SampleCodes.embedding, sampling_dicom.embedding))
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


@pytest.fixture()
def sampling_dataset(
    sampling_dicom: SamplingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_sampling_dataset(sampling_dicom, identifier)


@pytest.fixture()
def processing_dicom(
    processing: Processing,
    identifier: Union[str, SpecimenIdentifier],
    fixative: Optional[SpecimenFixativesCode],
    medium: Optional[SpecimenEmbeddingMediaCode],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    yield ProcessingDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=processing.date_time,
        description=processing.description,
        processing=processing.method,
        fixative=fixative,
        embedding=medium,
    )


def create_processing_dataset(
    processing_dicom: ProcessingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(processing_dicom))
    if processing_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, processing_dicom.date_time
            ),
        )
    if processing_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, processing_dicom.description
            )
        )
    if processing_dicom.processing is not None:
        items.append(
            create_code_item(
                SampleCodes.processing_description, processing_dicom.processing
            )
        )
    if processing_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, processing_dicom.fixative))
    if processing_dicom.embedding is not None:
        items.append(
            create_code_item(SampleCodes.embedding, processing_dicom.embedding)
        )
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


@pytest.fixture()
def processing_dataset(
    processing_dicom: ProcessingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_processing_dataset(processing_dicom, identifier)


@pytest.fixture()
def staining_dicom(
    staining: Staining,
    identifier: Union[str, SpecimenIdentifier],
    fixative: Optional[SpecimenFixativesCode],
    medium: Optional[SpecimenEmbeddingMediaCode],
    processing_method: Optional[SpecimenPreparationStepsCode],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    yield StainingDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=staining.date_time,
        description=staining.description,
        substances=staining.substances,
        fixative=fixative,
        embedding=medium,
        processing=processing_method,
    )


def create_staining_dataset(
    staining_dicom: StainingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(staining_dicom))
    if staining_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, staining_dicom.date_time
            ),
        )
    if staining_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, staining_dicom.description
            )
        )
    for substance in staining_dicom.substances:
        items.append(
            create_code_item(SampleCodes.using_substance, substance)
            if isinstance(substance, SpecimenStainsCode)
            else create_string_item(SampleCodes.using_substance, substance)
        )
    if staining_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, staining_dicom.fixative))
    if staining_dicom.embedding is not None:
        items.append(create_code_item(SampleCodes.embedding, staining_dicom.embedding))
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


@pytest.fixture()
def staining_dataset(
    staining_dicom: StainingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_staining_dataset(staining_dicom, identifier)


@pytest.fixture()
def receiving_dicom(
    receiving: Receiving,
    identifier: Union[str, SpecimenIdentifier],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    yield ReceivingDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=receiving.date_time,
        description=receiving.description,
        fixative=None,
        embedding=None,
        processing=None,
    )


def create_receiving_dataset(
    receiving_dicom: ReceivingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(receiving_dicom))
    if receiving_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, receiving_dicom.date_time
            ),
        )
    if receiving_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, receiving_dicom.description
            )
        )
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


@pytest.fixture()
def receiving_dataset(
    receiving_dicom: ReceivingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_receiving_dataset(receiving_dicom, identifier)


@pytest.fixture()
def storage_dicom(
    storage: Storage,
    identifier: Union[str, SpecimenIdentifier],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    yield StorageDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=storage.date_time,
        description=storage.description,
        fixative=None,
        embedding=None,
        processing=None,
    )


def create_storage_dataset(
    storage_dicom: StorageDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(storage_dicom))
    if storage_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, storage_dicom.date_time
            ),
        )
    if storage_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, storage_dicom.description
            )
        )
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


@pytest.fixture()
def storage_dataset(
    storage_dicom: StorageDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_storage_dataset(storage_dicom, identifier)


def create_description(
    slide_sample_id: str,
    slide_sample_uid: UID,
    specimen_id: str,
    block_id: str,
    specimen_type: AnatomicPathologySpecimenTypesCode,
    collection_method: SpecimenCollectionProcedureCode,
    fixative: SpecimenFixativesCode,
    specimen_sampling_method: SpecimenSamplingProcedureCode,
    embedding_medium: SpecimenEmbeddingMediaCode,
    block_sampling_method: SpecimenSamplingProcedureCode,
    block_type: AnatomicPathologySpecimenTypesCode,
    sample_localization: SpecimenLocalization,
    sampling_location: SamplingLocation,
    primary_anatomic_structures: Sequence[Code],
    stains: Sequence[SpecimenStainsCode],
    short_description: Optional[str] = None,
    detailed_description: Optional[str] = None,
    specimen_container: Optional[ContainerTypeCode] = None,
    block_container: Optional[ContainerTypeCode] = None,
):
    collection = create_collection_dataset(
        CollectionDicomModel(
            identifier=specimen_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=None,
            method=collection_method,
            processing=None,
            container=specimen_container,
        ),
        identifier=specimen_id,
    )
    fixation = create_processing_dataset(
        ProcessingDicomModel(
            identifier=specimen_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=fixative,
            embedding=None,
            processing=None,
        ),
        identifier=specimen_id,
    )
    sampling_to_block = create_sampling_dataset(
        SamplingDicomModel(
            identifier=specimen_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=None,
            method=specimen_sampling_method,
            parent_specimen_identifier=specimen_id,
            issuer_of_parent_specimen_identifier=None,
            parent_specimen_type=specimen_type,
            processing=None,
            location_reference=sampling_location.reference
            if sampling_location is not None
            else None,
            location_description=sampling_location.description
            if sampling_location is not None
            else None,
            location_x=sampling_location.x if sampling_location is not None else None,
            location_y=sampling_location.y if sampling_location is not None else None,
            location_z=sampling_location.z if sampling_location is not None else None,
            container=block_container,
        ),
        identifier=block_id,
    )
    embedding = create_processing_dataset(
        ProcessingDicomModel(
            identifier=block_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=embedding_medium,
            processing=None,
        ),
        identifier=block_id,
    )
    sampling_to_slide = create_sampling_dataset(
        SamplingDicomModel(
            identifier=block_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=None,
            method=block_sampling_method,
            parent_specimen_identifier=block_id,
            issuer_of_parent_specimen_identifier=None,
            parent_specimen_type=block_type,
            processing=None,
            location_reference=None,
            location_description=None,
            location_x=None,
            location_y=None,
            location_z=None,
        ),
        identifier=slide_sample_id,
    )
    staining = create_staining_dataset(
        StainingDicomModel(
            identifier=slide_sample_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            substances=list(stains),
            fixative=None,
            embedding=None,
            processing=None,
        ),
        identifier=slide_sample_id,
    )
    description = Dataset()
    description.SpecimenIdentifier = slide_sample_id
    description.SpecimenUID = slide_sample_uid
    description.SpecimenPreparationStepContentItemSequence = [
        collection,
        fixation,
        sampling_to_block,
        embedding,
        sampling_to_slide,
        staining,
    ]
    description.SpecimenTypeCodeSequence = [
        create_code_dataset(AnatomicPathologySpecimenTypesCode("Slide"))
    ]
    description.PrimaryAnatomicStructureSequence = [
        create_code_dataset(item) for item in primary_anatomic_structures
    ]
    description.SpecimenShortDescription = short_description
    description.SpecimenDetailedDescription = detailed_description
    if sample_localization is not None:
        sample_localization_sequence = []
        if sample_localization.reference is not None:
            sample_localization_sequence.append(
                create_string_item(
                    SampleCodes.location_frame_of_reference,
                    sample_localization.reference,
                )
            )
        if sample_localization.description is not None:
            sample_localization_sequence.append(
                create_string_item(
                    SampleCodes.location_of_specimen, sample_localization.description
                )
            )
        if sample_localization.x is not None:
            sample_localization_sequence.append(
                create_measurement_item(
                    SampleCodes.location_of_specimen_x, sample_localization.x
                )
            )
        if sample_localization.y is not None:
            sample_localization_sequence.append(
                create_measurement_item(
                    SampleCodes.location_of_specimen_x, sample_localization.y
                )
            )
        if sample_localization.y is not None:
            sample_localization_sequence.append(
                create_measurement_item(
                    SampleCodes.location_of_specimen_x, sample_localization.y
                )
            )
        if sample_localization.visual_marking is not None:
            sample_localization_sequence.append(
                create_string_item(
                    SampleCodes.visual_marking_of_specimen,
                    sample_localization.visual_marking,
                )
            )
        description.SpecimenLocalizationContentItemSequence = (
            sample_localization_sequence
        )
    return description


def create_identifier_items(identifier: Union[str, SpecimenIdentifier]):
    identifier_item = create_string_item(
        SampleCodes.identifier,
        identifier if isinstance(identifier, str) else identifier.value,
    )
    if not isinstance(identifier, SpecimenIdentifier) or identifier.issuer is None:
        return [identifier_item]
    issuer_item = create_string_item(
        SampleCodes.issuer_of_identifier,
        identifier.issuer.to_hl7v2(),
    )
    return [identifier_item, issuer_item]


def create_code_dataset(code: Union[Code, ConceptCode]):
    dataset = Dataset()
    dataset.CodeValue = code.value
    dataset.CodingSchemeDesignator = code.scheme_designator
    dataset.CodeMeaning = code.meaning
    dataset.CodingSchemeVersion = code.scheme_version
    return dataset


def create_string_item(name: Code, value: str):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.TextValue = value
    return dataset


def create_datetime_item(name: Code, value: datetime.datetime):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.DateTime = value
    return dataset


def create_code_item(name: Code, value: Union[Code, ConceptCode]):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.ConceptCodeSequence = [create_code_dataset(value)]
    return dataset


def create_measurement_item(name: Code, value: Measurement):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.NumericValue = DSfloat(value.value)
    dataset.FloatingPointValue = value.value
    dataset.MeasurementUnitsCodeSequence = [create_code_dataset(value.unit)]
    return dataset


def create_processing_type_item(step: SpecimenPreparationStepDicomModel):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(SampleCodes.processing_type)]
    if isinstance(step, CollectionDicomModel):
        processing_type_code = SampleCodes.specimen_collection
    elif isinstance(step, SamplingDicomModel):
        processing_type_code = SampleCodes.sampling_method
    elif isinstance(step, SpecimenPreparationStepDicomModel):
        processing_type_code = SampleCodes.sample_processing
    elif isinstance(step, StainingDicomModel):
        processing_type_code = SampleCodes.staining
    elif isinstance(step, ReceivingDicomModel):
        processing_type_code = SampleCodes.receiving
    elif isinstance(step, StorageDicomModel):
        processing_type_code = SampleCodes.storage
    else:
        raise NotImplementedError()
    dataset.ConceptCodeSequence = [create_code_dataset(processing_type_code)]
    return dataset


class TestSampleDicom:
    @pytest.mark.parametrize(
        ["slide_sample_ids", "slide_sample_uids", "specimen_ids"],
        [
            [
                ["slide sample"],
                [
                    UID(
                        "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445424"
                    )
                ],
                ["specimen"],
            ],
            [
                ["slide sample 1", "slide sample 2"],
                [
                    UID(
                        "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445424"
                    ),
                    UID(
                        "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445425"
                    ),
                ],
                ["specimen 1", "specimen 2"],
            ],
        ],
    )
    @pytest.mark.parametrize(
        "fixative", [SpecimenFixativesCode("Neutral Buffered Formalin")]
    )
    @pytest.mark.parametrize("medium", [SpecimenEmbeddingMediaCode("Paraffin wax")])
    @pytest.mark.parametrize("short_description", [None, "short description"])
    @pytest.mark.parametrize("detailed_description", [None, "detailed description"])
    @pytest.mark.parametrize(
        "specimen_container", [None, ContainerTypeCode("Specimen container")]
    )
    @pytest.mark.parametrize(
        "block_container", [None, ContainerTypeCode("Tissue cassette")]
    )
    def test_slide_sample_from_dataset(
        self,
        slide_sample_ids: Sequence[str],
        slide_sample_uids: Sequence[UID],
        specimen_ids: Sequence[str],
        block_id: str,
        specimen_type: AnatomicPathologySpecimenTypesCode,
        collection_method: SpecimenCollectionProcedureCode,
        fixative: SpecimenFixativesCode,
        specimen_sampling_method: SpecimenSamplingProcedureCode,
        medium: SpecimenEmbeddingMediaCode,
        block_sampling_method: SpecimenSamplingProcedureCode,
        block_type: AnatomicPathologySpecimenTypesCode,
        specimen_localization: SpecimenLocalization,
        sampling_location: SamplingLocation,
        primary_anatomic_structures: Sequence[Code],
        stains: Sequence[SpecimenStainsCode],
        short_description: Optional[str],
        detailed_description: Optional[str],
        specimen_container: Optional[ContainerTypeCode],
        block_container: Optional[ContainerTypeCode],
    ):
        # Arrange
        descriptions: List[Dataset] = []
        dataset = Dataset()
        for slide_sample_id, slide_sample_uid, specimen_id in zip(
            slide_sample_ids, slide_sample_uids, specimen_ids
        ):
            description = create_description(
                slide_sample_id,
                slide_sample_uid,
                specimen_id,
                block_id,
                specimen_type,
                collection_method,
                fixative,
                specimen_sampling_method,
                medium,
                block_sampling_method,
                block_type,
                specimen_localization,
                sampling_location,
                primary_anatomic_structures,
                stains,
                short_description=short_description,
                detailed_description=detailed_description,
                specimen_container=specimen_container,
                block_container=block_container,
            )

            descriptions.append(description)
        dataset.SpecimenDescriptionSequence = descriptions
        schema = SpecimenDescriptionDicomSchema()

        # Act
        models = [
            schema.load(description)
            for description in dataset.SpecimenDescriptionSequence
        ]
        slide_samples, stainings = SpecimenDicomParser().parse_descriptions(models)

        # Assert
        assert slide_samples is not None
        assert stainings is not None
        assert len(slide_samples) == len(slide_sample_ids)
        for slide_sample_index, slide_sample in enumerate(slide_samples):
            assert isinstance(slide_sample, SlideSample)
            assert slide_sample.identifier == slide_sample_ids[slide_sample_index]
            assert slide_sample.uid == slide_sample_uids[slide_sample_index]
            assert slide_sample.anatomical_sites == primary_anatomic_structures
            assert slide_sample.localization == specimen_localization
            assert slide_sample.sampled_from is not None
            assert slide_sample.sampled_from.method == block_sampling_method
            block = slide_sample.sampled_from.specimen
            assert isinstance(block, Sample)
            assert block.identifier == block_id
            assert block.type == block_type
            assert block.container == block_container
            embedding_step = block.steps[0]
            assert isinstance(embedding_step, Embedding)
            assert embedding_step.medium == medium
            assert len(block.sampled_from) == len(specimen_ids)
            for index, specimen_id in enumerate(specimen_ids):
                assert block.sampled_from[index].method == specimen_sampling_method
                specimen = block.sampled_from[index].specimen
                assert isinstance(specimen, ExtractedSpecimen)
                assert specimen.identifier == specimen_id
                assert specimen.type == specimen_type
                assert specimen.container == specimen_container
                fixation_step = specimen.steps[1]
                assert isinstance(fixation_step, Fixation)
                assert fixation_step.fixative == fixative
                collection_step = specimen.steps[0]
                assert isinstance(collection_step, Collection)
                assert collection_step.method == collection_method


class TestPreparationStepDicomSchema:
    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    @pytest.mark.parametrize(
        "container", [None, ContainerTypeCode("Specimen container")]
    )
    def test_serialize_collection_dicom(
        self,
        collection_dicom: CollectionDicomModel,
        identifier: Union[str, SpecimenIdentifier],
    ):
        # Arrange
        schema = CollectionDicomSchema()

        # Act
        serialized = schema.dump(collection_dicom)

        # Assert
        assert isinstance(serialized, list)
        item_iterator = iter(serialized)
        # First item should be identifier
        self.assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item can be container type
        if collection_dicom.container is not None:
            self.assert_next_item_equals_code(
                item_iterator, SampleCodes.container, collection_dicom.container
            )
        # Next item should be processing type
        self.assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.specimen_collection,
        )
        # Next item can be date time
        if collection_dicom.date_time is not None:
            self.assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                collection_dicom.date_time,
            )
        # Next item can be description
        if collection_dicom.description is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.processing_description,
                collection_dicom.description,
            )
        # Last item should be method
        self.assert_next_item_equals_code(
            item_iterator,
            SampleCodes.specimen_collection,
            collection_dicom.method,
        )
        # There should be no more items
        assert next(item_iterator, None) is None

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    @pytest.mark.parametrize(
        ["fixative", "medium"],
        [
            [None, None],
            [
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                SpecimenEmbeddingMediaCode("Paraffin wax"),
            ],
        ],
    )
    @pytest.mark.parametrize(
        "container", [None, ContainerTypeCode("Specimen container")]
    )
    def test_deserialize_collection_dicom(
        self,
        collection_dicom: CollectionDicomModel,
        collection_dataset: Dataset,
    ):
        # Arrange
        schema = CollectionDicomSchema()

        # Act
        deserialized = schema.load(
            collection_dataset.SpecimenPreparationStepContentItemSequence
        )

        # Assert
        assert isinstance(deserialized, CollectionDicomModel)
        assert deserialized.identifier == collection_dicom.identifier
        assert (
            deserialized.issuer_of_identifier == collection_dicom.issuer_of_identifier
        )
        assert deserialized.method == collection_dicom.method
        assert deserialized.date_time == collection_dicom.date_time
        assert deserialized.description == collection_dicom.description
        assert deserialized.fixative == collection_dicom.fixative
        assert deserialized.embedding == collection_dicom.embedding
        assert deserialized.container == collection_dicom.container

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    @pytest.mark.parametrize(
        "location",
        [
            None,
            SpecimenLocalization(
                "reference",
                "description",
                Measurement(1, UnitCode("cm")),
                Measurement(2, UnitCode("cm")),
                Measurement(3, UnitCode("cm")),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "container", [None, ContainerTypeCode("Specimen container")]
    )
    def test_serialize_sampling_dicom(
        self,
        sampling_dicom: SamplingDicomModel,
        identifier: Union[str, SpecimenIdentifier],
    ):
        # Arrange
        schema = SamplingDicomSchema()

        # Act
        serialized = schema.dump(sampling_dicom)

        # Assert
        assert isinstance(serialized, list)
        item_iterator = iter(serialized)
        # First item should be identifier
        self.assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item can be container type
        if sampling_dicom.container is not None:
            self.assert_next_item_equals_code(
                item_iterator, SampleCodes.container, sampling_dicom.container
            )
        # Next item should be processing type
        self.assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.sampling_method,
        )
        # Next item can be date time
        if sampling_dicom.date_time is not None:
            self.assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                sampling_dicom.date_time,
            )
        # Next item can be description
        if sampling_dicom.description is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.processing_description,
                sampling_dicom.description,
            )
        # Next item should be method
        self.assert_next_item_equals_code(
            item_iterator,
            SampleCodes.sampling_method,
            sampling_dicom.method,
        )
        # Next item should be parent specimen identifier
        self.assert_next_item_equals_string(
            item_iterator,
            SampleCodes.parent_specimen_identifier,
            sampling_dicom.parent_specimen_identifier,
        )
        # Next item can be parent specimen identifier issuer
        if sampling_dicom.issuer_of_parent_specimen_identifier is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_parent_specimen_identifier,
                sampling_dicom.issuer_of_parent_specimen_identifier,
            )
        # Next item should be parent specimen type
        self.assert_next_item_equals_code(
            item_iterator,
            SampleCodes.parent_specimen_type,
            sampling_dicom.parent_specimen_type,
        )
        # Next item can be location reference
        if sampling_dicom.location_reference is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.location_frame_of_reference,
                sampling_dicom.location_reference,
            )
        # Next item can be location description
        if sampling_dicom.location_description is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.location_of_sampling_site,
                sampling_dicom.location_description,
            )
        # Next item can be location x
        if sampling_dicom.location_x is not None:
            self.assert_next_item_equals_measurement(
                item_iterator,
                SampleCodes.location_of_sampling_site_x,
                sampling_dicom.location_x,
            )
        # Next item can be location y
        if sampling_dicom.location_y is not None:
            self.assert_next_item_equals_measurement(
                item_iterator,
                SampleCodes.location_of_sampling_site_y,
                sampling_dicom.location_y,
            )
        # Next item can be location z
        if sampling_dicom.location_z is not None:
            self.assert_next_item_equals_measurement(
                item_iterator,
                SampleCodes.location_of_sampling_site_z,
                sampling_dicom.location_z,
            )
        # There should be no more items
        assert next(item_iterator, None) is None

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    @pytest.mark.parametrize(
        ["fixative", "medium"],
        [
            [None, None],
            [
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                SpecimenEmbeddingMediaCode("Paraffin wax"),
            ],
        ],
    )
    @pytest.mark.parametrize(
        "location",
        [
            None,
            SpecimenLocalization(
                "reference",
                "description",
                Measurement(1, UnitCode("cm")),
                Measurement(2, UnitCode("cm")),
                Measurement(3, UnitCode("cm")),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "container", [None, ContainerTypeCode("Specimen container")]
    )
    def test_deserialize_sampling_dicom(
        self, sampling_dicom: SamplingDicomModel, sampling_dataset: Dataset
    ):
        # Arrange
        schema = SamplingDicomSchema()

        # Act
        deserialized = schema.load(
            sampling_dataset.SpecimenPreparationStepContentItemSequence
        )

        # Assert
        assert isinstance(deserialized, SamplingDicomModel)
        assert deserialized.identifier == sampling_dicom.identifier
        assert deserialized.issuer_of_identifier == sampling_dicom.issuer_of_identifier
        assert deserialized.method == sampling_dicom.method
        assert deserialized.date_time == sampling_dicom.date_time
        assert deserialized.description == sampling_dicom.description
        assert deserialized.fixative == sampling_dicom.fixative
        assert deserialized.embedding == sampling_dicom.embedding
        assert deserialized.location_reference == sampling_dicom.location_reference
        assert deserialized.location_description == sampling_dicom.location_description
        assert deserialized.location_x == sampling_dicom.location_x
        assert deserialized.location_y == sampling_dicom.location_y
        assert deserialized.location_z == sampling_dicom.location_z
        assert deserialized.container == sampling_dicom.container

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    @pytest.mark.parametrize(
        ["fixative", "medium"],
        [
            [None, None],
            [
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                SpecimenEmbeddingMediaCode("Paraffin wax"),
            ],
        ],
    )
    def test_serialize_processing_dicom(
        self,
        processing_dicom: ProcessingDicomModel,
        identifier: Union[str, SpecimenIdentifier],
    ):
        # Arrange
        schema = ProcessingDicomSchema()

        # Act
        serialized = schema.dump(processing_dicom)

        # Assert
        assert isinstance(serialized, list)
        item_iterator = iter(serialized)
        # First item should be identifier
        self.assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item should be processing type
        self.assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.sample_processing,
        )
        # Next item can be date time
        if processing_dicom.date_time is not None:
            self.assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                processing_dicom.date_time,
            )
        # Next item can be description
        if processing_dicom.description is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.processing_description,
                processing_dicom.description,
            )
        # Next item can be method
        if processing_dicom.processing is not None:
            self.assert_next_item_equals_code(
                item_iterator,
                SampleCodes.processing_description,
                processing_dicom.processing,
            )
        # Next item can be fixative
        if processing_dicom.fixative is not None:
            self.assert_next_item_equals_code(
                item_iterator, SampleCodes.fixative, processing_dicom.fixative
            )
        # Next item can be embedding
        if processing_dicom.embedding is not None:
            self.assert_next_item_equals_code(
                item_iterator, SampleCodes.embedding, processing_dicom.embedding
            )
        # There should be no more items
        assert next(item_iterator, None) is None

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    @pytest.mark.parametrize(
        ["fixative", "medium"],
        [
            [None, None],
            [
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                SpecimenEmbeddingMediaCode("Paraffin wax"),
            ],
        ],
    )
    def test_deserialize_processing_dicom(
        self,
        processing_dicom: ProcessingDicomModel,
        processing_dataset: Dataset,
    ):
        # Arrange

        schema = ProcessingDicomSchema()

        # Act
        deserialized = schema.load(
            processing_dataset.SpecimenPreparationStepContentItemSequence
        )

        # Assert
        assert isinstance(deserialized, ProcessingDicomModel)
        assert deserialized.identifier == processing_dicom.identifier
        assert (
            deserialized.issuer_of_identifier == processing_dicom.issuer_of_identifier
        )
        assert deserialized.processing == processing_dicom.processing
        assert deserialized.date_time == processing_dicom.date_time
        assert deserialized.description == processing_dicom.description
        assert deserialized.fixative == processing_dicom.fixative
        assert deserialized.embedding == processing_dicom.embedding

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    def test_serialize_staining_dicom(
        self,
        staining_dicom: StainingDicomModel,
        identifier: Union[str, SpecimenIdentifier],
    ):
        # Arrange
        schema = StainingDicomSchema()

        # Act
        serialized = schema.dump(staining_dicom)

        # Assert
        assert isinstance(serialized, list)
        item_iterator = iter(serialized)
        item_iterator = iter(serialized)
        # First item should be identifier
        self.assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item should be processing type
        self.assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.staining,
        )
        # Next item can be date time
        if staining_dicom.date_time is not None:
            self.assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                staining_dicom.date_time,
            )
        # Next item can be description
        if staining_dicom.description is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.processing_description,
                staining_dicom.description,
            )
        # Next item can be staining
        for substance in staining_dicom.substances:
            if isinstance(substance, SpecimenStainsCode):
                self.assert_next_item_equals_code(
                    item_iterator, SampleCodes.using_substance, substance
                )
            else:
                self.assert_next_item_equals_string(
                    item_iterator, SampleCodes.using_substance, substance
                )
        # There should be no more items
        assert next(item_iterator, None) is None

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    def test_deserialize_staining_dicom(
        self,
        staining_dicom: StainingDicomModel,
        staining_dataset: Dataset,
    ):
        # Arrange

        schema = StainingDicomSchema()

        # Act
        deserialized = schema.load(
            staining_dataset.SpecimenPreparationStepContentItemSequence
        )

        # Assert
        assert isinstance(deserialized, StainingDicomModel)
        assert deserialized.identifier == staining_dicom.identifier
        assert deserialized.issuer_of_identifier == staining_dicom.issuer_of_identifier
        assert deserialized.substances == staining_dicom.substances
        assert deserialized.date_time == staining_dicom.date_time
        assert deserialized.description == staining_dicom.description
        assert deserialized.fixative == staining_dicom.fixative
        assert deserialized.embedding == staining_dicom.embedding

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    def test_serialize_receiving_dicom(
        self,
        receiving_dicom: ReceivingDicomModel,
        identifier: Union[str, SpecimenIdentifier],
    ):
        # Arrange
        schema = ReceivingDicomSchema()

        # Act
        serialized = schema.dump(receiving_dicom)

        # Assert
        assert isinstance(serialized, list)
        item_iterator = iter(serialized)
        # First item should be identifier
        self.assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item should be processing type
        self.assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.receiving,
        )
        # Next item can be date time
        if receiving_dicom.date_time is not None:
            self.assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                receiving_dicom.date_time,
            )
        # Next item can be description
        if receiving_dicom.description is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.processing_description,
                receiving_dicom.description,
            )
        # There should be no more items
        assert next(item_iterator, None) is None

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    def test_deserialize_receiving_dicom(
        self,
        receiving_dicom: ReceivingDicomModel,
        receiving_dataset: Dataset,
    ):
        # Arrange

        schema = ReceivingDicomSchema()

        # Act
        deserialized = schema.load(
            receiving_dataset.SpecimenPreparationStepContentItemSequence
        )

        # Assert
        assert isinstance(deserialized, ReceivingDicomModel)
        assert deserialized.identifier == receiving_dicom.identifier
        assert deserialized.issuer_of_identifier == receiving_dicom.issuer_of_identifier
        assert deserialized.date_time == receiving_dicom.date_time
        assert deserialized.description == receiving_dicom.description

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    def test_serialize_storage_dicom(
        self,
        storage_dicom: StorageDicomModel,
        identifier: Union[str, SpecimenIdentifier],
    ):
        # Arrange
        schema = StorageDicomSchema()

        # Act
        serialized = schema.dump(storage_dicom)

        # Assert
        assert isinstance(serialized, list)
        item_iterator = iter(serialized)
        # First item should be identifier
        self.assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item should be processing type
        self.assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.storage,
        )
        # Next item can be date time
        if storage_dicom.date_time is not None:
            self.assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                storage_dicom.date_time,
            )
        # Next item can be description
        if storage_dicom.description is not None:
            self.assert_next_item_equals_string(
                item_iterator,
                SampleCodes.processing_description,
                storage_dicom.description,
            )
        # There should be no more items
        assert next(item_iterator, None) is None

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    def test_deserialize_storage_dicom(
        self,
        storage_dicom: StorageDicomModel,
        storage_dataset: Dataset,
    ):
        # Arrange

        schema = StorageDicomSchema()

        # Act
        deserialized = schema.load(
            storage_dataset.SpecimenPreparationStepContentItemSequence
        )

        # Assert
        assert isinstance(deserialized, StorageDicomModel)
        assert deserialized.identifier == storage_dicom.identifier
        assert deserialized.issuer_of_identifier == storage_dicom.issuer_of_identifier
        assert deserialized.date_time == storage_dicom.date_time
        assert deserialized.description == storage_dicom.description

    @pytest.mark.parametrize("reference", ["reference", None])
    @pytest.mark.parametrize("description", ["description", None])
    @pytest.mark.parametrize("x", [Measurement(1, UnitCode("mm")), None])
    @pytest.mark.parametrize("y", [Measurement(1, UnitCode("mm")), None])
    @pytest.mark.parametrize("z", [Measurement(1, UnitCode("mm")), None])
    @pytest.mark.parametrize("visual_marking", ["visual_marking", None])
    def test_serialize_specimen_location_dicom(
        self,
        reference: Optional[str],
        description: Optional[str],
        x: Optional[Measurement],
        y: Optional[Measurement],
        z: Optional[Measurement],
        visual_marking: Optional[str],
    ):
        # Arrange
        location = SpecimenLocalization(
            reference=reference,
            description=description,
            x=x,
            y=y,
            z=z,
            visual_marking=visual_marking,
        )
        schema = SpecimenLocalizationDicomSchema()

        # Act
        serialized = schema.dump(location)

        # Assert
        assert isinstance(serialized, list)
        item_iterator = iter(serialized)
        for string, name in [
            (reference, SampleCodes.location_frame_of_reference),
            (description, SampleCodes.location_of_specimen),
        ]:
            if string is not None:
                string_item = next(item_iterator)
                self.assert_item_name_equals_code(string_item, name)
                self.assert_item_string_equals_value(string_item, string)
        for measurement, name in [
            (x, SampleCodes.location_of_specimen_x),
            (y, SampleCodes.location_of_specimen_y),
            (z, SampleCodes.location_of_specimen_z),
        ]:
            if measurement is not None:
                measurement_item = next(item_iterator)
                self.assert_item_name_equals_code(measurement_item, name)
                self.assert_item_measurement_equals_value(measurement_item, measurement)

        if visual_marking is not None:
            visual_marking_item = next(item_iterator)
            self.assert_item_name_equals_code(
                visual_marking_item, SampleCodes.visual_marking_of_specimen
            )
            self.assert_item_string_equals_value(visual_marking_item, visual_marking)

    @pytest.mark.parametrize("reference", ["reference", None])
    @pytest.mark.parametrize("description", ["description", None])
    @pytest.mark.parametrize("x", [Measurement(1, UnitCode("mm")), None])
    @pytest.mark.parametrize("y", [Measurement(1, UnitCode("mm")), None])
    @pytest.mark.parametrize("z", [Measurement(1, UnitCode("mm")), None])
    @pytest.mark.parametrize("visual_marking", ["visual_marking", None])
    def test_deserialize_specimen_location_dicom(
        self,
        reference: Optional[str],
        description: Optional[str],
        x: Optional[Measurement],
        y: Optional[Measurement],
        z: Optional[Measurement],
        visual_marking: Optional[str],
    ):
        # Arrange
        sequence = []
        if reference is not None:
            sequence.append(
                create_string_item(SampleCodes.location_frame_of_reference, reference)
            )
        if description is not None:
            sequence.append(
                create_string_item(SampleCodes.location_of_specimen, description)
            )
        if x is not None:
            sequence.append(
                create_measurement_item(SampleCodes.location_of_specimen_x, x)
            )
        if y is not None:
            sequence.append(
                create_measurement_item(SampleCodes.location_of_specimen_y, y)
            )
        if z is not None:
            sequence.append(
                create_measurement_item(SampleCodes.location_of_specimen_z, z)
            )
        if visual_marking is not None:
            sequence.append(
                create_string_item(
                    SampleCodes.visual_marking_of_specimen, visual_marking
                )
            )
        schema = SpecimenLocalizationDicomSchema()

        # Act
        deserialized = schema.load(sequence)

        # Assert
        assert isinstance(deserialized, SpecimenLocalization)
        assert deserialized.reference == reference
        assert deserialized.description == description
        assert deserialized.x == x
        assert deserialized.y == y
        assert deserialized.z == z
        assert deserialized.visual_marking == visual_marking

    def assert_item_name_equals_code(self, item: Dataset, name: Code):
        self.assert_code_dataset_equals_code(item.ConceptNameCodeSequence[0], name)

    def assert_item_string_equals_value(self, item: Dataset, value: str):
        assert item.TextValue == value

    def assert_item_datetime_equals_value(
        self, item: Dataset, value: datetime.datetime
    ):
        assert item.DateTime == value

    def assert_item_code_equals_value(
        self, item: Dataset, value: Union[Code, ConceptCode]
    ):
        self.assert_code_dataset_equals_code(item.ConceptCodeSequence[0], value)

    def assert_item_measurement_equals_value(self, item: Dataset, value: Measurement):
        assert item.FloatingPointValue == value.value
        self.assert_code_dataset_equals_code(
            item.MeasurementUnitsCodeSequence[0],
            value.unit,
        )

    def assert_code_dataset_equals_code(
        self, item: Dataset, code: Union[Code, ConceptCode]
    ):
        assert item.CodeValue == code.value, (item.CodeMeaning, code.meaning)
        assert item.CodingSchemeDesignator == code.scheme_designator, (
            item.CodeMeaning,
            code.meaning,
        )

    def assert_next_item_equals_string(
        self, iterator: Iterator[Dataset], name: Code, value: str
    ):
        item = next(iterator)
        self.assert_item_name_equals_code(item, name)
        self.assert_item_string_equals_value(item, value)

    def assert_next_item_equals_code(
        self, iterator: Iterator[Dataset], name: Code, value: Union[Code, ConceptCode]
    ):
        item = next(iterator)
        self.assert_item_name_equals_code(item, name)
        self.assert_item_code_equals_value(item, value)

    def assert_next_item_equals_datetime(
        self, iterator: Iterator[Dataset], name: Code, value: datetime.datetime
    ):
        item = next(iterator)
        self.assert_item_name_equals_code(item, name)
        self.assert_item_datetime_equals_value(item, value)

    def assert_next_item_equals_measurement(
        self, iterator: Iterator[Dataset], name: Code, value: Measurement
    ):
        item = next(iterator)
        self.assert_item_name_equals_code(item, name)
        self.assert_item_measurement_equals_value(item, value)
