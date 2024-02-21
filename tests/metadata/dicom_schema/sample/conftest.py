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

from typing import Optional, Union

import pytest
from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import UID

from tests.metadata.dicom_schema.helpers import (
    create_code_dataset,
    create_collection_dataset,
    create_identifier_items,
    create_processing_dataset,
    create_receiving_dataset,
    create_sampling_dataset,
    create_staining_dataset,
    create_storage_dataset,
)
from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    ContainerTypeCode,
    SpecimenCollectionProcedureCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenPreparationStepsCode,
    SpecimenSamplingProcedureCode,
    SpecimenStainsCode,
)
from wsidicom.metadata.sample import (
    Collection,
    Processing,
    Receiving,
    SampleLocalization,
    Sampling,
    SamplingLocation,
    SpecimenIdentifier,
    Staining,
    Storage,
)
from wsidicom.metadata.schema.dicom.sample.model import (
    CollectionDicomModel,
    ProcessingDicomModel,
    ReceivingDicomModel,
    SamplingDicomModel,
    StainingDicomModel,
    StorageDicomModel,
)
from wsidicom.metadata.schema.dicom.sample.schema import SampleCodes


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
def sample_localization():
    yield SampleLocalization(description="left")


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
    specimen_type: Optional[AnatomicPathologySpecimenTypesCode],
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
        specimen_type=specimen_type,
    )


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
    specimen_type: Optional[AnatomicPathologySpecimenTypesCode],
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
        parent_specimen_type=sampling.specimen_type,
        processing=processing_method,
        location_reference=(
            sampling.location.reference if sampling.location is not None else None
        ),
        location_description=(
            sampling.location.description if sampling.location is not None else None
        ),
        location_x=sampling.location.x if sampling.location is not None else None,
        location_y=sampling.location.y if sampling.location is not None else None,
        location_z=sampling.location.z if sampling.location is not None else None,
        container=container,
        specimen_type=specimen_type,
    )


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
    container: Optional[ContainerTypeCode],
    specimen_type: Optional[AnatomicPathologySpecimenTypesCode],
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
        container=container,
        specimen_type=specimen_type,
    )


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
    container: Optional[ContainerTypeCode],
    specimen_type: Optional[AnatomicPathologySpecimenTypesCode],
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
        container=container,
        specimen_type=specimen_type,
    )


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
    fixative: Optional[SpecimenFixativesCode],
    medium: Optional[SpecimenEmbeddingMediaCode],
    processing_method: Optional[SpecimenPreparationStepsCode],
    container: Optional[ContainerTypeCode],
    specimen_type: Optional[AnatomicPathologySpecimenTypesCode],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    yield ReceivingDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=receiving.date_time,
        description=receiving.description,
        fixative=fixative,
        embedding=medium,
        processing=processing_method,
        container=container,
        specimen_type=specimen_type,
    )


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
    fixative: Optional[SpecimenFixativesCode],
    medium: Optional[SpecimenEmbeddingMediaCode],
    processing_method: Optional[SpecimenPreparationStepsCode],
    container: Optional[ContainerTypeCode],
    specimen_type: Optional[AnatomicPathologySpecimenTypesCode],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    yield StorageDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=storage.date_time,
        description=storage.description,
        fixative=fixative,
        embedding=medium,
        processing=processing_method,
        container=container,
        specimen_type=specimen_type,
    )


@pytest.fixture()
def storage_dataset(
    storage_dicom: StorageDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_storage_dataset(storage_dicom, identifier)


@pytest.fixture()
def invalid_preparation_step_dataset(invalid_preparation_step_type: str):
    identifier = "identifier"
    items = create_identifier_items(identifier)
    if invalid_preparation_step_type == "missing_processing_type":
        # Processing type item has wrong concept name code
        dataset = Dataset()
        dataset.ValueType = "CODE"
        dataset.ConceptNameCodeSequence = [
            create_code_dataset(Code("not valid", "SCT", "not valid"))
        ]
        dataset.ConceptCodeSequence = [create_code_dataset(SampleCodes.sampling_method)]
        items.append(dataset)
    elif invalid_preparation_step_type == "unknown_processing_type":
        # Processing type item has unknown concept code
        dataset = Dataset()
        dataset.ValueType = "CODE"
        dataset.ConceptNameCodeSequence = [
            create_code_dataset(SampleCodes.processing_type)
        ]
        dataset.ConceptCodeSequence = [
            create_code_dataset(Code("not valid", "SCT", "not valid"))
        ]
        items.append(dataset)
    elif invalid_preparation_step_type == "validation_error":
        # Sampling does not specify method
        dataset = Dataset()
        dataset.ValueType = "CODE"
        dataset.ConceptNameCodeSequence = [
            create_code_dataset(SampleCodes.processing_type)
        ]
        dataset.ConceptCodeSequence = [create_code_dataset(SampleCodes.sampling_method)]
        items.append(dataset)
    else:
        raise ValueError(
            f"Unknown invalid preparation step type: {invalid_preparation_step_type}"
        )

    dataset = Dataset()
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset
