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
from typing import List, Optional, Sequence, Union
from pydicom import Dataset
import pytest

from pydicom.uid import UID
from pydicom.sr.coding import Code
from tests.metadata.dicom_schema.helpers import (
    assert_item_measurement_equals_value,
    assert_item_name_equals_code,
    assert_item_string_equals_value,
    assert_next_item_equals_code,
    assert_next_item_equals_datetime,
    assert_next_item_equals_measurement,
    assert_next_item_equals_string,
    create_collection_dataset,
    create_description_dataset,
    create_measurement_item,
    create_processing_dataset,
    create_receiving_dataset,
    create_sampling_dataset,
    create_staining_dataset,
    create_storage_dataset,
    create_string_item,
)
from wsidicom.conceptcode import (
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
from wsidicom.metadata.schema.dicom.sample.model import (
    CollectionDicomModel,
    ProcessingDicomModel,
    StainingDicomModel,
    SamplingDicomModel,
    ReceivingDicomModel,
    StorageDicomModel,
)
from wsidicom.metadata.schema.dicom.sample.parser import SpecimenDicomParser
from wsidicom.metadata.schema.dicom.sample.schema import (
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
    Specimen,
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
        container=None,
        specimen_type=None,
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
        container=None,
        specimen_type=None,
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
        container=None,
        specimen_type=None,
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
        container=None,
        specimen_type=None,
    )


@pytest.fixture()
def storage_dataset(
    storage_dicom: StorageDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_storage_dataset(storage_dicom, identifier)


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
            description = create_description_dataset(
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
            if isinstance(slide_sample.sampled_from, Sampling):
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
                loaded_specimen_sampling = block.sampled_from[index]
                if isinstance(loaded_specimen_sampling, Sampling):
                    assert loaded_specimen_sampling.method == specimen_sampling_method
                specimen = loaded_specimen_sampling.specimen
                assert isinstance(specimen, Specimen)
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
    @pytest.mark.parametrize(
        "specimen_type", [None, AnatomicPathologySpecimenTypesCode("tissue specimen")]
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
        assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item can be container type
        if collection_dicom.container is not None:
            assert_next_item_equals_code(
                item_iterator, SampleCodes.container, collection_dicom.container
            )
        # Next item can be specimen type
        if collection_dicom.specimen_type is not None:
            assert_next_item_equals_code(
                item_iterator,
                SampleCodes.specimen_type,
                collection_dicom.specimen_type,
            )
        # Next item should be processing type
        assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.specimen_collection,
        )
        # Next item can be date time
        if collection_dicom.date_time is not None:
            assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                collection_dicom.date_time,
            )
        # Next item can be description
        if collection_dicom.description is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.processing_description,
                collection_dicom.description,
            )
        # Last item should be method
        assert_next_item_equals_code(
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
    @pytest.mark.parametrize(
        "specimen_type", [None, AnatomicPathologySpecimenTypesCode("tissue specimen")]
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
        assert deserialized.specimen_type == collection_dicom.specimen_type

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
    @pytest.mark.parametrize(
        "specimen_type", [None, AnatomicPathologySpecimenTypesCode("tissue specimen")]
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
        assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item can be container type
        if sampling_dicom.container is not None:
            assert_next_item_equals_code(
                item_iterator, SampleCodes.container, sampling_dicom.container
            )
        # Next item can be specimen type
        if sampling_dicom.specimen_type is not None:
            assert_next_item_equals_code(
                item_iterator,
                SampleCodes.specimen_type,
                sampling_dicom.specimen_type,
            )
        # Next item should be processing type
        assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.sampling_method,
        )
        # Next item can be date time
        if sampling_dicom.date_time is not None:
            assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                sampling_dicom.date_time,
            )
        # Next item can be description
        if sampling_dicom.description is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.processing_description,
                sampling_dicom.description,
            )
        # Next item should be method
        assert_next_item_equals_code(
            item_iterator,
            SampleCodes.sampling_method,
            sampling_dicom.method,
        )
        # Next item should be parent specimen identifier
        assert_next_item_equals_string(
            item_iterator,
            SampleCodes.parent_specimen_identifier,
            sampling_dicom.parent_specimen_identifier,
        )
        # Next item can be parent specimen identifier issuer
        if sampling_dicom.issuer_of_parent_specimen_identifier is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_parent_specimen_identifier,
                sampling_dicom.issuer_of_parent_specimen_identifier,
            )
        # Next item should be parent specimen type
        assert_next_item_equals_code(
            item_iterator,
            SampleCodes.parent_specimen_type,
            sampling_dicom.parent_specimen_type,
        )
        # Next item can be location reference
        if sampling_dicom.location_reference is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.location_frame_of_reference,
                sampling_dicom.location_reference,
            )
        # Next item can be location description
        if sampling_dicom.location_description is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.location_of_sampling_site,
                sampling_dicom.location_description,
            )
        # Next item can be location x
        if sampling_dicom.location_x is not None:
            assert_next_item_equals_measurement(
                item_iterator,
                SampleCodes.location_of_sampling_site_x,
                sampling_dicom.location_x,
            )
        # Next item can be location y
        if sampling_dicom.location_y is not None:
            assert_next_item_equals_measurement(
                item_iterator,
                SampleCodes.location_of_sampling_site_y,
                sampling_dicom.location_y,
            )
        # Next item can be location z
        if sampling_dicom.location_z is not None:
            assert_next_item_equals_measurement(
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
    @pytest.mark.parametrize(
        "specimen_type", [None, AnatomicPathologySpecimenTypesCode("tissue specimen")]
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
        assert deserialized.specimen_type == sampling_dicom.specimen_type

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
        assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item should be processing type
        assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.sample_processing,
        )
        # Next item can be date time
        if processing_dicom.date_time is not None:
            assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                processing_dicom.date_time,
            )
        # Next item can be description
        if processing_dicom.description is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.processing_description,
                processing_dicom.description,
            )
        # Next item can be method
        if processing_dicom.processing is not None:
            assert_next_item_equals_code(
                item_iterator,
                SampleCodes.processing_description,
                processing_dicom.processing,
            )
        # Next item can be fixative
        if processing_dicom.fixative is not None:
            assert_next_item_equals_code(
                item_iterator, SampleCodes.fixative, processing_dicom.fixative
            )
        # Next item can be embedding
        if processing_dicom.embedding is not None:
            assert_next_item_equals_code(
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
        assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item should be processing type
        assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.staining,
        )
        # Next item can be date time
        if staining_dicom.date_time is not None:
            assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                staining_dicom.date_time,
            )
        # Next item can be description
        if staining_dicom.description is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.processing_description,
                staining_dicom.description,
            )
        # Next item can be staining
        for substance in staining_dicom.substances:
            if isinstance(substance, SpecimenStainsCode):
                assert_next_item_equals_code(
                    item_iterator, SampleCodes.using_substance, substance
                )
            else:
                assert_next_item_equals_string(
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
        assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item should be processing type
        assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.receiving,
        )
        # Next item can be date time
        if receiving_dicom.date_time is not None:
            assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                receiving_dicom.date_time,
            )
        # Next item can be description
        if receiving_dicom.description is not None:
            assert_next_item_equals_string(
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
        assert_next_item_equals_string(
            item_iterator,
            SampleCodes.identifier,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            assert_next_item_equals_string(
                item_iterator,
                SampleCodes.issuer_of_identifier,
                identifier.issuer.to_hl7v2(),
            )
        # Next item should be processing type
        assert_next_item_equals_code(
            item_iterator,
            SampleCodes.processing_type,
            SampleCodes.storage,
        )
        # Next item can be date time
        if storage_dicom.date_time is not None:
            assert_next_item_equals_datetime(
                item_iterator,
                SampleCodes.datetime_of_processing,
                storage_dicom.date_time,
            )
        # Next item can be description
        if storage_dicom.description is not None:
            assert_next_item_equals_string(
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
                assert_item_name_equals_code(string_item, name)
                assert_item_string_equals_value(string_item, string)
        for measurement, name in [
            (x, SampleCodes.location_of_specimen_x),
            (y, SampleCodes.location_of_specimen_y),
            (z, SampleCodes.location_of_specimen_z),
        ]:
            if measurement is not None:
                measurement_item = next(item_iterator)
                assert_item_name_equals_code(measurement_item, name)
                assert_item_measurement_equals_value(measurement_item, measurement)

        if visual_marking is not None:
            visual_marking_item = next(item_iterator)
            assert_item_name_equals_code(
                visual_marking_item, SampleCodes.visual_marking_of_specimen
            )
            assert_item_string_equals_value(visual_marking_item, visual_marking)

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
