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
from typing import Optional, Union
from pydicom import Dataset
import pytest

from tests.metadata.dicom_schema.helpers import (
    assert_item_measurement_equals_value,
    assert_item_name_equals_code,
    assert_item_string_equals_value,
    assert_next_item_equals_code,
    assert_next_item_equals_datetime,
    assert_next_item_equals_measurement,
    assert_next_item_equals_string,
    create_measurement_item,
    create_string_item,
)
from wsidicom.conceptcode import (
    ContainerTypeCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenStainsCode,
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
from wsidicom.metadata.schema.dicom.sample.schema import (
    CollectionDicomSchema,
    ProcessingDicomSchema,
    ReceivingDicomSchema,
    SamplingDicomSchema,
    SpecimenLocalizationDicomSchema,
    StainingDicomSchema,
    SampleCodes,
    StorageDicomSchema,
)

from wsidicom.metadata.sample import (
    LocalIssuerOfIdentifier,
    Measurement,
    SpecimenLocalization,
    SpecimenIdentifier,
)


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
