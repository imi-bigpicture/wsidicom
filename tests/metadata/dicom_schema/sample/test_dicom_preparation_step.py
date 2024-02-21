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

import marshmallow
import pytest
from pydicom import Dataset
from pydicom.uid import UID

from tests.metadata.dicom_schema.helpers import (
    assert_initial_common_preparation_step_items,
    assert_item_measurement_equals_value,
    assert_item_name_equals_code,
    assert_item_string_equals_value,
    assert_last_common_preparation_step_items,
    assert_next_item_equals_code,
    assert_next_item_equals_measurement,
    assert_next_item_equals_string,
    create_description_dataset,
    create_measurement_item,
    create_string_item,
)
from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    ContainerTypeCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenPreparationStepsCode,
    SpecimenStainsCode,
    UnitCode,
)
from wsidicom.config import settings
from wsidicom.metadata.sample import (
    LocalIssuerOfIdentifier,
    Measurement,
    SampleLocalization,
    SpecimenIdentifier,
)
from wsidicom.metadata.schema.dicom.sample.model import (
    CollectionDicomModel,
    ProcessingDicomModel,
    ReceivingDicomModel,
    SamplingDicomModel,
    SpecimenDescriptionDicomModel,
    StainingDicomModel,
    StorageDicomModel,
)
from wsidicom.metadata.schema.dicom.sample.schema import (
    CollectionDicomSchema,
    PreparationStepDicomField,
    ProcessingDicomSchema,
    ReceivingDicomSchema,
    SampleCodes,
    SampleLocalizationDicomSchema,
    SamplingDicomSchema,
    SpecimenDescriptionDicomSchema,
    StainingDicomSchema,
    StorageDicomSchema,
)


@pytest.mark.parametrize(
    "identifier",
    [
        "identifier",
        SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
    ],
)
@pytest.mark.parametrize(
    [
        "container",
        "specimen_type",
        "date_time",
        "description",
        "processing_method",
        "fixative",
        "medium",
    ],
    [
        [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        [
            ContainerTypeCode("Specimen container"),
            AnatomicPathologySpecimenTypesCode("tissue specimen"),
            datetime.datetime(2021, 1, 1),
            "description",
            SpecimenPreparationStepsCode("Specimen clearing"),
            SpecimenFixativesCode("Neutral Buffered Formalin"),
            SpecimenEmbeddingMediaCode("Paraffin wax"),
        ],
    ],
)
class TestPreparationStepDicomSchema:
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
        assert_initial_common_preparation_step_items(
            item_iterator, collection_dicom, identifier, SampleCodes.specimen_collection
        )
        # Next item should be method
        assert_next_item_equals_code(
            item_iterator,
            SampleCodes.specimen_collection,
            collection_dicom.method,
        )
        assert_last_common_preparation_step_items(item_iterator, collection_dicom)
        # There should be no more items
        assert next(item_iterator, None) is None

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
        assert deserialized.processing == collection_dicom.processing

    @pytest.mark.parametrize(
        "location",
        [
            None,
            SampleLocalization(
                "reference",
                "description",
                Measurement(1, UnitCode("cm")),
                Measurement(2, UnitCode("cm")),
                Measurement(3, UnitCode("cm")),
            ),
        ],
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
        assert_initial_common_preparation_step_items(
            item_iterator,
            sampling_dicom,
            identifier,
            SampleCodes.sampling_of_tissue_specimen,
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
        assert_last_common_preparation_step_items(item_iterator, sampling_dicom)
        # There should be no more items
        assert next(item_iterator, None) is None

    @pytest.mark.parametrize(
        "location",
        [
            None,
            SampleLocalization(
                "reference",
                "description",
                Measurement(1, UnitCode("cm")),
                Measurement(2, UnitCode("cm")),
                Measurement(3, UnitCode("cm")),
            ),
        ],
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
        assert deserialized.processing == sampling_dicom.processing

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
        assert_initial_common_preparation_step_items(
            item_iterator, processing_dicom, identifier, SampleCodes.sample_processing
        )
        assert_last_common_preparation_step_items(item_iterator, processing_dicom)
        # There should be no more items
        assert next(item_iterator, None) is None

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
        assert deserialized.container == processing_dicom.container
        assert deserialized.specimen_type == processing_dicom.specimen_type

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
        assert_initial_common_preparation_step_items(
            item_iterator, staining_dicom, identifier, SampleCodes.staining
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
        assert_last_common_preparation_step_items(item_iterator, staining_dicom)
        # There should be no more items
        assert next(item_iterator, None) is None

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
        assert deserialized.processing == staining_dicom.processing
        assert deserialized.container == staining_dicom.container
        assert deserialized.specimen_type == staining_dicom.specimen_type

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
        assert_initial_common_preparation_step_items(
            item_iterator, receiving_dicom, identifier, SampleCodes.receiving
        )
        assert_last_common_preparation_step_items(item_iterator, receiving_dicom)
        # There should be no more items
        assert next(item_iterator, None) is None

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
        assert deserialized.fixative == receiving_dicom.fixative
        assert deserialized.embedding == receiving_dicom.embedding
        assert deserialized.processing == receiving_dicom.processing
        assert deserialized.container == receiving_dicom.container
        assert deserialized.specimen_type == receiving_dicom.specimen_type

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

        assert_initial_common_preparation_step_items(
            item_iterator, storage_dicom, identifier, SampleCodes.storage
        )
        assert_last_common_preparation_step_items(item_iterator, storage_dicom)
        # There should be no more items
        assert next(item_iterator, None) is None

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
        assert deserialized.fixative == storage_dicom.fixative
        assert deserialized.embedding == storage_dicom.embedding
        assert deserialized.processing == storage_dicom.processing
        assert deserialized.container == storage_dicom.container
        assert deserialized.specimen_type == storage_dicom.specimen_type


@pytest.mark.parametrize(
    "invalid_preparation_step_type",
    ["missing_processing_type", "unknown_processing_type", "validation_error"],
)
class TestPreparationStepDicomErrorHandling:
    def test_deserialize_step_validation_error_with_ignore_setting(
        self, invalid_preparation_step_dataset: Dataset
    ):
        # Arrange
        settings.ignore_specimen_preparation_step_on_validation_error = True
        field = PreparationStepDicomField(
            data_key="SpecimenPreparationStepContentItemSequence"
        )

        # Act
        deserialized = field._deserialize(invalid_preparation_step_dataset, None, {})

        # Assert
        assert deserialized is None

    def test_deserialize_step_validation_error_with_throw_setting(
        self, invalid_preparation_step_dataset: Dataset
    ):
        # Arrange
        settings.ignore_specimen_preparation_step_on_validation_error = False
        field = PreparationStepDicomField(
            data_key="SpecimenPreparationStepContentItemSequence"
        )

        # Act & Assert
        with pytest.raises(marshmallow.ValidationError):
            field._deserialize(invalid_preparation_step_dataset, None, {})

    def test_deserialize_steps_validation_error_with_ignore_setting(
        self,
        invalid_preparation_step_dataset: Dataset,
        slide_sample_id: str,
        slide_sample_uid: UID,
        staining_dataset: Dataset,
    ):
        # Arrange
        settings.ignore_specimen_preparation_step_on_validation_error = True
        description = create_description_dataset(
            slide_sample_id,
            slide_sample_uid,
            preparation_step_datasets=[
                invalid_preparation_step_dataset,
                staining_dataset,
            ],
        )
        dataset = Dataset()
        dataset.SpecimenDescriptionSequence = [description]

        schema = SpecimenDescriptionDicomSchema()

        # Act
        models = [
            schema.load(description)
            for description in dataset.SpecimenDescriptionSequence
        ]

        # Assert
        assert len(models) == 1
        model = models[0]
        assert isinstance(model, SpecimenDescriptionDicomModel)
        # Assert that the invalid preparation step was ignored
        assert len(model.steps) == 1
        step = model.steps[0]
        assert isinstance(step, StainingDicomModel)

    def test_deserialize_steps_validation_error_with_throw_setting(
        self,
        invalid_preparation_step_dataset: Dataset,
        slide_sample_id: str,
        slide_sample_uid: UID,
        staining_dataset: Dataset,
    ):
        # Arrange
        settings.ignore_specimen_preparation_step_on_validation_error = False
        description = create_description_dataset(
            slide_sample_id,
            slide_sample_uid,
            preparation_step_datasets=[
                invalid_preparation_step_dataset,
                staining_dataset,
            ],
        )
        dataset = Dataset()
        dataset.SpecimenDescriptionSequence = [description]

        schema = SpecimenDescriptionDicomSchema()

        # Act
        models = [
            schema.load(description)
            for description in dataset.SpecimenDescriptionSequence
        ]

        # Assert
        assert len(models) == 1
        model = models[0]
        assert isinstance(model, SpecimenDescriptionDicomModel)
        # Assert that all preparation step are ignored
        assert len(model.steps) == 0


class TestSampleLocalizationDicomSchema:
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
        location = SampleLocalization(
            reference=reference,
            description=description,
            x=x,
            y=y,
            z=z,
            visual_marking=visual_marking,
        )
        schema = SampleLocalizationDicomSchema()

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
        schema = SampleLocalizationDicomSchema()

        # Act
        deserialized = schema.load(sequence)

        # Assert
        assert isinstance(deserialized, SampleLocalization)
        assert deserialized.reference == reference
        assert deserialized.description == description
        assert deserialized.x == x
        assert deserialized.y == y
        assert deserialized.z == z
        assert deserialized.visual_marking == visual_marking
