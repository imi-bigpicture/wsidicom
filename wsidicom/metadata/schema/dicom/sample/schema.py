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

"""Schemas for serializing specimen description."""

import datetime
import logging
from collections.abc import Iterable, Sequence
from typing import Any

from marshmallow import ValidationError, fields, post_load
from pydicom import Dataset
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code
from pydicom.valuerep import VR

from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    ContainerTypeCode,
    SpecimenCollectionProcedureCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenPreparationProcedureCode,
    SpecimenPreparationStepsCode,
    SpecimenSamplingProcedureCode,
    SpecimenStainsCode,
    dataset_to_code,
)
from wsidicom.config import get_settings
from wsidicom.metadata.sample import Measurement, SampleLocalization
from wsidicom.metadata.schema.dicom.fields import (
    CodeDicomField,
    CodeItemDicomField,
    ContentItemDicomField,
    DatasetDicomField,
    DateTimeItemDicomField,
    DefaultOnValidationExceptionField,
    IssuerOfIdentifierDicomField,
    ListDicomField,
    MeasurementItemDicomField,
    SingleCodeSequenceDicomField,
    StringDicomField,
    StringItemDicomField,
    UidDicomField,
)
from wsidicom.metadata.schema.dicom.sample.model import (
    CollectionDicomModel,
    ProcessingDicomModel,
    ReceivingDicomModel,
    SamplingDicomModel,
    SpecimenDescriptionDicomModel,
    SpecimenPreparationStepDicomModel,
    StainingDicomModel,
    StorageDicomModel,
)
from wsidicom.metadata.schema.dicom.schema import (
    DicomSchema,
    ItemField,
    ItemSequenceDicomSchema,
    LoadType,
)
from wsidicom.tags import TextValueTag

logger = logging.getLogger(__name__)


class SampleCodes:
    identifier: Code = codes.DCM.SpecimenIdentifier  # type: ignore
    issuer_of_identifier: Code = codes.DCM.IssuerOfSpecimenIdentifier  # type: ignore
    processing_type: Code = codes.DCM.ProcessingType  # type: ignore
    sampling_method: Code = codes.DCM.SamplingMethod  # type: ignore
    datetime_of_processing: Code = codes.DCM.DatetimeOfProcessing  # type: ignore
    processing_description: Code = codes.DCM.ProcessingStepDescription  # type: ignore
    parent_specimen_identifier: Code = codes.DCM.ParentSpecimenIdentifier  # type: ignore
    issuer_of_parent_specimen_identifier: Code = (
        codes.DCM.IssuerOfParentSpecimenIdentifier
    )  # type: ignore
    parent_specimen_type: Code = codes.DCM.ParentSpecimenType  # type: ignore
    specimen_type: Code = codes.SCT.SpecimenType  # type: ignore
    specimen_collection: Code = codes.SCT.SpecimenCollection  # type: ignore
    sampling_of_tissue_specimen: Code = codes.SCT.SamplingOfTissueSpecimen  # type: ignore
    sample_processing: Code = codes.SCT.SpecimenProcessing  # type: ignore
    staining: Code = codes.SCT.Staining  # type: ignore
    using_substance: Code = codes.SCT.UsingSubstance  # type: ignore
    fixative: Code = codes.SCT.TissueFixative  # type: ignore
    embedding: Code = codes.SCT.TissueEmbeddingMedium  # type: ignore
    location_frame_of_reference: Code = codes.DCM.PositionFrameOfReference  # type: ignore
    location_of_sampling_site: Code = codes.DCM.LocationOfSamplingSite  # type: ignore
    location_of_sampling_site_x: Code = codes.DCM.LocationOfSamplingSiteXOffset  # type: ignore
    location_of_sampling_site_y: Code = codes.DCM.LocationOfSamplingSiteYOffset  # type: ignore
    location_of_sampling_site_z: Code = codes.DCM.LocationOfSamplingSiteZOffset  # type: ignore
    location_of_specimen: Code = codes.DCM.LocationOfSpecimen  # type: ignore
    location_of_specimen_x: Code = codes.DCM.LocationOfSpecimenXOffset  # type: ignore
    location_of_specimen_y: Code = codes.DCM.LocationOfSpecimenYOffset  # type: ignore
    location_of_specimen_z: Code = codes.DCM.LocationOfSpecimenZOffset  # type: ignore
    visual_marking_of_specimen: Code = codes.DCM.VisualMarkingOfSpecimen  # type: ignore
    container: Code = codes.SCT.SpecimenContainer  # type: ignore
    receiving: Code = codes.SCT.SpecimenReceiving  # type: ignore
    storage: Code = codes.DCM.SpecimenStorage  # type: ignore


class SampleLocalizationDicomSchema(ItemSequenceDicomSchema[SampleLocalization]):
    reference = StringItemDicomField(allow_none=True)
    description = StringItemDicomField(allow_none=True)
    x = MeasurementItemDicomField(allow_none=True)
    y = MeasurementItemDicomField(allow_none=True)
    z = MeasurementItemDicomField(allow_none=True)
    visual_marking = StringItemDicomField(allow_none=True)

    @property
    def load_type(self):
        return SampleLocalization

    @property
    def item_fields(self) -> dict[str, ItemField]:
        return {
            "reference": ItemField(
                SampleCodes.location_frame_of_reference, (str,), False
            ),
            "description": ItemField(SampleCodes.location_of_specimen, (str,), False),
            "x": ItemField(SampleCodes.location_of_specimen_x, (Measurement,), False),
            "y": ItemField(SampleCodes.location_of_specimen_y, (Measurement,), False),
            "z": ItemField(SampleCodes.location_of_specimen_z, (Measurement,), False),
            "visual_marking": ItemField(
                SampleCodes.visual_marking_of_specimen, (str,), False
            ),
        }


class BasePreparationStepDicomSchema(ItemSequenceDicomSchema[LoadType]):
    _dump_only_fields = ["processing_type"]

    identifier = StringItemDicomField()
    issuer_of_identifier = StringItemDicomField(allow_none=True, load_default=None)
    date_time = DateTimeItemDicomField(allow_none=True, load_default=None)
    description = StringItemDicomField(allow_none=True, load_default=None)
    fixative = CodeItemDicomField(
        load_type=SpecimenFixativesCode, allow_none=True, load_default=None
    )
    embedding = CodeItemDicomField(
        load_type=SpecimenEmbeddingMediaCode, allow_none=True, load_default=None
    )
    processing = CodeItemDicomField(
        load_type=SpecimenPreparationStepsCode, allow_none=True, load_default=None
    )
    container = CodeItemDicomField(
        load_type=ContainerTypeCode, allow_none=True, load_default=None
    )
    specimen_type = CodeItemDicomField(
        load_type=AnatomicPathologySpecimenTypesCode,
        allow_none=True,
        load_default=None,
    )

    @property
    def item_fields(self) -> dict[str, ItemField]:
        """TID 8001 Specimen Preparation, excluding Collection, Sampling, and Specimen
        fields."""
        return {
            "identifier": ItemField(SampleCodes.identifier, (str,), False),
            "issuer_of_identifier": ItemField(
                SampleCodes.issuer_of_identifier, (str,), False
            ),
            "container": ItemField(SampleCodes.container, (Code,), False),
            "specimen_type": ItemField(SampleCodes.specimen_type, (Code,), False),
            "processing_type": ItemField(SampleCodes.processing_type, (Code,), False),
            "date_time": ItemField(
                SampleCodes.datetime_of_processing,
                (datetime.datetime,),
                False,
            ),
            "description": ItemField(SampleCodes.processing_description, (str,), False),
            "processing": ItemField(SampleCodes.processing_description, (Code,), False),
            "fixative": ItemField(SampleCodes.fixative, (Code,), False),
            "embedding": ItemField(SampleCodes.embedding, (Code,), False),
        }


class SamplingDicomSchema(BasePreparationStepDicomSchema[SamplingDicomModel]):
    processing_type = CodeItemDicomField(
        load_type=SpecimenPreparationProcedureCode,
        dump_default=SampleCodes.sampling_of_tissue_specimen,
        dump_only=True,
    )
    method = CodeItemDicomField(SpecimenSamplingProcedureCode)
    parent_specimen_identifier = StringItemDicomField()
    issuer_of_parent_specimen_identifier = StringItemDicomField(
        allow_none=True,
        load_default=None,
    )
    parent_specimen_type = CodeItemDicomField(AnatomicPathologySpecimenTypesCode)
    location_reference = StringItemDicomField(allow_none=True)
    location_description = StringItemDicomField(allow_none=True)
    location_x = MeasurementItemDicomField(allow_none=True)
    location_y = MeasurementItemDicomField(allow_none=True)
    location_z = MeasurementItemDicomField(allow_none=True)

    @property
    def load_type(self):
        return SamplingDicomModel

    @property
    def item_fields(self) -> dict[str, ItemField]:
        return {
            "identifier": ItemField(SampleCodes.identifier, (str,), False),
            "issuer_of_identifier": ItemField(
                SampleCodes.issuer_of_identifier, (str,), False
            ),
            "container": ItemField(SampleCodes.container, (Code,), False),
            "specimen_type": ItemField(SampleCodes.specimen_type, (Code,), False),
            "processing_type": ItemField(SampleCodes.processing_type, (Code,), False),
            "date_time": ItemField(
                SampleCodes.datetime_of_processing,
                (datetime.datetime,),
                False,
            ),
            "description": ItemField(SampleCodes.processing_description, (str,), False),
            "processing": ItemField(SampleCodes.processing_description, (Code,), False),
            "method": ItemField(SampleCodes.sampling_method, (Code,), False),
            "parent_specimen_identifier": ItemField(
                SampleCodes.parent_specimen_identifier,
                (str,),
                False,
            ),
            "issuer_of_parent_specimen_identifier": ItemField(
                SampleCodes.issuer_of_parent_specimen_identifier,
                (str,),
                False,
            ),
            "parent_specimen_type": ItemField(
                SampleCodes.parent_specimen_type, (Code,), False
            ),
            "location_reference": ItemField(
                SampleCodes.location_frame_of_reference, (str,), False
            ),
            "location_description": ItemField(
                SampleCodes.location_of_sampling_site, (str,), False
            ),
            "location_x": ItemField(
                SampleCodes.location_of_sampling_site_x, (Measurement,), False
            ),
            "location_y": ItemField(
                SampleCodes.location_of_sampling_site_y, (Measurement,), False
            ),
            "location_z": ItemField(
                SampleCodes.location_of_sampling_site_z, (Measurement,), False
            ),
            "fixative": ItemField(SampleCodes.fixative, (Code,), False),
            "embedding": ItemField(SampleCodes.embedding, (Code,), False),
        }


class CollectionDicomSchema(BasePreparationStepDicomSchema[CollectionDicomModel]):
    processing_type = CodeItemDicomField(
        load_type=SpecimenPreparationProcedureCode,
        dump_default=SampleCodes.specimen_collection,
        dump_only=True,
    )
    method = CodeItemDicomField(SpecimenCollectionProcedureCode)

    @property
    def load_type(self):
        return CollectionDicomModel

    @property
    def item_fields(self) -> dict[str, ItemField]:
        return {
            "identifier": ItemField(SampleCodes.identifier, (str,), False),
            "issuer_of_identifier": ItemField(
                SampleCodes.issuer_of_identifier, (str,), False
            ),
            "container": ItemField(SampleCodes.container, (Code,), False),
            "specimen_type": ItemField(SampleCodes.specimen_type, (Code,), False),
            "processing_type": ItemField(SampleCodes.processing_type, (Code,), False),
            "date_time": ItemField(
                SampleCodes.datetime_of_processing,
                (datetime.datetime,),
                False,
            ),
            "description": ItemField(SampleCodes.processing_description, (str,), False),
            "processing": ItemField(SampleCodes.processing_description, (Code,), False),
            "method": ItemField(SampleCodes.specimen_collection, (Code,), False),
            "fixative": ItemField(SampleCodes.fixative, (Code,), False),
            "embedding": ItemField(SampleCodes.embedding, (Code,), False),
        }


class ProcessingDicomSchema(BasePreparationStepDicomSchema[ProcessingDicomModel]):
    processing_type = CodeItemDicomField(
        load_type=SpecimenPreparationProcedureCode,
        dump_default=SampleCodes.sample_processing,
        dump_only=True,
    )
    processing = CodeItemDicomField(
        load_type=SpecimenPreparationStepsCode, allow_none=True, load_default=None
    )

    @property
    def load_type(self):
        return ProcessingDicomModel


class SubstanceItemDicomField(
    ContentItemDicomField[str | Sequence[SpecimenStainsCode] | None]
):
    _code_item = CodeItemDicomField(SpecimenStainsCode)
    _string_item = StringItemDicomField()

    # Makes the several items a substance is written as, rather than one. An
    # item field makes one item, so the items are made one at a time: a list
    # field cannot say this, as it is typed by one type where an item field
    # dumps the dataset of an item and loads what the item states.
    def _serialize(
        self,
        value: str | Sequence[SpecimenStainsCode] | None,
        attr,
        obj,
        **kwargs,
    ) -> Sequence[Dataset] | None:
        if value is None:
            return None
        if isinstance(value, str):
            item = self._string_item._serialize(value, attr, obj, **kwargs)
            return None if item is None else [item]
        items = (
            self._code_item._serialize(code, attr, obj, **kwargs) for code in value
        )
        return [item for item in items if item is not None]

    def _deserialize(
        self, value: Dataset | Sequence[Dataset], attr, data, **kwargs
    ) -> str | Sequence[SpecimenStainsCode] | None:
        datasets = [value] if isinstance(value, Dataset) else value
        first_value = datasets[0]

        if first_value.ValueType == "TEXT" or TextValueTag in first_value:
            return self._string_item.deserialize(first_value, attr, data, **kwargs)
        return [
            self._code_item.deserialize(dataset, attr, data, **kwargs)
            for dataset in datasets
        ]


class StainingDicomSchema(BasePreparationStepDicomSchema[StainingDicomModel]):
    processing_type = CodeItemDicomField(
        load_type=SpecimenPreparationProcedureCode,
        dump_default=SampleCodes.staining,
        dump_only=True,
    )
    substances = SubstanceItemDicomField()

    @property
    def load_type(self):
        return StainingDicomModel

    @property
    def item_fields(self) -> dict[str, ItemField]:
        return {
            "identifier": ItemField(SampleCodes.identifier, (str,), False),
            "issuer_of_identifier": ItemField(
                SampleCodes.issuer_of_identifier, (str,), False
            ),
            "container": ItemField(SampleCodes.container, (Code,), False),
            "specimen_type": ItemField(SampleCodes.specimen_type, (Code,), False),
            "processing_type": ItemField(SampleCodes.processing_type, (Code,), False),
            "date_time": ItemField(
                SampleCodes.datetime_of_processing,
                (datetime.datetime,),
                False,
            ),
            "description": ItemField(SampleCodes.processing_description, (str,), False),
            "processing": ItemField(SampleCodes.processing_description, (Code,), False),
            "substances": ItemField(SampleCodes.using_substance, (str, Code), True),
            "fixative": ItemField(SampleCodes.fixative, (Code,), False),
            "embedding": ItemField(SampleCodes.embedding, (Code,), False),
        }


class ReceivingDicomSchema(BasePreparationStepDicomSchema[ReceivingDicomModel]):
    processing_type = CodeItemDicomField(
        load_type=SpecimenPreparationProcedureCode,
        dump_default=SampleCodes.receiving,
        dump_only=True,
    )

    @property
    def load_type(self):
        return ReceivingDicomModel


class StorageDicomSchema(BasePreparationStepDicomSchema[StorageDicomModel]):
    processing_type = CodeItemDicomField(
        load_type=SpecimenPreparationProcedureCode,
        dump_default=SampleCodes.storage,
        dump_only=True,
    )

    @property
    def load_type(self):
        return StorageDicomModel


class PreparationStepDicomField(
    DatasetDicomField[SpecimenPreparationStepDicomModel | None]
):
    """Field for a preparation step, whichever kind of step it is.

    A step is written as the content items of the sequence named by the data
    key, by the schema for the kind of step it is. The data key is the
    attribute of the item this makes, not one of the dataset the item is in.
    """

    PREPARATION_STEP_SCHEMAS: tuple[type[ItemSequenceDicomSchema], ...] = (
        SamplingDicomSchema,
        CollectionDicomSchema,
        ProcessingDicomSchema,
        StainingDicomSchema,
        ReceivingDicomSchema,
        StorageDicomSchema,
    )
    """The schemas a preparation step is written and read by, one for each kind of
    step. What kind a schema is for is the schema's to say, so the way to a schema
    from a step, and from the processing type a step states, is worked out from the
    schemas rather than stated again here."""

    _schema_by_step: dict[
        type[SpecimenPreparationStepDicomModel], type[ItemSequenceDicomSchema]
    ] = {schema().load_type: schema for schema in PREPARATION_STEP_SCHEMAS}

    _schema_by_processing_type: dict[Code, type[ItemSequenceDicomSchema]] = {
        schema().fields["processing_type"].dump_default: schema
        for schema in PREPARATION_STEP_SCHEMAS
    }

    def __init__(self, data_key: str, **kwargs):
        self._data_key = data_key
        super().__init__(data_key=data_key, **kwargs)

    def _serialize(
        self,
        value: SpecimenPreparationStepDicomModel | None,
        attr: str | None,
        obj: Any,
        **kwargs,
    ) -> Dataset | None:
        """The item holding the content items the step is written as."""
        if value is None:
            return None
        dataset = Dataset()
        self._set_attribute(dataset, self._data_key, self._subschema_dump(value))
        return dataset

    def _deserialize(
        self, value: Dataset, attr: str | None, data: Any, **kwargs
    ) -> SpecimenPreparationStepDicomModel | None:
        """The step the content items of the item hold."""
        return self._subschema_load(getattr(value, self._data_key))

    def _subschema_load(
        self, sequence: Iterable[Dataset]
    ) -> SpecimenPreparationStepDicomModel | None:
        """Select a schema and load and return step using the schema."""
        try:
            try:
                processing_type: Code = next(
                    dataset_to_code(item.ConceptCodeSequence[0])
                    for item in sequence
                    if dataset_to_code(item.ConceptNameCodeSequence[0])
                    == SampleCodes.processing_type
                )
            except StopIteration:
                raise ValidationError(
                    "Failed to load processing step due to missing processing type."
                ) from None
            try:
                schema = self._schema_by_processing_type[processing_type]
            except KeyError:
                raise ValidationError(
                    "Failed to load processing step due to unknown "
                    f"processing type {processing_type}."
                ) from None
            loaded = schema().load(sequence, many=False)
        except ValidationError as exception:
            error = "Failed to load processing step due to validation error."
            if get_settings().ignore_specimen_preparation_step_on_validation_error:
                logger.warning(error, exc_info=True)
                return None
            raise ValidationError(
                error + " Check the processing step for errors or set "
                "`settings.ignore_specimen_preparation_step_on_validation_error` "
                "to `True` to ignore this step."
            ) from exception
        assert isinstance(loaded, SpecimenPreparationStepDicomModel)
        return loaded

    def _subschema_dump(self, step: SpecimenPreparationStepDicomModel) -> list[Dataset]:
        """Select a schema and dump the step using the schema."""
        schema = self._schema_by_step[type(step)]
        dumped = schema().dump(step, many=False)
        assert isinstance(dumped, list)
        return dumped


class SpecimenDescriptionDicomSchema(DicomSchema[SpecimenDescriptionDicomModel]):
    identifier = StringDicomField(
        value_representation=VR.LO, data_key="SpecimenIdentifier"
    )
    uid = UidDicomField(data_key="SpecimenUID", dump_required=True)
    localization = fields.Nested(
        SampleLocalizationDicomSchema(),
        data_key="SpecimenLocalizationContentItemSequence",
        allow_none=True,
    )
    issuer_of_identifier = IssuerOfIdentifierDicomField(
        data_key="IssuerOfTheSpecimenIdentifierSequence", allow_none=True
    )
    steps = DefaultOnValidationExceptionField(
        fields.List(
            PreparationStepDicomField(
                data_key="SpecimenPreparationStepContentItemSequence"
            )
        ),
        data_key="SpecimenPreparationSequence",
        load_default=[],
    )
    anatomical_sites = ListDicomField(
        CodeDicomField(Code),
        data_key="PrimaryAnatomicStructureSequence",
        load_default=[],
    )
    specimen_type = SingleCodeSequenceDicomField(
        load_type=AnatomicPathologySpecimenTypesCode,
        data_key="SpecimenTypeCodeSequence",
        dump_default=AnatomicPathologySpecimenTypesCode("Slide"),
        dump_only=True,
    )
    short_description = StringDicomField(
        value_representation=VR.LO, data_key="SpecimenShortDescription", allow_none=True
    )
    detailed_description = StringDicomField(
        value_representation=VR.UT,
        data_key="SpecimenDetailedDescription",
        allow_none=True,
    )

    @property
    def load_type(self):
        return SpecimenDescriptionDicomModel

    @post_load
    def post_load(
        self, data: dict[str, Any], **kwargs
    ) -> SpecimenDescriptionDicomModel:
        """Remove None values from steps before loading to object."""
        data["steps"] = [step for step in data["steps"] if step is not None]
        return super().post_load(data, **kwargs)
