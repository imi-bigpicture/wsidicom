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
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
)

import marshmallow
from pydicom import Dataset
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

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
from wsidicom.metadata.dicom_schema.fields import (
    CodeDicomField,
    CodeItemDicomField,
    DateTimeItemDicomField,
    MeasurementtemDicomField,
    IssuerOfIdentifierDicomField,
    SingleCodeSequenceField,
    SingleItemSequenceDicomField,
    StringDicomField,
    StringItemDicomField,
    StringOrCodeItemDicomField,
)
from wsidicom.metadata.dicom_schema.sample.model import (
    CollectionDicomModel,
    ReceivingDicomModel,
    SpecimenPreparationStepDicomModel,
    ProcessingDicomModel,
    SamplingDicomModel,
    SpecimenDescriptionDicomModel,
    StainingDicomModel,
    StorageDicomModel,
)
from wsidicom.metadata.dicom_schema.schema import (
    DicomSchema,
    ItemField,
    ItemSequenceDicomSchema,
    LoadType,
)
from wsidicom.metadata.sample import Measurement, SpecimenLocalization


class SampleCodes:
    identifier: Code = codes.DCM.SpecimenIdentifier  # type: ignore
    issuer_of_identifier: Code = codes.DCM.IssuerOfSpecimenIdentifier  # type: ignore
    processing_type: Code = codes.DCM.ProcessingType  # type: ignore
    sampling_method: Code = codes.SCT.SamplingOfTissueSpecimen  # type: ignore
    datetime_of_processing: Code = codes.DCM.DatetimeOfProcessing  # type: ignore
    processing_description: Code = codes.DCM.ProcessingStepDescription  # type: ignore
    parent_specimen_identifier: Code = codes.DCM.ParentSpecimenIdentifier  # type: ignore
    issuer_of_parent_specimen_identifier: Code = codes.DCM.IssuerOfParentSpecimenIdentifier  # type: ignore
    parent_specimen_type: Code = codes.DCM.ParentSpecimenType  # type: ignore
    specimen_collection: Code = codes.SCT.SpecimenCollection  # type: ignore
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


class SpecimenLocalizationDicomSchema(ItemSequenceDicomSchema[SpecimenLocalization]):
    reference = StringItemDicomField(allow_none=True)
    description = StringItemDicomField(allow_none=True)
    x = MeasurementtemDicomField(allow_none=True)
    y = MeasurementtemDicomField(allow_none=True)
    z = MeasurementtemDicomField(allow_none=True)
    visual_marking = StringItemDicomField(allow_none=True)

    @property
    def load_type(self):
        return SpecimenLocalization

    @property
    def item_fields(self) -> Dict[str, ItemField]:
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


class SamplingDicomSchema(BasePreparationStepDicomSchema[SamplingDicomModel]):
    processing_type = CodeItemDicomField(
        load_type=SpecimenPreparationProcedureCode,
        dump_default=SampleCodes.sampling_method,
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
    location_x = MeasurementtemDicomField(allow_none=True)
    location_y = MeasurementtemDicomField(allow_none=True)
    location_z = MeasurementtemDicomField(allow_none=True)
    container = CodeItemDicomField(
        load_type=ContainerTypeCode, allow_none=True, load_default=None
    )

    @property
    def load_type(self):
        return SamplingDicomModel

    @property
    def item_fields(self) -> Dict[str, ItemField]:
        return {
            "identifier": ItemField(SampleCodes.identifier, (str,), False),
            "issuer_of_identifier": ItemField(
                SampleCodes.issuer_of_identifier, (str,), False
            ),
            "container": ItemField(SampleCodes.container, (Code,), False),
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
    container = CodeItemDicomField(
        load_type=ContainerTypeCode, allow_none=True, load_default=None
    )

    @property
    def load_type(self):
        return CollectionDicomModel

    @property
    def item_fields(self) -> Dict[str, ItemField]:
        return {
            "identifier": ItemField(SampleCodes.identifier, (str,), False),
            "issuer_of_identifier": ItemField(
                SampleCodes.issuer_of_identifier, (str,), False
            ),
            "container": ItemField(SampleCodes.container, (Code,), False),
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

    @property
    def item_fields(self) -> Dict[str, ItemField]:
        return {
            "identifier": ItemField(SampleCodes.identifier, (str,), False),
            "issuer_of_identifier": ItemField(
                SampleCodes.issuer_of_identifier, (str,), False
            ),
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


class StainingDicomSchema(BasePreparationStepDicomSchema[StainingDicomModel]):
    processing_type = CodeItemDicomField(
        load_type=SpecimenPreparationProcedureCode,
        dump_default=SampleCodes.staining,
        dump_only=True,
    )
    substances = marshmallow.fields.List(StringOrCodeItemDicomField(SpecimenStainsCode))

    @property
    def load_type(self):
        return StainingDicomModel

    @property
    def item_fields(self) -> Dict[str, ItemField]:
        return {
            "identifier": ItemField(SampleCodes.identifier, (str,), False),
            "issuer_of_identifier": ItemField(
                SampleCodes.issuer_of_identifier, (str,), False
            ),
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

    @property
    def item_fields(self) -> Dict[str, ItemField]:
        return {
            "identifier": ItemField(SampleCodes.identifier, (str,), False),
            "issuer_of_identifier": ItemField(
                SampleCodes.issuer_of_identifier, (str,), False
            ),
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


class StorageDicomSchema(BasePreparationStepDicomSchema[StorageDicomModel]):
    processing_type = CodeItemDicomField(
        load_type=SpecimenPreparationProcedureCode,
        dump_default=SampleCodes.storage,
        dump_only=True,
    )

    @property
    def load_type(self):
        return StorageDicomModel

    @property
    def item_fields(self) -> Dict[str, ItemField]:
        return {
            "identifier": ItemField(SampleCodes.identifier, (str,), False),
            "issuer_of_identifier": ItemField(
                SampleCodes.issuer_of_identifier, (str,), False
            ),
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


class PreparationStepDicomField(marshmallow.fields.Field):
    """Mapping step type to schema."""

    _type_to_schema_mapping: Dict[
        Type[SpecimenPreparationStepDicomModel], Type[ItemSequenceDicomSchema]
    ] = {
        SamplingDicomModel: SamplingDicomSchema,
        CollectionDicomModel: CollectionDicomSchema,
        ProcessingDicomModel: ProcessingDicomSchema,
        StainingDicomModel: StainingDicomSchema,
        ReceivingDicomModel: ReceivingDicomSchema,
        StorageDicomModel: StorageDicomSchema,
    }

    """Mapping key in serialized step to schema."""
    _processing_type_to_schema_mapping: Dict[Code, Type[ItemSequenceDicomSchema]] = {
        SampleCodes.sampling_method: SamplingDicomSchema,
        SampleCodes.specimen_collection: CollectionDicomSchema,
        SampleCodes.sample_processing: ProcessingDicomSchema,
        SampleCodes.staining: StainingDicomSchema,
        SampleCodes.receiving: ReceivingDicomSchema,
        SampleCodes.storage: StorageDicomSchema,
    }

    def _serialize(
        self,
        step: SpecimenPreparationStepDicomModel,
        attr: Optional[str],
        obj: Any,
        **kwargs
    ) -> List[Dataset]:
        """Serialize step to dataset."""
        return self._subschema_dump(step)

    def _deserialize(
        self, sequence: Iterable[Dataset], attr: Optional[str], data: Any, **kwargs
    ) -> SpecimenPreparationStepDicomModel:
        """Deserialize step from dataset."""
        return self._subschema_load(sequence)

    def _subschema_load(
        self, sequence: Iterable[Dataset]
    ) -> SpecimenPreparationStepDicomModel:
        """Select a schema and load and return step using the schema."""
        try:
            processing_type: Code = next(
                dataset_to_code(item.ConceptCodeSequence[0])
                for item in sequence
                if dataset_to_code(item.ConceptNameCodeSequence[0])
                == SampleCodes.processing_type
            )
            schema = self._processing_type_to_schema_mapping[processing_type]
        except (StopIteration, KeyError):
            raise NotImplementedError()
        loaded = schema().load(sequence, many=False)
        assert isinstance(loaded, SpecimenPreparationStepDicomModel)
        return loaded

    def _subschema_dump(self, step: SpecimenPreparationStepDicomModel) -> List[Dataset]:
        """Select a schema and dump the step using the schema."""
        schema = self._type_to_schema_mapping[type(step)]
        dumped = schema().dump(step, many=False)
        assert isinstance(dumped, list)
        return dumped


class SpecimenDescriptionDicomSchema(DicomSchema[SpecimenDescriptionDicomModel]):
    identifier = StringDicomField(data_key="SpecimenIdentifier")
    uid = StringDicomField(data_key="SpecimenUID")
    localization = marshmallow.fields.Nested(
        SpecimenLocalizationDicomSchema(),
        data_key="SpecimenLocalizationContentItemSequence",
        allow_none=True,
    )
    issuer_of_identifier = IssuerOfIdentifierDicomField(
        data_key="IssuerOfTheSpecimenIdentifierSequence", allow_none=True
    )
    steps = marshmallow.fields.List(
        SingleItemSequenceDicomField(
            PreparationStepDicomField(),
            data_key="SpecimenPreparationStepContentItemSequence",
        ),
        data_key="SpecimenPreparationStepContentItemSequence",
        load_default=[],
    )
    anatomical_sites = marshmallow.fields.List(
        CodeDicomField(Code),
        data_key="PrimaryAnatomicStructureSequence",
        load_default=[],
    )
    specimen_type = SingleCodeSequenceField(
        load_type=AnatomicPathologySpecimenTypesCode,
        data_key="SpecimenTypeCodeSequence",
        dump_default=AnatomicPathologySpecimenTypesCode("Slide"),
        dump_only=True,
    )
    short_description = StringDicomField(
        data_key="SpecimenShortDescription", allow_none=True
    )
    detailed_description = StringDicomField(
        data_key="SpecimenDetailedDescription", allow_none=True
    )

    @property
    def load_type(self):
        return SpecimenDescriptionDicomModel

    @marshmallow.post_load
    def post_load(self, data, **kwargs):
        print(data.keys())
        return super().post_load(data, **kwargs)
