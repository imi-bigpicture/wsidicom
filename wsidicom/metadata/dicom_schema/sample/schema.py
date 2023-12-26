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
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import marshmallow
from marshmallow import post_dump, pre_load
from pydicom import Dataset
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code

from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    SpecimenCollectionProcedureCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenPreparationProcedureCode,
    SpecimenPreparationStepsCode,
    SpecimenSamplingProcedureCode,
    SpecimenStainsCode,
)
from wsidicom.metadata.dicom_schema.fields import (
    CodeDicomField,
    CodeItemDicomField,
    DateTimeItemDicomField,
    IssuerOfIdentifierField,
    SingleCodeSequenceField,
    SingleItemSequenceDicomField,
    StringDicomField,
    StringItemDicomField,
    StringOrCodeItemDicomField,
)
from wsidicom.metadata.dicom_schema.sample.model import (
    CollectionDicomModel,
    SpecimenPreparationStepDicomModel,
    ProcessingDicomModel,
    SamplingDicomModel,
    SpecimenDescriptionDicomModel,
    StainingDicomModel,
)
from wsidicom.metadata.dicom_schema.schema import DicomSchema, LoadType


def dataset_to_code(dataset: Dataset) -> Code:
    return Code(
        dataset.CodeValue,
        dataset.CodingSchemeDesignator,
        dataset.CodeMeaning,
        dataset.get("CodingSchemeVersion", None),
    )


def dataset_to_type(dataset: Dataset) -> Type:
    if "ConceptCodeSequence" in dataset:
        return Code
    if "TextValue" in dataset:
        return str
    if "DateTime" in dataset:
        return datetime.datetime
    if "NumericValue" in dataset:
        return float
    raise NotImplementedError()


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


@dataclass
class ItemField:
    name: Code
    value_types: Tuple[Type, ...]
    many: bool


class BasePreparationStepDicomSchema(DicomSchema[LoadType]):
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

    @property
    @abstractmethod
    def item_fields(self) -> Dict[str, ItemField]:
        """Describe the fields in the schema.

        Fields should be ordered as in TID 8001. The key is the python name of the
        field, and the value is a ItemField with the DICOM code name of the field,
        the allowed value types (tuple of one or more types), and if the field can
        hold multiple values (e.g. is a list)."""
        raise NotImplementedError()

    @post_dump
    def post_dump(
        self, data: Dict[str, Union[Dataset, Sequence[Dataset]]], many: bool, **kwargs
    ) -> Dataset:
        """Format the preparation step content items into sequence in a dataset."""
        dataset = Dataset()
        dataset.SpecimenPreparationStepContentItemSequence = [
            self._name_item(flatten_item, name)
            for item, name in [
                (data[key], description.name)
                for key, description in self.item_fields.items()
            ]
            if item is not None
            for flatten_item in ([item] if isinstance(item, Dataset) else item)
        ]
        return dataset

    @pre_load
    def pre_load(self, dataset: Dataset, many: bool, **kwargs) -> Dict[str, Any]:
        """Parse the preparation step content items from a dataset into a dictionary."""
        data = {
            key: self._get_item(dataset, description)
            for key, description in self.item_fields.items()
        }
        data.pop("processing_type")
        return data

    @staticmethod
    def _name_item(item: Dataset, name: Code):
        """Add concept name code sequence to dataset."""
        name_dataset = Dataset()
        name_dataset.CodeValue = name.value
        name_dataset.CodingSchemeDesignator = name.scheme_designator
        name_dataset.CodeMeaning = name.meaning
        name_dataset.CodingSchemeVersion = name.scheme_version
        item.ConceptNameCodeSequence = [name_dataset]
        return item

    def _get_item(
        self, dataset: Dataset, field: ItemField
    ) -> Optional[Union[Dataset, List[Dataset]]]:
        """Get item dataset from dataset specimen preparation step content item sequence.

        Parameters
        ----------
        dataset: Dataset
            Dataset to get item from.
        field:
            Description of the field to get.

        Returns
        -------
        Optional[Union[Dataset, List[Dataset]]]
            Item dataset or datasets or None if not found.
        """
        items = (
            item
            for item in dataset.SpecimenPreparationStepContentItemSequence
            if dataset_to_code(item.ConceptNameCodeSequence[0]) == field.name
            and dataset_to_type(item) in field.value_types
        )
        if field.many:
            return list(items)
        return next(items, None)


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


class PreparationStepDicomSchema(marshmallow.Schema):
    """Mapping step type to schema."""

    _type_to_schema_mapping: Dict[
        Type[SpecimenPreparationStepDicomModel], Type[DicomSchema]
    ] = {
        SamplingDicomModel: SamplingDicomSchema,
        CollectionDicomModel: CollectionDicomSchema,
        ProcessingDicomModel: ProcessingDicomSchema,
        StainingDicomModel: StainingDicomSchema,
    }

    """Mapping key in serialized step to schema."""
    _processing_type_to_schema_mapping: Dict[Code, Type[DicomSchema]] = {
        SampleCodes.sampling_method: SamplingDicomSchema,
        SampleCodes.specimen_collection: CollectionDicomSchema,
        SampleCodes.sample_processing: ProcessingDicomSchema,
        SampleCodes.staining: StainingDicomSchema,
    }

    def dump(self, step: SpecimenPreparationStepDicomModel, **kwargs) -> Dataset:
        return self._subschema_dump(step)

    def load(self, dataset: Dataset, **kwargs):
        return self._subschema_load(dataset)

    def _subschema_load(self, dataset: Dataset) -> SpecimenPreparationStepDicomModel:
        """Select a schema and load and return step using the schema."""
        try:
            processing_type: Code = next(
                dataset_to_code(item.ConceptCodeSequence[0])
                for item in dataset.SpecimenPreparationStepContentItemSequence
                if dataset_to_code(item.ConceptNameCodeSequence[0])
                == SampleCodes.processing_type
            )
            schema = self._processing_type_to_schema_mapping[processing_type]
        except (StopIteration, KeyError):
            raise NotImplementedError()
        loaded = schema().load(dataset, many=False)
        assert isinstance(loaded, SpecimenPreparationStepDicomModel)
        return loaded

    def _subschema_dump(self, step: SpecimenPreparationStepDicomModel) -> Dataset:
        """Select a schema and dump the step using the schema."""
        schema = self._type_to_schema_mapping[type(step)]
        dumped = schema().dump(step, many=False)
        assert isinstance(dumped, Dataset)
        return dumped


class SpecimenDescriptionDicomSchema(DicomSchema[SpecimenDescriptionDicomModel]):
    identifier = StringDicomField(data_key="SpecimenIdentifier")
    uid = StringDicomField(data_key="SpecimenUID")
    # specimen_location:
    issuer_of_identifier = IssuerOfIdentifierField(
        data_key="IssuerOfTheSpecimenIdentifierSequence", allow_none=True
    )
    steps = marshmallow.fields.List(
        marshmallow.fields.Nested(PreparationStepDicomSchema()),
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
