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

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Type,
    Union,
)

from marshmallow import Schema, fields, pre_dump

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
    BaseSpecimen,
    Collection,
    Embedding,
    Fixation,
    PreparationStep,
    Processing,
    Receiving,
    Sample,
    SampledSpecimen,
    SampleLocalization,
    Sampling,
    SamplingLocation,
    SlideSample,
    Specimen,
    SpecimenIdentifier,
    Staining,
    Storage,
    UnknownSampling,
)
from wsidicom.metadata.schema.common import DataclassLoadingSchema, LoadingSchema
from wsidicom.metadata.schema.json.fields import (
    CodeJsonField,
    JsonFieldFactory,
    MeasurementJsonField,
    SpecimenIdentifierJsonField,
    UidJsonField,
)
from wsidicom.metadata.schema.json.sample.model import (
    BaseSpecimenJsonModel,
    PreparationAction,
    SampleJsonModel,
    SamplingConstraintJsonModel,
    SamplingJsonModel,
    SlideSampleJsonModel,
    SpecimenJsonModel,
)
from wsidicom.metadata.schema.json.sample.parser import SpecimenJsonParser


class SamplingLocationJsonSchema(LoadingSchema[SamplingLocation]):
    reference = fields.String(allow_none=True)
    description = fields.String(allow_none=True)
    x = MeasurementJsonField(allow_none=True)
    y = MeasurementJsonField(allow_none=True)
    z = MeasurementJsonField(allow_none=True)

    @property
    def load_type(self) -> Type[SamplingLocation]:
        return SamplingLocation


class SampleLocalizationJsonSchema(
    SamplingLocationJsonSchema, LoadingSchema[SamplingLocation]
):
    visual_marking = fields.String(allow_none=True)

    @property
    def load_type(self) -> Type[SamplingLocation]:
        return SampleLocalization


class SamplingConstraintJsonSchema(LoadingSchema[SamplingConstraintJsonModel]):
    """Schema for serializing and deserializing a `SamplingConstraintJsonModel`."""

    identifier = SpecimenIdentifierJsonField()
    sampling_step_index = fields.Integer()

    @property
    def load_type(self) -> Type[SamplingConstraintJsonModel]:
        return SamplingConstraintJsonModel

    @pre_dump
    def dump_simple(self, sampling: Sampling, **kwargs):
        return SamplingConstraintJsonModel.to_json_model(sampling)


class SamplingJsonSchema(DataclassLoadingSchema[SamplingJsonModel]):
    action = fields.Constant(PreparationAction.SAMPLING.value, dump_only=True)
    method = JsonFieldFactory.concept_code(SpecimenSamplingProcedureCode)(
        allow_none=True
    )
    sampling_constraints = fields.List(
        fields.Nested(SamplingConstraintJsonSchema, allow_none=True), allow_none=True
    )
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)
    location = fields.Nested(SamplingLocationJsonSchema, allow_none=True)

    @property
    def load_type(self) -> Type[SamplingJsonModel]:
        return SamplingJsonModel


class CollectionJsonSchema(DataclassLoadingSchema[Collection]):
    action = fields.Constant(PreparationAction.COLLECTION.value, dump_only=True)
    method = JsonFieldFactory.concept_code(SpecimenCollectionProcedureCode)()
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)

    @property
    def load_type(self) -> Type[Collection]:
        return Collection


class ProcessingJsonSchema(DataclassLoadingSchema[Processing]):
    action = fields.Constant(PreparationAction.PROCESSING.value, dump_only=True)
    method = JsonFieldFactory.concept_code(SpecimenPreparationStepsCode)(
        allow_none=True
    )
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)

    @property
    def load_type(self) -> Type[Processing]:
        return Processing


class EmbeddingJsonSchema(DataclassLoadingSchema[Embedding]):
    action = fields.Constant(PreparationAction.EMBEDDING.value, dump_only=True)
    medium = JsonFieldFactory.concept_code(SpecimenEmbeddingMediaCode)()
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)

    @property
    def load_type(self) -> Type[Embedding]:
        return Embedding


class FixationJsonSchema(DataclassLoadingSchema[Fixation]):
    action = fields.Constant(PreparationAction.FIXATION.value, dump_only=True)
    fixative = JsonFieldFactory.concept_code(SpecimenFixativesCode)()
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)

    @property
    def load_type(self) -> Type[Fixation]:
        return Fixation


class ReceivingJsonSchema(DataclassLoadingSchema[Receiving]):
    action = fields.Constant(PreparationAction.RECEIVING.value, dump_only=True)
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)

    @property
    def load_type(self) -> Type[Receiving]:
        return Receiving


class StorageJsonSchema(DataclassLoadingSchema[Storage]):
    action = fields.Constant(PreparationAction.STORAGE.value, dump_only=True)
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)

    @property
    def load_type(self) -> Type[Storage]:
        return Storage


class PreparationStepJsonSchema(Schema):
    """Schema to use to serialize/deserialize preparation steps."""

    """Mapping step type to schema."""
    _type_to_schema_mapping: Dict[Type[PreparationStep], Type[Schema]] = {
        Sampling: SamplingJsonSchema,
        UnknownSampling: SamplingJsonSchema,
        Collection: CollectionJsonSchema,
        Processing: ProcessingJsonSchema,
        Embedding: EmbeddingJsonSchema,
        Fixation: FixationJsonSchema,
        Receiving: ReceivingJsonSchema,
        Storage: StorageJsonSchema,
    }

    """Mapping key in serialized step to schema."""
    _action_to_schema_mapping: Dict[PreparationAction, Type[Schema]] = {
        PreparationAction.SAMPLING: SamplingJsonSchema,
        PreparationAction.COLLECTION: CollectionJsonSchema,
        PreparationAction.PROCESSING: ProcessingJsonSchema,
        PreparationAction.EMBEDDING: EmbeddingJsonSchema,
        PreparationAction.FIXATION: FixationJsonSchema,
        PreparationAction.RECEIVING: ReceivingJsonSchema,
        PreparationAction.STORAGE: StorageJsonSchema,
    }

    def dump(
        self,
        data: Union[PreparationStep, Iterable[PreparationStep]],
        **kwargs,
    ):
        if isinstance(data, PreparationStep):
            return self._subschema_dump(data)
        return [self._subschema_dump(item) for item in data]

    def load(
        self,
        data: Union[Dict[str, Any], Iterable[Dict[str, Any]]],
        **kwargs,
    ):
        if isinstance(data, dict):
            return self._subschema_load(data)  # type: ignore
        return [self._subschema_load(step) for step in data]

    def _subschema_load(
        self, step: Dict[str, Any]
    ) -> Union[PreparationStep, SamplingJsonModel]:
        """Select a schema and load and return step using the schema."""
        try:
            action = PreparationAction(step.pop("action"))
            schema = self._action_to_schema_mapping[action]
        except KeyError:
            raise NotImplementedError()
        loaded = schema().load(step, many=False)
        assert isinstance(loaded, (PreparationStep, SamplingJsonModel))
        return loaded

    def _subschema_dump(self, step: PreparationStep):
        """Select a schema and dump the step using the schema."""
        schema = self._type_to_schema_mapping[type(step)]
        return schema().dump(step, many=False)


class StainingJsonSchema(DataclassLoadingSchema[Staining]):
    substances = fields.List(
        JsonFieldFactory.concept_code(SpecimenStainsCode)(), allow_none=True
    )
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)

    @property
    def load_type(self) -> Type[Staining]:
        return Staining


class SpecimenJsonSchema(LoadingSchema[Specimen]):
    """Schema for extracted specimen that has not been sampled from other specimen."""

    identifier = SpecimenIdentifierJsonField()
    steps = fields.List(fields.Nested(PreparationStepJsonSchema()))
    type = JsonFieldFactory.concept_code(AnatomicPathologySpecimenTypesCode)(
        allow_none=True, load_default=None
    )
    container = JsonFieldFactory.concept_code(ContainerTypeCode)(
        allow_none=True, load_default=None
    )

    @property
    def load_type(self) -> Type[SpecimenJsonModel]:
        return SpecimenJsonModel


class SampleJsonSchema(LoadingSchema[SampleJsonModel]):
    """Schema for sampled specimen."""

    identifier = SpecimenIdentifierJsonField()
    steps = fields.List(fields.Nested(PreparationStepJsonSchema()))
    sampled_from = fields.List(fields.Nested(SamplingConstraintJsonSchema))
    type = JsonFieldFactory.concept_code(AnatomicPathologySpecimenTypesCode)(
        allow_none=True, load_default=None
    )
    container = JsonFieldFactory.concept_code(ContainerTypeCode)(
        allow_none=True, load_default=None
    )

    @property
    def load_type(self) -> Type[SampleJsonModel]:
        return SampleJsonModel


class SlideSampleJsonSchema(LoadingSchema[SlideSampleJsonModel]):
    """Schema for sampled specimen on a slide."""

    identifier = SpecimenIdentifierJsonField()
    steps = fields.List(fields.Nested(PreparationStepJsonSchema()))
    anatomical_sites = fields.List(CodeJsonField(), allow_none=True)
    sampled_from = fields.Nested(SamplingConstraintJsonSchema)
    uid = UidJsonField(allow_none=True)
    localization = fields.Nested(
        SampleLocalizationJsonSchema, allow_none=True, load_default=None
    )
    short_description = fields.String(allow_none=True, load_default=None)
    detailed_description = fields.String(allow_none=True, load_default=None)

    @property
    def load_type(self) -> Type[SlideSampleJsonModel]:
        return SlideSampleJsonModel


class BaseSpecimenJsonSchema(Schema):
    """Schema to use to serialize/deserialize specimens."""

    """Mapping specimen type to schema."""
    _type_to_schema_mapping: Dict[Type[BaseSpecimen], Type[Schema]] = {
        Specimen: SpecimenJsonSchema,
        Sample: SampleJsonSchema,
        SlideSample: SlideSampleJsonSchema,
    }

    """Mapping key in serialized specimen to schema."""
    _key_to_schema_mapping: Dict[str, Type[Schema]] = {
        "anatomical_sites": SlideSampleJsonSchema,
        "sampled_from": SampleJsonSchema,
        "type": SpecimenJsonSchema,
    }

    def dump(
        self,
        specimens: Union[BaseSpecimen, Iterable[BaseSpecimen]],
        **kwargs,
    ):
        if isinstance(specimens, BaseSpecimen):
            specimens = [specimens]

        all_specimens: Dict[Union[str, SpecimenIdentifier], BaseSpecimen] = {}
        for specimen in specimens:
            if isinstance(specimen, SampledSpecimen):
                all_specimens.update(self._get_samplings(specimen))

        return [self._subschema_dump(specimen) for specimen in all_specimens.values()]

    def load(
        self,
        data: Union[Dict[str, Any], Iterable[Dict[str, Any]]],
        **kwargs,
    ) -> List[BaseSpecimen]:
        """Load serialized specimen or list of specimen as `Specimen`."""
        if isinstance(data, dict):
            loaded = [self._subschema_load(data)]  # type: ignore
        else:
            loaded = [self._subschema_load(item) for item in data]
        specimen_parser = SpecimenJsonParser(loaded)
        return specimen_parser.create_specimens()

    @classmethod
    def _get_samplings(
        cls, sample: SampledSpecimen
    ) -> Dict[Union[str, SpecimenIdentifier], BaseSpecimen]:
        """Return a dictionary containing this specimen and all recursive sampled
        specimens."""
        samplings: Dict[Union[str, SpecimenIdentifier], BaseSpecimen] = {
            sample.identifier: sample
        }
        for sampling in sample.sampled_from_list:
            if not isinstance(sampling.specimen, SampledSpecimen):
                samplings.update({sampling.specimen.identifier: sampling.specimen})
            else:
                samplings.update(cls._get_samplings(sampling.specimen))
        return samplings

    @classmethod
    def _subschema_select(cls, specimen: Dict[str, Any]) -> Type[Schema]:
        try:
            return next(
                schema
                for key, schema in cls._key_to_schema_mapping.items()
                if key in specimen
            )
        except StopIteration:
            raise NotImplementedError()

    def _subschema_load(self, specimen: Dict[str, Any]) -> BaseSpecimenJsonModel:
        """Select a schema and load and return specimen using the schema."""
        schema = self._subschema_select(specimen)
        loaded = schema().load(specimen, many=False)
        assert isinstance(loaded, BaseSpecimenJsonModel)
        return loaded

    def _subschema_dump(self, specimen: BaseSpecimen) -> Dict[str, Any]:
        """Select a schema and dump the specimen using the schema."""
        schema = self._type_to_schema_mapping[type(specimen)]
        dumped = schema().dump(specimen)
        assert isinstance(dumped, dict)
        return dumped
