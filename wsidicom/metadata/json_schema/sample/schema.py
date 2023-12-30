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

"""
Schemata for serializing/deserializing Specimen.

To avoid duplication in the nested `sampled_from` attribute, the `SampledFrom` object
is serialized to a `SampledFromSimplified` replacing the linked specimen with the
identifier of the specimen.

The collection of specimens in a sampling hierarchy can be serialized/deserialized using
the `SpecimenJsonSchema`, which serializes all the contained specimens individually and, on
deserialization, recreates the specimen object linkage in the `sampled_from` attribute.
"""
import dataclasses
from abc import abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Type,
    TypeVar,
    Union,
)

from marshmallow import Schema, fields, post_load, pre_dump

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
from wsidicom.metadata.json_schema.fields import (
    CodeJsonField,
    JsonFieldFactory,
    MeasurementJsonField,
    SpecimenIdentifierJsonField,
    UidJsonField,
)
from wsidicom.metadata.json_schema.sample.model import (
    ExtractedSpecimenJsonModel,
    PreparationAction,
    SampleJsonModel,
    SamplingConstraintJsonModel,
    SamplingJsonModel,
    SlideSampleJsonModel,
    SpecimenJsonModel,
)
from wsidicom.metadata.json_schema.sample.parser import SpecimenJsonParser
from wsidicom.metadata.sample import (
    Collection,
    Embedding,
    ExtractedSpecimen,
    Fixation,
    PreparationStep,
    Processing,
    Receiving,
    Sample,
    SampledSpecimen,
    Sampling,
    SamplingLocation,
    SlideSample,
    Specimen,
    SpecimenIdentifier,
    SpecimenLocalization,
    Staining,
    Storage,
)

"""For sampling steps referencing other sampling steps in sampling_chain_constraints we
have a problem on how to serialize/deserialize. Suggest to change this linkage
with the identifier of the sampled specimen and the number of the sampling step. Add
a dataclass like:
class SamplingChaingConstraintSimplified:
    specimen: Union[str, SpecimenIdentifier]
    sampling_step_index: int

When we deserialize a specimens we check if the sampled specimen is already created,
and if not create that one first. The specimen in the sampling chain constraint should thus also
already be created.
"""


class SamplingLocationJsonSchema(Schema):
    reference = fields.String(allow_none=True)
    description = fields.String(allow_none=True)
    x = MeasurementJsonField(allow_none=True)
    y = MeasurementJsonField(allow_none=True)
    z = MeasurementJsonField(allow_none=True)

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs) -> SamplingLocation:
        return SamplingLocation(**data)


class SpecimenLocalizationJsonSchema(SamplingLocationJsonSchema):
    visual_marking = fields.String(allow_none=True)

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs) -> SpecimenLocalization:
        return SpecimenLocalization(**data)


class SamplingConstraintJsonSchema(Schema):
    """Schema for serializing and deserializing a `SamplingConstraintJsonModel`."""

    identifier = SpecimenIdentifierJsonField()
    sampling_step_index = fields.Integer()

    @pre_dump
    def dump_simple(self, sampling: Sampling, **kwargs):
        return SamplingConstraintJsonModel.to_json_model(sampling)

    @post_load
    def load_simple(self, data: Dict, **kwargs) -> SamplingConstraintJsonModel:
        return SamplingConstraintJsonModel(**data)


class BasePreparationStepJsonSchema(Schema):
    """Base Schema for serializing and deserializing a `PreparationStep`."""

    _load_class: Type[
        Union[
            SamplingJsonModel,
            Collection,
            Processing,
            Embedding,
            Fixation,
            Staining,
            Receiving,
            Storage,
        ]
    ]

    @post_load
    def post_load(
        self, data: Dict[str, Any], **kwargs
    ) -> Union[PreparationStep, SamplingJsonModel]:
        """Return a object of given load class using the defined dataclass fields."""
        return self._load_class(
            **{
                field.name: data[field.name]
                for field in dataclasses.fields(self._load_class)
                if field.name in data
            }
        )


class SamplingJsonSchema(BasePreparationStepJsonSchema):
    action = fields.Constant(PreparationAction.SAMPLING.value, dump_only=True)
    method = JsonFieldFactory.concept_code(SpecimenSamplingProcedureCode)()
    sampling_chain_constraints = fields.List(
        fields.Nested(SamplingConstraintJsonSchema, allow_none=True), allow_none=True
    )
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)
    location = fields.Nested(SamplingLocationJsonSchema, allow_none=True)
    _load_class = SamplingJsonModel


class CollectionJsonSchema(BasePreparationStepJsonSchema):
    action = fields.Constant(PreparationAction.COLLECTION.value, dump_only=True)
    method = JsonFieldFactory.concept_code(SpecimenCollectionProcedureCode)()
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)
    _load_class = Collection


class ProcessingJsonSchema(BasePreparationStepJsonSchema):
    action = fields.Constant(PreparationAction.PROCESSING.value, dump_only=True)
    method = JsonFieldFactory.concept_code(SpecimenPreparationStepsCode)(
        allow_none=True
    )
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)
    _load_class = Processing


class EmbeddingJsonSchema(BasePreparationStepJsonSchema):
    action = fields.Constant(PreparationAction.EMBEDDING.value, dump_only=True)
    medium = JsonFieldFactory.concept_code(SpecimenEmbeddingMediaCode)()
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)
    _load_class = Embedding


class FixationJsonSchema(BasePreparationStepJsonSchema):
    action = fields.Constant(PreparationAction.FIXATION.value, dump_only=True)
    fixative = JsonFieldFactory.concept_code(SpecimenFixativesCode)()
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)
    _load_class = Fixation


class StainingJsonSchema(BasePreparationStepJsonSchema):
    action = fields.Constant(PreparationAction.STAINING.value, dump_only=True)
    substances = fields.List(
        JsonFieldFactory.concept_code(SpecimenStainsCode)(), allow_none=True
    )
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)
    _load_class = Staining


class ReceivingJsonSchema(BasePreparationStepJsonSchema):
    action = fields.Constant(PreparationAction.RECEIVING.value, dump_only=True)
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)
    _load_class = Receiving


class StorageJsonSchema(BasePreparationStepJsonSchema):
    action = fields.Constant(PreparationAction.STORAGE.value, dump_only=True)
    date_time = fields.DateTime(allow_none=True)
    description = fields.String(allow_none=True)
    _load_class = Storage


class PreparationStepJsonSchema(Schema):
    """Schema to use to serialize/deserialize preparation steps."""

    """Mapping step type to schema."""
    _type_to_schema_mapping: Dict[Type[PreparationStep], Type[Schema]] = {
        Sampling: SamplingJsonSchema,
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
        PreparationAction.STAINING: StainingJsonSchema,
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


LoadType = TypeVar("LoadType")


class BaseSpecimenJsonSchema(Schema, Generic[LoadType]):
    """Base schema for specimen."""

    identifier = SpecimenIdentifierJsonField()
    steps = fields.List(fields.Nested(PreparationStepJsonSchema()))

    @property
    @abstractmethod
    def load_type(self) -> Type[LoadType]:
        """Return the specimen type to use for deserialization."""
        raise NotImplementedError()

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs) -> LoadType:
        """Return a specimen of given load type using the defined dataclass fields."""
        return self.load_type(**data)


class ExtractedSpecimenJsonSchema(BaseSpecimenJsonSchema[ExtractedSpecimen]):
    """Schema for extracted specimen that has not been sampled from other specimen."""

    type = JsonFieldFactory.concept_code(AnatomicPathologySpecimenTypesCode)()
    container = JsonFieldFactory.concept_code(ContainerTypeCode)(load_default=None)

    @property
    def load_type(self) -> Type[ExtractedSpecimenJsonModel]:
        return ExtractedSpecimenJsonModel


class SampleJsonSchema(BaseSpecimenJsonSchema[SampleJsonModel]):
    """Schema for sampled specimen."""

    sampled_from = fields.List(fields.Nested(SamplingConstraintJsonSchema))
    type = JsonFieldFactory.concept_code(AnatomicPathologySpecimenTypesCode)()
    container = JsonFieldFactory.concept_code(ContainerTypeCode)(load_default=None)

    @property
    def load_type(self) -> Type[SampleJsonModel]:
        return SampleJsonModel


class SlideSampleJsonSchema(BaseSpecimenJsonSchema[SlideSampleJsonModel]):
    """Schema for sampled specimen on a slide."""

    anatomical_sites = fields.List(CodeJsonField(), allow_none=True)
    sampled_from = fields.Nested(SamplingConstraintJsonSchema)
    uid = UidJsonField(allow_none=True)
    localization = fields.Nested(SpecimenLocalizationJsonSchema, allow_none=True)
    short_description = fields.String(allow_none=True)
    detailed_description = fields.String(allow_none=True)

    @property
    def load_type(self) -> Type[SlideSampleJsonModel]:
        return SlideSampleJsonModel


class SpecimenJsonSchema(Schema):
    """Schema to use to serialize/deserialize specimens."""

    """Mapping specimen type to schema."""
    _type_to_schema_mapping: Dict[Type[Specimen], Type[Schema]] = {
        ExtractedSpecimen: ExtractedSpecimenJsonSchema,
        Sample: SampleJsonSchema,
        SlideSample: SlideSampleJsonSchema,
    }

    """Mapping key in serialized specimen to schema."""
    _key_to_schema_mapping: Dict[str, Type[Schema]] = {
        "anatomical_sites": SlideSampleJsonSchema,
        "sampled_from": SampleJsonSchema,
        "type": ExtractedSpecimenJsonSchema,
    }

    def dump(
        self,
        specimens: Union[Specimen, Iterable[Specimen]],
        **kwargs,
    ):
        if isinstance(specimens, Specimen):
            specimens = [specimens]

        all_specimens: Dict[Union[str, SpecimenIdentifier], Specimen] = {}
        for specimen in specimens:
            if isinstance(specimen, SampledSpecimen):
                all_specimens.update(self._get_samplings(specimen))

        return [self._subschema_dump(specimen) for specimen in all_specimens.values()]

    def load(
        self,
        data: Union[Dict[str, Any], Iterable[Dict[str, Any]]],
        **kwargs,
    ) -> List[Specimen]:
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
    ) -> Dict[Union[str, SpecimenIdentifier], Specimen]:
        """Return a dictionary containing this specimen and all recursive sampled specimens."""
        samplings: Dict[Union[str, SpecimenIdentifier], Specimen] = {
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

    def _subschema_load(self, specimen: Dict[str, Any]) -> SpecimenJsonModel:
        """Select a schema and load and return specimen using the schema."""
        schema = self._subschema_select(specimen)
        loaded = schema().load(specimen, many=False)
        assert isinstance(loaded, SpecimenJsonModel)
        return loaded

    def _subschema_dump(self, specimen: Specimen) -> Dict[str, Any]:
        """Select a schema and dump the specimen using the schema."""
        schema = self._type_to_schema_mapping[type(specimen)]
        dumped = schema().dump(specimen)
        assert isinstance(dumped, dict)
        return dumped
