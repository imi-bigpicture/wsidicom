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

from abc import abstractmethod
from collections import UserDict
from pydicom.sr.coding import Code
from pydicom.uid import UID
import datetime
from dataclasses import dataclass

from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    SpecimenSamplingProcedureCode,
)
from wsidicom.metadata.sample import (
    ExtractedSpecimen,
    PreparationStep,
    Sample,
    SampledSpecimen,
    SamplingLocation,
    SlideSample,
    Specimen,
    SpecimenIdentifier,
    SpecimenLocalization,
    Sampling,
    Collection,
)


class SpecimenDictionary(UserDict[Union[str, SpecimenIdentifier], Specimen]):
    """Dictionary for specimens that creates missing specimens when accessed."""

    def __init__(
        self,
        create_missing: Callable[[Union[str, SpecimenIdentifier]], Specimen],
        *args,
        **kwargs,
    ):
        """Initialize dictionary.

        Parameters
        ----------
        create_missing: Callable[[Union[str, SpecimenIdentifier]], Specimen]
            Function that creates a specimen for a given identifier.
        """
        super().__init__(*args, **kwargs)
        self._create_missing = create_missing

    def __missing__(self, key: Union[str, SpecimenIdentifier]) -> Specimen:
        """Create missing specimen."""
        specimen = self._create_missing(key)
        self[key] = specimen
        return specimen


@dataclass
class SamplingConstraintJsonModel:
    """Simplified representation of a `Sampling` to use as sampling chain constraint,
    replacing the sampling with the identifier of the sampled specimen and the index of
    the sampling step within the step sequence of the specimen."""

    identifier: Union[str, SpecimenIdentifier]
    sampling_step_index: int

    @classmethod
    def to_json_model(cls, sampling: Sampling) -> "SamplingConstraintJsonModel":
        """Create json model for sampling."""
        return cls(
            identifier=sampling.specimen.identifier,
            sampling_step_index=sampling.specimen.samplings.index(sampling),
        )

    def from_json_model(
        self,
        specimens: SpecimenDictionary,
    ) -> Sampling:
        """Create sampling from json model.

        Parameters
        ----------
        specimens: SpecimenDictionary
            Dictionary providing specimens.

        Returns
        -------
        Sampling
            Created sampling.
        """
        specimen = specimens[self.identifier]
        return specimen.samplings[self.sampling_step_index]


@dataclass
class SamplingJsonModel:
    """Simplified representation of a `Sampling`, replacing the sampled specimen with
    the idententifier and sampling constratins with simplified sampling constraints."""

    method: SpecimenSamplingProcedureCode
    sampling_chain_constraints: Optional[Sequence[SamplingConstraintJsonModel]] = None
    date_time: Optional[datetime.datetime] = None
    location: Optional[SamplingLocation] = None
    description: Optional[str] = None

    def from_json_model(
        self,
        specimen: Specimen,
        specimens: SpecimenDictionary,
    ) -> Sampling:
        """Create sampling from json model.

        Parameters
        ----------
        specimen: Specimen
            The specimen that has been sampled.
        specimens: SpecimenDictionary
            Dictionary providing specimens.

        Returns
        -------
        Sampling
            Created sampling.
        """
        return Sampling(
            specimen,
            self.method,
            self._get_sampling_constraints(specimens),
            self.date_time,
            self.description,
            self.location,
        )

    def _get_sampling_constraints(
        self,
        specimens: SpecimenDictionary,
    ) -> Optional[List[Sampling]]:
        """
        Get list of constraint sampling this sampling.

        Parameters
        ---------
        specimens: SpecimenDictionary
            Dictionary providing specimens.

        Returns
        -------
        Optional[List[Sampling]]
            List of constraint samplings, or None if no constraints.
        """
        if self.sampling_chain_constraints is None:
            return None
        return [
            constraint.from_json_model(specimens)
            for constraint in self.sampling_chain_constraints
        ]


@dataclass
class SpecimenJsonModel:
    """Base json model for a specimen."""

    identifier: Union[str, SpecimenIdentifier]
    steps: List[Union[PreparationStep, SamplingJsonModel]]

    @abstractmethod
    def from_json_model(self, specimens: SpecimenDictionary) -> Specimen:
        """Return specimen created from this json model.

        Parameters
        ----------
        specimens: SpecimenDictionary
            Dictionary providing specimens.

        Returns
        -------
        Specimen
            Created specimen for this model.
        """
        raise NotImplementedError()


@dataclass
class ExtractedSpecimenJsonModel(SpecimenJsonModel):
    type: AnatomicPathologySpecimenTypesCode

    def from_json_model(
        self,
        specimens: SpecimenDictionary,
    ) -> ExtractedSpecimen:
        specimen = ExtractedSpecimen(
            identifier=self.identifier,
            type=self.type,
        )
        for index, step in enumerate(self.steps):
            if isinstance(step, SamplingJsonModel):
                step = step.from_json_model(specimen, specimens)
            elif isinstance(step, Collection):
                if index != 0:
                    raise ValueError(
                        (
                            "Collection step can only be added as first step to "
                            " an ExtractedSpecimen"
                        )
                    )
                specimen.extraction_step = step
            specimen.add(step)
        return specimen


@dataclass
class SampleJsonModel(SpecimenJsonModel):
    type: AnatomicPathologySpecimenTypesCode
    sampled_from: Sequence[SamplingConstraintJsonModel]

    def from_json_model(
        self,
        specimens: SpecimenDictionary,
    ) -> Sample:
        sample = Sample(
            identifier=self.identifier,
            type=self.type,
            sampled_from=[
                sample.from_json_model(specimens) for sample in self.sampled_from
            ],
        )
        for step in self.steps:
            if isinstance(step, SamplingJsonModel):
                step = step.from_json_model(sample, specimens)
            elif isinstance(step, Collection):
                raise ValueError(
                    "Collection step can only be added to an ExtractedSpecimen",
                )
            sample.add(step)
        return sample


@dataclass
class SlideSampleJsonModel(SpecimenJsonModel):
    anatomical_sites: Optional[Sequence[Code]] = None
    sampled_from: Optional[SamplingConstraintJsonModel] = None
    uid: Optional[UID] = None
    localization: Optional[SpecimenLocalization] = None
    short_description: Optional[str] = None
    detailed_description: Optional[str] = None

    def from_json_model(
        self,
        specimens: SpecimenDictionary,
    ) -> SlideSample:
        sample = SlideSample(
            identifier=self.identifier,
            anatomical_sites=self.anatomical_sites,
            sampled_from=self.sampled_from.from_json_model(specimens)
            if self.sampled_from
            else None,
            uid=self.uid,
            localization=self.localization,
            short_description=self.short_description,
            detailed_description=self.detailed_description,
        )
        for step in self.steps:
            if isinstance(step, SamplingJsonModel):
                raise ValueError("A SlideSample cannot be sampled to another specimen.")
            elif isinstance(step, Collection):
                raise ValueError(
                    "Collection step can only be added to an ExtractedSpecimen",
                )
            sample.add(step)
        return sample


class SpecimenFactory:
    """Factory for creating specimens from json models."""

    def __init__(self, specimen_models: Iterable[SpecimenJsonModel]):
        """Initiate factory.

        Parameters
        ----------
        specimen_models: Iterable[SpecimenJsonModel]
            Json models to create specimens from.
        """
        self._specimens = SpecimenDictionary(self._make_specimen)
        self._specimen_models_by_identifier: Dict[
            Union[str, SpecimenIdentifier], SpecimenJsonModel
        ] = {specimen.identifier: specimen for specimen in specimen_models}

    def create_specimens(self) -> List[Specimen]:
        """Create specimens

        Returns
        -------
        List[Specimen]
            List of created specimens.
        """
        for identifier in self._specimen_models_by_identifier:
            self._specimens[identifier] = self._make_specimen(identifier)
        sampled_specimens = [
            sampled_from.specimen.identifier
            for specimen in self._specimens.values()
            if isinstance(specimen, SampledSpecimen)
            for sampled_from in specimen.sampled_from_list
        ]
        return [
            specimen
            for specimen in self._specimens.values()
            if specimen.identifier not in sampled_specimens
        ]

    def _make_specimen(
        self,
        identifier: Union[str, SpecimenIdentifier],
    ) -> Specimen:
        """Create specimen by identifier from json model.

        Create nested specimens that the specimen is sampled from if needed.

        Parameters
        ----------
        identifier: Union[str, SpecimenIdentifier]
            Identifier of specimen to create

        Returns
        -------
        Specimen
            Specimen created from json model.

        """
        specimen_model = self._specimen_models_by_identifier[identifier]
        return specimen_model.from_json_model(self._specimens)
