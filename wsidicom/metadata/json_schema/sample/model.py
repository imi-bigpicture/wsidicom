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
    List,
    Optional,
    Sequence,
    Union,
)

from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    ContainerTypeCode,
    SpecimenSamplingProcedureCode,
)
from wsidicom.metadata.sample import (
    ExtractedSpecimen,
    PreparationStep,
    Sample,
    SamplingLocation,
    SlideSample,
    Specimen,
    SpecimenIdentifier,
    SpecimenLocalization,
    Sampling,
    Collection,
)


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
        specimens: UserDict[Union[str, SpecimenIdentifier], Specimen],
    ) -> Sampling:
        """Create sampling from json model.

        Parameters
        ----------
        specimens: UserDict[
            Union[str, SpecimenIdentifier], Specimen
        ]
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
        specimens: UserDict[Union[str, SpecimenIdentifier], Specimen],
    ) -> Sampling:
        """Create sampling from json model.

        Parameters
        ----------
        specimen: Specimen
            The specimen that has been sampled.
        specimens: UserDict[
            Union[str, SpecimenIdentifier], Specimen
        ]
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
        specimens: UserDict[Union[str, SpecimenIdentifier], Specimen],
    ) -> Optional[List[Sampling]]:
        """
        Get list of constraint sampling this sampling.

        Parameters
        ---------
        specimens: UserDict[
            Union[str, SpecimenIdentifier], Specimen
        ]
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
    def from_json_model(
        self, specimens: UserDict[Union[str, SpecimenIdentifier], Specimen]
    ) -> Specimen:
        """Return specimen created from this json model.

        Parameters
        ----------
        specimens: UserDict[
            Union[str, SpecimenIdentifier], Specimen
        ]
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
    container: Optional[ContainerTypeCode]

    def from_json_model(
        self,
        specimens: UserDict[Union[str, SpecimenIdentifier], Specimen],
    ) -> ExtractedSpecimen:
        specimen = ExtractedSpecimen(
            identifier=self.identifier, type=self.type, container=self.container
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
    container: Optional[ContainerTypeCode]

    def from_json_model(
        self,
        specimens: UserDict[Union[str, SpecimenIdentifier], Specimen],
    ) -> Sample:
        sample = Sample(
            identifier=self.identifier,
            type=self.type,
            sampled_from=[
                sample.from_json_model(specimens) for sample in self.sampled_from
            ],
            container=self.container,
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
        specimens: UserDict[Union[str, SpecimenIdentifier], Specimen],
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
