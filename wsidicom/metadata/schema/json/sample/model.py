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
from collections import UserDict
from dataclasses import dataclass
from enum import Enum
from typing import (
    List,
    Optional,
    Sequence,
    Union,
)

from pydicom.sr.coding import Code
from pydicom.uid import UID

from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    ContainerTypeCode,
    SpecimenSamplingProcedureCode,
)
from wsidicom.metadata.sample import (
    BaseSampling,
    Collection,
    Specimen,
    PreparationStep,
    Sample,
    Sampling,
    SamplingLocation,
    SlideSample,
    BaseSpecimen,
    SpecimenIdentifier,
    SampleLocalization,
    UnknownSampling,
)


class PreparationAction(Enum):
    SAMPLING = "sampling"
    COLLECTION = "collection"
    PROCESSING = "processing"
    STAINING = "staining"
    FIXATION = "fixation"
    EMBEDDING = "embedding"
    RECEIVING = "receiving"
    STORAGE = "storage"


@dataclass(frozen=True)
class SamplingConstraintJsonModel:
    """Simplified representation of a `Sampling` to use as sampling tree constraint,
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
        specimens: UserDict[Union[str, SpecimenIdentifier], BaseSpecimen],
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


@dataclass(frozen=True)
class SamplingJsonModel:
    """Simplified representation of a `Sampling`, replacing the sampled specimen with
    the idententifier and sampling constraints with simplified sampling constraints."""

    method: Optional[SpecimenSamplingProcedureCode] = None
    sampling_constraints: Optional[Sequence[SamplingConstraintJsonModel]] = None
    date_time: Optional[datetime.datetime] = None
    location: Optional[SamplingLocation] = None
    description: Optional[str] = None

    def from_json_model(
        self,
        specimen: BaseSpecimen,
        specimens: UserDict[Union[str, SpecimenIdentifier], BaseSpecimen],
    ) -> BaseSampling:
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
        if self.method is None or specimen.type is None:
            return UnknownSampling(
                specimen,
                self._get_sampling_constraints(specimens),
            )
        return Sampling(
            specimen,
            self.method,
            specimen.type,
            self._get_sampling_constraints(specimens),
            self.date_time,
            self.description,
            self.location,
        )

    def _get_sampling_constraints(
        self,
        specimens: UserDict[Union[str, SpecimenIdentifier], BaseSpecimen],
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
        if self.sampling_constraints is None:
            return None
        return [
            constraint.from_json_model(specimens)
            for constraint in self.sampling_constraints
        ]


@dataclass(frozen=True)
class BaseSpecimenJsonModel:
    """Base json model for a specimen."""

    identifier: Union[str, SpecimenIdentifier]
    steps: List[Union[PreparationStep, SamplingJsonModel]]

    @abstractmethod
    def from_json_model(
        self, specimens: UserDict[Union[str, SpecimenIdentifier], BaseSpecimen]
    ) -> BaseSpecimen:
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


@dataclass(frozen=True)
class SpecimenJsonModel(BaseSpecimenJsonModel):
    type: AnatomicPathologySpecimenTypesCode
    container: Optional[ContainerTypeCode]

    def from_json_model(
        self,
        specimens: UserDict[Union[str, SpecimenIdentifier], BaseSpecimen],
    ) -> Specimen:
        extraction_step = next(
            (step for step in self.steps if isinstance(step, Collection)), None
        )
        specimen = Specimen(
            identifier=self.identifier,
            extraction_step=extraction_step,
            type=self.type,
            container=self.container,
        )
        for step in self.steps:
            if isinstance(step, SamplingJsonModel):
                step = step.from_json_model(specimen, specimens)
            elif isinstance(step, Collection):
                if step != extraction_step:
                    raise ValueError(
                        "There should be only one Collection step.",
                    )
                continue
            specimen.add(step)
        return specimen


@dataclass(frozen=True)
class SampleJsonModel(BaseSpecimenJsonModel):
    type: AnatomicPathologySpecimenTypesCode
    sampled_from: Sequence[SamplingConstraintJsonModel]
    container: Optional[ContainerTypeCode]

    def from_json_model(
        self,
        specimens: UserDict[Union[str, SpecimenIdentifier], BaseSpecimen],
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
                    "Collection step can only be added to an Specimen",
                )
            sample.add(step)
        return sample


@dataclass(frozen=True)
class SlideSampleJsonModel(BaseSpecimenJsonModel):
    anatomical_sites: Optional[Sequence[Code]] = None
    sampled_from: Optional[SamplingConstraintJsonModel] = None
    uid: Optional[UID] = None
    localization: Optional[SampleLocalization] = None
    short_description: Optional[str] = None
    detailed_description: Optional[str] = None

    def from_json_model(
        self,
        specimens: UserDict[Union[str, SpecimenIdentifier], BaseSpecimen],
    ) -> SlideSample:
        sample = SlideSample(
            identifier=self.identifier,
            anatomical_sites=self.anatomical_sites,
            sampled_from=(
                self.sampled_from.from_json_model(specimens)
                if self.sampled_from
                else None
            ),
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
                    "Collection step can only be added to an Specimen",
                )
            sample.add(step)
        return sample
