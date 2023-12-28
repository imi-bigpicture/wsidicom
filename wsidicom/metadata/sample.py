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
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydicom.sr.coding import Code
from pydicom.uid import UID, generate_uid
from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    SpecimenCollectionProcedureCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenPreparationStepsCode,
    SpecimenSamplingProcedureCode,
    SpecimenStainsCode,
    UnitCode,
)


"""
A root specimen is created by giving it an identifier, a type, and optionally the
extraction method used.

To any specimen steps can be added. These are added to a list of steps, representing the
order that the specimen was processed.

A sampled specimen is created by one or more parent specimens, a sampling method, a type,
and an identifier.
The sampling can optionally be specific the to one or more specimens in the parent specimen.
A sampled specimen keeps track of the specimens it was sampled from. A sampling step is added to the parent
specimen representing the sampling (defining the sampling method, the sub-specimen
sampled (if not whole specimen), and the identifier of the new sample.)

Finally a slide sample will be made (using a create_slide_sample()-method?). The slide
sample has added position.

When DICOMizing the chain of samples, we start at the slide sample. For each specimen, we
process the steps in reverse order and convert the steps into SpecimenPreparationSteps.
For a Sample-specimen We then go to the linked sampled_from specimens and find the step
corresponding to the sampling, and parse all the steps prior to that step (again in reverse
order). Finally we end up at the TakenSpecimen which does not have a parent specimen.

When we parse the parent specimen steps for a sample, we only consider steps (processing
and sampling) the for sub-specimens in the parent the sample was sampled from, if specified.
E.g. there might be steps specific to one of the parent specimens samples, that is or is not
included. We only follow the sampled_from linkage of the given specimen.

Highdicom does not support the Specimen Receiving and Specimen Storage steps, so skip those.
Highdicom does not support Specimen Container and Specimen Type in the SpecimenPreparationStep,
consider making a PR.
"""


class UniversalIssuerType(Enum):
    """The type of a universal issuer of an identifier."""

    DNS = "DNS"  # An Internet dotted name. Either in ASCII or as integers
    EUI64 = "EUI64"  # An IEEE Extended Unique Identifier
    ISO = "ISO"  # An International Standards Organization Object Identifier
    URI = "IRI"  # Uniform Resource Identifier
    UUID = "UUID"  # The DCE Universal Unique Identifier
    X400 = "X400"  # An X.400 MHS identifier
    X500 = "X500"  # An X.500 directory name


class IssuerOfIdentifier(metaclass=ABCMeta):
    """Metaclass for an issuer of an identifier."""

    @abstractmethod
    def to_hl7v2(self) -> str:
        """Return HL7v2 representation of issuer."""
        raise NotImplementedError()

    @classmethod
    def from_hl7v2(cls, value: str) -> "IssuerOfIdentifier":
        """Return IssuerOfIdentifier from HL7v2 representation."""
        local_identifier, *universal_identifier = value.split("^")
        if local_identifier == "":
            local_identifier = None
        if len(universal_identifier) == 2:
            universal_identifier, universal_issuer_type = universal_identifier
            return UniversalIssuerOfIdentifier(
                identifier=universal_identifier,
                issuer_type=UniversalIssuerType(universal_issuer_type),
                local_identifier=local_identifier,
            )
        if local_identifier is not None:
            return LocalIssuerOfIdentifier(identifier=value)
        raise ValueError(
            "Could not parse string to local issuer of identifier or "
            "universal issuer of identifier."
        )


@dataclass(unsafe_hash=True)
class LocalIssuerOfIdentifier(IssuerOfIdentifier):
    """A local issuer of an identifier."""

    identifier: str

    def to_hl7v2(self) -> str:
        return self.identifier


@dataclass(unsafe_hash=True)
class UniversalIssuerOfIdentifier(IssuerOfIdentifier):
    """A universal issuer of an identifier. Can optinally also define a local identifer."""

    identifier: str
    issuer_type: UniversalIssuerType
    local_identifier: Optional[str] = None

    def to_hl7v2(self) -> str:
        identifier = self.local_identifier or ""
        identifier += f"^{self.identifier}^{self.issuer_type.name}"
        return identifier


@dataclass
class Measurement:
    value: float
    unit: UnitCode


@dataclass
class SamplingLocation:
    """The location of a sampling.

    Parameters
    ----------
    reference: Optinal[str] = None
        Description of coordinate system and origin reference point used for the
        location.
    description: Optional[str] = None
        Description of the location.
    x: Optional[Measurement] = None
        The x-coordinate of the location.
    y: Optional[Measurement] = None
        The y-coordinate of the location.
    z: Optional[Measurement] = None
        The z-coordinate of the location.
    """

    reference: Optional[str] = None
    description: Optional[str] = None
    x: Optional[Measurement] = None
    y: Optional[Measurement] = None
    z: Optional[Measurement] = None


@dataclass
class SpecimenLocalization(SamplingLocation):
    """The location of a specimen.

    Parameters
    ----------
    reference: Optinal[str] = None
        Description of coordinate system and origin reference point used for the
        location.
    description: Optional[str] = None
        Description of the location.
    x: Optional[Measurement] = None
        The x-coordinate of the location.
    y: Optional[Measurement] = None
        The y-coordinate of the location.
    z: Optional[Measurement] = None
        The z-coordinate of the location.
    visual_marking: Optional[str] = None
        Description of visual marking of the specimen, for example ink or shape.
    """

    visual_marking: Optional[str] = None


@dataclass(unsafe_hash=True)
class SpecimenIdentifier:
    """A specimen identifier including an optional issuer."""

    value: str
    issuer: Optional[IssuerOfIdentifier] = None

    def __eq__(self, other: Any):
        """Determine if other specimen identifiers is equal to this."""
        if isinstance(other, str):
            return self.value == other and self.issuer is None
        if isinstance(other, SpecimenIdentifier):
            return self.value == other.value and self.issuer == other.issuer
        return False

    def to_string_identifier_and_issuer(self) -> Tuple[str, Optional[str]]:
        """Format into string identifier and IssuerOfIdentifier."""
        if self.issuer is None:
            return self.value, None
        return self.value, self.issuer.to_hl7v2()

    @classmethod
    def get_string_identifier_and_issuer(
        cls, identifier: Union[str, "SpecimenIdentifier"]
    ) -> Tuple[str, Optional[str]]:
        """Return string identifier and optional issuer of identifier object."""
        if isinstance(identifier, str):
            return identifier, None
        return identifier.to_string_identifier_and_issuer()

    @classmethod
    def get_identifier_and_issuer(
        cls, identifier: Union[str, "SpecimenIdentifier"]
    ) -> Tuple[str, Optional[IssuerOfIdentifier]]:
        """Return string identifier and optional issuer of identifier object."""
        if isinstance(identifier, str):
            return identifier, None
        return identifier.value, identifier.issuer


class PreparationStep(metaclass=ABCMeta):
    """
    Metaclass for a preparation step for a specimen.

    A preparation step is an action performed on a specimen.
    """


@dataclass
class Sampling(PreparationStep):
    """
    The sampling of a specimen into samples that can be used to create new specimens.

    See
    https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8110.html
    for allowed sampling methods.
    """

    specimen: "Specimen"
    method: SpecimenSamplingProcedureCode
    sampling_chain_constraints: Optional[Sequence["Sampling"]] = None
    date_time: Optional[datetime.datetime] = None
    description: Optional[str] = None
    location: Optional[SamplingLocation] = None


@dataclass
class Collection(PreparationStep):
    """
    The collection of a specimen from a body.

    See
    https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8109.html
    for allowed collection methods.
    """

    method: SpecimenCollectionProcedureCode
    date_time: Optional[datetime.datetime] = None
    description: Optional[str] = None


@dataclass
class Processing(PreparationStep):
    """
    Other processing steps, such as heating or clearing, made on a specimen.

    See
    https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8113.html
    for allowed proo´cessing methods.
    """

    method: SpecimenPreparationStepsCode
    date_time: Optional[datetime.datetime] = None
    description: Optional[str] = None


@dataclass
class Embedding(PreparationStep):
    """
    Embedding of a specimen in an medium.

    See
    https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8115.html
    for allowed mediums.
    """

    medium: SpecimenEmbeddingMediaCode
    date_time: Optional[datetime.datetime] = None
    description: Optional[str] = None


@dataclass
class Fixation(PreparationStep):
    """
    Fixation of a specimen using a fixative.

    See
    https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8114.html
    for allowed fixatives.
    """

    fixative: SpecimenFixativesCode
    date_time: Optional[datetime.datetime] = None
    description: Optional[str] = None


@dataclass
class Staining(PreparationStep):
    """
    Staining of a specimen using staining substances.

    The substances can be given either as string or coded values. See
    https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8112.html
    for allowed stain codes.
    """

    substances: List[Union[str, SpecimenStainsCode]]
    date_time: Optional[datetime.datetime] = None
    description: Optional[str] = None


class Specimen(metaclass=ABCMeta):
    """Metaclass for a specimen."""

    def __init__(
        self,
        identifier: Union[str, SpecimenIdentifier],
        type: AnatomicPathologySpecimenTypesCode,
        steps: Sequence[PreparationStep],
    ):
        self.identifier = identifier
        self.type = type
        self.steps = list(steps)

    @property
    def samplings(self) -> List[Sampling]:
        """Return list of samplings done on the specimen."""
        return [step for step in self.steps if isinstance(step, Sampling)]

    @abstractmethod
    def add(self, step: PreparationStep) -> None:
        """Add a preparation step to the sequence of steps for the specimen."""
        raise NotImplementedError()


class SampledSpecimen(Specimen, metaclass=ABCMeta):
    """Metaclass for a specimen thas has been sampled from one or more specimens."""

    def __init__(
        self,
        identifier: Union[str, SpecimenIdentifier],
        type: AnatomicPathologySpecimenTypesCode,
        sampled_from: Optional[Union[Sampling, Sequence[Sampling]]],
        steps: Sequence[PreparationStep],
    ):
        super().__init__(identifier, type, steps)
        if sampled_from is None:
            sampled_from = []
        elif isinstance(sampled_from, Sampling):
            sampled_from = [sampled_from]
        elif not isinstance(sampled_from, list):
            sampled_from = list(sampled_from)
        self._sampled_from = sampled_from

    def add(self, step: PreparationStep) -> None:
        if isinstance(step, Collection):
            raise ValueError(
                "A collection step can only be added to specimens of type `ExtractedSpecimen`"
            )
        self.steps.append(step)

    def get_samplings(self) -> Dict[Union[str, SpecimenIdentifier], Specimen]:
        """Return a dictionary containing this specimen and all recursive sampled specimens."""
        samplings: Dict[Union[str, SpecimenIdentifier], Specimen] = {
            self.identifier: self
        }
        for sampling in self._sampled_from:
            if not isinstance(sampling.specimen, SampledSpecimen):
                samplings.update({sampling.specimen.identifier: sampling.specimen})
            else:
                samplings.update(sampling.specimen.get_samplings())
        return samplings

    def _check_sampling_constraints(
        self, constraints: Optional[Sequence[Sampling]]
    ) -> None:
        if constraints is None:
            return

        def recursive_search(sampling: Sampling, specimen: Specimen) -> bool:
            """Recursively search for sampling in samplings for specimen."""
            if sampling in specimen.samplings:
                return True
            if isinstance(specimen, SampledSpecimen):
                return any(
                    recursive_search(sampling, sampling.specimen)
                    for sampling in specimen._sampled_from
                )
            return False

        for constraint in constraints:
            if not recursive_search(constraint, self):
                raise ValueError(
                    "Could not create sampling as specimen was not sampled "
                    f"from {constraint}"
                )


@dataclass
class ExtractedSpecimen(Specimen):
    """A specimen that has been extracted/taken from a patient in some way. Does not
    need to represent the actual first specimen in the collection chain, but should
    represent the first known (i.e. that we have metadata for) specimen in the collection
    chain."""

    identifier: Union[str, SpecimenIdentifier]
    type: AnatomicPathologySpecimenTypesCode
    extraction_step: Optional[Collection] = None
    steps: List[PreparationStep] = field(default_factory=list)

    def __post_init__(self):
        if self.extraction_step is not None:
            # Add self.extraction step as first in list of steps
            self.steps.insert(
                0,
                self.extraction_step,
            )
        else:
            # If extraction step in steps, set it as self.extraction_step
            extraction_step = next(
                (step for step in self.steps if isinstance(step, Collection)), None
            )
            if extraction_step is not None:
                self.extraction_step = extraction_step
        super().__init__(identifier=self.identifier, type=self.type, steps=self.steps)

    def add(self, step: PreparationStep) -> None:
        if isinstance(step, Collection) and len(self.steps) != 0:
            raise ValueError("A Collection-step must be the first step.")
        self.steps.append(step)

    def sample(
        self,
        method: SpecimenSamplingProcedureCode,
        date_time: Optional[datetime.datetime] = None,
        description: Optional[str] = None,
        location: Optional[SamplingLocation] = None,
    ) -> Sampling:
        """Create a sampling from the specimen that can be used to create a new sample."""
        sampling = Sampling(
            specimen=self,
            method=method,
            sampling_chain_constraints=None,
            date_time=date_time,
            description=description,
            location=location,
        )
        self.add(sampling)
        return sampling


@dataclass
class Sample(SampledSpecimen):
    """A specimen that has been sampled from one or more other specimens."""

    identifier: Union[str, SpecimenIdentifier]
    type: AnatomicPathologySpecimenTypesCode
    sampled_from: Sequence[Sampling]
    steps: Sequence[PreparationStep] = field(default_factory=list)

    def __post_init__(self):
        super().__init__(
            identifier=self.identifier,
            type=self.type,
            sampled_from=self.sampled_from,
            steps=self.steps,
        )

    def sample(
        self,
        method: SpecimenSamplingProcedureCode,
        date_time: Optional[datetime.datetime] = None,
        description: Optional[str] = None,
        sampling_chain_constraints: Optional[Sequence[Sampling]] = None,
        location: Optional[SamplingLocation] = None,
    ) -> Sampling:
        """Create a sampling from the specimen that can be used to create a new sample."""
        self._check_sampling_constraints(sampling_chain_constraints)
        if sampling_chain_constraints is not None:
            for sampling_chain_constraint in sampling_chain_constraints:
                assert isinstance(sampling_chain_constraint, Sampling)
        sampling = Sampling(
            specimen=self,
            method=method,
            sampling_chain_constraints=sampling_chain_constraints,
            date_time=date_time,
            description=description,
            location=location,
        )
        self.add(sampling)
        return sampling


@dataclass
class SlideSample(SampledSpecimen):
    """A sample that has been placed on a slide."""

    identifier: Union[str, SpecimenIdentifier]
    anatomical_sites: Optional[Sequence[Code]] = None
    sampled_from: Optional[Sampling] = None
    uid: Optional[UID] = None
    localization: Optional[SpecimenLocalization] = None
    steps: List[PreparationStep] = field(default_factory=list)
    short_description: Optional[str] = None
    detailed_description: Optional[str] = None

    def __post_init__(self):
        super().__init__(
            identifier=self.identifier,
            type=AnatomicPathologySpecimenTypesCode("Slide"),
            sampled_from=self.sampled_from,
            steps=self.steps,
        )

    @cached_property
    def default_uid(self) -> UID:
        """Uid used if not set."""
        if self.uid is not None:
            return self.uid
        return generate_uid()
