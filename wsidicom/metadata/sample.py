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

"""Models for specimen description."""

import datetime
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

from pydicom.sr.coding import Code
from pydicom.uid import UID

from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    ContainerTypeCode,
    SpecimenCollectionProcedureCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenPreparationStepsCode,
    SpecimenSamplingProcedureCode,
    SpecimenStainsCode,
    UnitCode,
)


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
    """Metaclass for an issuer of an issuer of identifier."""

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


@dataclass(frozen=True)
class LocalIssuerOfIdentifier(IssuerOfIdentifier):
    """A local issuer of an identifier.

    Parameters
    ----------
    identifier: str
        The identifier of the local issuer.
    """

    identifier: str

    def to_hl7v2(self) -> str:
        return self.identifier


@dataclass(frozen=True)
class UniversalIssuerOfIdentifier(IssuerOfIdentifier):
    """A universal issuer of an identifier.

    Parameters
    ----------
    identifier: str
        The identifier of the universal issuer.
    issuer_type: UniversalIssuerType
        The type of the universal issuer.
    local_identifier: str | None = None
        Optional local identifier of the universal issuer.
    """

    identifier: str
    issuer_type: UniversalIssuerType
    local_identifier: str | None = None

    def to_hl7v2(self) -> str:
        identifier = self.local_identifier or ""
        identifier += f"^{self.identifier}^{self.issuer_type.name}"
        return identifier


@dataclass(frozen=True)
class Measurement:
    """A measurement with a value and a unit.

    Parameters
    ----------
    value: float
        The value of the measurement.
    unit: UnitCode
        The unit of the measurement.
    """

    value: float
    unit: UnitCode


@dataclass(frozen=True)
class SamplingLocation:
    """The location of a sampling.

    Parameters
    ----------
    reference: str | None = None
        Description of coordinate system and origin reference point used for the
        location.
    description: str | None = None
        Description of the location.
    x: Measurement | None = None
        The x-coordinate of the location.
    y: Measurement | None = None
        The y-coordinate of the location.
    z: Measurement | None = None
        The z-coordinate of the location.
    """

    reference: str | None = None
    description: str | None = None
    x: Measurement | None = None
    y: Measurement | None = None
    z: Measurement | None = None


@dataclass(frozen=True)
class SampleLocalization(SamplingLocation):
    """The location of a sample on a slide.

    Parameters
    ----------
    reference: str | None = None
        Description of coordinate system and origin reference point used for the
        location.
    description: str | None = None
        Description of the location.
    x: Measurement | None = None
        The x-coordinate of the location.
    y: Measurement | None = None
        The y-coordinate of the location.
    z: Measurement | None = None
        The z-coordinate of the location.
    visual_marking: str | None = None
        Description of visual marking of the sample, for example ink or shape.
    """

    visual_marking: str | None = None


@dataclass(frozen=True)
class SpecimenIdentifier:
    """A specimen identifier including an optional issuer.

    Parameters
    ----------
    value: str
        The value of the identifier.
    issuer: IssuerOfIdentifier | None = None
        Optional issuer of the identifier.
    """

    value: str
    issuer: IssuerOfIdentifier | None = None

    def __eq__(self, other: object) -> bool:
        """Determine if other specimen identifiers is equal to this."""
        if isinstance(other, str):
            return self.value == other and self.issuer is None
        if isinstance(other, SpecimenIdentifier):
            return self.value == other.value and self.issuer == other.issuer
        return False

    def to_string_identifier_and_issuer(self) -> tuple[str, str | None]:
        """Format into string identifier and IssuerOfIdentifier."""
        if self.issuer is None:
            return self.value, None
        return self.value, self.issuer.to_hl7v2()

    @classmethod
    def get_string_identifier_and_issuer(
        cls, identifier: Union[str, "SpecimenIdentifier"]
    ) -> tuple[str, str | None]:
        """Return string identifier and optional issuer of identifier object."""
        if isinstance(identifier, str):
            return identifier, None
        return identifier.to_string_identifier_and_issuer()

    @classmethod
    def get_identifier_and_issuer(
        cls, identifier: Union[str, "SpecimenIdentifier"]
    ) -> tuple[str, IssuerOfIdentifier | None]:
        """Return string identifier and optional issuer of identifier object."""
        if isinstance(identifier, str):
            return identifier, None
        return identifier.value, identifier.issuer

    def simplify(self) -> Union[str, "SpecimenIdentifier"]:
        """Return a simplified version of the identifier."""
        if self.issuer is None:
            return self.value
        return self

    def relaxed_matching(self, other: "SpecimenIdentifier") -> bool:
        """Determine if other specimen identifiers is equal to this, but does not
        require issuer to be the same if one of the issuers is None."""
        if self == other:
            return True
        if self.value != other.value:
            return False
        return self.issuer is None or other.issuer is None


class PreparationStep:
    """
    Metaclass for a preparation step for a specimen.

    A preparation step is an action performed on a specimen.
    """


class BaseSampling(PreparationStep, metaclass=ABCMeta):
    """Either a `Sampling` or a `UnknownSampling`."""

    specimen: "BaseSpecimen"
    sampling_constraints: Sequence["BaseSampling"] | None


@dataclass(frozen=True)
class Sampling(BaseSampling):
    """
    The sampling of a specimen into samples that can be used to create new specimens.

    Parameters
    ----------
    specimen: BaseSpecimen
        The specimen that was sampled.
    method: SpecimenSamplingProcedureCode
        Method used for sampling. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8110.html
        or `SpecimenSamplingProcedureCode.meanings` for allowed sampling methods.
    specimen_type: AnatomicPathologySpecimenTypesCode
        The type of the specimen. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8103.html
         or `AnatomicPathologySpecimenTypesCode.meanings` for allowed specimen types.
    sampling_constraints: Sequence[BaseSampling] | None = None
        Optional constraints on the sampling tree. The constraints can be used to
        define a single branch from the sample to a end specimen in the sampling tree.
    date_time: datetime.datetime | None = None
        Optional date and time when the sampling was performed.
    description: str | None = None
        Optional description of the sampling.
    location: SamplingLocation | None = None
        Optional location of the sampling in the parent specimen.
    """

    specimen: "BaseSpecimen"
    method: SpecimenSamplingProcedureCode
    specimen_type: AnatomicPathologySpecimenTypesCode = field(repr=False)
    sampling_constraints: Sequence[BaseSampling] | None = None
    date_time: datetime.datetime | None = None
    description: str | None = None
    location: SamplingLocation | None = None


@dataclass(frozen=True)
class UnknownSampling(BaseSampling):
    """
    Represent an unknown relation between a parent specimen and a child sample.

    The relation between the parent and child can be direct (e.g. from block to slide)
    or several steps in between can be missing (e.g. from specimen to slide). If
    possible is preferred to use the `Sampling` class instead.

    Parameters
    ----------
    specimen: BaseSpecimen
        The specimen that was sampled.
    sampling_constraints: Sequence[BaseSampling] | None = None
        Optional constraints on the sampling tree. The constraints should be used to
        define a single branch for the sample in the sampling tree.
    """

    specimen: "BaseSpecimen"
    sampling_constraints: Sequence[BaseSampling] | None = None


@dataclass(frozen=True)
class Collection(PreparationStep):
    """
    The collection of a specimen from a body.

    Parameters
    ----------
    method: SpecimenCollectionProcedureCode
        Method used for collection. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8109.html
        or `SpecimenCollectionProcedureCode.meanings` for allowed collection methods.
    date_time: datetime.datetime | None = None
        Optional date and time when the collection was performed.
    description: str | None = None
        Optional description of the collection.
    """

    method: SpecimenCollectionProcedureCode
    date_time: datetime.datetime | None = None
    description: str | None = None


@dataclass(frozen=True)
class Processing(PreparationStep):
    """
    Other processing steps, such as heating or clearing, made on a specimen.

    Parameters
    ----------
    method: SpecimenPreparationStepsCode
        Method used for processing. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8113.html
        or `SpecimenPreparationStepsCode.meanings` for allowed processing methods.
    date_time: datetime.datetime | None = None
        Optional date and time when the processing was performed.
    description: str | None = None
        Optional description of the processing.
    """

    method: SpecimenPreparationStepsCode | None = None
    date_time: datetime.datetime | None = None
    description: str | None = None


@dataclass(frozen=True)
class Embedding(PreparationStep):
    """
    Embedding of a specimen in an medium.

    Parameters
    ----------
    medium: SpecimenEmbeddingMediaCode
        The medium used for embedding. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8115.html
        or `SpecimenEmbeddingMediaCode.meanings` for allowed mediums.
    date_time: datetime.datetime | None = None
        Optional date and time when the embedding was performed.
    description: str | None = None
        Optional description of the embedding.
    """

    medium: SpecimenEmbeddingMediaCode
    date_time: datetime.datetime | None = None
    description: str | None = None


@dataclass(frozen=True)
class Fixation(PreparationStep):
    """
    Fixation of a specimen using a fixative.

    Parameters
    ----------
    fixative: SpecimenFixativesCode
        The fixative used for fixation. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8114.html
        or `SpecimenFixativesCode.meanings` for allowed fixatives.
    date_time: datetime.datetime | None = None
        Optional date and time when the fixation was performed.
    description: str | None = None
        Optional description of the fixation.
    """

    fixative: SpecimenFixativesCode
    date_time: datetime.datetime | None = None
    description: str | None = None


@dataclass(frozen=True)
class Staining(PreparationStep):
    """
    Staining of a specimen using staining substances.

    Parameters
    ----------
    substances: str | Sequence[SpecimenStainsCode]
        The substances used for staining. Either a single string or a list of codes. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8112.html
        or SpecimenStainsCode.meanings` for allowed stain codes.
    date_time: datetime.datetime | None = None
        Optional date and time when the staining was performed.
    description: str | None = None
        Optional description of the staining.
    """

    substances: str | Sequence[SpecimenStainsCode]
    date_time: datetime.datetime | None = None
    description: str | None = None


@dataclass(frozen=True)
class Receiving(PreparationStep):
    """
    Reception of a specimen.

    Parameters
    ----------
    date_time: datetime.datetime | None = None
        Optional date and time when the reception was performed.
    description: str | None = None
        Optional description of the reception.
    """

    date_time: datetime.datetime | None = None
    description: str | None = None


@dataclass(frozen=True)
class Storage(PreparationStep):
    """
    Storage of a specimen.

    Parameters
    ----------
    date_time: datetime.datetime | None = None
        Optional date and time when the storage was performed.
    description: str | None = None
        Optional description of the storage.
    """

    date_time: datetime.datetime | None = None
    description: str | None = None


class BaseSpecimen(metaclass=ABCMeta):
    """Metaclass for a specimen."""

    steps: list[PreparationStep]
    identifier: str | SpecimenIdentifier
    type: AnatomicPathologySpecimenTypesCode | None
    container: ContainerTypeCode | None

    @property
    def samplings(self) -> list[Sampling]:
        """Return list of samplings done on the specimen."""
        return [step for step in self.steps if isinstance(step, Sampling)]

    def add(self, step: PreparationStep) -> None:
        """Add a preparation step to the sequence of steps for the specimen."""
        if not isinstance(step, BaseSampling) and any(
            isinstance(present_step, BaseSampling) for present_step in self.steps
        ):
            raise ValueError(
                "Only additional samplings can be added to a specimen that has "
                "been sampled."
            )
        self.steps.append(step)


class SampledSpecimen(BaseSpecimen, metaclass=ABCMeta):
    """Metaclass for a specimen thas has been sampled from one or more specimens."""

    sampled_from: BaseSampling | list[BaseSampling] | None

    @property
    def sampled_from_list(self) -> list[BaseSampling]:
        if self.sampled_from is None:
            return []
        if isinstance(self.sampled_from, BaseSampling):
            return [self.sampled_from]
        return self.sampled_from

    def add(self, step: PreparationStep) -> None:
        if isinstance(step, Collection):
            raise ValueError(
                "A collection step can only be added to specimens of type `Specimen`"
            )
        super().add(step)

    def _check_sampling_constraints_in_sampling_tree(
        self, constraints: Sequence[BaseSampling] | None
    ) -> None:
        """Check that the sampling tree for the specimen contains the constraints."""
        if constraints is None:
            return

        def recursive_search(sampling: BaseSampling, specimen: BaseSpecimen) -> bool:
            """Recursively search for sampling in samplings for specimen."""
            if sampling in specimen.samplings:
                return True
            if isinstance(specimen, SampledSpecimen):
                return any(
                    recursive_search(sampling, sampling.specimen)
                    for sampling in specimen.sampled_from_list
                )
            return False

        for constraint in constraints:
            if not recursive_search(constraint, self):
                raise ValueError(
                    "Could not create sampling as specimen was not sampled "
                    f"from {constraint}"
                )

    def is_sampling_constraint_is_unambiguous(
        self,
        constraints: Iterable[str | SpecimenIdentifier],
    ):
        """Check that the sampling tree for the specimen is is not branching when
        following the constraints."""
        if len(self.sampled_from_list) == 0:
            # No sampling tree, is unambiguous
            return True
        if len(self.sampled_from_list) > 1:
            # Check that at one and only one of the samplings are in the constraints
            samplings_in_constraints = (
                sampling
                for sampling in self.sampled_from_list
                if sampling.specimen.identifier in constraints
            )
            first_sampling_in_constraints = next(samplings_in_constraints, None)
            if first_sampling_in_constraints is None:
                return False
            if next(samplings_in_constraints, None) is not None:
                return False
            sampling_to_check = first_sampling_in_constraints
        else:
            # Only one sampling to check
            sampling_to_check = self.sampled_from_list[0]

        if not isinstance(sampling_to_check.specimen, SampledSpecimen):
            # Sampling tree ends non-sampled specimen, is unambiguous
            return True

        # Update the constrain and check the parent specimen
        sub_constraint = set(constraints)
        if sampling_to_check.sampling_constraints is not None:
            sub_constraint.update(
                constraint.specimen.identifier
                for constraint in sampling_to_check.sampling_constraints
            )
        return sampling_to_check.specimen.is_sampling_constraint_is_unambiguous(
            sub_constraint
        )


@dataclass(frozen=True)
class Specimen(BaseSpecimen):
    """A specimen that has been extracted/taken from a patient in some way. Does not
    need to represent the actual first specimen in the collection tree, but should
    represent the first known (i.e. that we have metadata for) specimen in the
    collection tree.

    Parameters
    ----------
    identifier: str | SpecimenIdentifier
        The identifier of the specimen.
    extraction_step: Collection | None = None
        Optional collection step for the specimen.
    type: AnatomicPathologySpecimenTypesCode | None = None
        Optional type of the specimen. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8103.html
        or `AnatomicPathologySpecimenTypesCode.meanings` for allowed specimen types.
    steps: Sequence[PreparationStep] = []
        Optional sequence of preparation steps for the specimen.
    container: ContainerTypeCode | None = None
        Optional container type of the specimen. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8101.html
        or `ContainerTypeCode.meanings` for allowed container types.
    """

    identifier: str | SpecimenIdentifier
    extraction_step: Collection | None = None
    type: AnatomicPathologySpecimenTypesCode | None = None
    steps: list[PreparationStep] = field(default_factory=list)
    container: ContainerTypeCode | None = None

    def __post_init__(self):
        if self.extraction_step is not None and (
            len(self.steps) == 0
            or (len(self.steps) > 0 and self.steps[0] != self.extraction_step)
        ):
            self.steps.insert(
                0,
                self.extraction_step,
            )

    def add(self, step: PreparationStep) -> None:
        if isinstance(step, Collection) and len(self.steps) != 0:
            raise ValueError("A Collection-step must be the first step.")
        super().add(step)

    def sample(
        self,
        method: SpecimenSamplingProcedureCode | None = None,
        date_time: datetime.datetime | None = None,
        description: str | None = None,
        location: SamplingLocation | None = None,
    ) -> BaseSampling:
        """Create a sampling from the specimen that can be used to create a new
        sample."""
        if method is None or self.type is None:
            sampling = UnknownSampling(
                specimen=self,
                sampling_constraints=None,
            )
        else:
            sampling = Sampling(
                specimen=self,
                method=method,
                specimen_type=self.type,
                sampling_constraints=None,
                date_time=date_time,
                description=description,
                location=location,
            )
        self.add(sampling)
        return sampling


@dataclass(frozen=True)
class Sample(SampledSpecimen):
    """A specimen that has been sampled from one or more other specimens.

    Parameters
    ----------
    identifier: str | SpecimenIdentifier
        The identifier of the sample.
    sampled_from: Sequence[BaseSampling]
        Sequence of samplings that the sample was made from.
    type: AnatomicPathologySpecimenTypesCode | None = None
        Optional type of the sample. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8103.html
        or `AnatomicPathologySpecimenTypesCode.meanings` for allowed specimen types.
    steps: Sequence[PreparationStep] = []
        Optional sequence of preparation steps for the sample.
    container: ContainerTypeCode | None = None
        Optional container type of the sample. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8101.html
        or `ContainerTypeCode.meanings` for allowed container types.
    """

    identifier: str | SpecimenIdentifier
    # A sample is always sampled from a sequence of samplings: intentional
    # narrowing of the base's broader type that pyright's invariant
    # variable-override rule cannot express.
    sampled_from: Sequence[BaseSampling]  # pyright: ignore
    type: AnatomicPathologySpecimenTypesCode | None = None
    steps: list[PreparationStep] = field(default_factory=list)
    container: ContainerTypeCode | None = None

    def sample(
        self,
        method: SpecimenSamplingProcedureCode | None = None,
        date_time: datetime.datetime | None = None,
        description: str | None = None,
        sampling_constraints: Sequence[BaseSampling] | None = None,
        location: SamplingLocation | None = None,
    ) -> BaseSampling:
        """Create a sampling from the specimen that can be used to create a new
        sample."""
        self._check_sampling_constraints_in_sampling_tree(sampling_constraints)
        if sampling_constraints is not None:
            for sampling_constraint in sampling_constraints:
                assert isinstance(sampling_constraint, Sampling)

        if method is None or self.type is None:
            sampling = UnknownSampling(
                specimen=self,
                sampling_constraints=sampling_constraints,
            )
        else:
            sampling = Sampling(
                specimen=self,
                specimen_type=self.type,
                method=method,
                sampling_constraints=sampling_constraints,
                date_time=date_time,
                description=description,
                location=location,
            )
        self.add(sampling)
        return sampling


@dataclass(frozen=True)
class SlideSample(SampledSpecimen):
    """A sample that has been placed on a slide.

    Parameters
    ----------
    identifier: str | SpecimenIdentifier
        The identifier of the slide sample.
    anatomical_sites: Sequence[Code] | None = None
        Optional primary anatomic structures of interest in the slide sample.
    sampled_from: BaseSampling | None = None
        Optional sampling that the slide sample was made from.
    uid: UID | None = None
        Optional UID of the slide sample.
    localization: SampleLocalization | None = None
        Optional localization of the slide sample on the slide.
    steps: Sequence[PreparationStep] = []
        Optional sequence of preparation steps for the slide sample.
    short_description: str | None = None
        Optional short description of the slide sample. Should max be 64 characters.
    detailed_description: str | None = None
        Optional detailed description of the slide sample.
    """

    identifier: str | SpecimenIdentifier
    anatomical_sites: Sequence[Code] | None = None
    # A slide sample is sampled from at most one sampling, and always has a fixed
    # slide container and type: intentional narrowing of the base's broader types
    # that pyright's invariant variable-override rule cannot express.
    sampled_from: BaseSampling | None = None  # pyright: ignore
    uid: UID | None = None
    localization: SampleLocalization | None = None
    steps: list[PreparationStep] = field(default_factory=list)
    short_description: str | None = None
    detailed_description: str | None = None
    container: ContainerTypeCode = field(  # pyright: ignore
        init=False, default=ContainerTypeCode("Microscope slide")
    )
    type: AnatomicPathologySpecimenTypesCode = field(  # pyright: ignore
        init=False, default=AnatomicPathologySpecimenTypesCode("Slide")
    )

    def __post_init__(self):
        self._check_slide_sample_sampling_constraints()

    def _check_slide_sample_sampling_constraints(self):
        """Check that the sampling tree for the slide sample is not branching."""
        if self.sampled_from is None or not isinstance(
            self.sampled_from.specimen, SampledSpecimen
        ):
            # No sampling constraints possible
            return
        if not self.sampled_from.specimen.is_sampling_constraint_is_unambiguous(
            [
                constraint.specimen.identifier
                for constraint in self.sampled_from.sampling_constraints
            ]
            if self.sampled_from.sampling_constraints is not None
            else []
        ):
            raise ValueError("Sampling constraints for slide sample are ambiguous.")
