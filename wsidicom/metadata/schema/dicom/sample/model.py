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

"""Models for use during DICOM specimen serialization."""

import datetime
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import (
    Optional,
)

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
)
from wsidicom.metadata.sample import (
    BaseSpecimen,
    Collection,
    Embedding,
    Fixation,
    IssuerOfIdentifier,
    Measurement,
    PreparationStep,
    Processing,
    Receiving,
    SampleLocalization,
    Sampling,
    SamplingLocation,
    SpecimenIdentifier,
    Staining,
    Storage,
    UnknownSampling,
)


class SpecimenPreparationStepDicomModel:
    def __init__(
        self,
        identifier: str,
        issuer_of_identifier: str | None = None,
        date_time: datetime.datetime | None = None,
        description: str | None = None,
        fixative: SpecimenFixativesCode | None = None,
        embedding: SpecimenEmbeddingMediaCode | None = None,
        processing: SpecimenPreparationStepsCode | None = None,
        specimen_type: AnatomicPathologySpecimenTypesCode | None = None,
        container: ContainerTypeCode | None = None,
    ):
        self.identifier = identifier
        self.issuer_of_identifier = issuer_of_identifier
        self.date_time = date_time
        self.description = description
        self.fixative = fixative
        self.embedding = embedding
        self.processing = processing
        self.specimen_type = specimen_type
        self.container = container

    @classmethod
    def from_step(
        cls,
        step: PreparationStep,
        specimen: BaseSpecimen,
    ) -> Optional["SpecimenPreparationStepDicomModel"]:
        """Return DICOM model for the step.

        Parameters
        ----------
        step: PreparationStep
            Step to convert into DICOM model.
        specimen: BaseSpecimen
            Specimen that was processed.

        Returns
        -------
        SpecimenPreparationStepDicomModel | None
            DICOM model for the step, or None if no step should be produced.
        """
        if isinstance(step, Sampling):
            if step.specimen.type is None:
                return None
            return SamplingDicomModel.from_step(step, specimen)
        if isinstance(step, UnknownSampling):
            return None
        if isinstance(step, Collection):
            return CollectionDicomModel.from_step(step, specimen)
        if isinstance(step, (Processing, Embedding, Fixation)):
            return ProcessingDicomModel.from_step(step, specimen)
        if isinstance(step, Staining):
            return StainingDicomModel.from_step(step, specimen)
        if isinstance(step, Receiving):
            return ReceivingDicomModel.from_step(step, specimen)
        if isinstance(step, Storage):
            return StorageDicomModel.from_step(step, specimen)

        raise NotImplementedError(f"Unknown preparation step type {type(step)}.")

    @property
    def specimen_identifier(self) -> SpecimenIdentifier:
        if self.issuer_of_identifier is None:
            issuer = None
        else:
            issuer = IssuerOfIdentifier.from_hl7v2(self.issuer_of_identifier)
        return SpecimenIdentifier(
            self.identifier,
            issuer,
        )


@dataclass
class SamplingDicomModel(SpecimenPreparationStepDicomModel):
    identifier: str
    method: SpecimenSamplingProcedureCode
    parent_specimen_identifier: str
    parent_specimen_type: AnatomicPathologySpecimenTypesCode
    issuer_of_identifier: str | None = None
    issuer_of_parent_specimen_identifier: str | None = None
    date_time: datetime.datetime | None = None
    description: str | None = None
    fixative: SpecimenFixativesCode | None = None
    embedding: SpecimenEmbeddingMediaCode | None = None
    processing: SpecimenPreparationStepsCode | None = None
    specimen_type: AnatomicPathologySpecimenTypesCode | None = None
    container: ContainerTypeCode | None = None
    location_reference: str | None = None
    location_description: str | None = None
    location_x: Measurement | None = None
    location_y: Measurement | None = None
    location_z: Measurement | None = None

    def __post_init__(self):
        super().__init__(
            identifier=self.identifier,
            issuer_of_identifier=self.issuer_of_identifier,
            date_time=self.date_time,
            description=self.description,
            fixative=self.fixative,
            embedding=self.embedding,
            processing=self.processing,
            specimen_type=self.specimen_type,
            container=self.container,
        )

    @classmethod
    def from_step(
        cls,
        sampling: Sampling,
        specimen: BaseSpecimen,
    ) -> "SamplingDicomModel":
        """Return Dicom dataset for the step.

        Parameters
        ----------
        sampling: Sampling
            Step to convert into dataset.
        specimen_identifier: str | SpecimenIdentifier:
            Identifier for the specimen that was processed.

        Parameters
        ----------
        SpecimenPreparationStep:
            Dicom dataset describing the processing step.

        """
        if sampling.specimen.type is None:
            raise ValueError("Sampled specimen must have a specimen type.")
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen.identifier
        )
        (
            parent_identifier,
            parent_issuer,
        ) = SpecimenIdentifier.get_string_identifier_and_issuer(
            sampling.specimen.identifier
        )
        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            date_time=sampling.date_time,
            description=sampling.description,
            method=sampling.method,
            parent_specimen_identifier=parent_identifier,
            issuer_of_parent_specimen_identifier=parent_issuer,
            parent_specimen_type=sampling.specimen.type,
            location_reference=(
                sampling.location.reference if sampling.location is not None else None
            ),
            location_description=(
                sampling.location.description if sampling.location is not None else None
            ),
            location_x=sampling.location.x if sampling.location is not None else None,
            location_y=sampling.location.y if sampling.location is not None else None,
            location_z=sampling.location.z if sampling.location is not None else None,
            container=specimen.container,
            specimen_type=specimen.type,
        )

    @property
    def parent_identifier(self) -> SpecimenIdentifier:
        if self.issuer_of_parent_specimen_identifier is None:
            issuer = None
        else:
            issuer = IssuerOfIdentifier.from_hl7v2(
                self.issuer_of_parent_specimen_identifier
            )
        return SpecimenIdentifier(
            self.parent_specimen_identifier,
            issuer,
        )

    @property
    def sampling_location(self) -> SamplingLocation | None:
        if any(
            item is not None
            for item in (
                self.location_reference,
                self.location_description,
                self.location_x,
                self.location_y,
                self.location_z,
            )
        ):
            return SamplingLocation(
                reference=self.location_reference,
                description=self.location_description,
                x=self.location_x,
                y=self.location_y,
                z=self.location_z,
            )
        return None


@dataclass
class CollectionDicomModel(SpecimenPreparationStepDicomModel):
    identifier: str
    method: SpecimenCollectionProcedureCode
    issuer_of_identifier: str | None = None
    date_time: datetime.datetime | None = None
    description: str | None = None
    fixative: SpecimenFixativesCode | None = None
    embedding: SpecimenEmbeddingMediaCode | None = None
    processing: SpecimenPreparationStepsCode | None = None
    specimen_type: AnatomicPathologySpecimenTypesCode | None = None
    container: ContainerTypeCode | None = None

    def __post_init__(self):
        super().__init__(
            identifier=self.identifier,
            issuer_of_identifier=self.issuer_of_identifier,
            date_time=self.date_time,
            description=self.description,
            fixative=self.fixative,
            embedding=self.embedding,
            processing=self.processing,
            specimen_type=self.specimen_type,
            container=self.container,
        )

    @classmethod
    def from_step(
        cls,
        collection: Collection,
        specimen: BaseSpecimen,
    ) -> "CollectionDicomModel":
        """Return Dicom dataset for the step.

        Parameters
        ----------
        collection: Collection
            Step to convert into dataset.
        specimen_identifier: str | SpecimenIdentifier:
            Identifier for the specimen that was processed.

        Parameters
        ----------
        SpecimenPreparationStep:
            Dicom dataset describing the processing step.

        """
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen.identifier
        )
        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            date_time=collection.date_time,
            description=collection.description,
            method=collection.method,
            container=specimen.container,
            specimen_type=specimen.type,
        )


@dataclass
class ReceivingDicomModel(SpecimenPreparationStepDicomModel):
    identifier: str
    issuer_of_identifier: str | None = None
    date_time: datetime.datetime | None = None
    description: str | None = None
    fixative: SpecimenFixativesCode | None = None
    embedding: SpecimenEmbeddingMediaCode | None = None
    processing: SpecimenPreparationStepsCode | None = None
    specimen_type: AnatomicPathologySpecimenTypesCode | None = None
    container: ContainerTypeCode | None = None

    @classmethod
    def from_step(
        cls,
        receiving: Receiving,
        specimen: BaseSpecimen,
    ):
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen.identifier
        )
        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            date_time=receiving.date_time,
            description=receiving.description,
            container=specimen.container,
            specimen_type=specimen.type,
        )


@dataclass
class StorageDicomModel(SpecimenPreparationStepDicomModel):
    identifier: str
    issuer_of_identifier: str | None = None
    date_time: datetime.datetime | None = None
    description: str | None = None
    fixative: SpecimenFixativesCode | None = None
    embedding: SpecimenEmbeddingMediaCode | None = None
    processing: SpecimenPreparationStepsCode | None = None
    specimen_type: AnatomicPathologySpecimenTypesCode | None = None
    container: ContainerTypeCode | None = None

    @classmethod
    def from_step(
        cls,
        storage: Storage,
        specimen: BaseSpecimen,
    ):
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen.identifier
        )
        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            date_time=storage.date_time,
            description=storage.description,
            container=specimen.container,
            specimen_type=specimen.type,
        )


@dataclass
class ProcessingDicomModel(SpecimenPreparationStepDicomModel):
    identifier: str
    issuer_of_identifier: str | None = None
    date_time: datetime.datetime | None = None
    description: str | None = None
    fixative: SpecimenFixativesCode | None = None
    embedding: SpecimenEmbeddingMediaCode | None = None
    processing: SpecimenPreparationStepsCode | None = None
    specimen_type: AnatomicPathologySpecimenTypesCode | None = None
    container: ContainerTypeCode | None = None

    @classmethod
    def from_step(
        cls,
        processing: Processing | Embedding | Fixation,
        specimen: BaseSpecimen,
    ) -> "ProcessingDicomModel":
        """Return Dicom dataset for the step.

        Parameters
        ----------
        processing: Processing
            Step to convert into dataset.
        specimen_identifier: str | SpecimenIdentifier:
            Identifier for the specimen that was processed.

        Parameters
        ----------
        SpecimenPreparationStep:
            Dicom dataset describing the processing step.

        """
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen.identifier
        )
        method = processing.method if isinstance(processing, Processing) else None
        fixative = processing.fixative if isinstance(processing, Fixation) else None
        embedding = processing.medium if isinstance(processing, Embedding) else None

        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            date_time=processing.date_time,
            description=processing.description,
            fixative=fixative,
            embedding=embedding,
            processing=method,
            container=specimen.container,
            specimen_type=specimen.type,
        )


@dataclass
class StainingDicomModel(SpecimenPreparationStepDicomModel):
    identifier: str
    substances: str | Sequence[SpecimenStainsCode]
    issuer_of_identifier: str | None = None
    date_time: datetime.datetime | None = None
    description: str | None = None
    fixative: SpecimenFixativesCode | None = None
    embedding: SpecimenEmbeddingMediaCode | None = None
    processing: SpecimenPreparationStepsCode | None = None
    specimen_type: AnatomicPathologySpecimenTypesCode | None = None
    container: ContainerTypeCode | None = None

    @classmethod
    def from_step(
        cls,
        staining: Staining,
        specimen: BaseSpecimen,
    ) -> "StainingDicomModel":
        """Return Dicom dataset for the step.

        Parameters
        ----------
        specimen_identifier: str | SpecimenIdentifier:
            Identifier for the specimen that was processed.

        Parameters
        ----------
        SpecimenPreparationStep:
            Dicom dataset describing the processing step.

        """
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen.identifier
        )
        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            date_time=staining.date_time,
            description=staining.description,
            substances=staining.substances,
            container=specimen.container,
            specimen_type=specimen.type,
        )


@dataclass
class SpecimenDescriptionDicomModel:
    """A sample that has been placed on a slide."""

    identifier: str
    uid: UID
    steps: list[SpecimenPreparationStepDicomModel] = field(default_factory=list)
    anatomical_sites: list[Code] = field(default_factory=list)
    issuer_of_identifier: IssuerOfIdentifier | None = None
    short_description: str | None = None
    detailed_description: str | None = None
    localization: SampleLocalization | None = None

    @property
    def specimen_identifier(self) -> SpecimenIdentifier:
        return SpecimenIdentifier(self.identifier, self.issuer_of_identifier)
