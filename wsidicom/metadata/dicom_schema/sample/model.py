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
from abc import ABCMeta
from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Union,
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
    Collection,
    Embedding,
    Fixation,
    IssuerOfIdentifier,
    Measurement,
    PreparationStep,
    Processing,
    Receiving,
    Sampling,
    SamplingLocation,
    Specimen,
    SpecimenIdentifier,
    SpecimenLocalization,
    Staining,
    Storage,
)


@dataclass
class SpecimenPreparationStepDicomModel(metaclass=ABCMeta):
    identifier: str
    issuer_of_identifier: Optional[str]
    date_time: Optional[datetime.datetime]
    description: Optional[str]
    fixative: Optional[SpecimenFixativesCode]
    embedding: Optional[SpecimenEmbeddingMediaCode]
    processing: Optional[SpecimenPreparationStepsCode]

    @classmethod
    def from_step(
        cls,
        step: PreparationStep,
        specimen: Specimen,
    ) -> "SpecimenPreparationStepDicomModel":
        if isinstance(step, Sampling):
            return SamplingDicomModel.from_step(
                step, specimen.identifier, specimen.container
            )
        if isinstance(step, Collection):
            return CollectionDicomModel.from_step(
                step, specimen.identifier, specimen.container
            )
        if isinstance(step, (Processing, Embedding, Fixation)):
            return ProcessingDicomModel.from_step(step, specimen.identifier)
        if isinstance(step, Staining):
            return StainingDicomModel.from_step(step, specimen.identifier)
        raise NotImplementedError()

    def get_identifier(self) -> Union[str, SpecimenIdentifier]:
        if self.issuer_of_identifier is None:
            return self.identifier
        return SpecimenIdentifier(
            self.identifier,
            IssuerOfIdentifier.from_hl7v2(self.issuer_of_identifier),
        )


@dataclass
class SamplingDicomModel(SpecimenPreparationStepDicomModel):
    method: SpecimenSamplingProcedureCode
    parent_specimen_identifier: str
    parent_specimen_type: AnatomicPathologySpecimenTypesCode
    issuer_of_parent_specimen_identifier: Optional[str]
    location_reference: Optional[str] = None
    location_description: Optional[str] = None
    location_x: Optional[Measurement] = None
    location_y: Optional[Measurement] = None
    location_z: Optional[Measurement] = None
    container: Optional[ContainerTypeCode] = None

    @classmethod
    def from_step(
        cls,
        sampling: Sampling,
        specimen_identifier: Union[str, SpecimenIdentifier],
        container: Optional[ContainerTypeCode],
    ) -> "SamplingDicomModel":
        """Return Dicom dataset for the step.

        Parameters
        ----------
        sampling: Sampling
            Step to convert into dataset.
        specimen_identifier: Union[str, SpecimenIdentifier]:
            Identifier for the specimen that was processed.

        Parameters
        ----------
        SpecimenPreparationStep:
            Dicom dataset describing the processing step.

        """
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen_identifier
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
            fixative=None,
            embedding=None,
            processing=None,
            location_reference=sampling.location.reference
            if sampling.location is not None
            else None,
            location_description=sampling.location.description
            if sampling.location is not None
            else None,
            location_x=sampling.location.x if sampling.location is not None else None,
            location_y=sampling.location.y if sampling.location is not None else None,
            location_z=sampling.location.z if sampling.location is not None else None,
            container=container,
        )

    def get_parent_identifier(self) -> Union[str, SpecimenIdentifier]:
        if (
            self.issuer_of_parent_specimen_identifier is not None
            and self.issuer_of_parent_specimen_identifier != ""
        ):
            return SpecimenIdentifier(
                self.parent_specimen_identifier,
                IssuerOfIdentifier.from_hl7v2(
                    self.issuer_of_parent_specimen_identifier
                ),
            )
        return self.parent_specimen_identifier

    @property
    def sampling_location(self) -> Optional[SamplingLocation]:
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
    method: SpecimenCollectionProcedureCode
    container: Optional[ContainerTypeCode] = None

    @classmethod
    def from_step(
        cls,
        collection: Collection,
        specimen_identifier: Union[str, SpecimenIdentifier],
        container: Optional[ContainerTypeCode],
    ) -> "CollectionDicomModel":
        """Return Dicom dataset for the step.

        Parameters
        ----------
        collection: Collection
            Step to convert into dataset.
        specimen_identifier: Union[str, SpecimenIdentifier]:
            Identifier for the specimen that was processed.

        Parameters
        ----------
        SpecimenPreparationStep:
            Dicom dataset describing the processing step.

        """
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen_identifier
        )
        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            date_time=collection.date_time,
            description=collection.description,
            method=collection.method,
            fixative=None,
            embedding=None,
            processing=None,
            container=container,
        )


@dataclass
class ReceivingDicomModel(SpecimenPreparationStepDicomModel):
    @classmethod
    def from_step(
        cls,
        receiving: Receiving,
        specimen_identifier: Union[str, SpecimenIdentifier],
    ):
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen_identifier
        )
        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            date_time=receiving.date_time,
            description=receiving.description,
            fixative=None,
            embedding=None,
            processing=None,
        )


@dataclass
class StorageDicomModel(SpecimenPreparationStepDicomModel):
    @classmethod
    def from_step(
        cls,
        storage: Storage,
        specimen_identifier: Union[str, SpecimenIdentifier],
    ):
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen_identifier
        )
        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            date_time=storage.date_time,
            description=storage.description,
            fixative=None,
            embedding=None,
            processing=None,
        )


@dataclass
class ProcessingDicomModel(SpecimenPreparationStepDicomModel):
    @classmethod
    def from_step(
        cls,
        processing: Union[Processing, Embedding, Fixation],
        specimen_identifier: Union[str, SpecimenIdentifier],
    ) -> "ProcessingDicomModel":
        """Return Dicom dataset for the step.

        Parameters
        ----------
        processing: Processing
            Step to convert into dataset.
        specimen_identifier: Union[str, SpecimenIdentifier]:
            Identifier for the specimen that was processed.

        Parameters
        ----------
        SpecimenPreparationStep:
            Dicom dataset describing the processing step.

        """
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen_identifier
        )
        if isinstance(processing, Processing):
            method = processing.method
        else:
            method = None
        if isinstance(processing, Fixation):
            fixative = processing.fixative
        else:
            fixative = None
        if isinstance(processing, Embedding):
            embedding = processing.medium
        else:
            embedding = None

        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            date_time=processing.date_time,
            description=processing.description,
            fixative=fixative,
            embedding=embedding,
            processing=method,
        )


@dataclass
class StainingDicomModel(SpecimenPreparationStepDicomModel):
    substances: List[Union[str, SpecimenStainsCode]]

    @classmethod
    def from_step(
        cls, staining: Staining, specimen_identifier: Union[str, SpecimenIdentifier]
    ) -> "StainingDicomModel":
        """Return Dicom dataset for the step.

        Parameters
        ----------
        specimen_identifier: Union[str, SpecimenIdentifier]:
            Identifier for the specimen that was processed.

        Parameters
        ----------
        SpecimenPreparationStep:
            Dicom dataset describing the processing step.

        """
        identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(
            specimen_identifier
        )
        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            date_time=staining.date_time,
            description=staining.description,
            substances=staining.substances,
            fixative=None,
            embedding=None,
            processing=None,
        )


@dataclass
class SpecimenDescriptionDicomModel:
    """A sample that has been placed on a slide."""

    identifier: str
    uid: UID
    steps: List[SpecimenPreparationStepDicomModel]
    # specimen_type: AnatomicPathologySpecimenTypesCode
    anatomical_sites: List[Code]
    issuer_of_identifier: Optional[IssuerOfIdentifier] = None
    short_description: Optional[str] = None
    detailed_description: Optional[str] = None
    localization: Optional[SpecimenLocalization] = None

    def get_identifier(self) -> Union[str, SpecimenIdentifier]:
        if self.issuer_of_identifier is None:
            return self.identifier
        return SpecimenIdentifier(self.identifier, self.issuer_of_identifier)
