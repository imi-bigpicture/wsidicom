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
import logging
from abc import ABCMeta
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pydicom.sr.coding import Code
from pydicom.uid import UID

from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
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
    ExtractedSpecimen,
    Fixation,
    IssuerOfIdentifier,
    PreparationStep,
    Processing,
    Sample,
    SampledSpecimen,
    Sampling,
    SlideSample,
    SlideSamplePosition,
    Specimen,
    SpecimenIdentifier,
    Staining,
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
        cls, step: PreparationStep, specimen_identifier: Union[str, SpecimenIdentifier]
    ) -> "SpecimenPreparationStepDicomModel":
        if isinstance(step, Sampling):
            return SamplingDicomModel.from_step(step, specimen_identifier)
        if isinstance(step, Collection):
            return CollectionDicomModel.from_step(step, specimen_identifier)
        if isinstance(step, (Processing, Embedding, Fixation)):
            return ProcessingDicomModel.from_step(step, specimen_identifier)
        if isinstance(step, Staining):
            return StainingDicomModel.from_step(step, specimen_identifier)
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
    issuer_of_parent_specimen_identifier: Optional[str]
    parent_specimen_type: AnatomicPathologySpecimenTypesCode

    @classmethod
    def from_step(
        cls, sampling: Sampling, specimen_identifier: Union[str, SpecimenIdentifier]
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


@dataclass
class CollectionDicomModel(SpecimenPreparationStepDicomModel):
    method: SpecimenCollectionProcedureCode

    @classmethod
    def from_step(
        cls,
        collection: Collection,
        specimen_identifier: Union[str, SpecimenIdentifier],
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

    @classmethod
    def to_dicom_model(
        cls,
        slide_sample: SlideSample,
        stains: Optional[Sequence[Staining]] = None,
    ) -> "SpecimenDescriptionDicomModel":
        """Create a DICOM specimen description for the specimen."""
        if stains is None:
            stains = []
        if slide_sample.uid is None:
            sample_uid = slide_sample.default_uid
        else:
            sample_uid = slide_sample.uid
        sample_preparation_steps = cls._get_steps(slide_sample, stains)
        identifier, issuer = SpecimenIdentifier.get_identifier_and_issuer(
            slide_sample.identifier
        )
        if isinstance(slide_sample.position, str):
            position = slide_sample.position
        elif isinstance(slide_sample.position, SlideSamplePosition):
            position = slide_sample.position.to_tuple()
        else:
            position = None
        return cls(
            identifier=identifier,
            issuer_of_identifier=issuer,
            uid=sample_uid,
            steps=sample_preparation_steps,
            anatomical_sites=list(slide_sample.anatomical_sites)
            if slide_sample.anatomical_sites is not None
            else [],
            # specimen_type=slide_sample.type,
            short_description=slide_sample.short_description,
            detailed_description=slide_sample.detailed_description,
        )

    @classmethod
    def from_dicom_model(
        cls, specimen_descriptions: Iterable["SpecimenDescriptionDicomModel"]
    ) -> Tuple[Optional[List[SlideSample]], Optional[List[Staining]]]:
        """
        Parse specimen descriptions into samples and stainings.

        Parameters
        ----------
        specimen_descriptions: Iterable["SpecimenDescriptionDicomModel"]
            Specimen descriptions to parse.

        Returns
        ----------
        Optional[Tuple[List["SlideSample"], List[Staining]]]
            Samples and stainings parsed from descriptions.

        """

        created_specimens: Dict[
            Union[str, SpecimenIdentifier], Union[ExtractedSpecimen, Sample]
        ] = {}
        slide_samples: List[SlideSample] = []
        stainings: List[Staining] = []
        for specimen_description in specimen_descriptions:
            slide_sample = cls._create_slide_sample(
                specimen_description, created_specimens, stainings
            )
            slide_samples.append(slide_sample)

        return slide_samples, stainings

    @classmethod
    def _get_steps(
        cls, slide_sample: SlideSample, stainings: Optional[Iterable[Staining]]
    ) -> List[SpecimenPreparationStepDicomModel]:
        sample_preparation_steps = [
            step
            for sampling in slide_sample._sampled_from
            for step in cls._get_steps_for_sampling(sampling, slide_sample.identifier)
        ]
        if stainings is None:
            return sample_preparation_steps
        for staining in stainings:
            step = StainingDicomModel.from_step(staining, slide_sample.identifier)
            sample_preparation_steps.append(step)
        return sample_preparation_steps

    @classmethod
    def _get_steps_for_sampling(
        cls, sampling: Sampling, sample_identifier: Union[str, SpecimenIdentifier]
    ) -> List[SpecimenPreparationStepDicomModel]:
        """Return DICOM steps for the specimen the sample was sampled from."""
        if isinstance(sampling.specimen, SampledSpecimen):
            steps = [
                step
                for sampling in sampling.specimen._sampled_from
                if sampling.sampling_chain_constraints is None
                or sampling in sampling.sampling_chain_constraints
                for step in cls._get_steps_for_sampling(
                    sampling, sampling.specimen.identifier
                )
            ]
        else:
            steps = []
        steps.extend(
            SpecimenPreparationStepDicomModel.from_step(
                step, sampling.specimen.identifier
            )
            for step in cls._get_steps_before_sampling(sampling.specimen, sampling)
        )
        steps.append(SamplingDicomModel.from_step(sampling, sample_identifier))
        return steps

    @classmethod
    def _get_steps_before_sampling(
        cls, specimen: Specimen, sampling: Sampling
    ) -> Iterator[PreparationStep]:
        """Return the steps in specimen that occurred before the given sampling."""
        for step in specimen.steps:
            if isinstance(step, Sampling):
                # Break if sampling step for this sample, otherwise skip
                if step == sampling:
                    break
                continue
            yield step

    @classmethod
    def _parse_preparation_steps_for_specimen(
        cls,
        identifier: Union[str, SpecimenIdentifier],
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[Optional[SpecimenPreparationStepDicomModel]],
        ],
        existing_specimens: Dict[
            Union[str, SpecimenIdentifier], Union[ExtractedSpecimen, Sample]
        ],
        stop_at_step: Optional[SpecimenPreparationStepDicomModel] = None,
    ) -> Tuple[List[PreparationStep], List[Sampling]]:
        """
        Parse preparation steps and samplings for a specimen.

        Creates or updates parent specimens.

        Parameters
        ----------
        identifier: Union[str, SpecimenIdentifier]
            The identifier of the specimen to parse.
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[Optional[SpecimenPreparationStepDicomModel]],
        ]
            DICOM preparation steps ordered by specimen identifier.
        existing_specimens: Dict[
            Union[str, SpecimenIdentifier], Union[ExtractedSpecimen, Sample]
        ]
            Existing specimens ordered by specimen identifier.
        stop_at_step: Optional[SpecimenPreparationStepDicomModel] = None:
            Step in the list of steps for this identifier at which the list should not
            be processed further.

        Returns
        ----------
        Tuple[List[PreparationStep], List[Sampling]]
            Parsed PreparationSteps and Samplings for the specimen.

        """
        if stop_at_step is not None:
            if (
                not isinstance(stop_at_step, SamplingDicomModel)
                or stop_at_step.get_parent_identifier() != identifier
            ):
                raise ValueError(
                    "Stop at step should be a parent SpecimenSampling step."
                )

        samplings: List[Sampling] = []
        preparation_steps: List[PreparationStep] = []
        for index, step in enumerate(steps_by_identifier[identifier]):
            if stop_at_step is not None and stop_at_step == step:
                # We should not parse the rest of the list
                break
            if step is None:
                # This step has already been parsed, skip to next.
                continue
            if step.identifier != identifier:
                # This is OK if SpecimenSampling with matching parent identifier
                if (
                    not isinstance(step, SamplingDicomModel)
                    or step.get_parent_identifier() != identifier
                ):
                    error = (
                        f"Got step of unexpected type {type(step)}"
                        f"or identifier {step.identifier} for specimen {identifier}"
                    )
                    raise ValueError(error)
                # Skip to next
                continue

            if isinstance(step, StainingDicomModel):
                # Stainings are handled elsewhere
                pass
            elif isinstance(step, CollectionDicomModel):
                any_sampling_steps = any(
                    sampling_step
                    for sampling_step in steps_by_identifier[identifier]
                    if sampling_step is not None
                    and isinstance(sampling_step, SamplingDicomModel)
                    and sampling_step.identifier == identifier
                )
                if index != 0 or any_sampling_steps:
                    raise ValueError(
                        (
                            "Collection step should be first step and there should not "
                            "be any sampling steps."
                        )
                    )
                preparation_steps.append(
                    Collection(
                        step.method,
                        date_time=step.date_time,
                        description=step.description,
                    )
                )
            elif isinstance(step, ProcessingDicomModel):
                pass

            elif isinstance(step, SamplingDicomModel):
                parent_identifier = step.get_parent_identifier()
                if parent_identifier in existing_specimens:
                    # Parent already exists. Parse any non-parsed steps
                    parent = existing_specimens[parent_identifier]
                    (
                        parent_steps,
                        sampling_constraints,
                    ) = cls._parse_preparation_steps_for_specimen(
                        parent_identifier, steps_by_identifier, existing_specimens, step
                    )
                    for parent_step in parent_steps:
                        # Only add step if an equivalent does not exists
                        if not any(step == parent_step for step in parent.steps):
                            parent.add(parent_step)
                    if isinstance(parent, Sample):
                        parent._sampled_from.extend(sampling_constraints)
                else:
                    # Need to create parent
                    parent = cls._create_specimen(
                        parent_identifier,
                        step.parent_specimen_type,
                        steps_by_identifier,
                        existing_specimens,
                        step,
                    )
                    if isinstance(parent, Sample):
                        sampling_constraints = parent.sampled_from
                    else:
                        sampling_constraints = None
                    existing_specimens[parent_identifier] = parent

                if isinstance(parent, Sample):
                    # If Sample create sampling with constraint
                    sampling = parent.sample(
                        method=step.method,
                        date_time=step.date_time,
                        description=step.description,
                        sampling_chain_constraints=sampling_constraints,
                    )
                else:
                    # Extracted specimen can not have constraint
                    sampling = parent.sample(
                        method=step.method,
                        date_time=step.date_time,
                        description=step.description,
                    )

                samplings.append(sampling)
            else:
                raise NotImplementedError(f"Step of type {type(step)}")
            if step.fixative is not None:
                preparation_steps.append(
                    Fixation(
                        fixative=step.fixative,
                        date_time=step.date_time,
                        description=step.description,
                    )
                )
            if step.embedding is not None:
                preparation_steps.append(
                    Embedding(
                        medium=step.embedding,
                        date_time=step.date_time,
                        description=step.description,
                    )
                )
            if step.processing is not None:
                preparation_steps.append(
                    Processing(
                        step.processing,
                        date_time=step.date_time,
                        description=step.description,
                    )
                )

            # Clear this step so that it will not be processed again
            steps_by_identifier[identifier][index] = None
        return preparation_steps, samplings

    @classmethod
    def _create_specimen(
        cls,
        identifier: Union[str, SpecimenIdentifier],
        specimen_type: AnatomicPathologySpecimenTypesCode,
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[Optional[SpecimenPreparationStepDicomModel]],
        ],
        existing_specimens: Dict[
            Union[str, SpecimenIdentifier], Union[ExtractedSpecimen, Sample]
        ],
        stop_at_step: SpecimenPreparationStepDicomModel,
    ) -> Union[ExtractedSpecimen, Sample]:
        """
        Create an ExtractedSpecimen or Sample.

        Parameters
        ----------
        identifier: Union[str, SpecimenIdentifier]
            The identifier of the specimen to create.
        specimen_type: AnatomicPathologySpecimenTypesCode
            The coded type of the specimen to create.
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[Optional[SpecimenPreparationStepDicomModel]],
        ]
            DICOM preparation steps ordered by specimen identifier.
        existing_specimens: Dict[
            Union[str, SpecimenIdentifier], Union[ExtractedSpecimen, Sample]
        ]
            Existing specimens ordered by specimen identifier.
        stop_at_step: SpecimenPreparationStepDicomModel,
            Stop processing steps for this specimen at this step in the list.

        Returns
        ----------
        Union[ExtractedSpecimen, Sample]
            Created ExtracedSpecimen, if the specimen has no parents, or Specimen.

        """
        logging.debug(f"Creating specimen with identifier {identifier}")
        preparation_steps, samplings = cls._parse_preparation_steps_for_specimen(
            identifier, steps_by_identifier, existing_specimens, stop_at_step
        )

        if len(samplings) == 0:
            return ExtractedSpecimen(
                identifier=identifier,
                type=specimen_type,
                steps=preparation_steps,
            )
        return Sample(
            identifier=identifier,
            type=specimen_type,
            sampled_from=samplings,
            steps=preparation_steps,
        )

    @classmethod
    def _create_slide_sample(
        cls,
        description: "SpecimenDescriptionDicomModel",
        existing_specimens: Dict[
            Union[str, SpecimenIdentifier], Union[ExtractedSpecimen, Sample]
        ],
        existing_stainings: List[Staining],
    ) -> SlideSample:
        """
        Create a SlideSample from Specimen Description.

        Contained parent specimens and stainings are created or updated.

        Parameters
        ----------
        description: SpecimenDescriptionDicomModel,
            Specimen description to parse.
        existing_specimens: Dict[
            Union[str, SpecimenIdentifier], Union[ExtractedSpecimen, Sample]
        ]
            Dictionary with existing specimens. New/updated specimens this Specimen
            Description are updated/added.
        existing_stainings: List[Staining]
            List of existing stainings. New stainings from this Specimen Description are
            added.

        Returns
        ----------
        SlideSample
            Parsed sample.

        """
        # Sort the steps based on specimen identifier.
        # Sampling steps are put into to both sampled and parent bucket.
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[Optional[SpecimenPreparationStepDicomModel]],
        ] = defaultdict(list)
        for step in description.steps:
            if isinstance(step, StainingDicomModel):
                staining = Staining(
                    step.substances,
                    date_time=step.date_time,
                    description=step.description,
                )
                if not any(staining == existing for existing in existing_stainings):
                    existing_stainings.append(staining)
            elif isinstance(step, SamplingDicomModel):
                parent_identifier = step.parent_specimen_identifier
                steps_by_identifier[parent_identifier].append(step)
            identifier = step.get_identifier()
            steps_by_identifier[identifier].append(step)

        identifier = cls._get_identifier_from_description(description)
        preparation_steps, samplings = cls._parse_preparation_steps_for_specimen(
            identifier, steps_by_identifier, existing_specimens
        )

        if len(samplings) > 1:
            raise ValueError("Should be max one sampling, got.", len(samplings))
        # TODO add position when highdicom support
        return SlideSample(
            identifier=identifier,
            anatomical_sites=[],
            sampled_from=next(iter(samplings), None),
            uid=description.uid,
            # position=
            steps=preparation_steps,
        )

    @classmethod
    def _get_identifier_from_description(
        cls, description: "SpecimenDescriptionDicomModel"
    ) -> Union[str, SpecimenIdentifier]:
        if description.issuer_of_identifier is None:
            return description.identifier
        return SpecimenIdentifier(
            description.identifier, description.issuer_of_identifier
        )
