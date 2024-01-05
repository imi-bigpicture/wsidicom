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

"""Module with DICOM specimen description parser."""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    ContainerTypeCode,
)
from wsidicom.metadata.sample import (
    BaseSampling,
    BaseSpecimen,
    Collection,
    Embedding,
    Fixation,
    PreparationStep,
    Processing,
    Receiving,
    Sample,
    SlideSample,
    Specimen,
    SpecimenIdentifier,
    Staining,
    Storage,
)
from wsidicom.metadata.schema.dicom.sample.model import (
    CollectionDicomModel,
    ProcessingDicomModel,
    ReceivingDicomModel,
    SamplingDicomModel,
    SpecimenDescriptionDicomModel,
    SpecimenPreparationStepDicomModel,
    StainingDicomModel,
    StorageDicomModel,
)


@dataclass
class ParsedSpecimen:
    """Parsed specimen steps."""

    preparation_steps: List[PreparationStep]
    sampling: Optional[BaseSampling]
    container: Optional[ContainerTypeCode]
    specimen_type: Optional[AnatomicPathologySpecimenTypesCode]


class SpecimenDicomParser:
    def __init__(self):
        self._created_specimens: Dict[Union[str, SpecimenIdentifier], BaseSpecimen] = {}
        self._created_stainings: List[Staining] = []

    def parse_descriptions(
        self, specimen_descriptions: Iterable[SpecimenDescriptionDicomModel]
    ) -> Tuple[List[SlideSample], List[Staining]]:
        """
        Parse specimen descriptions into samples and stainings.

        Parameters
        ----------
        specimen_descriptions: Iterable["SpecimenDescriptionDicomModel"]
            Specimen descriptions to parse.

        Returns
        ----------
        Tuple[List[SlideSample], List[Staining]]
            Samples and stainings parsed from descriptions.

        """

        slide_samples: List[SlideSample] = []
        for specimen_description in specimen_descriptions:
            slide_sample = self._create_slide_sample(specimen_description)
            slide_samples.append(slide_sample)

        return slide_samples, self._created_stainings

    def _create_slide_sample(
        self, description: SpecimenDescriptionDicomModel
    ) -> SlideSample:
        """
        Create a SlideSample from Specimen Description.

        Contained parent specimens and stainings are created or updated.

        Parameters
        ----------
        description: SpecimenDescriptionDicomModel,
            Specimen description to parse.

        Returns
        ----------
        SlideSample
            Parsed sample.

        """
        steps_by_identifier = self._order_steps_by_identifier(description.steps)
        stainings = self._parse_stainings(description.steps)
        for staining in stainings:
            if not any(
                existing_staining == staining
                for existing_staining in self._created_stainings
            ):
                self._created_stainings.append(staining)
        identifier = description.specimen_identifier
        parsed_specimen = self._parse_preparation_steps_for_specimen(
            identifier, steps_by_identifier
        )
        return SlideSample(
            identifier=identifier,
            anatomical_sites=description.anatomical_sites,
            sampled_from=parsed_specimen.sampling,
            uid=description.uid,
            localization=description.localization,
            steps=parsed_specimen.preparation_steps,
        )

    def _parse_stainings(
        self,
        steps: List[SpecimenPreparationStepDicomModel],
    ) -> List[Staining]:
        """
        Parse stainings from steps.

        Parameters
        ----------
        steps: List[SpecimenPreparationStepDicomModel]
            Steps to parse.

        Returns
        ----------
        List[Staining]
            Parsed stainings.

        """
        return [
            Staining(
                step.substances, date_time=step.date_time, description=step.description
            )
            for step in steps
            if isinstance(step, StainingDicomModel)
        ]

    def _order_steps_by_identifier(
        self,
        steps: List[SpecimenPreparationStepDicomModel],
    ) -> Dict[Union[str, SpecimenIdentifier], List[SpecimenPreparationStepDicomModel],]:
        """
        Parse steps into dictionary of steps by specimen identifier.

        Parameters
        ----------
        steps: List[SpecimenPreparationStepDicomModel]
            Steps to parse.

        Returns
        -------
        Dict[
            Union[str, SpecimenIdentifier],
            List[SpecimenPreparationStepDicomModel],
        ]:
            Steps ordered by specimen identifier.
        """
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[SpecimenPreparationStepDicomModel],
        ] = defaultdict(list)
        for step in steps:
            identifier = step.specimen_identifier
            steps_by_identifier[identifier].append(step)
        return steps_by_identifier

    def _parse_preparation_steps_for_specimen(
        self,
        identifier: Union[str, SpecimenIdentifier],
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[SpecimenPreparationStepDicomModel],
        ],
    ) -> ParsedSpecimen:
        """
        Parse preparation steps and samplings for a specimen.

        Creates or updates parent specimens.

        Parameters
        ----------
        identifier: Union[str, SpecimenIdentifier]
            The identifier of the specimen to parse.
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[SpecimenPreparationStepDicomModel],
        ]
            DICOM preparation steps ordered by specimen identifier.
        stop_at_step: Optional[SpecimenPreparationStepDicomModel] = None:
            Step in the list of steps for this identifier at which the list should not
            be processed further.

        Returns
        ----------
        ParsedSpecimen
            Parsed steps for specimen.

        """
        container: Optional[ContainerTypeCode] = None
        specimen_type: Optional[AnatomicPathologySpecimenTypesCode] = None
        sampling: Optional[BaseSampling] = None
        preparation_steps: List[PreparationStep] = []
        for index, step in enumerate(steps_by_identifier[identifier]):
            if step is None:
                # This step has already been parsed, skip to next.
                continue
            if step.specimen_identifier != identifier:
                # This is OK if SpecimenSampling with matching parent identifier
                if (
                    not isinstance(step, SamplingDicomModel)
                    or step.parent_identifier != identifier
                ):
                    error = (
                        f"Got step of unexpected type {type(step)}"
                        f"or identifier {step.identifier} for specimen {identifier}"
                    )
                    raise ValueError(error)
                # Skip to next
                continue
            parsed_step = self._parse_step(index, step, identifier, steps_by_identifier)
            preparation_steps.extend(parsed_step.preparation_steps)
            if parsed_step.sampling is not None:
                if sampling is not None:
                    raise ValueError(
                        "Should not be more than one sampling for a specimen."
                    )
                sampling = parsed_step.sampling

            # Update container and specimen type
            if step.specimen_type is not None:
                if specimen_type is not None and specimen_type != step.specimen_type:
                    raise ValueError(
                        f"Got mismatching specimen types {specimen_type} and "
                        f"{step.specimen_type} in steps for specimen {identifier}."
                    )
                specimen_type = step.specimen_type
            if step.container is not None:
                if container is not None and container != step.container:
                    raise ValueError(
                        f"Got mismatching container types {container} and "
                        f"{step.container} in steps for specimen {identifier}."
                    )
                container = step.container
        return ParsedSpecimen(
            preparation_steps=preparation_steps,
            sampling=sampling,
            container=container,
            specimen_type=specimen_type,
        )

    def _parse_step(
        self,
        index: int,
        step: SpecimenPreparationStepDicomModel,
        identifier: Union[str, SpecimenIdentifier],
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[SpecimenPreparationStepDicomModel],
        ],
    ) -> ParsedSpecimen:
        """Parse step and return nested preparation steps and samplings.

        Parameters
        ----------
        index: int
            Index of step in list of steps for this identifier.
        step: SpecimenPreparationStepDicomModel
            Step to parse.
        identifier: Union[str, SpecimenIdentifier]
            Identifier of specimen to parse.
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[SpecimenPreparationStepDicomModel],
        ]
            DICOM preparation steps ordered by specimen identifier.

        Returns
        -------
        ParsedSpecimen
            Parsed step for specimen.
        """
        preparation_steps: List[PreparationStep] = []
        sampling: Optional[BaseSampling] = None

        if isinstance(step, CollectionDicomModel):
            preparation_steps.append(
                self._create_collection_step(
                    index, step, identifier, steps_by_identifier
                )
            )
        elif isinstance(step, SamplingDicomModel):
            sampling = self._create_sampling_step(index, step, steps_by_identifier)
        elif index == 0:
            # First step was not a collection step or a sampling step, this specimen
            # thus has no defined parent.
            sampling = self._create_unknown_sampling(identifier, steps_by_identifier)
        elif isinstance(step, ProcessingDicomModel):
            if step.processing is not None:
                # Only add processing step if has processing code.
                preparation_steps.append(
                    Processing(step.processing, step.date_time, step.description)
                )
        elif isinstance(step, ReceivingDicomModel):
            preparation_steps.append(Receiving(step.date_time, step.description))
        elif isinstance(step, StorageDicomModel):
            preparation_steps.append(Storage(step.date_time, step.description))
        elif isinstance(step, StainingDicomModel):
            # Stainings are handled elsewhere
            pass
        else:
            raise NotImplementedError(f"Step of type {type(step)}")
        # Fixative, embedding, and processing (for other than processing step)
        # are parsed into separate steps
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
        if step.processing is not None and not isinstance(step, ProcessingDicomModel):
            preparation_steps.append(
                Processing(
                    step.processing,
                    date_time=step.date_time,
                    description=step.description,
                )
            )

        return ParsedSpecimen(
            preparation_steps=preparation_steps,
            sampling=sampling,
            container=step.container,
            specimen_type=step.specimen_type,
        )

    def _create_collection_step(
        self,
        index: int,
        step: CollectionDicomModel,
        identifier: Union[str, SpecimenIdentifier],
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[SpecimenPreparationStepDicomModel],
        ],
    ) -> Collection:
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
        return Collection(
            step.method,
            date_time=step.date_time,
            description=step.description,
        )

    def _create_sampling_step(
        self,
        index: int,
        step: SamplingDicomModel,
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[SpecimenPreparationStepDicomModel],
        ],
    ) -> BaseSampling:
        if index != 0:
            raise ValueError("Sampling step should be first step.")
        parent_identifier = step.parent_identifier
        sampling_constraints: Optional[Sequence[BaseSampling]] = None
        if parent_identifier in self._created_specimens:
            pass
            # Parent already exists.
            parent = self._created_specimens[parent_identifier]
            if isinstance(parent, Sample):
                sampling_constraints = parent.sampled_from
            parsed_parent = self._parse_preparation_steps_for_specimen(
                parent_identifier, steps_by_identifier
            )
            existing_parent_steps_without_samplings = [
                parent_step
                for parent_step in parent.steps
                if not isinstance(parent_step, BaseSampling)
            ]
            if (
                not parsed_parent.preparation_steps
                == existing_parent_steps_without_samplings
            ):
                raise ValueError(
                    f"Specimen {parent_identifier} already exists with different steps."
                )
            if (
                parsed_parent.sampling is not None
                and isinstance(parent, Sample)
                and parsed_parent.sampling not in parent.sampled_from_list
            ):
                parent.sampled_from_list.append(parsed_parent.sampling)
                sampling_constraints = [parsed_parent.sampling]
        else:
            # Need to create parent
            parent = self._create_specimen(
                parent_identifier,
                step.parent_specimen_type,
                steps_by_identifier,
            )
            if isinstance(parent, Sample):
                sampling_constraints = parent.sampled_from
            self._created_specimens[parent_identifier] = parent
        if isinstance(parent, Sample):
            # If Sample create sampling with constraint
            return parent.sample(
                method=step.method,
                date_time=step.date_time,
                description=step.description,
                sampling_chain_constraints=sampling_constraints,
                location=step.sampling_location,
            )
        if isinstance(parent, Specimen):
            # Extracted specimen can not have constraint
            return parent.sample(
                method=step.method,
                date_time=step.date_time,
                description=step.description,
                location=step.sampling_location,
            )
        raise ValueError(f"Unknown parent type {type(parent)}.")

    def _create_unknown_sampling(
        self,
        identifier: Union[str, SpecimenIdentifier],
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[SpecimenPreparationStepDicomModel],
        ],
    ) -> Optional[BaseSampling]:
        # If not first described specimen create a UnknownSampling from previous
        # listed specimen.
        this_specimen_index = list(steps_by_identifier.keys()).index(identifier)
        if this_specimen_index == 0:
            return None
        previous_specimen_identifier = list(steps_by_identifier.keys())[
            this_specimen_index - 1
        ]
        previous_specimen = self._create_specimen(
            previous_specimen_identifier,
            None,
            steps_by_identifier,
        )
        sampling = previous_specimen.sample()
        return sampling

    def _create_specimen(
        self,
        identifier: Union[str, SpecimenIdentifier],
        specimen_type: Optional[AnatomicPathologySpecimenTypesCode],
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[SpecimenPreparationStepDicomModel],
        ],
    ) -> Union[Specimen, Sample]:
        """
        Create an Specimen or Sample.

        Parameters
        ----------
        identifier: Union[str, SpecimenIdentifier]
            The identifier of the specimen to create.
        specimen_type: AnatomicPathologySpecimenTypesCode
            The coded type of the specimen to create.
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[SpecimenPreparationStepDicomModel],
        ]
            DICOM preparation steps ordered by specimen identifier.


        Returns
        ----------
        Union[Specimen, Sample]
            Created Specimen, if the specimen has no parents, or Specimen.

        """
        logging.debug(f"Creating specimen with identifier {identifier}")
        parsed_specimen = self._parse_preparation_steps_for_specimen(
            identifier, steps_by_identifier
        )
        if specimen_type is None:
            specimen_type = parsed_specimen.specimen_type
        elif (
            parsed_specimen.specimen_type is not None
            and specimen_type != parsed_specimen.specimen_type
        ):
            raise ValueError(
                f"Specimen type {specimen_type} and step specimen type "
                f"{parsed_specimen.specimen_type} do not match."
            )
        if parsed_specimen.sampling is None:
            if len(parsed_specimen.preparation_steps) > 0 and isinstance(
                parsed_specimen.preparation_steps[0], Collection
            ):
                collection_step = parsed_specimen.preparation_steps.pop(0)
                assert isinstance(collection_step, Collection)
            else:
                collection_step = None
            return Specimen(
                identifier=identifier,
                extraction_step=collection_step,
                type=specimen_type,
                steps=parsed_specimen.preparation_steps,
                container=parsed_specimen.container,
            )
        return Sample(
            identifier=identifier,
            type=specimen_type,
            sampled_from=[parsed_specimen.sampling]
            if parsed_specimen.sampling is not None
            else [],
            steps=parsed_specimen.preparation_steps,
            container=parsed_specimen.container,
        )
