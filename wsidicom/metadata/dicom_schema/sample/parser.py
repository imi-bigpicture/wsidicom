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

import logging
from collections import defaultdict
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)


from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
)
from wsidicom.metadata.dicom_schema.sample.model import (
    CollectionDicomModel,
    ProcessingDicomModel,
    SamplingDicomModel,
    SpecimenDescriptionDicomModel,
    SpecimenPreparationStepDicomModel,
    StainingDicomModel,
)

from wsidicom.metadata.sample import (
    Collection,
    Embedding,
    ExtractedSpecimen,
    Fixation,
    PreparationStep,
    Processing,
    Sample,
    Sampling,
    SlideSample,
    SpecimenIdentifier,
    Staining,
)


class SpecimenDicomParser:
    def __init__(self):
        self._created_specimens = {}
        self._created_stainings: List[Staining] = []

    def parse_descriptions(
        self, specimen_descriptions: Iterable[SpecimenDescriptionDicomModel]
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

        slide_samples: List[SlideSample] = []
        stainings: List[Staining] = []
        for specimen_description in specimen_descriptions:
            slide_sample = self._create_slide_sample(specimen_description)
            slide_samples.append(slide_sample)

        return slide_samples, stainings

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
                if not any(
                    staining == existing for existing in self._created_stainings
                ):
                    self._created_stainings.append(staining)
            elif isinstance(step, SamplingDicomModel):
                parent_identifier = step.parent_specimen_identifier
                steps_by_identifier[parent_identifier].append(step)
            identifier = step.get_identifier()
            steps_by_identifier[identifier].append(step)

        identifier = description.get_identifier()
        preparation_steps, samplings = self._parse_preparation_steps_for_specimen(
            identifier, steps_by_identifier
        )

        if len(samplings) > 1:
            raise ValueError("Should be max one sampling, got.", len(samplings))
        return SlideSample(
            identifier=identifier,
            anatomical_sites=[],
            sampled_from=next(iter(samplings), None),
            uid=description.uid,
            localization=description.localization,
            steps=preparation_steps,
        )

    def _parse_preparation_steps_for_specimen(
        self,
        identifier: Union[str, SpecimenIdentifier],
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[Optional[SpecimenPreparationStepDicomModel]],
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
                if parent_identifier in self._created_specimens:
                    # Parent already exists. Parse any non-parsed steps
                    parent = self._created_specimens[parent_identifier]
                    (
                        parent_steps,
                        sampling_constraints,
                    ) = self._parse_preparation_steps_for_specimen(
                        parent_identifier, steps_by_identifier, step
                    )
                    for parent_step in parent_steps:
                        # Only add step if an equivalent does not exists
                        if not any(step == parent_step for step in parent.steps):
                            parent.add(parent_step)
                    if isinstance(parent, Sample):
                        parent._sampled_from.extend(sampling_constraints)
                else:
                    # Need to create parent
                    parent = self._create_specimen(
                        parent_identifier,
                        step.parent_specimen_type,
                        steps_by_identifier,
                        step,
                    )
                    if isinstance(parent, Sample):
                        sampling_constraints = parent.sampled_from
                    else:
                        sampling_constraints = None
                    self._created_specimens[parent_identifier] = parent
                if isinstance(parent, Sample):
                    # If Sample create sampling with constraint
                    sampling = parent.sample(
                        method=step.method,
                        date_time=step.date_time,
                        description=step.description,
                        sampling_chain_constraints=sampling_constraints,
                        location=step.sampling_location,
                    )
                elif isinstance(parent, ExtractedSpecimen):
                    # Extracted specimen can not have constraint
                    sampling = parent.sample(
                        method=step.method,
                        date_time=step.date_time,
                        description=step.description,
                        location=step.sampling_location,
                    )
                else:
                    raise ValueError()

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

    def _create_specimen(
        self,
        identifier: Union[str, SpecimenIdentifier],
        specimen_type: AnatomicPathologySpecimenTypesCode,
        steps_by_identifier: Dict[
            Union[str, SpecimenIdentifier],
            List[Optional[SpecimenPreparationStepDicomModel]],
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
        preparation_steps, samplings = self._parse_preparation_steps_for_specimen(
            identifier, steps_by_identifier, stop_at_step
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
