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

from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    ContainerTypeCode,
)
from wsidicom.config import settings
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


@dataclass(frozen=True)
class ParsedSpecimen:
    """Parsed specimen steps."""

    preparation_steps: List[PreparationStep]
    sampling: Optional[BaseSampling]
    container: Optional[ContainerTypeCode]
    specimen_type: Optional[AnatomicPathologySpecimenTypesCode]


class StepsDirectory:
    """Class for holding specimen preparation steps ordered by specimen identifier."""

    def __init__(self, steps: List[SpecimenPreparationStepDicomModel]):
        """Initialize StepsDirectory.

        Parameters
        ----------
        steps: List[SpecimenPreparationStepDicomModel]
            Steps to order.
        """
        self._steps_by_identifier = self._order_steps_by_identifier(steps)

    def get(
        self, identifier: SpecimenIdentifier
    ) -> List[SpecimenPreparationStepDicomModel]:
        """Get steps matching identifier.

        Parameters
        ----------
        identifier: SpecimenIdentifier
            Specimen identifier to get steps for.

        Returns
        -------
        List[SpecimenPreparationStepDicomModel]
            Steps for specimen.
        """
        return self._get_and_clear_steps_for_identifier(identifier)

    def get_previous_specimen_identifier(
        self, identifier: SpecimenIdentifier
    ) -> Optional[SpecimenIdentifier]:
        """Get identifier of previous specimen.

        Parameters
        ----------
        identifier: SpecimenIdentifier
            Identifier of specimen to get previous specimen for.

        Returns
        -------
        Optional[SpecimenIdentifier]
            Identifier of previous specimen, if any.
        """
        matching_identifier = self._get_identifier(
            identifier, self._steps_by_identifier
        )
        if matching_identifier is None:
            return None
        this_specimen_index = list(self._steps_by_identifier.keys()).index(
            matching_identifier
        )
        if this_specimen_index == 0:
            return None
        return list(self._steps_by_identifier.keys())[this_specimen_index - 1]

    @property
    def is_empty(self) -> bool:
        return all(steps is None for steps in self._steps_by_identifier.values())

    def _order_steps_by_identifier(
        self,
        steps: List[SpecimenPreparationStepDicomModel],
    ) -> Dict[SpecimenIdentifier, Optional[List[SpecimenPreparationStepDicomModel]]]:
        """
        Parse steps into dictionary of steps by specimen identifier.

        Parameters
        ----------
        steps: List[SpecimenPreparationStepDicomModel]
            Steps to parse.

        Returns
        -------
        Dict[SpecimenIdentifier, Optional[List[SpecimenPreparationStepDicomModel]]]:
            Steps ordered by specimen identifier.
        """
        steps_by_identifier: Dict[
            SpecimenIdentifier, Optional[List[SpecimenPreparationStepDicomModel]]
        ] = {}
        for step in steps:
            identifier = step.specimen_identifier
            matching_identifier = self._get_identifier(identifier, steps_by_identifier)
            if matching_identifier not in steps_by_identifier:
                steps_by_identifier[identifier] = [step]
            else:
                existing_steps = steps_by_identifier[matching_identifier]
                if existing_steps is None:
                    steps_by_identifier[matching_identifier] = [step]
                else:
                    existing_steps.append(step)
        return steps_by_identifier

    def _get_and_clear_steps_for_identifier(
        self,
        identifier: SpecimenIdentifier,
    ) -> List[SpecimenPreparationStepDicomModel]:
        """Return and clear steps matching identifier.

        If strict matching return steps by key. Otherwise return steps with relaxed
        matching.

        Steps are cleared after being returned to ensure that they are not parsed
        multiple times. The key remains in the dictionary to enable checking that all
        identifiers have been parsed.

        Parameters
        ----------
        identifier: SpecimenIdentifier
            Identifier to match.

        Returns
        -------
        List[SpecimenPreparationStepDicomModel]
            Steps matching identifier.
        """
        matching_identifier = self._get_identifier(
            identifier, self._steps_by_identifier
        )
        if matching_identifier not in self._steps_by_identifier:
            # Specimen does not have any describing preparation steps
            return []
        steps = self._steps_by_identifier[matching_identifier]
        if steps is None:
            raise ValueError(
                f"Tried to get steps for identifier {matching_identifier} that "
                "has already been parsed."
            )
        # Clear steps for identifier, as the specimen should not be parsed again.
        self._steps_by_identifier[matching_identifier] = None
        return steps

    @staticmethod
    def _get_identifier(
        identifier: SpecimenIdentifier,
        steps_by_identifier: Dict[
            SpecimenIdentifier, Optional[List[SpecimenPreparationStepDicomModel]]
        ],
    ) -> Optional[SpecimenIdentifier]:
        """Get identifier matching identifier.

        Parameters
        ----------
        identifier: SpecimenIdentifier
            Identifier to match.
        steps_by_identifier: Dict[
            SpecimenIdentifier,
            Optional[List[SpecimenPreparationStepDicomModel]]
        ]
            Steps ordered by identifier.

        Returns
        -------
        Optional[SpecimenIdentifier]
            Identifier matching identifier.
        """

        if (
            settings.strict_specimen_identifier_check
            or identifier in steps_by_identifier
        ):
            return identifier
        return next(
            (
                existing_identifier
                for existing_identifier in steps_by_identifier.keys()
                if existing_identifier.relaxed_matching(identifier)
            ),
            None,
        )


class SpecimenDicomParser:
    """Parser for DICOM Specimen Description models deserialized from DICOM dataset.

    The DICOM Specimen Description contains a list of Specimen Preparation Steps
    describing the preparation of a specimen. The steps are ordered chronologically and
    describes specimen collection, sampling, staining, etc. for (potentially) the whole
    specimen branch from the extracted specimen to the slide sample.

    We assume (and check) that one Specimen Description only describes one branch of the
    specimen tree, i.e. if for example a slide sample is sampled from a block that has
    been prepared from two different extracted specimens, the Specimen Description will
    only contain steps for one of the extracted specimens. There might be another slide
    sample on the slide with a Specimen Description containing steps for the other
    extracted specimen.

    We also assume (and check) that when the sample specimen occurs in multiple
    Specimen Descriptions it has the same steps, i.e. the specimen has not been further
    processed after performing one of the samplings.

    We assume (and check) that all slide samples on a slide have been stained with the
    same stainings.

    When parsing multiple Specimen Descriptions, we merge any specimens that
    have the same identifier, i.e. two slide samples from the same block will have the
    samme block object as parent specimen.

    Any Specimen Preparation Step can specify a processing method, fixative, and/or
    embedding, e.g. a sampling step can optionally also define a fixative. We create
    separate steps for each of these (processing method, fixative, embedding).

    The parsing of the Specimen Description starts with parsing the steps for the
    slide sample, and then recursively parsing the steps for the parent specimen. Parent
    specimens are either identified by a sampling step or by assuming that the specimen
    was sampled from the previous specimen in the Specimen Description (if any). The
    later sampling will produce an `UnknownSampling`, as the sampling step is missing
    properties as `method` and it can't be certain that the found parent specimen is the
    immediate parent specimen or a specimen further up the specimen branch.
    """

    def __init__(self):
        self._parsed_specimens: Dict[SpecimenIdentifier, BaseSpecimen] = {}
        self._parsed_stainings: List[Staining] = []

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
        slide_samples = [
            self._parsed_description(specimen_description)
            for specimen_description in specimen_descriptions
        ]
        return slide_samples, self._parsed_stainings

    def _parsed_description(
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
        stainings = self._parse_stainings(description.steps)
        if len(self._parsed_stainings) == 0:
            self._parsed_stainings = list(stainings)
        elif any(staining not in self._parsed_stainings for staining in stainings):
            raise NotImplementedError(
                "Handling slide samples with different stainings is not implemented."
            )
        steps_directory = StepsDirectory(description.steps)
        identifier = description.specimen_identifier
        parsed_specimen = self._parse_preparation_steps_for_specimen(
            identifier, steps_directory
        )
        if not steps_directory.is_empty:
            logging.warning(
                "Specimen description contained steps for specimens that were not "
                "parsed. If this is due to steps missing issuer of identifiers, try "
                "setting `settings.strict_specimen_identifier_check` to `False`."
            )
        return SlideSample(
            identifier=identifier.simplify(),
            anatomical_sites=description.anatomical_sites,
            sampled_from=parsed_specimen.sampling,
            uid=description.uid,
            localization=description.localization,
            steps=parsed_specimen.preparation_steps,
        )

    def _parse_stainings(
        self,
        steps: List[SpecimenPreparationStepDicomModel],
    ) -> Iterator[Staining]:
        """
        Parse stainings from steps.

        Parameters
        ----------
        steps: List[SpecimenPreparationStepDicomModel]
            Steps to parse.

        Returns
        ----------
        Iterator[Staining]
            Parsed stainings.

        """
        return (
            Staining(
                step.substances, date_time=step.date_time, description=step.description
            )
            for step in steps
            if isinstance(step, StainingDicomModel)
        )

    def _parse_preparation_steps_for_specimen(
        self, identifier: SpecimenIdentifier, steps_directory: StepsDirectory
    ) -> ParsedSpecimen:
        """
        Parse preparation steps and samplings for a specimen.

        Creates or updates parent specimens.

        Parameters
        ----------
        identifier: SpecimenIdentifier
            The identifier of the specimen to parse.
        steps_directory: StepsDirectory
            DICOM preparation steps ordered by specimen identifier.

        Returns
        ----------
        ParsedSpecimen
            Parsed steps for specimen.

        """
        container: Optional[ContainerTypeCode] = None
        specimen_type: Optional[AnatomicPathologySpecimenTypesCode] = None
        sampling: Optional[BaseSampling] = None
        preparation_steps: List[PreparationStep] = []
        steps = steps_directory.get(identifier)
        for index, step in enumerate(steps):
            parsed_step = self._parse_step(
                index, step, identifier, steps_directory, steps
            )
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
        identifier: SpecimenIdentifier,
        steps_directory: StepsDirectory,
        steps: List[SpecimenPreparationStepDicomModel],
    ) -> ParsedSpecimen:
        """Parse step and return nested preparation steps and samplings.

        Parameters
        ----------
        index: int
            Index of step in list of steps for this identifier.
        step: SpecimenPreparationStepDicomModel
            Step to parse.
        identifier: SpecimenIdentifier
            Identifier of specimen to parse.
        steps_directory: StepsDirectory
            DICOM preparation steps ordered by specimen identifier.
        steps: List[SpecimenPreparationStepDicomModel]
            All steps for this identifier.

        Returns
        -------
        ParsedSpecimen
            Parsed step for specimen.
        """
        preparation_steps: List[PreparationStep] = []
        sampling: Optional[BaseSampling] = None
        # First handle collection and sampling steps that should be on index 0
        if index == 0:
            if isinstance(step, CollectionDicomModel):
                preparation_steps.append(
                    self._create_collection_step(step, identifier, steps)
                )
            elif isinstance(step, SamplingDicomModel):
                sampling = self._create_sampling_step(step, steps_directory)
            else:
                # First step was not a collection step or a sampling step, this specimen
                # thus has no defined parent. Add a unknown sampling to previous
                # specimen if any
                sampling = self._create_unknown_sampling(identifier, steps_directory)

        if isinstance(step, ProcessingDicomModel):
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
        elif isinstance(step, (CollectionDicomModel, SamplingDicomModel)):
            if index != 0:
                raise ValueError(
                    f"Found unexpected step of type {type(step)} at index {index}. "
                    "Collection and sampling steps should only be on index 0"
                )
        else:
            raise NotImplementedError(f"Step of type {type(step)} at index {index}.")
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
        step: CollectionDicomModel,
        identifier: SpecimenIdentifier,
        steps: List[SpecimenPreparationStepDicomModel],
    ) -> Collection:
        """Create collection step from collection model.

        Parameters
        ----------
        step: CollectionDicomModel
            Collection model to create step from.
        identifier: SpecimenIdentifier
            Identifier of the specimen the collection belongs to.
        steps_directory: StepsDirectory
            DICOM preparation steps ordered by specimen identifier.
        steps: List[SpecimenPreparationStepDicomModel]
            All steps for this identifier.

        Returns
        -------
        Collection
            Collection step created from model.
        """
        any_sampling_steps = any(
            sampling_step
            for sampling_step in steps
            if sampling_step is not None
            and isinstance(sampling_step, SamplingDicomModel)
            and sampling_step.identifier == identifier
        )
        if any_sampling_steps:
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
        step: SamplingDicomModel,
        steps_directory: StepsDirectory,
    ) -> BaseSampling:
        """Create sampling step from sampling model.

        Will create any needed parent specimens.

        Parameters
        ----------
        step: SamplingDicomModel
            Sampling model to create step from.
        steps_directory: StepsDirectory
            DICOM preparation steps ordered by specimen identifier.

        Returns
        -------
        BaseSampling
            Sampling step created from model.
        """
        parent, sampling_constraints = self._get_parent_for_sampling(
            step, steps_directory
        )
        if isinstance(parent, Sample):
            # If Sample create sampling with constraint
            return parent.sample(
                method=step.method,
                date_time=step.date_time,
                description=step.description,
                sampling_constraints=sampling_constraints,
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

    def _get_parent_for_sampling(
        self,
        sampling: SamplingDicomModel,
        steps_directory: StepsDirectory,
    ) -> Tuple[BaseSpecimen, Optional[Sequence[BaseSampling]]]:
        """Get parent specimen and any sampling constraints for sampling.

        Creates any needed parent specimens.

        Parameters
        ----------
        sampling: SamplingDicomModel
            Sampling to get parent for.
        steps_directory: StepsDirectory
            DICOM preparation steps ordered by specimen identifier.

        Returns
        -------
        Tuple[BaseSpecimen, Optional[Sequence[BaseSampling]]]
            Parent specimen and any sampling constraints.
        """
        parent_identifier = sampling.parent_identifier
        sampling_constraints: Optional[Sequence[BaseSampling]] = None
        if parent_identifier not in self._parsed_specimens:
            # Parent does not exist, create it.
            parent = self._create_specimen(
                parent_identifier,
                sampling.parent_specimen_type,
                steps_directory,
            )
            if isinstance(parent, Sample):
                sampling_constraints = parent.sampled_from
            self._parsed_specimens[parent_identifier] = parent
            return parent, sampling_constraints

        # Parent already exists. Check that parents preparation steps match and
        # add the new sampling tree to the parent.
        parent = self._parsed_specimens[parent_identifier]
        parsed_parent = self._parse_preparation_steps_for_specimen(
            parent_identifier, steps_directory
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
            isinstance(parent, Sample)
            and parsed_parent.sampling is not None
            and parsed_parent.sampling not in parent.sampled_from_list
        ):
            # Parsed parent adds a new sampling branch
            parent.sampled_from_list.append(parsed_parent.sampling)
            sampling_constraints = [parsed_parent.sampling]
        return parent, sampling_constraints

    def _create_unknown_sampling(
        self,
        identifier: SpecimenIdentifier,
        steps_directory: StepsDirectory,
    ) -> Optional[BaseSampling]:
        """Create unknown sampling from previous specimen.

        Creates any needed parent specimens.

        Parameters
        ----------
        identifier: SpecimenIdentifier
            Identifier of the specimen the sampling belongs to.
        steps_directory: StepsDirectory
            DICOM preparation steps ordered by specimen identifier.

        Returns
        -------
        Optional[BaseSampling]
            Unknown sampling created from previous specimen.
        """

        previous_specimen = self._get_previous_specimen(identifier, steps_directory)
        if previous_specimen is None:
            return None
        sampling = previous_specimen.sample()
        return sampling

    def _get_previous_specimen(
        self,
        identifier: SpecimenIdentifier,
        steps_directory: StepsDirectory,
    ) -> Optional[Union[Specimen, Sample]]:
        """Get previous described specimen.

        Will create any needed parent specimens.

        Parameters
        ----------
        identifier: SpecimenIdentifier
            Identifier of the specimen to get previous specimen for.
        steps_directory: StepsDirectory
            DICOM preparation steps ordered by specimen identifier.

        Returns
        -------
        Optional[Union[Specimen, Sample]]
            Previous specimen, if any.
        """
        previous_specimen_identifier = steps_directory.get_previous_specimen_identifier(
            identifier
        )
        if previous_specimen_identifier is None:
            return None
        return self._create_specimen(
            previous_specimen_identifier,
            None,
            steps_directory,
        )

    def _create_specimen(
        self,
        identifier: SpecimenIdentifier,
        specimen_type: Optional[AnatomicPathologySpecimenTypesCode],
        steps_directory: StepsDirectory,
    ) -> Union[Specimen, Sample]:
        """
        Create an Specimen or Sample.

        Parameters
        ----------
        identifier: SpecimenIdentifier
            The identifier of the specimen to create.
        specimen_type: AnatomicPathologySpecimenTypesCode
            The coded type of the specimen to create.
        steps_directory: StepsDirectory
            DICOM preparation steps ordered by specimen identifier.


        Returns
        ----------
        Union[Specimen, Sample]
            Created Specimen, if the specimen has no parents, or Specimen.

        """
        logging.debug(f"Creating specimen with identifier {identifier}")
        parsed_specimen = self._parse_preparation_steps_for_specimen(
            identifier, steps_directory
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
                identifier=identifier.simplify(),
                extraction_step=collection_step,
                type=specimen_type,
                steps=parsed_specimen.preparation_steps,
                container=parsed_specimen.container,
            )
        return Sample(
            identifier=identifier.simplify(),
            type=specimen_type,
            sampled_from=(
                [parsed_specimen.sampling]
                if parsed_specimen.sampling is not None
                else []
            ),
            steps=parsed_specimen.preparation_steps,
            container=parsed_specimen.container,
        )
