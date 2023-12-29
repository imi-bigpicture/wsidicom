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

from typing import (
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Union,
)
from wsidicom.conceptcode import ContainerTypeCode

from wsidicom.metadata.dicom_schema.sample.model import (
    SamplingDicomModel,
    SpecimenDescriptionDicomModel,
    SpecimenPreparationStepDicomModel,
    StainingDicomModel,
)
from wsidicom.metadata.sample import (
    PreparationStep,
    SampledSpecimen,
    Sampling,
    SlideSample,
    Specimen,
    SpecimenIdentifier,
    Staining,
)


class SpecimenDicomFormatter:
    @classmethod
    def to_dicom(
        cls,
        slide_sample: SlideSample,
        stains: Optional[Sequence[Staining]] = None,
    ) -> SpecimenDescriptionDicomModel:
        """Create a DICOM specimen description for slide sample."""
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
        return SpecimenDescriptionDicomModel(
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
            localization=slide_sample.localization,
        )

    @classmethod
    def _get_steps(
        cls, slide_sample: SlideSample, stainings: Optional[Iterable[Staining]]
    ) -> List[SpecimenPreparationStepDicomModel]:
        sample_preparation_steps = [
            step
            for sampling in slide_sample.sampled_from_list
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
        cls,
        sampling: Sampling,
        sample_identifier: Union[str, SpecimenIdentifier],
        container: Optional[ContainerTypeCode] = None,
    ) -> List[SpecimenPreparationStepDicomModel]:
        """Return DICOM steps for the specimen the sample was sampled from."""
        if isinstance(sampling.specimen, SampledSpecimen):
            steps = [
                step
                for sampling in sampling.specimen._sampled_from
                if sampling.sampling_chain_constraints is None
                or sampling in sampling.sampling_chain_constraints
                for step in cls._get_steps_for_sampling(
                    sampling, sampling.specimen.identifier, sampling.specimen.container
                )
            ]
        else:
            steps = []
        steps.extend(
            SpecimenPreparationStepDicomModel.from_step(step, sampling.specimen)
            for step in cls._get_steps_before_sampling(sampling.specimen, sampling)
        )
        steps.append(
            SamplingDicomModel.from_step(sampling, sample_identifier, container)
        )
        return steps

    @staticmethod
    def _get_steps_before_sampling(
        specimen: Specimen, sampling: Sampling
    ) -> Iterator[PreparationStep]:
        """Return the steps in specimen that occurred before the given sampling."""
        for step in specimen.steps:
            if isinstance(step, Sampling):
                # Break if sampling step for this sample, otherwise skip
                if step == sampling:
                    break
                continue
            yield step
