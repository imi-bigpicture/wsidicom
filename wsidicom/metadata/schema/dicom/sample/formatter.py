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

"""Module with DICOM specimen description formatter."""

from typing import (
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
)

from wsidicom.metadata.sample import (
    BaseSampling,
    BaseSpecimen,
    PreparationStep,
    SampledSpecimen,
    Sampling,
    SlideSample,
    SpecimenIdentifier,
    Staining,
)
from wsidicom.metadata.schema.dicom.sample.model import (
    SamplingDicomModel,
    SpecimenDescriptionDicomModel,
    SpecimenPreparationStepDicomModel,
    StainingDicomModel,
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
            for step in cls._get_steps_for_sampling(sampling, slide_sample)
        ]
        if stainings is None:
            return sample_preparation_steps
        for staining in stainings:
            step = StainingDicomModel.from_step(staining, slide_sample)
            sample_preparation_steps.append(step)
        return sample_preparation_steps

    @classmethod
    def _get_steps_for_sampling(
        cls,
        sampling: BaseSampling,
        sample: BaseSpecimen,
    ) -> List[SpecimenPreparationStepDicomModel]:
        """Return DICOM steps for the specimen the sample was sampled from."""
        if isinstance(sampling.specimen, SampledSpecimen):
            steps = [
                step
                for sub_sampling in sampling.specimen.sampled_from_list
                if sampling.sampling_chain_constraints is None
                or sub_sampling in sampling.sampling_chain_constraints
                for step in cls._get_steps_for_sampling(sub_sampling, sampling.specimen)
            ]
        else:
            steps = []
        steps.extend(
            dicom_step
            for dicom_step in [
                SpecimenPreparationStepDicomModel.from_step(step, sampling.specimen)
                for step in cls._get_steps_before_sampling(sampling.specimen, sampling)
            ]
            if dicom_step is not None
        )
        if isinstance(sampling, Sampling):
            steps.append(SamplingDicomModel.from_step(sampling, sample))
        return steps

    @staticmethod
    def _get_steps_before_sampling(
        specimen: BaseSpecimen, sampling: BaseSampling
    ) -> Iterator[PreparationStep]:
        """Return the steps in specimen that occurred before the given sampling."""
        for step in specimen.steps:
            if isinstance(step, BaseSampling):
                # Break if sampling step for this sample, otherwise skip
                if step == sampling:
                    break
                continue
            yield step
