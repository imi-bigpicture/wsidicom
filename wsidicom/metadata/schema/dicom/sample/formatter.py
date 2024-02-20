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
    Optional,
    Sequence,
)

from wsidicom.metadata.sample import (
    BaseSampling,
    BaseSpecimen,
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
    """Formatter for producing DICOM Specimen Description models from slide samples.

    Converting a slide sample to a DICOM Specimen Description requires producing a
    list of all the preparation steps that were performed on the sample. As a parent
    to a sample may have been sampled from multiple specimens, we use the sampling
    constraints to limit the sampling tree to a single branch."""

    @classmethod
    def to_dicom(
        cls,
        slide_sample: SlideSample,
        stains: Optional[Sequence[Staining]] = None,
    ) -> SpecimenDescriptionDicomModel:
        """Create a DICOM specimen description for slide sample.

        Parameters
        ----------
        slide_sample : SlideSample
            The slide sample to convert to DICOM.
        stains : Optional[Sequence[Staining]], optional
            Stainings performed on the slide sample.

        Returns
        -------
        SpecimenDescriptionDicomModel
            DICOM model describing the slide sample.
        """
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
            steps=list(sample_preparation_steps),
            anatomical_sites=(
                list(slide_sample.anatomical_sites)
                if slide_sample.anatomical_sites is not None
                else []
            ),
            short_description=slide_sample.short_description,
            detailed_description=slide_sample.detailed_description,
            localization=slide_sample.localization,
        )

    @classmethod
    def _get_steps(
        cls, slide_sample: SlideSample, stainings: Optional[Iterable[Staining]]
    ) -> Iterator[SpecimenPreparationStepDicomModel]:
        """Get all the steps performed on the slide sample, following the sampling
        branch specified in the sampling constraints.

        Parameters
        ----------
        slide_sample : SlideSample
            The slide sample to get steps for
        stainings: Optional[Iterable[Staining]]
            Stainings performed on the slide sample.

        Returns
        -------
        Iterator[SpecimenPreparationStepDicomModel]
            Iterator of DICOM steps performed on the slide sample.
        """
        yield from (
            step
            for sampling in slide_sample.sampled_from_list
            for step in cls._get_steps_for_sampling(sampling, slide_sample)
        )
        if stainings is not None:
            for staining in stainings:
                yield StainingDicomModel.from_step(staining, slide_sample)

    @classmethod
    def _get_steps_for_sampling(
        cls,
        sampling: BaseSampling,
        sample: BaseSpecimen,
    ) -> Iterator[SpecimenPreparationStepDicomModel]:
        """Return DICOM steps for the specimen the sample was sampled from."""
        if isinstance(sampling.specimen, SampledSpecimen):
            yield from (
                step
                for sub_sampling in sampling.specimen.sampled_from_list
                if sampling.sampling_constraints is None
                or sub_sampling in sampling.sampling_constraints
                for step in cls._get_steps_for_sampling(sub_sampling, sampling.specimen)
            )

        yield from (
            dicom_step
            for dicom_step in [
                SpecimenPreparationStepDicomModel.from_step(step, sampling.specimen)
                for step in sampling.specimen.steps
                if not isinstance(step, BaseSampling)
            ]
            if dicom_step is not None
        )
        if isinstance(sampling, Sampling):
            yield SamplingDicomModel.from_step(sampling, sample)
