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

from typing import List, Optional, Sequence, Union

import pytest
from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import UID

from tests.metadata.dicom_schema.helpers import (
    assert_next_item_equals_code,
    assert_next_item_equals_measurement,
    assert_next_item_equals_string,
    create_description_dataset,
    create_processing_dataset,
    create_sampling_dataset,
    create_specimen_preparation_sequence,
)
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
from wsidicom.config import settings
from wsidicom.metadata.sample import (
    Collection,
    Embedding,
    Fixation,
    LocalIssuerOfIdentifier,
    Processing,
    Sample,
    SampleLocalization,
    Sampling,
    SamplingLocation,
    SlideSample,
    Specimen,
    SpecimenIdentifier,
    Staining,
)
from wsidicom.metadata.schema.dicom.sample.formatter import SpecimenDicomFormatter
from wsidicom.metadata.schema.dicom.sample.model import (
    ProcessingDicomModel,
    SamplingDicomModel,
)
from wsidicom.metadata.schema.dicom.sample.parser import SpecimenDicomParser
from wsidicom.metadata.schema.dicom.sample.schema import (
    SampleCodes,
    SpecimenDescriptionDicomSchema,
)


class TestSampleDicom:
    @pytest.mark.parametrize(
        ["slide_sample_ids", "slide_sample_uids", "specimen_ids"],
        [
            [
                ["slide sample"],
                [
                    UID(
                        "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445424"
                    )
                ],
                ["specimen"],
            ],
            [
                ["slide sample 1", "slide sample 2"],
                [
                    UID(
                        "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445424"
                    ),
                    UID(
                        "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445425"
                    ),
                ],
                ["specimen 1", "specimen 2"],
            ],
        ],
    )
    @pytest.mark.parametrize(
        "fixative", [SpecimenFixativesCode("Neutral Buffered Formalin")]
    )
    @pytest.mark.parametrize("medium", [SpecimenEmbeddingMediaCode("Paraffin wax")])
    @pytest.mark.parametrize("short_description", [None, "short description"])
    @pytest.mark.parametrize("detailed_description", [None, "detailed description"])
    @pytest.mark.parametrize(
        "specimen_container", [None, ContainerTypeCode("Specimen container")]
    )
    @pytest.mark.parametrize(
        "block_container", [None, ContainerTypeCode("Tissue cassette")]
    )
    @pytest.mark.parametrize(
        "substances",
        [
            "HE",
            [
                SpecimenStainsCode("hematoxylin stain"),
                SpecimenStainsCode("water soluble eosin stain"),
            ],
        ],
    )
    def test_slide_sample_from_dataset(
        self,
        slide_sample_ids: Sequence[str],
        slide_sample_uids: Sequence[UID],
        specimen_ids: Sequence[str],
        block_id: str,
        specimen_type: AnatomicPathologySpecimenTypesCode,
        collection_method: SpecimenCollectionProcedureCode,
        fixative: SpecimenFixativesCode,
        specimen_sampling_method: SpecimenSamplingProcedureCode,
        medium: SpecimenEmbeddingMediaCode,
        block_sampling_method: SpecimenSamplingProcedureCode,
        block_type: AnatomicPathologySpecimenTypesCode,
        sample_localization: SampleLocalization,
        sampling_location: SamplingLocation,
        primary_anatomic_structures: Sequence[Code],
        substances: Union[str, Sequence[SpecimenStainsCode]],
        short_description: Optional[str],
        detailed_description: Optional[str],
        specimen_container: Optional[ContainerTypeCode],
        block_container: Optional[ContainerTypeCode],
    ):
        # Arrange
        descriptions: List[Dataset] = []
        dataset = Dataset()
        for slide_sample_id, slide_sample_uid, specimen_id in zip(
            slide_sample_ids, slide_sample_uids, specimen_ids
        ):
            preparation_steps = create_specimen_preparation_sequence(
                slide_sample_id,
                specimen_id,
                block_id,
                specimen_type,
                collection_method,
                fixative,
                specimen_sampling_method,
                medium,
                block_sampling_method,
                block_type,
                sampling_location,
                substances,
                specimen_container,
                block_container,
            )
            description = create_description_dataset(
                slide_sample_id,
                slide_sample_uid,
                primary_anatomic_structures,
                sample_localization,
                short_description=short_description,
                detailed_description=detailed_description,
                preparation_step_datasets=preparation_steps,
            )

            descriptions.append(description)
        dataset.SpecimenDescriptionSequence = descriptions
        schema = SpecimenDescriptionDicomSchema()

        # Act
        models = [
            schema.load(description)
            for description in dataset.SpecimenDescriptionSequence
        ]
        slide_samples, stainings = SpecimenDicomParser().parse_descriptions(models)

        # Assert
        assert slide_samples is not None
        assert stainings is not None
        for staining in stainings:
            assert isinstance(staining, Staining)
            assert staining.substances == substances
        assert len(slide_samples) == len(slide_sample_ids)
        for slide_sample_index, slide_sample in enumerate(slide_samples):
            assert isinstance(slide_sample, SlideSample)
            assert slide_sample.identifier == slide_sample_ids[slide_sample_index]
            assert slide_sample.uid == slide_sample_uids[slide_sample_index]
            assert slide_sample.anatomical_sites == primary_anatomic_structures
            assert slide_sample.localization == sample_localization
            assert slide_sample.sampled_from is not None
            if isinstance(slide_sample.sampled_from, Sampling):
                assert slide_sample.sampled_from.method == block_sampling_method
            block = slide_sample.sampled_from.specimen
            assert isinstance(block, Sample)
            assert block.identifier == block_id
            assert block.type == block_type
            assert block.container == block_container
            embedding_step = block.steps[0]
            assert isinstance(embedding_step, Embedding)
            assert embedding_step.medium == medium
            assert len(block.sampled_from) == len(specimen_ids)
            for index, specimen_id in enumerate(specimen_ids):
                loaded_specimen_sampling = block.sampled_from[index]
                if isinstance(loaded_specimen_sampling, Sampling):
                    assert loaded_specimen_sampling.method == specimen_sampling_method
                specimen = loaded_specimen_sampling.specimen
                assert isinstance(specimen, Specimen)
                assert specimen.identifier == specimen_id
                assert specimen.type == specimen_type
                assert specimen.container == specimen_container
                fixation_step = specimen.steps[1]
                assert isinstance(fixation_step, Fixation)
                assert fixation_step.fixative == fixative
                collection_step = specimen.steps[0]
                assert isinstance(collection_step, Collection)
                assert collection_step.method == collection_method

    @pytest.mark.parametrize(
        ["slide_sample_ids", "slide_sample_uids", "specimen_ids"],
        [
            [
                ["slide sample"],
                [
                    UID(
                        "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445424"
                    )
                ],
                ["specimen"],
            ],
            [
                ["slide sample 1", "slide sample 2"],
                [
                    UID(
                        "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445424"
                    ),
                    UID(
                        "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445425"
                    ),
                ],
                ["specimen 1", "specimen 2"],
            ],
        ],
    )
    @pytest.mark.parametrize(
        "fixative", [SpecimenFixativesCode("Neutral Buffered Formalin")]
    )
    @pytest.mark.parametrize("medium", [SpecimenEmbeddingMediaCode("Paraffin wax")])
    @pytest.mark.parametrize("short_description", [None, "short description"])
    @pytest.mark.parametrize("detailed_description", [None, "detailed description"])
    @pytest.mark.parametrize(
        "specimen_container", [None, ContainerTypeCode("Specimen container")]
    )
    @pytest.mark.parametrize(
        "block_container", [None, ContainerTypeCode("Tissue cassette")]
    )
    @pytest.mark.parametrize(
        "substances",
        [
            "HE",
            [
                SpecimenStainsCode("hematoxylin stain"),
                SpecimenStainsCode("water soluble eosin stain"),
            ],
        ],
    )
    def test_slide_sample_to_dataset(
        self,
        slide_sample_ids: Sequence[str],
        slide_sample_uids: Sequence[UID],
        specimen_ids: Sequence[str],
        block_id: str,
        specimen_type: AnatomicPathologySpecimenTypesCode,
        collection_method: SpecimenCollectionProcedureCode,
        fixative: SpecimenFixativesCode,
        specimen_sampling_method: SpecimenSamplingProcedureCode,
        medium: SpecimenEmbeddingMediaCode,
        block_sampling_method: SpecimenSamplingProcedureCode,
        block_type: AnatomicPathologySpecimenTypesCode,
        sample_localization: SampleLocalization,
        sampling_location: SamplingLocation,
        primary_anatomic_structures: Sequence[Code],
        substances: Union[str, Sequence[SpecimenStainsCode]],
        short_description: Optional[str],
        detailed_description: Optional[str],
        specimen_container: Optional[ContainerTypeCode],
        block_container: Optional[ContainerTypeCode],
    ):
        # Arrange
        slide_samples = []
        for slide_sample_id, slide_sample_uid, specimen_id in zip(
            slide_sample_ids, slide_sample_uids, specimen_ids
        ):
            specimen = Specimen(
                identifier=specimen_id,
                extraction_step=Collection(method=collection_method),
                type=specimen_type,
                container=specimen_container,
                steps=[
                    Fixation(fixative=fixative),
                ],
            )
            block = Sample(
                identifier=block_id,
                sampled_from=[
                    specimen.sample(
                        method=specimen_sampling_method,
                        location=sampling_location,
                    )
                ],
                type=block_type,
                container=block_container,
                steps=[
                    Embedding(medium=medium),
                ],
            )
            slide_sample = SlideSample(
                identifier=slide_sample_id,
                uid=slide_sample_uid,
                anatomical_sites=primary_anatomic_structures,
                localization=sample_localization,
                sampled_from=block.sample(
                    method=block_sampling_method,
                ),
                short_description=short_description,
                detailed_description=detailed_description,
            )
            slide_samples.append(slide_sample)
        staining = Staining(substances)

        schema = SpecimenDescriptionDicomSchema()

        # Act
        descriptions = []
        for slide_sample in slide_samples:
            dicom_model = SpecimenDicomFormatter.to_dicom(slide_sample, [staining])
            description = schema.dump(dicom_model)
            descriptions.append(description)

        dataset = Dataset()
        dataset.SpecimenDescriptionSequence = descriptions

        # Assert
        for index, (slide_sample_id, slide_sample_uid, specimen_id) in enumerate(
            zip(slide_sample_ids, slide_sample_uids, specimen_ids)
        ):
            description = dataset.SpecimenDescriptionSequence[index]
            assert isinstance(description, Dataset)
            assert description.SpecimenIdentifier == slide_sample_id
            assert description.SpecimenUID == slide_sample_uid
            if short_description is not None:
                assert description.SpecimenShortDescription == short_description
            else:
                assert "SpecimenShortDescription" not in description
            if detailed_description is not None:
                assert description.SpecimenDetailedDescription == detailed_description
            else:
                assert "SpecimenDetailedDescription" not in description
            assert len(description.PrimaryAnatomicStructureSequence) == len(
                primary_anatomic_structures
            )
            for index, primary_anatomic_structure in enumerate(
                primary_anatomic_structures
            ):
                assert (
                    description.PrimaryAnatomicStructureSequence[index].CodeValue
                    == primary_anatomic_structure.value
                )
                assert (
                    description.PrimaryAnatomicStructureSequence[
                        index
                    ].CodingSchemeDesignator
                    == primary_anatomic_structure.scheme_designator
                )
                assert (
                    description.PrimaryAnatomicStructureSequence[index].CodeMeaning
                    == primary_anatomic_structure.meaning
                )

            step_iterator = iter(description.SpecimenPreparationSequence)

            # Collection
            step = next(step_iterator)
            step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
            assert_next_item_equals_string(
                step_item_iterator, SampleCodes.identifier, specimen_id
            )
            if specimen_container is not None:
                assert_next_item_equals_code(
                    step_item_iterator, SampleCodes.container, specimen_container
                )
            assert_next_item_equals_code(
                step_item_iterator, SampleCodes.specimen_type, specimen_type
            )
            assert_next_item_equals_code(
                step_item_iterator,
                SampleCodes.processing_type,
                SampleCodes.specimen_collection,
            )
            assert_next_item_equals_code(
                step_item_iterator, SampleCodes.specimen_collection, collection_method
            )
            assert next(step_item_iterator, None) is None

            # Fixation
            step = next(step_iterator)
            step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
            assert_next_item_equals_string(
                step_item_iterator, SampleCodes.identifier, specimen_id
            )
            if specimen_container is not None:
                assert_next_item_equals_code(
                    step_item_iterator, SampleCodes.container, specimen_container
                )
            assert_next_item_equals_code(
                step_item_iterator, SampleCodes.specimen_type, specimen_type
            )
            assert_next_item_equals_code(
                step_item_iterator,
                SampleCodes.processing_type,
                SampleCodes.sample_processing,
            )
            assert_next_item_equals_code(
                step_item_iterator, SampleCodes.fixative, fixative
            )
            assert next(step_item_iterator, None) is None

            # Sampling to block
            step = next(step_iterator)
            step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
            assert_next_item_equals_string(
                step_item_iterator, SampleCodes.identifier, block_id
            )
            if block_container is not None:
                assert_next_item_equals_code(
                    step_item_iterator, SampleCodes.container, block_container
                )
            assert_next_item_equals_code(
                step_item_iterator, SampleCodes.specimen_type, block_type
            )
            assert_next_item_equals_code(
                step_item_iterator,
                SampleCodes.processing_type,
                SampleCodes.sampling_of_tissue_specimen,
            )
            assert_next_item_equals_code(
                step_item_iterator,
                SampleCodes.sampling_method,
                specimen_sampling_method,
            )
            assert_next_item_equals_string(
                step_item_iterator, SampleCodes.parent_specimen_identifier, specimen_id
            )
            assert_next_item_equals_code(
                step_item_iterator, SampleCodes.parent_specimen_type, specimen_type
            )
            if sampling_location is not None:
                if sampling_location.reference is not None:
                    assert_next_item_equals_string(
                        step_item_iterator,
                        SampleCodes.location_frame_of_reference,
                        sampling_location.reference,
                    )
                if sampling_location.description is not None:
                    assert_next_item_equals_string(
                        step_item_iterator,
                        SampleCodes.location_of_sampling_site,
                        sampling_location.description,
                    )
                if sampling_location.x is not None:
                    assert_next_item_equals_measurement(
                        step_item_iterator,
                        SampleCodes.location_of_sampling_site_x,
                        sampling_location.x,
                    )
                if sampling_location.y is not None:
                    assert_next_item_equals_measurement(
                        step_item_iterator,
                        SampleCodes.location_of_sampling_site_y,
                        sampling_location.y,
                    )
                if sampling_location.z is not None:
                    assert_next_item_equals_measurement(
                        step_item_iterator,
                        SampleCodes.location_of_sampling_site_z,
                        sampling_location.z,
                    )
            assert next(step_item_iterator, None) is None

            # Embedding
            step = next(step_iterator)
            step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
            assert_next_item_equals_string(
                step_item_iterator, SampleCodes.identifier, block_id
            )
            if block_container is not None:
                assert_next_item_equals_code(
                    step_item_iterator, SampleCodes.container, block_container
                )
            assert_next_item_equals_code(
                step_item_iterator, SampleCodes.specimen_type, block_type
            )
            assert_next_item_equals_code(
                step_item_iterator,
                SampleCodes.processing_type,
                SampleCodes.sample_processing,
            )
            assert_next_item_equals_code(
                step_item_iterator, SampleCodes.embedding, medium
            )
            assert next(step_item_iterator, None) is None

            # Sampling to slide sample
            step = next(step_iterator)
            step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
            assert_next_item_equals_string(
                step_item_iterator, SampleCodes.identifier, slide_sample_id
            )
            assert_next_item_equals_code(
                step_item_iterator,
                SampleCodes.container,
                ContainerTypeCode("Microscope slide"),
            )
            assert_next_item_equals_code(
                step_item_iterator,
                SampleCodes.specimen_type,
                AnatomicPathologySpecimenTypesCode("Slide"),
            )
            assert_next_item_equals_code(
                step_item_iterator,
                SampleCodes.processing_type,
                SampleCodes.sampling_of_tissue_specimen,
            )
            assert_next_item_equals_code(
                step_item_iterator, SampleCodes.sampling_method, block_sampling_method
            )
            assert_next_item_equals_string(
                step_item_iterator, SampleCodes.parent_specimen_identifier, block_id
            )
            assert_next_item_equals_code(
                step_item_iterator, SampleCodes.parent_specimen_type, block_type
            )
            assert next(step_item_iterator, None) is None

            # Staining
            step = next(step_iterator)
            step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
            assert_next_item_equals_string(
                step_item_iterator, SampleCodes.identifier, slide_sample_id
            )
            assert_next_item_equals_code(
                step_item_iterator,
                SampleCodes.container,
                ContainerTypeCode("Microscope slide"),
            )
            assert_next_item_equals_code(
                step_item_iterator,
                SampleCodes.specimen_type,
                AnatomicPathologySpecimenTypesCode("Slide"),
            )
            assert_next_item_equals_code(
                step_item_iterator,
                SampleCodes.processing_type,
                SampleCodes.staining,
            )
            if isinstance(substances, str):
                assert_next_item_equals_string(
                    step_item_iterator, SampleCodes.using_substance, substances
                )
            else:
                for substance in substances:
                    assert_next_item_equals_code(
                        step_item_iterator, SampleCodes.using_substance, substance
                    )
            assert next(step_item_iterator, None) is None

    def test_sampling_from_not_defined_specimen(
        self,
        slide_sample_id: str,
        slide_sample_uid: UID,
        block_id: str,
        block_sampling_method: SpecimenSamplingProcedureCode,
        block_type: AnatomicPathologySpecimenTypesCode,
    ):
        # Arrange
        sampling = create_sampling_dataset(
            SamplingDicomModel(
                identifier=slide_sample_id,
                method=block_sampling_method,
                parent_specimen_identifier=block_id,
                parent_specimen_type=block_type,
            ),
            identifier=slide_sample_id,
        )
        description = create_description_dataset(
            slide_sample_id,
            slide_sample_uid,
            preparation_step_datasets=[sampling],
        )
        dataset = Dataset()
        dataset.SpecimenDescriptionSequence = [description]

        schema = SpecimenDescriptionDicomSchema()

        # Act
        models = [
            schema.load(description)
            for description in dataset.SpecimenDescriptionSequence
        ]
        slide_samples, stainings = SpecimenDicomParser().parse_descriptions(models)

        # Assert
        assert len(stainings) == 0
        assert len(slide_samples) == 1
        slide_sample = slide_samples[0]
        assert slide_sample.sampled_from is not None
        assert isinstance(slide_sample.sampled_from, Sampling)
        assert slide_sample.sampled_from.method == block_sampling_method
        assert isinstance(slide_sample.sampled_from.specimen, Specimen)
        assert slide_sample.sampled_from.specimen.identifier == block_id
        assert slide_sample.sampled_from.specimen.type == block_type

    @pytest.mark.parametrize(
        ["setting", "slide_sample_issuer", "processing_issuer", "processing_expected"],
        [
            (False, "issuer 1", "issuer 1", True),
            (False, "issuer 1", "issuer 2", False),
            (False, "issuer 1", None, True),
            (False, None, "issuer 1", True),
            (False, None, None, True),
            (True, "issuer 1", "issuer 1", True),
            (True, "issuer 1", "issuer 2", False),
            (True, "issuer 1", None, False),
            (True, None, "issuer 1", False),
            (True, None, None, True),
        ],
    )
    def test_handling_of_preparation_step_with_mismatching_issuer(
        self,
        slide_sample_id: str,
        slide_sample_uid: UID,
        setting: bool,
        slide_sample_issuer: Optional[str],
        processing_issuer: Optional[str],
        processing_expected: bool,
    ):
        # Arrange
        settings.strict_specimen_identifier_check = setting
        if slide_sample_issuer is not None:
            local_slide_sample_issuer = LocalIssuerOfIdentifier(slide_sample_issuer)
            slide_sample_identifier = SpecimenIdentifier(
                slide_sample_id, local_slide_sample_issuer
            )
        else:
            local_slide_sample_issuer = None
            slide_sample_identifier = slide_sample_id
        if processing_issuer is not None:
            processing_identifier = SpecimenIdentifier(
                slide_sample_id, LocalIssuerOfIdentifier(processing_issuer)
            )
        else:
            processing_identifier = slide_sample_id
        processing_1_method = SpecimenPreparationStepsCode("Specimen clearing")
        processing_1 = create_processing_dataset(
            ProcessingDicomModel(
                identifier=slide_sample_id,
                processing=processing_1_method,
                issuer_of_identifier=slide_sample_issuer,
            ),
            identifier=slide_sample_identifier,
        )
        processing_2_method = SpecimenPreparationStepsCode("Specimen freezing")
        processing_2 = create_processing_dataset(
            ProcessingDicomModel(
                identifier=slide_sample_id,
                processing=processing_2_method,
                issuer_of_identifier=processing_issuer,
            ),
            identifier=processing_identifier,
        )
        description = create_description_dataset(
            slide_sample_id,
            slide_sample_uid,
            preparation_step_datasets=[processing_1, processing_2],
            slide_sample_issuer=local_slide_sample_issuer,
        )
        dataset = Dataset()
        dataset.SpecimenDescriptionSequence = [description]

        schema = SpecimenDescriptionDicomSchema()

        # Act
        models = [
            schema.load(description)
            for description in dataset.SpecimenDescriptionSequence
        ]
        slide_samples, stainings = SpecimenDicomParser().parse_descriptions(models)

        # Assert
        assert len(slide_samples) == 1
        slide_sample = slide_samples[0]
        if processing_expected:
            assert len(slide_sample.steps) == 2
            assert isinstance(slide_sample.steps[0], Processing)
            assert slide_sample.steps[0].method == processing_1_method
            assert isinstance(slide_sample.steps[1], Processing)
            assert slide_sample.steps[1].method == processing_2_method
        else:
            assert len(slide_sample.steps) == 1
            assert isinstance(slide_sample.steps[0], Processing)
            assert slide_sample.steps[0].method == processing_1_method
