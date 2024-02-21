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
from typing import List, Sequence

import pytest
from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.uid import UID

from tests.metadata.dicom_schema.helpers import (
    assert_next_item_equals_code,
    assert_next_item_equals_datetime,
    assert_next_item_equals_string,
    create_code_dataset,
    create_code_item,
    create_datetime_item,
    create_specimen_preparation_dataset,
    create_string_item,
)
from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    ContainerTypeCode,
    SpecimenCollectionProcedureCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenSamplingProcedureCode,
    SpecimenStainsCode,
)
from wsidicom.metadata.sample import (
    Collection,
    Embedding,
    Fixation,
    LocalIssuerOfIdentifier,
    Receiving,
    Sample,
    Sampling,
    SamplingLocation,
    SlideSample,
    Specimen,
    SpecimenIdentifier,
    Staining,
    UnknownSampling,
)
from wsidicom.metadata.schema.dicom.sample.formatter import SpecimenDicomFormatter
from wsidicom.metadata.schema.dicom.sample.parser import SpecimenDicomParser
from wsidicom.metadata.schema.dicom.sample.schema import (
    SampleCodes,
    SpecimenDescriptionDicomSchema,
)

"""
 Test based on example in
https://dicom.nema.org/medical/dicom/current/output/chtml/part17/sect_NN.6.2.html
The specimen type is changed from "Anatomic part" to "tissue specimen" as
"Anatomic part" is not part of CID 8103.
The embedding medium is changed from "Paraffin" to "Parrafin wax". as
"Paraffin" is not part of CID 8114.
Issuer of identifier has been added to last steps as an issuer is defined
in the dataset
"""


@pytest.fixture
def issuer_of_identifier():
    return "Case Medical Center"


@pytest.fixture
def slide_identifier():
    return "S07-100 A 5 1"


@pytest.fixture
def specimen_uid():
    return UID("1.2.840.99790.986.33.1677.1.1.19.5")


@pytest.fixture
def slide_short_description():
    return "Part A: LEFT UPPER LOBE, Block 5: Mass (2 pc), Slide 1: H&E"


@pytest.fixture
def slide_detailed_description():
    return (
        "A: Received fresh for intraoperative consultation, labeled with the "
        'patient\'s name, number and "left upper lobe," is a pink-tan, '
        "wedge-shaped segment of soft tissue, 6.9 x 4.2 x 1.0 cm. The pleural "
        "surface is pink-tan and glistening with a stapled line measuring 12.0 cm. "
        "in length. The pleural surface shows a 0.5 cm. area of puckering. "
        "The pleural surface is inked black. The cut surface reveals a 1.2 x 1.1 "
        "cm, white-gray, irregular mass abutting the pleural surface and deep to "
        "the puckered area. The remainder of the cut surface is red-brown and "
        "congested. No other lesions are identified. Representative sections are "
        "submitted."
        ""
        'Block 5: "Mass" (2 pieces)'
    )


@pytest.fixture
def primary_anatomic_structure():
    return Code("44714003", "SCT", "Left upper lobe of lung")


@pytest.fixture
def specimen_identifier():
    return "S07-100 A"


@pytest.fixture
def specimen_container():
    return ContainerTypeCode("Specimen container")


@pytest.fixture
def specimen_type():
    return AnatomicPathologySpecimenTypesCode("tissue specimen")


@pytest.fixture
def collection_method():
    return SpecimenCollectionProcedureCode("Excision")


@pytest.fixture
def collection_description():
    return "Taken"


@pytest.fixture
def collection_datetime():
    return datetime.datetime(2007, 3, 23, 8, 27)


@pytest.fixture
def specimen_receiving_datetime():
    return datetime.datetime(2007, 3, 23, 9, 43)


@pytest.fixture
def block_identifier():
    return "S07-100 A 5"


@pytest.fixture
def block_container():
    return ContainerTypeCode("Tissue cassette")


@pytest.fixture
def block_type():
    return AnatomicPathologySpecimenTypesCode("Gross specimen")


@pytest.fixture
def block_sampling_method():
    return SpecimenSamplingProcedureCode("Dissection")


@pytest.fixture
def block_sampling_description():
    return "Block Creation"


@pytest.fixture
def block_sampling_location():
    return "Mass"


@pytest.fixture
def block_fixation_datetime():
    return datetime.datetime(2007, 3, 23, 19, 0)


@pytest.fixture
def block_fixative():
    return SpecimenFixativesCode("Formalin")


@pytest.fixture
def block_fixation_description():
    return "Standard Block Processing (Formalin)"


@pytest.fixture
def block_embedding_datetime():
    return datetime.datetime(2007, 3, 24, 5, 0)


@pytest.fixture
def block_embedding_medium():
    return SpecimenEmbeddingMediaCode("Paraffin wax")


@pytest.fixture
def block_embedding_description():
    return "Embedding (paraffin)"


@pytest.fixture
def staining_datetime():
    return datetime.datetime(2007, 3, 24, 7, 0)


@pytest.fixture
def staining_substances():
    return [
        SpecimenStainsCode("hematoxylin stain"),
        SpecimenStainsCode("water soluble eosin stain"),
    ]


class TestExampleSpecimenModule:
    def test_load_example_specimen_module(
        self,
        issuer_of_identifier: str,
        slide_identifier: str,
        specimen_uid: UID,
        slide_short_description: str,
        slide_detailed_description: str,
        primary_anatomic_structure: Code,
        specimen_identifier: str,
        specimen_container: ContainerTypeCode,
        specimen_type: AnatomicPathologySpecimenTypesCode,
        collection_method: SpecimenCollectionProcedureCode,
        collection_description: str,
        collection_datetime: datetime.datetime,
        specimen_receiving_datetime: datetime.datetime,
        block_identifier: str,
        block_container: ContainerTypeCode,
        block_type: AnatomicPathologySpecimenTypesCode,
        block_sampling_method: SpecimenSamplingProcedureCode,
        block_sampling_description: str,
        block_sampling_location: str,
        block_fixation_datetime: datetime.datetime,
        block_fixative: SpecimenFixativesCode,
        block_fixation_description: str,
        block_embedding_datetime: datetime.datetime,
        block_embedding_medium: SpecimenEmbeddingMediaCode,
        block_embedding_description: str,
        staining_datetime: datetime.datetime,
        staining_substances: Sequence[SpecimenStainsCode],
    ):
        # Arrange
        steps: List[List[Dataset]] = [
            # Part Collection in OR
            [
                create_string_item(SampleCodes.identifier, specimen_identifier),
                create_string_item(
                    SampleCodes.issuer_of_identifier, issuer_of_identifier
                ),
                create_code_item(
                    SampleCodes.container,
                    specimen_container,
                ),
                create_code_item(
                    SampleCodes.specimen_type,
                    specimen_type,
                ),
                create_code_item(
                    SampleCodes.processing_type, SampleCodes.specimen_collection
                ),
                create_datetime_item(
                    SampleCodes.datetime_of_processing, collection_datetime
                ),
                create_string_item(
                    SampleCodes.processing_description, collection_description
                ),
                create_code_item(
                    SampleCodes.specimen_collection,
                    collection_method,
                ),
            ],
            # Specimen received in Pathology department
            [
                create_string_item(SampleCodes.identifier, specimen_identifier),
                create_string_item(
                    SampleCodes.issuer_of_identifier, issuer_of_identifier
                ),
                create_code_item(
                    SampleCodes.container,
                    specimen_container,
                ),
                create_code_item(
                    SampleCodes.specimen_type,
                    specimen_type,
                ),
                create_code_item(SampleCodes.processing_type, SampleCodes.receiving),
                create_datetime_item(
                    SampleCodes.datetime_of_processing,
                    specimen_receiving_datetime,
                ),
            ],
            # Sampling to block
            [
                create_string_item(SampleCodes.identifier, block_identifier),
                create_string_item(
                    SampleCodes.issuer_of_identifier, issuer_of_identifier
                ),
                create_code_item(
                    SampleCodes.container,
                    block_container,
                ),
                create_code_item(SampleCodes.specimen_type, block_type),
                create_code_item(
                    SampleCodes.processing_type, SampleCodes.sampling_of_tissue_specimen
                ),
                create_string_item(
                    SampleCodes.processing_description, block_sampling_description
                ),
                create_code_item(
                    SampleCodes.sampling_method,
                    block_sampling_method,
                ),
                create_string_item(
                    SampleCodes.parent_specimen_identifier,
                    specimen_identifier,
                ),
                create_string_item(
                    SampleCodes.issuer_of_parent_specimen_identifier,
                    issuer_of_identifier,
                ),
                create_code_item(
                    SampleCodes.parent_specimen_type,
                    specimen_type,
                ),
                create_string_item(
                    SampleCodes.location_of_sampling_site,
                    block_sampling_location,
                ),
            ],
            # Block processing
            [
                create_string_item(SampleCodes.identifier, block_identifier),
                create_string_item(
                    SampleCodes.issuer_of_identifier, issuer_of_identifier
                ),
                create_code_item(
                    SampleCodes.container,
                    block_container,
                ),
                create_code_item(SampleCodes.specimen_type, block_type),
                create_code_item(
                    SampleCodes.processing_type, SampleCodes.sample_processing
                ),
                create_datetime_item(
                    SampleCodes.datetime_of_processing,
                    block_fixation_datetime,
                ),
                create_string_item(
                    SampleCodes.processing_description,
                    block_fixation_description,
                ),
                create_code_item(
                    SampleCodes.fixative,
                    block_fixative,
                ),
            ],
            # Block embedding
            [
                create_string_item(SampleCodes.identifier, block_identifier),
                create_string_item(
                    SampleCodes.issuer_of_identifier, issuer_of_identifier
                ),
                create_code_item(
                    SampleCodes.container,
                    block_container,
                ),
                create_code_item(SampleCodes.specimen_type, block_type),
                create_code_item(
                    SampleCodes.processing_type, SampleCodes.sample_processing
                ),
                create_datetime_item(
                    SampleCodes.datetime_of_processing,
                    block_embedding_datetime,
                ),
                create_string_item(
                    SampleCodes.processing_description, block_embedding_description
                ),
                create_code_item(
                    SampleCodes.embedding,
                    block_embedding_medium,
                ),
            ],
            # Slide Staining
            [
                create_string_item(SampleCodes.identifier, slide_identifier),
                create_string_item(
                    SampleCodes.issuer_of_identifier, issuer_of_identifier
                ),
                create_code_item(SampleCodes.processing_type, SampleCodes.staining),
                create_datetime_item(
                    SampleCodes.datetime_of_processing,
                    staining_datetime,
                ),
                create_code_item(SampleCodes.using_substance, staining_substances[0]),
                create_code_item(SampleCodes.using_substance, staining_substances[1]),
            ],
        ]
        dataset = Dataset()
        dataset.SpecimenIdentifier = slide_identifier
        issuer_of_identifier_dataset = Dataset()
        issuer_of_identifier_dataset.LocalNamespaceEntityID = issuer_of_identifier
        dataset.IssuerOfTheSpecimenIdentifierSequence = [issuer_of_identifier_dataset]
        dataset.SpecimenUID = specimen_uid
        dataset.SpecimenShortDescription = slide_short_description
        dataset.SpecimenDetailedDescription = slide_detailed_description
        dataset.PrimaryAnatomicStructureSequence = [
            create_code_dataset(primary_anatomic_structure)
        ]
        dataset.SpecimenPreparationSequence = [
            create_specimen_preparation_dataset(step) for step in steps
        ]
        schema = SpecimenDescriptionDicomSchema()

        # Act
        model = schema.load(dataset)
        slide_samples, stainings = SpecimenDicomParser().parse_descriptions([model])

        # Assert
        assert slide_samples is not None
        assert len(slide_samples) == 1
        slide_sample = slide_samples[0]
        assert isinstance(slide_sample, SlideSample)
        assert slide_sample.identifier == SpecimenIdentifier(
            slide_identifier, LocalIssuerOfIdentifier(issuer_of_identifier)
        )
        assert slide_sample.uid == specimen_uid
        assert slide_sample.anatomical_sites == [primary_anatomic_structure]
        assert slide_sample.localization is None
        assert slide_sample.sampled_from is not None
        assert isinstance(slide_sample.sampled_from, UnknownSampling)
        block = slide_sample.sampled_from.specimen
        assert isinstance(block, Sample)
        assert block.identifier == SpecimenIdentifier(
            block_identifier, LocalIssuerOfIdentifier(issuer_of_identifier)
        )
        assert block.type == block_type
        assert block.container == block_container
        fixation_step = block.steps[0]
        assert isinstance(fixation_step, Fixation)
        assert fixation_step.fixative == block_fixative
        assert fixation_step.date_time == block_fixation_datetime
        assert fixation_step.description == block_fixation_description
        embedding_step = block.steps[1]
        assert isinstance(embedding_step, Embedding)
        assert embedding_step.medium == block_embedding_medium
        assert embedding_step.date_time == block_embedding_datetime
        assert embedding_step.description == block_embedding_description
        sampling_to_slide_step = block.steps[2]
        assert isinstance(sampling_to_slide_step, UnknownSampling)
        assert len(block.sampled_from) == 1
        specimen = block.sampled_from[0].specimen
        assert isinstance(specimen, Specimen)
        assert specimen.identifier == SpecimenIdentifier(
            specimen_identifier, LocalIssuerOfIdentifier(issuer_of_identifier)
        )
        assert specimen.type == specimen_type
        assert specimen.container == specimen_container
        collection_step = specimen.steps[0]
        assert isinstance(collection_step, Collection)
        assert collection_step.method == collection_method
        assert collection_step.date_time == collection_datetime
        assert collection_step.description == collection_description
        receiving_step = specimen.steps[1]
        assert isinstance(receiving_step, Receiving)
        assert receiving_step.date_time == specimen_receiving_datetime
        sampling_to_block_step = specimen.steps[2]
        assert isinstance(sampling_to_block_step, Sampling)
        assert sampling_to_block_step.method == block_sampling_method
        assert sampling_to_block_step.description == block_sampling_description
        assert sampling_to_block_step.location == SamplingLocation(
            description=block_sampling_location
        )
        assert stainings is not None
        assert len(stainings) == 1
        staining = stainings[0]
        assert staining.substances == staining_substances
        assert staining.date_time == staining_datetime

    def test_create_example_specimen_module(
        self,
        issuer_of_identifier: str,
        slide_identifier: str,
        specimen_uid: UID,
        slide_short_description: str,
        slide_detailed_description: str,
        primary_anatomic_structure: Code,
        specimen_identifier: str,
        specimen_container: ContainerTypeCode,
        specimen_type: AnatomicPathologySpecimenTypesCode,
        collection_method: SpecimenCollectionProcedureCode,
        collection_description: str,
        collection_datetime: datetime.datetime,
        specimen_receiving_datetime: datetime.datetime,
        block_identifier: str,
        block_container: ContainerTypeCode,
        block_type: AnatomicPathologySpecimenTypesCode,
        block_sampling_method: SpecimenSamplingProcedureCode,
        block_sampling_description: str,
        block_sampling_location: str,
        block_fixation_datetime: datetime.datetime,
        block_fixative: SpecimenFixativesCode,
        block_fixation_description: str,
        block_embedding_datetime: datetime.datetime,
        block_embedding_medium: SpecimenEmbeddingMediaCode,
        block_embedding_description: str,
        staining_datetime: datetime.datetime,
        staining_substances: Sequence[SpecimenStainsCode],
    ):
        local_issuer_of_identifier = LocalIssuerOfIdentifier(issuer_of_identifier)
        specimen = Specimen(
            identifier=SpecimenIdentifier(
                specimen_identifier, local_issuer_of_identifier
            ),
            extraction_step=Collection(
                method=collection_method,
                date_time=collection_datetime,
                description=collection_description,
            ),
            type=specimen_type,
            container=specimen_container,
            steps=[
                Receiving(date_time=specimen_receiving_datetime),
            ],
        )

        block_sampling = specimen.sample(
            method=block_sampling_method,
            description=block_sampling_description,
            location=SamplingLocation(description=block_sampling_location),
        )
        block = Sample(
            identifier=SpecimenIdentifier(block_identifier, local_issuer_of_identifier),
            sampled_from=[block_sampling],
            type=block_type,
            steps=[
                Fixation(
                    fixative=block_fixative,
                    date_time=block_fixation_datetime,
                    description=block_fixation_description,
                ),
                Embedding(
                    medium=block_embedding_medium,
                    date_time=block_embedding_datetime,
                    description=block_embedding_description,
                ),
            ],
            container=block_container,
        )
        slide_sampling = block.sample()
        slide_sample = SlideSample(
            identifier=SpecimenIdentifier(slide_identifier, local_issuer_of_identifier),
            anatomical_sites=[primary_anatomic_structure],
            sampled_from=slide_sampling,
            uid=specimen_uid,
            short_description=slide_short_description,
            detailed_description=slide_detailed_description,
        )

        staining = Staining(
            substances=staining_substances,
            date_time=staining_datetime,
        )

        schema = SpecimenDescriptionDicomSchema()

        # Act
        dicom_model = SpecimenDicomFormatter.to_dicom(slide_sample, [staining])
        dataset = schema.dump(dicom_model)

        # Assert
        assert isinstance(dataset, Dataset)
        assert dataset.SpecimenIdentifier == slide_identifier
        assert len(dataset.IssuerOfTheSpecimenIdentifierSequence) == 1
        assert (
            dataset.IssuerOfTheSpecimenIdentifierSequence[0].LocalNamespaceEntityID
            == issuer_of_identifier
        )
        assert dataset.SpecimenUID == specimen_uid
        assert dataset.SpecimenShortDescription == slide_short_description
        assert dataset.SpecimenDetailedDescription == slide_detailed_description
        assert len(dataset.PrimaryAnatomicStructureSequence) == 1

        assert (
            dataset.PrimaryAnatomicStructureSequence[0].CodeValue
            == primary_anatomic_structure.value
        )
        assert (
            dataset.PrimaryAnatomicStructureSequence[0].CodingSchemeDesignator
            == primary_anatomic_structure.scheme_designator
        )
        assert (
            dataset.PrimaryAnatomicStructureSequence[0].CodeMeaning
            == primary_anatomic_structure.meaning
        )

        step_iterator = iter(dataset.SpecimenPreparationSequence)
        # Part Collection in OR
        step = next(step_iterator)
        step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.identifier, specimen_identifier
        )
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.issuer_of_identifier, issuer_of_identifier
        )
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
        assert_next_item_equals_datetime(
            step_item_iterator, SampleCodes.datetime_of_processing, collection_datetime
        )
        assert_next_item_equals_string(
            step_item_iterator,
            SampleCodes.processing_description,
            collection_description,
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.specimen_collection, collection_method
        )
        assert next(step_item_iterator, None) is None

        # Specimen received in Pathology department
        step = next(step_iterator)
        step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.identifier, specimen_identifier
        )
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.issuer_of_identifier, issuer_of_identifier
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.container, specimen_container
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.specimen_type, specimen_type
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.processing_type, SampleCodes.receiving
        )
        assert_next_item_equals_datetime(
            step_item_iterator,
            SampleCodes.datetime_of_processing,
            specimen_receiving_datetime,
        )
        assert next(step_item_iterator, None) is None

        # Sampling to block
        step = next(step_iterator)
        step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.identifier, block_identifier
        )
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.issuer_of_identifier, issuer_of_identifier
        )
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
        assert_next_item_equals_string(
            step_item_iterator,
            SampleCodes.processing_description,
            block_sampling_description,
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.sampling_method, block_sampling_method
        )
        assert_next_item_equals_string(
            step_item_iterator,
            SampleCodes.parent_specimen_identifier,
            specimen_identifier,
        )
        assert_next_item_equals_string(
            step_item_iterator,
            SampleCodes.issuer_of_parent_specimen_identifier,
            issuer_of_identifier,
        )
        assert_next_item_equals_code(
            step_item_iterator,
            SampleCodes.parent_specimen_type,
            specimen_type,
        )
        assert_next_item_equals_string(
            step_item_iterator,
            SampleCodes.location_of_sampling_site,
            block_sampling_location,
        )
        assert next(step_item_iterator, None) is None

        # Block processing
        step = next(step_iterator)
        step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.identifier, block_identifier
        )
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.issuer_of_identifier, issuer_of_identifier
        )
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

        assert_next_item_equals_datetime(
            step_item_iterator,
            SampleCodes.datetime_of_processing,
            block_fixation_datetime,
        )
        assert_next_item_equals_string(
            step_item_iterator,
            SampleCodes.processing_description,
            block_fixation_description,
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.fixative, block_fixative
        )
        assert next(step_item_iterator, None) is None

        # Block embedding
        step = next(step_iterator)
        step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.identifier, block_identifier
        )
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.issuer_of_identifier, issuer_of_identifier
        )
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
        assert_next_item_equals_datetime(
            step_item_iterator,
            SampleCodes.datetime_of_processing,
            block_embedding_datetime,
        )
        assert_next_item_equals_string(
            step_item_iterator,
            SampleCodes.processing_description,
            block_embedding_description,
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.embedding, block_embedding_medium
        )
        assert next(step_item_iterator, None) is None

        # Slide Staining
        step = next(step_iterator)
        step_item_iterator = iter(step.SpecimenPreparationStepContentItemSequence)
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.identifier, slide_identifier
        )
        assert_next_item_equals_string(
            step_item_iterator, SampleCodes.issuer_of_identifier, issuer_of_identifier
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.container, slide_sample.container
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.specimen_type, slide_sample.type
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.processing_type, SampleCodes.staining
        )
        assert_next_item_equals_datetime(
            step_item_iterator, SampleCodes.datetime_of_processing, staining_datetime
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.using_substance, staining_substances[0]
        )
        assert_next_item_equals_code(
            step_item_iterator, SampleCodes.using_substance, staining_substances[1]
        )
        assert next(step_item_iterator, None) is None

        assert next(step_iterator, None) is None
