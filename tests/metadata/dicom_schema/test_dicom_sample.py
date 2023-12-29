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
from typing import List, Optional, Sequence, Union
from pydicom import Dataset
from pydicom.valuerep import DSfloat
import pytest

from pydicom.uid import UID
from pydicom.sr.coding import Code
from wsidicom.conceptcode import (
    ConceptCode,
    SpecimenCollectionProcedureCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenPreparationStepsCode,
    SpecimenStainsCode,
    SpecimenSamplingProcedureCode,
    AnatomicPathologySpecimenTypesCode,
    UnitCode,
)
from wsidicom.metadata.dicom_schema.sample.model import (
    CollectionDicomModel,
    SpecimenPreparationStepDicomModel,
    ProcessingDicomModel,
    StainingDicomModel,
    SamplingDicomModel,
)
from wsidicom.metadata.dicom_schema.sample.parser import SpecimenDicomParser
from wsidicom.metadata.dicom_schema.sample.schema import (
    CollectionDicomSchema,
    ProcessingDicomSchema,
    SamplingDicomSchema,
    SpecimenDescriptionDicomSchema,
    SpecimenLocalizationDicomSchema,
    StainingDicomSchema,
    SampleCodes,
)

from wsidicom.metadata.sample import (
    Collection,
    Embedding,
    ExtractedSpecimen,
    Fixation,
    LocalIssuerOfIdentifier,
    Measurement,
    Sample,
    SlideSample,
    SpecimenLocalization,
    SpecimenIdentifier,
    Sampling,
    Processing,
    Staining,
)


@pytest.fixture()
def specimen_id():
    yield "specimen"


@pytest.fixture()
def slide_sample_id():
    yield "slide sample"


@pytest.fixture()
def slide_sample_uid():
    yield UID("1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445424")


@pytest.fixture()
def specimen_type():
    yield AnatomicPathologySpecimenTypesCode("tissue specimen")


@pytest.fixture()
def collection_method():
    yield SpecimenCollectionProcedureCode("Excision")


@pytest.fixture()
def specimen_sampling_method():
    yield SpecimenSamplingProcedureCode("Dissection")


@pytest.fixture()
def block_id():
    yield "block"


@pytest.fixture()
def block_sampling_method():
    yield SpecimenSamplingProcedureCode("Block sectioning")


@pytest.fixture()
def block_type():
    yield AnatomicPathologySpecimenTypesCode("tissue specimen")


@pytest.fixture()
def position():
    yield SpecimenLocalization(description="left")


@pytest.fixture()
def primary_anatomic_structures():
    yield [Code("value", "schema", "meaning")]


@pytest.fixture()
def stains():
    yield [
        SpecimenStainsCode("hematoxylin stain"),
        SpecimenStainsCode("water soluble eosin stain"),
    ]


@pytest.fixture()
def collection_dicom(
    collection: Collection,
    identifier: Union[str, SpecimenIdentifier],
    fixative: Optional[SpecimenFixativesCode],
    embedding: Optional[SpecimenEmbeddingMediaCode],
    processing_method: Optional[SpecimenPreparationStepsCode],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    yield CollectionDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=collection.date_time,
        description=collection.description,
        fixative=fixative,
        embedding=embedding,
        method=collection.method,
        processing=processing_method,
    )


def create_collection_dataset(
    collection_dicom: CollectionDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(collection_dicom))
    if collection_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, collection_dicom.date_time
            ),
        )
    if collection_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, collection_dicom.description
            )
        )
    items.append(
        create_code_item(SampleCodes.specimen_collection, collection_dicom.method)
    )
    if collection_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, collection_dicom.fixative))
    if collection_dicom.embedding is not None:
        items.append(
            create_code_item(SampleCodes.embedding, collection_dicom.embedding)
        )
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


@pytest.fixture()
def collection_dataset(
    collection_dicom: CollectionDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_collection_dataset(collection_dicom, identifier)


@pytest.fixture()
def sampling_dicom(
    sampling: Sampling,
    identifier: Union[str, SpecimenIdentifier],
    fixative: Optional[SpecimenFixativesCode],
    embedding: Optional[SpecimenEmbeddingMediaCode],
    processing_method: Optional[SpecimenPreparationStepsCode],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    (
        parent_identifier,
        parent_issuer,
    ) = SpecimenIdentifier.get_string_identifier_and_issuer(
        sampling.specimen.identifier
    )
    yield SamplingDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=sampling.date_time,
        description=sampling.description,
        method=sampling.method,
        fixative=fixative,
        embedding=embedding,
        parent_specimen_identifier=parent_identifier,
        issuer_of_parent_specimen_identifier=parent_issuer,
        parent_specimen_type=sampling.specimen.type,
        processing=processing_method,
        location_reference=sampling.location.reference
        if sampling.location is not None
        else None,
        location_description=sampling.location.description
        if sampling.location is not None
        else None,
        location_x=sampling.location.x if sampling.location is not None else None,
        location_y=sampling.location.y if sampling.location is not None else None,
        location_z=sampling.location.z if sampling.location is not None else None,
    )


def create_sampling_dataset(
    sampling_dicom: SamplingDicomModel, identifier: Union[str, SpecimenIdentifier]
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(sampling_dicom))
    if sampling_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, sampling_dicom.date_time
            ),
        )
    if sampling_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, sampling_dicom.description
            )
        )
    items.append(create_code_item(SampleCodes.sampling_method, sampling_dicom.method))
    items.append(
        create_string_item(
            SampleCodes.parent_specimen_identifier,
            sampling_dicom.parent_specimen_identifier,
        )
    )
    if sampling_dicom.issuer_of_parent_specimen_identifier is not None:
        items.append(
            create_string_item(
                SampleCodes.issuer_of_parent_specimen_identifier,
                sampling_dicom.issuer_of_parent_specimen_identifier,
            )
        )
    items.append(
        create_code_item(
            SampleCodes.parent_specimen_type, sampling_dicom.parent_specimen_type
        )
    )
    if sampling_dicom.location_reference is not None:
        items.append(
            create_string_item(
                SampleCodes.location_frame_of_reference,
                sampling_dicom.location_reference,
            )
        )
    if sampling_dicom.location_description is not None:
        items.append(
            create_string_item(
                SampleCodes.location_of_sampling_site,
                sampling_dicom.location_description,
            )
        )
    if sampling_dicom.location_x is not None:
        items.append(
            create_measurement_item(
                SampleCodes.location_of_sampling_site_x, sampling_dicom.location_x
            )
        )
    if sampling_dicom.location_y is not None:
        items.append(
            create_measurement_item(
                SampleCodes.location_of_sampling_site_y, sampling_dicom.location_y
            )
        )
    if sampling_dicom.location_z is not None:
        items.append(
            create_measurement_item(
                SampleCodes.location_of_sampling_site_z, sampling_dicom.location_z
            )
        )
    if sampling_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, sampling_dicom.fixative))
    if sampling_dicom.embedding is not None:
        items.append(create_code_item(SampleCodes.embedding, sampling_dicom.embedding))
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


@pytest.fixture()
def sampling_dataset(
    sampling_dicom: SamplingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_sampling_dataset(sampling_dicom, identifier)


@pytest.fixture()
def processing_dicom(
    processing: Processing,
    identifier: Union[str, SpecimenIdentifier],
    fixative: Optional[SpecimenFixativesCode],
    embedding: Optional[SpecimenEmbeddingMediaCode],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    yield ProcessingDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=processing.date_time,
        description=processing.description,
        processing=processing.method,
        fixative=fixative,
        embedding=embedding,
    )


def create_processing_dataset(
    processing_dicom: ProcessingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(processing_dicom))
    if processing_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, processing_dicom.date_time
            ),
        )
    if processing_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, processing_dicom.description
            )
        )
    if processing_dicom.processing is not None:
        items.append(
            create_code_item(
                SampleCodes.processing_description, processing_dicom.processing
            )
        )
    if processing_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, processing_dicom.fixative))
    if processing_dicom.embedding is not None:
        items.append(
            create_code_item(SampleCodes.embedding, processing_dicom.embedding)
        )
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


@pytest.fixture()
def processing_dataset(
    processing_dicom: ProcessingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_processing_dataset(processing_dicom, identifier)


@pytest.fixture()
def staining_dicom(
    staining: Staining,
    identifier: Union[str, SpecimenIdentifier],
    fixative: Optional[SpecimenFixativesCode],
    embedding: Optional[SpecimenEmbeddingMediaCode],
    processing_method: Optional[SpecimenPreparationStepsCode],
):
    identifier, issuer = SpecimenIdentifier.get_string_identifier_and_issuer(identifier)
    yield StainingDicomModel(
        identifier=identifier,
        issuer_of_identifier=issuer,
        date_time=staining.date_time,
        description=staining.description,
        substances=staining.substances,
        fixative=fixative,
        embedding=embedding,
        processing=processing_method,
    )


def create_staining_dataset(
    staining_dicom: StainingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(staining_dicom))
    if staining_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, staining_dicom.date_time
            ),
        )
    if staining_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, staining_dicom.description
            )
        )
    for substance in staining_dicom.substances:
        items.append(
            create_code_item(SampleCodes.using_substance, substance)
            if isinstance(substance, SpecimenStainsCode)
            else create_string_item(SampleCodes.using_substance, substance)
        )
    if staining_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, staining_dicom.fixative))
    if staining_dicom.embedding is not None:
        items.append(create_code_item(SampleCodes.embedding, staining_dicom.embedding))
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


@pytest.fixture()
def staining_dataset(
    staining_dicom: StainingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    yield create_staining_dataset(staining_dicom, identifier)


@pytest.fixture()
def description_dataset(
    collection_dataset: Dataset,
):
    pass


def create_description(
    slide_sample_id: str,
    slide_sample_uid: UID,
    specimen_id: str,
    block_id: str,
    specimen_type: AnatomicPathologySpecimenTypesCode,
    collection_method: SpecimenCollectionProcedureCode,
    fixative: SpecimenFixativesCode,
    specimen_sampling_method: SpecimenSamplingProcedureCode,
    embedding_medium: SpecimenEmbeddingMediaCode,
    block_sampling_method: SpecimenSamplingProcedureCode,
    block_type: AnatomicPathologySpecimenTypesCode,
    position: SpecimenLocalization,
    primary_anatomic_structures: Sequence[Code],
    stains: Sequence[SpecimenStainsCode],
    short_description: Optional[str] = None,
    detailed_description: Optional[str] = None,
):
    collection = create_collection_dataset(
        CollectionDicomModel(
            identifier=specimen_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=None,
            method=collection_method,
            processing=None,
        ),
        identifier=specimen_id,
    )
    fixation = create_processing_dataset(
        ProcessingDicomModel(
            identifier=specimen_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=fixative,
            embedding=None,
            processing=None,
        ),
        identifier=specimen_id,
    )
    sampling_to_block = create_sampling_dataset(
        SamplingDicomModel(
            identifier=specimen_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=None,
            method=specimen_sampling_method,
            parent_specimen_identifier=specimen_id,
            issuer_of_parent_specimen_identifier=None,
            parent_specimen_type=specimen_type,
            processing=None,
            location_reference=position.reference if position is not None else None,
            location_description=position.description if position is not None else None,
            location_x=position.x if position is not None else None,
            location_y=position.y if position is not None else None,
            location_z=position.z if position is not None else None,
        ),
        identifier=block_id,
    )
    embedding = create_processing_dataset(
        ProcessingDicomModel(
            identifier=block_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=embedding_medium,
            processing=None,
        ),
        identifier=block_id,
    )
    sampling_to_slide = create_sampling_dataset(
        SamplingDicomModel(
            identifier=block_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=None,
            method=block_sampling_method,
            parent_specimen_identifier=block_id,
            issuer_of_parent_specimen_identifier=None,
            parent_specimen_type=block_type,
            processing=None,
            location_reference=None,
            location_description=None,
            location_x=None,
            location_y=None,
            location_z=None,
        ),
        identifier=slide_sample_id,
    )
    staining = create_staining_dataset(
        StainingDicomModel(
            identifier=slide_sample_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            substances=list(stains),
            fixative=None,
            embedding=None,
            processing=None,
        ),
        identifier=slide_sample_id,
    )
    description = Dataset()
    description.SpecimenIdentifier = slide_sample_id
    description.SpecimenUID = slide_sample_uid
    description.SpecimenPreparationStepContentItemSequence = [
        collection,
        fixation,
        sampling_to_block,
        embedding,
        sampling_to_slide,
        staining,
    ]
    description.SpecimenTypeCodeSequence = [
        create_code_dataset(AnatomicPathologySpecimenTypesCode("Slide"))
    ]
    description.PrimaryAnatomicStructureSequence = [
        create_code_dataset(item) for item in primary_anatomic_structures
    ]
    description.SpecimenShortDescription = short_description
    description.SpecimenDetailedDescription = detailed_description
    return description


def create_identifier_items(identifier: Union[str, SpecimenIdentifier]):
    identifier_item = create_string_item(
        SampleCodes.identifier,
        identifier if isinstance(identifier, str) else identifier.value,
    )
    if not isinstance(identifier, SpecimenIdentifier) or identifier.issuer is None:
        return [identifier_item]
    issuer_item = create_string_item(
        SampleCodes.issuer_of_identifier,
        identifier.issuer.to_hl7v2(),
    )
    return [identifier_item, issuer_item]


def create_code_dataset(code: Union[Code, ConceptCode]):
    dataset = Dataset()
    dataset.CodeValue = code.value
    dataset.CodingSchemeDesignator = code.scheme_designator
    dataset.CodeMeaning = code.meaning
    dataset.CodingSchemeVersion = code.scheme_version
    return dataset


def create_string_item(name: Code, value: str):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.TextValue = value
    return dataset


def create_datetime_item(name: Code, value: datetime.datetime):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.DateTime = value
    return dataset


def create_code_item(name: Code, value: Union[Code, ConceptCode]):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.ConceptCodeSequence = [create_code_dataset(value)]
    return dataset


def create_measurement_item(name: Code, value: Measurement):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.NumericValue = DSfloat(value.value)
    dataset.FloatingPointValue = value.value
    dataset.MeasurementUnitsCodeSequence = [create_code_dataset(value.unit)]
    return dataset


def create_processing_type_item(step: SpecimenPreparationStepDicomModel):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(SampleCodes.processing_type)]
    if isinstance(step, CollectionDicomModel):
        processing_type_code = SampleCodes.specimen_collection
    elif isinstance(step, SamplingDicomModel):
        processing_type_code = SampleCodes.sampling_method
    elif isinstance(step, SpecimenPreparationStepDicomModel):
        processing_type_code = SampleCodes.sample_processing
    elif isinstance(step, StainingDicomModel):
        processing_type_code = SampleCodes.staining
    else:
        raise NotImplementedError()
    dataset.ConceptCodeSequence = [create_code_dataset(processing_type_code)]
    return dataset


class TestSampleDicom:
    # def test_collection_from_dataset(self):
    #     # Arrange
    #     method = SpecimenCollectionProcedureCode("Excision")
    #     dataset = SpecimenPreparationStep(
    #         "identifier",
    #         SpecimenCollection(
    #             procedure=method.code,
    #         ),
    #         processing_description="description",
    #     )

    #     # Act
    #     collection = CollectionDicomModel.from_dataset(dataset)

    #     # Assert
    #     assert collection.method == method

    # def test_processing_from_dataset(self):
    #     # Arrange
    #     method = SpecimenPreparationStepsCode("Specimen clearing")
    #     dataset = SpecimenPreparationStep(
    #         "identifier",
    #         SpecimenProcessing(
    #             description=method.code,
    #         ),
    #     )

    #     # Act
    #     processing = ProcessingDicomModel.from_dataset(dataset)

    #     # Assert
    #     assert processing.method == method

    # def test_embedding_from_dataset(self):
    #     # Arrange
    #     medium = SpecimenEmbeddingMediaCode("Paraffin wax")
    #     dataset = SpecimenPreparationStep(
    #         "identifier",
    #         SpecimenProcessing(
    #             description="Embedding",
    #         ),
    #         embedding_medium=medium.code,
    #     )

    #     # Act
    #     embedding = EmbeddingDicom.from_dataset(dataset)

    #     # Assert
    #     assert embedding.medium == medium

    # def test_fixation_from_dataset(self):
    #     # Arrange
    #     fixative = SpecimenFixativesCode("Neutral Buffered Formalin")
    #     dataset = SpecimenPreparationStep(
    #         "identifier",
    #         SpecimenProcessing(
    #             description="Fixation",
    #         ),
    #         fixative=fixative.code,
    #     )

    #     # Act
    #     fixation = FixationDicom.from_dataset(dataset)

    #     # Assert
    #     assert fixation.fixative == fixative

    # @pytest.mark.parametrize(
    #     "stains",
    #     [
    #         ["stain"],
    #         ["stain 1", "stain 2"],
    #         [SpecimenStainsCode("hematoxylin stain").code],
    #         [
    #             SpecimenStainsCode("hematoxylin stain").code,
    #             SpecimenStainsCode("water soluble eosin stain").code,
    #         ],
    #     ],
    # )
    # def test_staining_from_dataset(self, stains: List[Union[str, Code]]):
    #     # Arrange
    #     dataset = SpecimenPreparationStep("identifier", SpecimenStaining(stains))

    #     # Act
    #     staining = StainingDicomModel.from_dataset(dataset)

    #     # Assert
    #     assert staining.substances == stains

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
    @pytest.mark.parametrize("embedding", [SpecimenEmbeddingMediaCode("Paraffin wax")])
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
        embedding: SpecimenEmbeddingMediaCode,
        block_sampling_method: SpecimenSamplingProcedureCode,
        block_type: AnatomicPathologySpecimenTypesCode,
        position: SpecimenLocalization,
        primary_anatomic_structures: Sequence[Code],
        stains: Sequence[SpecimenStainsCode],
    ):
        # Arrange
        descriptions: List[Dataset] = []
        dataset = Dataset()
        for slide_sample_id, slide_sample_uid, specimen_id in zip(
            slide_sample_ids, slide_sample_uids, specimen_ids
        ):
            description = create_description(
                slide_sample_id,
                slide_sample_uid,
                specimen_id,
                block_id,
                specimen_type,
                collection_method,
                fixative,
                specimen_sampling_method,
                embedding,
                block_sampling_method,
                block_type,
                position,
                primary_anatomic_structures,
                stains,
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
        assert len(slide_samples) == len(slide_sample_ids)
        for slide_sample_index, slide_sample in enumerate(slide_samples):
            assert isinstance(slide_sample, SlideSample)
            assert slide_sample.identifier == slide_sample_ids[slide_sample_index]
            assert slide_sample.uid == slide_sample_uids[slide_sample_index]
            # assert slide_sample.anatomical_sites == primary_anatomic_structures
            # assert slide_sample.position == position
            assert slide_sample.sampled_from is not None
            assert slide_sample.sampled_from.method == block_sampling_method
            block = slide_sample.sampled_from.specimen
            assert isinstance(block, Sample)
            assert block.identifier == block_id
            assert block.type == block_type
            embedding_step = block.steps[0]
            assert isinstance(embedding_step, Embedding)
            assert embedding_step.medium == embedding
            assert len(block.sampled_from) == len(specimen_ids)
            for index, specimen_id in enumerate(specimen_ids):
                assert block.sampled_from[index].method == specimen_sampling_method
                specimen = block.sampled_from[index].specimen
                assert isinstance(specimen, ExtractedSpecimen)
                assert specimen.identifier == specimen_id
                assert specimen.type == specimen_type
                fixation_step = specimen.steps[1]
                assert isinstance(fixation_step, Fixation)
                assert fixation_step.fixative == fixative
                collection_step = specimen.steps[0]
                assert isinstance(collection_step, Collection)
                assert collection_step.method == collection_method

    pass


class TestPreparationStepDicomSchema:
    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    def test_serialize_collection_dicom(
        self,
        collection_dicom: CollectionDicomModel,
        identifier: Union[str, SpecimenIdentifier],
    ):
        # Arrange
        schema = CollectionDicomSchema()

        # Act
        serialized = schema.dump(collection_dicom)

        # Assert
        assert isinstance(serialized, list)
        # First item should be identifier
        item_iterator = iter(serialized)
        identifier_item = next(item_iterator)
        self.assert_item_name_equals_code(identifier_item, SampleCodes.identifier)
        self.assert_item_string_equals_value(
            identifier_item,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            issuer_of_identifier_item = next(item_iterator)
            self.assert_item_name_equals_code(
                issuer_of_identifier_item, SampleCodes.issuer_of_identifier
            )
            self.assert_item_string_equals_value(
                issuer_of_identifier_item, identifier.issuer.to_hl7v2()
            )
        # Next item should be processing type
        processing_type_item = next(item_iterator)
        self.assert_item_name_equals_code(
            processing_type_item, SampleCodes.processing_type
        )
        self.assert_item_code_equals_value(
            processing_type_item, SampleCodes.specimen_collection
        )
        # Next item can be date time
        if collection_dicom.date_time is not None:
            date_time_item = next(item_iterator)
            self.assert_item_name_equals_code(
                date_time_item, SampleCodes.datetime_of_processing
            )
            self.assert_item_datetime_equals_value(
                date_time_item, collection_dicom.date_time
            )
        # Next item can be description
        if collection_dicom.description is not None:
            collection_item = next(item_iterator)
            self.assert_item_name_equals_code(
                collection_item, SampleCodes.processing_description
            )
            self.assert_item_string_equals_value(
                collection_item, collection_dicom.description
            )
        # Last item should be method
        method_item = next(item_iterator)
        self.assert_item_name_equals_code(method_item, SampleCodes.specimen_collection)
        self.assert_item_code_equals_value(method_item, collection_dicom.method.code)
        # There should be no more items
        assert next(item_iterator, None) is None

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    @pytest.mark.parametrize(
        ["fixative", "embedding"],
        [
            [None, None],
            [
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                SpecimenEmbeddingMediaCode("Paraffin wax"),
            ],
        ],
    )
    def test_deserialize_collection_dicom(
        self,
        collection_dicom: CollectionDicomModel,
        collection_dataset: Dataset,
    ):
        # Arrange
        schema = CollectionDicomSchema()

        # Act
        deserialized = schema.load(
            collection_dataset.SpecimenPreparationStepContentItemSequence
        )

        # Assert
        assert isinstance(deserialized, CollectionDicomModel)
        assert deserialized.identifier == collection_dicom.identifier
        assert (
            deserialized.issuer_of_identifier == collection_dicom.issuer_of_identifier
        )
        assert deserialized.method == collection_dicom.method
        assert deserialized.date_time == collection_dicom.date_time
        assert deserialized.description == collection_dicom.description
        assert deserialized.fixative == collection_dicom.fixative
        assert deserialized.embedding == collection_dicom.embedding

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    @pytest.mark.parametrize(
        "location",
        [
            None,
            SpecimenLocalization(
                "reference",
                "description",
                Measurement(1, UnitCode("cm")),
                Measurement(2, UnitCode("cm")),
                Measurement(3, UnitCode("cm")),
            ),
        ],
    )
    def test_serialize_sampling_dicom(
        self,
        sampling_dicom: SamplingDicomModel,
        identifier: Union[str, SpecimenIdentifier],
    ):
        # Arrange
        schema = SamplingDicomSchema()

        # Act
        serialized = schema.dump(sampling_dicom)

        # Assert
        assert isinstance(serialized, list)
        # First item should be identifier
        item_iterator = iter(serialized)
        identifier_item = next(item_iterator)
        self.assert_item_name_equals_code(identifier_item, SampleCodes.identifier)
        self.assert_item_string_equals_value(
            identifier_item,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            issuer_of_identifier_item = next(item_iterator)
            self.assert_item_name_equals_code(
                issuer_of_identifier_item, SampleCodes.issuer_of_identifier
            )
            self.assert_item_string_equals_value(
                issuer_of_identifier_item, identifier.issuer.to_hl7v2()
            )
        # Next item should be processing type
        processing_type_item = next(item_iterator)
        self.assert_item_name_equals_code(
            processing_type_item, SampleCodes.processing_type
        )
        self.assert_item_code_equals_value(
            processing_type_item, SampleCodes.sampling_method
        )
        # Next item can be date time
        if sampling_dicom.date_time is not None:
            date_time_item = next(item_iterator)
            self.assert_item_name_equals_code(
                date_time_item, SampleCodes.datetime_of_processing
            )
            self.assert_item_datetime_equals_value(
                date_time_item, sampling_dicom.date_time
            )
        # Next item can be description
        if sampling_dicom.description is not None:
            collection_item = next(item_iterator)
            self.assert_item_name_equals_code(
                collection_item, SampleCodes.processing_description
            )
            self.assert_item_string_equals_value(
                collection_item, sampling_dicom.description
            )
        # Next item should be method
        method_item = next(item_iterator)
        self.assert_item_name_equals_code(method_item, SampleCodes.sampling_method)
        self.assert_item_code_equals_value(method_item, sampling_dicom.method.code)
        # Next item should be parent specimen identifier
        parent_specimen_identifier_item = next(item_iterator)
        self.assert_item_name_equals_code(
            parent_specimen_identifier_item, SampleCodes.parent_specimen_identifier
        )
        self.assert_item_string_equals_value(
            parent_specimen_identifier_item, sampling_dicom.parent_specimen_identifier
        )
        # Next item can be parent specimen identifier issuer
        if sampling_dicom.issuer_of_parent_specimen_identifier is not None:
            parent_specimen_identifier_issuer_item = next(item_iterator)
            self.assert_item_name_equals_code(
                parent_specimen_identifier_issuer_item,
                SampleCodes.issuer_of_parent_specimen_identifier,
            )
            self.assert_item_string_equals_value(
                parent_specimen_identifier_issuer_item,
                sampling_dicom.issuer_of_parent_specimen_identifier,
            )
        # Next item should be parent specimen type
        parent_specimen_type_item = next(item_iterator)
        self.assert_item_name_equals_code(
            parent_specimen_type_item, SampleCodes.parent_specimen_type
        )
        self.assert_item_code_equals_value(
            parent_specimen_type_item,
            sampling_dicom.parent_specimen_type,
        )
        # Next item can be location reference
        if sampling_dicom.location_reference is not None:
            location_reference_item = next(item_iterator)
            self.assert_item_name_equals_code(
                location_reference_item, SampleCodes.location_frame_of_reference
            )
            self.assert_item_string_equals_value(
                location_reference_item, sampling_dicom.location_reference
            )
        # Next item can be location description
        if sampling_dicom.location_description is not None:
            location_description_item = next(item_iterator)
            self.assert_item_name_equals_code(
                location_description_item, SampleCodes.location_of_sampling_site
            )
            self.assert_item_string_equals_value(
                location_description_item, sampling_dicom.location_description
            )
        # Next item can be location x
        if sampling_dicom.location_x is not None:
            location_x_item = next(item_iterator)
            self.assert_item_name_equals_code(
                location_x_item, SampleCodes.location_of_sampling_site_x
            )
            self.assert_item_measurement_equals_value(
                location_x_item, sampling_dicom.location_x
            )
        # Next item can be location y
        if sampling_dicom.location_y is not None:
            location_y_item = next(item_iterator)
            self.assert_item_name_equals_code(
                location_y_item, SampleCodes.location_of_sampling_site_y
            )
            self.assert_item_measurement_equals_value(
                location_y_item, sampling_dicom.location_y
            )
        # Next item can be location z
        if sampling_dicom.location_z is not None:
            location_z_item = next(item_iterator)
            self.assert_item_name_equals_code(
                location_z_item, SampleCodes.location_of_sampling_site_z
            )
            self.assert_item_measurement_equals_value(
                location_z_item, sampling_dicom.location_z
            )
        # There should be no more items
        assert next(item_iterator, None) is None

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    @pytest.mark.parametrize(
        ["fixative", "embedding"],
        [
            [None, None],
            [
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                SpecimenEmbeddingMediaCode("Paraffin wax"),
            ],
        ],
    )
    @pytest.mark.parametrize(
        "location",
        [
            None,
            SpecimenLocalization(
                "reference",
                "description",
                Measurement(1, UnitCode("cm")),
                Measurement(2, UnitCode("cm")),
                Measurement(3, UnitCode("cm")),
            ),
        ],
    )
    def test_deserialize_sampling_dicom(
        self, sampling_dicom: SamplingDicomModel, sampling_dataset: Dataset
    ):
        # Arrange
        schema = SamplingDicomSchema()

        # Act
        deserialized = schema.load(
            sampling_dataset.SpecimenPreparationStepContentItemSequence
        )

        # Assert
        assert isinstance(deserialized, SamplingDicomModel)
        assert deserialized.identifier == sampling_dicom.identifier
        assert deserialized.issuer_of_identifier == sampling_dicom.issuer_of_identifier
        assert deserialized.method == sampling_dicom.method
        assert deserialized.date_time == sampling_dicom.date_time
        assert deserialized.description == sampling_dicom.description
        assert deserialized.fixative == sampling_dicom.fixative
        assert deserialized.embedding == sampling_dicom.embedding
        assert deserialized.location_reference == sampling_dicom.location_reference
        assert deserialized.location_description == sampling_dicom.location_description
        assert deserialized.location_x == sampling_dicom.location_x
        assert deserialized.location_y == sampling_dicom.location_y
        assert deserialized.location_z == sampling_dicom.location_z

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    def test_serialize_processing_dicom(
        self,
        processing_dicom: ProcessingDicomModel,
        identifier: Union[str, SpecimenIdentifier],
    ):
        # Arrange
        schema = ProcessingDicomSchema()

        # Act
        serialized = schema.dump(processing_dicom)

        # Assert
        assert isinstance(serialized, list)
        # First item should be identifier
        item_iterator = iter(serialized)
        identifier_item = next(item_iterator)
        self.assert_item_name_equals_code(identifier_item, SampleCodes.identifier)
        self.assert_item_string_equals_value(
            identifier_item,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            issuer_of_identifier_item = next(item_iterator)
            self.assert_item_name_equals_code(
                issuer_of_identifier_item, SampleCodes.issuer_of_identifier
            )
            self.assert_item_string_equals_value(
                issuer_of_identifier_item, identifier.issuer.to_hl7v2()
            )
        # Next item should be processing type
        processing_type_item = next(item_iterator)
        self.assert_item_name_equals_code(
            processing_type_item, SampleCodes.processing_type
        )
        self.assert_item_code_equals_value(
            processing_type_item, SampleCodes.sample_processing
        )
        # Next item can be date time
        if processing_dicom.date_time is not None:
            date_time_item = next(item_iterator)
            self.assert_item_name_equals_code(
                date_time_item, SampleCodes.datetime_of_processing
            )
            self.assert_item_datetime_equals_value(
                date_time_item, processing_dicom.date_time
            )
        # Next item can be description
        if processing_dicom.description is not None:
            collection_item = next(item_iterator)
            self.assert_item_name_equals_code(
                collection_item, SampleCodes.processing_description
            )
            self.assert_item_string_equals_value(
                collection_item, processing_dicom.description
            )
        # Next item can be method
        if processing_dicom.processing is not None:
            collection_item = next(item_iterator)
            self.assert_item_name_equals_code(
                collection_item, SampleCodes.processing_description
            )
            self.assert_item_code_equals_value(
                collection_item, processing_dicom.processing
            )
        # There should be no more items
        assert next(item_iterator, None) is None

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    @pytest.mark.parametrize(
        ["fixative", "embedding"],
        [
            [None, None],
            [
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                SpecimenEmbeddingMediaCode("Paraffin wax"),
            ],
        ],
    )
    def test_deserialize_processing_dicom(
        self,
        processing_dicom: ProcessingDicomModel,
        processing_dataset: Dataset,
    ):
        # Arrange

        schema = ProcessingDicomSchema()

        # Act
        deserialized = schema.load(
            processing_dataset.SpecimenPreparationStepContentItemSequence
        )

        # Assert
        assert isinstance(deserialized, ProcessingDicomModel)
        assert deserialized.identifier == processing_dicom.identifier
        assert (
            deserialized.issuer_of_identifier == processing_dicom.issuer_of_identifier
        )
        assert deserialized.processing == processing_dicom.processing
        assert deserialized.date_time == processing_dicom.date_time
        assert deserialized.description == processing_dicom.description
        assert deserialized.fixative == processing_dicom.fixative
        assert deserialized.embedding == processing_dicom.embedding

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    def test_serialize_staining_dicom(
        self,
        staining_dicom: StainingDicomModel,
        identifier: Union[str, SpecimenIdentifier],
    ):
        # Arrange
        schema = StainingDicomSchema()

        # Act
        serialized = schema.dump(staining_dicom)

        # Assert
        assert isinstance(serialized, list)
        # First item should be identifier
        item_iterator = iter(serialized)
        identifier_item = next(item_iterator)
        self.assert_item_name_equals_code(identifier_item, SampleCodes.identifier)
        self.assert_item_string_equals_value(
            identifier_item,
            identifier if isinstance(identifier, str) else identifier.value,
        )
        # Next item can be issuer of identifier
        if isinstance(identifier, SpecimenIdentifier) and identifier.issuer is not None:
            issuer_of_identifier_item = next(item_iterator)
            self.assert_item_name_equals_code(
                issuer_of_identifier_item, SampleCodes.issuer_of_identifier
            )
            self.assert_item_string_equals_value(
                issuer_of_identifier_item, identifier.issuer.to_hl7v2()
            )
        # Next item should be processing type
        processing_type_item = next(item_iterator)
        self.assert_item_name_equals_code(
            processing_type_item, SampleCodes.processing_type
        )
        self.assert_item_code_equals_value(processing_type_item, SampleCodes.staining)
        # Next item can be date time
        if staining_dicom.date_time is not None:
            date_time_item = next(item_iterator)
            self.assert_item_name_equals_code(
                date_time_item, SampleCodes.datetime_of_processing
            )
            self.assert_item_datetime_equals_value(
                date_time_item, staining_dicom.date_time
            )
        # Next item can be description
        if staining_dicom.description is not None:
            collection_item = next(item_iterator)
            self.assert_item_name_equals_code(
                collection_item, SampleCodes.processing_description
            )
            self.assert_item_string_equals_value(
                collection_item, staining_dicom.description
            )
        # Next item can be method
        for substance in staining_dicom.substances:
            substance_item = next(item_iterator)
            self.assert_item_name_equals_code(
                substance_item,
                SampleCodes.using_substance,
            )
            if isinstance(substance, SpecimenStainsCode):
                self.assert_item_code_equals_value(substance_item, substance)
            else:
                self.assert_item_string_equals_value(substance_item, substance)

        # There should be no more items
        assert next(item_iterator, None) is None

    @pytest.mark.parametrize(
        ["date_time", "description"],
        [
            [datetime.datetime(2021, 1, 1), "description"],
            [None, None],
        ],
    )
    @pytest.mark.parametrize(
        "identifier",
        [
            "identifier",
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
        ],
    )
    @pytest.mark.parametrize(
        ["fixative", "embedding"],
        [
            [None, None],
            [
                SpecimenFixativesCode("Neutral Buffered Formalin"),
                SpecimenEmbeddingMediaCode("Paraffin wax"),
            ],
        ],
    )
    def test_deserialize_staining_dicom(
        self,
        staining_dicom: StainingDicomModel,
        staining_dataset: Dataset,
    ):
        # Arrange

        schema = StainingDicomSchema()

        # Act
        deserialized = schema.load(
            staining_dataset.SpecimenPreparationStepContentItemSequence
        )

        # Assert
        assert isinstance(deserialized, StainingDicomModel)
        assert deserialized.identifier == staining_dicom.identifier
        assert deserialized.issuer_of_identifier == staining_dicom.issuer_of_identifier
        assert deserialized.substances == staining_dicom.substances
        assert deserialized.date_time == staining_dicom.date_time
        assert deserialized.description == staining_dicom.description
        assert deserialized.fixative == staining_dicom.fixative
        assert deserialized.embedding == staining_dicom.embedding

    @pytest.mark.parametrize("reference", ["reference", None])
    @pytest.mark.parametrize("description", ["description", None])
    @pytest.mark.parametrize("x", [Measurement(1, UnitCode("mm")), None])
    @pytest.mark.parametrize("y", [Measurement(1, UnitCode("mm")), None])
    @pytest.mark.parametrize("z", [Measurement(1, UnitCode("mm")), None])
    @pytest.mark.parametrize("visual_marking", ["visual_marking", None])
    def test_serialize_specimen_location_dicom(
        self,
        reference: Optional[str],
        description: Optional[str],
        x: Optional[Measurement],
        y: Optional[Measurement],
        z: Optional[Measurement],
        visual_marking: Optional[str],
    ):
        # Arrange
        location = SpecimenLocalization(
            reference=reference,
            description=description,
            x=x,
            y=y,
            z=z,
            visual_marking=visual_marking,
        )
        schema = SpecimenLocalizationDicomSchema()

        # Act
        serialized = schema.dump(location)

        # Assert
        assert isinstance(serialized, list)
        item_iterator = iter(serialized)
        for string, name in [
            (reference, SampleCodes.location_frame_of_reference),
            (description, SampleCodes.location_of_specimen),
        ]:
            if string is not None:
                string_item = next(item_iterator)
                self.assert_item_name_equals_code(string_item, name)
                self.assert_item_string_equals_value(string_item, string)
        for measurement, name in [
            (x, SampleCodes.location_of_specimen_x),
            (y, SampleCodes.location_of_specimen_y),
            (z, SampleCodes.location_of_specimen_z),
        ]:
            if measurement is not None:
                measurement_item = next(item_iterator)
                self.assert_item_name_equals_code(measurement_item, name)
                self.assert_item_measurement_equals_value(measurement_item, measurement)

        if visual_marking is not None:
            visual_marking_item = next(item_iterator)
            self.assert_item_name_equals_code(
                visual_marking_item, SampleCodes.visual_marking_of_specimen
            )
            self.assert_item_string_equals_value(visual_marking_item, visual_marking)

    def assert_item_name_equals_code(self, item: Dataset, name: Code):
        self.assert_code_dataset_equals_code(item.ConceptNameCodeSequence[0], name)

    def assert_item_string_equals_value(self, item: Dataset, value: str):
        assert item.TextValue == value

    def assert_item_datetime_equals_value(
        self, item: Dataset, value: datetime.datetime
    ):
        assert item.DateTime == value

    def assert_item_code_equals_value(
        self, item: Dataset, value: Union[Code, ConceptCode]
    ):
        self.assert_code_dataset_equals_code(item.ConceptCodeSequence[0], value)

    def assert_item_measurement_equals_value(self, item: Dataset, value: Measurement):
        assert item.FloatingPointValue == value.value
        self.assert_code_dataset_equals_code(
            item.MeasurementUnitsCodeSequence[0],
            value.unit,
        )

    def assert_code_dataset_equals_code(
        self, item: Dataset, code: Union[Code, ConceptCode]
    ):
        assert item.CodeValue == code.value, (item.CodeMeaning, code.meaning)
        assert item.CodingSchemeDesignator == code.scheme_designator, (
            item.CodeMeaning,
            code.meaning,
        )
