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
from typing import Any, Dict

import pytest
from pydicom.sr.coding import Code
from pydicom.uid import UID

from tests.metadata.json_schema.helpers import assert_dict_equals_code
from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    SpecimenCollectionProcedureCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenPreparationStepsCode,
    SpecimenSamplingProcedureCode,
    UnitCode,
)
from wsidicom.metadata.json_schema.sample.model import (
    ExtractedSpecimenJsonModel,
    SampleJsonModel,
    SlideSampleJsonModel,
)
from wsidicom.metadata.json_schema.sample.schema import (
    ExtractedSpecimenJsonSchema,
    PreparationStepJsonSchema,
    SampleJsonSchema,
    SamplingConstraintJsonSchema,
    SamplingJsonModel,
    SamplingConstraintJsonModel,
    SlideSampleJsonSchema,
    SpecimenJsonSchema,
    SpecimenLocalizationJsonSchema,
)
from wsidicom.metadata.sample import (
    Collection,
    Embedding,
    ExtractedSpecimen,
    Fixation,
    Measurement,
    Processing,
    Sample,
    SamplingLocation,
    SlideSample,
    SpecimenLocalization,
)


class TestSampleJsonSchema:
    @pytest.mark.parametrize(
        ["localization", "expected"],
        [
            (
                SpecimenLocalization(
                    "reference",
                    "description",
                    Measurement(1, UnitCode("mm")),
                    Measurement(2, UnitCode("mm")),
                    Measurement(3, UnitCode("mm")),
                    "marking",
                ),
                {
                    "reference": "reference",
                    "description": "description",
                    "x": {
                        "value": 1,
                        "unit": "mm",
                    },
                    "y": {
                        "value": 2,
                        "unit": "mm",
                    },
                    "z": {
                        "value": 3,
                        "unit": "mm",
                    },
                    "visual_marking": "marking",
                },
            ),
            (
                SpecimenLocalization(),
                {
                    "reference": None,
                    "description": None,
                    "x": None,
                    "y": None,
                    "z": None,
                    "visual_marking": None,
                },
            ),
        ],
    )
    def test_specimen_localization_serialize(
        self, localization: SpecimenLocalization, expected: Dict[str, Any]
    ):
        # Arrange

        # Act
        dumped = SpecimenLocalizationJsonSchema().dump(localization)

        # Assert
        assert isinstance(dumped, dict)
        assert dumped == expected

    @pytest.mark.parametrize(
        ["localization", "expected"],
        [
            (
                {
                    "reference": "slide",
                    "description": "left",
                    "x": {
                        "value": 1,
                        "unit": "mm",
                    },
                    "y": {
                        "value": 2,
                        "unit": "mm",
                    },
                    "z": {
                        "value": 3,
                        "unit": "mm",
                    },
                    "visual_marking": "marking",
                },
                SpecimenLocalization(
                    "slide",
                    "left",
                    Measurement(1, UnitCode("mm")),
                    Measurement(2, UnitCode("mm")),
                    Measurement(3, UnitCode("mm")),
                    "marking",
                ),
            ),
            ({}, SpecimenLocalization()),
        ],
    )
    def test_specimen_localization_deserialize(
        self, localization: Dict[str, Any], expected: SpecimenLocalization
    ):
        # Arrange

        # Act
        loaded = SpecimenLocalizationJsonSchema().load(localization)

        # Assert
        assert isinstance(loaded, SpecimenLocalization)
        assert loaded == expected

    def test_sampling_constraint_serialize(self, extracted_specimen: ExtractedSpecimen):
        # Arrange
        sampling_chain_constraint = extracted_specimen.sample(
            SpecimenSamplingProcedureCode("Dissection")
        )

        # Act
        dumped = SamplingConstraintJsonSchema().dump(sampling_chain_constraint)

        # Arrange
        assert isinstance(dumped, dict)
        assert dumped["identifier"] == sampling_chain_constraint.specimen.identifier
        assert dumped["sampling_step_index"] == 0

    def test_sampling_constraint_deserialize(self):
        # Arrange
        dumped = {"identifier": "specimen", "sampling_step_index": 1}

        # Act
        loaded = SamplingConstraintJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, SamplingConstraintJsonModel)
        assert loaded.identifier == dumped["identifier"]
        assert loaded.sampling_step_index == dumped["sampling_step_index"]

    def test_sampling_serialize(self):
        # Arrange
        specimen = ExtractedSpecimen(
            "specimen",
            AnatomicPathologySpecimenTypesCode("Gross specimen"),
        )
        sampling_1 = specimen.sample(
            SpecimenSamplingProcedureCode("Dissection"),
            datetime.datetime(2023, 8, 5),
            "description",
        )
        sample = Sample(
            "sample",
            AnatomicPathologySpecimenTypesCode("Tissue section"),
            [sampling_1],
            [],
        )
        sampling_2 = sample.sample(
            SpecimenSamplingProcedureCode("Block sectioning"),
            datetime.datetime(2023, 8, 5),
            "description",
            [sampling_1],
            SamplingLocation(
                "reference",
                "description",
                Measurement(1, UnitCode("mm")),
                Measurement(2, UnitCode("mm")),
                Measurement(3, UnitCode("mm")),
            ),
        )
        assert sampling_2.date_time is not None
        assert sampling_2.location is not None
        assert sampling_2.location.x is not None
        assert sampling_2.location.y is not None
        assert sampling_2.location.z is not None

        # Act
        dumped = PreparationStepJsonSchema().dump(sampling_2)

        # Assert
        assert isinstance(dumped, dict)
        assert_dict_equals_code(dumped["sampling_method"], sampling_2.method)
        assert dumped["date_time"] == sampling_2.date_time.isoformat()
        assert dumped["description"] == sampling_2.description
        assert (
            dumped["sampling_chain_constraints"][0]["identifier"] == specimen.identifier
        )
        assert dumped["sampling_chain_constraints"][0]["sampling_step_index"] == 0
        assert isinstance(dumped["location"], dict)
        assert dumped["location"]["reference"] == sampling_2.location.reference
        assert dumped["location"]["description"] == sampling_2.location.description
        assert dumped["location"]["x"]["value"] == sampling_2.location.x.value
        assert dumped["location"]["x"]["unit"] == sampling_2.location.x.unit.value
        assert dumped["location"]["y"]["value"] == sampling_2.location.y.value
        assert dumped["location"]["y"]["unit"] == sampling_2.location.y.unit.value
        assert dumped["location"]["z"]["value"] == sampling_2.location.z.value
        assert dumped["location"]["z"]["unit"] == sampling_2.location.z.unit.value

    def test_sampling_deserialize(self):
        # Arrange
        dumped = {
            "sampling_method": {
                "value": "434472006",
                "scheme_designator": "SCT",
                "meaning": "Block sectioning",
            },
            "sampling_chain_constraints": [
                {"identifier": "specimen", "sampling_step_index": 0}
            ],
            "date_time": "2023-08-05T00:00:00",
            "description": "description",
            "location": {
                "reference": "reference",
                "description": "description",
                "x": {
                    "value": 1,
                    "unit": "mm",
                },
                "y": {
                    "value": 2,
                    "unit": "mm",
                },
                "z": {
                    "value": 3,
                    "unit": "mm",
                },
            },
        }

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, SamplingJsonModel)
        assert loaded.sampling_chain_constraints is not None
        assert_dict_equals_code(dumped["sampling_method"], loaded.method)
        assert loaded.date_time == datetime.datetime.fromisoformat(dumped["date_time"])
        assert loaded.description == dumped["description"]
        assert (
            loaded.sampling_chain_constraints[0].identifier
            == dumped["sampling_chain_constraints"][0]["identifier"]
        )
        assert (
            loaded.sampling_chain_constraints[0].sampling_step_index
            == dumped["sampling_chain_constraints"][0]["sampling_step_index"]
        )
        assert isinstance(loaded.location, SamplingLocation)
        assert loaded.location.reference == dumped["location"]["reference"]
        assert loaded.location.description == dumped["location"]["description"]
        assert loaded.location.x == Measurement(
            dumped["location"]["x"]["value"],
            UnitCode(dumped["location"]["x"]["unit"]),
        )
        assert loaded.location.y == Measurement(
            dumped["location"]["y"]["value"],
            UnitCode(dumped["location"]["y"]["unit"]),
        )
        assert loaded.location.z == Measurement(
            dumped["location"]["z"]["value"],
            UnitCode(dumped["location"]["z"]["unit"]),
        )

    def test_collection_serialize(self):
        # Arrange
        collection = Collection(
            SpecimenCollectionProcedureCode("Excision"),
            datetime.datetime(2023, 8, 5),
            "description",
        )
        assert collection.date_time is not None

        # Act
        dumped = PreparationStepJsonSchema().dump(collection)

        # Assert
        assert isinstance(dumped, dict)
        assert_dict_equals_code(dumped["extraction_method"], collection.method)
        assert dumped["date_time"] == collection.date_time.isoformat()
        assert dumped["description"] == collection.description

    def test_collection_deserialize(self):
        # Arrange
        dumped = {
            "extraction_method": {
                "value": "65801008",
                "scheme_designator": "SCT",
                "meaning": "Excision",
            },
            "date_time": "2023-08-05T00:00:00",
            "description": "description",
        }

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Collection)
        assert_dict_equals_code(dumped["extraction_method"], loaded.method)
        assert loaded.date_time == datetime.datetime.fromisoformat(dumped["date_time"])
        assert loaded.description == dumped["description"]

    def test_processing_serialize(self):
        # Arrange
        processing = Processing(
            SpecimenPreparationStepsCode("Specimen clearing"),
            datetime.datetime(2023, 8, 5),
        )
        assert processing.date_time is not None

        # Act
        dumped = PreparationStepJsonSchema().dump(processing)

        # Assert
        assert isinstance(dumped, dict)
        assert_dict_equals_code(dumped["processing_method"], processing.method)
        assert dumped["date_time"] == processing.date_time.isoformat()

    def test_processing_deserialize(self):
        # Arrange
        dumped = {
            "processing_method": {
                "value": "433452008",
                "scheme_designator": "SCT",
                "meaning": "Specimen clearing",
            },
            "date_time": "2023-08-05T00:00:00",
        }

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Processing)
        assert_dict_equals_code(dumped["processing_method"], loaded.method)
        assert loaded.date_time == datetime.datetime.fromisoformat(dumped["date_time"])

    def test_embedding_serialize(self):
        # Arrange
        embedding = Embedding(
            SpecimenEmbeddingMediaCode("Paraffin wax"),
            datetime.datetime(2023, 8, 5),
        )
        assert embedding.date_time is not None

        # Act
        dumped = PreparationStepJsonSchema().dump(embedding)

        # Assert
        assert isinstance(dumped, dict)
        assert_dict_equals_code(dumped["medium"], embedding.medium)
        assert dumped["date_time"] == embedding.date_time.isoformat()

    def test_embedding_deserialize(self):
        # Arrange
        dumped = {
            "medium": {
                "value": "311731000",
                "scheme_designator": "SCT",
                "meaning": "Paraffin wax",
            },
            "date_time": "2023-08-05T00:00:00",
        }

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Embedding)
        assert_dict_equals_code(dumped["medium"], loaded.medium)
        assert loaded.date_time == datetime.datetime.fromisoformat(dumped["date_time"])

    def test_fixation_serialize(self):
        # Arrange
        fixation = Fixation(
            SpecimenFixativesCode("Neutral Buffered Formalin"),
            datetime.datetime(2023, 8, 5),
        )
        assert fixation.date_time is not None

        # Act
        dumped = PreparationStepJsonSchema().dump(fixation)

        # Assert
        assert isinstance(dumped, dict)
        assert_dict_equals_code(dumped["fixative"], fixation.fixative)
        assert dumped["date_time"] == fixation.date_time.isoformat()

    def test_fixation_deserialize(self):
        # Arrange
        dumped = {
            "fixative": {
                "value": "434162003",
                "scheme_designator": "SCT",
                "meaning": "Neutral Buffered Formalin",
            },
            "date_time": "2023-08-05T00:00:00",
        }

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Fixation)
        assert_dict_equals_code(dumped["fixative"], loaded.fixative)
        assert loaded.date_time == datetime.datetime.fromisoformat(dumped["date_time"])

    def test_extracted_specimen_serialize(self, extracted_specimen: ExtractedSpecimen):
        # Arrange
        assert extracted_specimen.extraction_step is not None
        assert extracted_specimen.extraction_step.date_time is not None

        # Act
        dumped = ExtractedSpecimenJsonSchema().dump(extracted_specimen)

        # Assert

        assert isinstance(dumped, dict)
        assert dumped["identifier"] == extracted_specimen.identifier
        assert_dict_equals_code(
            dumped["steps"][0]["extraction_method"],
            extracted_specimen.extraction_step.method,
        )
        assert (
            dumped["steps"][0]["date_time"]
            == extracted_specimen.extraction_step.date_time.isoformat()
        )
        assert_dict_equals_code(dumped["type"], extracted_specimen.type)

    def test_extracted_specimen_deserialize(self):
        # Arrange
        dumped = {
            "identifier": "specimen",
            "steps": [
                {
                    "extraction_method": {
                        "value": "65801008",
                        "scheme_designator": "SCT",
                        "meaning": "Excision",
                    },
                    "date_time": "2023-08-05T00:00:00",
                    "description": "description",
                }
            ],
            "type": {
                "value": "430861001",
                "scheme_designator": "SCT",
                "meaning": "Gross specimen",
            },
        }

        # Act
        loaded = ExtractedSpecimenJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, ExtractedSpecimenJsonModel)
        assert loaded.identifier == dumped["identifier"]
        collection = loaded.steps[0]
        assert isinstance(collection, Collection)
        assert_dict_equals_code(
            dumped["steps"][0]["extraction_method"], collection.method
        )
        assert collection.date_time == datetime.datetime.fromisoformat(
            dumped["steps"][0]["date_time"]
        )
        assert collection.description == dumped["steps"][0]["description"]
        assert isinstance(loaded.type, AnatomicPathologySpecimenTypesCode)
        assert_dict_equals_code(dumped["type"], loaded.type)

    def test_sample_serialize(self, sample: Sample):
        # Arrange
        processing = sample.steps[0]
        assert isinstance(processing, Processing)
        assert processing.date_time is not None

        # Act
        dumped = SampleJsonSchema().dump(sample)

        # Assert
        assert isinstance(dumped, dict)
        assert dumped["identifier"] == sample.identifier
        assert_dict_equals_code(
            dumped["steps"][0]["processing_method"], processing.method
        )
        assert dumped["steps"][0]["date_time"] == processing.date_time.isoformat()
        assert_dict_equals_code(dumped["type"], sample.type)
        assert (
            dumped["sampled_from"][0]["identifier"]
            == sample.sampled_from[0].specimen.identifier
        )
        assert dumped["sampled_from"][0]["sampling_step_index"] == 0

    def test_sampled_specimen_deserialize(self):
        # Arrange
        dumped = {
            "identifier": "sample",
            "steps": [
                {
                    "processing_method": {
                        "value": "433452008",
                        "scheme_designator": "SCT",
                        "meaning": "Specimen clearing",
                    },
                    "date_time": "2023-08-05T00:00:00",
                }
            ],
            "sampled_from": [{"identifier": "specimen", "sampling_step_index": 1}],
            "type": {
                "value": "430856003",
                "scheme_designator": "SCT",
                "meaning": "Tissue section",
            },
        }

        # Act
        loaded = SampleJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, SampleJsonModel)
        assert loaded.identifier == dumped["identifier"]
        processing = loaded.steps[0]
        assert isinstance(processing, Processing)
        assert_dict_equals_code(
            dumped["steps"][0]["processing_method"], processing.method
        )
        assert processing.date_time == datetime.datetime.fromisoformat(
            dumped["steps"][0]["date_time"]
        )
        assert isinstance(loaded.type, AnatomicPathologySpecimenTypesCode)
        assert_dict_equals_code(dumped["type"], loaded.type)

    def test_slide_sample_serialize(self, slide_sample: SlideSample):
        # Arrange
        assert slide_sample.sampled_from is not None
        assert slide_sample.anatomical_sites is not None
        assert slide_sample.localization is not None

        # Act
        dumped = SlideSampleJsonSchema().dump(slide_sample)

        # Assert
        assert isinstance(dumped, dict)
        assert dumped["identifier"] == slide_sample.identifier
        assert_dict_equals_code(
            dumped["anatomical_sites"][0], slide_sample.anatomical_sites[0]
        )
        assert (
            dumped["sampled_from"]["identifier"]
            == slide_sample.sampled_from.specimen.identifier
        )
        assert dumped["sampled_from"]["sampling_step_index"] == 0
        assert dumped["uid"] == str(slide_sample.uid)
        assert (
            dumped["localization"]["reference"] == slide_sample.localization.reference
        )
        assert (
            dumped["localization"]["description"]
            == slide_sample.localization.description
        )
        assert dumped["localization"]["x"] == slide_sample.localization.x
        assert dumped["localization"]["y"] == slide_sample.localization.y
        assert dumped["localization"]["z"] == slide_sample.localization.z
        assert (
            dumped["localization"]["visual_marking"]
            == slide_sample.localization.visual_marking
        )

    def test_slide_sample_deserialize(self):
        # Arrange
        dumped = {
            "identifier": "sample",
            "steps": [],
            "anatomical_sites": [
                {"value": "value", "scheme_designator": "scheme", "meaning": "meaning"}
            ],
            "sampled_from": {"identifier": "sample", "sampling_step_index": 1},
            "uid": "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445423",
            "localization": {"description": "left"},
        }

        # Act
        loaded = SlideSampleJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, SlideSampleJsonModel)
        assert loaded.identifier == dumped["identifier"]
        assert loaded.anatomical_sites is not None
        anatomical_site = loaded.anatomical_sites[0]
        assert isinstance(anatomical_site, Code)
        assert_dict_equals_code(dumped["anatomical_sites"][0], anatomical_site)
        assert loaded.uid == UID(dumped["uid"])
        assert loaded.localization is not None
        assert loaded.localization.description == dumped["localization"]["description"]

    def test_full_slide_sample_serialize(self, slide_sample: SlideSample):
        # Arrange
        assert slide_sample.sampled_from is not None
        sample = slide_sample.sampled_from.specimen
        assert isinstance(sample, Sample)
        specimen = sample.sampled_from[0].specimen

        # Act
        dumped = SpecimenJsonSchema().dump(slide_sample)

        # Assert
        assert isinstance(dumped, list)
        dumpled_slide_sample = dumped[0]
        dumpled_sample = dumped[1]
        dumpled_specimen = dumped[2]

        assert isinstance(dumpled_slide_sample, dict)
        assert dumpled_slide_sample["identifier"] == slide_sample.identifier
        assert (
            dumpled_slide_sample["sampled_from"]["identifier"]
            == slide_sample.sampled_from.specimen.identifier
        )
        assert dumpled_slide_sample["sampled_from"]["sampling_step_index"] == 0
        assert isinstance(dumpled_sample, dict)
        assert dumpled_sample["identifier"] == sample.identifier
        assert (
            dumpled_sample["sampled_from"][0]["identifier"]
            == sample.sampled_from[0].specimen.identifier
        )
        assert dumpled_sample["sampled_from"][0]["sampling_step_index"] == 0
        assert isinstance(dumpled_specimen, dict)
        assert dumpled_specimen["identifier"] == specimen.identifier

    def test_full_slide_sample_deserialize(self):
        dumped = [
            {
                "identifier": "slide sample",
                "steps": [],
                "anatomical_sites": [
                    {
                        "value": "value",
                        "scheme_designator": "scheme",
                        "meaning": "meaning",
                    }
                ],
                "sampled_from": {"identifier": "sample", "sampling_step_index": 0},
                "uid": "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445423",
                "localization": {"description": "left"},
            },
            {
                "identifier": "sample",
                "steps": [
                    {
                        "processing_method": {
                            "value": "433452008",
                            "scheme_designator": "SCT",
                            "meaning": "Specimen clearing",
                        },
                        "date_time": "2023-08-05T00:00:00",
                    },
                    {
                        "sampling_method": {
                            "value": "434472006",
                            "scheme_designator": "SCT",
                            "meaning": "Block sectioning",
                        },
                        "sampling_chain_constraints": None,
                        "date_time": "2023-08-05T00:00:00",
                        "description": "Sectioning to slide",
                    },
                ],
                "sampled_from": [{"identifier": "specimen", "sampling_step_index": 0}],
                "type": {
                    "value": "430856003",
                    "scheme_designator": "SCT",
                    "meaning": "Tissue section",
                },
            },
            {
                "identifier": "specimen",
                "steps": [
                    {
                        "extraction_method": {
                            "value": "65801008",
                            "scheme_designator": "SCT",
                            "meaning": "Excision",
                        },
                        "date_time": "2023-08-05T00:00:00",
                        "description": "description",
                    },
                    {
                        "sampling_method": {
                            "value": "122459003",
                            "scheme_designator": "SCT",
                            "meaning": "Dissection",
                        },
                        "sampling_chain_constraints": None,
                        "date_time": "2023-08-05T00:00:00",
                        "description": "Sampling to block",
                    },
                ],
                "type": {
                    "value": "430861001",
                    "scheme_designator": "SCT",
                    "meaning": "Gross specimen",
                },
            },
        ]

        # Act
        loaded = SpecimenJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, list)
        loaded_slide_sample = loaded[0]
        assert isinstance(loaded_slide_sample, SlideSample)
        assert loaded_slide_sample.identifier == dumped[0]["identifier"]
        assert loaded_slide_sample.sampled_from is not None
        sample = loaded_slide_sample.sampled_from.specimen
        assert isinstance(sample, Sample)
        assert sample.identifier == dumped[1]["identifier"]
        specimen = sample.sampled_from[0].specimen
        assert isinstance(specimen, ExtractedSpecimen)
        assert specimen.identifier == dumped[2]["identifier"]

    def test_slide_sample_rountrip(self, slide_sample: SlideSample):
        # Arrange

        # Act
        dumped = SpecimenJsonSchema().dump(slide_sample)
        print(dumped)
        loaded = SpecimenJsonSchema().load(dumped)
        print(slide_sample)
        print(loaded[0])

        # Assert
        assert str(loaded[0]) == str(slide_sample)
