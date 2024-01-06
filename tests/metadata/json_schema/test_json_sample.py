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
    SpecimenSamplingProcedureCode,
    UnitCode,
)
from wsidicom.metadata.schema.json.sample.model import (
    SpecimenJsonModel,
    SampleJsonModel,
    SlideSampleJsonModel,
)
from wsidicom.metadata.schema.json.sample.schema import (
    SpecimenJsonSchema,
    PreparationAction,
    PreparationStepJsonSchema,
    SampleJsonSchema,
    SamplingConstraintJsonModel,
    SamplingConstraintJsonSchema,
    SamplingJsonModel,
    SlideSampleJsonSchema,
    BaseSpecimenJsonSchema,
    SampleLocalizationJsonSchema,
)
from wsidicom.metadata.sample import (
    Collection,
    Embedding,
    Specimen,
    Fixation,
    Measurement,
    Processing,
    Receiving,
    Sample,
    Sampling,
    SamplingLocation,
    SlideSample,
    SampleLocalization,
    Storage,
    UnknownSampling,
)


class TestSampleJsonSchema:
    @pytest.mark.parametrize(
        ["localization", "expected"],
        [
            (
                SampleLocalization(
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
                SampleLocalization(),
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
    def test_sample_localization_serialize(
        self, localization: SampleLocalization, expected: Dict[str, Any]
    ):
        # Arrange

        # Act
        dumped = SampleLocalizationJsonSchema().dump(localization)

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
                SampleLocalization(
                    "slide",
                    "left",
                    Measurement(1, UnitCode("mm")),
                    Measurement(2, UnitCode("mm")),
                    Measurement(3, UnitCode("mm")),
                    "marking",
                ),
            ),
            ({}, SampleLocalization()),
        ],
    )
    def test_sample_localization_deserialize(
        self, localization: Dict[str, Any], expected: SampleLocalization
    ):
        # Arrange

        # Act
        loaded = SampleLocalizationJsonSchema().load(localization)

        # Assert
        assert isinstance(loaded, SampleLocalization)
        assert loaded == expected

    def test_sampling_constraint_serialize(self, extracted_specimen: Specimen):
        # Arrange
        sampling_constraint = extracted_specimen.sample(
            SpecimenSamplingProcedureCode("Dissection")
        )

        # Act
        dumped = SamplingConstraintJsonSchema().dump(sampling_constraint)

        # Arrange
        assert isinstance(dumped, dict)
        assert dumped["identifier"] == sampling_constraint.specimen.identifier
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
        specimen = Specimen(
            "specimen",
            Collection(SpecimenCollectionProcedureCode("Specimen collection")),
            AnatomicPathologySpecimenTypesCode("Gross specimen"),
        )
        sampling_1 = specimen.sample(
            SpecimenSamplingProcedureCode("Dissection"),
            datetime.datetime(2023, 8, 5),
            "description",
        )
        sample = Sample(
            "sample",
            [sampling_1],
            AnatomicPathologySpecimenTypesCode("Tissue section"),
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
        assert isinstance(sampling_2, Sampling)
        assert sampling_2.date_time is not None
        assert sampling_2.location is not None
        assert sampling_2.location.x is not None
        assert sampling_2.location.y is not None
        assert sampling_2.location.z is not None

        # Act
        dumped = PreparationStepJsonSchema().dump(sampling_2)

        # Assert
        assert isinstance(dumped, dict)
        assert_dict_equals_code(dumped["method"], sampling_2.method)
        assert dumped["date_time"] == sampling_2.date_time.isoformat()
        assert dumped["description"] == sampling_2.description
        assert dumped["sampling_constraints"][0]["identifier"] == specimen.identifier
        assert dumped["sampling_constraints"][0]["sampling_step_index"] == 0
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
            "action": "sampling",
            "method": {
                "value": "434472006",
                "scheme_designator": "SCT",
                "meaning": "Block sectioning",
            },
            "sampling_constraints": [
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
        assert loaded.sampling_constraints is not None
        if "method" in dumped:
            assert loaded.method is not None
            assert_dict_equals_code(dumped["method"], loaded.method)
        else:
            assert loaded.method is None
        assert loaded.date_time == datetime.datetime.fromisoformat(dumped["date_time"])
        assert loaded.description == dumped["description"]
        assert (
            loaded.sampling_constraints[0].identifier
            == dumped["sampling_constraints"][0]["identifier"]
        )
        assert (
            loaded.sampling_constraints[0].sampling_step_index
            == dumped["sampling_constraints"][0]["sampling_step_index"]
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

    def test_unkown_sampling_serialize(self):
        # Arrange
        specimen = Specimen(
            "specimen",
            Collection(SpecimenCollectionProcedureCode("Specimen collection")),
            AnatomicPathologySpecimenTypesCode("Gross specimen"),
        )
        sampling_1 = specimen.sample(
            SpecimenSamplingProcedureCode("Dissection"),
            datetime.datetime(2023, 8, 5),
            "description",
        )
        sample = Sample(
            "sample",
            [sampling_1],
            AnatomicPathologySpecimenTypesCode("Tissue section"),
            [],
        )
        sampling_2 = sample.sample(sampling_constraints=[sampling_1])
        assert isinstance(sampling_2, UnknownSampling)

        # Act
        dumped = PreparationStepJsonSchema().dump(sampling_2)

        # Assert
        assert isinstance(dumped, dict)
        assert "method" not in dumped
        assert "date_time" not in dumped
        assert "description" not in dumped
        assert dumped["sampling_constraints"][0]["identifier"] == specimen.identifier
        assert dumped["sampling_constraints"][0]["sampling_step_index"] == 0
        assert "location" not in dumped

    def test_unkown_sampling_deserialize(self):
        # Arrange
        dumped = {
            "action": "sampling",
            "sampling_constraints": [
                {"identifier": "specimen", "sampling_step_index": 0}
            ],
        }

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, SamplingJsonModel)
        assert loaded.sampling_constraints is not None
        assert loaded.method is None

        assert loaded.date_time is None
        assert loaded.description is None
        assert (
            loaded.sampling_constraints[0].identifier
            == dumped["sampling_constraints"][0]["identifier"]
        )
        assert (
            loaded.sampling_constraints[0].sampling_step_index
            == dumped["sampling_constraints"][0]["sampling_step_index"]
        )
        assert loaded.location is None

    def test_collection_serialize(self, collection: Collection):
        # Arrange

        # Act
        dumped = PreparationStepJsonSchema().dump(collection)

        # Assert
        assert isinstance(dumped, dict)
        assert_dict_equals_code(dumped["method"], collection.method)
        if collection.date_time is not None:
            assert dumped["date_time"] == collection.date_time.isoformat()
        else:
            assert "date_time" not in dumped
        if collection.description is not None:
            assert dumped["description"] == collection.description
        else:
            assert "description" not in dumped

    def test_collection_deserialize(self, collection: Collection):
        # Arrange
        dumped = {
            "action": "collection",
            "method": {
                "value": collection.method.value,
                "scheme_designator": collection.method.scheme_designator,
                "meaning": collection.method.meaning,
            },
        }
        if collection.date_time is not None:
            dumped["date_time"] = collection.date_time.isoformat()
        if collection.description is not None:
            dumped["description"] = collection.description

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Collection)
        assert loaded.method == collection.method
        assert loaded.date_time == collection.date_time
        assert loaded.description == collection.description

    def test_processing_serialize(self, processing: Processing):
        # Arrange

        # Act
        dumped = PreparationStepJsonSchema().dump(processing)

        # Assert
        assert isinstance(dumped, dict)
        if processing.method is not None:
            assert_dict_equals_code(dumped["method"], processing.method)
        else:
            assert "method" not in dumped
        if processing.date_time is not None:
            assert dumped["date_time"] == processing.date_time.isoformat()
        else:
            assert "date_time" not in dumped
        if processing.description is not None:
            assert dumped["description"] == processing.description
        else:
            assert "description" not in dumped

    def test_processing_deserialize(self, processing: Processing):
        # Arrange
        dumped: Dict[str, Any] = {
            "action": "processing",
        }
        if processing.method is not None:
            dumped["method"] = {
                "value": processing.method.value,
                "scheme_designator": processing.method.scheme_designator,
                "meaning": processing.method.meaning,
            }
        if processing.date_time is not None:
            dumped["date_time"] = processing.date_time.isoformat()
        if processing.description is not None:
            dumped["description"] = processing.description

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Processing)
        assert loaded.method == processing.method
        assert loaded.date_time == processing.date_time
        assert loaded.description == processing.description

    def test_embedding_serialize(self, embedding: Embedding):
        # Arrange

        # Act
        dumped = PreparationStepJsonSchema().dump(embedding)

        # Assert
        assert isinstance(dumped, dict)
        assert_dict_equals_code(dumped["medium"], embedding.medium)
        if embedding.date_time is not None:
            assert dumped["date_time"] == embedding.date_time.isoformat()
        else:
            assert "date_time" not in dumped
        if embedding.description is not None:
            assert dumped["description"] == embedding.description
        else:
            assert "description" not in dumped

    def test_embedding_deserialize(self, embedding: Embedding):
        # Arrange
        dumped = {
            "action": "embedding",
            "medium": {
                "value": embedding.medium.value,
                "scheme_designator": embedding.medium.scheme_designator,
                "meaning": embedding.medium.meaning,
            },
        }
        if embedding.date_time is not None:
            dumped["date_time"] = embedding.date_time.isoformat()
        if embedding.description is not None:
            dumped["description"] = embedding.description

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Embedding)
        assert loaded.medium == embedding.medium
        assert loaded.date_time == embedding.date_time
        assert loaded.description == embedding.description

    def test_fixation_serialize(self, fixation: Fixation):
        # Arrange

        # Act
        dumped = PreparationStepJsonSchema().dump(fixation)

        # Assert
        assert isinstance(dumped, dict)
        assert_dict_equals_code(dumped["fixative"], fixation.fixative)
        if fixation.date_time is not None:
            assert dumped["date_time"] == fixation.date_time.isoformat()
        else:
            assert "date_time" not in dumped
        if fixation.description is not None:
            assert dumped["description"] == fixation.description
        else:
            assert "description" not in dumped

    def test_fixation_deserialize(self, fixation: Fixation):
        # Arrange
        dumped = {
            "action": "fixation",
            "fixative": {
                "value": fixation.fixative.value,
                "scheme_designator": fixation.fixative.scheme_designator,
                "meaning": fixation.fixative.meaning,
            },
        }
        if fixation.date_time is not None:
            dumped["date_time"] = fixation.date_time.isoformat()
        if fixation.description is not None:
            dumped["description"] = fixation.description

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Fixation)
        assert loaded.fixative == fixation.fixative
        assert loaded.date_time == fixation.date_time
        assert loaded.description == fixation.description

    def test_receiving_serialize(self, receiving: Receiving):
        # Arrange

        # Act
        dumped = PreparationStepJsonSchema().dump(receiving)

        # Assert
        assert isinstance(dumped, dict)
        assert dumped["action"] == PreparationAction.RECEIVING.value
        if receiving.date_time is not None:
            assert dumped["date_time"] == receiving.date_time.isoformat()
        else:
            assert "date_time" not in dumped
        if receiving.description is not None:
            assert dumped["description"] == receiving.description
        else:
            assert "description" not in dumped

    def test_receiving_deserialize(self, receiving: Receiving):
        # Arrange
        dumped = {"action": PreparationAction.RECEIVING.value}
        if receiving.date_time is not None:
            dumped["date_time"] = receiving.date_time.isoformat()
        if receiving.description is not None:
            dumped["description"] = receiving.description

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Receiving)
        assert loaded.date_time == receiving.date_time
        assert loaded.description == receiving.description

    def test_storage_serialize(self, storage: Storage):
        # Arrange

        # Act
        dumped = PreparationStepJsonSchema().dump(storage)

        # Assert
        assert isinstance(dumped, dict)
        assert dumped["action"] == PreparationAction.STORAGE.value
        if storage.date_time is not None:
            assert dumped["date_time"] == storage.date_time.isoformat()
        else:
            assert "date_time" not in dumped
        if storage.description is not None:
            assert dumped["description"] == storage.description
        else:
            assert "description" not in dumped

    def test_storage_deserialize(self, storage: Storage):
        # Arrange
        dumped = {"action": PreparationAction.STORAGE.value}
        if storage.date_time is not None:
            dumped["date_time"] = storage.date_time.isoformat()
        if storage.description is not None:
            dumped["description"] = storage.description

        # Act
        loaded = PreparationStepJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Storage)
        assert loaded.date_time == storage.date_time
        assert loaded.description == storage.description

    def test_extracted_specimen_serialize(self, extracted_specimen: Specimen):
        # Arrange

        # Act
        dumped = SpecimenJsonSchema().dump(extracted_specimen)

        # Assert

        assert isinstance(dumped, dict)
        assert dumped["identifier"] == extracted_specimen.identifier
        if extracted_specimen.extraction_step is not None:
            assert_dict_equals_code(
                dumped["steps"][0]["method"],
                extracted_specimen.extraction_step.method,
            )
            if extracted_specimen.extraction_step.date_time is not None:
                assert (
                    dumped["steps"][0]["date_time"]
                    == extracted_specimen.extraction_step.date_time.isoformat()
                )
            else:
                assert "date_time" not in dumped["steps"][0]
        else:
            assert "steps" not in dumped["steps"][0]
            assert "date_time" not in dumped["steps"][0]
        if extracted_specimen.type is not None:
            assert_dict_equals_code(dumped["type"], extracted_specimen.type)
        else:
            assert "type" not in dumped
        if extracted_specimen.container is not None:
            assert_dict_equals_code(dumped["container"], extracted_specimen.container)
        else:
            assert "container" not in dumped

    def test_extracted_specimen_deserialize(self):
        # Arrange
        dumped = {
            "identifier": "specimen",
            "steps": [
                {
                    "action": "collection",
                    "method": {
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
            "container": {
                "value": "434711009",
                "scheme_designator": "SCT",
                "meaning": "Specimen container",
            },
        }

        # Act
        loaded = SpecimenJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, SpecimenJsonModel)
        assert loaded.identifier == dumped["identifier"]
        collection = loaded.steps[0]
        assert isinstance(collection, Collection)
        assert_dict_equals_code(dumped["steps"][0]["method"], collection.method)
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

        # Act
        dumped = SampleJsonSchema().dump(sample)

        # Assert
        assert isinstance(dumped, dict)
        assert dumped["identifier"] == sample.identifier
        if processing.method is not None:
            assert_dict_equals_code(dumped["steps"][0]["method"], processing.method)
        else:
            assert "method" not in dumped["steps"][0]
        if processing.date_time is not None:
            assert dumped["steps"][0]["date_time"] == processing.date_time.isoformat()
        else:
            assert "date_time" not in dumped["steps"][0]
        if sample.type is not None:
            assert_dict_equals_code(dumped["type"], sample.type)
        else:
            assert "type" not in dumped
        assert (
            dumped["sampled_from"][0]["identifier"]
            == sample.sampled_from[0].specimen.identifier
        )
        assert dumped["sampled_from"][0]["sampling_step_index"] == 0
        if sample.container is not None:
            assert_dict_equals_code(dumped["container"], sample.container)
        else:
            assert "container" not in dumped

    def test_sampled_specimen_deserialize(self):
        # Arrange
        dumped = {
            "identifier": "sample",
            "steps": [
                {
                    "action": "processing",
                    "method": {
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
            "container": {
                "value": "434464009",
                "scheme_designator": "SCT",
                "meaning": "Tissue cassette",
            },
        }

        # Act
        loaded = SampleJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, SampleJsonModel)
        assert loaded.identifier == dumped["identifier"]
        processing = loaded.steps[0]
        assert isinstance(processing, Processing)
        if processing.method is not None:
            assert_dict_equals_code(dumped["steps"][0]["method"], processing.method)
        else:
            assert "method" not in dumped["steps"][0]
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
        dumped = BaseSpecimenJsonSchema().dump(slide_sample)

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
                        "action": "processing",
                        "method": {
                            "value": "433452008",
                            "scheme_designator": "SCT",
                            "meaning": "Specimen clearing",
                        },
                        "date_time": "2023-08-05T00:00:00",
                    },
                    {
                        "action": "sampling",
                        "method": {
                            "value": "434472006",
                            "scheme_designator": "SCT",
                            "meaning": "Block sectioning",
                        },
                        "sampling_constraints": None,
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
                        "action": "collection",
                        "method": {
                            "value": "65801008",
                            "scheme_designator": "SCT",
                            "meaning": "Excision",
                        },
                        "date_time": "2023-08-05T00:00:00",
                        "description": "description",
                    },
                    {
                        "action": "sampling",
                        "method": {
                            "value": "122459003",
                            "scheme_designator": "SCT",
                            "meaning": "Dissection",
                        },
                        "sampling_constraints": None,
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
        loaded = BaseSpecimenJsonSchema().load(dumped)

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
        assert isinstance(specimen, Specimen)
        assert specimen.identifier == dumped[2]["identifier"]

    def test_slide_sample_rountrip(self, slide_sample: SlideSample):
        # Arrange

        # Act
        dumped = BaseSpecimenJsonSchema().dump(slide_sample)
        loaded = BaseSpecimenJsonSchema().load(dumped)

        # Assert
        assert str(loaded[0]) == str(slide_sample)
