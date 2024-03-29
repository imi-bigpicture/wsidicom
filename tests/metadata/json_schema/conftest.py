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

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytest

from wsidicom.metadata.sample import (
    SpecimenIdentifier,
    UniversalIssuerOfIdentifier,
)


@pytest.fixture()
def json_slide_identifier(slide_identifier: Optional[Union[str, SpecimenIdentifier]]):
    if slide_identifier is None:
        yield None
    elif isinstance(slide_identifier, str):
        yield slide_identifier
    elif not isinstance(slide_identifier.issuer, UniversalIssuerOfIdentifier):
        yield {
            "value": slide_identifier.value,
        }
    else:
        yield {
            "value": slide_identifier.value,
            "issuer": {
                "identifier": slide_identifier.issuer.identifier,
                "type": slide_identifier.issuer.issuer_type.name,
                "local": slide_identifier.issuer.local_identifier,
            },
        }


@pytest.fixture()
def json_slide(json_slide_identifier: Union[str, Dict[str, Any]]):
    yield {
        "identifier": json_slide_identifier,
        "stainings": [
            {
                "substances": [
                    {
                        "value": "12710003",
                        "scheme_designator": "SCT",
                        "meaning": "hematoxylin stain",
                    },
                    {
                        "value": "36879007",
                        "scheme_designator": "SCT",
                        "meaning": "water soluble eosin stain",
                    },
                ],
                "date_time": "2023-08-05T00:00:00",
            }
        ],
        "samples": [
            {
                "identifier": "Sample 1",
                "steps": [],
                "anatomical_sites": [
                    {
                        "value": "value",
                        "scheme_designator": "schema",
                        "meaning": "meaning",
                    }
                ],
                "sampled_from": {"identifier": "block 1", "sampling_step_index": 0},
                "uid": "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445423",
                "localization": {"description": "left"},
            },
            {
                "identifier": "block 1",
                "steps": [
                    {
                        "action": "embedding",
                        "medium": {
                            "value": "311731000",
                            "scheme_designator": "SCT",
                            "meaning": "Paraffin wax",
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
                        "sampling_constraints": [
                            {"identifier": "part 1", "sampling_step_index": 0}
                        ],
                        "date_time": "2023-08-05T00:00:00",
                        "description": "Sampling to slide",
                    },
                    {
                        "action": "sampling",
                        "method": {
                            "value": "434472006",
                            "scheme_designator": "SCT",
                            "meaning": "Block sectioning",
                        },
                        "sampling_constraints": [
                            {"identifier": "part 2", "sampling_step_index": 0}
                        ],
                        "date_time": "2023-08-05T00:00:00",
                        "description": "Sampling to slide",
                    },
                ],
                "sampled_from": [
                    {"identifier": "part 1", "sampling_step_index": 0},
                    {"identifier": "part 2", "sampling_step_index": 0},
                ],
                "type": {
                    "value": "119376003",
                    "scheme_designator": "SCT",
                    "meaning": "tissue specimen",
                },
            },
            {
                "identifier": "part 1",
                "steps": [
                    {
                        "action": "collection",
                        "method": {
                            "value": "17636008",
                            "scheme_designator": "SCT",
                            "meaning": "Specimen collection",
                        },
                        "date_time": "2023-08-05T00:00:00",
                        "description": "Extracted",
                    },
                    {
                        "action": "fixation",
                        "fixative": {
                            "value": "434162003",
                            "scheme_designator": "SCT",
                            "meaning": "Neutral Buffered Formalin",
                        },
                        "date_time": "2023-08-05T00:00:00",
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
                    "value": "119376003",
                    "scheme_designator": "SCT",
                    "meaning": "tissue specimen",
                },
            },
            {
                "identifier": "part 2",
                "steps": [
                    {
                        "action": "collection",
                        "method": {
                            "value": "17636008",
                            "scheme_designator": "SCT",
                            "meaning": "Specimen collection",
                        },
                        "date_time": "2023-08-05T00:00:00",
                        "description": "Extracted",
                    },
                    {
                        "action": "fixation",
                        "fixative": {
                            "value": "434162003",
                            "scheme_designator": "SCT",
                            "meaning": "Neutral Buffered Formalin",
                        },
                        "date_time": "2023-08-05T00:00:00",
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
                    "value": "119376003",
                    "scheme_designator": "SCT",
                    "meaning": "tissue specimen",
                },
            },
            {
                "identifier": "Sample 2",
                "steps": [],
                "anatomical_sites": [
                    {
                        "value": "value",
                        "scheme_designator": "schema",
                        "meaning": "meaning",
                    }
                ],
                "sampled_from": {"identifier": "block 1", "sampling_step_index": 1},
                "uid": "1.2.826.0.1.3680043.8.498.11522107373528810886192809691753445424",
                "localization": {"description": "right"},
            },
        ],
    }


@pytest.fixture
def icc_file(icc_profile: bytes, tmp_path: Path):
    icc_filename = "test.icc"
    icc_filepath = tmp_path.joinpath(icc_filename)
    with open(tmp_path.joinpath(icc_filepath), "wb") as file:
        file.write(icc_profile)
    yield icc_filepath
