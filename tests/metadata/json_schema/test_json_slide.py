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

from typing import Any, Dict

import pytest

from tests.metadata.json_schema.helpers import assert_dict_equals_code
from wsidicom.conceptcode import SpecimenStainsCode
from wsidicom.metadata.sample import (
    LocalIssuerOfIdentifier,
    SpecimenIdentifier,
    UniversalIssuerOfIdentifier,
    UniversalIssuerType,
)
from wsidicom.metadata.schema.json.slide import SlideJsonSchema
from wsidicom.metadata.slide import Slide


class TestSlideJsonSchema:
    @pytest.mark.parametrize(
        "slide_identifier",
        [
            None,
            "identifier",
            SpecimenIdentifier("identifier"),
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
            SpecimenIdentifier(
                "identifier",
                UniversalIssuerOfIdentifier("issuer", UniversalIssuerType.UUID),
            ),
        ],
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
    def test_slide_serialize(self, slide: Slide):
        # Arrange
        assert slide.samples is not None
        sample = slide.samples[0]
        assert sample.sampled_from is not None
        block_1 = sample.sampled_from.specimen

        # Act
        dumped = SlideJsonSchema().dump(slide)

        # Assert
        assert isinstance(dumped, dict)
        if isinstance(slide.identifier, SpecimenIdentifier):
            assert dumped["identifier"]["value"] == slide.identifier.value
            if isinstance(slide.identifier.issuer, UniversalIssuerOfIdentifier):
                assert (
                    dumped["identifier"]["issuer"]["identifier"]
                    == slide.identifier.issuer.identifier
                )
                assert (
                    dumped["identifier"]["issuer"]["issuer_type"]
                    == slide.identifier.issuer.issuer_type.name
                )
        else:
            assert dumped["identifier"] == slide.identifier
        if slide.stainings is not None:
            for index, staining in enumerate(slide.stainings):
                dumped_staining = dumped["stainings"][index]
                if isinstance(staining.substances, str):
                    assert dumped_staining["substances"] == staining.substances
                else:
                    for stain_index, stain in enumerate(staining.substances):
                        assert isinstance(stain, SpecimenStainsCode)
                        assert_dict_equals_code(
                            dumped_staining["substances"][stain_index], stain
                        )
        else:
            assert "stainings" not in dumped

        dumped_sample = dumped["samples"][0]
        assert dumped_sample["identifier"] == sample.identifier
        if sample.anatomical_sites is not None:
            assert_dict_equals_code(
                dumped_sample["anatomical_sites"][0], sample.anatomical_sites[0]
            )
        else:
            assert "anatomical_sites" not in dumped_sample
        assert dumped_sample["uid"] == str(sample.uid)
        if sample.localization is not None:
            assert (
                dumped_sample["localization"]["description"]
                == sample.localization.description
            )
        else:
            assert "localization" not in dumped_sample
        dumped_block_1 = dumped["samples"][1]
        assert dumped_block_1["identifier"] == block_1.identifier
        if block_1.type is not None:
            assert_dict_equals_code(dumped_block_1["type"], block_1.type)

    @pytest.mark.parametrize(
        "slide_identifier",
        [
            None,
            "identifier",
            SpecimenIdentifier("identifier"),
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
            SpecimenIdentifier(
                "identifier",
                UniversalIssuerOfIdentifier("issuer", UniversalIssuerType.UUID),
            ),
        ],
    )
    def test_slide_deserialize(self, json_slide: Dict[str, Any]):
        # Arrange

        # Act
        loaded = SlideJsonSchema().load(json_slide)

        # Assert
        assert isinstance(loaded, Slide)
