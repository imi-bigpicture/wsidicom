#    Copyright 2026 SECTRA AB
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

import pytest
from pydicom.dataset import Dataset
from pydicom.uid import UID

from wsidicom.metadata import (
    Equipment,
    Image,
    Label,
    Patient,
    Pyramid,
    Series,
    Slide,
    Study,
    WsiMetadata,
)
from wsidicom.metadata.sample import SlideSample
from wsidicom.metadata.uid_generator import CallableUidGenerator


@pytest.fixture
def metadata() -> WsiMetadata:
    return WsiMetadata(
        study=Study(),
        series=Series(),
        patient=Patient(),
        equipment=Equipment(),
        slide=Slide(samples=[SlideSample(identifier="s1")]),
        pyramid=Pyramid(image=Image(), optical_paths=[]),
        label=Label(),
    )


@pytest.mark.unittest
class TestUidGenerator:
    def test_callable_default_produces_unique_per_role(self, metadata: WsiMetadata):
        # Arrange
        gen = CallableUidGenerator()

        # Act
        produced = {
            gen.study_uid(metadata.study),
            gen.series_uid(metadata.series),
            gen.frame_of_reference_uid(metadata),
            gen.dimension_organization_uid(metadata),
            gen.sample_uid(SlideSample(identifier="s1")),
            gen.pyramid_uid(metadata.pyramid),
            gen.sop_uid(Dataset()),
            gen.annotation_group_uid(),
        }

        # Assert
        assert len(produced) == 8

    @pytest.mark.parametrize(
        ["identifier", "expected"],
        [
            ["11", "1.2.3.11"],
            ["22", "1.2.3.22"],
        ],
    )
    def test_subclass_sees_entity_context(self, identifier: str, expected: str):
        """Subclasses receive the entity, so they can derive UIDs from it."""

        # Arrange
        class IdentifierBased(CallableUidGenerator):
            def study_uid(self, study: Study) -> UID:
                return UID(f"1.2.3.{study.identifier}")

        gen = IdentifierBased()

        # Act
        study_uid = gen.study_uid(Study(identifier=identifier))

        # Assert
        assert study_uid == expected
