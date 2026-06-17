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

from dataclasses import replace
from pathlib import Path

import pytest
from pydicom.tag import BaseTag
from pydicom.uid import generate_uid

from wsidicom import WsiDicom
from wsidicom.metadata import Patient, WsiMetadata
from wsidicom.metadata.uid_generator import CallableUidGenerator

ANONYMIZED_NAME = "Anonymous"


@pytest.mark.unittest
class TestSaveWithMetadata:
    @staticmethod
    def _clean_metadata(wsi: WsiDicom) -> WsiMetadata:
        """Return source metadata with an anonymized patient.

        The pyramid uid is populated as it is a Type 1 attribute required when
        dumping, and may be absent from the source metadata.
        """
        return replace(
            wsi.metadata,
            patient=Patient(name=ANONYMIZED_NAME),
            pyramid=replace(wsi.metadata.pyramid, uid=generate_uid()),
        )

    @staticmethod
    def _add_private_tag(wsi: WsiDicom, value: str = "private secret") -> BaseTag:
        """Add a private tag to every instance dataset and return its tag.

        Stands in for a private tag or other unhandled attribute that may carry
        PHI and that is not modeled by the metadata schema.
        """
        tag: BaseTag | None = None
        for pyramid in wsi.pyramids:
            for level in pyramid:
                for instance in level.instances.values():
                    block = instance.dataset.private_block(
                        0x0009, "WSIDICOM TEST", create=True
                    )
                    block.add_new(0x01, "LO", value)
                    tag = block.get_tag(0x01)
        assert tag is not None
        return tag

    def test_replace_metadata_replaces_phi_and_drops_unmodeled_tags(
        self,
        wsi: WsiDicom,
        uid_generator: CallableUidGenerator,
        tmp_path: Path,
    ):
        # Arrange
        private_tag = self._add_private_tag(wsi)
        assert str(wsi.metadata.patient.name) != ANONYMIZED_NAME
        metadata = self._clean_metadata(wsi)
        output = tmp_path.joinpath("replace")
        output.mkdir()

        # Act
        wsi.save(output, uid_generator=uid_generator, metadata=metadata)

        # Assert
        with WsiDicom.open(output) as saved:
            dataset = saved.pyramids[0].base_level.default_instance.dataset
            assert str(dataset.PatientName) == ANONYMIZED_NAME
            assert private_tag not in dataset

    def test_update_metadata_replaces_phi_and_keeps_unmodeled_tags(
        self,
        wsi: WsiDicom,
        uid_generator: CallableUidGenerator,
        tmp_path: Path,
    ):
        # Arrange
        private_tag = self._add_private_tag(wsi)
        metadata = self._clean_metadata(wsi)
        output = tmp_path.joinpath("update")
        output.mkdir()

        # Act
        wsi.save(
            output,
            uid_generator=uid_generator,
            metadata=metadata,
            replace_metadata=False,
        )

        # Assert
        with WsiDicom.open(output) as saved:
            dataset = saved.pyramids[0].base_level.default_instance.dataset
            assert str(dataset.PatientName) == ANONYMIZED_NAME
            assert private_tag in dataset

    def test_default_icc_profile_inserted_for_color_when_missing(
        self,
        wsi: WsiDicom,
        uid_generator: CallableUidGenerator,
        tmp_path: Path,
    ):
        # Arrange — color fixture whose optical path has no ICC profile. ICC
        # Profile is required (Type 1C) for non-MONOCHROME2 images, so save()
        # should insert a default one.
        metadata = self._clean_metadata(wsi)
        output = tmp_path.joinpath("icc_default")
        output.mkdir()

        # Act
        wsi.save(output, uid_generator=uid_generator, metadata=metadata)

        # Assert
        with WsiDicom.open(output) as saved:
            dataset = saved.pyramids[0].base_level.default_instance.dataset
            optical_path = dataset.OpticalPathSequence[0]
            assert "ICCProfile" in optical_path
            assert len(optical_path.ICCProfile) > 0

    def test_icc_profile_in_metadata_is_preserved(
        self,
        wsi: WsiDicom,
        uid_generator: CallableUidGenerator,
        tmp_path: Path,
    ):
        # Arrange — supply an ICC profile via the metadata optical paths; it
        # should be kept rather than replaced by an auto-inserted default.
        icc_profile = b"custom icc profile"
        metadata = self._clean_metadata(wsi)
        pyramid = metadata.pyramid
        optical_paths = [
            replace(optical_path, icc_profile=icc_profile)
            for optical_path in pyramid.optical_paths
        ]
        metadata = replace(
            metadata, pyramid=replace(pyramid, optical_paths=optical_paths)
        )
        output = tmp_path.joinpath("icc_in_metadata")
        output.mkdir()

        # Act
        wsi.save(output, uid_generator=uid_generator, metadata=metadata)

        # Assert
        with WsiDicom.open(output) as saved:
            dataset = saved.pyramids[0].base_level.default_instance.dataset
            optical_path = dataset.OpticalPathSequence[0]
            assert optical_path.ICCProfile == icc_profile

    def test_no_metadata_keeps_source_dataset(
        self,
        wsi: WsiDicom,
        uid_generator: CallableUidGenerator,
        tmp_path: Path,
    ):
        # Arrange
        private_tag = self._add_private_tag(wsi)
        original_name = str(wsi.metadata.patient.name)
        output = tmp_path.joinpath("none")
        output.mkdir()

        # Act
        wsi.save(output, uid_generator=uid_generator)

        # Assert
        with WsiDicom.open(output) as saved:
            dataset = saved.pyramids[0].base_level.default_instance.dataset
            assert str(dataset.PatientName) == original_name
            assert private_tag in dataset
