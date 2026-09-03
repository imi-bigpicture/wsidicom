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

from contextlib import suppress
from dataclasses import FrozenInstanceError
from typing import cast

import pytest
from pydicom import config as pydicom_config
from pydicom.uid import generate_uid

from wsidicom import config
from wsidicom.config import (
    Settings,
    get_settings,
    set_default_settings,
    use_settings,
)
from wsidicom.metadata import Series
from wsidicom.metadata.schema.dicom import SeriesDicomSchema
from wsidicom.options import DicomValueValidationOption, ResampleFilterOption


@pytest.mark.unittest
class TestSettings:
    """The immutable `Settings` value: defaults, coercion and immutability."""

    def test_read_filter_defaults_to_bilinear(self):
        # Arrange
        configured = Settings()

        # Act & Assert
        assert configured.resampling_filter == ResampleFilterOption.BILINEAR

    def test_pyramid_filter_defaults_to_box(self):
        # Arrange
        configured = Settings()

        # Act & Assert
        assert configured.pyramid_resampling_filter == ResampleFilterOption.BOX

    def test_read_and_pyramid_filters_are_separate_fields(self):
        # Act — set only the read filter
        configured = Settings(resampling_filter=ResampleFilterOption.LANCZOS)

        # Assert — the pyramid filter keeps its default
        assert configured.pyramid_resampling_filter == ResampleFilterOption.BOX

    def test_option_field_accepts_its_string_value(self):
        # Arrange — the string form (cast since the field is typed as the enum)
        string_value = cast(ResampleFilterOption, "box")

        # Act — coerced in __post_init__
        configured = Settings(resampling_filter=string_value)

        # Assert — coerced to the enum member
        assert configured.resampling_filter == ResampleFilterOption.BOX

    def test_is_immutable(self):
        # Arrange
        configured = Settings()

        # Act & Assert
        with pytest.raises(FrozenInstanceError):
            configured.resampling_filter = ResampleFilterOption.BOX  # type: ignore[misc]


@pytest.mark.unittest
class TestDefaultSettings:
    """The process-wide default and the `use_settings` scope: `get_settings`
    returns the scope-active `Settings` when one is active, else the default;
    `set_default_settings` changes the default."""

    @pytest.fixture(autouse=True)
    def _restore_default_settings(self):
        # The default is an immutable Settings that the setters rebind; snapshot
        # and restore the reference so a test does not leak into the others.
        original = config._default_settings
        yield
        config._default_settings = original

    def test_resolves_to_default_outside_a_scope(self):
        # Act
        resampling_filter = get_settings().resampling_filter

        # Assert
        assert resampling_filter == ResampleFilterOption.BILINEAR

    def test_use_settings_activates_within_scope_and_restores_after(self):
        # Arrange
        custom = Settings(resampling_filter=ResampleFilterOption.LANCZOS)

        # Act & Assert — active inside the scope, default again after
        with use_settings(custom):
            assert get_settings().resampling_filter == ResampleFilterOption.LANCZOS
        assert get_settings().resampling_filter == ResampleFilterOption.BILINEAR

    def test_set_default_settings_replaces_the_default(self):
        # Act
        set_default_settings(Settings(resampling_filter=ResampleFilterOption.BOX))

        # Assert
        assert get_settings().resampling_filter == ResampleFilterOption.BOX

    def test_default_change_targets_the_default_not_the_active_scope(self):
        # Arrange
        custom = Settings(resampling_filter=ResampleFilterOption.LANCZOS)

        # Act — changing the default inside a scope still targets the default
        with use_settings(custom):
            set_default_settings(Settings(resampling_filter=ResampleFilterOption.BOX))
            active = get_settings().resampling_filter

        # Assert — reads honoured the active `custom`; the default took the change
        assert active == ResampleFilterOption.LANCZOS
        assert get_settings().resampling_filter == ResampleFilterOption.BOX


@pytest.mark.unittest
class TestPydicomGlobalValidationMode:
    """The mode is passed per value, so pydicom's global mode is never set."""

    @pytest.mark.parametrize("option", list(DicomValueValidationOption))
    def test_pydicom_global_validation_mode_is_never_touched(
        self, option: DicomValueValidationOption
    ):
        """Whichever mode wsidicom is set to, it reaches pydicom per value.

        Checking through pydicom's global mode would be the other way to do it,
        and would leave a process using wsidicom with a mode it did not choose.
        """
        # Arrange
        before = pydicom_config.settings.reading_validation_mode
        settings = Settings(dicom_value_validation=option)
        # Series Description is LO, which allows 64
        series = Series(uid=generate_uid(), description="X" * 100)

        # Act
        with use_settings(settings):
            with suppress(ValueError):
                SeriesDicomSchema().dump(series)
            during = pydicom_config.settings.reading_validation_mode

        # Assert
        assert during == before
        assert pydicom_config.settings.reading_validation_mode == before
