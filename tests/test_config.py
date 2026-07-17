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

from dataclasses import FrozenInstanceError

import pytest

from wsidicom import config
from wsidicom.config import Settings, settings, use_settings
from wsidicom.options import ResampleFilterOption


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
        # Act
        configured = Settings(resampling_filter="box")

        # Assert — coerced to the enum member
        assert configured.resampling_filter == ResampleFilterOption.BOX

    def test_is_immutable(self):
        # Arrange
        configured = Settings()

        # Act & Assert
        with pytest.raises(FrozenInstanceError):
            configured.resampling_filter = ResampleFilterOption.BOX  # type: ignore[misc]


@pytest.mark.unittest
class TestSettingsContext:
    """The global `settings`: reads resolve to the context-active `Settings` (via
    `use_settings`) or the default; writes replace the process default."""

    @pytest.fixture(autouse=True)
    def _restore_default_settings(self):
        # Proxy writes replace the process default; snapshot and restore it so
        # a test writing through `settings` does not leak into the others.
        original = config._default_settings
        yield
        config._default_settings = original

    def test_resolves_to_default_outside_a_context(self):
        # Act
        resampling_filter = settings.resampling_filter

        # Assert
        assert resampling_filter == ResampleFilterOption.BILINEAR

    def test_use_settings_overrides_within_scope_and_restores_after(self):
        # Arrange
        custom = Settings(resampling_filter=ResampleFilterOption.LANCZOS)

        # Act & Assert — active inside the scope, default again after
        with use_settings(custom):
            assert settings.resampling_filter == ResampleFilterOption.LANCZOS
        assert settings.resampling_filter == ResampleFilterOption.BILINEAR

    def test_proxy_write_replaces_the_default(self):
        # Act — writing through the proxy configures the process default
        settings.resampling_filter = ResampleFilterOption.BOX

        # Assert
        assert settings.resampling_filter == ResampleFilterOption.BOX

    def test_proxy_write_targets_the_default_not_the_active_settings(self):
        # Arrange
        custom = Settings(resampling_filter=ResampleFilterOption.LANCZOS)

        # Act — a proxy write inside a context still targets the default
        with use_settings(custom):
            settings.resampling_filter = ResampleFilterOption.BOX
            active = settings.resampling_filter

        # Assert — reads honoured the active `custom`; the default took the write
        assert active == ResampleFilterOption.LANCZOS
        assert settings.resampling_filter == ResampleFilterOption.BOX
