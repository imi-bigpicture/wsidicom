#    Copyright 2022 SECTRA AB
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


from typing import Optional

from PIL import Image as Pillow


class Settings:
    """Class containing settings. Settings are to be accessed through the
    global variable settings."""

    def __init__(self) -> None:
        self._strict_uid_check = False
        self._strict_tile_size_check = True
        self._strict_attribute_check = False
        self._strict_specimen_identifier_check = True
        self._focal_plane_distance_threshold = 0.000001
        self._pyramids_origin_threshold = 0.02
        self._prefered_decoder: Optional[str] = None
        self._open_web_theads: Optional[int] = None
        self._pillow_resampling_filter = Pillow.Resampling.BILINEAR
        self._ignore_specimen_preparation_step_on_validation_error = True
        self._truncate_long_dicom_strings = False
        self._decoded_frame_cache_size = 100 * 1024 * 1024
        self._encoded_frame_cache_size = 100 * 1024 * 1024
        self._level_scale_tolerance = 1e-2

    @property
    def strict_uid_check(self) -> bool:
        """If frame of reference uid needs to match."""
        return self._strict_uid_check

    @strict_uid_check.setter
    def strict_uid_check(self, value: bool) -> None:
        self._strict_uid_check = value

    @property
    def strict_attribute_check(self) -> bool:
        """If attribute marked with Requirement.STRICT is required."""
        return self._strict_attribute_check

    @strict_attribute_check.setter
    def strict_attribute_check(self, value: bool) -> None:
        self._strict_attribute_check = value

    @property
    def strict_specimen_identifier_check(self) -> bool:
        """If `True` the issuer of two specimen identifiers needs to match or both be
        None for the identifiers to match. If `False` the identifiers will match also if
        either issuer is None. Either way the identifier needs to match."""
        return self._strict_specimen_identifier_check

    @strict_specimen_identifier_check.setter
    def strict_specimen_identifier_check(self, value: bool) -> None:
        self._strict_specimen_identifier_check = value

    @property
    def strict_tile_size_check(self) -> bool:
        """If tile size need to match for levels. If `False` the tile size accross
        levels are allowed to be non-uniform."""
        return self._strict_tile_size_check

    @strict_tile_size_check.setter
    def strict_tile_size_check(self, value: bool) -> None:
        self._strict_tile_size_check = value

    @property
    def focal_plane_distance_threshold(self) -> float:
        """Threshold in mm for which distances between focal planes are
        considered to be equal. Default is 1 nm, as distance between focal
        planes are likely larger than this.
        """
        return self._focal_plane_distance_threshold

    @focal_plane_distance_threshold.setter
    def focal_plane_distance_threshold(self, value: float) -> None:
        self._focal_plane_distance_threshold = value

    @property
    def pyramids_origin_threshold(self) -> float:
        """Threshold in mm for the distance between origins of instances
        to group them into the same pyramid. Default is 0.02 mm.
        """
        return self._pyramids_origin_threshold

    @pyramids_origin_threshold.setter
    def pyramids_origin_threshold(self, value: float) -> None:
        self._pyramids_origin_threshold = value

    @property
    def prefered_decoder(self) -> Optional[str]:
        """Name of preferred decoder to use."""
        return self._prefered_decoder

    @prefered_decoder.setter
    def prefered_decoder(self, value: Optional[str]) -> None:
        self._prefered_decoder = value

    @property
    def open_web_theads(self) -> Optional[int]:
        """Number of threads to use when opening web instances."""
        return self._open_web_theads

    @open_web_theads.setter
    def open_web_theads(self, value: Optional[int]) -> None:
        self._open_web_theads = value

    @property
    def pillow_resampling_filter(self) -> Pillow.Resampling:
        """The resampling filter to use when rescaling images."""
        return self._pillow_resampling_filter

    @pillow_resampling_filter.setter
    def pillow_resampling_filter(self, value: Pillow.Resampling) -> None:
        self._pillow_resampling_filter = value

    @property
    def ignore_specimen_preparation_step_on_validation_error(self) -> bool:
        """If ignore specimen preparation steps that fails to validate. If false all
        steps will be ignored if one fails to validate."""
        return self._ignore_specimen_preparation_step_on_validation_error

    @ignore_specimen_preparation_step_on_validation_error.setter
    def ignore_specimen_preparation_step_on_validation_error(self, value: bool) -> None:
        self._ignore_specimen_preparation_step_on_validation_error = value

    @property
    def truncate_long_dicom_strings_on_validation_error(self) -> bool:
        """If long DICOM strings should be truncated. This is only used if
        `pydicom.settings.writing_validation_mode` is set to `pydicom.config.RAISE`. If
        set to `True` long strings will be truncated if needed to pass validation."""
        return self._truncate_long_dicom_strings

    @truncate_long_dicom_strings_on_validation_error.setter
    def truncate_long_dicom_strings_on_validation_error(self, value: bool) -> None:
        self._truncate_long_dicom_strings = value

    @property
    def decoded_frame_cache_size(self) -> int:
        """Size of the decoded frame cache. Default is 100 MB."""
        return self._decoded_frame_cache_size

    @decoded_frame_cache_size.setter
    def decoded_frame_cache_size(self, value: int) -> None:
        self._decoded_frame_cache_size = value

    @property
    def encoded_frame_cache_size(self) -> int:
        """Size of the encoded frame cache. Default is 100 MB."""
        return self._encoded_frame_cache_size

    @encoded_frame_cache_size.setter
    def encoded_frame_cache_size(self, value: int) -> None:
        self._encoded_frame_cache_size = value

    @property
    def level_scale_tolerance(self) -> float:
        """Tolerance for level scale comparison. Default is 1e-2."""
        return self._level_scale_tolerance

    @level_scale_tolerance.setter
    def level_scale_tolerance(self, value: float) -> None:
        self._level_scale_tolerance = value


settings = Settings()
"""Global settings variable."""
