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


class Settings:
    """Class containing settings. Settings are to be accessed through the
    global variable settings."""

    def __init__(self) -> None:
        self._strict_uid_check = False
        self._strict_attribute_check = False
        self._parse_pixel_data_on_load = True
        self._focal_plane_distance_threshold = 0.000001
        self._stitching_workers = None

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
    def parse_pixel_data_on_load(self) -> bool:
        """If to parse pixel data for frame positions on file load."""
        return self._parse_pixel_data_on_load

    @parse_pixel_data_on_load.setter
    def parse_pixel_data_on_load(self, value: bool) -> None:
        self._parse_pixel_data_on_load = value

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
    def stitching_workers(self) -> Optional[int]:
        """The number of threads to use for stitching tiles. Set to None to let the
        implementation decided. Set to 1 to dissable threading."""
        return self._stitching_workers

    @stitching_workers.setter
    def stitching_workers(self, value: Optional[int]) -> None:
        self._stitching_workers = value


settings = Settings()
"""Global settings variable."""
