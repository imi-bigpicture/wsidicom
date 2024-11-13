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

"""Module with default values for DICOM serialization."""

import datetime

from wsidicom.conceptcode import (
    ContainerComponentTypeCode,
    ContainerTypeCode,
    IlluminationCode,
    IlluminationColorCode,
)
from wsidicom.geometry import SizeMm
from wsidicom.metadata.image import FocusMethod


class Defaults:
    string = "Unknown"
    date_time = datetime.datetime.fromtimestamp(0, datetime.timezone.utc)
    optical_path_identifier = "0"
    illumination_type = IlluminationCode("Brightfield illumination")
    illumination = IlluminationColorCode("Full Spectrum")
    slide_container_type = ContainerTypeCode("Microscope slide")
    slide_component_type = ContainerComponentTypeCode("Microscope slide cover slip")
    slide_material = "GLASS"
    focus_method = FocusMethod.AUTO
    slide_size_without_label = SizeMm(25, 50)
    image_coordinate_system_rotation: float = 180


defaults = Defaults()
