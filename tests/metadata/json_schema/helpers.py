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

from typing import Any, Dict, Optional, Sequence, Union

from pydicom.sr.coding import Code

from wsidicom.conceptcode import ConceptCode, IlluminationColorCode
from wsidicom.geometry import SizeMm
from wsidicom.metadata.image import Image, ImageCoordinateSystem, LossyCompression
from wsidicom.metadata.optical_path import (
    ConstantLutSegment,
    DiscreteLutSegment,
    LinearLutSegment,
    Lut,
    LutSegment,
    OpticalPath,
)


def assert_dict_equals_code(
    dumped_code: Dict[str, str], expected_code: Union[Code, ConceptCode]
):
    assert dumped_code["value"] == expected_code.value
    assert dumped_code["scheme_designator"] == expected_code.scheme_designator
    assert dumped_code["meaning"] == expected_code.meaning
    assert dumped_code.get("scheme_version", None) == expected_code.scheme_version


def assert_lut_is_equal(dumped: Dict[str, Any], lut: Lut):
    assert dumped["bits"] == 8 * lut.data_type().itemsize
    for dumped_component, component in zip(
        (dumped["red"], dumped["green"], dumped["blue"]),
        (lut.red, lut.green, lut.blue),
    ):
        assert len(dumped_component) == len(component)
        for dumped_segment, segment in zip(dumped_component, component):
            assert_lut_segment_is_equal(dumped_segment, segment)


def assert_lut_segment_is_equal(dumped: Dict[str, Any], segment: LutSegment):
    if isinstance(segment, LinearLutSegment):
        assert dumped["start_value"] == segment.start_value
        assert dumped["end_value"] == segment.end_value
        assert dumped["length"] == segment.length
    elif isinstance(segment, ConstantLutSegment):
        assert dumped["value"] == segment.value
        assert dumped["length"] == segment.length
    elif isinstance(segment, DiscreteLutSegment):
        assert dumped["values"] == segment.values


def assert_image_coordinate_system_is_equal(
    dumped: Optional[Dict[str, Any]],
    image_coordinate_system: Optional[ImageCoordinateSystem],
):
    if image_coordinate_system is None:
        assert dumped is None
        return
    assert dumped is not None
    assert dumped["origin"]["x"] == image_coordinate_system.origin.x
    assert dumped["origin"]["y"] == image_coordinate_system.origin.y
    assert dumped["rotation"] == image_coordinate_system.rotation
    if image_coordinate_system.z_offset is None:
        assert dumped["z_offset"] is None
    else:
        assert dumped["z_offset"] == image_coordinate_system.z_offset


def assert_pixel_spacing_is_equal(
    dumped: Optional[Dict[str, Any]],
    pixel_spacing: Optional[SizeMm],
):
    if pixel_spacing is None:
        assert dumped is None
        return
    assert dumped is not None
    assert dumped["width"] == pixel_spacing.width
    assert dumped["height"] == pixel_spacing.height


def assert_lossy_compression_is_equal(
    dumped: Optional[Sequence[Dict[str, Any]]],
    lossy_compression: Optional[Sequence[LossyCompression]],
):
    if lossy_compression is None:
        assert dumped is None
        return
    assert dumped is not None
    assert len(dumped) == len(lossy_compression)
    for dumped_compression, expected_compression in zip(dumped, lossy_compression):
        assert dumped_compression["method"] == expected_compression.method.value
        assert dumped_compression["ratio"] == expected_compression.ratio


def assert_image_is_equal(dumped: Dict[str, Any], image: Image):
    if image.acquisition_datetime is None:
        assert dumped["acquisition_datetime"] is None
    else:
        assert dumped["acquisition_datetime"] == image.acquisition_datetime.isoformat()
    if image.focus_method is None:
        assert dumped["focus_method"] is None
    else:
        assert dumped["focus_method"] == image.focus_method.value
    if image.extended_depth_of_field is None:
        assert dumped["extended_depth_of_field"] is None
    else:
        assert (
            dumped["extended_depth_of_field"]["number_of_focal_planes"]
            == image.extended_depth_of_field.number_of_focal_planes
        )
        assert (
            dumped["extended_depth_of_field"]["distance_between_focal_planes"]
            == image.extended_depth_of_field.distance_between_focal_planes
        )
    assert_image_coordinate_system_is_equal(
        dumped.get("image_coordinate_system"), image.image_coordinate_system
    )

    assert_pixel_spacing_is_equal(dumped.get("pixel_spacing"), image.pixel_spacing)

    if image.focal_plane_spacing is None:
        assert dumped["focal_plane_spacing"] is None
    else:
        assert dumped["focal_plane_spacing"] == image.focal_plane_spacing
    if image.depth_of_field is None:
        assert dumped["depth_of_field"] is None
    else:
        assert dumped["depth_of_field"] == image.depth_of_field
    assert_lossy_compression_is_equal(
        dumped.get("lossy_compressions"), image.lossy_compressions
    )


def assert_optical_path_is_equal(dumped: Dict[str, Any], optical_path: OpticalPath):
    assert dumped["identifier"] == optical_path.identifier
    assert dumped["description"] == optical_path.description
    if optical_path.illumination_types is None:
        assert dumped.get("illumination_types") is None
    else:
        assert len(dumped["illumination_types"]) == len(optical_path.illumination_types)
        for dumped_type, expected_type in zip(
            dumped["illumination_types"], optical_path.illumination_types
        ):
            assert_dict_equals_code(dumped_type, expected_type)
    if optical_path.light_path_filter is None:
        assert dumped.get("light_path_filter") is None
    elif optical_path.light_path_filter is None:
        assert dumped.get("light_path_filter") is None
    else:
        if isinstance(optical_path.illumination, IlluminationColorCode):
            assert_dict_equals_code(dumped["illumination"], optical_path.illumination)
        else:
            assert dumped["illumination"] == optical_path.illumination
        assert optical_path.light_path_filter.filters is not None
        assert_dict_equals_code(
            dumped["light_path_filter"]["filters"][0],
            optical_path.light_path_filter.filters[0],
        )
        assert (
            dumped["light_path_filter"]["nominal"]
            == optical_path.light_path_filter.nominal
        )
        assert (
            dumped["light_path_filter"]["low_pass"]
            == optical_path.light_path_filter.low_pass
        )
        assert (
            dumped["light_path_filter"]["high_pass"]
            == optical_path.light_path_filter.high_pass
        )
    if optical_path.image_path_filter is None:
        assert dumped.get("image_path_filter") is None
    else:
        assert optical_path.image_path_filter.filters is not None
        assert_dict_equals_code(
            dumped["image_path_filter"]["filters"][0],
            optical_path.image_path_filter.filters[0],
        )
        assert (
            dumped["image_path_filter"]["nominal"]
            == optical_path.image_path_filter.nominal
        )
        assert (
            dumped["image_path_filter"]["low_pass"]
            == optical_path.image_path_filter.low_pass
        )
        assert (
            dumped["image_path_filter"]["high_pass"]
            == optical_path.image_path_filter.high_pass
        )
    if optical_path.objective is None:
        assert dumped.get("objective") is None
    else:
        assert optical_path.objective.lenses is not None
        assert_dict_equals_code(
            dumped["objective"]["lenses"][0],
            optical_path.objective.lenses[0],
        )
        assert (
            dumped["objective"]["condenser_power"]
            == optical_path.objective.condenser_power
        )
        assert (
            dumped["objective"]["objective_power"]
            == optical_path.objective.objective_power
        )
        assert (
            dumped["objective"]["objective_numerical_aperture"]
            == optical_path.objective.objective_numerical_aperture
        )

    if optical_path.lut is not None:
        assert_lut_is_equal(dumped["lut"], optical_path.lut)
