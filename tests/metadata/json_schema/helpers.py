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

from typing import Any, Dict, Union

from pydicom.sr.coding import Code

from wsidicom.conceptcode import ConceptCode
from wsidicom.metadata.optical_path import (
    ConstantLutSegment,
    DiscreteLutSegment,
    LinearLutSegment,
    Lut,
    LutSegment,
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
