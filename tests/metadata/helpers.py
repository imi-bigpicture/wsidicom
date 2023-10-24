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

from typing import Any, Dict, Sequence, Union
from pydicom import Dataset

from pydicom.sr.coding import Code
from wsidicom.conceptcode import ConceptCode
from wsidicom.metadata.optical_path import (
    ConstantLutSegment,
    DiscreteLutSegment,
    LinearLutSegment,
    Lut,
    LutSegment,
)


def bool_to_dicom_literal(value: bool) -> str:
    if value:
        return "YES"
    return "NO"


def assert_dict_equals_code(
    dumped_code: Dict[str, str], expected_code: Union[Code, ConceptCode]
):
    assert dumped_code["value"] == expected_code.value
    assert dumped_code["scheme_designator"] == expected_code.scheme_designator
    assert dumped_code["meaning"] == expected_code.meaning
    assert dumped_code.get("scheme_version", None) == expected_code.scheme_version


def assert_dicom_code_dataset_equals_code(
    code_dataset: Dataset, expected_code: Union[Code, ConceptCode]
):
    assert code_dataset.CodeValue == expected_code.value
    assert code_dataset.CodingSchemeDesignator == expected_code.scheme_designator
    assert code_dataset.CodeMeaning == expected_code.meaning
    assert code_dataset.CodingSchemeVersion == expected_code.scheme_version


def assert_dicom_code_sequence_equals_codes(
    code_sequence: Sequence[Dataset], expected_codes: Sequence[Union[Code, ConceptCode]]
):
    assert len(code_sequence) == len(expected_codes)
    for code_dataset, expected_code in zip(code_sequence, expected_codes):
        assert_dicom_code_dataset_equals_code(code_dataset, expected_code)


def assert_dicom_bool_equals_bool(dicom_bool: str, expected_bool: bool):
    if expected_bool:
        assert dicom_bool == "YES"
    else:
        assert dicom_bool == "NO"


def code_to_code_dataset(code: Code):
    dataset = Dataset()
    dataset.CodeValue = code.value
    dataset.CodingSchemeDesignator = code.scheme_designator
    dataset.CodeMeaning = code.meaning
    dataset.CodingSchemeVersion = code.scheme_version

    return dataset


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
