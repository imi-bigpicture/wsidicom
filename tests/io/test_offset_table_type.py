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

import pytest
from wsidicom.file.io.frame_index.offset_table_type import OffsetTableType


@pytest.mark.parametrize(
    ["string", "expected_result"],
    [
        ("BOT", OffsetTableType.BASIC),
        ("bot", OffsetTableType.BASIC),
        ("EOT", OffsetTableType.EXTENDED),
        ("eot", OffsetTableType.EXTENDED),
        ("EMPTY", OffsetTableType.EMPTY),
        ("empty", OffsetTableType.EMPTY),
        ("NONE", OffsetTableType.NONE),
        ("none", OffsetTableType.NONE),
    ],
)
class TestOffsetTableType:
    def test_from_string(self, string: str, expected_result: OffsetTableType):
        # Arrange
        # Act
        result = OffsetTableType.from_string(string)

        # Assert
        assert result == expected_result
