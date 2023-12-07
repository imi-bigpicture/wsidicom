#    Copyrigh 2023 SECTRA AB
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

from enum import Enum


class OffsetTableType(Enum):
    """Offset table type."""

    NONE = "none"
    EMPTY = "empty"
    BASIC = "BOT"
    EXTENDED = "EOT"

    @classmethod
    def from_string(cls, offset_table: str) -> "OffsetTableType":
        """Return OffsetTableType parsed from string."""
        if offset_table.strip().lower() == "none":
            return OffsetTableType.NONE
        if offset_table.strip().lower() == "empty":
            return OffsetTableType.EMPTY
        if offset_table.strip().lower() == "eot":
            return OffsetTableType.EXTENDED
        if offset_table.strip().lower() == "bot":
            return OffsetTableType.BASIC
        raise ValueError(f"Unknown offset table type: {offset_table}")
