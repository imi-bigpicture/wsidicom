#    Copyright 2021 SECTRA AB
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


class WsiDicomError(Exception):
    """Raised for general error."""

    def __init__(self, error: str):
        self.error = error

    def __str__(self):
        return self.error


class WsiDicomFileError(Exception):
    """Raised if read file data is incorrect."""

    def __init__(self, file: str, error: str):
        self.file = file
        self.error = error

    def __str__(self):
        return f"{self.file}: {self.error}"


class WsiDicomMatchError(Exception):
    """Raised if item in group that should match does not match."""

    def __init__(self, item: str, group: str):
        self.item = item
        self.group = group

    def __str__(self):
        return f"{self.item} does not match {self.group}"


class WsiDicomUidDuplicateError(Exception):
    """Raised if unique UID is encountered twice."""

    def __init__(self, item: str, group: str):
        self.item = item
        self.group = group

    def __str__(self):
        return f"{self.item} is duplicated in {self.group}"


class WsiDicomNotFoundError(Exception):
    """Raised if requested item is not found"""

    def __init__(self, item: str, not_found_in: str):
        self.item = item
        self.not_found_in = not_found_in

    def __str__(self):
        return f"{self.item} not found in {self.not_found_in}"


class WsiDicomOutOfBoundsError(Exception):
    """Raised if requested item is out of bonds"""

    def __init__(self, error: str, bonds: str):
        self.error = error
        self.bonds = bonds

    def __str__(self):
        return f"{self.error} is out of bonds of {self.bonds}"


class WsiDicomStrictRequirementError(Exception):
    """Raised if attribute required in strict mode is missing."""


class WsiDicomRequirementError(Exception):
    """Raised if required attribute is missing."""


class WsiDicomNoResolutionError(Exception):
    """Raised if method is not possible as resolution is missing in image
    data."""


class WsiDicomNotSupportedError(Exception):
    """Raised if opened instance is not supported."""


class WsiDicomBotOverflow(Exception):
    """Raised if image data is to large to fit into basic table offset."""
