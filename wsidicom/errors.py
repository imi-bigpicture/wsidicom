from pathlib import Path
from .geometry import Point


class WsiDicomFileError(Exception):
    """Raised if read file data is incorrect."""

    def __init__(self, filepath: Path, error: str):
        self.filepath = filepath
        self.error = error

    def __str__(self):
        return f"{self.filepath}: {self.error}"


class WsiDicomMatchError(Exception):
    """Raised if item in group that should match doesnt match."""

    def __init__(self, item: str, group: str):
        self.item = item
        self.group = group

    def __str__(self):
        return f"{self.item} doesnt match {self.group}"


class WsiDicomUidDuplicateError(Exception):
    """Raised if unique UID is encountered twice."""

    def __init__(self, item: str, group: str):
        self.item = item
        self.group = group

    def __str__(self):
        return f"{self.item} is duplicated in {self.group}"


class WsiDicomSparse(Exception):
    """Raised for sparsed index if requested frame is valid but not in
    pixel data.
    """

    def __init__(self, tile: Point):
        self.tile = tile

    def __str__(self):
        return f"{self.tile} not found in pixel data"


class WsiDicomNotFoundError(Exception):
    """Raised if requested item is not found"""

    def __init__(self, item: str, not_found_in: str):
        self.item = item
        self.not_found_in = not_found_in

    def __str__(self):
        return f"{self.item} not found in {self.not_found_in}"


class WsiDicomOutOfBondsError(Exception):
    """Raised if requested item is out of bonds"""

    def __init__(self, error: str, bonds: str):
        self.error = error
        self.bonds = bonds

    def __str__(self):
        return f"{self.error} is out of bonds of {self.bonds}"
