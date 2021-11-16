from pathlib import Path


class WsiDicomError(Exception):
    """Raised for general error."""

    def __init__(self, error: str):
        self.error = error

    def __str__(self):
        return self.error


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
