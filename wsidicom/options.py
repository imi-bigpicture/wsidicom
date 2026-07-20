#    Copyright 2026 SECTRA AB
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

"""User-facing options for controlling how WsiDicom works."""

from dataclasses import dataclass
from enum import Enum, Flag, auto
from typing import TypeVar

OptionT = TypeVar("OptionT", bound="Option")


class Option(Enum):
    """Base for settings option enums, naming a choice by its string value."""

    @classmethod
    def coerce(cls: type[OptionT], value: "OptionT | str | None") -> "OptionT | None":
        """Coerce ``value`` to a member: a member or its string value (e.g.
        ``"pillow"``) returns the member, ``None`` passes through, and an unknown
        string raises ``ValueError``.
        """
        return None if value is None else cls(value)


class ResampleFilterOption(Option):
    """Backend-neutral resampling filter, mapped by each downsampler to its
    native equivalent."""

    NEAREST = "nearest"
    """Does not average, so it aliases when shrinking; use only where exact source
    pixel values matter (e.g. label or mask images)."""
    BOX = "box"
    """Equal-weighted area average. Exact and ringing-free but the crudest
    antialiasing filter; the pyramid-generation default."""
    BILINEAR = "bilinear"
    """Linear (triangle) kernel; the read default. Antialiases without the ringing
    of BICUBIC and LANCZOS."""
    HAMMING = "hamming"
    """Windowed kernel between BOX and BILINEAR; sharper than BILINEAR without
    BOX's local dislocations."""
    BICUBIC = "bicubic"
    """Cubic kernel; sharper than BILINEAR but can overshoot and ring at
    high-contrast edges."""
    LANCZOS = "lanczos"
    """Truncated-sinc kernel; the sharpest filter, with the most ringing."""


class DownsamplerOption(Option):
    """Downsampler backend to use when rescaling images."""

    OPENCV = "opencv"
    PILLOW = "pillow"


class DecoderOption(Option):
    """Decoder to prefer to use when decoding image frames."""

    IMAGECODECS = "imagecodecs"
    PYLIBJPEG_OPENJPEG = "pylibjpeg_openjpeg"
    PILLOW = "pillow"
    PYLIBJPEG_RLE = "pylibjpeg_rle"
    IMAGECODECS_RLE = "imagecodecs_rle"
    PYLIBJPEG_LS = "pylibjpeg_ls"
    PYDICOM = "pydicom"


class InstanceSplit(Flag):
    """Controls how optical paths and focal planes are split across written instances.

    By default all optical paths and focal planes of a pyramid level (or group)
    are combined into a single instance. Flags can be combined, e.g.
    ``InstanceSplit.FOCAL_PLANE | InstanceSplit.OPTICAL_PATH`` writes one instance
    per (focal plane, optical path) pair.
    """

    NONE = 0
    """All optical paths and focal planes in one instance (default)."""
    FOCAL_PLANE = auto()
    """Write a separate instance per focal plane."""
    OPTICAL_PATH = auto()
    """Write a separate instance per optical path."""


@dataclass(frozen=True)
class ConcatenationByFrames:
    """Split each level into concatenated instances of at most `count` frames."""

    count: int


@dataclass(frozen=True)
class ConcatenationByBytes:
    """Split each level into concatenated instances whose encapsulated pixel data
    is at most `count` bytes. The dataset header is not counted, so set `count` below
    any hard limit.
    """

    count: int
