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

"""Option enums naming the choices that settings select between."""

from enum import Enum
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
