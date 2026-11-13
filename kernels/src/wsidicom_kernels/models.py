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

"""The public enum accepted by wsidicom_kernels.

``ResizeFilter`` is a ``StrEnum``, so a member is itself the wire value the
Rust extension expects; the public functions also accept the raw strings.
"""

from enum import StrEnum


class ResizeFilter(StrEnum):
    """The filter source pixels are weighted with when resizing.

    Taken by :func:`~wsidicom_kernels.downscale` and
    :func:`~wsidicom_kernels.resize`. Its support grows with the reduction
    ratio, so when shrinking each output pixel averages the source area it
    covers (antialiasing).
    """

    NEAREST = "nearest"
    """One source pixel per output pixel, no averaging; aliases when shrinking.
    For label and mask images."""
    BOX = "box"
    """Equal-weighted area average; the fastest, and a native SIMD kernel at a
    power-of-2 factor. Exact and ringing-free, the crudest antialiasing."""
    BILINEAR = "bilinear"
    """Linear (triangle) kernel; smooth, without the ringing of CATMULLROM and
    LANCZOS3."""
    CATMULLROM = "catmullrom"
    """Catmull-Rom cubic kernel; sharper than BILINEAR, with overshoot and
    ringing at high-contrast edges."""
    LANCZOS3 = "lanczos3"
    """Three-lobe windowed sinc; the sharpest, with the most ringing."""
