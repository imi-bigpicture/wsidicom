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

"""Fast tile compositing and downscaling.

Operates on planar ``(H, W)`` or interleaved HWC ``(H, W, C)`` arrays, dtype
uint8 / uint16 / float32. Planar is the universal core — the only layout that
expresses subsampled chroma (YCbCr 4:2:0), so compose/downscale those per plane.
Overlaps in :func:`compose` resolve last-write-wins (no blending).

The implementation lives in :mod:`wsidicom_kernels.core` over the PyO3 extension
:mod:`wsidicom_kernels._kernels`; this module only re-exports the public API.
"""

from importlib.metadata import version

from wsidicom_kernels.core import (
    compose,
    downscale,
    downscale_grid,
    resize,
    rgb_to_rgbx,
    stitch,
)
from wsidicom_kernels.models import Interpolation, ResizeFilter

__version__ = version("wsidicom-kernels")

__all__ = [
    "Interpolation",
    "ResizeFilter",
    "compose",
    "downscale",
    "downscale_grid",
    "resize",
    "rgb_to_rgbx",
    "stitch",
]
