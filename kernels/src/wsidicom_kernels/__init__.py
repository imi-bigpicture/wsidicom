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

"""Fast, accurate image resizing and pyramid downscaling.

Operates on planar ``(H, W)`` or interleaved RGB ``(H, W, 3)`` arrays, dtype
uint8 / uint16. Subsampled chroma (YCbCr 4:2:0) is resized per plane.

The implementation lives in :mod:`wsidicom_kernels.core` over the PyO3 extension
:mod:`wsidicom_kernels._kernels`; this module only re-exports the public API.
"""

from importlib.metadata import version

from wsidicom_kernels.core import (
    downscale,
    downscale_grid,
    resize,
)
from wsidicom_kernels.models import ResizeFilter

__version__ = version("wsidicom-kernels")

__all__ = [
    "ResizeFilter",
    "downscale",
    "downscale_grid",
    "resize",
]
