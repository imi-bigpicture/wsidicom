#    Copyright 2021, 2022, 2023 SECTRA AB
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

"""Downsampler for image processing."""

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from wsidicom.config import settings
from wsidicom.geometry import Size
from wsidicom.options import DownsamplerOption, ResampleFilterOption

try:
    import cv2
except ImportError:
    cv2 = None


class Downsampler(ABC):
    """Abstract interface for downsampling images."""

    @classmethod
    def create(
        cls,
        resample: ResampleFilterOption | None = None,
        preferred: DownsamplerOption | None = None,
    ) -> "Downsampler":
        """Create a downsampler that uses a resampling filter.

        Selects an available backend for the filter, honoring the backend
        preference and falling back when a backend does not support the filter.
        Both arguments default to settings when None, so callers (and tests) can
        override the filter or the backend without touching global settings.

        Parameters
        ----------
        resample: ResampleFilterOption | None = None
            Resampling filter to use. Defaults to ``settings.resampling_filter``.
        preferred: DownsamplerOption | None = None
            Backend to prefer. Defaults to ``settings.preferred_downsampler``;
            when that is also None, the fastest available backend is chosen.

        Returns
        -------
        Downsampler
            Downsampler for the filter.
        """
        if resample is None:
            resample = settings.resampling_filter
        if preferred is None:
            preferred = settings.preferred_downsampler
        if preferred != DownsamplerOption.PILLOW:
            cv2_downsampler = Cv2Downsampler.from_filter(resample)
            if cv2_downsampler is not None:
                return cv2_downsampler
        return PillowDownsampler(resample=resample)

    @classmethod
    def create_for_read(cls) -> "Downsampler":
        """Create the downsampler for rescaling on read, using the configured
        read filter (``settings.resampling_filter``)."""
        return cls.create(resample=settings.resampling_filter)

    @classmethod
    def create_for_pyramid(cls) -> "Downsampler":
        """Create the downsampler for generating pyramid levels, using the
        configured pyramid filter (``settings.pyramid_resampling_filter``)."""
        return cls.create(resample=settings.pyramid_resampling_filter)

    @property
    def commutes_with_stitch(self) -> bool:
        """Whether downsample-then-stitch is equivalent to stitch-then-downsample."""
        return False

    @abstractmethod
    def downsample(self, array: np.ndarray, output_size: Size) -> np.ndarray:
        """Downsample an array to the specified output size.

        The resample is the single Pillow (or other backend) boundary; callers
        stay in numpy on both sides.

        Parameters
        ----------
        array: np.ndarray
            Input array to downsample, ``(rows, columns[, samples])``.
        output_size: Size
            Target output size in pixels.

        Returns
        -------
        np.ndarray
            Downsampled array.
        """
        raise NotImplementedError()

    def reduce_by_half(self, array: np.ndarray) -> np.ndarray:
        """Downsample an array by an exact factor of two in each dimension.

        Used by the pyramid write path's reduce-then-stitch fast path (taken
        when ``commutes_with_stitch``). The default resamples to half via
        ``downsample``; backends with a cheaper exact 2x reducer override it.

        Parameters
        ----------
        array: np.ndarray
            Input array, ``(rows, columns[, samples])``.

        Returns
        -------
        np.ndarray
            The array at half size in each spatial dimension.
        """
        half = Size(max(array.shape[1] // 2, 1), max(array.shape[0] // 2, 1))
        return self.downsample(array, half)

    @abstractmethod
    def thumbnail(self, array: np.ndarray, max_size: Size) -> np.ndarray:
        """Resample an array to fit within ``max_size``.

        Unlike ``downsample`` (which resamples to an exact size), this fits the
        array within ``max_size`` while preserving aspect ratio and never
        upscaling.

        Parameters
        ----------
        array: np.ndarray
            Input array to resample, ``(rows, columns[, samples])``.
        max_size: Size
            Upper size limit in pixels.

        Returns
        -------
        np.ndarray
            Resampled array, no larger than ``max_size`` in either dimension.
        """
        raise NotImplementedError()


class PillowDownsampler(Downsampler):
    """Downsampler using Pillow's resize with configurable resampling filter."""

    _PILLOW_FILTERS = {
        ResampleFilterOption.NEAREST: Image.Resampling.NEAREST,
        ResampleFilterOption.BOX: Image.Resampling.BOX,
        ResampleFilterOption.BILINEAR: Image.Resampling.BILINEAR,
        ResampleFilterOption.HAMMING: Image.Resampling.HAMMING,
        ResampleFilterOption.BICUBIC: Image.Resampling.BICUBIC,
        ResampleFilterOption.LANCZOS: Image.Resampling.LANCZOS,
    }

    def __init__(self, resample: ResampleFilterOption = ResampleFilterOption.BILINEAR):
        """Create a Pillow-based downsampler.

        Parameters
        ----------
        resample: ResampleFilterOption
            Resampling filter to use. Default is BILINEAR.
        """
        self._filter = resample
        self._resample = self._PILLOW_FILTERS[resample]

    @property
    def commutes_with_stitch(self) -> bool:
        """Whether downsample-then-stitch equals stitch-then-downsample.

        True only for BOX, whose 2x2 footprint stays within each block, so a
        per-tile `Image.reduce(2)` matches downsampling the whole level. BILINEAR
        does not: Pillow scales the triangle kernel with the factor, so a 2x
        reduce is a ~4-tap triangle that both differs from a box and reaches
        across tile boundaries.
        """
        return self._filter == ResampleFilterOption.BOX

    def downsample(self, array: np.ndarray, output_size: Size) -> np.ndarray:
        resized = self._to_image(array).resize(
            (output_size.width, output_size.height),
            resample=self._resample,
        )
        return self._from_image(resized, array.dtype)

    def reduce_by_half(self, array: np.ndarray) -> np.ndarray:
        if self._filter != ResampleFilterOption.BOX:
            # Only BOX's footprint equals an equal-weighted 2x2 average; other
            # filters scale their kernel with the factor, so a real 2x resize is
            # required rather than Image.reduce(2).
            return super().reduce_by_half(array)
        # Image.reduce(2) is that 2x2 average, matching a real BOX 2x downscale;
        # kept as the write-path fast path so pixels match stitch-then-downsample.
        return self._from_image(self._to_image(array).reduce(2), array.dtype)

    def thumbnail(self, array: np.ndarray, max_size: Size) -> np.ndarray:
        # PIL's Image.thumbnail resamples in place and returns None.
        image = self._to_image(array)
        image.thumbnail((max_size.width, max_size.height), resample=self._resample)
        return self._from_image(image, array.dtype)

    def _to_image(self, array: np.ndarray) -> Image.Image:
        """Wrap an array as a Pillow image, widening 16-bit ``I;16`` to ``I``
        since Pillow cannot resample ``I;16``. ``_from_image`` narrows the
        resampled result back to the input dtype.
        """
        image = Image.fromarray(np.ascontiguousarray(array))
        if image.mode.startswith("I;16"):
            return image.convert("I")
        return image

    @staticmethod
    def _from_image(image: Image.Image, dtype: np.dtype) -> np.ndarray:
        """Convert a resampled image back to an array of the input dtype,
        undoing the ``I;16`` to ``I`` widening ``_to_image`` applies."""
        array = np.asarray(image)
        if array.dtype != dtype:
            # The 16-bit resize ran in Pillow's signed "I" (int32) mode; an
            # overshooting filter (BICUBIC/LANCZOS/HAMMING) can ring outside the
            # target range, so clamp before narrowing to avoid uint wraparound.
            return np.clip(array, 0, np.iinfo(dtype).max).astype(dtype)
        return array


class Cv2Downsampler(Downsampler):
    """Downsampler using OpenCV's resize. Requires the optional ``opencv`` extra."""

    def __init__(self, interpolation: int | None = None):
        """Create an OpenCV-based downsampler.

        Parameters
        ----------
        interpolation: Optional[int]
            OpenCV interpolation flag (e.g. ``cv2.INTER_AREA``). Defaults to
            ``cv2.INTER_AREA``, which is the recommended filter for shrinking.
        """
        if cv2 is None:
            raise ImportError(
                "Cv2Downsampler requires opencv; install the 'opencv' extra."
            )
        self._cv2 = cv2
        self._interpolation = cv2.INTER_AREA if interpolation is None else interpolation

    @classmethod
    def is_available(cls) -> bool:
        """Whether OpenCV is installed."""
        return cv2 is not None

    @classmethod
    def from_filter(cls, resample: ResampleFilterOption) -> "Cv2Downsampler | None":
        """Return a downsampler using cv2's equivalent of a resampling filter, or
        None if opencv is unavailable or the filter has no suitable cv2 match (so
        the caller falls back to Pillow).

        Only NEAREST and BOX qualify: INTER_NEAREST_EXACT reproduces NEAREST
        bit-exactly, and INTER_AREA matches BOX (bit-exact at the 2x pyramid
        reduce). cv2's linear/cubic/lanczos use fixed support and alias when
        shrinking, so they are not stand-ins for Pillow's antialiasing filters.
        """
        if cv2 is None:
            return None
        equivalents = {
            ResampleFilterOption.NEAREST: cv2.INTER_NEAREST_EXACT,
            ResampleFilterOption.BOX: cv2.INTER_AREA,
        }
        interpolation = equivalents.get(resample)
        if interpolation is None:
            return None
        return cls(interpolation)

    @property
    def commutes_with_stitch(self) -> bool:
        """Whether downsample-then-stitch is equivalent to stitch-then-downsample.

        True for INTER_AREA: an exact integer downscale performs pixel-area
        averaging, identical whether applied before or after stitching.
        """
        return self._interpolation == self._cv2.INTER_AREA

    def downsample(self, array: np.ndarray, output_size: Size) -> np.ndarray:
        return self._cv2.resize(
            array,
            (output_size.width, output_size.height),
            interpolation=self._interpolation,
        )

    def thumbnail(self, array: np.ndarray, max_size: Size) -> np.ndarray:
        height, width = array.shape[0], array.shape[1]
        factor = min(max_size.width / width, max_size.height / height, 1.0)
        if factor == 1.0:
            return array
        output_size = Size(
            max(round(width * factor), 1), max(round(height * factor), 1)
        )
        return self.downsample(array, output_size)
