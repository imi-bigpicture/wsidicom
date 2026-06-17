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

from PIL import Image

from wsidicom.geometry import Size


class Downsampler(ABC):
    """Abstract interface for downsampling images."""

    @property
    def commutes_with_stitch(self) -> bool:
        """Whether downsample-then-stitch is equivalent to stitch-then-downsample."""
        return False

    @abstractmethod
    def downsample(self, image: Image.Image, output_size: Size) -> Image.Image:
        """Downsample an image to the specified output size.

        Parameters
        ----------
        image: Image.Image
            Input image to downsample.
        output_size: Size
            Target output size in pixels.

        Returns
        -------
        Image.Image
            Downsampled image.
        """
        raise NotImplementedError()

    @abstractmethod
    def thumbnail(self, image: Image.Image, max_size: Size) -> Image.Image:
        """Resample an image to fit within ``max_size``.

        Unlike ``downsample`` (which resamples to an exact size), this fits the
        image within ``max_size`` while preserving aspect ratio and never
        upscaling.

        The input ``image`` may be modified in place; callers should use the
        returned image and not reuse the argument afterwards.

        Parameters
        ----------
        image: Image.Image
            Input image to resample. May be modified in place.
        max_size: Size
            Upper size limit in pixels.

        Returns
        -------
        Image.Image
            Resampled image, no larger than ``max_size`` in either dimension.
        """
        raise NotImplementedError()


class PillowDownsampler(Downsampler):
    """Downsampler using Pillow's resize with configurable resampling filter."""

    def __init__(self, resample: Image.Resampling = Image.Resampling.BILINEAR):
        """Create a Pillow-based downsampler.

        Parameters
        ----------
        resample: Image.Resampling
            Resampling filter to use. Default is BILINEAR.
        """
        self._resample = resample

    @property
    def commutes_with_stitch(self) -> bool:
        """Whether downsample-then-stitch is equivalent to stitch-then-downsample.

        True for BOX and BILINEAR resampling filters. For these filters, an
        exact 2x downscale produces equal-weighted 2x2 averaging, which
        `Image.reduce(2)` computes identically.
        """
        return self._resample in (
            Image.Resampling.BOX,
            Image.Resampling.BILINEAR,
        )

    def downsample(self, image: Image.Image, output_size: Size) -> Image.Image:
        return image.resize(
            (output_size.width, output_size.height),
            resample=self._resample,
        )

    def thumbnail(self, image: Image.Image, max_size: Size) -> Image.Image:
        # PIL's Image.thumbnail resamples in place and returns None.
        image.thumbnail((max_size.width, max_size.height), resample=self._resample)
        return image
