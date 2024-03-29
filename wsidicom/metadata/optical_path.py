#    Copyright 2023 SECTRA AB
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

"""Optical path model."""

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Optional, Sequence, Type, TypeVar, Union

import numpy as np

from wsidicom.conceptcode import (
    IlluminationCode,
    IlluminationColorCode,
    ImagePathFilterCode,
    LenseCode,
    LightPathFilterCode,
)

LutDataType = Type[Union[np.uint8, np.uint16]]


class LutSegment(metaclass=ABCMeta):
    """Metaclass for a LUT segment."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def array(self, data_type: LutDataType) -> np.ndarray:
        """Return the segment as an array of the given data type."""
        raise NotImplementedError()


@dataclass(frozen=True)
class DiscreteLutSegment(LutSegment):
    """A discrete segmented defined by a sequence of values.

    Note that when serializing, if the next segment is a Linear segment its start value
    will be included in this segment.

    Parameters
    ----------
    values: Sequence[int]
        The discrete values of the segment.
    """

    values: Sequence[int]

    def __len__(self) -> int:
        return len(self.values)

    def array(self, data_type: LutDataType) -> np.ndarray:
        return np.array(self.values, dtype=data_type)


@dataclass(frozen=True)
class LinearLutSegment(LutSegment):
    """A linear segment defined by start and end values and a length.

    Note that the corresponding Dicom Linear Segment Type does not define the start
    value. When serialized to Dicom the start value will be included in the previous
    segment (as last value in a Discrete or Linear segment) and the length reduced by
    one.

    Parameters
    ----------
    start_value: int
        The start value of the linear segment.
    end_value: int
        The end value of the linear segment.
    length: int
        The length of the linear segment.
    """

    start_value: int
    end_value: int
    length: int

    def __len__(self) -> int:
        return self.length

    def array(self, data_type: LutDataType) -> np.ndarray:
        return np.linspace(
            start=self.start_value,
            stop=self.end_value,
            num=self.length,
            dtype=data_type,
        )


@dataclass(frozen=True)
class ConstantLutSegment(LutSegment):
    """A constant segment defined by a value and a length.

    Parameters
    ----------
    value: int
        The value of the constant segment.
    length: int
        The length of the constant segment.
    """

    value: int
    length: int

    def __len__(self) -> int:
        return self.length

    def array(self, data_type: LutDataType) -> np.ndarray:
        return np.full(
            self.length,
            self.value,
            dtype=data_type,
        )


@dataclass(frozen=True)
class Lut:
    """Represents a LUT.

    Parameters
    ----------
    red: Sequence[LutSegment]
        The red segments of the LUT.
    green: Sequence[LutSegment]
        The green segments of the LUT.
    blue: Sequence[LutSegment]
        The blue segments of the LUT.
    data_type: LutDataType
        The data type of the LUT.
    """

    red: Sequence[LutSegment]
    green: Sequence[LutSegment]
    blue: Sequence[LutSegment]
    data_type: LutDataType

    @property
    def bits(self) -> int:
        if self.data_type == np.uint8:
            return 8
        if self.data_type == np.uint16:
            return 16
        raise ValueError("Value type should be 'np.uint8' or np.uint16'.")

    @property
    def table(self) -> np.ndarray:
        return np.stack(
            [
                np.concatenate([segment.array(self.data_type) for segment in self.red]),
                np.concatenate(
                    [segment.array(self.data_type) for segment in self.green]
                ),
                np.concatenate(
                    [segment.array(self.data_type) for segment in self.blue]
                ),
            ]
        )

    @property
    def length(self) -> int:
        length_by_color = set(
            sum(len(segment) for segment in color)
            for color in (self.red, self.green, self.blue)
        )
        if len(length_by_color) != 1:
            raise ValueError("Color components have different length.")
        return length_by_color.pop()

    def array(self, mode: str) -> np.ndarray:
        """Return flattened representation of the lookup table with order
        suitable for use with Pillows point(). The lookup table is scaled to
        either 8 or 16 bit depending on mode.

        Parameters
        ----------
        mode: str
            Image mode to produce lookup table for.

        Returns
        ----------
        np.ndarray
            Lookup table ordered by rgb, rgb ...
        """
        if mode == "L" or mode == "I":
            bits = 16
        else:
            bits = 8
        return self.table.flatten() / (2**self.bits / 2**bits)


OpticalFilterCodeType = TypeVar(
    "OpticalFilterCodeType",
    LightPathFilterCode,
    ImagePathFilterCode,
)
OpticalFilterType = TypeVar("OpticalFilterType", bound="OpticalFilter")


@dataclass(frozen=True)
class OpticalFilter(Generic[OpticalFilterCodeType]):
    """Metaclass for filter conditions for optical path.

    Parameters
    ----------
    filters: Sequence[OpticalFilterCodeType] = []
        The filters used.
    nominal: Optional[float] = None
        The nominal value of the filter in nm.
    low_pass: Optional[float] = None
        The low pass value of the filter in nm.
    high_pass: Optional[float] = None
        The high pass value of the filter in nm.
    """

    filters: Sequence[OpticalFilterCodeType] = field(default_factory=list)
    nominal: Optional[float] = None
    low_pass: Optional[float] = None
    high_pass: Optional[float] = None


class LightPathFilter(OpticalFilter[LightPathFilterCode]):
    """Set of light path filter conditions for optical path.

    Parameters
    ----------
    filters: Sequence[LightPathFilterCode] = []
        The filters used. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8124.html
        or `LightPathFilterCode.meanings` for a list of valid codes.
    nominal: Optional[float] = None
        The nominal value of the filter in nm.
    low_pass: Optional[float] = None
        The low pass value of the filter in nm.
    high_pass: Optional[float] = None
        The high pass value of the filter in nm.
    """


class ImagePathFilter(OpticalFilter[ImagePathFilterCode]):
    """Set of image path filter conditions for optical path.

    Parameters
    ----------
    filters: Sequence[ImagePathFilterCode] = []
        The filters used. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8124.html
        or ImagePathFilterCode.meanings` for a list of valid codes.
    nominal: Optional[float] = None
        The nominal value of the filter in nm.
    low_pass: Optional[float] = None
        The low pass value of the filter in nm.
    high_pass: Optional[float] = None
        The high pass value of the filter in nm.
    """


@dataclass(frozen=True)
class Objectives:
    """Set of lens conditions for optical path.

    Parameters
    ----------
    lenses: Sequence[LenseCode] = []
        Lenses used. See
        https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8121.html
        or `LensCode.meanings` for a list of valid codes.
    condenser_power: Optional[float] = None
        The condenser power.
    objective_power: Optional[float] = None
        The objective power.
    objective_numerical_aperture: Optional[float] = None
        The objective numerical aperture.
    """

    lenses: Sequence[LenseCode] = field(default_factory=list)
    condenser_power: Optional[float] = None
    objective_power: Optional[float] = None
    objective_numerical_aperture: Optional[float] = None


@dataclass(frozen=True)
class OpticalPath:
    """
    Optical path metadata.

    Corresponds to the `Required`, `Required, Empty if Unknown`, and selected
    `Optional` attributes for an Optical Path Sequence item in the Optical Path Module:
    https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.html

    Parameters
    identifier : Optional[str] = None
        Identifier of the optical path.
    description : Optional[str] = None
        Description of the optical path.
    illumination_types : Optional[Sequence[IlluminationCode]] = None
        Illumination types used in the optical path.
    illumination : Optional[Union[float, IlluminationColorCode]] = None
        Illumination used in the optical path.
    icc_profile : Optional[bytes] = None
        ICC profile for the optical path.
    lut : Optional[Lut] = None
        Lookup table to use for the optical path.
    light_path_filter : Optional[LightPathFilter] = None
        Light path filter used in the optical path.
    image_path_filter : Optional[ImagePathFilter] = None
        Image path filter used in the optical path.
    objective : Optional[Objectives] = None
        Objectives used in the optical path.
    """

    identifier: Optional[str] = None
    description: Optional[str] = None
    illumination_types: Optional[Sequence[IlluminationCode]] = None
    illumination: Optional[Union[float, IlluminationColorCode]] = None
    icc_profile: Optional[bytes] = None
    lut: Optional[Lut] = None
    light_path_filter: Optional[LightPathFilter] = None
    image_path_filter: Optional[ImagePathFilter] = None
    objective: Optional[Objectives] = None
