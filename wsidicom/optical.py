#    Copyright 2021 SECTRA AB
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

import logging
import struct
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL.Image import Image as PILImage
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from wsidicom.conceptcode import (
    ChannelDescriptionCode,
    ConceptCode,
    IlluminationCode,
    IlluminationColorCode,
    IlluminatorCode,
    ImagePathFilterCode,
    LenseCode,
    LightPathFilterCode,
)
from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.instance import WsiInstance


class Lut:
    """Represents a LUT."""

    def __init__(self, lut_sequence: DicomSequence):
        """Read LUT from a DICOM LUT sequence.

        Parameters
        ----------
        size: int
            the number of entries in the table
        bits: int
            the bits for each entry (currently forced to 16)
        """
        self._lut_item = lut_sequence[0]
        length, first, bits = self._lut_item.RedPaletteColorLookupTableDescriptor
        self._length = length
        self._bits = bits
        if bits == 8:
            self._type = np.dtype(np.uint8)
        else:
            self._type = np.dtype(np.uint16)
        self._byte_format = "HHH"  # Do we need to set endianness?
        self.table = self._parse_lut(self._lut_item)

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
        return self.table.flatten() / (2**self._bits / 2**bits)

    def insert_into_ds(self, ds: Dataset) -> Dataset:
        """Codes and insert object into sequence in dataset.

        Parameters
        ----------
        ds: Dataset
           Dataset to insert into.

        Returns
        ----------
        Dataset
            Dataset with object inserted.

        """
        ds.PaletteColorLookupTableSequence = DicomSequence([self._lut_item])
        return ds

    @classmethod
    def from_ds(cls, ds: Dataset) -> Optional["Lut"]:
        if "PaletteColorLookupTableSequence" in ds:
            try:
                return cls(ds.PaletteColorLookupTableSequence)
            except Exception:
                logging.error("Failed to parse LUT", exc_info=True)
                return None
        return None

    def get(self) -> np.ndarray:
        """Return 2D representation of the lookup table.

        Returns
        ----------
        np.ndarray
            Lookup table ordered by color x entry
        """
        return self.table

    def _parse_color(self, segmented_lut_data: bytes):
        LENGTH = 6
        parsed_table = np.ndarray((0,), dtype=self._type)
        for segment in range(int(len(segmented_lut_data) / LENGTH)):
            segment_bytes = segmented_lut_data[
                segment * LENGTH : segment * LENGTH + LENGTH
            ]
            lut_type, lut_length, lut_value = struct.unpack(
                self._byte_format, segment_bytes
            )
            if lut_type == 0:
                parsed_table = self._add_discret(parsed_table, lut_length, lut_value)
            elif lut_type == 1:
                parsed_table = self._add_linear(parsed_table, lut_length, lut_value)
            else:
                raise NotImplementedError("Unknown lut segment type")
        return parsed_table

    def _parse_lut(self, lut: Dataset) -> np.ndarray:
        """Parse a dicom Palette Color Lookup Table Sequence item.

        Parameters
        ----------
        lut: Dataset
            A Palette Color Lookup Table Sequence item
        """
        tables = [
            lut.SegmentedRedPaletteColorLookupTableData,
            lut.SegmentedGreenPaletteColorLookupTableData,
            lut.SegmentedBluePaletteColorLookupTableData,
        ]
        parsed_tables = np.zeros((len(tables), self._length), dtype=self._type)

        for color, table in enumerate(tables):
            parsed_tables[color] = self._parse_color(table)
        return parsed_tables

    @classmethod
    def _insert(cls, table: np.ndarray, segment: np.ndarray):
        """Insert a segment into the lookup table of channel.

        Parameters
        ----------
        channel: int
            The channel (r=0, g=1, b=2) to operate on
        segment: np.ndarray
            The segment to insert
        """
        table = np.append(table, segment)
        return table

    @classmethod
    def _add_discret(cls, table: np.ndarray, length: int, value: int):
        """Add a discret segment into the lookup table of channel.

        Parameters
        ----------
        channel: int
            The channel (r=0, g=1, b=2) to operate on
        length: int
            The length of the discret segment
        value: int
            The value of the deiscret segment
        """
        segment = np.full(length, value, dtype=table.dtype)
        table = cls._insert(table, segment)
        return table

    @classmethod
    def _add_linear(cls, table: np.ndarray, length: int, value: int):
        """Add a linear segment into the lookup table of channel.

        Parameters
        ----------
        channel: int
            The channel (r=0, g=1, b=2) to operate on
        length: int
            The length of the discret segment
        value: int
            The value of the deiscret segment
        """
        # Default shift segment by one to not include first value
        # (same as last value)
        start_position = 1
        # If no last value, set it to 0 and include
        # first value in segment
        try:
            last_value = table[-1]
        except IndexError:
            last_value = 0
            start_position = 0
        segment = np.linspace(
            start=last_value, stop=value, num=start_position + length, dtype=table.dtype
        )
        table = cls._insert(table, segment[start_position:])
        return table


@dataclass
class OpticalFilter(metaclass=ABCMeta):
    """Metaclass for filter conditions for optical path"""

    filters: Optional[List[ConceptCode]]
    nominal: Optional[float]
    low_pass: Optional[float]
    high_pass: Optional[float]

    @classmethod
    @abstractmethod
    def from_ds(cls, ds: Dataset) -> "OpticalFilter":
        raise NotImplementedError()

    def insert_into_ds(self, ds: Dataset) -> Dataset:
        """Codes and insert object into dataset.

        Parameters
        ----------
        ds: Dataset
           Optical path sequence item.

        Returns
        ----------
        Dataset
            Dataset with object inserted.

        """
        if self.nominal is not None:
            ds.LightPathFilterPassBand = self.nominal
        if self.low_pass is not None and self.high_pass is not None:
            ds.LightPathFilterPassThroughwavelength = [self.low_pass, self.high_pass]
        if self.filters is not None:
            for filter in self.filters:
                ds = filter.insert_into_ds(ds)
        return ds


@dataclass
class LightPathFilter(OpticalFilter):
    """Set of light path filter conditions for optical path"""

    filters: Optional[List[LightPathFilterCode]]

    @classmethod
    def from_ds(cls, ds: Dataset) -> "LightPathFilter":
        """Returns LightPathFilter object read from dataset
        (optical path sequence item).

        Parameters
        ----------
        ds: Dataset
           Optical path sequence item.

        Returns
        ----------
        LightPathFilter
            Object containing light path filter conditions for optical path.

        """
        filter_band = getattr(ds, "LightPathFilterPassBand", [None, None])
        return cls(
            filters=LightPathFilterCode.from_ds(ds),
            nominal=getattr(ds, "LightPathFilterPassThroughwavelengthh", None),
            low_pass=filter_band[0],
            high_pass=filter_band[1],
        )


@dataclass
class ImagePathFilter(OpticalFilter):
    """Set of image path filter conditions for optical path"""

    filters: Optional[List[ImagePathFilterCode]]

    @classmethod
    def from_ds(cls, ds: Dataset) -> "ImagePathFilter":
        """Returns ImagePathFilter object read from dataset
        (optical path sequence item).

        Parameters
        ----------
        ds: Dataset
           Optical path sequence item.

        Returns
        ----------
        ImagePathFilter
            Object containing image path filter conditions for optical path.

        """
        filter_band = getattr(ds, "ImagePathFilterPassBand", [None, None])
        return cls(
            filters=ImagePathFilterCode.from_ds(ds),
            nominal=getattr(ds, "ImagePathFilterPassThroughwavelengthh", None),
            low_pass=filter_band[0],
            high_pass=filter_band[1],
        )


@dataclass
class Illumination:
    """Set of illumination conditions for optical path"""

    def __init__(
        self,
        illumination_method: Optional[Sequence[IlluminationCode]] = None,
        illumination_wavelength: Optional[float] = None,
        illumination_color: Optional[IlluminationColorCode] = None,
        illuminator: Optional[IlluminatorCode] = None,
    ):
        # if illumination_color is None and illumination_wavelength is None:
        #     raise ValueError("Illumination color or wavelength need to be set")
        if illumination_method is None:
            illumination_method = []
        self.illumination_method = illumination_method
        self.illumination_wavelength = illumination_wavelength
        self.illumination_color = illumination_color
        self.illuminator = illuminator

    @classmethod
    def from_ds(cls, ds: Dataset) -> "Illumination":
        """Returns Illuminatin object read from dataset (optical path sequence
        item).

        Parameters
        ----------
        ds: Dataset
           Optical path sequence item.

        Returns
        ----------
        Illumination
            Object containing illumination conditions for optical path.

        """
        return cls(
            illumination_method=IlluminationCode.from_ds(ds),
            illumination_wavelength=getattr(ds, "IlluminationWaveLength", None),
            illumination_color=IlluminationColorCode.from_ds(ds),
            illuminator=IlluminatorCode.from_ds(ds),
        )

    def insert_into_ds(self, ds: Dataset) -> Dataset:
        """Codes and insert object into dataset.

        Parameters
        ----------
        ds: Dataset
           Optical path sequence item.

        Returns
        ----------
        Dataset
            Dataset with object inserted.

        """
        if self.illumination_wavelength is not None:
            ds.Illuminationwavelengthh = self.illumination_wavelength
        if self.illumination_color is not None:
            ds = self.illumination_color.insert_into_ds(ds)
        if self.illuminator is not None:
            ds = self.illuminator.insert_into_ds(ds)
        for item in self.illumination_method:
            ds = item.insert_into_ds(ds)
        return ds


@dataclass
class Lenses:
    """Set of lens conditions for optical path"""

    lenses: Optional[List[LenseCode]]
    condenser_power: Optional[float]
    objective_power: Optional[float]
    objective_na: Optional[float]

    @classmethod
    def from_ds(cls, ds: Dataset) -> "Lenses":
        """Returns Lenses object read from dataset (optical path sequence
        item).

        Parameters
        ----------
        ds: Dataset
           Optical path sequence item.

        Returns
        ----------
        Lenses
            Object containing lense conditions for optical path.

        """
        return cls(
            lenses=LenseCode.from_ds(ds),
            condenser_power=getattr(ds, "CondenserLensPower", None),
            objective_power=getattr(ds, "ObjectiveLensPower", None),
            objective_na=getattr(ds, "ObjectiveLensNumericalAperture", None),
        )

    def insert_into_ds(self, ds: Dataset) -> Dataset:
        """Codes and insert object into dataset.

        Parameters
        ----------
        ds: Dataset
           Optical path sequence item.

        Returns
        ----------
        Dataset
            Dataset with object inserted.

        """
        if self.condenser_power is not None:
            ds.CondenserLensPower = self.condenser_power
        if self.objective_power is not None:
            ds.ObjectiveLensPower = self.objective_power
        if self.objective_na is not None:
            ds.ObjectiveLensNumericalAperture = self.objective_na
        if self.lenses is not None:
            for lense in self.lenses:
                ds = lense.insert_into_ds(ds)
        return ds


@dataclass
class OpticalPath:
    """Represents an optical path"""

    def __init__(
        self,
        identifier: str,
        illumination: Illumination,
        photometric_interpretation: str,
        description: Optional[str] = None,
        icc_profile: Optional[bytes] = None,
        lut: Optional[Lut] = None,
        light_path_filter: Optional[LightPathFilter] = None,
        image_path_filter: Optional[ImagePathFilter] = None,
        channel_description: Optional[Sequence[ChannelDescriptionCode]] = None,
        lenses: Optional[Lenses] = None,
    ):
        """Create a OpticalPath from identifier, illumination, photometric
        interpretation, and optional attributes.

        Parameters
        ----------
        identifier: str
            String identifier for the optical path.
        illumination: Illumination
            The illumination condition used in the optical path.
        photometric_interpretation: str
            The photometric interpretation of the optical path.
        description: Optional[str] = None
            Optional description of the optical path.
        icc_profile: Optional[bytes] = None
            Optional ICC profile for the optical path.
        lut: Optional[Lut] = None
            Optional Look-up table for the optical path.
        light_path_filter: Optional[LightPathFilter] = None
            Optional light path filter description for the optical path.
        image_path_filter: Optional[ImagePathFilter] = None
            Optional image path filter description for the optical path.
        channel_description: Optional[Sequence[ChannelDescriptionCode]] = None
            Optional channel description for the optical path.
        lenses: Optional[Lenses] = None
            Optional lens description for the optical path.
        """

        self.identifier = identifier
        self.illumination = illumination
        self.description = description
        self.icc_profile = icc_profile
        self.lut = lut
        self.light_path_filter = light_path_filter
        self.image_path_filter = image_path_filter
        self.channel_description = channel_description
        self.lenses = lenses

    def __str__(self):
        string = self.identifier
        if self.description is not None:
            string += " - " + self.description
        return string

    def to_ds(self) -> Dataset:
        ds = Dataset()
        ds.OpticalPathIdentifier = self.identifier
        ds = self.illumination.insert_into_ds(ds)
        if self.description is not None:
            ds.OpticalPathDescription = self.description
        if self.icc_profile is not None:
            ds.ICCProfile = self.icc_profile
        if self.lut is not None:
            ds = self.lut.insert_into_ds(ds)
        if self.light_path_filter is not None:
            ds = self.light_path_filter.insert_into_ds(ds)
        if self.image_path_filter is not None:
            ds = self.image_path_filter.insert_into_ds(ds)
        if self.channel_description is not None:
            for item in self.channel_description:
                ds = item.insert_into_ds(ds)
        if self.lenses is not None:
            ds = self.lenses.insert_into_ds(ds)
        return ds

    @classmethod
    def from_ds(cls, ds: Dataset, photometric_interpretation: str) -> "OpticalPath":
        """Create new optical path item populated with optical path
        identifier, description, icc profile name and lookup table.

        Parameters
        ----------
        optical_path: Dataset
            Optical path dataset containing the optical path data
        photometric_interpretation: str
            Photometric interprentation for parent dataset.

        Returns
        ----------
        OpticalPath
            New optical path item
        """
        return OpticalPath(
            identifier=str(ds.OpticalPathIdentifier),
            illumination=Illumination.from_ds(ds),
            photometric_interpretation=photometric_interpretation,
            description=getattr(ds, "OpticalPathDescription", None),
            icc_profile=getattr(ds, "ICCProfile", None),
            lut=Lut.from_ds(ds),
            light_path_filter=LightPathFilter.from_ds(ds),
            image_path_filter=ImagePathFilter.from_ds(ds),
            channel_description=ChannelDescriptionCode.from_ds(ds),
            lenses=Lenses.from_ds(ds),
        )


class OpticalManager:
    """Store optical paths loaded from dicom files."""

    def __init__(
        self,
        optical_paths: Optional[Sequence[OpticalPath]] = None,
    ):
        """Create a OpticalManager from list of OpticalPaths.

        Parameters
        ----------
         optical_paths: Optional[Sequence[OpticalPath]] = None
            List of OpticalPaths.
        """
        if optical_paths is None:
            self._optical_paths = {}
        else:
            self._optical_paths: Dict[str, OpticalPath] = {
                optical_path.identifier: optical_path for optical_path in optical_paths
            }

    @classmethod
    def open(cls, instances: Sequence[WsiInstance]) -> "OpticalManager":
        """Parse optical path sequence in listed instances and create an
        OpticalManager out of the found (unique) OpticalPaths.

        Parameters
        ----------
        files: Sequence[WsiInstance]
            List of WsiDicom instances to parse

        Returns
        ----------
        OpticalManager
            OpticalManager for the found OpticalPaths
        """
        optical_paths: Dict[str, OpticalPath] = {}
        for instance in instances:
            for dataset in instance.datasets:
                optical_path_sequence = dataset.optical_path_sequence
                if optical_path_sequence is None:
                    continue
                for optical_path in optical_path_sequence:
                    identifier = str(optical_path.OpticalPathIdentifier)
                    if identifier not in optical_paths:
                        path = OpticalPath.from_ds(
                            optical_path, dataset.PhotometricInterpretation
                        )
                        optical_paths[identifier] = path
        return cls(list(optical_paths.values()))

    def insert_into_ds(self, ds: Dataset) -> Dataset:
        """Codes and insert object into dataset.

        Parameters
        ----------
        ds: Dataset
           DICOM dataset.

        Returns
        ----------
        Dataset
            Dataset with object inserted.

        """
        ds.NumberOfOpticalPaths = len(self._optical_paths)
        ds.OpticalPathSequence = DicomSequence(
            [optical_path.to_ds() for optical_path in self._optical_paths.values()]
        )
        return ds

    def get(self, identifier: str) -> OpticalPath:
        try:
            return self._optical_paths[identifier]
        except KeyError:
            raise WsiDicomNotFoundError(
                f"identifier {identifier}", "optical path manager"
            )

    def apply_lut(self, image: PILImage, identifier: str) -> PILImage:
        """Apply LUT of identifier to image. Converts gray scale image to RGB.

        Parameters
        ----------
        image: PILImage
            Pillow image to apply LUT to.
        identifier: str
            The identifier of the LUT to apply

        Returns
        ----------
        Image
            Image with LUT applied.
        """

        path = self.get(identifier)
        lut = path.lut
        if lut is None:
            raise WsiDicomNotFoundError(
                f"Lut for identifier {identifier}", "optical path manager"
            )
        # if(image.mode == 'L'):
        #     image = image.convert('RGB')
        return image.point(lut.array(image.mode))
