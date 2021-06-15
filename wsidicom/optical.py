import struct
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union

import numpy as np
from PIL import Image
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from .conceptcode import (ChannelDescriptionCode, IlluminationCode,
                          IlluminationColorCode, IlluminatorCode,
                          ImagePathFilterCode, LenseCode, LightPathFilterCode)
from .errors import WsiDicomNotFoundError
from .file import WsiDicomFile


class Lut:
    def __init__(self, lut_sequence: DicomSequence):
        """Stores RGB lookup tables.

        Parameters
        ----------
        size: int
            the number of entries in the table
        bits: int
            the bits for each entry (currently forced to 16)
        """
        self._lut_item = lut_sequence[0]
        length, first, bits = \
            self._lut_item.RedPaletteColorLookupTableDescriptor
        self._length = length
        self._type: type
        self._bits = bits
        if bits == 8:
            self._type = np.uint8
        else:
            self._type = np.uint16
        self._byte_format = 'HHH'  # Do we need to set endianess?
        self.table = self._parse_lut(self._lut_item)

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
        ds.PaletteColorLookupTableSequence = DicomSequence[self._lut_item]

    @classmethod
    def from_ds(cls, ds: Dataset) -> Optional['Lut']:
        if('PaletteColorLookupTableSequence' in ds):
            return cls(ds.PaletteColorLookupTableSequence)
        return None

    def get(self) -> np.ndarray:
        """Return 2D representation of the lookup table.

        Returns
        ----------
        np.ndarray
            Lookup table ordered by color x entry
        """
        return self.table

    def get_flat(self) -> np.ndarray:
        """Return 1D representation of the lookup table.
        Suitable for use with pillows point function.

        Returns
        ----------
        np.ndarray
            Lookup table ordered by rgb, rgb ...
        """
        return self.table.flatten()

    def _parse_color(self, segmented_lut_data: bytes):
        LENGTH = 6
        parsed_table = np.ndarray((0, ), dtype=self._type)
        for segment in range(int(len(segmented_lut_data)/LENGTH)):
            segment_bytes = segmented_lut_data[
                segment*LENGTH:segment*LENGTH+LENGTH
            ]
            lut_type, lut_length, lut_value = struct.unpack(
                self._byte_format,
                segment_bytes
            )
            if(lut_type == 0):
                parsed_table = self._add_discret(
                    parsed_table,
                    lut_length,
                    lut_value
                )
            elif(lut_type == 1):
                parsed_table = self._add_linear(
                    parsed_table,
                    lut_length,
                    lut_value
                )
            else:
                raise NotImplementedError("Unkown lut segment type")
        return parsed_table

    def _parse_lut(self, lut: DicomSequence) -> np.ndarray:
        """Parse a dicom Palette Color Lookup Table Sequence item.

        Parameters
        ----------
        lut: DicomSequence
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
        """Insert a segement into the lookup table of channel.

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
        """Add a discret segement into the lookup table of channel.

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
        """Add a linear segement into the lookup table of channel.

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
            start=last_value,
            stop=value,
            num=start_position+length,
            dtype=table.dtype
        )
        table = cls._insert(table, segment[start_position:])
        return table


@dataclass
class OpticalFilter(metaclass=ABCMeta):
    filters: Optional[Union[LightPathFilterCode, ImagePathFilterCode]]
    nominal: Optional[float]
    low_pass: Optional[float]
    high_pass: Optional[float]

    @classmethod
    @abstractmethod
    def from_ds(cls, ds: Dataset):
        raise NotImplementedError

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
            ds.LightPathFilterPassThroughWavelength = [
                self.low_pass,
                self.high_pass
            ]
        if self.filters is not None:
            for filter in self.filters:
                ds = filter.insert_into_ds(ds)
        return ds


@dataclass
class LightPathFilter(OpticalFilter):
    filters: Optional[List[LightPathFilterCode]]

    @classmethod
    def from_ds(cls, ds: Dataset) -> 'LightPathFilter':
        filter_band = getattr(ds, 'LightPathFilterPassBand', [None, None])
        return cls(
            filters=LightPathFilterCode.from_ds(ds),
            nominal=getattr(ds, 'LightPathFilterPassThroughWavelength', None),
            low_pass=filter_band[0],
            high_pass=filter_band[1]
        )


@dataclass
class ImagePathFilter(OpticalFilter):
    filters: Optional[List[ImagePathFilterCode]]

    @classmethod
    def from_ds(cls, ds: Dataset) -> 'ImagePathFilter':
        filter_band = getattr(ds, 'ImagePathFilterPassBand', [None, None])
        return cls(
            filters=ImagePathFilterCode.from_ds(ds),
            nominal=getattr(ds, 'ImagePathFilterPassThroughWavelength', None),
            low_pass=filter_band[0],
            high_pass=filter_band[1]
        )


@dataclass
class Illumination:
    """Set of illumination conditions for optical path"""
    illumination_method: List[IlluminationCode]
    illumination_wavelengt: Optional[float]
    illumination_color: Optional[IlluminationColorCode]
    illuminator: Optional[IlluminatorCode]

    @classmethod
    def from_ds(cls, ds: Dataset) -> 'Illumination':
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
            illumination_wavelengt=getattr(ds, 'IlluminationWaveLength', None),
            illumination_color=IlluminationColorCode.from_ds(ds),
            illuminator=IlluminatorCode.from_ds(ds)
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
        if self.illumination_wavelengt is not None:
            ds.IlluminationWaveLength = self.illumination_wavelengt
        if self.illumination_color is not None:
            ds = self.illumination_color.insert_into_ds(ds)
        if self.illuminator is not None:
            ds = self.illuminator.insert_into_ds(ds)
        for item in self.illumination_method:
            ds = item.insert_into_ds(ds)
        return ds


@dataclass
class Lenses:
    lenses: Optional[List[LenseCode]]
    condenser_power: Optional[float]
    objective_power: Optional[float]
    objective_na: Optional[float]

    @classmethod
    def from_ds(cls, ds: Dataset) -> 'Lenses':
        return cls(
            lenses=LenseCode.from_ds(ds),
            condenser_power=getattr(ds, 'CondenserLensPower', None),
            objective_power=getattr(ds, 'ObjectiveLensPower', None),
            objective_na=getattr(ds, 'ObjectiveLensNumericalAperture', None)
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
    identifier: str
    illumination: Illumination
    description: Optional[str]
    icc_profile: Optional[bytes]
    lut: Optional[Lut]
    light_path_filter: Optional[LightPathFilter]
    image_path_filter: Optional[ImagePathFilter]
    channel_description: Optional[List[ChannelDescriptionCode]]
    lenses: Optional[Lenses]

    def __str__(self):
        return self.pretty_str()

    def pretty_str(
        self,
        indent: int = 0,
        depth: int = None
    ) -> str:
        if(self.description == ''):
            return self.identifier
        return self.identifier + ':' + self.description

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
            ds = self.light_path_filter.insert_into_ds(ds)
        if self.channel_description is not None:
            for item in self.channel_description:
                ds = item.insert_into_ds(ds)
        if self.lenses is not None:
            ds = self.lenses.insert_into_ds(ds)
        return ds

    @classmethod
    def from_ds(
        cls,
        ds: Dataset
    ) -> 'OpticalPath':
        """Create new optical path item populated with optical path
        identifier, description, icc profile name and lookup table.

        Parameters
        ----------
        optical_path: Dataset
            Optical path dataset containing the optical path data

        Returns
        ----------
        OpticalPath
            New optical path item
        """
        return OpticalPath(
            identifier=str(ds.OpticalPathIdentifier),
            illumination=Illumination.from_ds(ds),
            description=getattr(ds, 'OpticalPathDescription', None),
            icc_profile=getattr(ds, 'ICCProfile', None),
            lut=Lut.from_ds(ds),
            light_path_filter=LightPathFilter.from_ds(ds),
            image_path_filter=ImagePathFilter.from_ds(ds),
            channel_description=ChannelDescriptionCode.from_ds(ds),
            lenses=Lenses.from_ds(ds),
        )


class OpticalManager:
    def __init__(
        self,
        optical_paths: Dict[str, OpticalPath] = None,
    ):
        """Store optical paths loaded from dicom files.
        """

        self._optical_paths: Dict[str, OpticalPath] = optical_paths

    @classmethod
    def open(cls, files: List[WsiDicomFile]) -> 'OpticalManager':
        optical_paths: Dict[str, OpticalPath] = {}
        for file in files:
            for optical_ds in file.optical_path_sequence:
                identifier = str(optical_ds.OpticalPathIdentifier)
                if identifier not in optical_paths:
                    path = OpticalPath.from_ds(optical_ds)
                    optical_paths[identifier] = path
        return OpticalManager(optical_paths)

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
        ds.OpticalPathSequence = DicomSequence([
            optical_path.to_ds()
            for optical_path in self._optical_paths.values()
        ])
        return ds

    def get(self, identifier: str) -> OpticalPath:
        """Return the optical path item with identifier.

        Parameters
        ----------
        identifier: str
            The unique optical identifier to get

        Returns
        ----------
        OpticalPath
            The OpticalPath item
        """
        try:
            return self._optical_paths[identifier]
        except KeyError:
            raise WsiDicomNotFoundError(
                f"identifier {identifier}",
                "optical path manager"
            )

    def get_lut(self, identifer: str) -> Lut:
        """Return lookup table for optical path with identifier.

        Parameters
        ----------
        identifier: str
            The unique optical identifier to get the lookup table for

        Returns
        ----------
        Optional[Lut]
            The Lut for the optical path, or None if not set
        """
        path = self.get(identifer)
        if path.lut is None:
            raise WsiDicomNotFoundError(
                f"Lut for identifier {identifer}",
                "optical path manager"
            )
        return path.lut

    def apply_lut(self, image: Image, identifier: str) -> Image:
        """Apply LUT of identifier to image. Converts gray scale image to RGB.

        Parameters
        ----------
        image: Image
            Pillow image to apply LUT to.
        identifier: str
            The identifier of the LUT to apply

        Returns
        ----------
        Image
            Image with LUT applied.
        """
        if(image.mode == 'L'):
            image = image.convert('RGB')
        lut = self.get_lut(identifier)
        lut_array = lut.get_flat()/(2**lut._bits/256)
        return image.point(lut_array)

    @staticmethod
    def get_path_identifers(optical_path_sequence: DicomSequence) -> List[str]:
        return list({
            str(optical_ds.OpticalPathIdentifier)
            for optical_ds in optical_path_sequence
        })
