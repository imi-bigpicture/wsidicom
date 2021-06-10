import io
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image, ImageCms
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.sr.coding import Code

from .errors import WsiDicomNotFoundError
from .file import WsiDicomFile


@dataclass
class IccProfile:
    name: str
    description: str
    profile: bytes

    @staticmethod
    def read_profile(ds: Dataset) -> ImageCms.ImageCmsProfile:
        icc_profile: bytes = ds.ICCProfile
        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
        return profile

    @classmethod
    def read_name(cls, ds: Dataset) -> Optional[str]:
        """Read Icc profile name from dataset.

        Parameters
        ----------
        optical_path_ds: Dataset
            A dataset containing an icc profile

        Returns
        ----------
        Optional[str]
            The icc profile name
        """
        try:
            profile = cls.read_profile(ds)
            icc_profile_name = str(ImageCms.getProfileName(profile))
            return icc_profile_name
        except (ImageCms.PyCMSError, AttributeError):
            # No profile found or profile not valid
            return None

    @classmethod
    def from_ds(cls, ds: Dataset) -> 'IccProfile':
        profile = cls.read_profile(ds)
        return IccProfile(
            name=str(ImageCms.getProfileName(profile)),
            description=str(ImageCms.getProfileDescription(profile)),
            profile=profile
        )


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

    @property
    def sequence(self) -> DicomSequence:
        return DicomSequence[self._lut_item]

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
class OpticalPath:
    identifier: str
    icc_profile_name: str
    illumination_types: List[Code]
    illumination_wavelengt: Optional[float]
    illumination_color: Optional[Code]
    description: Optional[str]
    lut: Optional[Lut]

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

    @classmethod
    def from_ds(cls, ds: Dataset) -> 'OpticalPath':
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
            icc_profile_name=IccProfile.read_name(ds),
            illumination_types=cls.get_illumination_type_codes(ds),
            illumination_wavelengt=getattr(ds, 'IlluminationWaveLength', None),
            illumination_color=cls.get_illumination_color(ds),
            description=str(getattr(ds, 'OpticalPathDescription', None)),
            lut=cls.get_lut(ds)
        )

    @staticmethod
    def get_lut(ds: Dataset) -> Optional[Lut]:
        if('PaletteColorLookupTableSequence' in ds):
            return Lut(ds.PaletteColorLookupTableSequence)
        return None

    @staticmethod
    def get_illumination_color(ds: Dataset) -> Optional[Code]:
        if 'IlluminationColorCodeSequence' not in ds:
            return None
        code_ds = ds.IlluminationColorCodeSequence[0]
        code = Code(
            value=code_ds.CodeValue,
            scheme_designator=code_ds.CodingSchemeDesignator,
            meaning=code_ds.CodeMeaning,
            scheme_version=getattr(ds, 'CodingSchemeVersion', None)
        )
        return code

    @staticmethod
    def get_illumination_type_codes(ds: Dataset) -> List[Code]:
        codes: List[Code] = []
        for code_ds in ds.IlluminationTypeCodeSequence:
            code = Code(
                value=code_ds.CodeValue,
                scheme_designator=code_ds.CodingSchemeDesignator,
                meaning=code_ds.CodeMeaning,
                scheme_version=getattr(ds, 'CodingSchemeVersion', None)
            )
            codes.append(code)
        return codes


class OpticalManager:
    def __init__(
        self,
        optical_paths: List[OpticalPath] = None,
        icc_profiles: List[IccProfile] = None
    ):
        """Store optical paths and icc profiles loaded from dicom files.
        """

        self._optical_paths: Dict[str, OpticalPath] = optical_paths
        self._icc_profiles: Dict[str, IccProfile] = icc_profiles

    @classmethod
    def open(cls, files: List[WsiDicomFile]) -> 'OpticalManager':
        optical_paths: Dict[str, OpticalPath] = {}
        icc_profiles: Dict[str, IccProfile] = {}
        for file in files:
            optical_paths, icc_profiles = cls._open_file(
               file,
               optical_paths,
               icc_profiles
            )
        return OpticalManager(optical_paths, icc_profiles)

    @staticmethod
    def _open_file(
        file: WsiDicomFile,
        optical_paths: Dict[str, OpticalPath],
        icc_profiles: Dict[str, IccProfile]
    ) -> Tuple[Dict[str, OpticalPath], Dict[str, IccProfile]]:
        for optical_ds in file.optical_path_sequence:
            identifier = str(optical_ds.OpticalPathIdentifier)
            if identifier not in optical_paths:
                path = OpticalPath.from_ds(optical_ds)
                optical_paths[identifier] = path
                icc_profile_name = IccProfile.read_name(optical_ds)
                if (
                    icc_profile_name is not None
                    and icc_profile_name not in icc_profiles
                ):
                    profile = IccProfile.from_ds(optical_ds)
                    name = profile.name
                    icc_profiles[name] = profile
        return optical_paths, icc_profiles

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

    def get_icc(self, identifer: str) -> bytes:
        """Return icc profile for optical path with identifier.

        Parameters
        ----------
        identifier: str
            The unique optical identifier to get the lookup table for

        Returns
        ----------
        bytes
            The Icc profile in bytes
        """
        path = self.get(identifer)
        name = path.icc_profile_name
        try:
            return self._icc_profiles[name].profile
        except KeyError:
            raise WsiDicomNotFoundError(
                f"icc profile {name}", "optical path manager"
            )

    @staticmethod
    def get_path_identifers(optical_path_sequence: DicomSequence) -> List[str]:
        found_identifiers: Set[str] = set()
        for optical_ds in optical_path_sequence:
            identifier = str(optical_ds.OpticalPathIdentifier)
            found_identifiers.add(identifier)
        return list(found_identifiers)
