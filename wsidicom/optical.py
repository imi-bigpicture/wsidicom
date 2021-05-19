from dataclasses import dataclass
import struct
import numpy as np
from typing import Dict, List, Optional
from PIL import Image, ImageCms
import io


from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence

from .errors import WsiDicomNotFoundError
from .uid import BaseUids


@dataclass
class IccProfile:
    name: str
    description: str
    profile: bytes


class Lut:
    def __init__(self, size: int, bits: int):
        """Stores RGB lookup tables.

        Parameters
        ----------
        size: int
            the number of entries in the table
        bits: int
            the bits for each entry (currently forced to 16)
        """
        self.bits = bits
        self._type: type
        if bits == 8:
            self._type = np.uint8
        else:
            self._type = np.uint16
        self.table = np.zeros((3, size), self._type)
        self._position = [0, 0, 0]
        self._last_value = [-1, -1, -1]
        self._byte_format = 'HHH'  # Do we need to set endianess?

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

    def parse_lut(self, lut: DicomSequence):
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
        for color, table in enumerate(tables):
            for i in range(int(len(table)/6)):
                segment_bytes = table[i*6:i*6+6]
                segment_type, length, value = struct.unpack(
                    self._byte_format,
                    segment_bytes
                )
                if(segment_type == 0):
                    self._add_discret(color, length, value)
                elif(segment_type == 1):
                    self._add_linear(color, length, value)

    def _insert(self, channel: int, segment: np.ndarray):
        """Insert a segement into the lookup table of channel.

        Parameters
        ----------
        channel: int
            The channel (r=0, g=1, b=2) to operate on
        segment: np.ndarray
            The segment to insert
        """
        length = len(segment)
        position = self._position[channel]
        self.table[channel, position:position+length] = segment
        self._position[channel] += length
        self._last_value[channel] = segment[length - 1]

    def _add_discret(self, channel: int, length: int, value: int):
        """Add a discret segement into the lookup table of channel.

        Parameters
        ----------
        channel: int
            The channel (r=0, g=1, b=2) to operate on
        length: int
            The lenght of the discret segment
        value: int
            The value of the deiscret segment
        """
        segment = np.full(length, value, dtype=self._type)
        self._insert(channel, segment)

    def _add_linear(self, channel: int, lenght: int, value: int):
        """Add a linear segement into the lookup table of channel.

        Parameters
        ----------
        channel: int
            The channel (r=0, g=1, b=2) to operate on
        length: int
            The lenght of the discret segment
        value: int
            The value of the deiscret segment
        """
        # Default shift segment by one to not include first value
        # (same as last value)
        start_position = 1
        # If no last value, set it to 0 and include
        # first value in segment
        if self._last_value[channel] == -1:
            self._last_value[channel] = 0
            start_position = 0
        segment = np.linspace(
            start=self._last_value[channel],
            stop=value,
            num=start_position+lenght,
            dtype=self._type
        )
        self._insert(channel, segment[start_position:])


@dataclass
class OpticalPath:
    identifier: str
    description: str
    icc_profile_name: str
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


class OpticalManager:
    def __init__(self, uids: BaseUids):
        """Store optical paths and icc profiles loaded from dicom files.
        Parameters
        ----------
        uids: BaseUids
            Base uids this manager is bound to
        """

        self.uids = uids
        self._optical_paths: Dict[str, OpticalPath] = {}
        self._icc_profiles: Dict[str, IccProfile] = {}

    def add(self, sequence: DicomSequence) -> List[str]:
        """Add optical paths parsed from pydicom optical path sequence.

        Parameters
        ----------
        ds: DicomSequence
            A pydicom Optical Path Sequence

        Returns
        ----------
        List[str]
            A list of found identifiers
        """
        found_identifiers: List[str] = []
        for optical_path in sequence:
            identifier = str(optical_path.OpticalPathIdentifier)
            found_identifiers.append(identifier)
            # Is optical path with identifier already known?
            try:
                self.get(identifier)
            except WsiDicomNotFoundError:
                path = self._new_optical_path(identifier, optical_path)
                # self._optical_paths.append(path)
                self._optical_paths[identifier] = path
        return found_identifiers

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
        path = self._optical_paths.get(identifier)
        if path is None:
            raise WsiDicomNotFoundError(
                f"identifier {identifier}",
                "optical path manager"
            )
        return path

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
        lut_array = lut.get_flat()/(2**lut.bits/256)
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
        return self._get_icc_bytes(path.icc_profile_name)

    def _get_icc_bytes(self, name: str) -> bytes:
        """Get Icc profile bytes by icc profile name.

        Parameters
        ----------
        name: str
            The name of the icc profile

        Returns
        ----------
        IccProfile
            The profile with specified name
        """
        icc_profile = self._get_icc(name)
        return icc_profile.profile

    def _get_icc(self, name: str) -> IccProfile:
        """Get a Icc profile item by icc profile name.

        Parameters
        ----------
        name: str
            The name of the icc profile

        Returns
        ----------
        IccProfile
            The profile with specified name
        """
        icc_profile = self._icc_profiles.get(name)
        if icc_profile is None:
            raise WsiDicomNotFoundError(
                f"icc profile {name}", "optical path manager"
            )
        return icc_profile

    def _add_icc(self, name: str, profile: IccProfile):
        """Add Icc profile item to manager

        Parameters
        ----------
        name: str
            Profile name
        profile: IccProfile
            The profile to add

        """
        self._icc_profiles[name] = profile

    def _read_icc(self, optical_path_ds: Dataset) -> str:
        """Read Icc profile item from dataset. If profile is not already saved,
        saves the profile. Returns the profile name if succesfull.

        Parameters
        ----------
        optical_path_ds: Dataset
            A dataset containing an icc profile

        Returns
        ----------
        str
            The icc profile name
        """
        try:
            icc_profile: bytes = optical_path_ds.ICCProfile
            profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
            icc_profile_name = str(ImageCms.getProfileName(profile))
            # Is the ICC name already known?
            try:
                self._get_icc(icc_profile_name)
            except WsiDicomNotFoundError:
                # Save profile
                description = ImageCms.getProfileDescription(profile)
                new_profile = IccProfile(
                    name=icc_profile_name,
                    description=description,
                    profile=icc_profile
                )
                self._add_icc(icc_profile_name, new_profile)
            return icc_profile_name
        except (ImageCms.PyCMSError, AttributeError):
            # No profile found or profile not valid
            return ""

    def _new_optical_path(
        self,
        identifier: str,
        optical_path: Dataset
    ) -> OpticalPath:
        """Create new optical path item populated with optical path
        identifier, description, icc profile name and lookup table.

        Parameters
        ----------
        identifier: str
            Identifier of the new optical path
        optical_path: Dataset
            Optical path dataset containing the optical path data

        Returns
        ----------
        OpticalPath
            New optical path item
        """
        # New optical path, try to load ICC, get description and lut
        icc_profile_name = self._read_icc(optical_path)
        description = getattr(
            optical_path,
            'OpticalPathDescription',
            ''
        )
        lut: Optional[Lut] = None
        if('PaletteColorLookupTableSequence' in optical_path):
            sequence = optical_path.PaletteColorLookupTableSequence[0]
            lenght, first, bits = \
                sequence.RedPaletteColorLookupTableDescriptor
            lut = Lut(lenght, bits)
            lut.parse_lut(sequence)
        # Create a new OpticalPath and add
        path = OpticalPath(
            identifier=identifier,
            description=description,
            icc_profile_name=icc_profile_name,
            lut=lut
        )
        return path
