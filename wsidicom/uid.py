from pydicom.uid import UID as Uid
from dataclasses import dataclass

WSI_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.77.1.6'


@dataclass
class BaseUids:
    """Represents the UIDs that should be common for all files in the wsi."""
    study_instance: Uid
    series_instance: Uid
    frame_of_reference: Uid

    def __str__(self) -> str:
        return (
            f"BaseUids study: {self.study_instance}, "
            f"series: {self.series_instance}, "
            f"frame of reference {self.frame_of_reference}"
        )


@dataclass
class FileUids:
    """Represents the UIDs in a DICOM-file."""
    instance: Uid
    concatenation: Uid
    base: BaseUids

    @property
    def identifier(self) -> Uid:
        """Return identifier for the instance the file belongs to. Either its
        own intance uid or, if not none, the concnatenation uid.

        Returns
        ----------
        Uid
            Identifier uid for the instance the file belongs to.
        """
        if self.concatenation is not None:
            return self.concatenation
        return self.instance

    def match(self, other: 'FileUids') -> bool:
        """Return true if concatenation uid is not none, matches other
        concantenatin uid and base uids match.

        Parameters
        ----------
        other: FileUids
            File uids to match to.

        Returns
        ----------
        bool
            True if concatenation is matching.
        """
        if isinstance(other, FileUids):
            return (
                self.concatenation is not None and
                self.concatenation == other.concatenation and
                self.base == other.base
            )
        raise NotImplementedError
