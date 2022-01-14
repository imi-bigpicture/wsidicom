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

from dataclasses import dataclass
from typing import Optional

from pydicom.uid import UID as Uid

from wsidicom.config import settings

WSI_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.77.1.6'
ANN_SOP_CLASS_UID = '1.2.840.10008.5.1.4.1.1.91.1'


@dataclass
class BaseUids:
    """Represents the UIDs that should be common for all files in the wsi."""
    study_instance: Uid
    series_instance: Uid
    frame_of_reference: Optional[Uid] = None

    def __str__(self) -> str:
        return (
            f"BaseUids study: {self.study_instance}, "
            f"series: {self.series_instance}, "
            f"frame of reference {self.frame_of_reference}"
        )

    def __eq__(self, other: 'BaseUids') -> bool:
        if isinstance(other, BaseUids):
            return (
                self.study_instance == other.study_instance and
                self.series_instance == other.series_instance and
                (
                    self.frame_of_reference == other.frame_of_reference or
                    self.frame_of_reference is None or
                    other.frame_of_reference is None
                )
            )
        return NotImplemented

    def matches(self, other: 'BaseUids') -> bool:
        if settings.strict_uid_check:
            return self == other

        return (
            self.study_instance == other.study_instance and
            self.series_instance == other.series_instance
        )


@dataclass
class FileUids:
    """Represents the UIDs in a DICOM-file."""
    instance: Uid
    concatenation: Optional[Uid]
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

    def __eq__(self, other: 'FileUids') -> bool:
        """Return true if concatenation uid is not none, matches other
        concatenation uid and base uids match.

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
        return NotImplemented
