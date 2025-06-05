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

from pydicom.uid import (
    UID,
    AllTransferSyntaxes,
    UID_dictionary,  # type: ignore[import-untyped]
)

from wsidicom.config import settings
from wsidicom.errors import WsiDicomStrictRequirementError

transfer_syntaxes_to_register = {
    "1.2.840.10008.1.2.4.110": (
        "JPEG XL Lossless",
        "Transfer Syntax",
        "",
        "",
        "JPEGXLLossless",
    ),
    "1.2.840.10008.1.2.4.111": (
        "JPEG XL JPEG Recompression",
        "Transfer Syntax",
        "",
        "",
        "JPEGXLJPEGRecompression",
    ),
    "1.2.840.10008.1.2.4.112": (
        "JPEG XL",
        "Transfer Syntax",
        "",
        "",
        "JPEGXL",
    ),
}
for ts_uid, ts_info in transfer_syntaxes_to_register.items():
    if ts_uid not in UID_dictionary:
        UID_dictionary[ts_uid] = ts_info
    if UID(ts_uid) not in AllTransferSyntaxes:
        AllTransferSyntaxes.append(UID(ts_uid))

JPEGXLLossless = UID("1.2.840.10008.1.2.4.110")

JPEGXLJPEGRecompression = UID("1.2.840.10008.1.2.4.111")

JPEGXL = UID("1.2.840.10008.1.2.4.112")


@dataclass(frozen=True)
class SlideUids:
    """Represents the UIDs that should be common for all files of a slide."""

    study_instance: UID
    series_instance: UID
    frame_of_reference: Optional[UID] = None

    def __post_init__(self) -> None:
        if settings.strict_uid_check and self.frame_of_reference is None:
            raise WsiDicomStrictRequirementError(
                "Frame of reference uid is missing and strict uid check is " "enabled"
            )

    def __str__(self) -> str:
        return (
            f"SlideUids study: {self.study_instance}, "
            f"series: {self.series_instance}, "
            f"frame of reference {self.frame_of_reference}"
        )

    def __eq__(self, other: "SlideUids") -> bool:
        if isinstance(other, SlideUids):
            return (
                self.study_instance == other.study_instance
                and self.series_instance == other.series_instance
                and (
                    self.frame_of_reference == other.frame_of_reference
                    or self.frame_of_reference is None
                    or other.frame_of_reference is None
                )
            )
        return NotImplemented

    def matches(self, other: "SlideUids") -> bool:
        if settings.strict_uid_check:
            return (
                self.study_instance == other.study_instance
                and self.frame_of_reference == other.frame_of_reference
            )

        return self.study_instance == other.study_instance


@dataclass(frozen=True)
class FileUids:
    """Represents the UIDs in a DICOM-file."""

    instance: UID
    concatenation: Optional[UID]
    slide: SlideUids

    @property
    def identifier(self) -> UID:
        """Return identifier for the instance the file belongs to. Either its
        own instance uid or, if not none, the concnatenation uid.

        Returns
        -------
        UID
            Identifier uid for the instance the file belongs to.
        """
        if self.concatenation is not None:
            return self.concatenation
        return self.instance

    def __eq__(self, other: "FileUids") -> bool:
        """Return true if concatenation uid is not none, matches other
        concatenation uid and base uids match.

        Parameters
        ----------
        other: FileUids
            File uids to match to.

        Returns
        -------
        bool
            True if concatenation is matching.
        """
        if isinstance(other, FileUids):
            return (
                self.concatenation is not None
                and self.concatenation == other.concatenation
                and self.slide == other.slide
            )
        return NotImplemented
