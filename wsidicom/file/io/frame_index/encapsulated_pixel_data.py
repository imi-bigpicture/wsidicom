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

"""Abstract frame index for files with encapsulated data."""

from typing import Optional

from pydicom.tag import ItemTag

from wsidicom.errors import WsiDicomFileError
from wsidicom.file.io.frame_index.frame_index import FrameIndex


class EncapsulatedPixelData(FrameIndex):
    def _validate_pixel_data_start(self):
        """Check that pixel data tag is present and that the tag length is
        set as undefined. Raises WsiDicomFileError otherwise.

        """
        super()._validate_pixel_data_start(None)

    def _read_bot_length(self) -> Optional[int]:
        """Read the length of the basic table offset (BOT). Returns None if BOT
        is empty.

        Returns
        ----------
        Optional[int]
            BOT length.
        """
        BOT_BYTES = 4
        if self._file.read_tag() != ItemTag:
            raise WsiDicomFileError(
                str(self._file), "Basic offset table did not start with an ItemTag"
            )
        bot_length = self._file.read_UL()
        if bot_length == 0:
            return None
        elif bot_length % BOT_BYTES:
            raise WsiDicomFileError(
                str(self._file),
                f"Basic offset table should be a multiple of {BOT_BYTES} bytes",
            )
        return bot_length
