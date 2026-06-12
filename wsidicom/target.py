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

from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Sequence, Union

from pydicom.uid import UID

from wsidicom.codec import Encoder
from wsidicom.codec import Settings as EncoderSettings
from wsidicom.series import Labels, Overviews, Pyramids


class Target(metaclass=ABCMeta):
    """A target should be initiated with a path or similar, and enable saving of
    instances into that path."""

    def __init__(
        self,
        uid_generator: Callable[..., UID],
        workers: int,
        include_pyramids: Optional[Sequence[int]] = None,
        include_levels: Optional[Sequence[int]] = None,
        add_missing_levels: bool = False,
        transcoding: Optional[Union[EncoderSettings, Encoder]] = None,
        force_transcoding: bool = False,
    ) -> None:
        """Initiate a target.

        Parameters
        ----------
        uid_generator: Callable[..., UID]
            Uid generator to use.
        workers: int
            Maximum number of thread workers to use.
        include_pyramids: Optional[Sequence[int]] = None
            Optional list indices (in present pyramids) to include.
        include_levels: Optional[Sequence[int]] = None
            Optional list indices (in present levels) to include, e.g. [0, 1]
            includes the two lowest levels. Negative indices can be used,
            e.g. [-1, -2] includes the two highest levels.
        add_missing_levels: bool = False
            If to add missing dyadic levels up to the single tile level.
        transcoding: Optional[Union[EncoderSettings, Encoder]] = None
            Optional settings or encoder for transcoding image data. If None, image data
            will be copied as is.
        force_transcoding: bool = False
            If to force transcoding even if transfer syntax already matches the encoding
            settings.
        """
        self._uid_generator = uid_generator
        self._workers = workers
        self._include_pyramids = include_pyramids
        self._include_levels = include_levels
        self._add_missing_levels = add_missing_levels
        self._instance_number = 0
        self._force_transcoding = force_transcoding
        if isinstance(transcoding, EncoderSettings):
            self._transcoder = Encoder.create_for_settings(transcoding)
        else:
            self._transcoder = transcoding
        self.__enter__()

    @abstractmethod
    def save(
        self,
        pyramids: Pyramids,
        labels: Optional[Labels],
        overviews: Optional[Overviews],
        include_thumbnails: bool,
    ) -> None:
        """Save pyramids, labels, and overviews to the target.

        Parameters
        ----------
        pyramids: Pyramids
            Pyramids to save.
        labels: Optional[Labels]
            Labels to save, or None to skip.
        overviews: Optional[Overviews]
            Overviews to save, or None to skip.
        include_thumbnails: bool
            If to include thumbnails from pyramids.
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Should close any by the object opened resources."""
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
