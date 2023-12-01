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
from typing import Callable, Optional

from pydicom.uid import UID

from wsidicom.codec import Settings as EncoderSettings
from wsidicom.series import Labels, Levels, Overviews

"""A Target enables creating new instances."""


class Target(metaclass=ABCMeta):
    """A target should be initiated with a path or similar, and enable saving of
    instances into that path."""

    def __init__(
        self,
        uid_generator: Callable[..., UID],
        workers: int,
        chunk_size: int,
        add_missing_levels: bool = False,
        transcode_settings: Optional[EncoderSettings] = None,
    ) -> None:
        """Initiate a target.

        Parameters
        ----------
        uid_generator: Callable[..., UID]
            Uid generator to use.
        workers: int
            Maximum number of thread workers to use.
        chunk_size: int
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.
        add_missing_levels: bool = False
            If to add missing dyadic levels up to the single tile level.
        """
        self._uid_generator = uid_generator
        self._workers = workers
        self._chunk_size = chunk_size
        self._add_missing_levels = add_missing_levels
        self._instance_number = 0
        self._transcode_settings = transcode_settings
        self.__enter__()

    @abstractmethod
    def save_levels(self, levels: Levels) -> None:
        """Should save the levels to the target."""
        raise NotImplementedError()

    @abstractmethod
    def save_labels(self, labels: Labels) -> None:
        """Should save the labels to the target."""
        raise NotImplementedError()

    @abstractmethod
    def save_overviews(self, overviews: Overviews) -> None:
        """Should save the overviews to the target."""
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """Should close any by the object opened resources."""
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
