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
from collections.abc import Sequence

from wsidicom.codec import Encoder
from wsidicom.codec import Settings as EncoderSettings
from wsidicom.metadata.uid_generator import UidGenerator
from wsidicom.series import Labels, Overviews, Pyramids

"""A Target enables creating new instances."""


class Target(metaclass=ABCMeta):
    """A target should be initiated with a path or similar, and enable saving of
    instances into that path."""

    def __init__(
        self,
        uid_generator: UidGenerator,
        workers: int,
        chunk_size: int,
        include_pyramids: Sequence[int] | None = None,
        include_levels: Sequence[int] | None = None,
        add_missing_levels: bool = False,
        transcoding: EncoderSettings | Encoder | None = None,
        force_transcoding: bool = False,
    ) -> None:
        """Initiate a target.

        Parameters
        ----------
        uid_generator: UidGenerator
            Generator for producing UIDs.
        workers: int
            Maximum number of thread workers to use.
        chunk_size: int
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.
        include_pyramids: Sequence[int] | None = None
            Optional list indices (in present pyramids) to include.
        include_levels: Sequence[int] | None = None
            Optional list indices (in present levels) to include, e.g. [0, 1]
            includes the two lowest levels. Negative indices can be used,
            e.g. [-1, -2] includes the two highest levels.
        add_missing_levels: bool = False
            If to add missing dyadic levels up to the single tile level.
        transcoding: EncoderSettings | Encoder | None = None
            Optional settings or encoder for transcoding image data. If None, image data
            will be copied as is.
        force_transcoding: bool = False
            If to force transcoding even if transfer syntax already matches the encoding
            settings.
        """
        self._uid_generator = uid_generator
        self._workers = workers
        self._chunk_size = chunk_size
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
    def save_pyramids(self, pyramids: Pyramids, include_thumbnails: bool) -> None:
        """Should save the pyramids to the target."""
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

    @staticmethod
    def _select_included_levels(
        candidate_levels: Sequence[int],
        include_indices: Sequence[int] | None = None,
    ) -> set[int]:
        """Return the set of pyramid levels to include from candidate_levels.

        Parameters
        ----------
        candidate_levels: Sequence[int]
            Pyramid levels eligible for inclusion. When the caller wants to
            allow missing levels to be generated, this should be the extended
            level list (e.g. ``range(0, highest_level + 1)``); otherwise it
            should be the list of natively present levels.
        include_indices: Sequence[int] | None = None
            Optional list of indices (into ``candidate_levels``) to include,
            e.g. ``[0, 1]`` includes the two lowest. Negative indices can be
            used, e.g. ``[-1, -2]`` includes the two highest. Out-of-range
            indices are silently ignored. ``None`` selects all candidates;
            an empty sequence selects none.

        Returns
        -------
        set[int]
            Set of pyramid levels to include.
        """
        if include_indices is None:
            return set(candidate_levels)
        candidate_count = len(candidate_levels)
        valid_indices = [
            index
            for index in include_indices
            if -candidate_count <= index < candidate_count
        ]
        return {candidate_levels[index] for index in valid_indices}
