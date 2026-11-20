#    Copyright 2026 SECTRA AB
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

"""Sources of the encoded tiles to write for a group instance."""

from abc import ABCMeta, abstractmethod
from collections.abc import Iterator, Sequence

from wsidicom.codec import Encoder, Recoder
from wsidicom.group import Instances


class GroupTileSource(metaclass=ABCMeta):
    """Gives the encoded tiles to write for the frames of group instances."""

    @abstractmethod
    def iter_tiles(
        self,
        instances: Instances,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
    ) -> Iterator[bytes]:
        """Iterate the encoded tiles to write, in frame order.

        Parameters
        ----------
        instances: Instances
            Source instances to read tiles from.
        focal_planes: Sequence[float]
            Focal planes to read tiles of.
        optical_paths: Sequence[str]
            Optical paths to read tiles of.

        Returns
        -------
        Iterator[bytes]
            The encoded tiles to write.
        """
        raise NotImplementedError()


class CopyingGroupTileSource(GroupTileSource):
    """Gives the encoded tiles of the source unchanged."""

    def iter_tiles(
        self,
        instances: Instances,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
    ) -> Iterator[bytes]:
        return instances.iter_encoded_tiles(focal_planes, optical_paths)


class RecodingGroupTileSource(GroupTileSource):
    """Gives the encoded tiles of the source recoded."""

    def __init__(self, recoder: Recoder) -> None:
        """Create a source that recodes the encoded tiles.

        Parameters
        ----------
        recoder: Recoder
            Recoder to recode the tiles with.
        """
        self._recoder = recoder

    def iter_tiles(
        self,
        instances: Instances,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
    ) -> Iterator[bytes]:
        return (
            self._recoder.recode(tile)
            for tile in instances.iter_encoded_tiles(focal_planes, optical_paths)
        )


class TranscodingGroupTileSource(GroupTileSource):
    """Gives the tiles of the source decoded and encoded again."""

    def __init__(self, encoder: Encoder) -> None:
        """Create a source that transcodes the tiles.

        Parameters
        ----------
        encoder: Encoder
            Encoder to encode the decoded tiles with.
        """
        self._encoder = encoder

    def iter_tiles(
        self,
        instances: Instances,
        focal_planes: Sequence[float],
        optical_paths: Sequence[str],
    ) -> Iterator[bytes]:
        return (
            self._encoder.encode(tile)
            for tile in instances.iter_decoded_tiles(focal_planes, optical_paths)
        )
