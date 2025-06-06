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

"""A target for writing WSI DICOM files to disk."""

from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from pydicom.uid import UID, VLWholeSlideMicroscopyImageStorage
from pydicom.valuerep import MAX_VALUE_LEN
from upath import UPath

from wsidicom.cache import DecodedFrameCache, EncodedFrameCache
from wsidicom.codec import Encoder
from wsidicom.codec import Settings as EncoderSettings
from wsidicom.config import settings
from wsidicom.file.io import (
    OffsetTableType,
    WsiDicomReader,
    WsiDicomWriter,
)
from wsidicom.file.wsidicom_file_image_data import WsiDicomFileImageData
from wsidicom.file.wsidicom_stream_opener import WsiDicomStreamOpener
from wsidicom.geometry import Size, SizeMm
from wsidicom.group import Label, Level, Overview, Thumbnail
from wsidicom.instance import ImageData, WsiInstance
from wsidicom.series import Labels, Overviews, Pyramid, Pyramids
from wsidicom.tags import LossyImageCompressionMethodTag, LossyImageCompressionRatioTag
from wsidicom.target import Target


class WsiDicomFileTarget(Target):
    """Target for writing WSI DICOM instances to disk."""

    def __init__(
        self,
        output_path: Union[str, Path, UPath],
        uid_generator: Callable[..., UID],
        workers: int,
        chunk_size: int,
        offset_table: Optional[OffsetTableType] = None,
        include_pyramids: Optional[Sequence[int]] = None,
        include_levels: Optional[Sequence[int]] = None,
        add_missing_levels: bool = False,
        transcoding: Optional[Union[EncoderSettings, Encoder]] = None,
        file_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Create a WsiDicomFileTarget.

        Parameters
        ----------
        output_path: Union[str, Path, UPath]
            Folder path to save files to.
        uid_generator: Callable[..., UID]
            Uid generator to use.
        workers: int
            Maximum number of thread workers to use.
        chunk_size: int
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.
        offset_table: OffsetTableType
            Offset table to use.
        include_pyramids: Optional[Sequence[int]] = None
            Optional list indices (in present pyramids) to include.
        include_levels: Optional[Sequence[int]] = None
            Optional list indices (in all pyramids) to include, e.g. [0, 1]
            includes the two lowest levels. Negative indicies can be used,
            e.g. [-1, -2] includes the two highest levels.
        add_missing_levels: bool
            If to add missing dyadic levels up to the single tile level.
        transcoding: Optional[Union[EncoderSettings, Encoder]] = None,
            Optional settings or encoder for transcoding image data. If None, image data
            will be copied as is.
        file_options: Optional[Dict[str, Any]] = None
            Keyword arguments for saving files to output path.
        """
        self._output_path = UPath(output_path)
        self._offset_table = offset_table
        self._filepaths: List[UPath] = []
        self._opened_files: List[WsiDicomReader] = []
        self._file_options = file_options
        self._decoded_frame_cache = DecodedFrameCache(settings.decoded_frame_cache_size)
        self._encoded_frame_cache = EncodedFrameCache(settings.encoded_frame_cache_size)
        super().__init__(
            uid_generator,
            workers,
            chunk_size,
            include_pyramids,
            include_levels,
            add_missing_levels,
            transcoding,
        )

    @property
    def filepaths(self) -> List[UPath]:
        """Return filepaths for created files."""
        return self._filepaths

    def save_pyramids(self, pyramids: Pyramids, include_thumbnails: bool):
        if self._include_pyramids is not None:
            pyramids_to_save = [pyramids[index] for index in self._include_pyramids]
        else:
            pyramids_to_save = pyramids
        for pyramid in pyramids_to_save:
            self._save_pyramid(pyramid, include_thumbnails)

    def save_labels(self, labels: Labels):
        """Save labels to target."""
        for label in labels:
            self._save_group(label, 1)

    def save_overviews(self, overviews: Overviews):
        """Save overviews to target."""
        for overview in overviews:
            self._save_group(overview, 1)

    def close(self) -> None:
        """Close any opened level files."""
        for file in self._opened_files:
            file.close()

    def _save_pyramid(self, pyramid: Pyramid, include_thumbnails: bool):
        """Save pyramid to target."""
        # Collection of new pyramid levels.
        new_levels: List[Level] = []
        highest_level_in_file = pyramid.pyramid_indices[-1]
        lowest_single_tile_level = pyramid.lowest_single_tile_level
        highest_level = max(highest_level_in_file, lowest_single_tile_level)
        for pyramid_level in range(highest_level + 1):
            if not self._is_included_level(
                pyramid_level,
                pyramid.pyramid_indices,
                self._add_missing_levels,
                self._include_levels,
            ):
                continue
            if pyramid_level in pyramid.pyramid_indices:
                level = pyramid.get(pyramid_level)
                self._save_group(level, 1)

            elif self._add_missing_levels:
                # Create scaled level from closest level, prefer from original levels
                closest_level = pyramid.get_closest_by_level(pyramid_level)
                closest_new_level = next(
                    (
                        level
                        for level in sorted(
                            new_levels, key=lambda x: x.level, reverse=True
                        )
                        if level.level < pyramid_level
                        and level.level > closest_level.level
                    ),
                    None,
                )
                if closest_new_level is not None:
                    closest_level = closest_new_level
                scale = int(2 ** (pyramid_level - closest_level.level))
                new_level = self._save_and_open_level(
                    closest_level,
                    pyramid.pixel_spacing,
                    scale,
                )
                new_levels.append(new_level)
        if include_thumbnails and pyramid.thumbnails is not None:
            for group in pyramid.thumbnails.groups:
                self._save_group(group, 1)

    def _save_and_open_level(
        self,
        level: Level,
        base_pixel_spacing: SizeMm,
        scale: int,
    ) -> Level:
        """Save level and return a new level from the created files."""
        filepaths = self._save_group(level, scale)
        instances = self._open_files(filepaths)
        return Level(instances, base_pixel_spacing)

    def _save_group(
        self,
        group: Union[Label, Level, Overview, Thumbnail],
        scale: int,
    ) -> List[UPath]:
        """Save group to target."""
        if not isinstance(scale, int) or scale < 1:
            raise ValueError(f"Scale must be positive integer, got {scale}.")
        filepaths: List[UPath] = []
        for instances in self._group_instances_to_file(group):
            uid = self._uid_generator()
            filepath = self._output_path.joinpath(uid + ".dcm")

            image_data_list = self._list_image_data(instances)
            focal_planes, optical_paths, tiled_size = self._get_frame_information(
                image_data_list
            )

            dataset = instances[0].dataset.as_tiled_full(
                focal_planes, optical_paths, tiled_size, scale
            )
            if self._transcoder is not None:
                transcoder = self._transcoder
            elif instances[0].image_data.transcoder is not None:
                transcoder = instances[0].image_data.transcoder
            else:
                transcoder = None
            if transcoder is not None:
                if (
                    transcoder.bits != instances[0].image_data.bits
                    or transcoder.samples_per_pixel
                    != instances[0].image_data.samples_per_pixel
                ):
                    raise ValueError(
                        "Transcode settings must match image data bits and "
                        "photometric interpretation."
                    )
                transfer_syntax = transcoder.transfer_syntax
                dataset.PhotometricInterpretation = (
                    transcoder.photometric_interpretation
                )
                if transcoder.lossy_method:
                    dataset.LossyImageCompression = "01"
                    ratios = dataset.get_multi_value(LossyImageCompressionRatioTag)
                    # Reserve space for new ratio
                    ratios.append(" " * MAX_VALUE_LEN["DS"])
                    methods = dataset.get_multi_value(LossyImageCompressionMethodTag)
                    methods.append(transcoder.lossy_method.value)
                    dataset.LossyImageCompressionRatio = ratios
                    dataset.LossyImageCompressionMethod = methods
            else:
                transfer_syntax = instances[0].image_data.transfer_syntax
            if self._offset_table is not None:
                offset_table = self._offset_table
            elif transfer_syntax.is_encapsulated:
                offset_table = OffsetTableType.BASIC
            else:
                offset_table = OffsetTableType.NONE
            with WsiDicomWriter.open(
                filepath, transfer_syntax, offset_table, self._file_options
            ) as writer:
                writer.write(
                    uid,
                    dataset,
                    image_data_list,
                    self._workers,
                    self._chunk_size,
                    self._instance_number,
                    scale,
                    transcoder,
                )
            filepaths.append(filepath)
            self._instance_number += 1
        self._filepaths.extend(filepaths)
        return filepaths

    def _open_files(self, filepaths: Iterable[UPath]) -> List[WsiInstance]:
        readers = [
            WsiDicomReader(stream)
            for stream in WsiDicomStreamOpener(self._file_options).open(
                filepaths, VLWholeSlideMicroscopyImageStorage
            )
        ]
        self._opened_files.extend(readers)
        return [
            WsiInstance(
                [reader.dataset for reader in readers],
                WsiDicomFileImageData(
                    readers, self._decoded_frame_cache, self._encoded_frame_cache
                ),
            )
        ]

    @staticmethod
    def _group_instances_to_file(
        group: Union[Label, Level, Overview, Thumbnail],
    ) -> List[List[WsiInstance]]:
        """
        Group instances by properties that can't differ in a DICOM-file.

        Returns
        -------
        List[List[WsiInstance]]
            Instances grouped by common properties.
        """
        groups: Dict[
            Tuple[str, UID, bool, Optional[int], Optional[float], str],
            List[WsiInstance],
        ] = defaultdict(list)

        for instance in group.instances.values():
            groups[
                instance.image_data.photometric_interpretation,
                instance.image_data.transfer_syntax,
                instance.ext_depth_of_field,
                instance.ext_depth_of_field_planes,
                instance.ext_depth_of_field_plane_distance,
                instance.focus_method,
            ].append(instance)
        return list(groups.values())

    @staticmethod
    def _list_image_data(
        instances: Iterable[WsiInstance],
    ) -> Dict[Tuple[str, float], ImageData]:
        """
        Sort ImageData in instances by optical path and focal plane.

        Parameters
        ----------
        instances: Iterable[WsiInstance]
            List of instances with optical paths and focal planes to list and
            sort.

        Returns
        -------
        Dict[Tuple[str, float], ImageData]:
            ImageData sorted by optical path and focal plane.
        """
        output: Dict[Tuple[str, float], ImageData] = {}
        for instance in instances:
            for optical_path in instance.optical_paths:
                for z in sorted(instance.focal_planes):
                    if (optical_path, z) not in output:
                        output[optical_path, z] = instance.image_data
        return output

    @staticmethod
    def _get_frame_information(
        data: Dict[Tuple[str, float], ImageData],
    ) -> Tuple[List[float], List[str], Size]:
        """Return optical_paths, focal planes, and tiled size."""
        focal_planes_by_optical_path: Dict[str, Set[float]] = defaultdict(set)
        all_focal_planes: Set[float] = set()
        tiled_sizes: Set[Size] = set()
        for (optical_path, focal_plane), image_data in data.items():
            focal_planes_by_optical_path[optical_path].add(focal_plane)
            all_focal_planes.add(focal_plane)
            tiled_sizes.add(image_data.tiled_size)

        focal_planes_sparse_by_optical_path = any(
            optical_path_focal_planes != all_focal_planes
            for optical_path_focal_planes in focal_planes_by_optical_path.values()
        )
        if focal_planes_sparse_by_optical_path:
            raise ValueError("Each optical path must have the same focal planes.")

        if len(tiled_sizes) != 1:
            raise ValueError(f"Expected only one tiled size, found {len(tiled_sizes)}.")
        tiled_size = list(tiled_sizes)[0]
        return (
            sorted(list(all_focal_planes)),
            sorted(list(focal_planes_by_optical_path.keys())),
            tiled_size,
        )
