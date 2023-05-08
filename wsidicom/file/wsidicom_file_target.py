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
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from pydicom.uid import UID

from wsidicom.file.wsidicom_file import WsiDicomFile
from wsidicom.file.wsidicom_file_base import OffsetTableType
from wsidicom.file.wsidicom_file_image_data import WsiDicomFileImageData
from wsidicom.file.wsidicom_file_writer import WsiDicomFileWriter
from wsidicom.geometry import Size, SizeMm
from wsidicom.group import Group, Level
from wsidicom.instance import ImageData, WsiInstance
from wsidicom.series import Labels, Levels, Overviews
from wsidicom.target import Target


class WsiDicomFileTarget(Target):
    """Target for writing WSI DICOM instances to disk."""

    def __init__(
        self,
        output_path: Path,
        uid_generator: Callable[..., UID],
        workers: int,
        chunk_size: int,
        offset_table: Optional[str],
        add_missing_levels: bool = False,
    ):
        """
        Create a WsiDicomFileTarget.

        Parameters
        ----------
        output_path: Path
            Folder path to save files to.
        uid_generator: Callable[..., UID]
            Uid generator to use.
        workers: int
            Maximum number of thread workers to use.
        chunk_size: int
            Chunk size (number of tiles) to process at a time. Actual chunk
            size also depends on minimun_chunk_size from image_data.
        offset_table: Optional[str]
            Offset table to use, 'bot' basic offset table, 'eot' extended
            offset table, None - no offset table.
        add_missing_levels: bool = False
            If to add missing dyadic levels up to the single tile level.
        """
        self._output_path = output_path
        self._offset_table = OffsetTableType.from_string(offset_table)
        self._filepaths: List[Path] = []
        self._opened_files: List[WsiDicomFile] = []
        super().__init__(uid_generator, workers, chunk_size, add_missing_levels)

    @property
    def filepaths(self) -> List[Path]:
        """Return filepaths for created files."""
        return self._filepaths

    def save_levels(self, levels: Levels):
        """Save levels to target."""
        # Collection of new pyramid levels.
        new_levels: List[Level] = []
        highest_level_in_file = levels.levels[-1]
        lowest_single_tile_level = levels.lowest_single_tile_level
        highest_level = max(highest_level_in_file, lowest_single_tile_level)
        for pyramid_level in range(highest_level + 1):
            if pyramid_level in levels.levels:
                level = levels.get_level(pyramid_level)
                self._save_group(level, 1)
            elif self._add_missing_levels:
                # Create scaled level from closest level, prefer from original levels
                closest_level = levels.get_closest_by_level(pyramid_level)
                closest_new_level = next(
                    (
                        level
                        for level in new_levels
                        if level.level < pyramid_level
                        and level.level > closest_level.level
                    ),
                    None,
                )
                if closest_new_level is not None:
                    closest_level = closest_new_level
                scale = int(2 ** (pyramid_level - closest_level.level))
                new_level = self._save_and_open_level(
                    closest_level, levels.pixel_spacing, scale
                )
                new_levels.append(new_level)

    def save_labels(self, labels: Labels):
        """Save labels to target."""
        for label in labels.groups:
            self._save_group(label)

    def save_overviews(self, overviews: Overviews):
        """Save overviews to target."""
        for overview in overviews.groups:
            self._save_group(overview)

    def close(self) -> None:
        """Close any opened level files."""
        for file in self._opened_files:
            file.close()

    def _save_and_open_level(
        self, level: Level, base_pixel_spacing: SizeMm, scale: int = 1
    ) -> Level:
        """Save level and return a new level from the created files."""
        filepaths = self._save_group(level, scale)
        instances = self._open_files(filepaths)
        return Level(instances, base_pixel_spacing)

    def _save_group(self, group: Group, scale: int = 1) -> List[Path]:
        """Save group to target."""
        if not isinstance(scale, int) or scale < 1:
            raise ValueError(f"Scale must be positive integer, got {scale}.")
        filepaths: List[Path] = []
        for instances in self._group_instances_to_file(group):
            uid = self._uid_generator()
            filepath = Path(self._output_path).joinpath(uid + ".dcm")
            transfer_syntax = instances[0].image_data.transfer_syntax
            image_data_list = self._list_image_data(instances)
            focal_planes, optical_paths, tiled_size = self._get_frame_information(
                image_data_list
            )
            dataset = instances[0].dataset.as_tiled_full(
                focal_planes, optical_paths, tiled_size, scale
            )
            with WsiDicomFileWriter.open(filepath) as wsi_file:
                wsi_file.write(
                    uid,
                    transfer_syntax,
                    dataset,
                    image_data_list,
                    self._workers,
                    self._chunk_size,
                    self._offset_table,
                    self._instance_number,
                    scale,
                )
            filepaths.append(filepath)
            self._instance_number += 1
        self._filepaths.extend(filepaths)
        return filepaths

    def _open_files(self, filepaths: Iterable[Path]) -> List[WsiInstance]:
        files = [WsiDicomFile.open(filepath) for filepath in filepaths]
        self._opened_files.extend(files)
        return [
            WsiInstance([file.dataset for file in files], WsiDicomFileImageData(files))
        ]

    @staticmethod
    def _group_instances_to_file(group: Group) -> List[List[WsiInstance]]:
        """
        Group instances by properties that can't differ in a DICOM-file.

        Returns
        ----------
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
        ----------
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
        data: Dict[Tuple[str, float], ImageData]
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
