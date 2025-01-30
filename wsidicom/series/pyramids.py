from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from wsidicom.config import settings
from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.instance.instance import WsiInstance
from wsidicom.series.pyramid import Pyramid


class Pyramids:
    """A collection of pyramids."""

    def __init__(self, pyramids: Iterable[Pyramid]):
        self._pyramids = list(pyramids)

    def __getitem__(self, index: int) -> Pyramid:
        """Get pyramid by index.

        Parameters
        ----------
        index: int
            Index of pyramid to get

        Returns
        -------
        Pyramid
            The pyramid at index in the pyramids.
        """
        return self._pyramids[index]

    def __len__(self) -> int:
        return len(self._pyramids)

    def __iter__(self):
        return iter(self._pyramids)

    def get(self, index: int) -> Pyramid:
        """Get pyramid by index.

        Parameters
        ----------
        index: int
            Index of pyramid to get

        Returns
        -------
        Pyramid
            The pyramid at index in the pyramids.
        """
        try:
            return self[index]
        except IndexError:
            raise WsiDicomNotFoundError(f"Pyramid index {index}", "pyramids")

    @classmethod
    def open(
        cls, instances: Iterable[WsiInstance], thumbnails: Iterable[WsiInstance]
    ) -> "Pyramids":
        """Return Pyramids object created from wsi instances.

        Parameters
        ----------
        instances: Iterable[WsiInstance]
            Instances to create pyramids from.
        thumbnails: Iterable[WsiInstance]
            Instances to create thumbnails for pyramids from.

        Returns
        -------
        Pyramids
            Pyramids created from wsi instances. Each pyramid has the same
            image coordinate system and depth of field.
        """
        level_instances_grouped_by_pyramid = cls._group_instances_into_pyramids(
            instances
        )
        thumbnail_instances_grouped_by_pyramid = cls._group_instances_into_pyramids(
            thumbnails
        )
        pyramids = (
            Pyramid.open(
                level_instances,
                thumbnail_instances_grouped_by_pyramid.pop(group, []),
            )
            for group, level_instances in level_instances_grouped_by_pyramid.items()
        )

        return cls(pyramids)

    @classmethod
    def _group_instances_into_pyramids(cls, instances: Iterable[WsiInstance]) -> Dict[
        Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[int, float]]],
        List[WsiInstance],
    ]:
        """Return instances grouped and sorted by pyramid.

        Parameters
        ----------
        instances: Iterable[WsiInstance]
            Instances to group by pyramid

        Returns
        -------
        Dict[
            Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[int, float]]],
            List[WsiInstance],
        ]
            Instances grouped by image coordinate system and depth of field.

        """
        grouped_instances: Dict[
            Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[int, float]]],
            List[WsiInstance],
        ] = defaultdict(list)
        for instance in sorted(
            instances, key=lambda x: x.size.to_tuple(), reverse=True
        ):
            if instance.image_coordinate_system is not None:
                image_coordinate_system = (
                    instance.image_coordinate_system.origin.x,
                    instance.image_coordinate_system.origin.y,
                    instance.image_coordinate_system.rotation,
                )
            else:
                image_coordinate_system = None
            if instance.ext_depth_of_field:
                assert instance.ext_depth_of_field_planes is not None
                assert instance.ext_depth_of_field_plane_distance is not None
                ext_depth_of_field = (
                    instance.ext_depth_of_field_planes,
                    instance.ext_depth_of_field_plane_distance,
                )
            else:
                ext_depth_of_field = None

            if instance.image_coordinate_system is not None:
                existing_group_match = next(
                    (
                        (image_cs, ext_dof)
                        for (image_cs, ext_dof), group in grouped_instances.items()
                        if (
                            all(
                                instance.image_coordinate_system.origin_and_rotation_match(
                                    instance_in_group.image_coordinate_system,
                                    origin_threshold=settings.pyramids_origin_threshold,
                                )
                                for instance_in_group in group
                            )
                            and ext_depth_of_field == ext_dof
                        )
                    ),
                    None,
                )
                if existing_group_match is not None:
                    image_coordinate_system, ext_depth_of_field = existing_group_match

            grouped_instances[image_coordinate_system, ext_depth_of_field].append(
                instance
            )
        return grouped_instances
