from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple
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
    def open(cls, instances: Iterable[WsiInstance]) -> "Pyramids":
        # Sort instances by image coordinate system and extended focus
        # Create Levels for each group.
        # Create Pyramids from Levels.
        instances_grouped_by_pyramid = cls._group_instances_into_pyramids(instances)
        return cls(
            [Pyramid.open(instances) for instances in instances_grouped_by_pyramid]
        )

    @classmethod
    def _group_instances_into_pyramids(
        cls, instances: Iterable[WsiInstance]
    ) -> Iterable[List[WsiInstance]]:
        """Return instances grouped and sorted by pyramid.

        Parameters
        ----------
        instances: Iterable[WsiInstance]
            Instances to group by pyramid

        Returns
        -------
        Iterable[List[WsiInstance]]:
            Instances grouped by pyramid.

        """
        grouped_instances: Dict[
            Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[int, float]]],
            List[WsiInstance],
        ] = defaultdict(list)
        for instance in instances:
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
            grouped_instances[image_coordinate_system, ext_depth_of_field].append(
                instance
            )
        return grouped_instances.values()
