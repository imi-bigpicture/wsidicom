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

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    DefaultDict,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset, validate_file_meta
from pydicom.filereader import dcmread
from pydicom.filewriter import dcmwrite
from pydicom.sequence import Sequence as DicomSequence
from pydicom.uid import (
    ExplicitVRBigEndian,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    generate_uid,
)
from pydicom.values import convert_numbers

from wsidicom.conceptcode import (
    AnnotationCategoryCode,
    AnnotationTypeCode,
    MeasurementCode,
    UnitCode,
)

from .geometry import PointMm, RegionMm, SizeMm
from .uid import ANN_SOP_CLASS_UID, UID, SlideUids


@dataclass
class LabColor:
    l: int
    a: int
    b: int


def dcm_to_list(item: bytes, format: str) -> List[Any]:
    """Convert item to list of values using specified type format.

    Parameters
    ----------
    item: bytes
        Bytes to convert.
    format: str
        Struct format character.

    Returns
    ----------
    List[Any]
        List of values
    """
    converted = convert_numbers(item, is_little_endian=True, struct_format=format)
    if not isinstance(converted, list):
        return [converted]
    return converted


@dataclass
class Measurement:
    """Represents a measurement.

    Parameters
        ----------
        code: MeasurementCode
            Type of measurement.
        value: float
            Value of measurement.
        unit: UnitCode
            Unit of measurement.
    """

    code: MeasurementCode
    value: float
    unit: UnitCode

    def same_type(self, other_code: MeasurementCode, other_unit: UnitCode):
        """Return true if measurement base is same.

        Parameters
        ----------
        other_code: MeasurementCode
            Other base to compare to.
        other_unit: UnitCode
            Other base to compare to.

        Returns
        ----------
        bool
            True if bases are same.
        """
        return self.code == other_code and self.unit == other_unit

    @staticmethod
    def _get_measurement_type_from_ds(ds: Dataset) -> Tuple[MeasurementCode, UnitCode]:
        """Get measurement type from dataset.

        Parameters
        ----------
        ds: Dataset
            Dataset containing measurement sequence item (group).

        Returns
        ----------
        MeasurementType
            Measurement type of the measurement group.
        """
        code = MeasurementCode.from_ds(ds)
        if code is None:
            raise ValueError(f"Dataset is missing {MeasurementCode.sequence_name}.")
        unit = UnitCode.from_ds(ds)
        if unit is None:
            raise ValueError(f"Dataset is missing {UnitCode.sequence_name}.")
        return code, unit

    @staticmethod
    def _get_measurement_values_from_ds(ds: Dataset) -> List[float]:
        """Get measurement values from dataset.

        Parameters
        ----------
        ds: Dataset
            Dataset containing measurement sequence item (group).

        Returns
        ----------
        List[float]
            Measurement values of the measurement group.
        """
        return dcm_to_list(ds.MeasurementValuesSequence[0].FloatingPointValues, "f")

    @staticmethod
    def _get_measurement_indices_from_ds(
        ds: Dataset,
        annotation_count: int,
    ) -> List[int]:
        """Get measurement indices from dataset. Annotation index is stored as
        starting at index 1. If Annotation index list is
        missing measurements are one-to-one to annotations.

        Parameters
        ----------
        ds: Dataset
            Dataset containing measurement sequence item (group).
        annotation_count: int
            Number of annotations.

        Returns
        ----------
        List[int]
            Measurement indices of the measurement group.
        """
        measurement_ds = ds.MeasurementValuesSequence[0]
        if "AnnotationIndexList" in measurement_ds:
            indices = dcm_to_list(measurement_ds.AnnotationIndexList, "l")
            return [index - 1 for index in indices]
        return list(range(annotation_count))

    @classmethod
    def _iterate_measurement_sequence(
        cls, sequence: DicomSequence, annotation_count: int
    ) -> Iterator[Tuple[int, "Measurement"]]:
        """Return generator for measurements in dataset. Yields a tuple of
        the index of the annotation for the measurement and the measurement.

        Parameters
        ----------
        sequence: DicomSequence
            Dicom measurement sequence.
        annotation_count: int
            Number of annotations in group.

        Returns
        ----------
        Iterator[Tuple[int, 'Measurement']]:
            Generator for annotation index and measurement.
        """
        for group_ds in sequence:
            code, unit = cls._get_measurement_type_from_ds(group_ds)
            values = cls._get_measurement_values_from_ds(group_ds)
            indices = cls._get_measurement_indices_from_ds(group_ds, annotation_count)
            if len(values) != len(indices):
                raise ValueError("number of indices needs to be same as measurements")
            for index, annotation_index in enumerate(indices):
                yield (annotation_index, Measurement(code, values[index], unit))

    @classmethod
    def get_measurements_from_ds(
        cls, sequence: DicomSequence, annotation_count: int
    ) -> Dict[int, List["Measurement"]]:
        """Get measurements from dataset.

        Parameters
        ----------
        sequence: DicomSequence
            Dicom measurement sequence.
        annotation_count: int
            Number of annotations in group.

        Returns
        ----------
        Dict[int, List[Measurement]]
            Dict of measurements grouped by annotation number as key.
        """
        measurements: DefaultDict[int, List[Measurement]] = defaultdict(list)
        for annotation_index, measurement in cls._iterate_measurement_sequence(
            sequence, annotation_count
        ):
            measurements[annotation_index].append(measurement)
        return measurements


class Geometry(metaclass=ABCMeta):
    name: str

    @property
    @abstractmethod
    def data(self) -> List[float]:
        """Return geometry content as a list of floats"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def box(self) -> RegionMm:
        """Return Region that contains the geometry."""
        raise NotImplementedError()

    @abstractmethod
    def to_coords(self) -> List[Tuple[float, float]]:
        """Return geometry content as a list of tuple of floats"""
        raise NotImplementedError()

    @abstractmethod
    def to_list_coords(self) -> List[List[float]]:
        raise NotImplementedError()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_list(cls, list: Sequence[float]) -> "Geometry":
        """Return geometry object created from list of floats"""
        raise NotImplementedError()

    @classmethod
    def list_to_coords(cls, data: Sequence[float]) -> List[Tuple[float, float]]:
        """Return coordinates in list of floats as list of tuple of floats

        Parameters
        ----------
        data: Sequence[float]
            List of float to convert.

        Returns
        ----------
        List[Tuple[float, float]]
            List of coordinates (Tuple of floats).
        """
        x_indices = range(0, len(data), 2)
        y_indices = range(1, len(data), 2)
        return [(float(data[x]), float(data[y])) for x, y in zip(x_indices, y_indices)]

    @staticmethod
    def _coordinates_from_dict(
        dictionary: Union[
            Dict[str, Union[str, float]], Sequence[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str,
    ) -> List[Tuple[float, float]]:
        """Return coordinates in dictionary as list of tuple of floats

        Parameters
        ----------
        dictionary: Union[
            Dict[str, Union[str, float]],
            Sequence[Dict[str, Union[str, float]]]
            ]:
            Dictionary to convert.
        x: str
            Name of the x coordinate.
        y: str
            Name of the y coordinate.

        Returns
        ----------
        List[Tuple[float, float]]
            List of coordinates (Tuple of floats).
        """
        if not isinstance(dictionary, Sequence):
            coordinates = [dictionary]
        else:
            coordinates = dictionary
        return [
            (float(coordinate[x]), float(coordinate[y])) for coordinate in coordinates
        ]

    @classmethod
    @abstractmethod
    def from_dict(
        cls,
        dictionary: Union[
            Dict[str, Union[str, float]], Sequence[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str,
    ) -> "Geometry":
        raise NotImplementedError()

    @classmethod
    def from_shapely_like(cls, item: Any) -> "Geometry":
        """Return Geometry from shapely-like object. Object needs to have
        shapely-like attributes.

        Parameters
        ----------
        item
            Object with shapely-like attributes.

        Returns
        ----------
        Geometry
            Geometry created from object.
        """
        try:
            geometry_type = getattr(item, "geom_type")
            if geometry_type == "Point":
                coordinates = getattr(item, "coords")
                return Point(*coordinates[0])
            elif geometry_type == "Polygon":
                interiors = getattr(item, "interiors")
                exterior = getattr(item, "exterior")
                exterior_coords = getattr(exterior, "coords")
                if len(interiors) == 0:
                    return Polygon(exterior_coords)
            elif geometry_type == "LineString":
                coordinates = getattr(item, "coords")
                return Polyline(coordinates)
            raise NotImplementedError("Not a supported shapely like object")
        except AttributeError as exception:
            raise ValueError("Not a shapely like object") from exception

    @classmethod
    def from_geojson(cls, dictionary: Dict[str, Any]) -> List["Geometry"]:
        """Return geometries from geojson geometry dictionary. Note that
        MultiPoint returns multiple points.

        Parameters
        ----------
        dictionary
            Geojson geometry dictionary.

        Returns
        ----------
        Geometry
            Geometry created from Geojson dictionary.
        """
        try:
            coordinates = dictionary["coordinates"]
            annotation_type = dictionary["type"]
        except KeyError:
            raise ValueError("Not a geojson object")
        if not isinstance(coordinates, list) or not isinstance(annotation_type, str):
            raise ValueError("Not a geojson object")
        if annotation_type == "Point":
            return [Point.from_list(coordinates)]
        elif annotation_type == "MultiPoint":
            return [Point.from_list(point) for point in coordinates]
        elif annotation_type == "Polygon":
            return [Polygon(coordinates[0])]
        elif annotation_type == "LineString":
            return [Polyline(coordinates)]
        raise NotImplementedError("Not supported geojson geometry")


@dataclass
class Point(Geometry):
    """Geometry consisting of a single point"""

    name = "POINT"

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, point: "Point") -> bool:
        if isinstance(point, Point):
            return self.x == point.x and self.y == point.y
        else:
            raise NotImplementedError("Comparing Point to non-Point")

    @property
    def data(self) -> List[float]:
        return [self.x, self.y]

    @property
    def box(self) -> RegionMm:
        return RegionMm(PointMm(self.x, self.y), SizeMm(0, 0))

    def __str__(self) -> str:
        return f"x: {self.x}, y: {self.y}"

    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"

    def to_coords(self) -> List[Tuple[float, float]]:
        return [(self.x, self.y)]

    def to_list_coords(self) -> List[List[float]]:
        return [[self.x, self.y]]

    def __len__(self) -> int:
        return 1

    @classmethod
    def from_list(cls, list: List[float]) -> "Point":
        coordinates = cls.list_to_coords(list)
        return cls(*coordinates[0])

    @classmethod
    def multiple_from_dict(
        cls,
        dictionary: Union[
            Dict[str, Union[str, float]], Sequence[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str,
    ) -> List["Point"]:
        coordinates = cls._coordinates_from_dict(dictionary, x, y)
        return [cls(*point) for point in coordinates]

    @classmethod
    def from_dict(
        cls,
        dictionary: Union[
            Dict[str, Union[str, float]], Sequence[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str,
    ) -> "Point":
        coordinates = cls._coordinates_from_dict(dictionary, x, y)
        return cls(*coordinates[0])


@dataclass
class Polyline(Geometry):
    """Geometry consisting of connected lines."""

    name = "POLYLINE"

    def __init__(self, points: Sequence[Tuple[float, float]]):
        self.points: List[Point] = [Point(point[0], point[1]) for point in points]

    def __len__(self) -> int:
        return len(self.points)

    def __repr__(self) -> str:
        return f"Polyline({self.to_coords()})"

    def to_coords(self) -> List[Tuple[float, float]]:
        return [(point.x, point.y) for point in self.points]

    def to_list_coords(self) -> List[List[float]]:
        return [[point.x, point.y] for point in self.points]

    @property
    def data(self) -> List[float]:
        return [value for point in self.points for value in point.data]

    @property
    def box(self) -> RegionMm:
        top: float = self.points[0].y
        left: float = self.points[0].x
        bottom: float = self.points[0].y
        right: float = self.points[0].x
        for point in self.points[1:]:
            if point.y > top:
                top = point.y
            elif point.y < bottom:
                bottom = point.y
            if point.x > right:
                right = point.x
            elif point.y < left:
                left = point.x
        return RegionMm(PointMm(left, bottom), SizeMm(right - left, top - bottom))

    @classmethod
    def from_dict(
        cls,
        dictionary: Union[
            Dict[str, Union[str, float]], Sequence[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str,
    ) -> "Polyline":
        coordinates = cls._coordinates_from_dict(dictionary, x, y)
        return cls(coordinates)

    @classmethod
    def from_list(cls, list: Sequence[float]) -> "Polyline":
        coordinates = cls.list_to_coords(list)
        return cls(coordinates)


@dataclass
class Polygon(Polyline):
    """Geometry consisting of connected lines implicitly closed."""

    name = "POLYGON"

    def __init__(self, points: Sequence[Tuple[float, float]]):
        super().__init__(points)

    @classmethod
    def from_list(cls, list: Sequence[float]) -> "Polygon":
        coordinates = cls.list_to_coords(list)
        return cls(coordinates)

    @classmethod
    def from_dict(
        cls,
        dictionary: Union[
            Dict[str, Union[str, float]], Sequence[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str,
    ) -> "Polygon":
        coordinates = cls._coordinates_from_dict(dictionary, x, y)
        return cls(coordinates)

    def __repr__(self) -> str:
        return f"Polygon({self.to_coords()})"


class Annotation:
    def __init__(
        self, geometry: Geometry, measurements: Optional[Sequence[Measurement]] = None
    ):
        """Represents an annotation, with geometry and an optional list of
        measurements.

        Parameters
        ----------
        geometry: Geometry
            Geometry of the annotation.
        measurements: Optional[Sequence[Measurement]]
            Optional measurements of the annotation.
        """

        self._geometry = geometry
        if measurements is None:
            measurements = []
        self._measurements = measurements

    def __repr__(self) -> str:
        return f"Annotation({self.geometry}, {self.measurements})"

    def __eq__(self, other: "Annotation") -> bool:
        if not isinstance(other, Annotation):
            return NotImplemented
        return (
            self.geometry == other.geometry and self.measurements == other.measurements
        )

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def measurements(self) -> Sequence[Measurement]:
        return self._measurements

    def get_measurements(
        self, code: MeasurementCode, unit: UnitCode
    ) -> List[Measurement]:
        """Return measurements of specified type.

        Parameters
        ----------
        code: MeasurementCode
            Code of measurement type to get.
        unit: UnitCode
            Unit of measurement type to get.

        Returns
        ----------
        List[Measurement]
            List of measurements of specified type.
        """
        return [
            measurement
            for measurement in self.measurements
            if measurement.same_type(code, unit)
        ]

    def get_measurement_values(
        self, code: MeasurementCode, unit: UnitCode
    ) -> List[float]:
        """Return values for measurements of specified type.

        Parameters
        ----------
        code: MeasurementCode
            Code of measurement type to get.
        unit: UnitCode
            Unit of measurement type to get.

        Returns
        ----------
        List[Measurement]
            List of values for measurements of specified type.
        """
        return [
            measurement.value
            for measurement in self.measurements
            if measurement.same_type(code, unit)
        ]


GeometryType = TypeVar("GeometryType", bound=Geometry)
AnnotationGroupType = TypeVar("AnnotationGroupType", bound="AnnotationGroup")


class AnnotationGroup(Generic[GeometryType]):
    _geometry_type: Type[Geometry]

    def __init__(
        self,
        annotations: Sequence[Annotation],
        label: str,
        category_code: AnnotationCategoryCode,
        type_code: AnnotationTypeCode,
        description: Optional[str] = None,
        color: Optional[LabColor] = None,
        is_double: bool = True,
        instance: Optional[UID] = None,
    ):
        """Represents a group of annotations of the same type.

        Parameters
        ----------
        annotations: Sequence[Annotation]
            Annotations in the group.
        label: str
            Group label
        category_code: AnnotationCategoryCode
            Group category code.
        type_code: AnnotationTypeCode
            Group type code.
        description: Optional[str] = None
            Group description.
        color: Optional[LabColor] = None
            Recommended CIELAB color.
        is_double: bool
            If group is stored with double float
        instance: Optional[Uid]
            Uid this group was created from.

        """
        self.validate_type(annotations)
        self._z_planes: List[float] = []
        self._optical_paths: List[str] = []
        self._uid = generate_uid()
        self._category_code = category_code
        self._type_code = type_code
        self._is_double = is_double
        if self._is_double:
            self._point_data_type = np.float64
        else:
            self._point_data_type = np.float32

        self._annotations = annotations
        self._label = label
        self._description = description
        self._color = color
        self._instance = instance

    def __len__(self) -> int:
        return len(self.annotations)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.annotations}, {self.label}, "
            f"{self.category_code}, {self.type_code}, "
            f"{self.description}, {self._is_double})"
        )

    def __eq__(self, other: "AnnotationGroup") -> bool:
        if not isinstance(other, AnnotationGroup):
            return NotImplemented
        return (
            self.annotations == other.annotations
            and self.label == other.label
            and self.type_code == other.type_code
            and self.category_code == other.category_code
            and self.description == other.description
            and self.color == other.color
            and self._is_double == other._is_double
        )

    @property
    def category_code(self) -> AnnotationCategoryCode:
        return self._category_code

    @property
    def type_code(self) -> AnnotationTypeCode:
        return self._type_code

    @property
    def annotations(self) -> Sequence[Annotation]:
        return self._annotations

    @property
    def label(self) -> str:
        return self._label

    @property
    def description(self) -> Optional[str]:
        return self._description

    @property
    def color(self) -> Optional[LabColor]:
        return self._color

    @property
    def point_coordinates_data(self) -> np.ndarray:
        """Return coordinates for annotations in group np array.

        Returns
        ----------
        np.ndarray
            Coordinates in annotation group.
        """
        coordinate_list: List[float] = []
        for annotation in self.annotations:
            coordinate_list += annotation.geometry.data
        return np.array(coordinate_list, dtype=self._point_data_type)

    @property
    def number_of_annotations(self) -> int:
        """Return number of annotations in group.

        Returns
        ----------
        int
            Number of annotations in group.
        """
        return len(self.annotations)

    @property
    def geometry_type(self) -> Type[Geometry]:
        return self._geometry_type

    @property
    def annotation_type(self) -> str:
        return self._geometry_type.name

    def __getitem__(
        self, index: Union[int, Sequence[int]]
    ) -> Union[Annotation, List[Annotation]]:
        if isinstance(index, Sequence):
            return [self.annotations[i] for i in index]
        return self.annotations[index]

    @classmethod
    def from_ds(
        cls: Type[AnnotationGroupType], ds: Dataset, instance: UID
    ) -> AnnotationGroupType:
        """Return annotation group from Annotation Group Sequence dataset.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group.
        instance: Uid
            Uid this group was created from.

        Returns
        ----------
        AnnotationGroupType
            Annotation group from dataset.
        """
        is_double = cls._is_ds_double(ds)
        annotations = cls._get_annotations_from_ds(ds)
        label = cls._get_label_from_ds(ds)
        description = getattr(ds, "AnnotationGroupDescription", None)
        type_code = AnnotationTypeCode.from_ds(ds)
        if type_code is None:
            raise ValueError(f"Dataset is missing {AnnotationTypeCode.sequence_name}.")
        category_code = AnnotationCategoryCode.from_ds(ds)
        if category_code is None:
            raise ValueError(
                f"Dataset is missing {AnnotationCategoryCode.sequence_name}."
            )
        color = getattr(ds, "RecommendedDisplayCIELabValue", None)
        return cls(
            annotations,
            label,
            category_code,
            type_code,
            description,
            color,
            is_double,
            instance,
        )

    @staticmethod
    def _get_label_from_ds(ds: Dataset) -> str:
        """Return group label from dataset.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group label.

        Returns
        ----------
        str
            Annotation group label from dataset.
        """
        return str(ds.AnnotationGroupLabel)

    @staticmethod
    def _get_focal_planes_from_ds(ds: Dataset) -> List[float]:
        """Return focal planes from dataset. If annotation applies to all focal
        planes returns empty list.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group focal planes.

        Returns
        ----------
        List[float]
            Annotation group focal planes from dataset.
        """
        if ds.AnnotationAppliesToAllZPlanes == "YES":
            return []
        try:
            z_planes: List[float] = ds.CommonZCoordinateValue
        except AttributeError:
            raise NotImplementedError(
                "Only 3D annotations with common z coordinate is supported"
            )
        return z_planes

    @staticmethod
    def _get_optical_paths_from_ds(ds: Dataset) -> List[str]:
        """Return optical paths from dataset. If annotation applies to all
        optical paths returns empty list.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group optical paths.

        Returns
        ----------
        List[str]
            Annotation group optical paths from dataset.
        """
        if ds.AnnotationAppliesToAllOpticalPaths == "YES":
            return []
        optical_paths: List[str] = ds.ReferencedOpticalPathIdentifier
        return optical_paths

    @staticmethod
    def _get_count_from_ds(ds: Dataset) -> int:
        """Return number of annotations in group.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group.

        Returns
        ----------
        int
            Number of annotations in group
        """
        return int(ds.NumberOfAnnotations)

    @classmethod
    def _get_measurements_from_ds(
        cls,
        ds: Dataset,
    ) -> Dict[int, List[Measurement]]:
        """Return measurements grouped by annotation index from ds.

        Parameters
        ----------
        ds: Dataset
            Dataset containing measurement sequence.

        Returns
        ----------
        Dict[int, List[Measurement]]:
            Measurements grouped by annotation index.
        """
        if "MeasurementsSequence" not in ds:
            return {}
        annotation_count = cls._get_count_from_ds(ds)
        return Measurement.get_measurements_from_ds(
            ds.MeasurementsSequence, annotation_count
        )

    @staticmethod
    def _is_ds_double(ds: Dataset) -> bool:
        """Return true if group in dataset stores coordinates as double float.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group.

        Returns
        ----------
        bool
            True if group uses double float.
        """
        return "DoublePointCoordinatesData" in ds

    @classmethod
    def _get_coordinates_from_ds(cls, ds: Dataset) -> List[Tuple[float, float]]:
        """Return annotation coordinates.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group.

        Returns
        ----------
        List[float]
            The coordinates for the annotations in the group.
        """
        if cls._is_ds_double(ds):
            data = dcm_to_list(ds.DoublePointCoordinatesData, "d")
        else:
            data = dcm_to_list(ds.PointCoordinatesData, "f")
        return Geometry.list_to_coords(data)

    @classmethod
    @abstractmethod
    def _get_geometries_from_ds(cls, ds: Dataset) -> List[Geometry]:
        """Abstract method for getting geometries from dataset.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group.

        Returns
        ----------
        List[Geometry]
            Geometries in the annotation group.
        """
        raise NotImplementedError()

    @classmethod
    def _get_annotations_from_ds(cls, ds: Dataset) -> List[Annotation]:
        """Return annotation coordinates.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group.

        Returns
        ----------
        List[float]
            The coordinates for the annotations in the group.
        """
        measurements = cls._get_measurements_from_ds(ds)
        geometries = cls._get_geometries_from_ds(ds)

        return [
            Annotation(geometry, measurements.get(i))
            for i, geometry in enumerate(geometries)
        ]

    @property
    def measurement_types(self) -> List[Tuple[MeasurementCode, UnitCode]]:
        """Return measurement types in annotation group.

        Returns
        ----------
        List[Tuple[MeasurementCode, UnitCode]]
            Measurement code-unit-pairs in annotation group.
        """
        pairs: Set[Tuple[MeasurementCode, UnitCode]] = set()
        for annotation in self.annotations:
            for measurement in annotation.measurements:
                pair = (measurement.code, measurement.unit)
                pairs.add(pair)
        return list(pairs)

    def get_measurements(
        self, code: MeasurementCode, unit: UnitCode
    ) -> List[Measurement]:
        """Return measurements of specified code-unit-pair.

        Parameters
        ----------
        code: MeasurementCode
            Type of measurement type to get.
        unit: UnitCode
            Unit of measurement type to get.

        Returns
        ----------
        List[Measurement]
            Measurements of specified type.
        """
        measurements: List[Measurement] = []
        for annotation in self.annotations:
            measurements += annotation.get_measurements(code, unit)
        return measurements

    def create_measurement_indices(
        self, code: MeasurementCode, unit: UnitCode
    ) -> np.ndarray:
        """Return measurement indices for all measurements of specified type.
        Indices are stored starting at index 1.

        Parameters
        ----------
        code: MeasurementCode
            Type of measurement type to create.
        unit: UnitCode
            Unit of measurement type to create.

        Returns
        ----------
        np.ndarray
            Measurement indices in annotation group of specified type.
        """
        indices: List[int] = []
        for i, annotation in enumerate(self.annotations):
            indices += [i + 1] * len(annotation.get_measurements(code, unit))
        return np.array(indices, dtype=np.int32)

    def _create_measurement_values(
        self, code: MeasurementCode, unit: UnitCode
    ) -> np.ndarray:
        """Return measurement values for all measurements of specified type.

        Parameters
        ----------
        code: MeasurementCode
            Code of measurement type to create.
        unit: UnitCode
            Unit of measurement type to create.

        Returns
        ----------
        np.ndarray
            Measurement values in annotation group of specified type.
        """
        values: List[float] = []
        for annotation in self.annotations:
            values += annotation.get_measurement_values(code, unit)
        return np.array(values, dtype=np.float32)

    def _measurements_is_one_to_one(
        self, code: MeasurementCode, unit: UnitCode
    ) -> bool:
        """Check if all annotations in group have strictly one measurement
        of specified type-unit-pair.

        Parameters
        ----------
        code: MeasurementCode
            Code of measurement type to check.
        unit: UnitCode
            Unit of measurement type to check.

        Returns
        ----------
        bool
            True if this measurement code-unit-pair maps one-to-one to
            annotations.
        """
        for annotation in self.annotations:
            if not len(annotation.get_measurements(code, unit)) == 1:
                return False
        return True

    def _create_measurement_value_sequence(
        self, code: MeasurementCode, unit: UnitCode
    ) -> DicomSequence:
        """Return Measurement Value Sequence of measurements of specified
        code and unit.

        Parameters
        ----------
        code: MeasurementCode
            Code of measurement create sequence for.
        unit: UnitCode
            Unit of measurement create sequence for.

        Returns
        ----------
        DicomSequence
            A sequence of the measurement values of specified type and unit.
        """
        measurements_ds = Dataset()
        # According to the standard the indices is not needed if each
        # annotation has one and only one measurement.
        if not self._measurements_is_one_to_one(code, unit):
            indices = self.create_measurement_indices(code, unit)
            measurements_ds.AnnotationIndexList = indices
        values = self._create_measurement_values(code, unit)
        measurements_ds.FloatingPointValues = values
        measurement_value_sequence = DicomSequence([measurements_ds])
        return measurement_value_sequence

    def _create_measurement_sequence_item(
        self, code: MeasurementCode, unit: UnitCode
    ) -> Dataset:
        """Return Measurement Sequence item of measurements of specified code
        and unit.

        Parameters
        ----------
        code: MeasurementCode
            Code of measurement to create sequence for.
        unit: UnitCode
            Unit of measurement to create sequence for.

        Returns
        ----------
        Dataset
            A measurement sequence item for the specified measurement type and
            unit.
        """
        ds = Dataset()
        ds.MeasurementValuesSequence = self._create_measurement_value_sequence(
            code, unit
        )
        code.insert_into_ds(ds)
        unit.insert_into_ds(ds)
        return ds

    def _set_planes_in_ds(self, ds: Dataset) -> Dataset:
        """Insert the group focal plane attributes into the Annotation Group
        Sequence.

        Parameters
        ----------
        ds: Dataset
            The Annotation Group Sequence.

        Returns
        ----------
        Dataset
            The Annotation Group Sequence with focal plane attributes.
        """
        if len(self._z_planes) == 0:
            ds.AnnotationAppliesToAllZPlanes = "YES"
        else:
            ds.AnnotationAppliesToAllZPlanes = "NO"
            ds.CommonZCoordinateValue = self._z_planes
        return ds

    def _set_optical_paths_in_ds(self, ds: Dataset) -> Dataset:
        """Insert the group optical path attributes into the Annotation Group
        Sequence.

        Parameters
        ----------
        ds: Dataset
            The Annotation Group Sequence.

        Returns
        ----------
        Dataset
            The Annotation Group Sequence with optical path attributes.
        """
        if len(self._optical_paths) == 0:
            ds.AnnotationAppliesToAllOpticalPaths = "YES"
        else:
            ds.AnnotationAppliesToAllOpticalPaths = "NO"
            ds.ReferencedOpticalPathIdentifier = self._optical_paths
        return ds

    def _set_coordinates_data_in_ds(self, ds: Dataset) -> Dataset:
        """Insert the group point coordinates into the Annotation Group
        Sequence.

        Parameters
        ----------
        ds: Dataset
            The Annotation Group Sequence.

        Returns
        ----------
        Dataset
            The Annotation Group Sequence with inserted coordinates.
        """
        if self._is_double:
            ds.DoublePointCoordinatesData = self.point_coordinates_data
        else:
            ds.PointCoordinatesData = self.point_coordinates_data
        return ds

    def _set_measurement_sequence_in_ds(self, ds: Dataset) -> Dataset:
        """Insert group measurements into the Annotation Group Sequence.

        Parameters
        ----------
        ds: Dataset
            The Annotation Group Sequence.

        Returns
        ----------
        Dataset
            The Annotation Group Sequence with inserted measurements.
        """
        ds.MeasurementsSequence = DicomSequence(
            [
                self._create_measurement_sequence_item(*measurement_type)
                for measurement_type in self.measurement_types
            ]
        )
        return ds

    def to_ds(self, group_number: int) -> Dataset:
        """Return annotation group as a Annotation Group Sequence
        item.

        Parameters
        ----------
        group_number: int
            The index of the group.

        Returns
        ----------
        Dataset
            Dataset containing the group.
        """
        ds = Dataset()
        ds.AnnotationGroupNumber = group_number
        ds.AnnotationGroupUID = self._uid
        ds.NumberOfAnnotations = len(self.annotations)
        ds.GraphicType = self.annotation_type
        ds.AnnotationGroupLabel = self.label
        if self.description is not None:
            ds.AnnotationGroupDescription = self.description
        ds = self.category_code.insert_into_ds(ds)
        ds = self.type_code.insert_into_ds(ds)
        ds = self._set_coordinates_data_in_ds(ds)
        ds = self._set_planes_in_ds(ds)
        ds = self._set_optical_paths_in_ds(ds)
        ds = self._set_measurement_sequence_in_ds(ds)
        if self.color is not None:
            ds.RecommendedDisplayCIELabValue = self.color
        # AUTOMATIC and SEMIAUTOMATIC requires a
        # Annotation Algorithm Identification Sequence
        ds.AnnotationGroupGenerationType = "MANUAL"
        return ds

    @classmethod
    def validate_type(cls, annotations: Sequence[Annotation]):
        """Check that list of annotations are of the requested type.

        Parameters
        ----------
        annotations: Sequence[Annotation]
            List of annotations to check
        """
        for annotation in annotations:
            if not isinstance(annotation.geometry, cls._geometry_type):
                raise TypeError(
                    f"annotation type {type(annotation.geometry)}"
                    f" does not match Group type code {cls._geometry_type}"
                )

    @classmethod
    def _get_group_type_by_geometry(
        cls, geometry_type: Type[Geometry]
    ) -> Type["AnnotationGroup"]:
        """Return AnnotationGroup class for geometry type.

        Parameters
        ----------
        geometry_type: type
            The geometry type to get AnnotationGroup class for.

        """
        if geometry_type == Point:
            return PointAnnotationGroup
        elif geometry_type == Polyline:
            return PolylineAnnotationGroup
        elif geometry_type == Polygon:
            return PolygonAnnotationGroup
        raise NotImplementedError(f"{type(geometry_type)} is not supported")

    @classmethod
    def from_geometries(
        cls,
        geometries: Sequence[Geometry],
        label: str,
        category_code: AnnotationCategoryCode,
        type_code: AnnotationTypeCode,
        is_double: bool = True,
    ) -> "AnnotationGroup":
        """Return AnnotationGroup created from list of geometries. The group
        type is determined by the first geometry, and all geometries needs to
        have the same type.

        Parameters
        ----------
        geometries: Sequence[Geometries]
            Geometries in the group.
        label: str
            Group label
        category_code: AnnotationCategoryCode
            Group category code.
        type_code: AnnotationTypeCode
            Group type code.

        Returns
        ----------
        AnnotationGroup
            Group created from geometries.

        """
        geometry_type = type(geometries[0])
        group_type = cls._get_group_type_by_geometry(geometry_type)
        group: AnnotationGroup = group_type(
            annotations=[Annotation(geometry) for geometry in geometries],
            label=label,
            category_code=category_code,
            type_code=type_code,
            is_double=is_double,
        )
        return group


class PointAnnotationGroup(AnnotationGroup[Point]):
    """Point annotation group"""

    _geometry_type = Point

    @classmethod
    def _get_geometries_from_ds(cls, ds: Dataset) -> List[Point]:
        """Returns point geometries from dataset.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group.

        Returns
        ----------
        List[Geometry]
            Point geometries in the annotation group.
        """
        coordinate_list = cls._get_coordinates_from_ds(ds)
        points = [Point(*coordinates) for coordinates in coordinate_list]
        return points


class PolylineAnnotationGroupMeta(AnnotationGroup[GeometryType]):
    """Meta class for line annotation group"""

    geometry_type: Type[Geometry]

    @property
    def point_index_list(self) -> np.ndarray:
        """Return point index list for annotations in group. Indices are stored
        starting at index 1 and in relation to geometry data length.

        Returns
        ----------
        np.ndarray
            List of indices in annotation group
        """
        index = 1
        indices: List[int] = []
        for annotation in self.annotations:
            indices.append(index)
            index += len(annotation.geometry.data)
        return np.array(indices, dtype=np.int32)

    @staticmethod
    def _get_indices_from_ds(ds: Dataset) -> List[int]:
        """Return line start indices from sup 222 dataset. Indices are stored
        starting at with value 1, and are in relation to non-pared coordinates.
        Returned list starts at 0 and is in relation to paired coordinates.

        Parameters
        ----------
        ds: Dataset
            Dataset containing sup 222 indices.

        Returns
        ----------
        List[int]
            List of indices in dataset.
        """
        return [
            (value - 1) // 2
            for value in dcm_to_list(ds.LongPrimitivePointIndexList, "l")
        ]

    @classmethod
    def _get_geometries_from_ds(cls, ds: Dataset) -> List[GeometryType]:
        """Returns line geometries from dataset. Each line geometry consists of
        multiple points, and the first coordinate in the coordinate list is

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group.

        Returns
        ----------
        List[Geometry]
            Polyline geometries in the annotation group.
        """
        indices = cls._get_indices_from_ds(ds)
        number_of_geometries = ds.NumberOfAnnotations
        if number_of_geometries != len(indices):
            raise ValueError(
                "Number of indices must be the same as number of geometries"
            )
        coordinates = cls._get_coordinates_from_ds(ds)
        indices += [len(coordinates)]  # Add end for last geometry
        geometries: List[GeometryType] = []
        for index in range(number_of_geometries):
            start = indices[index]
            end = indices[index + 1]
            line_coordinates = coordinates[start:end]
            lines = cls._get_line_geometry_from_coords(line_coordinates)
            geometries.append(lines)
        return geometries

    @staticmethod
    @abstractmethod
    def _get_line_geometry_from_coords(
        coordinates: Sequence[Tuple[float, float]]
    ) -> GeometryType:
        raise NotImplementedError()

    def to_ds(self, group_number: int) -> Dataset:
        """Return annotation group as a Annotation Group Sequence item.

        Parameters
        ----------
        group_number: int
            The index of the group.

        Returns
        ----------
        Dataset
            Dataset containing the group.
        """
        ds = super().to_ds(group_number)
        ds.LongPrimitivePointIndexList = self.point_index_list
        return ds


class PolylineAnnotationGroup(PolylineAnnotationGroupMeta[Polyline]):
    _geometry_type = Polyline

    @staticmethod
    def _get_line_geometry_from_coords(coordinates: Sequence[Tuple[float, float]]):
        return Polyline(coordinates)


class PolygonAnnotationGroup(PolylineAnnotationGroupMeta[Polygon]):
    _geometry_type = Polygon

    @staticmethod
    def _get_line_geometry_from_coords(coordinates: Sequence[Tuple[float, float]]):
        return Polygon(coordinates)


class AnnotationInstance:
    """Class for handling microscope bulk simple annotations according to
    sup-222. Point, polyline, and polygon graphic types are implemented,
    ellipse and rectangle graphic types are not implemented. Annotation must
    have common z-coordinate or apply to all z planes (annotation with 3D
    PointCoordinateData is not implemented.)
    """

    def __init__(
        self,
        groups: Sequence[AnnotationGroup],
        coordinate_type: str,
        slide_uids: SlideUids,
    ):
        """Represents a collection of annotation groups.

        Parameters
        ----------
        annotations: Sequence[AnnotationGroup]
            List of annotations group
        coordinate_type: str
            If coordinates are volume-related ('volume') or image-related
            ('image').
        frame_of_referenc: Uid
            Frame of reference uid of image that the annotations belong to
        """
        self.groups = groups
        if coordinate_type not in ["image", "volume"]:
            raise ValueError("Coordinate type should be 'image' or 'volume'")
        self.coordinate_type = coordinate_type
        self.slide_uids = slide_uids
        self.datetime = datetime.now()
        self.modality = "ANN"
        self.series_number: int

    def __repr__(self) -> str:
        return f"AnnotationInstance({self.groups}, " f"{self.slide_uids})"

    def save(
        self,
        path: Union[str, Path],
        little_endian: bool = True,
        implicit_vr: bool = False,
        uid_generator: Callable[..., UID] = generate_uid,
    ):
        """Write annotations to DICOM file according to sup 222.
        Note that the file will miss DICOM attributes that has not yet been
        implemented.

        Parameters
        ----------
        path: Union[str, Path]
            Path to write DICOM file to
        little_endian: bool
            Write DICOM file as little endian
        implicit_vr: bool
            Write DICOM file with implicit value representation
        """
        ds = Dataset()
        ds.is_little_endian = little_endian
        ds.is_implicit_VR = implicit_vr
        bulk_sequence = DicomSequence()
        for index, annotation_group in enumerate(self.groups):
            if isinstance(annotation_group, AnnotationGroup):
                bulk_sequence.append(annotation_group.to_ds(index + 1))
            else:
                raise NotImplementedError(
                    f"Group type: {type(annotation_group)} not supported"
                )
        ds.AnnotationGroupSequence = bulk_sequence
        if self.coordinate_type == "image":
            ds.AnnotationCoordinateType = "2D"
        elif self.coordinate_type == "volume":
            ds.AnnotationCoordinateType = "3D"
        if self.slide_uids.frame_of_reference is not None:
            ds.FrameOfReferenceUID = self.slide_uids.frame_of_reference
        ds.StudyInstanceUID = self.slide_uids.study_instance
        ds.SeriesInstanceUID = self.slide_uids.series_instance
        ds.SOPInstanceUID = uid_generator()
        ds.SOPClassUID = ANN_SOP_CLASS_UID

        meta_ds = FileMetaDataset()
        if little_endian and implicit_vr:
            transfer_syntax = ImplicitVRLittleEndian
        elif little_endian and not implicit_vr:
            transfer_syntax = ExplicitVRLittleEndian
        elif not little_endian and not implicit_vr:
            transfer_syntax = ExplicitVRBigEndian
        else:
            raise NotImplementedError("Unsupported transfer syntax")

        meta_ds.TransferSyntaxUID = transfer_syntax
        meta_ds.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        meta_ds.MediaStorageSOPClassUID = ANN_SOP_CLASS_UID
        meta_ds.FileMetaInformationGroupLength = 0  # Updated on write
        validate_file_meta(meta_ds)
        file_ds = FileDataset(
            preamble=b"\x00" * 128,
            filename_or_obj=path,
            file_meta=meta_ds,
            dataset=ds,
            is_implicit_VR=implicit_vr,
            is_little_endian=little_endian,
        )
        dcmwrite(path, file_ds)

    @classmethod
    def open(
        cls, files: Iterable[Union[str, Path, BinaryIO]]
    ) -> Iterable["AnnotationInstance"]:
        """Read annotations from DICOM files according to sup 222.

        Parameters
        ----------
        files: Sequence[Union[str, Path, BinaryIO]]
            Files with DICOM annotations to read.
        """
        return (cls.open_dataset(dcmread(file)) for file in files)

    @classmethod
    def open_dataset(cls, dataset: Dataset) -> "AnnotationInstance":
        """Read annotations from DICOM dataset according to sup 222.

        Parameters
        ----------
        dataset: Dataset
            DICOM annotation dataset to read.

        Returns
        ----------
        List[AnnotationGroup]
            Annotation groups read from dataset.
        """
        groups: List[AnnotationGroup] = []
        slide_uids: Optional[SlideUids] = None

        frame_of_reference_uid = getattr(dataset, "FrameOfReferenceUID", None)
        if dataset.AnnotationCoordinateType == "2D":
            coordinate_type = "image"
        elif dataset.AnnotationCoordinateType == "3D":
            coordinate_type = "volume"
        else:
            raise ValueError("Unknown coordinate type")
        if coordinate_type == "volume" and frame_of_reference_uid is None:
            raise ValueError(
                "volume annotation corrindate type requires frame of reference"
            )

        instance = dataset.SOPInstanceUID
        if slide_uids is None:
            slide_uids = SlideUids(
                dataset.StudyInstanceUID,
                dataset.SeriesInstanceUID,
                frame_of_reference_uid,
            )
        else:
            if slide_uids != SlideUids(
                dataset.StudyInstanceUID,
                dataset.SeriesInstanceUID,
                frame_of_reference_uid,
            ):
                raise ValueError("Base uids should match")
        for annotation_ds in dataset.AnnotationGroupSequence:
            annotation_type = annotation_ds.GraphicType
            if annotation_type == "POINT":
                annotation_class = PointAnnotationGroup
            elif annotation_type == "POLYLINE":
                annotation_class = PolylineAnnotationGroup
            elif annotation_type == "POLYGON":
                annotation_class = PolygonAnnotationGroup
            else:
                raise NotImplementedError("Unsupported Graphic type")
            annotation = annotation_class.from_ds(annotation_ds, instance)
            groups.append(annotation)
        return cls(groups, coordinate_type, slide_uids)

    def __getitem__(self, index: int) -> AnnotationGroup:
        """Return annotation group by index (group number).

        Parameters
        ----------
        index: int
            Group number of group to return

        Returns
        ----------
        Annotation
            Annotation group with group number
        """
        return self.groups[index]

    def __len__(self) -> int:
        return len(self.groups)
