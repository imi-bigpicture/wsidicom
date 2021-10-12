from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (Any, Callable, DefaultDict, Dict, Generator, List,
                    Optional, Set, Tuple, Union)

import numpy as np
import pydicom
from pydicom import config
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.sr.codedict import Code, codes

from .uid import ANN_SOP_CLASS_UID, BaseUids, Uid

config.enforce_valid_values = True
config.future_behavior()


@dataclass
class LabColor:
    l: int
    a: int
    b: int


def dcm_to_list(item: bytes, type: str) -> List:
    """Convert item to list of values using specified type format.

    Parameters
    ----------
    item: bytes
        Bytes to convert.
    type: str
        Struct format character.

    Returns
    ----------
    List
        List of values
    """
    converted = pydicom.values.convert_numbers(
        item,
        is_little_endian=True,
        struct_format=type
    )
    if not isinstance(converted, list):
        return [converted]
    return converted


class ConceptCode(Code):
    """Help functions for handling SR codes.
    Provides functions for converting between Code and dicom dataset.
    For CIDs that are not-yet standardized, functions for creating Code from
    code meaning is provided using the CID definitions in the sup 222 draft.
    For standardized CIDs one can use pydicom.sr.codedict to create codes."""
    code_dictionaries = {
        'measurement': {'Area': Code('42798000', 'SCT', 'Area')},  # noqa
        'typecode': {
            'Nucleus': Code('84640000', 'SCT', 'Nucleus'),  # noqa
            'Entire cell': Code('362837007', 'SCT', 'Entire cell')  # noqa
        }
    }
    measurement_dictionary = {'Area': Code('42798000', 'SCT', 'Area')}  # noqa

    typecode_dictionary = {
        'Nucleus': Code('84640000', 'SCT', 'Nucleus'),  # noqa
        'Entire cell': Code('362837007', 'SCT', 'Entire cell')  # noqa
    }

    def __hash__(self):
        return hash((
            self.value,
            self.scheme_designator,
            self.meaning,
            self.scheme_version
        ))

    @property
    def sequence(self) -> DicomSequence:
        """Return code as DICOM sequence.

        Returns
        ----------
        DicomSequence
            Dicom sequence of dataset containing code.

        """
        ds = Dataset()
        ds.CodeValue = self.value
        ds.CodingSchemeDesignator = self.scheme_designator
        ds.CodeMeaning = self.meaning
        if self.scheme_version is not None:
            ds.CodeSchemeVersion = self.scheme_version
        sequence = DicomSequence([ds])
        return sequence

    @classmethod
    def measurement(cls, value: Union[str, Dataset]) -> 'ConceptCode':
        """Return measurement code for value. Value can be a code meaning (str)
        or a DICOM dataset containing the code.

        Parameters
        ----------
        value: Union[str, Dataset]
            The value for creating the code.

        Returns
        ----------
        ConceptCode
            Measurement code created from value.

        """
        if isinstance(value, str):
            return cls._from_dict('measurement', value)
        elif isinstance(value, Dataset):
            return cls._from_ds(value, 'ConceptNameCodeSequence')
        raise NotImplementedError(value)

    @classmethod
    def type(cls, value: Union[str, Dataset]) -> 'ConceptCode':
        """Return typecode code for value. Value can be a code meaning
        (str) or a DICOM dataset containing the code.

        Parameters
        ----------
        value: Union[str, Dataset]
            The value for creating the code.

        Returns
        ----------
        ConceptCode
            typecode code created from value.

        """
        if isinstance(value, str):
            return cls._from_dict('typecode', value)
        elif isinstance(value, Dataset):
            return cls._from_ds(value, 'AnnotationPropertyTypeCodeSequence')
        raise NotImplementedError(value)

    @classmethod
    def category(cls, value: Union[str, Dataset]) -> 'ConceptCode':
        """Return categorycode code for value. Value can be a code meaning
        (str) or a DICOM dataset containing the code.

        Parameters
        ----------
        value: Union[str, Dataset]
            The value for creating the code.

        Returns
        ----------
        ConceptCode
            categorycode code created from value.

        """
        if isinstance(value, str):
            return cls._from_cid('cid7150', value)
        elif isinstance(value, Dataset):
            return cls._from_ds(
                value,
                'AnnotationPropertyCategoryCodeSequence'
            )
        raise NotImplementedError(value)

    @classmethod
    def unit(cls, value: Union[str, Dataset]) -> 'ConceptCode':
        """Return unit code for value. Value can be a code meaning
        (str) or a DICOM dataset containing the code.

        Parameters
        ----------
        value: Union[str, Dataset]
            The value for creating the code.

        Returns
        ----------
        ConceptCode
            Unit code created from value.

        """
        if isinstance(value, str):
            return cls._from_ucum(value)
        elif isinstance(value, Dataset):
            return cls._from_ds(value, 'MeasurementUnitsCodeSequence')
        raise NotImplementedError(value)

    @classmethod
    def _from_ds(cls, ds: Dataset, sequence_name: str) -> 'ConceptCode':
        """Return ConceptCode from sequence in dataset.

        Parameters
        ----------
        ds: Dataset
            Datasete containing code sequence.
        sequence_name: str
            Name of the sequence containing code.

        Returns
        ----------
        ConceptCode
            Code created from sequence in dataset.

        """
        code_ds = getattr(ds, sequence_name)[0]
        value = code_ds.CodeValue
        scheme = code_ds.CodingSchemeDesignator
        meaning = code_ds.CodeMeaning
        version = getattr(code_ds, 'CodeSchemeVersion', None)
        return cls(
            value=value,
            scheme_designator=scheme,
            meaning=meaning,
            scheme_version=version
        )

    @classmethod
    def _from_dict(cls, dict_name: str, meaning: str) -> 'ConceptCode':
        """Return ConceptCode from dictionary.

        Parameters
        ----------
        dict_name: str
            Dictionary name to get code from.
        meaning: str
            Code meaning of  code to get.

        Returns
        ----------
        ConceptCode
            Code from dictionary.

        """
        try:
            code_dict = cls.code_dictionaries[dict_name]
        except KeyError:
            raise NotImplementedError("Unkown dictionary")
        try:
            code = code_dict[meaning]
        except KeyError:
            raise NotImplementedError("Unsupported code")
        return cls(*code)

    @classmethod
    def _from_ucum(cls, unit: str) -> 'ConceptCode':
        """Return UCUM scheme ConceptCode.

        Parameters
        ----------
        meaning: str
            Code meaning.

        Returns
        ----------
        ConceptCode
            Code created from meaning.

        """
        return cls(
            value=unit,
            scheme_designator='UCUM',
            meaning=unit
        )

    @classmethod
    def _from_cid(cls, cid: str, meaning: str) -> 'ConceptCode':
        """Return ConceptCode from CID and meaning. For a list of CIDs, see
        http://dicom.nema.org/medical/dicom/current/output/chtml/part16/chapter_B.html # noqa

        Parameters
        ----------
        cid: str
            CID to use.
        meaning: str
            Code meaning to get.

        Returns
        ----------
        ConceptCode
            Code created from CID and meaning.

        """
        try:
            cid = getattr(codes, cid)
        except AttributeError:
            raise NotImplementedError("Unsupported cid")
        try:
            code = getattr(cid, meaning)
        except AttributeError:
            raise NotImplementedError("Unsupported code")
        return cls(*code)


@dataclass
class Measurement:
    def __init__(self, type: ConceptCode, value: float, unit: ConceptCode):
        """Represents a measurement.

        Parameters
        ----------
        type: str
            Type of measurement.
        value: float
            Value of measurement.
        unit: str
            Unit of measruement.
        """
        self.type = type
        self.value = value
        self.unit = unit

    def __repr__(self) -> str:
        return f"Measurement({self.type}, {self.value},  {self.unit})"

    def same_type(self, other_type: ConceptCode, other_unit: ConceptCode):
        """Return true if measurement base is same.

        Parameters
        ----------
        other_base: MeasurementType
            Other base to compare to.

        Returns
        ----------
        bool
            True if bases are same.
        """
        return (
            self.type == other_type and
            self.unit == other_unit
        )

    @staticmethod
    def _get_measurement_type_from_ds(
        ds: Dataset
    ) -> Tuple[ConceptCode, ConceptCode]:
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
        type = ConceptCode.measurement(ds)
        unit = ConceptCode.unit(ds)
        return (type, unit)

    @staticmethod
    def _get_measurement_values_from_ds(
        ds: Dataset
    ) -> List[float]:
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
        return dcm_to_list(
            ds.MeasurementValuesSequence[0].FloatingPointValues,
            'f'
        )

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
        if 'AnnotationIndexList' in measurement_ds:
            indices = dcm_to_list(measurement_ds.AnnotationIndexList, 'l')
            return [index-1 for index in indices]
        return list(range(annotation_count))

    @classmethod
    def _iterate_measurement_sequence(
        cls,
        sequence: DicomSequence,
        annotation_count: int
    ) -> Generator[Tuple[int, 'Measurement'], None, None]:
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
        Generator[Tuple[int, 'Measurement'], None, None]:
            Generator for annotation index and measurement.
        """
        for group_ds in sequence:
            type, unit = cls._get_measurement_type_from_ds(group_ds)
            values = cls._get_measurement_values_from_ds(group_ds)
            indices = cls._get_measurement_indices_from_ds(
                group_ds,
                annotation_count
            )
            if len(values) != len(indices):
                raise ValueError(
                    "number of indices needs to be same as measurements"
                )
            for index, annotation_index in enumerate(indices):
                yield (
                    annotation_index,
                    Measurement(type, values[index], unit)
                )

    @classmethod
    def get_measurements_from_ds(
        cls,
        sequence: DicomSequence,
        annotation_count: int
    ) -> Dict[int, List['Measurement']]:
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
            sequence,
            annotation_count
        ):
            measurements[annotation_index].append(measurement)
        return dict(measurements)


class Geometry(metaclass=ABCMeta):
    @property
    @abstractmethod
    def data(self) -> List[float]:
        """Return geometry content as a list of floats"""
        raise NotImplementedError

    @abstractmethod
    def to_coords(self) -> List[Tuple[float, float]]:
        """Return geometry content as a list of tuple of floats"""
        raise NotImplementedError

    @abstractmethod
    def to_list_coords(self) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_coords(
        cls,
        coords: Union[Tuple[float, float], List[Tuple[float, float]]]
    ):
        """Return geometry object created from list of coordinates"""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_list(cls, list: List[float]):
        """Return geometry object created from list of floats"""
        raise NotImplementedError

    @classmethod
    def list_to_coords(self, data: List[float]) -> List[Tuple[float, float]]:
        """Return cordinates in list of floats as list of tuple of floats

        Parameters
        ----------
        data: List[float]
            List of float to convert.

        Returns
        ----------
        List[Tuple[float, float]]
            List of coordinates (Tuple of floats).
        """
        x_indices = range(0, len(data), 2)
        y_indices = range(1, len(data), 2)
        coords: List[Tuple[float, float]] = [
            (float(data[x]), float(data[y]))
            for x, y in zip(x_indices, y_indices)
        ]
        return coords

    @staticmethod
    def _coordinates_from_dict(
        dictionary: Union[
            Dict[str, Union[str, float]],
            List[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str
    ) -> List[Tuple[float, float]]:
        """Return coordinates in dictionary as list of tuple of floats

        Parameters
        ----------
        dictionary: Union[
            Dict[str, Union[str, float]],
            List[Dict[str, Union[str, float]]]
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
        if not isinstance(dictionary, list):
            coords = [dictionary]
        else:
            coords = dictionary
        return [
            (float(coordinate[x]), float(coordinate[y]))
            for coordinate in coords
        ]

    @classmethod
    @abstractmethod
    def from_dict(
        cls,
        dictionary: Union[
            Dict[str, Union[str, float]],
            List[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str,
    ):
        raise NotImplementedError

    @classmethod
    def from_shapely_like(cls, object) -> 'Geometry':
        """Return Geometry from shapely-like object. Object needs to have
        shapely-like attributes.

        Parameters
        ----------
        ojbect
            Object with shapely-like attributes.

        Returns
        ----------
        Geometry
            Geometry created from object.
        """
        geometry_type = object.geom_type
        try:
            if geometry_type == "Point":
                return Point(*object.coords[0])
            elif geometry_type == "Polygon":
                if list(object.interiors) == []:
                    return Polygon(list(object.exterior.coords))
            elif geometry_type == "LineString":
                return Polyline(object.coords)
            raise NotImplementedError("Not a supported shapely like object")
        except AttributeError:
            raise ValueError("Not a shapely like object")

    @classmethod
    def from_geojson(
        cls,
        dictionary: Dict[str, Any]
    ) -> List['Geometry']:
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
            coordinates: List[Any] = dictionary["coordinates"]
            annotation_type: str = dictionary["type"]
        except KeyError:
            raise ValueError("Not a geojson object")
        if(annotation_type == 'Point'):
            points: List[float] = coordinates
            return [Point.from_list(points)]
        elif(annotation_type == 'MultiPoint'):
            points: List[List[float]] = coordinates
            return [Point.from_list(point) for point in points]
        elif(annotation_type == 'Polygon'):
            polylines: List[List[List[float]]] = coordinates
            return [Polygon.from_coords(polylines[0])]
        elif(annotation_type == 'LineString'):
            polyline: List[List[float]] = coordinates
            return [Polyline.from_coords(polyline)]
        raise NotImplementedError("Not supported geojson geometry")


@dataclass
class Point(Geometry):
    """Geometry consisting of a single point"""
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __eq__(self, point) -> bool:
        if isinstance(point, Point):
            return (
                self.x == point.x and self.y == point.y
            )
        else:
            raise NotImplementedError("Comparing Point to non-Point")

    @property
    def data(self) -> List[float]:
        return [self.x, self.y]

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
    def from_coords(
        cls,
        coords: Union[Tuple[float, float], List[Tuple[float, float]]]
    ) -> 'Point':
        if not isinstance(coords, tuple):
            if len(coords) != 1:
                raise ValueError("Input has more than two points")
            coords = coords[0]
        return cls(*coords)

    @classmethod
    def from_list(cls, list: List[float]) -> 'Point':
        coordinates = cls.list_to_coords(list)
        return cls.from_coords(coordinates)

    @classmethod
    def multiple_from_dict(
        cls,
        dictionary: Union[
            Dict[str, Union[str, float]],
            List[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str,
    ) -> List['Point']:
        coords = cls._coordinates_from_dict(dictionary, x, y)
        return [cls(*point) for point in coords]

    @classmethod
    def from_dict(
        cls,
        dictionary: Union[
            Dict[str, Union[str, float]],
            List[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str,
    ) -> 'Point':
        coords = cls._coordinates_from_dict(dictionary, x, y)
        return cls.from_coords(coords)


@dataclass
class Polyline(Geometry):
    """Geometry consisting of connected lines."""
    def __init__(self, points: List[Tuple[float, float]]):
        self.points: List[Point] = [
            Point(point[0], point[1]) for point in points
        ]

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

    @classmethod
    def from_coords(
        cls,
        coords: Union[Tuple[float, float], List[Tuple[float, float]]]
    ) -> 'Polyline':
        if isinstance(coords, tuple):
            coords = [coords]
        return cls(coords)

    @classmethod
    def from_dict(
        cls,
        dictionary: Union[
            Dict[str, Union[str, float]],
            List[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str,
    ) -> 'Polyline':
        coords = cls._coordinates_from_dict(dictionary, x, y)
        return cls.from_coords(coords)

    @classmethod
    def from_list(cls, list: List[float]) -> 'Polyline':
        coordinates = cls.list_to_coords(list)
        return cls.from_coords(coordinates)


@dataclass
class Polygon(Polyline):
    """Geometry consisting of connected lines implicity closed."""
    def __init__(self, points: List[Tuple[float, float]]):
        super().__init__(points)

    @classmethod
    def from_coords(
        cls,
        coords: Union[Tuple[float, float], List[Tuple[float, float]]]
    ) -> 'Polygon':
        if isinstance(coords, tuple):
            coords = [coords]
        return cls(coords)

    @classmethod
    def from_list(cls, list: List[float]) -> 'Polygon':
        coordinates = cls.list_to_coords(list)
        return cls.from_coords(coordinates)

    @classmethod
    def from_dict(
        cls,
        dictionary: Union[
            Dict[str, Union[str, float]],
            List[Dict[str, Union[str, float]]]
        ],
        x: str,
        y: str,
    ) -> 'Polygon':
        coords = cls._coordinates_from_dict(dictionary, x, y)
        return cls.from_coords(coords)

    def __repr__(self) -> str:
        return f"Polygon({self.to_coords()})"


class Annotation:
    def __init__(
        self,
        geometry: Geometry,
        measurements: List[Measurement] = None
    ):
        """Represents an annotation, with geometry and an optional list of
        measurements.

        Parameters
        ----------
        geometry: Geometry
            Geometry of the annotation.
        measurements: List[Measurement]
            Optional measurements of the annotation.
        """

        self._geometry = geometry
        self._measurements: List[Measurement] = []
        if measurements is not None:
            self._measurements = measurements

    def __repr__(self) -> str:
        return f"Annotation({self.geometry}, {self.measurements})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Annotation):
            raise NotImplemented(other)
        return (
            self.geometry == other.geometry and
            self.measurements == other.measurements
        )

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def measurements(self) -> List[Measurement]:
        return self._measurements

    def get_measurements(
        self,
        type: ConceptCode,
        unit: ConceptCode
    ) -> List[Measurement]:
        """Return measurements of specified type.

        Parameters
        ----------
        measurement_type: MeasurementType
            Type of measurements to get.

        Returns
        ----------
        List[Measurement]
            List of measurements of specified type.
        """
        return [
            measurement
            for measurement in self.measurements
            if measurement.same_type(type, unit)
        ]

    def get_measurement_values(
        self,
        type: ConceptCode,
        unit: ConceptCode
    ) -> List[float]:
        """Return values for measurements of specified type.

        Parameters
        ----------
        measurement_type: MeasurementType
            Type of measurements to get.

        Returns
        ----------
        List[Measurement]
            List of values for measurements of specified type.
        """
        return [
            measurement.value for measurement in self.measurements
            if measurement.same_type(type, unit)
        ]


class AnnotationGroup:
    _geometry_type: type

    def __init__(
        self,
        annotations: List[Annotation],
        label: str,
        categorycode: ConceptCode,
        typecode: ConceptCode,
        description: str = None,
        color: LabColor = None,
        is_double: bool = True,
        instance: Uid = None
    ):
        """Represents a group of annotations of the same type.

        Parameters
        ----------
        annotations: List[Annotation]
            Annotations in the group.
        label: str
            Group label
        categorycode: ConceptCode
            Group categorycode.
        typecode: ConceptCode
            Group typecode.
        instance: Uid
            Uid this group was created from.
        description: str
            Group description.
        color: LabColor
            Recommended CIELAB color.
        is_double: bool
            If group is stored with double float
        """
        self.validate_type(annotations, self._geometry_type)
        self._z_planes: List[float] = []
        self._optical_paths: List[str] = []
        self._uid = pydicom.uid.generate_uid()
        self._categorycode = categorycode
        self._typecode = typecode
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
            f"{self.categorycode}, {self.typecode}, "
            f"{self.description}, {self._is_double})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, AnnotationGroup):
            raise NotImplemented(other)
        return (
            self.annotations == other.annotations and
            self.label == other.label and
            self.typecode == other.typecode and
            self.categorycode == other.categorycode and
            self.description == other.description and
            self.color == other.color and
            self._is_double == other._is_double
        )

    @property
    def categorycode(self) -> ConceptCode:
        return self._categorycode

    @property
    def typecode(self) -> ConceptCode:
        return self._typecode

    @property
    def annotations(self) -> List[Annotation]:
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
            Coordiantes in annotation group.
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
    @abstractmethod
    def annotation_type(self) -> str:
        raise NotImplementedError

    def __getitem__(self, index: Union[int, List[int]]):
        if isinstance(index, list):
            return [self[i] for i in index]
        return self.annotations[index]

    @classmethod
    def from_ds(
        cls,
        ds: Dataset,
        instance: Uid
    ) -> 'AnnotationGroup':
        """Return annotation group from Annotation Group Sequence dataset.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group.
        instance: Uid
            Uid this group was created from.

        Returns
        ----------
        AnnotationGroup
            Annotation group from dataset.
        """
        is_double = cls._is_ds_double(ds)
        annotations = cls._get_annotations_from_ds(ds)
        label = cls._get_label_from_ds(ds)
        description = getattr(ds, 'AnnotationGroupDescription', None)
        typecode = ConceptCode.type(ds)
        categorycode = ConceptCode.category(ds)
        color = getattr(ds, 'RecommendedDisplayCIELabValue', None)
        return cls(
            annotations,
            label,
            categorycode,
            typecode,
            description,
            color,
            is_double,
            instance
        )

    @staticmethod
    def _get_label_from_ds(
        ds: Dataset
    ) -> str:
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
    def _get_focal_planes_from_ds(
        ds: Dataset
    ) -> List[float]:
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
        if ds.AnnotationAppliesToAllZPlanes == 'YES':
            return []
        z_planes: List[float] = ds.CommonZCoordinateValue
        return z_planes

    @staticmethod
    def _get_optical_paths_from_ds(
        ds: Dataset
    ) -> List[str]:
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
        if ds.AnnotationAppliesToAllOpticalPaths == 'YES':
            return []
        optical_paths: List[str] = ds.ReferencedOpticalPathIdentifier
        return optical_paths

    @staticmethod
    def _get_count_from_ds(
        ds: Dataset
    ) -> int:
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
        if 'MeasurementsSequence' not in ds:
            return {}
        annotation_count = cls._get_count_from_ds(ds)
        return Measurement.get_measurements_from_ds(
            ds.MeasurementsSequence,
            annotation_count
        )

    @staticmethod
    def _is_ds_double(
        ds: Dataset
    ) -> bool:
        """Return true if group in dataset stores coordiantes as double float.

        Parameters
        ----------
        ds: Dataset
            Dataset containing annotation group.

        Returns
        ----------
        bool
            True if group uses double float.
        """
        return 'DoublePointCoordinatesData' in ds

    @classmethod
    def _get_coordinates_from_ds(
        cls,
        ds: Dataset
    ) -> List[Tuple[float, float]]:
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
            data = dcm_to_list(ds.DoublePointCoordinatesData, 'd')
        else:
            data = dcm_to_list(ds.PointCoordinatesData, 'f')
        return Geometry.list_to_coords(data)

    @classmethod
    @abstractmethod
    def _get_geometries_from_ds(
        cls,
        ds: Dataset
    ):
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
        raise NotImplementedError

    @classmethod
    def _get_annotations_from_ds(
        cls,
        ds: Dataset
    ) -> List[Annotation]:
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
    def measurement_types(self) -> List[Tuple[ConceptCode, ConceptCode]]:
        """Return measurement types in annotation group.

        Returns
        ----------
        List[Tuple[ConceptCode, ConceptCode]]
            Measurement type-unit-pairs in annotation group.
        """
        pairs: Set[Tuple[ConceptCode, ConceptCode]] = set()
        for annotation in self.annotations:
            for measurement in annotation.measurements:
                pair = (measurement.type, measurement.unit)
                pairs.add(pair)
        return list(pairs)

    def get_measurements(
        self,
        type: ConceptCode,
        unit: ConceptCode
    ) -> List[Measurement]:
        """Return measurements of specified type-unit-pair.

        Parameters
        ----------
        type: ConceptCode
            Type of measurement to get
        unit: ConceptCode
            Unit of measurement to get

        Returns
        ----------
        List[Measurement]
            Measurements of specified type.
        """
        measurements: List[Measurement] = []
        for annotation in self.annotations:
            measurements += annotation.get_measurements(type, unit)
        return measurements

    def create_measurement_indices(
        self,
        type: ConceptCode,
        unit: ConceptCode
    ) -> np.ndarray:
        """Return measurement indicies for all measurements of specified type.
        Indices are stored starting at index 1.

        Parameters
        ----------
        type: ConceptCode
            Type of measurement to get
        unit: ConceptCode
            Unit of measurement to get

        Returns
        ----------
        np.ndarray
            Measurement indices in annotation group of specified type.
        """
        indices: List[int] = []
        for i, annotation in enumerate(self.annotations):
            indices += [i + 1] * len(annotation.get_measurements(type, unit))
        return np.array(indices, dtype=int)

    def _create_measurement_values(
        self,
        type: ConceptCode,
        unit: ConceptCode
    ) -> np.ndarray:
        """Return measurement values for all measurements of specified type.

        Parameters
        ----------
        type: ConceptCode
            Type of measurement to get
        unit: ConceptCode
            Unit of measurement to get

        Returns
        ----------
        np.ndarray
            Measurement values in annotation group of specified type.
        """
        values: List[float] = []
        for annotation in self.annotations:
            values += annotation.get_measurement_values(type, unit)
        return np.array(values, dtype=np.float32)

    def _measurements_is_one_to_one(
        self,
        type: ConceptCode,
        unit: ConceptCode
    ) -> bool:
        """Check if all annotations in group have strictly one measurement
        of specified type-unit-pair.

        Parameters
        ----------
        type: ConceptCode
            Type of measurement to check
        unit: ConceptCode
            Unit of measurement to check

        Returns
        ----------
        bool
            True if this measurement type-unit-pair maps one-to-one to
            annotations.
        """
        for annotation in self.annotations:
            if not len(annotation.get_measurements(type, unit)) == 1:
                return False
        return True

    def _create_measurement_value_sequence(
        self,
        type: ConceptCode,
        unit: ConceptCode
    ) -> DicomSequence:
        """Return Measurement Value Sequence of measurements of specified
        type and unit.

        Parameters
        ----------
        type: ConceptCode
            Type of measurement create sequence for.
        unit: ConceptCode
            Unit of measurement create sequence for.

        Returns
        ----------
        DicomSequence
            A sequence of the measurement values of specified type and unit.
        """
        measurements_ds = Dataset()
        # According to the standard the indices is not needed if each
        # annotation has one and only one measurement.
        if not self._measurements_is_one_to_one(type, unit):
            indices = self.create_measurement_indices(type, unit)
            measurements_ds.AnnotationIndexList = indices
        values = self._create_measurement_values(type, unit)
        measurements_ds.FloatingPointValues = values
        measurement_value_sequence = DicomSequence([measurements_ds])
        return measurement_value_sequence

    def _create_measurement_sequence_item(
        self,
        type: ConceptCode,
        unit: ConceptCode
    ) -> Dataset:
        """Return Measurement Sequence item of measurements of specified type
        and unit.

        Parameters
        ----------
        type: ConceptCode
            Type of measurement create sequence for.
        unit: ConceptCode
            Unit of measurement create sequence for.

        Returns
        ----------
        Dataset
            A measurement sequence item for the specified measurement type and
            unit.
        """
        ds = Dataset()
        ds.MeasurementValuesSequence = self._create_measurement_value_sequence(
            type, unit
        )
        ds.ConceptNameCodeSequence = type.sequence
        ds.MeasurementUnitsCodeSequence = unit.sequence
        return ds

    def _set_planes_in_ds(
        self,
        ds: Dataset
    ) -> Dataset:
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
        if self._z_planes is []:
            ds.AnnotationAppliesToAllZPlanes = 'YES'
        else:
            ds.AnnotationAppliesToAllZPlanes = 'NO'
            ds.CommonZCoordinateValue = self._z_planes
        return ds

    def _set_optical_paths_in_ds(
        self,
        ds: Dataset
    ) -> Dataset:
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
        if self._optical_paths is []:
            ds.AnnotationAppliesToAllOpticalPaths = 'YES'
        else:
            ds.AnnotationAppliesToAllOpticalPaths = 'NO'
            ds.ReferencedOpticalPathIdentifier = self._optical_paths
        return ds

    def _set_coordiantes_data_in_ds(
        self,
        ds: Dataset
    ) -> Dataset:
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

    def _set_measurement_sequence_in_ds(
        self,
        ds: Dataset
    ) -> Dataset:
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
                self._create_measurement_sequence_item(
                    *measurement_type
                )
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
        ds.AnnotationGroupNumber = group_number + 1
        ds.AnnotationGroupUID = self._uid
        ds.NumberOfAnnotations = len(self.annotations)
        ds.GraphicType = self.annotation_type
        ds.AnnotationGroupLabel = self.label
        if self.description is not None:
            ds.AnnotationGroupDescription = self.description
        ds.AnnotationPropertyCategoryCodeSequence = self.categorycode.sequence
        ds.AnnotationPropertyTypeCodeSequence = self.typecode.sequence
        ds = self._set_coordiantes_data_in_ds(ds)
        ds = self._set_planes_in_ds(ds)
        ds = self._set_optical_paths_in_ds(ds)
        ds = self._set_measurement_sequence_in_ds(ds)
        if self.color is not None:
            ds.RecommendedDisplayCIELabValue = self.color
        # AUTOMATIC and SEMIAUTOMATIC requires a
        # Annotation Algorithm Identification Sequence
        ds.AnnotationGroupGenerationType = 'MANUAL'
        return ds

    @staticmethod
    def validate_type(
        annotations: List[Annotation],
        geometry_type: type
    ):
        """Check that list of annotations are of the requested type.

        Parameters
        ----------
        annotations: List[Annotation]
            List of annotations to check
        geometry_type: type
            Requested type
        """
        for annotation in annotations:
            if not isinstance(annotation.geometry, geometry_type):
                raise TypeError(
                    f'annotation type {type(annotation.geometry)}'
                    f' does not match Group typecode {geometry_type}'
                )

    @classmethod
    def _get_group_type_by_geometry(cls, geometry_type: type):
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
        geometries: List[Geometry],
        label: str,
        categorycode: ConceptCode,
        typecode: ConceptCode,
        is_double: bool = True
    ) -> 'AnnotationGroup':
        """Return AnnotationGroup created from list of geometries. The group
        type is determined by the first geometry, and all geometries needs to
        have the same type.

        Parameters
        ----------
        geometries: List[Geometries]
            Geometries in the group.
        label: str
            Group label
        categorycode: ConceptCode
            Group categorycode.
        typecode: ConceptCode
            Group typecode.

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
            categorycode=categorycode,
            typecode=typecode,
            is_double=is_double
        )
        return group


class PointAnnotationGroup(AnnotationGroup):
    """Point annotation group"""
    _geometry_type = Point

    @property
    def annotation_type(self) -> str:
        return "POINT"

    @classmethod
    def _get_geometries_from_ds(
        cls,
        ds: Dataset
    ) -> List[Point]:
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
        points = [
            Point.from_coords(coordiantes) for coordiantes in coordinate_list
        ]
        return points


class PolylineAnnotationGroupMeta(AnnotationGroup):
    """Meta class for line annotation goup"""
    _geometry_type: type

    @property
    def annotation_type(self) -> str:
        raise NotImplementedError

    @property
    def point_index_list(self) -> np.ndarray:
        """Return point index list for annotations in group. Indices are stored
        starting at index 1 and in relation to geometry data lenght.

        Returns
        ----------
        np.ndarray
            List of indices in annotation group
        """
        index = 1
        indices = []
        for annotation in self.annotations:
            indices.append(index)
            index += len(annotation.geometry.data)
        return np.array(indices, dtype=int)

    @staticmethod
    def _get_indices_from_ds(
        ds: Dataset
    ) -> List[int]:
        """Return line start indices from sup 222 dataset. Indices are stored
        starting at with value 1, and are in relation to non-pared coordinates.
        Returned list starts at 0 and is in relation to paired coordiantes.

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
            (value - 1)//2
            for value
            in dcm_to_list(ds.LongPrimitivePointIndexList, 'l')
        ]

    @classmethod
    def _get_geometries_from_ds(
        cls,
        ds: Dataset
    ) -> List[Geometry]:
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
        geometries: List[Geometry] = []
        for index in range(number_of_geometries):
            start = indices[index]
            end = indices[index+1]
            line_coordinates = coordinates[start:end]
            lines = cls._get_line_geometry_from_coords(line_coordinates)
            geometries.append(lines)
        return geometries

    @staticmethod
    @abstractmethod
    def _get_line_geometry_from_coords(coords: List[Tuple[float, float]]):
        raise NotImplementedError

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


class PolylineAnnotationGroup(PolylineAnnotationGroupMeta):
    _geometry_type = Polyline

    @property
    def annotation_type(self) -> str:
        return "POLYLINE"

    @staticmethod
    def _get_line_geometry_from_coords(coords: List[Tuple[float, float]]):
        return Polyline.from_coords(coords)


class PolygonAnnotationGroup(PolylineAnnotationGroupMeta):
    _geometry_type = Polygon

    @property
    def annotation_type(self) -> str:
        return "POLYGON"

    @staticmethod
    def _get_line_geometry_from_coords(coords: List[Tuple[float, float]]):
        return Polygon.from_coords(coords)


class AnnotationInstance:
    def __init__(
        self,
        groups: List[AnnotationGroup],
        base_uids: BaseUids
    ):
        """Reoresents a collection of annotation groups.

        Parameters
        ----------
        annotations: List[AnnotationGroup]
            List of annotations group
        frame_of_referenc: Uid
            Frame of reference uid of image that the annotations belong to
        """
        self.groups = groups
        self.coordinate_type = '3D'
        self.base_uids = base_uids
        self.datetime = datetime.now()
        self.modality = 'ANN'
        self.series_number: int

    def __repr__(self) -> str:
        return (
            f"AnnotationInstance({self.groups}, "
            f"{self.base_uids})"
        )

    def save(
        self,
        path: str,
        little_endian: bool = True,
        implicit_vr: bool = False,
        uid_generator: Callable[..., Uid] = pydicom.uid.generate_uid
    ):
        """Write annotations to DICOM file according to sup 222.
        Note that the file will miss important DICOM attributes that has not
        yet been implemented.

        Parameters
        ----------
        path: Path
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
            if(isinstance(annotation_group, AnnotationGroup)):
                bulk_sequence.append(annotation_group.to_ds(index+1))
            else:
                raise NotImplementedError(
                    f"Group type: {type(annotation_group)} not supported"
                )
        ds.AnnotationGroupSequence = bulk_sequence
        ds.AnnotationCoordinateType = self.coordinate_type
        ds.FrameOfReferenceUID = self.base_uids.frame_of_reference
        ds.StudyInstanceUID = self.base_uids.study_instance
        ds.SeriesInstanceUID = self.base_uids.series_instance
        ds.SOPInstanceUID = uid_generator()
        ds.SOPClassUID = ANN_SOP_CLASS_UID

        meta_ds = pydicom.dataset.FileMetaDataset()
        if little_endian and implicit_vr:
            transfer_syntax = pydicom.uid.ImplicitVRLittleEndian
        elif little_endian and not implicit_vr:
            transfer_syntax = pydicom.uid.ExplicitVRLittleEndian
        elif not little_endian and not implicit_vr:
            transfer_syntax = pydicom.uid.ExplicitVRBigEndian
        else:
            raise NotImplementedError("Unsupported transfer syntax")

        meta_ds.TransferSyntaxUID = transfer_syntax
        meta_ds.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        meta_ds.MediaStorageSOPClassUID = ANN_SOP_CLASS_UID
        meta_ds.FileMetaInformationGroupLength = 0  # Updated on write
        pydicom.dataset.validate_file_meta(meta_ds)
        file_ds = pydicom.dataset.FileDataset(
            preamble=b'\x00' * 128,
            filename_or_obj=path,
            file_meta=meta_ds,
            dataset=ds,
            is_implicit_VR=implicit_vr,
            is_little_endian=little_endian
        )
        pydicom.filewriter.dcmwrite(path, file_ds)

    @classmethod
    def open(cls, paths: List[Path]) -> 'AnnotationInstance':
        """Read annotations from DICOM file according to sup 222.

        Parameters
        ----------
        paths: List[Path]
            Paths to DICOM annotation files to read.
        """
        groups: List[AnnotationGroup] = []
        base_uids: BaseUids = None
        for path in paths:
            ds = pydicom.filereader.dcmread(path)
            if ds.file_meta.MediaStorageSOPClassUID != ANN_SOP_CLASS_UID:
                raise ValueError("SOP Class UID of file is wrong")

            is_3D = (ds.AnnotationCoordinateType == '3D')
            if not is_3D:
                raise NotImplementedError(
                    "Only support annotations of '3D' type"
                )
            base_uids = BaseUids(
                ds.StudyInstanceUID,
                ds.SeriesInstanceUID,
                ds.FrameOfReferenceUID
            )
            instance = ds.SOPInstanceUID
            if base_uids is None:
                base_uids = BaseUids(
                    ds.StudyInstanceUID,
                    ds.SeriesInstanceUID,
                    ds.FrameOfReferenceUID
                )
            else:
                if base_uids != BaseUids(
                    ds.StudyInstanceUID,
                    ds.SeriesInstanceUID,
                    ds.FrameOfReferenceUID
                ):
                    raise ValueError("Base uids should match")
            for annotation_ds in ds.AnnotationGroupSequence:
                annotation_type = annotation_ds.GraphicType
                if(annotation_type == 'POINT'):
                    annotation_class = PointAnnotationGroup
                elif(annotation_type == 'POLYLINE'):
                    annotation_class = PolylineAnnotationGroup
                elif(annotation_type == 'POLYGON'):
                    annotation_class = PolygonAnnotationGroup
                else:
                    raise NotImplementedError("Unsupported Graphic type")
                annotation = annotation_class.from_ds(annotation_ds, instance)
                groups.append(annotation)
        return cls(groups, base_uids)

    def __getitem__(
        self,
        index: int
    ) -> AnnotationGroup:
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
