#    Copyright 2021, 2023 SECTRA AB
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

import json
import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pytest
import xmltodict
from pydicom.uid import generate_uid
from shapely import wkt
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from wsidicom import WsiDicom
from wsidicom.conceptcode import (
    AnnotationCategoryCode,
    AnnotationTypeCode,
    MeasurementCode,
    UnitCode,
)
from wsidicom.graphical_annotations import (
    Annotation,
    AnnotationGroup,
    AnnotationInstance,
    Geometry,
    LabColor,
    Measurement,
    Point,
    PointAnnotationGroup,
    Polygon,
    PolygonAnnotationGroup,
    Polyline,
    PolylineAnnotationGroup,
)
from wsidicom.uid import SlideUids

ANNOTATION_FOLDER = Path("tests/testdata/annotation")

type_code = AnnotationTypeCode("Nucleus")
category_code = AnnotationCategoryCode("Tissue")
slide_uids = SlideUids(
    study_instance=generate_uid(),
    series_instance=generate_uid(),
    frame_of_reference=generate_uid(),
)


test_files = {
    os.path.basename(folder): list(folder.iterdir())
    for folder in ANNOTATION_FOLDER.iterdir()
}

area = MeasurementCode("Area")
pixels = UnitCode("Pixels")
measurement0 = Measurement(area, 5, pixels)
measurement1 = Measurement(area, 10, pixels)
measurement2 = Measurement(area, 15, pixels)
point = Point(1, 1)


@pytest.mark.unittest
class TestWsiDicomAnnotation:
    @pytest.mark.parametrize("file_path", test_files["qupath_geojson"])
    def test_qupath_geojson(self, file_path: Path):
        with open(file_path) as f:
            # Arrange
            input_dict: Dict[str, Any] = json.load(f)
            group = AnnotationGroup.from_geometries(
                Geometry.from_geojson(input_dict["geometry"]),
                label=input_dict["properties"]["name"],
                category_code=category_code,
                type_code=type_code,
            )

            # Act
            dicom = AnnotationInstance([group], "volume", slide_uids)
            dicom = self.dicom_round_trip(dicom)

            # Assert
            output_group = dicom[0]
            geometry_type, coordinates = self.annotation_group_to_qupath(output_group)
            output_dict = deepcopy(input_dict)
            output_dict["geometry"]["type"] = geometry_type
            output_dict["geometry"]["coordinates"] = coordinates
            output_dict["properties"]["name"] = output_group.label
            assert input_dict == output_dict

    @pytest.mark.parametrize("file_path", test_files["qupath_geojson_advanced"])
    def test_qupath_geojson_advanced(self, file_path: Path):
        with open(file_path) as f:
            # Arrange
            input_dict: Dict = json.load(f)
            # Group annotations by type and type using key

            @dataclass(frozen=True)
            class Key:
                annotation_type: type
                type_code: AnnotationTypeCode

            @dataclass(frozen=True)
            class Value:
                label: str
                color: LabColor
                geometries: List[Geometry]

            grouped_annotations: Dict[Key, Value] = {}

            # For each annotation, make a type-category key and insert the
            # annotation in the correct dict-group. If no group exists,
            # create one using the key and insert the label.
            for input_annotation in input_dict:
                geometries = Geometry.from_geojson(input_annotation["geometry"])
                group_key = Key(
                    annotation_type=type(geometries[0]),
                    type_code=self.qupath_get_type_code(input_annotation),
                )
                try:
                    group = grouped_annotations[group_key]
                except KeyError:
                    group = Value(
                        label=self.qupath_get_label(input_annotation),
                        color=LabColor(0, 0, 0),
                        geometries=[],
                    )
                    grouped_annotations[group_key] = group
                group.geometries.extend(geometries)

            assert grouped_annotations != {}

            # For each group of annotations (same type and category) make
            # an annotation group
            annotation_groups: List[AnnotationGroup] = []
            for group_keys, group_values in grouped_annotations.items():
                annotation_group = AnnotationGroup.from_geometries(
                    geometries=group_values.geometries,
                    label=group_values.label,
                    category_code=category_code,
                    type_code=group_keys.type_code,
                )
                annotation_groups.append(annotation_group)
            assert annotation_groups != []

            # Act
            # Make a group collection and do dicom round-trip
            dicom = AnnotationInstance(annotation_groups, "volume", slide_uids)
            dicom = self.dicom_round_trip(dicom)

            # Assert
            assert dicom.groups != []
            # For each annotation group, produce a type-category_code key.
            # Get the original group using the key and check that the
            # annotations are the same.
            for output_group in dicom.groups:
                assert output_group.annotations != []
                key = Key(
                    annotation_type=output_group.geometry_type,
                    type_code=output_group.type_code,
                )
                input_group = grouped_annotations[key]
                for i, output_annotation in enumerate(output_group.annotations):
                    input_annotation = input_group.geometries[i]
                    print(len(output_annotation.geometry))
                    print(len(input_annotation))
                    print(type(output_annotation.geometry))
                    print(type(input_annotation))
                    assert output_annotation.geometry == input_annotation

    @pytest.mark.parametrize("file_path", test_files["asap"])
    def test_asap(self, file_path: Path):
        with open(file_path) as f:
            # Arrange
            annotation_xml = xmltodict.parse(f.read())["ASAP_Annotations"]
            input_dict: Dict[str, Any] = annotation_xml["Annotations"]["Annotation"]
            group = AnnotationGroup.from_geometries(
                self.asap_to_geometries(input_dict),
                label=input_dict["@Name"],
                category_code=category_code,
                type_code=type_code,
            )

            # Act
            dicom = AnnotationInstance([group], "volume", slide_uids)
            dicom = self.dicom_round_trip(dicom)

            # Assert
            output_group = dicom[0]
            output_dict = deepcopy(input_dict)
            geometry_type, output_coords = self.asap_annotation_group_to_geometries(
                output_group
            )
            output_dict["@Name"] = output_group.label
            output_dict["@Type"] = geometry_type
            if len(output_coords) == 1:
                output_dict["Coordinates"]["Coordinate"] = output_coords[0]
            else:
                output_dict["Coordinates"]["Coordinate"] = output_coords
            self.maxDiff = None
            assert input_dict == output_dict

    @pytest.mark.parametrize("file_path", test_files["sectra"])
    def test_sectra(self, file_path: Path):
        with open(file_path) as f:
            # Arrange
            input_dict: Union[Dict[str, Any], List[Dict[str, Any]]] = json.load(f)
            if isinstance(input_dict, list):
                input_dict = input_dict[0]
            group = AnnotationGroup.from_geometries(
                [self.sectra_to_geometry(input_dict)],
                label=input_dict["name"],
                category_code=category_code,
                type_code=type_code,
            )

            # Act
            dicom = AnnotationInstance([group], "volume", slide_uids)
            dicom = self.dicom_round_trip(dicom)

            # Assert
            output_group = dicom[0]
            output_annotation = output_group.annotations[0]
            geometry_type, result_coordinates = self.annotation_to_sectra(
                output_annotation
            )
            output_dict = deepcopy(input_dict)
            output_dict["type"] = geometry_type
            output_dict["content"] = result_coordinates
            if input_dict["name"] is not None:
                output_dict["name"] = output_group.label
            self.maxDiff = None
            assert input_dict == output_dict

    @pytest.mark.parametrize("file_path", test_files["cytomine"])
    def test_cytomine(self, file_path: Path):
        with open(file_path) as f:
            # Arrange
            input_dict = json.load(f)
            geometry = wkt.loads(input_dict["annotation"]["location"])
            # wkt.loads trims excess decimals, so we do the same to the
            # input
            input_dict["annotation"]["location"] = wkt.dumps(geometry, trim=True)
            group = AnnotationGroup.from_geometries(
                [Geometry.from_shapely_like(geometry)],
                label=str(input_dict["annotation"]["id"]),
                category_code=category_code,
                type_code=type_code,
            )

            # Act
            dicom = AnnotationInstance([group], "volume", slide_uids)
            dicom = self.dicom_round_trip(dicom)

            # Assert
            output_group = dicom[0]
            output_annotation = output_group.annotations[0]
            output_geometry = self.annotation_to_shapely(output_annotation)
            output_dict = deepcopy(input_dict)
            output_dict["annotation"]["location"] = output_geometry.wkt
            output_dict["annotation"]["id"] = int(dicom[0].label)
            assert input_dict == output_dict

    @pytest.mark.parametrize(
        "input_geometry",
        [
            ShapelyPoint(15123.21, 12410.01),
            ShapelyPolygon(
                [
                    (26048, 17269.375),
                    (27408, 16449.375),
                    (27300, 15557.375),
                    (25056, 16817.375),
                    (26048, 17269.375),
                ]
            ),
            ShapelyLineString(
                [
                    (27448, 18266.75),
                    (29040, 19194.75),
                    (29088, 16618.75),
                    (27464, 16874.75),
                    (26984, 17562.75),
                ]
            ),
        ],
    )
    def test_shapely(self, input_geometry):
        # Arrange
        group = AnnotationGroup.from_geometries(
            [Geometry.from_shapely_like(input_geometry)],
            label="shapely test",
            category_code=category_code,
            type_code=type_code,
        )

        # Act
        dicom = AnnotationInstance([group], "volume", slide_uids)
        dicom = self.dicom_round_trip(dicom)

        # Assert
        output_group = dicom[0]
        output_annotation = output_group.annotations[0]
        output_geometry = self.annotation_to_shapely(output_annotation)
        assert input_geometry == output_geometry

    @pytest.mark.parametrize(
        "input_annotations",
        [
            [Annotation(Point(0.0, 0.1))],
            [Annotation(Point(0.0, 0.1)), Annotation(Point(0.0, 0.1))],
        ],
    )
    def test_point_annotation(self, input_annotations: List[Annotation]):
        # Arrange

        # Act
        group = PointAnnotationGroup(
            input_annotations, "test", category_code, type_code
        )

        # Assert
        assert input_annotations == group.annotations
        for input_annotation, annotation in zip(input_annotations, group.annotations):
            assert annotation == input_annotation

    @pytest.mark.parametrize(
        "input_annotations",
        [
            [Annotation(Polyline([(0.0, 0.1), (1.0, 1.1), (2.0, 2.1)]))],
            [
                Annotation(
                    Polyline([(10.0, 10.1), (11.0, 11.1), (12.0, 12.1), (13.0, 13.1)])
                ),
                Annotation(
                    Polyline(
                        [
                            (20.0, 20.1),
                            (21.0, 11.1),
                            (22.0, 22.1),
                            (23.0, 23.1),
                            (24.0, 24.1),
                        ]
                    )
                ),
            ],
        ],
    )
    def test_line_annotation(self, input_annotations: List[Annotation]):
        # Arrange

        # Act
        group = PolylineAnnotationGroup(
            input_annotations, "test", category_code, type_code
        )

        # Assert
        assert input_annotations == group.annotations
        for input_annotation, annotation in zip(input_annotations, group.annotations):
            assert annotation == input_annotation

    def test_float_32_to_32(self):
        # Arrange
        np_input = np.array([0.0254, 0.12405], dtype=np.float32)
        input_point = Point(float(np_input[0]), float(np_input[1]))
        input_annotation = Annotation(input_point)
        group = PointAnnotationGroup(
            [input_annotation], "test", category_code, type_code, is_double=False
        )

        # act
        dicom = AnnotationInstance([group], "volume", slide_uids)
        dicom = self.dicom_round_trip(dicom)

        # Assert
        output_group = dicom[0]
        if isinstance(output_group, AnnotationGroup):
            output_point = output_group.annotations[0].geometry.to_coords()[0]
            np_output = np.array(output_point, dtype=np.float32)
            assert np_input.all() == np_output.all()

    def test_float_32_to_64(self):
        # Arrange
        np_input = np.array([0.0254, 0.12405], dtype=np.float32)
        input_point = Point(float(np_input[0]), float(np_input[1]))
        input_annotation = Annotation(input_point)
        group = PointAnnotationGroup(
            [input_annotation], "test", category_code, type_code, is_double=True
        )

        # Act
        dicom = AnnotationInstance([group], "volume", slide_uids)
        dicom = self.dicom_round_trip(dicom)

        # Assert
        output_group = dicom[0]
        if isinstance(output_group, AnnotationGroup):
            output_point = output_group.annotations[0].geometry.to_coords()[0]
            np_output = np.array(output_point, dtype=np.float32)
            assert np_input.all() == np_output.all()

    def test_float_64_to_64(self):
        # Arrange
        np_input = np.array([0.0254, 0.12405], dtype=np.float64)
        input_point = Point(float(np_input[0]), float(np_input[1]))

        input_annotation = Annotation(input_point)
        group = PointAnnotationGroup(
            [input_annotation], "test", category_code, type_code, is_double=True
        )

        # Act
        dicom = AnnotationInstance([group], "volume", slide_uids)
        dicom = self.dicom_round_trip(dicom)

        # Assert
        output_group = dicom[0]
        if isinstance(output_group, AnnotationGroup):
            output_point = output_group.annotations[0].geometry.to_coords()[0]
            np_output = np.array(output_point, dtype=np.float64)
            assert np_input.all() == np_output.all()

    def test_float_64_to_32(self):
        # Arrange
        np_input = np.array([0.0254, 0.12405], dtype=np.float64)
        input_point = Point(float(np_input[0]), float(np_input[1]))

        input_annotation = Annotation(input_point)
        group = PointAnnotationGroup(
            [input_annotation], "test", category_code, type_code, is_double=False
        )

        # Act
        dicom = AnnotationInstance([group], "volume", slide_uids)
        dicom = self.dicom_round_trip(dicom)

        # Assert
        output_group = dicom[0]
        if isinstance(output_group, AnnotationGroup):
            output_point = output_group.annotations[0].geometry.to_coords()[0]
            np_output = np.array(output_point, dtype=np.float64)
            assert np_input.all() == np_output.all()

    @pytest.mark.parametrize(
        "measurements",
        [
            [Measurement(MeasurementCode("Area"), 5, UnitCode("Pixels"))],
            [
                Measurement(MeasurementCode("Area"), 5, UnitCode("Pixels")),
                Measurement(MeasurementCode("Area"), 10, UnitCode("Pixels")),
            ],
        ],
    )
    def test_measurement(self, measurements: List[Measurement]):
        # Arrange
        point = Point(1, 1)

        # Act
        annotation = Annotation(point, measurements)

        # Assert
        assert annotation.measurements == measurements

    @pytest.mark.parametrize(
        ["annotation1", "annotation2", "expected_result"],
        [
            [
                Annotation(point, [measurement0]),
                Annotation(point, [measurement1]),
                True,
            ],
            [Annotation(point, [measurement0]), Annotation(point, []), False],
            [
                Annotation(point, [measurement0, measurement1]),
                Annotation(point, [measurement2]),
                False,
            ],
        ],
    )
    def test_measurement_one_to_one(
        self, annotation1: Annotation, annotation2: Annotation, expected_result: bool
    ):
        # Arrange
        annotation_group = PointAnnotationGroup(
            [annotation1, annotation2], "label", category_code, type_code
        )

        # Assert
        assert (
            annotation_group._measurements_is_one_to_one(area, pixels)
            == expected_result
        )

    def test_measurement_cycle(self):
        # Arrange
        area = MeasurementCode("Area")
        pixels = UnitCode("Pixels")
        measurement0 = Measurement(area, 5, pixels)
        measurement1 = Measurement(area, 10, pixels)
        point = Point(1, 1)
        annotation0 = Annotation(point, [measurement0])
        annotation1 = Annotation(point, [measurement1])
        annotation_group = PointAnnotationGroup(
            [annotation0, annotation1], "label", category_code, type_code
        )

        # Act
        dicom = AnnotationInstance([annotation_group], "volume", slide_uids)
        dicom = self.dicom_round_trip(dicom)

        # Assert
        output_group = dicom[0]

        if isinstance(output_group, PointAnnotationGroup):
            annotation0 = output_group.annotations[0]

            assert measurement0 == annotation0.get_measurements(area, pixels)[0]

    def test_large_number_of_annotations(self):
        # Arrange
        count = 10000
        point_annotations = [
            Annotation(Point(float(i), float(i))) for i in range(count)
        ]

        group = PointAnnotationGroup(
            point_annotations, "test", category_code, type_code
        )

        # Act
        dicom = AnnotationInstance([group], "volume", slide_uids)
        dicom = self.dicom_round_trip(dicom)

        # Assert
        output_group = dicom[0]
        if isinstance(output_group, PointAnnotationGroup):
            for i in range(count):
                assert Point(float(i), float(i)) == output_group.annotations[i].geometry

    def test_make_annotated_wsi_slide(self, wsi: WsiDicom, tmp_path: Path):
        # Arrange
        point_annotation = Annotation(Point(10.0, 20.0))
        group = PointAnnotationGroup(
            annotations=[point_annotation],
            label="group label",
            category_code=AnnotationCategoryCode("Tissue"),
            type_code=AnnotationTypeCode("Nucleus"),
            description="description",
        )
        assert wsi.uids is not None
        annotation_file_path = tmp_path.joinpath("annotation_for_slide.dcm")

        # Act
        annotations = AnnotationInstance([group], "volume", wsi.uids)
        annotations.save(annotation_file_path)

        # Assert
        with WsiDicom.open(tmp_path) as wsi:
            output_group = wsi.annotations[0][0]
        assert output_group == group

    @classmethod
    def dicom_round_trip(cls, dicom: AnnotationInstance) -> AnnotationInstance:
        """Saves the annotation collection to temporary file, reads the file
        and return the parsed annotation collection.

        Parameters
        ----------
        dicom: AnnotationInstance
            Collection of annotations to save.

        Returns
        -------
        AnnotationInstance
            Read back annotation collection.

        """
        with TemporaryDirectory() as tempdir:
            filename = "annotation_round_trip.dcm"
            dcm_path = str(Path(tempdir).joinpath(filename))
            dicom.save(dcm_path)
            return list(AnnotationInstance.open([dcm_path]))[0]

    @staticmethod
    def annotation_group_to_qupath(
        group: AnnotationGroup,
    ) -> Tuple[str, Union[List[float], List[List[float]], List[List[List[float]]]]]:
        if isinstance(group, PointAnnotationGroup):
            if len(group) == 1:
                return ("Point", group.annotations[0].geometry.to_list_coords()[0])
            else:
                coordinates = []
                for annotation in group.annotations:
                    coordinates += annotation.geometry.to_list_coords()
                return ("MultiPoint", coordinates)
        elif isinstance(group, PolylineAnnotationGroup):
            return ("LineString", group.annotations[0].geometry.to_list_coords())
        elif isinstance(group, PolygonAnnotationGroup):
            return ("Polygon", [group.annotations[0].geometry.to_list_coords()])
        raise NotImplementedError()

    @staticmethod
    def qupath_get_type_code(annotation: Dict) -> AnnotationTypeCode:
        type_code = annotation["properties"]["classification"]["name"]
        return AnnotationTypeCode(type_code)

    @staticmethod
    def qupath_get_color(annotation: Dict) -> Tuple[int, int, int]:
        color: Tuple[int, int, int] = annotation["properties"]["color"]
        return color

    @staticmethod
    def qupath_get_label(annotation: Dict) -> str:
        label: str = annotation["properties"]["name"]
        return label

    @staticmethod
    def asap_annotation_group_to_geometries(
        group: AnnotationGroup,
    ) -> Tuple[str, List[OrderedDict]]:
        increase_order_by_annotation = False
        if isinstance(group, PointAnnotationGroup):
            if len(group) == 1:
                geometry_type = "Dot"
            else:
                geometry_type = "PointSet"
                increase_order_by_annotation = True
        elif isinstance(group, PolygonAnnotationGroup):
            geometry_type = "Polygon"
        else:
            raise NotImplementedError(group)
        group_coordinates = [
            OrderedDict(
                {
                    "@Order": str(
                        point_index + annotation_index * increase_order_by_annotation
                    ),
                    "@X": str(item[0]),
                    "@Y": str(item[1]),
                }
            )
            for annotation_index, annotation in enumerate(group.annotations)
            for point_index, item in enumerate(annotation.geometry.to_coords())
        ]
        return geometry_type, group_coordinates

    @classmethod
    def asap_to_geometries(cls, dictionary: Dict[str, Any]) -> List[Geometry]:
        annotation_type: str = dictionary["@Type"]
        coordinate_dict = dictionary["Coordinates"]["Coordinate"]
        if annotation_type == "Dot":
            return [Point.from_dict(coordinate_dict, "@X", "@Y")]
        elif annotation_type == "PointSet":
            points = Point.multiple_from_dict(coordinate_dict, "@X", "@Y")
            return points  # type: ignore
        elif annotation_type == "Polygon":
            return [Polygon.from_dict(coordinate_dict, "@X", "@Y")]

        raise NotImplementedError(annotation_type)

    @staticmethod
    def annotation_to_sectra(
        annotation: Annotation,
    ) -> Tuple[str, List[Dict[str, float]]]:
        if type(annotation.geometry) == Polyline:
            geometry_type = "Polyline"
        elif type(annotation.geometry) == Polygon:
            geometry_type = "Area"
        else:
            raise NotImplementedError()
        group_coordinates = [
            {"x": float(item[0]), "y": float(item[1])}
            for item in annotation.geometry.to_coords()
        ]
        return geometry_type, group_coordinates

    @staticmethod
    def sectra_to_geometry(dictionary: Dict[str, Any]) -> Geometry:
        geometry_type: str = dictionary["type"]
        if geometry_type == "Area":
            return Polygon.from_dict(dictionary["content"], "x", "y")
        elif geometry_type == "Polyline":
            return Polyline.from_dict(dictionary["content"], "x", "y")
        raise NotImplementedError()

    @staticmethod
    def annotation_to_shapely(annotation: Annotation):
        if type(annotation.geometry) == Point:
            return ShapelyPoint(annotation.geometry.to_coords())
        elif type(annotation.geometry) == Polyline:
            return ShapelyLineString(annotation.geometry.to_coords())
        elif type(annotation.geometry) == Polygon:
            return ShapelyPolygon(annotation.geometry.to_coords())
        raise NotImplementedError(annotation)
