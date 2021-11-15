import json
import os
import unittest
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, Union
from pydicom.dataset import FileDataset

from pydicom.uid import generate_uid

import numpy as np
import pytest
import shapely
import xmltodict
from shapely import wkt
from wsidicom.graphical_annotations import (Annotation, AnnotationGroup,
                                            AnnotationInstance, ConceptCode,
                                            Geometry, LabColor, Measurement,
                                            Point, PointAnnotationGroup,
                                            Polygon, PolygonAnnotationGroup,
                                            Polyline, PolylineAnnotationGroup)
from wsidicom import WsiDicom
from wsidicom.uid import BaseUids

from .data_gen import create_layer_file

wsidicom_test_data_dir = os.environ.get("WSIDICOM_TESTDIR", "C:/temp/wsidicom")
sub_data_dir = "annotation"
data_dir = wsidicom_test_data_dir + '/' + sub_data_dir
typecode = ConceptCode.type('Nucleus')
categorycode = ConceptCode.category('Tissue')
base_uids = BaseUids(
    study_instance=generate_uid(),
    series_instance=generate_uid(),
    frame_of_reference=generate_uid()
)


@pytest.mark.unittest
class WsiDicomAnnotationTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_files: Dict[str, List[Path]]
        self.tempdir: TemporaryDirectory
        self.slide: WsiDicom

    @classmethod
    def setUpClass(cls):
        cls.tempdir = TemporaryDirectory()
        dirpath = Path(cls.tempdir.name)
        test_file_path = dirpath.joinpath("test_im_annotated.dcm")
        create_layer_file(test_file_path)
        cls.slide = WsiDicom.open(cls.tempdir.name)

        cls.test_files = {}
        folders = cls._get_folders()
        for folder in folders:
            folder_name = os.path.basename(folder)
            cls.test_files[folder_name] = [
                Path(folder).joinpath(item)
                for item in os.listdir(folder)
            ]

    @classmethod
    def tearDownClass(cls):
        cls.slide.close()
        cls.tempdir.cleanup()

    @classmethod
    def _get_folders(cls) -> List[Path]:
        return [
            Path(data_dir).joinpath(item)
            for item in os.listdir(data_dir)
        ]

    @classmethod
    def dicom_round_trip(
        cls,
        dicom: AnnotationInstance
    ) -> AnnotationInstance:
        """Saves the annotation collection to temporary file, reads the file
        and return the parsed annotation collection.

        Parameters
        ----------
        dicom: AnnotationInstance
            Collection of annotations to save.

        Returns
        ----------
        AnnotationInstance
            Read back annotation collection.

        """
        tempdir = TemporaryDirectory()
        dirpath = Path(tempdir.name)
        filename = "annotation_round_trip.dcm"
        dcm_path = str(dirpath.joinpath(filename))
        dicom.save(dcm_path)
        read_annotations = AnnotationInstance.open([dcm_path])[0]
        tempdir.cleanup()
        return read_annotations

    @staticmethod
    def annotation_group_to_qupath(
        group: AnnotationGroup
    ) -> Tuple[str, Union[
            List[float],
            List[List[float]],
            List[List[List[float]]]
    ]]:
        coordinates: Union[
            List[float],
            List[List[float]],
            List[List[List[float]]]
        ]
        if(isinstance(group, PointAnnotationGroup)):
            if len(group) == 1:
                geometry_type = 'Point'
                annotation = group.annotations[0]
                coordinates = annotation.geometry.to_list_coords()[0]
            else:
                geometry_type = 'MultiPoint'
                coordinates = []
                for annotation in group.annotations:
                    coordinates += annotation.geometry.to_list_coords()
        elif(isinstance(group, PolylineAnnotationGroup)):
            geometry_type = 'LineString'
            annotation = group.annotations[0]
            coordinates = annotation.geometry.to_list_coords()
        elif(isinstance(group, PolygonAnnotationGroup)):
            geometry_type = 'Polygon'
            annotation = group.annotations[0]
            coordinates = [annotation.geometry.to_list_coords()]
        else:
            raise NotImplementedError
        return (geometry_type, coordinates)

    def test_qupath_geojson(self):
        files = self.test_files['qupath_geojson']
        for file_path in files:
            with open(file_path) as f:
                print(file_path)
                input_dict: Dict[str, Any] = json.load(f)
                group = AnnotationGroup.from_geometries(
                    Geometry.from_geojson(input_dict["geometry"]),
                    label=input_dict["properties"]["name"],
                    categorycode=categorycode,
                    typecode=typecode
                )
                dicom = AnnotationInstance([group], 'volume', base_uids)
                dicom = self.dicom_round_trip(dicom)
                output_group = dicom[0]
                geometry_type, coordinates = (
                    self.annotation_group_to_qupath(output_group)
                )
                output_dict = deepcopy(input_dict)
                output_dict["geometry"]["type"] = geometry_type
                output_dict["geometry"]["coordinates"] = coordinates
                output_dict["properties"]["name"] = output_group.label
                self.assertDictEqual(input_dict, output_dict)

    @staticmethod
    def qupath_get_typecode(annotation: Dict) -> ConceptCode:
        typecode = annotation["properties"]["classification"]["name"]
        return ConceptCode.type(typecode)

    @staticmethod
    def qupath_get_color(annotation: Dict) -> Tuple[int, int, int]:
        color: Tuple[int, int, int] = annotation["properties"]["color"]
        return color

    @staticmethod
    def qupath_get_label(annotation: Dict) -> str:
        label: str = annotation["properties"]["name"]
        return label

    def test_qupath_geojson_advanced(self):
        files = self.test_files['qupath_geojson_advanced']
        for file_path in files:
            with open(file_path) as f:
                print(file_path)
                input_dict: Dict = json.load(f)
                # Group annotations by type and type using key

                @dataclass(unsafe_hash=True)
                class Key:
                    annotation_type: type
                    typecode: ConceptCode

                @dataclass(unsafe_hash=True)
                class Value:
                    label: str
                    color: LabColor
                    geometries: List[Geometry]

                grouped_annotations: Dict[Key, Value] = {}

                # For each annotation, make a type-category key and insert the
                # annotation in the correct dict-group. If no group exists,
                # create one using the key and insert the label.
                for input_annotation in input_dict:
                    geometries = Geometry.from_geojson(
                        input_annotation["geometry"]
                    )
                    group_key = Key(
                        annotation_type=type(geometries[0]),
                        typecode=self.qupath_get_typecode(
                            input_annotation
                        )
                    )
                    try:
                        group = grouped_annotations[group_key]
                    except KeyError:
                        group = Value(
                            label=self.qupath_get_label(input_annotation),
                            color=LabColor(0, 0, 0),
                            geometries=[]
                        )
                        grouped_annotations[group_key] = group
                    group.geometries += geometries

                self.assertNotEqual(grouped_annotations, {})

                # For each group of annotations (same type and category) make
                # an annotation group
                annotation_groups: List[AnnotationGroup] = []
                for group_keys, group_values in grouped_annotations.items():
                    annotation_group = AnnotationGroup.from_geometries(
                        geometries=group_values.geometries,
                        label=group_values.label,
                        categorycode=categorycode,
                        typecode=group_keys.typecode
                    )
                    annotation_groups.append(annotation_group)
                self.assertNotEqual(annotation_groups, [])

                # Make a group collection and do dicom round-trip
                dicom = AnnotationInstance(
                    annotation_groups,
                    'volume',
                    base_uids
                )
                dicom = self.dicom_round_trip(dicom)

                self.assertNotEqual(dicom.groups, [])
                # For each annotation group, produce a type-categorycode key.
                # Get the original group using the key and check that the
                # annotations are the same.
                for output_group in dicom.groups:
                    self.assertNotEqual(output_group.annotations, [])
                    key = Key(
                        annotation_type=output_group.geometry_type,
                        typecode=output_group.typecode
                    )
                    input_group = grouped_annotations[key]
                    output = enumerate(output_group.annotations)
                    for i, output_annotation in output:
                        input_annotation = input_group.geometries[i]
                        self.assertEqual(
                            output_annotation.geometry,
                            input_annotation
                        )

    @staticmethod
    def asap_annotation_group_to_geometries(
        group: AnnotationGroup
    ) -> Tuple[str, List[OrderedDict]]:
        increase_order_by_annotation = False
        if(isinstance(group, PointAnnotationGroup)):
            if len(group) == 1:
                geometry_type = 'Dot'
            else:
                geometry_type = 'PointSet'
                increase_order_by_annotation = True
        elif(isinstance(group, PolygonAnnotationGroup)):
            geometry_type = 'Polygon'
        else:
            raise NotImplementedError(group)
        group_coordinates = [
            OrderedDict({
                            '@Order': str(
                                point_index +
                                annotation_index * increase_order_by_annotation
                            ),
                            '@X': str(item[0]),
                            '@Y': str(item[1])
            })
            for annotation_index, annotation in enumerate(group.annotations)
            for point_index, item in enumerate(annotation.geometry.to_coords())
        ]
        return geometry_type, group_coordinates

    @classmethod
    def asap_to_geometries(cls, dictionary: Dict[str, Any]) -> List[Geometry]:
        annotation_type: str = dictionary["@Type"]
        coordinate_dict = dictionary['Coordinates']['Coordinate']
        if(annotation_type == 'Dot'):
            return [Point.from_dict(coordinate_dict, "@X", "@Y")]
        elif(annotation_type == 'PointSet'):
            return Point.multiple_from_dict(coordinate_dict, "@X", "@Y")
        elif(annotation_type == 'Polygon'):
            return [Polygon.from_dict(coordinate_dict, "@X", "@Y")]

        raise NotImplementedError(annotation_type)

    def test_asap(self):
        files = self.test_files['asap']
        for file_path in files:
            with open(file_path) as f:
                print(file_path)
                annotation_xml = xmltodict.parse(f.read())["ASAP_Annotations"]
                input_dict: Dict[str, Any] = (
                    annotation_xml["Annotations"]["Annotation"]
                )
                group = AnnotationGroup.from_geometries(
                    self.asap_to_geometries(input_dict),
                    label=input_dict["@Name"],
                    categorycode=categorycode,
                    typecode=typecode
                )
                dicom = AnnotationInstance([group], 'volume', base_uids)
                dicom = self.dicom_round_trip(dicom)
                output_group = dicom[0]
                output_dict = deepcopy(input_dict)
                geometry_type, output_coords = (
                    self.asap_annotation_group_to_geometries(output_group)
                )
                output_dict["@Name"] = output_group.label
                output_dict['@Type'] = geometry_type
                if len(output_coords) == 1:
                    output_dict["Coordinates"]["Coordinate"] = output_coords[0]
                else:
                    output_dict["Coordinates"]["Coordinate"] = output_coords
                self.maxDiff = None
                self.assertDictEqual(input_dict, output_dict)

    @staticmethod
    def annotation_to_sectra(
        annotation: Annotation
    ) -> Tuple[str, List[Dict[str, float]]]:
        if(type(annotation.geometry) == Polyline):
            geometry_type = 'Polyline'
        elif(type(annotation.geometry) == Polygon):
            geometry_type = 'Area'
        else:
            raise NotImplementedError
        group_coordinates = [
            {'x': float(item[0]), 'y': float(item[1])}
            for item in annotation.geometry.to_coords()
        ]
        return geometry_type, group_coordinates

    @staticmethod
    def sectra_to_geometry(
        dictionary: Dict[str, Any]
    ) -> Geometry:
        geometry_type: str = dictionary["type"]
        if(geometry_type == 'Area'):
            return Polygon.from_dict(dictionary["content"], "x", "y")
        elif(geometry_type == 'Polyline'):
            return Polyline.from_dict(dictionary["content"], "x", "y")
        raise NotImplementedError()

    def test_sectra(self):
        files = self.test_files['pathologycore']
        for file_path in files:
            with open(file_path) as f:
                print(file_path)
                input_dict: Union[Dict[str, Any], List[Dict[str, Any]]] = (
                    json.load(f)
                )
                if(isinstance(input_dict, list)):
                    input_dict = input_dict[0]
                group = AnnotationGroup.from_geometries(
                    [self.sectra_to_geometry(input_dict)],
                    label=input_dict["name"],
                    categorycode=categorycode,
                    typecode=typecode
                )
                dicom = AnnotationInstance([group], 'volume', base_uids)
                dicom = self.dicom_round_trip(dicom)
                output_group = dicom[0]
                output_annotation = output_group.annotations[0]
                geometry_type, result_coordinates = (
                    self.annotation_to_sectra(output_annotation)
                )
                output_dict = deepcopy(input_dict)
                output_dict["type"] = geometry_type
                output_dict["content"] = result_coordinates
                if input_dict["name"] is not None:
                    output_dict["name"] = output_group.label
                self.maxDiff = None
                self.assertDictEqual(input_dict, output_dict)

    def test_cytomine(self):
        files = self.test_files['cytomine']
        for file_path in files:
            with open(file_path) as f:
                print(file_path)
                input_dict = json.load(f)
                geometry = wkt.loads(input_dict["annotation"]["location"])
                # wkt.loads trims excess decimals, so we do the same to the
                # input
                input_dict["annotation"]["location"] = wkt.dumps(
                    geometry, trim=True
                )
                group = AnnotationGroup.from_geometries(
                    [Geometry.from_shapely_like(geometry)],
                    label=str(input_dict["annotation"]["id"]),
                    categorycode=categorycode,
                    typecode=typecode
                )
                dicom = AnnotationInstance([group], 'volume', base_uids)
                dicom = self.dicom_round_trip(dicom)
                output_group = dicom[0]
                output_annotation = output_group.annotations[0]
                output_geometry = self.annotation_to_shapely(output_annotation)
                output_dict = deepcopy(input_dict)
                output_dict["annotation"]["location"] = output_geometry.wkt
                output_dict["annotation"]["id"] = int(dicom[0].label)
                self.assertDictEqual(input_dict, output_dict)

    @staticmethod
    def annotation_to_shapely(annotation: Annotation):
        if(type(annotation.geometry) == Point):
            return shapely.geometry.Point(annotation.geometry.to_coords())
        elif(type(annotation.geometry) == Polyline):
            return shapely.geometry.LineString(annotation.geometry.to_coords())
        elif(type(annotation.geometry) == Polygon):
            return shapely.geometry.Polygon(annotation.geometry.to_coords())
        raise NotImplementedError(annotation)

    def test_shapely(self):
        input_geometries = [
            shapely.geometry.Point(15123.21, 12410.01),
            shapely.geometry.Polygon([
                (26048, 17269.375),
                (27408, 16449.375),
                (27300, 15557.375),
                (25056, 16817.375),
                (26048, 17269.375)
            ]),
            shapely.geometry.LineString([
                (27448, 18266.75),
                (29040, 19194.75),
                (29088, 16618.75),
                (27464, 16874.75),
                (26984, 17562.75)
            ])
        ]
        for input_geometry in input_geometries:
            print(input_geometry)
            group = AnnotationGroup.from_geometries(
                [Geometry.from_shapely_like(input_geometry)],
                label="shapely test",
                categorycode=categorycode,
                typecode=typecode
            )
            dicom = AnnotationInstance([group], 'volume', base_uids)
            dicom = self.dicom_round_trip(dicom)
            output_group = dicom[0]
            output_annotation = output_group.annotations[0]
            output_geometry = self.annotation_to_shapely(output_annotation)
            self.assertEqual(input_geometry, output_geometry)

    def test_point_annotation(self):
        input_annotation = Annotation(Point(0.0, 0.1))
        group = PointAnnotationGroup(
            [input_annotation],
            'test',
            categorycode,
            typecode
        )
        output_group = group[0]
        self.assertEqual(input_annotation, output_group)

        input_annotations = [
            Annotation(Point(0.0, 0.1)),
            Annotation(Point(0.0, 0.1))
        ]
        group = PointAnnotationGroup(
            input_annotations,
            'test',
            categorycode,
            typecode
        )

        output_group = group[0]
        self.assertEqual(input_annotations[0], output_group)

        output_group = group[1]
        self.assertEqual(input_annotations[1], output_group)

        output_groups = group[list(range(0, 2))]
        self.assertEqual(input_annotations, output_groups)

    def test_line_annotation(self):
        input_annotation = Annotation(
            Polyline([(0.0, 0.1), (1.0, 1.1), (2.0, 2.1)])
        )
        group = PolylineAnnotationGroup(
            [input_annotation],
            'test',
            categorycode,
            typecode
        )
        output_group = group[0]
        self.assertEqual(input_annotation, output_group)

        input_annotations = [
            Annotation(Polyline(
                [(10.0, 10.1), (11.0, 11.1), (12.0, 12.1), (13.0, 13.1)])
            ),
            Annotation(Polyline(
                [(20.0, 20.1), (21.0, 11.1), (22.0, 22.1), (23.0, 23.1),
                 (24.0, 24.1)])
            )
        ]
        group = PolylineAnnotationGroup(
            input_annotations,
            'test',
            categorycode,
            typecode
        )
        output_group = group[0]
        self.assertEqual(input_annotations[0], output_group)

        output_group = group[1]
        self.assertEqual(input_annotations[1], output_group)

        output_groups = group[list(range(0, 2))]
        self.assertEqual(input_annotations, output_groups)

    def test_float_32_to_32(self):
        np_input = np.array([0.0254, 0.12405], dtype=np.float32)
        input_point = Point(float(np_input[0]), float(np_input[1]))
        input_annotation = Annotation(input_point)
        group = PointAnnotationGroup(
            [input_annotation],
            "test",
            categorycode,
            typecode,
            is_double=False
        )

        dicom = AnnotationInstance([group], 'volume', base_uids)
        dicom = self.dicom_round_trip(dicom)
        output_group = dicom[0]

        if(isinstance(output_group, AnnotationGroup)):
            output_point = output_group.annotations[0].geometry.to_coords()[0]
            np_output = np.array(output_point, dtype=np.float32)
            self.assertEqual(np_input.all(), np_output.all())

    def test_float_32_to_64(self):
        np_input = np.array([0.0254, 0.12405], dtype=np.float32)
        input_point = Point(float(np_input[0]), float(np_input[1]))
        input_annotation = Annotation(input_point)
        group = PointAnnotationGroup(
            [input_annotation],
            "test",
            categorycode,
            typecode,
            is_double=True
        )
        dicom = AnnotationInstance([group], 'volume', base_uids)
        dicom = self.dicom_round_trip(dicom)
        output_group = dicom[0]

        if(isinstance(output_group, AnnotationGroup)):
            output_point = output_group.annotations[0].geometry.to_coords()[0]
            np_output = np.array(output_point, dtype=np.float32)
            self.assertEqual(np_input.all(), np_output.all())

    def test_float_64_to_64(self):
        np_input = np.array([0.0254, 0.12405], dtype=np.float64)
        input_point = Point(float(np_input[0]), float(np_input[1]))

        input_annotation = Annotation(input_point)
        group = PointAnnotationGroup(
            [input_annotation],
            "test",
            categorycode,
            typecode,
            is_double=True
        )
        dicom = AnnotationInstance([group], 'volume', base_uids)
        dicom = self.dicom_round_trip(dicom)
        output_group = dicom[0]

        if(isinstance(output_group, AnnotationGroup)):
            output_point = output_group.annotations[0].geometry.to_coords()[0]
            np_output = np.array(output_point, dtype=np.float64)
            self.assertEqual(np_input.all(), np_output.all())

    def test_float_64_to_32(self):
        np_input = np.array([0.0254, 0.12405], dtype=np.float64)
        input_point = Point(float(np_input[0]), float(np_input[1]))

        input_annotation = Annotation(input_point)
        group = PointAnnotationGroup(
            [input_annotation],
            "test",
            categorycode,
            typecode,
            is_double=False
        )

        dicom = AnnotationInstance([group], 'volume', base_uids)
        dicom = self.dicom_round_trip(dicom)
        output_group = dicom[0]

        if(isinstance(output_group, AnnotationGroup)):
            output_point = output_group.annotations[0].geometry.to_coords()[0]
            np_output = np.array(output_point, dtype=np.float64)
            self.assertEqual(np_input.all(), np_output.all())

    def test_measurement(self):
        area = ConceptCode.measurement('Area')
        pixels = ConceptCode.unit('Pixels')
        measurement0 = Measurement(area, 5, pixels)
        point = Point(1, 1)
        annotation = Annotation(point, [measurement0])
        output = annotation.measurements[0]
        self.assertEqual(measurement0, output)

        measurement1 = Measurement(area, 10, pixels)
        annotation = Annotation(point, [measurement0, measurement1])
        outputs = annotation.measurements
        self.assertEqual([measurement0, measurement1], outputs)

    def test_measurement_one_to_one(self):
        area = ConceptCode.measurement('Area')
        pixels = ConceptCode.unit('Pixels')
        measurement0 = Measurement(area, 5, pixels)
        measurement1 = Measurement(area, 10, pixels)
        measurement2 = Measurement(area, 15, pixels)
        point = Point(1, 1)
        annotation0 = Annotation(point, [measurement0])
        annotation1 = Annotation(point, [measurement1])
        annotation_group = PointAnnotationGroup(
            [annotation0, annotation1],
            "label",
            categorycode,
            typecode
        )

        self.assertTrue(
            annotation_group._measurements_is_one_to_one(area, pixels)
        )

        annotation0 = Annotation(point, [measurement0])
        annotation1 = Annotation(point, [])
        annotation_group = PointAnnotationGroup(
            [annotation0, annotation1],
            "label",
            categorycode,
            typecode
        )
        self.assertFalse(
            annotation_group._measurements_is_one_to_one(area, pixels)
        )

        annotation0 = Annotation(point, [measurement0, measurement1])
        annotation1 = Annotation(point, [measurement2])
        annotation_group = PointAnnotationGroup(
            [annotation0, annotation1],
            "label",
            categorycode,
            typecode
        )
        self.assertFalse(
            annotation_group._measurements_is_one_to_one(area, pixels)
        )

    def test_measurement_cycle(self):
        area = ConceptCode.measurement('Area')
        pixels = ConceptCode.unit('Pixels')
        measurement0 = Measurement(area, 5, pixels)
        measurement1 = Measurement(area, 10, pixels)
        measurement2 = Measurement(area, 15, pixels)
        point = Point(1, 1)
        annotation0 = Annotation(point, [measurement0])
        annotation1 = Annotation(point, [measurement1])
        annotation_group = PointAnnotationGroup(
            [annotation0, annotation1],
            "label",
            categorycode,
            typecode
        )

        dicom = AnnotationInstance([annotation_group], 'volume', base_uids)
        dicom = self.dicom_round_trip(dicom)

        output_group = dicom[0]

        if(isinstance(output_group, PointAnnotationGroup)):
            annotation0 = output_group.annotations[0]

            self.assertEqual(
                measurement0,
                annotation0.get_measurements(area, pixels)[0]
            )

    def test_large_number_of_annotations(self):
        count = 10000
        point_annotations = [
            Annotation(Point(float(i), float(i)))
            for i in range(count)
        ]

        print(type(point_annotations[0]))
        group = PointAnnotationGroup(
            point_annotations,
            'test',
            categorycode,
            typecode
        )

        dicom = AnnotationInstance(
                [group],
                'volume',
                base_uids
        )
        dicom = self.dicom_round_trip(dicom)
        output_group = dicom[0]
        if isinstance(output_group, PointAnnotationGroup):
            for i in range(count):
                self.assertEqual(
                    Point(float(i), float(i)),
                    output_group.annotations[i].geometry
                )

    def test_make_annotated_wsi_slide(self):
        point_annotation = Annotation(Point(10.0, 20.0))
        group = PointAnnotationGroup(
            annotations=[point_annotation],
            label='group label',
            categorycode=ConceptCode.category('Tissue'),
            typecode=ConceptCode.type('Nucleus'),
            description='description'
        )
        annotations = AnnotationInstance(
            [group],
            'volume',
            self.slide.uids
        )
        dirpath = Path(self.tempdir.name)
        annotation_file_path = dirpath.joinpath("annotation_for_slide.dcm")
        annotations.save(annotation_file_path)

        slide = WsiDicom.open(self.tempdir.name)
        output_group = slide.annotations[0][0]
        slide.close()
        self.assertEqual(output_group, group)
