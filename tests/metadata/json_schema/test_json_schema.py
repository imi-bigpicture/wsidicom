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

from datetime import date, datetime
from pathlib import Path

import pytest

from tests.metadata.json_schema.helpers import (
    assert_dict_equals_code,
    assert_image_is_equal,
    assert_optical_path_is_equal,
)
from wsidicom.conceptcode import Code, IlluminationColorCode
from wsidicom.geometry import PointMm, SizeMm
from wsidicom.metadata import (
    Equipment,
    ExtendedDepthOfField,
    FocusMethod,
    Image,
    ImageCoordinateSystem,
    Label,
    OpticalPath,
    Overview,
    Patient,
    Pyramid,
    Series,
    Study,
)
from wsidicom.metadata.schema.json import (
    EquipmentJsonSchema,
    ImageJsonSchema,
    LabelJsonSchema,
    OpticalPathJsonSchema,
    OverviewJsonSchema,
    PatientJsonSchema,
    SeriesJsonSchema,
    StudyJsonSchema,
    WsiMetadataJsonSchema,
)
from wsidicom.metadata.schema.json.pyramid import PyramidJsonSchema
from wsidicom.metadata.wsi import WsiMetadata


class TestJsonSchema:
    @pytest.mark.parametrize(
        ["manufacturer", "model_name", "serial_number", "versions"],
        [
            ["manufacturer", "model_name", "serial_number", ["version"]],
            ["manufacturer", "model_name", "serial_number", ["version 1", "version 2"]],
            [None, None, None, None],
        ],
    )
    def test_equipment_serialize(self, equipment: Equipment):
        # Arrange

        # Act
        dumped = EquipmentJsonSchema().dump(equipment)

        # Assert
        assert isinstance(dumped, dict)
        assert dumped["manufacturer"] == equipment.manufacturer
        assert dumped["model_name"] == equipment.model_name
        assert dumped["device_serial_number"] == equipment.device_serial_number
        assert dumped["software_versions"] == equipment.software_versions

    @pytest.mark.parametrize(
        ["manufacturer", "model_name", "serial_number", "versions"],
        [
            ["manufacturer", "model_name", "serial_number", ["version"]],
            ["manufacturer", "model_name", "serial_number", ["version 1", "version 2"]],
            [None, None, None, None],
        ],
    )
    def test_equipment_deserialize(self, equipment: Equipment):
        # Arrange
        dumped = {
            "manufacturer": equipment.manufacturer,
            "model_name": equipment.model_name,
            "device_serial_number": equipment.device_serial_number,
            "software_versions": equipment.software_versions,
        }

        # Act
        loaded = EquipmentJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Equipment)
        assert loaded == equipment

    @pytest.mark.parametrize(
        [
            "acquisition_datetime",
            "focus_method",
            "extended_depth_of_field",
            "image_coordinate_system",
            "pixel_spacing",
            "focal_plane_spacing",
            "depth_of_field",
        ],
        [
            [
                datetime(2023, 8, 5),
                FocusMethod.AUTO,
                ExtendedDepthOfField(5, 0.5),
                ImageCoordinateSystem(PointMm(20.0, 30.0), 90.0),
                SizeMm(0.01, 0.01),
                0.001,
                0.01,
            ],
            [
                datetime(2023, 8, 5, 12, 13, 14, 150),
                FocusMethod.MANUAL,
                ExtendedDepthOfField(15, 0.5),
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0),
                SizeMm(0.02, 0.02),
                0.002,
                0.02,
            ],
            [None, None, None, None, None, None, None],
        ],
    )
    def test_image_serialize(self, image: Image):
        # Arrange

        # Act
        dumped = ImageJsonSchema().dump(image)

        # Assert
        assert isinstance(dumped, dict)
        assert_image_is_equal(dumped, image)

    @pytest.mark.parametrize(
        ["pixel_spacing", "image_coordinate_system"],
        [
            [
                SizeMm(0.01, 0.01),
                ImageCoordinateSystem(PointMm(20.0, 30.0), 90.0),
            ],
            [None, None],
        ],
    )
    def test_image_deserialize(self, image: Image):
        # Arrange
        dumped = {
            "acquisition_datetime": (
                image.acquisition_datetime.isoformat()
                if image.acquisition_datetime
                else None
            ),
            "focus_method": (image.focus_method.value if image.focus_method else None),
            "extended_depth_of_field": (
                {
                    "number_of_focal_planes": image.extended_depth_of_field.number_of_focal_planes,
                    "distance_between_focal_planes": image.extended_depth_of_field.distance_between_focal_planes,
                }
                if image.extended_depth_of_field
                else None
            ),
            "image_coordinate_system": (
                {
                    "origin": {
                        "x": image.image_coordinate_system.origin.x,
                        "y": image.image_coordinate_system.origin.y,
                    },
                    "rotation": image.image_coordinate_system.rotation,
                    "z_offset": image.image_coordinate_system.z_offset,
                }
                if image.image_coordinate_system
                else None
            ),
            "pixel_spacing": (
                {
                    "width": image.pixel_spacing.width,
                    "height": image.pixel_spacing.height,
                }
                if image.pixel_spacing
                else None
            ),
            "focal_plane_spacing": image.focal_plane_spacing,
            "depth_of_field": image.depth_of_field,
            "lossy_compressions": (
                [
                    {"method": compression.method.value, "ratio": compression.ratio}
                    for compression in image.lossy_compressions
                ]
                if image.lossy_compressions
                else None
            ),
        }

        # Act
        loaded = ImageJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Image)
        assert loaded == image

    @pytest.mark.parametrize(
        [
            "image_contains_phi",
            "image_coordinate_system",
            "pixel_spacing",
        ],
        [
            [
                True,
                ImageCoordinateSystem(PointMm(20.0, 30.0), 90.0),
                SizeMm(0.01, 0.01),
            ],
            [
                False,
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0),
                SizeMm(0.02, 0.02),
            ],
            [True, None, None],
        ],
    )
    def test_pyramid_serialize(self, pyramid: Pyramid):
        # Arrange

        # Act
        dumped = PyramidJsonSchema().dump(pyramid)

        # Assert
        assert isinstance(dumped, dict)
        assert_image_is_equal(dumped["image"], pyramid.image)
        assert len(dumped["optical_paths"]) == len(pyramid.optical_paths)
        for dumped_path, expected_path in zip(
            dumped["optical_paths"], pyramid.optical_paths
        ):
            assert_optical_path_is_equal(dumped_path, expected_path)
        assert dumped["contains_phi"] == pyramid.contains_phi
        assert dumped["comments"] == pyramid.comments

    @pytest.mark.parametrize(
        [
            "image_contains_phi",
            "image_coordinate_system",
            "pixel_spacing",
        ],
        [
            [
                True,
                ImageCoordinateSystem(PointMm(20.0, 30.0), 90.0),
                SizeMm(0.01, 0.01),
            ],
            [
                False,
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0),
                SizeMm(0.02, 0.02),
            ],
            [True, None, None],
        ],
    )
    def test_pyramid_deserialize(self, pyramid: Pyramid, icc_file: Path):
        # Arrange
        optical_paths = [
            OpticalPathJsonSchema().dump(path) for path in pyramid.optical_paths
        ]
        for optical_path in optical_paths:
            optical_path["icc_profile"] = str(icc_file)  # type: ignore[index]
        dumped = {
            "image": ImageJsonSchema().dump(pyramid.image),
            "optical_paths": optical_paths,
            "uid": pyramid.uid,
            "description": pyramid.description,
            "label": pyramid.label,
            "contains_phi": pyramid.contains_phi,
            "comments": pyramid.comments,
        }

        # Act
        loaded = PyramidJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Pyramid)
        assert loaded == pyramid

    @pytest.mark.parametrize(
        [
            "label_text",
            "label_barcode",
            "label_contains_phi",
            "label_image_coordinate_system",
            "label_pixel_spacing",
        ],
        [
            [
                "Label text",
                "label barcode",
                True,
                ImageCoordinateSystem(PointMm(20.0, 30.0), 90.0),
                SizeMm(0.01, 0.01),
            ],
            [
                "Label text 2",
                "label barcode 2",
                False,
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0),
                SizeMm(0.02, 0.02),
            ],
            [None, None, True, None, None],
        ],
    )
    def test_label_serialize(self, label: Label):
        # Arrange

        # Act
        dumped = LabelJsonSchema().dump(label)

        # Assert
        assert isinstance(dumped, dict)
        assert dumped["text"] == label.text
        assert dumped["barcode"] == label.barcode
        if label.image is None:
            assert "image" not in dumped
        else:
            assert_image_is_equal(dumped["image"], label.image)
        if label.optical_paths is None:
            assert "optical_paths" not in dumped
        else:
            assert len(dumped["optical_paths"]) == len(label.optical_paths)
            for dumped_path, expected_path in zip(
                dumped["optical_paths"], label.optical_paths
            ):
                assert_optical_path_is_equal(dumped_path, expected_path)
        assert dumped["contains_phi"] == label.contains_phi
        assert dumped["comments"] == label.comments

    @pytest.mark.parametrize(
        [
            "label_text",
            "label_barcode",
            "label_contains_phi",
            "label_image_coordinate_system",
            "label_pixel_spacing",
        ],
        [
            [
                "Label text",
                "label barcode",
                True,
                ImageCoordinateSystem(PointMm(20.0, 30.0), 90.0),
                SizeMm(0.01, 0.01),
            ],
            [
                "Label text 2",
                "label barcode 2",
                False,
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0),
                SizeMm(0.02, 0.02),
            ],
            [None, None, True, None, None],
        ],
    )
    def test_label_deserialize(self, label: Label, icc_file: Path):
        # Arrange
        optical_paths = (
            [OpticalPathJsonSchema().dump(path) for path in label.optical_paths]
            if label.optical_paths
            else []
        )
        for optical_path in optical_paths:
            optical_path["icc_profile"] = str(icc_file)  # type: ignore[index]
        dumped = {
            "text": label.text,
            "barcode": label.barcode,
            "image": ImageJsonSchema().dump(label.image) if label.image else None,
            "optical_paths": optical_paths,
            "contains_phi": label.contains_phi,
            "comments": label.comments,
        }

        # Act
        loaded = LabelJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Label)
        assert loaded == label

    @pytest.mark.parametrize(
        [
            "overview_contains_phi",
            "overview_image_contains_slide_label",
            "overview_image_coordinate_system",
            "overview_pixel_spacing",
        ],
        [
            [
                True,
                True,
                ImageCoordinateSystem(PointMm(20.0, 30.0), 90.0),
                SizeMm(0.01, 0.01),
            ],
            [
                False,
                False,
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0),
                SizeMm(0.02, 0.02),
            ],
            [True, True, None, None],
        ],
    )
    def test_overview_serialize(self, overview: Overview):
        # Arrange

        # Act
        dumped = OverviewJsonSchema().dump(overview)

        # Assert
        assert isinstance(dumped, dict)
        assert_image_is_equal(dumped["image"], overview.image)
        assert len(dumped["optical_paths"]) == len(overview.optical_paths)
        for dumped_path, expected_path in zip(
            dumped["optical_paths"], overview.optical_paths
        ):
            assert_optical_path_is_equal(dumped_path, expected_path)
        assert dumped["contains_label"] == overview.contains_label
        assert dumped["contains_phi"] == overview.contains_phi
        assert dumped["comments"] == overview.comments

    @pytest.mark.parametrize(
        [
            "overview_contains_phi",
            "overview_image_contains_slide_label",
            "overview_image_coordinate_system",
            "overview_pixel_spacing",
        ],
        [
            [
                True,
                True,
                ImageCoordinateSystem(PointMm(20.0, 30.0), 90.0),
                SizeMm(0.01, 0.01),
            ],
            [
                False,
                False,
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0),
                SizeMm(0.02, 0.02),
            ],
            [True, True, None, None],
        ],
    )
    def test_overview_deserialize(self, overview: Overview, icc_file: Path):
        # Arrange
        optical_paths = [
            OpticalPathJsonSchema().dump(path) for path in overview.optical_paths
        ]
        for optical_path in optical_paths:
            optical_path["icc_profile"] = str(icc_file)  # type: ignore[index]
        dumped = {
            "image": ImageJsonSchema().dump(overview.image),
            "optical_paths": optical_paths,
            "contains_phi": overview.contains_phi,
            "contains_label": overview.contains_label,
            "comments": overview.comments,
        }

        # Act
        loaded = OverviewJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Overview)
        assert loaded == overview

    @pytest.mark.parametrize(
        "illumination", [IlluminationColorCode("Full Spectrum"), 400.0]
    )
    def test_optical_path_serialize(self, optical_path: OpticalPath):
        # Arrange

        # Act
        dumped = OpticalPathJsonSchema().dump(optical_path)

        # Assert
        assert isinstance(dumped, dict)
        assert_optical_path_is_equal(dumped, optical_path)

    @pytest.mark.parametrize(
        "illumination",
        [
            IlluminationColorCode("Full Spectrum"),
            400.0,
        ],
    )
    def test_optical_path_deserialize(self, optical_path: OpticalPath, icc_file: Path):
        dumped = {
            "identifier": optical_path.identifier,
            "description": optical_path.description,
            "illumination_types": (
                [
                    {
                        "value": illumination_type.value,
                        "scheme_designator": illumination_type.scheme_designator,
                        "meaning": illumination_type.meaning,
                    }
                    for illumination_type in optical_path.illumination_types
                ]
                if optical_path.illumination_types
                else []
            ),
            "light_path_filter": (
                {
                    "filters": (
                        [
                            {
                                "value": filter.value,
                                "scheme_designator": filter.scheme_designator,
                                "meaning": filter.meaning,
                            }
                            for filter in optical_path.light_path_filter.filters
                        ]
                        if optical_path.light_path_filter.filters
                        else []
                    ),
                    "nominal": optical_path.light_path_filter.nominal,
                    "low_pass": optical_path.light_path_filter.low_pass,
                    "high_pass": optical_path.light_path_filter.high_pass,
                }
                if optical_path.light_path_filter
                else None
            ),
            "image_path_filter": (
                {
                    "filters": (
                        [
                            {
                                "value": filter.value,
                                "scheme_designator": filter.scheme_designator,
                                "meaning": filter.meaning,
                            }
                            for filter in optical_path.image_path_filter.filters
                        ]
                        if optical_path.image_path_filter.filters
                        else []
                    ),
                    "nominal": optical_path.image_path_filter.nominal,
                    "low_pass": optical_path.image_path_filter.low_pass,
                    "high_pass": optical_path.image_path_filter.high_pass,
                }
                if optical_path.image_path_filter
                else None
            ),
            "objective": (
                {
                    "lenses": (
                        [
                            {
                                "value": lens.value,
                                "scheme_designator": lens.scheme_designator,
                                "meaning": lens.meaning,
                            }
                            for lens in optical_path.objective.lenses
                        ]
                        if optical_path.objective.lenses
                        else []
                    ),
                    "condenser_power": optical_path.objective.condenser_power,
                    "objective_power": optical_path.objective.objective_power,
                    "objective_numerical_aperture": optical_path.objective.objective_numerical_aperture,
                }
                if optical_path.objective
                else None
            ),
            "lut": {
                "bits": 16,
                "red": [{"value": 0, "length": 256}],
                "green": [{"value": 0, "length": 256}],
                "blue": [{"start_value": 0, "end_value": 255, "length": 256}],
            },
            "icc_profile": (str(icc_file) if optical_path.icc_profile else None),
            "illumination": (
                {
                    "value": optical_path.illumination.value,
                    "scheme_designator": optical_path.illumination.scheme_designator,
                    "meaning": optical_path.illumination.meaning,
                }
                if isinstance(optical_path.illumination, IlluminationColorCode)
                else optical_path.illumination
            ),
        }

        # Act
        loaded = OpticalPathJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, OpticalPath)
        assert loaded == optical_path

    def test_patient_serialize(self, patient: Patient):
        # Arrange

        # Act
        dumped = PatientJsonSchema().dump(patient)

        # Assert
        assert patient.birth_date is not None
        assert patient.sex is not None
        assert patient.de_identification is not None
        assert patient.de_identification.methods is not None
        assert patient.comments is not None
        assert isinstance(dumped, dict)
        assert dumped["name"] == patient.name
        assert dumped["identifier"] == patient.identifier
        assert dumped["birth_date"] == date.isoformat(patient.birth_date)
        assert dumped["sex"] == patient.sex.value
        if isinstance(patient.species_description, Code):
            assert isinstance(patient.species_description, Code)
            assert_dict_equals_code(
                dumped["species_description"],
                patient.species_description,
            )
        else:
            assert dumped["species_description"] == patient.species_description

        assert (
            dumped["de_identification"]["identity_removed"]
            == patient.de_identification.identity_removed
        )
        method = patient.de_identification.methods[0]
        if isinstance(method, Code):
            assert_dict_equals_code(
                dumped["de_identification"]["methods"][0],
                method,
            )
        else:
            assert dumped["de_identification"]["methods"][0] == method
        assert dumped["comments"] == patient.comments

    @pytest.mark.parametrize(
        "species_description",
        [
            "specimen description",
            Code("value", "scheme", "meaning"),
        ],
    )
    @pytest.mark.parametrize(
        "patient_deidentification_method",
        [
            "identity removed",
            Code("value", "scheme", "meaning"),
            None,
        ],
    )
    def test_patient_deserialize(
        self,
        patient: Patient,
    ):
        # Arrange
        dumped = {
            "name": patient.name,
            "identifier": patient.identifier,
            "birth_date": (
                patient.birth_date.isoformat() if patient.birth_date else None
            ),
            "sex": patient.sex.value if patient.sex else None,
            "de_identification": (
                {
                    "identity_removed": True,
                    "methods": (
                        [
                            (
                                {
                                    "value": method.value,
                                    "scheme_designator": method.scheme_designator,
                                    "meaning": method.meaning,
                                }
                                if isinstance(method, Code)
                                else method
                            )
                            for method in patient.de_identification.methods
                        ]
                        if patient.de_identification.methods
                        else []
                    ),
                }
                if patient.de_identification is not None
                else None
            ),
            "species_description": (
                {
                    "value": patient.species_description.value,
                    "scheme_designator": patient.species_description.scheme_designator,
                    "meaning": patient.species_description.meaning,
                }
                if isinstance(patient.species_description, Code)
                else patient.species_description
            ),
            "comments": patient.comments,
        }

        # Act
        loaded = PatientJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Patient)
        assert loaded == patient

    def test_series_serialize(self, series: Series):
        # Arrange

        # Act
        dumped = SeriesJsonSchema().dump(series)

        # Assert
        assert isinstance(dumped, dict)
        assert dumped["uid"] == str(series.uid)
        assert dumped["number"] == series.number
        assert dumped["description"] == series.description
        assert dumped["body_part_examined"] == series.body_part_examined

    def test_series_deserialize(self, series: Series):
        dumped = {
            "uid": str(series.uid),
            "number": series.number,
            "description": series.description,
            "body_part_examined": series.body_part_examined,
        }

        # Act
        loaded = SeriesJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Series)
        assert loaded == series

    def test_study_serialize(self, study: Study):
        # Arrange

        # Act
        dumped = StudyJsonSchema().dump(study)

        # Assert
        assert study.date is not None
        assert study.time is not None
        assert isinstance(dumped, dict)
        dumped["uid"] = str(study.uid)
        dumped["identifier"] = study.identifier
        dumped["date"] = study.date.isoformat()
        dumped["time"] = study.time.isoformat()
        dumped["accession_number"] = study.accession_number
        dumped["referring_physician_name"] = study.referring_physician_name
        dumped["description"] = study.description

    def test_study_deserialize(self, study: Study):
        # Arrange
        dumped = {
            "uid": str(study.uid),
            "identifier": study.identifier,
            "date": study.date.isoformat() if study.date else None,
            "time": study.time.isoformat() if study.time else None,
            "accession_number": study.accession_number,
            "referring_physician_name": study.referring_physician_name,
            "description": study.description,
        }

        # Act
        loaded = StudyJsonSchema().load(dumped)

        # Assert
        assert isinstance(loaded, Study)
        assert loaded.uid == study.uid
        assert loaded.identifier == study.identifier
        assert loaded.date == study.date
        assert loaded.time == study.time
        assert loaded.accession_number == study.accession_number
        assert loaded.referring_physician_name == study.referring_physician_name
        assert loaded.description == study.description

    def test_metadata_serialize(
        self,
        wsi_metadata: WsiMetadata,
    ):
        # Arrange

        # Act
        dumped = WsiMetadataJsonSchema().dump(wsi_metadata)

        # Assert
        assert isinstance(dumped, dict)
        assert "study" in dumped
        assert "series" in dumped
        assert "patient" in dumped
        assert "equipment" in dumped
        assert "pyramid" in dumped
        assert "slide" in dumped
        assert "label" in dumped
        assert "overview" in dumped
