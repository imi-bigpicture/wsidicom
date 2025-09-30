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

from datetime import datetime
from typing import Optional

import pytest
from pydicom import Dataset

from tests.metadata.dicom_schema.helpers import (
    assert_dicom_code_dataset_equals_code,
    assert_dicom_code_sequence_equals_codes,
    assert_dicom_equipment_equals_equipment,
    assert_dicom_image_equals_image,
    assert_dicom_issuer_of_identifier_equals_issuer_of_identifier,
    assert_dicom_label_equals_label,
    assert_dicom_optical_path_equals_optical_path,
    assert_dicom_overview_equals_overview,
    assert_dicom_patient_equals_patient,
    assert_dicom_series_equals_series,
    assert_dicom_study_equals_study,
    bool_to_dicom_literal,
)
from wsidicom.codec import LossyCompressionIsoStandard
from wsidicom.conceptcode import (
    IlluminationColorCode,
    ImagePathFilterCode,
    LenseCode,
    LightPathFilterCode,
)
from wsidicom.geometry import Orientation, PointMm, SizeMm
from wsidicom.instance import ImageType
from wsidicom.metadata import (
    Equipment,
    ExtendedDepthOfField,
    FocusMethod,
    Image,
    ImageCoordinateSystem,
    ImagePathFilter,
    Label,
    LightPathFilter,
    LocalIssuerOfIdentifier,
    LossyCompression,
    Objectives,
    OpticalPath,
    Overview,
    Patient,
    Pyramid,
    Series,
    Slide,
    SlideSample,
    SpecimenIdentifier,
    Study,
    UniversalIssuerOfIdentifier,
    UniversalIssuerType,
    WsiMetadata,
)
from wsidicom.metadata.schema.dicom.defaults import defaults
from wsidicom.metadata.schema.dicom.equipment import EquipmentDicomSchema
from wsidicom.metadata.schema.dicom.image import (
    ImageCoordinateSystemDicomSchema,
    ImageDicomSchema,
)
from wsidicom.metadata.schema.dicom.label import (
    LabelDicomSchema,
    LabelOnlyDicomSchema,
)
from wsidicom.metadata.schema.dicom.optical_path import (
    OpticalPathDicomSchema,
)
from wsidicom.metadata.schema.dicom.overview import OverviewDicomSchema
from wsidicom.metadata.schema.dicom.patient import PatientDicomSchema
from wsidicom.metadata.schema.dicom.series import SeriesDicomSchema
from wsidicom.metadata.schema.dicom.slide import SlideDicomSchema
from wsidicom.metadata.schema.dicom.study import StudyDicomSchema
from wsidicom.metadata.schema.dicom.wsi import WsiMetadataDicomSchema


class TestDicomSchema:
    @pytest.mark.parametrize(
        ["manufacturer", "model_name", "serial_number", "versions"],
        [
            ["manufacturer", "model_name", "serial_number", ["version"]],
            ["manufacturer", "model_name", "serial_number", ["version 1", "version 2"]],
        ],
    )
    def test_serialize_eqipment(self, equipment: Equipment):
        # Arrange
        schema = EquipmentDicomSchema()
        assert equipment.software_versions is not None

        # Act
        serialized = schema.dump(equipment)

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_equipment_equals_equipment(serialized, equipment)

    def test_serialize_defaualt_eqipment(self):
        # Arrange
        equipment = Equipment()
        schema = EquipmentDicomSchema()

        # Act
        serialized = schema.dump(equipment)

        # Assert
        assert isinstance(serialized, Dataset)
        assert serialized.Manufacturer == defaults.string
        assert serialized.ManufacturerModelName == defaults.string
        assert serialized.DeviceSerialNumber == defaults.string
        assert serialized.SoftwareVersions == defaults.string

    @pytest.mark.parametrize(
        ["manufacturer", "model_name", "serial_number", "versions"],
        [
            ["manufacturer", "model_name", "serial_number", ["version"]],
            ["manufacturer", "model_name", "serial_number", ["version 1", "version 2"]],
            [None, None, None, None],
        ],
    )
    def test_deserialize_equipment(
        self, dicom_equipment: Dataset, equipment: Equipment
    ):
        # Arrange
        schema = EquipmentDicomSchema()

        # Act
        deserialized = schema.load(dicom_equipment)

        # Assert
        assert isinstance(deserialized, Equipment)
        assert deserialized == equipment

    @pytest.mark.parametrize(
        [
            "acquisition_datetime",
            "focus_method",
            "extended_depth_of_field",
            "image_coordinate_system",
            "pixel_spacing",
            "focal_plane_spacing",
            "depth_of_field",
            "lossy_compressions",
        ],
        [
            [
                datetime(2023, 8, 5),
                FocusMethod.AUTO,
                ExtendedDepthOfField(5, 0.5),
                ImageCoordinateSystem(PointMm(20.0, 30.0), 90.0),
                SizeMm(0.5, 0.5),
                0.25,
                2.5,
                None,
            ],
            [
                datetime(2023, 8, 5, 12, 13, 14, 150),
                FocusMethod.MANUAL,
                ExtendedDepthOfField(15, 0.5),
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0, 1.0),
                SizeMm(0.5, 0.5),
                0.25,
                2.5,
                [
                    LossyCompression(LossyCompressionIsoStandard.JPEG_LOSSY, 0.5),
                    LossyCompression(
                        LossyCompressionIsoStandard.JPEG_2000_IRREVERSIBLE, 0.2
                    ),
                ],
            ],
        ],
    )
    def test_serialize_image(self, image: Image):
        # Arrange
        schema = ImageDicomSchema()

        # Act
        serialized = schema.dump(image)

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_image_equals_image(serialized, image)

    def test_serialize_default_image(self):
        # Arrange
        image = Image()
        schema = ImageDicomSchema()

        # Act
        serialized = schema.dump(image)

        # Assert
        assert isinstance(serialized, Dataset)
        assert serialized.AcquisitionDateTime == defaults.date_time
        assert serialized.FocusMethod == defaults.focus_method.name
        assert serialized.ExtendedDepthOfField == "NO"
        assert "LossyImageCompressionMethod" not in serialized
        assert "LossyImageCompressionRatio" not in serialized
        assert serialized.LossyImageCompression == "00"

    @pytest.mark.parametrize(
        [
            "acquisition_datetime",
            "focus_method",
            "extended_depth_of_field",
            "image_coordinate_system",
            "pixel_spacing",
            "focal_plane_spacing",
            "depth_of_field",
            "lossy_compressions",
        ],
        [
            [
                datetime(2023, 8, 5),
                FocusMethod.AUTO,
                ExtendedDepthOfField(5, 0.5),
                ImageCoordinateSystem(PointMm(20.0, 30.0), 90.0),
                SizeMm(0.5, 0.5),
                0.25,
                2.5,
                None,
            ],
            [
                datetime(2023, 8, 5, 12, 13, 14, 150),
                FocusMethod.MANUAL,
                ExtendedDepthOfField(15, 0.5),
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0, 1.0),
                SizeMm(0.5, 0.5),
                0.25,
                2.5,
                [
                    LossyCompression(LossyCompressionIsoStandard.JPEG_LOSSY, 0.5),
                    LossyCompression(
                        LossyCompressionIsoStandard.JPEG_2000_IRREVERSIBLE, 0.2
                    ),
                ],
            ],
            [None, None, None, None, None, None, None, None],
        ],
    )
    @pytest.mark.parametrize("valid_dicom", [True, False])
    def test_deserialize_image(self, dicom_image: Dataset, image: Image):
        # Arrange
        schema = ImageDicomSchema()

        # Act
        deserialized = schema.load(dicom_image)
        assert isinstance(deserialized, Image)
        assert deserialized == image

    def test_serialize_label_base(self, label: Label):
        # Arrange
        schema = LabelOnlyDicomSchema()

        # Act
        serialized = schema.dump(label)

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_label_equals_label(serialized, label)
        assert "SpecimenLabelInImage" not in serialized

    def test_serialize_label_image(self, label: Label):
        # Arrange
        schema = LabelDicomSchema()

        # Act
        serialized = schema.dump(label)

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_label_equals_label(serialized, label)
        assert serialized.SpecimenLabelInImage == bool_to_dicom_literal(True)

    def test_serialize_default_label_base(self):
        # Arrange
        label = Label()
        schema = LabelOnlyDicomSchema()

        # Act
        serialized = schema.dump(label)

        # Assert
        assert isinstance(serialized, Dataset)

        assert "LabelText" not in serialized
        assert "BarcodeValue" not in serialized
        assert "SpecimenLabelInImage" not in serialized
        assert "BurnedInAnnotation" not in serialized

    def test_serialize_default_label_image(self):
        # Arrange
        label = Label()
        schema = LabelDicomSchema()

        # Act
        serialized = schema.dump(label)

        # Assert
        assert isinstance(serialized, Dataset)
        assert serialized.LabelText is None
        assert serialized.BarcodeValue is None
        assert serialized.SpecimenLabelInImage == "YES"
        assert serialized.BurnedInAnnotation == "YES"

    @pytest.mark.parametrize("image_type", [ImageType.OVERVIEW, ImageType.VOLUME])
    def test_deserialize_label_base(self, dicom_label: Dataset, label: Label):
        # Arrange
        schema = LabelOnlyDicomSchema()

        # Act
        deserialized = schema.load(dicom_label)

        # Assert
        assert isinstance(deserialized, Label)
        assert deserialized.text == label.text
        assert deserialized.barcode == label.barcode
        assert deserialized.image is None
        assert deserialized.optical_paths is None

    @pytest.mark.parametrize("image_type", [ImageType.LABEL])
    def test_deserialize_label_image(self, dicom_label: Dataset, label: Label):
        # Arrange
        schema = LabelDicomSchema()

        # Act
        deserialized = schema.load(dicom_label)

        # Assert
        assert isinstance(deserialized, Label)
        assert deserialized == label

    def test_serialize_overview(self, overview: Overview):
        # Arrange
        schema = OverviewDicomSchema()

        # Act
        serialized = schema.dump(overview)

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_overview_equals_overview(serialized, overview)

    def test_serialize_default_overview(self):
        # Arrange
        overview = Overview(image=Image(), optical_paths=[OpticalPath()])
        schema = OverviewDicomSchema()

        # Act
        serialized = schema.dump(overview)

        # Assert
        assert isinstance(serialized, Dataset)
        # if image_type == ImageType.LABEL:
        #     assert serialized.LabelText is None
        #     assert serialized.BarcodeValue is None
        #     assert serialized.SpecimenLabelInImage == "YES"
        #     assert serialized.BurnedInAnnotation == "YES"
        # else:
        #     assert "LabelText" not in serialized
        #     assert "BarcodeValue" not in serialized
        #     assert "SpecimenLabelInImage" not in serialized
        #     assert "BurnedInAnnotation" not in serialized

    def test_deserialize_overview(self, dicom_overview: Dataset, overview: Overview):
        # Arrange
        schema = OverviewDicomSchema()

        # Act
        deserialized = schema.load(dicom_overview)

        # Assert
        assert isinstance(deserialized, Overview)
        # if image_type == ImageType.LABEL:
        #     assert deserialized.text == label.text
        #     assert deserialized.barcode == label.barcode
        # elif image_type == ImageType.VOLUME:
        #     assert deserialized.text is None
        #     assert deserialized.barcode is None
        # elif image_type == ImageType.OVERVIEW:
        #     assert deserialized.text is None
        #     assert deserialized.barcode is None

    @pytest.mark.parametrize(
        "illumination", [IlluminationColorCode("Full Spectrum"), 400.0]
    )
    def test_serialize_optical_path(self, optical_path: OpticalPath):
        # Arrange
        schema = OpticalPathDicomSchema()

        # Act
        serialized = schema.dump(optical_path)

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_optical_path_equals_optical_path(serialized, optical_path)

    def test_serialize_default_optical_path(self):
        # Arrange
        optical_path = OpticalPath()
        schema = OpticalPathDicomSchema()

        # Act
        serialized = schema.dump(optical_path)

        # Assert
        assert isinstance(serialized, Dataset)
        assert "OpticalPathIdentifier" in serialized
        assert_dicom_code_sequence_equals_codes(
            serialized.IlluminationTypeCodeSequence, [defaults.illumination_type]
        )
        assert "OpticalPathDescription" not in serialized
        assert "IlluminationWaveLength" not in serialized
        assert_dicom_code_dataset_equals_code(
            serialized.IlluminationColorCodeSequence[0], defaults.illumination
        )
        assert "LensesCodeSequence" not in serialized
        assert "CondenserLensPower" not in serialized
        assert "ObjectiveLensPower" not in serialized
        assert "ObjectiveLensNumericalAperture" not in serialized
        assert "LightPathFilterTypeStackCodeSequence" not in serialized
        assert "LightPathFilterPassThroughWavelength" not in serialized
        assert "LightPathFilterPassBand" not in serialized
        assert "ImagePathFilterTypeStackCodeSequence" not in serialized
        assert "ImagePathFilterPassThroughWavelength" not in serialized
        assert "ImagePathFilterPassBand" not in serialized
        assert "ICCProfile" not in serialized

    @pytest.mark.parametrize(
        "illumination", [IlluminationColorCode("Full Spectrum"), 400.0, None]
    )
    @pytest.mark.parametrize(
        ["light_path_filter", "image_path_filter", "objectives"],
        [
            [
                LightPathFilter(
                    [LightPathFilterCode("Green optical filter")],
                    500,
                    400,
                    600,
                ),
                ImagePathFilter(
                    [ImagePathFilterCode("Red optical filter")],
                    500,
                    400,
                    600,
                ),
                Objectives(
                    [LenseCode("High power non-immersion lens")], 10.0, 20.0, 0.5
                ),
            ],
            [None, None, None],
        ],
    )
    def test_deserialize_optical_path(
        self, dicom_optical_path: Dataset, optical_path: OpticalPath
    ):
        # Arrange
        schema = OpticalPathDicomSchema()

        # Act
        deserialized = schema.load(dicom_optical_path)

        # Assert
        assert deserialized == optical_path

    def test_serialize_patient(self, patient: Patient):
        # Arrange
        schema = PatientDicomSchema()

        # Act
        serialized = schema.dump(patient)

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_patient_equals_patient(serialized, patient)

    def test_serialize_default_patient(self):
        # Arrange
        patient = Patient()
        schema = PatientDicomSchema()

        # Act
        serialized = schema.dump(patient)
        assert isinstance(serialized, Dataset)
        assert serialized.PatientName is None
        assert serialized.PatientID is None
        assert serialized.PatientBirthDate is None
        assert serialized.PatientSex is None
        assert "PatientSpeciesDescription" not in serialized
        assert "PatientSpeciesCodeSequence" not in serialized
        assert "PatientIdentityRemoved" not in serialized
        assert "DeidentificationMethod" not in serialized
        assert "DeidentificationMethodCodeSequence" not in serialized

    def test_deserialize_patient(self, dicom_patient: Dataset, patient: Patient):
        # Arrange

        schema = PatientDicomSchema()

        # Act
        deserialized = schema.load(dicom_patient)

        # Assert
        assert isinstance(deserialized, Patient)
        assert deserialized == patient

    def test_serialize_series(self, series: Series):
        # Arrange
        schema = SeriesDicomSchema()

        # Act
        serialized = schema.dump(series)

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_series_equals_series(serialized, series)

    def test_serialize_default_series(self):
        # Arrange
        series = Series()
        schema = SeriesDicomSchema()

        # Act
        serialized = schema.dump(series)

        # Assert
        assert isinstance(serialized, Dataset)
        assert serialized.SeriesInstanceUID == series.default_uid
        assert serialized.SeriesNumber == 1

    def test_deserialize_series(self, dicom_series: Dataset, series: Series):
        # Arrange

        schema = SeriesDicomSchema()

        # Act
        deserialized = schema.load(dicom_series)

        # Assert
        assert isinstance(deserialized, Series)
        assert deserialized == series

    def test_serialize_study(self, study: Study):
        # Arrange
        schema = StudyDicomSchema()

        # Act
        serialized = schema.dump(study)

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_study_equals_study(serialized, study)

    def test_serialize_default_study(self):
        # Arrange
        study = Study()
        schema = StudyDicomSchema()

        # Act
        serialized = schema.dump(study)

        # Assert
        assert isinstance(serialized, Dataset)
        assert serialized.StudyInstanceUID == study.default_uid
        assert serialized.StudyID is None
        assert serialized.StudyDate is None
        assert serialized.StudyTime is None
        assert serialized.AccessionNumber is None
        assert serialized.ReferringPhysicianName is None
        assert "StudyDescription" not in serialized

    def test_deserialize_study(self, dicom_study: Dataset, study: Study):
        # Arrange
        schema = StudyDicomSchema()

        # Act
        deserialized = schema.load(dicom_study)

        # Assert
        assert isinstance(deserialized, Study)
        assert deserialized == study

    @pytest.mark.parametrize(
        "slide_identifier",
        [
            None,
            "identifier",
            SpecimenIdentifier("identifier"),
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
            SpecimenIdentifier(
                "identifier",
                UniversalIssuerOfIdentifier("issuer", UniversalIssuerType.UUID),
            ),
        ],
    )
    def test_serialize_slide(self, slide: Slide):
        # Arrange
        schema = SlideDicomSchema()
        if slide.samples is not None:
            expected_samples = slide.samples
        else:
            expected_samples = [SlideSample(identifier=defaults.string)]

        # Act
        serialized = schema.dump(slide)

        # Assert
        assert isinstance(serialized, Dataset)
        if slide.identifier is None:
            assert serialized.ContainerIdentifier == defaults.string
        elif isinstance(slide.identifier, str):
            assert serialized.ContainerIdentifier == slide.identifier
        else:
            assert serialized.ContainerIdentifier == slide.identifier.value
            if slide.identifier.issuer is not None:
                assert "IssuerOfTheContainerIdentifierSequence" in serialized
                assert len(serialized.IssuerOfTheContainerIdentifierSequence) == 1
                assert_dicom_issuer_of_identifier_equals_issuer_of_identifier(
                    serialized.IssuerOfTheContainerIdentifierSequence[0],
                    slide.identifier.issuer,
                )
        assert_dicom_code_dataset_equals_code(
            serialized.ContainerTypeCodeSequence[0], defaults.slide_container_type
        )
        assert len(serialized.SpecimenDescriptionSequence) == len(expected_samples)
        for specimen_description, sample in zip(
            serialized.SpecimenDescriptionSequence, expected_samples
        ):
            assert specimen_description.SpecimenIdentifier == sample.identifier
            assert specimen_description.SpecimenUID == sample.uid

    def test_serialize_default_slide(self):
        # Arrange
        slide = Slide()
        schema = SlideDicomSchema()
        expected_samples = [SlideSample(identifier=defaults.string)]

        # Act
        serialized = schema.dump(slide)

        # Assert
        assert isinstance(serialized, Dataset)
        assert serialized.ContainerIdentifier == defaults.string
        assert_dicom_code_dataset_equals_code(
            serialized.ContainerTypeCodeSequence[0], defaults.slide_container_type
        )
        assert len(serialized.SpecimenDescriptionSequence) == len(expected_samples)
        for specimen_description, sample in zip(
            serialized.SpecimenDescriptionSequence, expected_samples
        ):
            assert specimen_description.SpecimenIdentifier == sample.identifier

    @pytest.mark.parametrize(
        "slide_identifier",
        [
            None,
            "identifier",
            SpecimenIdentifier("identifier"),
            SpecimenIdentifier("identifier", LocalIssuerOfIdentifier("issuer")),
            SpecimenIdentifier(
                "identifier",
                UniversalIssuerOfIdentifier("issuer", UniversalIssuerType.UUID),
            ),
        ],
    )
    def test_deserialize_slide(self, slide: Slide):
        # Arrange
        schema = SlideDicomSchema()
        serialized = schema.dump(slide)

        # Act
        deserialized = schema.load(serialized)

        # Assert
        assert isinstance(deserialized, Slide)
        # assert deserialized == slide

    @pytest.mark.parametrize(
        "illumination", [IlluminationColorCode("Full Spectrum"), 400.0]
    )
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
                None,
                None,
                None,
            ],
            [
                datetime(2023, 8, 5, 12, 13, 14, 150),
                FocusMethod.MANUAL,
                ExtendedDepthOfField(15, 0.5),
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0),
                None,
                None,
                None,
            ],
        ],
    )
    @pytest.mark.parametrize(
        "image_type", [ImageType.LABEL, ImageType.OVERVIEW, ImageType.VOLUME]
    )
    def test_serialize_wsi_metadata(
        self,
        wsi_metadata: WsiMetadata,
        equipment: Equipment,
        pyramid: Pyramid,
        label: Label,
        overview: Overview,
        study: Study,
        series: Series,
        patient: Patient,
        image_type: ImageType,
    ):
        # Arrange
        schema = WsiMetadataDicomSchema()

        # Act
        serialized = schema.dump(
            wsi_metadata, image_type=image_type, require_icc_profile=False
        )

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_equipment_equals_equipment(serialized, equipment)
        assert_dicom_series_equals_series(serialized, series)
        assert_dicom_study_equals_study(serialized, study)
        if image_type == ImageType.VOLUME:
            assert_dicom_image_equals_image(serialized, pyramid.image)
        elif image_type == ImageType.OVERVIEW:
            assert_dicom_image_equals_image(serialized, overview.image)
            assert_dicom_overview_equals_overview(serialized, overview)
        elif image_type == ImageType.LABEL:
            if label.image is None:
                assert_dicom_image_equals_image(
                    serialized,
                    Image(
                        acquisition_datetime=defaults.date_time,
                        focus_method=defaults.focus_method,
                    ),
                )
            else:
                assert_dicom_image_equals_image(serialized, label.image)
            if label.optical_paths is None:
                assert len(serialized.OpticalPathSequence) == 1
                assert_dicom_optical_path_equals_optical_path(
                    serialized.OpticalPathSequence[0],
                    OpticalPath(
                        identifier=defaults.optical_path_identifier,
                        illumination_types=[defaults.illumination_type],
                    ),
                )
            else:
                assert len(serialized.OpticalPathSequence) == len(label.optical_paths)
                for dicom_optical_path, optical_path in zip(
                    serialized.OpticalPathSequence, label.optical_paths
                ):
                    assert_dicom_optical_path_equals_optical_path(
                        dicom_optical_path,
                        optical_path,
                    )
        assert_dicom_label_equals_label(serialized, label)
        assert_dicom_patient_equals_patient(serialized, patient)

    def test_serialize_wsi_metadata_without_uids(
        self,
        equipment: Equipment,
        pyramid: Pyramid,
        label: Label,
        overview: Overview,
        study: Study,
        series: Series,
        patient: Patient,
        image_type: ImageType,
    ):
        # Arrange
        slide = Slide(stainings=[], samples=[])
        wsi_metadata = WsiMetadata(
            study=study,
            series=series,
            patient=patient,
            equipment=equipment,
            slide=slide,
            pyramid=pyramid,
            label=label,
            overview=overview,
        )
        schema = WsiMetadataDicomSchema()

        # Act
        serialized = schema.dump(
            wsi_metadata, image_type=image_type, require_icc_profile=False
        )

        # Assert
        assert "DimensionOrganizationSequence" in serialized
        assert len(serialized.DimensionOrganizationSequence) == 1
        assert (
            serialized.DimensionOrganizationSequence[0].DimensionOrganizationUID
            == wsi_metadata.default_dimension_organization_uids[0]
        )

    def test_deserialize_wsi_metadata_from_multiple_datasets(
        self,
        wsi_metadata: WsiMetadata,
        equipment: Equipment,
        pyramid: Pyramid,
        label: Label,
        overview: Overview,
        study: Study,
        series: Series,
        patient: Patient,
        pyramid_dataset: Dataset,
        dicom_label_label: Dataset,
        dicom_overview: Dataset,
    ):
        # Arrange
        schema = WsiMetadataDicomSchema()

        # Act
        deserialized = schema.load(pyramid_dataset, dicom_label_label, dicom_overview)

        # Assert
        assert isinstance(deserialized, WsiMetadata)
        assert deserialized.equipment == equipment
        assert deserialized.pyramid == pyramid
        assert deserialized.label == label
        assert deserialized.overview == overview
        assert deserialized.study == study
        assert deserialized.series == series
        assert deserialized.patient == patient
        assert (
            deserialized.frame_of_reference_uid
            == wsi_metadata.default_frame_of_reference_uid
        )
        assert (
            deserialized.dimension_organization_uids
            == wsi_metadata.default_dimension_organization_uids
        )

    def test_deserialize_wsi_metadata_from_empty_dataset(
        self,
    ):
        # Arrange
        dataset = Dataset()
        schema = WsiMetadataDicomSchema()

        # Act
        deserialized = schema.load(dataset)

        # Assert
        assert isinstance(deserialized, WsiMetadata)

    @pytest.mark.parametrize("z_offset", [1.0, None])
    def test_serialize_image_coordinate_system(
        self,
        z_offset: Optional[float],
    ):
        # Arrange
        image_coordinate_system = ImageCoordinateSystem(
            PointMm(20.0, 30.0), 90, z_offset
        )

        schema = ImageCoordinateSystemDicomSchema()
        # Act
        serialized = schema.dump(image_coordinate_system)

        # Assert
        assert isinstance(serialized, Dataset)
        assert (
            "TotalPixelMatrixOriginSequence" in serialized
            and len(serialized.TotalPixelMatrixOriginSequence) == 1
        )
        total_pixel_matrix_origin = serialized.TotalPixelMatrixOriginSequence[0]

        assert (
            total_pixel_matrix_origin.XOffsetInSlideCoordinateSystem
            == image_coordinate_system.origin.x
        )
        assert (
            total_pixel_matrix_origin.YOffsetInSlideCoordinateSystem
            == image_coordinate_system.origin.y
        )
        if z_offset is not None:
            assert total_pixel_matrix_origin.ZOffsetInSlideCoordinateSystem == z_offset
        assert "ImageOrientationSlide" in serialized
        assert len(serialized.ImageOrientationSlide) == 6
        assert serialized.ImageOrientationSlide == list(
            image_coordinate_system.orientation.values
        )

    @pytest.mark.parametrize("origin", [PointMm(20.0, 30.0), None])
    @pytest.mark.parametrize("orientation", [Orientation.from_rotation(90), None])
    @pytest.mark.parametrize("z_offset", [1.0, None])
    def test_deserialize_image_coordinate_system(
        self,
        origin: Optional[PointMm],
        orientation: Optional[Orientation],
        z_offset: Optional[float],
    ):
        # Arrange
        dataset = Dataset()
        origin_dataset = Dataset()
        if origin is not None:
            origin_dataset.XOffsetInSlideCoordinateSystem = origin.x
            origin_dataset.YOffsetInSlideCoordinateSystem = origin.y
        if z_offset is not None:
            origin_dataset.ZOffsetInSlideCoordinateSystem = z_offset
        if len(origin_dataset) > 0:
            dataset.TotalPixelMatrixOriginSequence = [origin_dataset]
        if orientation is not None:
            dataset.ImageOrientationSlide = list(orientation.values)

        schema = ImageCoordinateSystemDicomSchema()

        # Act
        deserialized = schema.load(dataset)

        # Assert
        if origin is not None and orientation is not None:
            assert isinstance(deserialized, ImageCoordinateSystem)
            assert deserialized.origin == origin
            assert deserialized.rotation == orientation.rotation
            assert deserialized.z_offset == z_offset
        else:
            assert deserialized is None

    @pytest.mark.parametrize(
        ["image_type", "default_rotation", "expected_origin"],
        [
            (ImageType.VOLUME, 0, PointMm(0, 0)),
            (ImageType.VOLUME, 90, PointMm(25, 0)),
            (ImageType.VOLUME, 180, PointMm(25, 50)),
            (ImageType.VOLUME, 270, PointMm(0, 50)),
            (ImageType.THUMBNAIL, 0, PointMm(0, 0)),
            (ImageType.THUMBNAIL, 90, PointMm(25, 0)),
            (ImageType.THUMBNAIL, 180, PointMm(25, 50)),
            (ImageType.THUMBNAIL, 270, PointMm(0, 50)),
            (ImageType.OVERVIEW, 0, PointMm(0, 0)),
            (ImageType.OVERVIEW, 90, PointMm(25, 0)),
            (ImageType.OVERVIEW, 180, PointMm(25, 75)),
            (ImageType.OVERVIEW, 270, PointMm(0, 75)),
            (ImageType.LABEL, 0, PointMm(0, 50)),
            (ImageType.LABEL, 90, PointMm(25, 50)),
            (ImageType.LABEL, 180, PointMm(25, 75)),
            (ImageType.LABEL, 270, PointMm(0, 75)),
        ],
    )
    def test_insert_default_image_coordinate_system(
        self, image_type: ImageType, default_rotation: int, expected_origin: PointMm
    ):
        # Arrange
        image = Image()
        defaults.image_coordinate_system_rotation = default_rotation

        # Act
        result = WsiMetadataDicomSchema._insert_default_image_coordinate_system(
            image, image_type
        )

        # Assert
        assert result.image_coordinate_system is not None
        assert result.image_coordinate_system.origin == expected_origin
        assert result.image_coordinate_system.rotation == default_rotation
