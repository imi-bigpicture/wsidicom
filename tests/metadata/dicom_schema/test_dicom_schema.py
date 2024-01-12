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
    assert_dicom_patient_equals_patient,
    assert_dicom_series_equals_series,
    assert_dicom_study_equals_study,
)
from wsidicom.codec import LossyCompressionIsoStandard
from wsidicom.conceptcode import (
    IlluminationColorCode,
    ImagePathFilterCode,
    LenseCode,
    LightPathFilterCode,
)
from wsidicom.geometry import PointMm, SizeMm
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
    Objectives,
    OpticalPath,
    Patient,
    Series,
    Slide,
    Study,
    WsiMetadata,
)
from wsidicom.metadata.image import LossyCompression
from wsidicom.metadata.sample import (
    LocalIssuerOfIdentifier,
    SlideSample,
    SpecimenIdentifier,
    UniversalIssuerOfIdentifier,
    UniversalIssuerType,
)
from wsidicom.metadata.schema.dicom.defaults import Defaults
from wsidicom.metadata.schema.dicom.equipment import EquipmentDicomSchema
from wsidicom.metadata.schema.dicom.image import ImageDicomSchema
from wsidicom.metadata.schema.dicom.label import LabelDicomSchema
from wsidicom.metadata.schema.dicom.optical_path import (
    OpticalPathDicomSchema,
)
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
        assert serialized.Manufacturer == Defaults.string
        assert serialized.ManufacturerModelName == Defaults.string
        assert serialized.DeviceSerialNumber == Defaults.string
        assert serialized.SoftwareVersions == Defaults.string

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
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0),
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
        assert serialized.AcquisitionDateTime == Defaults.date_time
        assert serialized.FocusMethod == Defaults.focus_method.name
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
                ImageCoordinateSystem(PointMm(50.0, 20.0), 180.0),
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
    def test_deserialize_image(self, dicom_image: Dataset, image: Image):
        # Arrange
        schema = ImageDicomSchema()

        # Act
        deserialized = schema.load(dicom_image)
        assert isinstance(deserialized, Image)
        assert deserialized == image

    @pytest.mark.parametrize(
        "image_type", [ImageType.LABEL, ImageType.OVERVIEW, ImageType.VOLUME]
    )
    def test_serialize_label(self, label: Label, image_type: ImageType):
        # Arrange
        schema = LabelDicomSchema(context={"image_type": image_type})

        # Act
        serialized = schema.dump(label)

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_label_equals_label(serialized, label, image_type)

    @pytest.mark.parametrize(
        "image_type", [ImageType.LABEL, ImageType.OVERVIEW, ImageType.VOLUME]
    )
    def test_serialize_default_label(self, image_type: ImageType):
        # Arrange
        label = Label()
        schema = LabelDicomSchema(context={"image_type": image_type})

        # Act
        serialized = schema.dump(label)

        # Assert
        assert isinstance(serialized, Dataset)
        if image_type == ImageType.LABEL:
            assert serialized.LabelText is None
            assert serialized.BarcodeValue is None
            assert serialized.SpecimenLabelInImage == "YES"
            assert serialized.BurnedInAnnotation == "YES"
        else:
            assert "LabelText" not in serialized
            assert "BarcodeValue" not in serialized
            assert serialized.SpecimenLabelInImage == "NO"
            assert serialized.BurnedInAnnotation == "NO"

    @pytest.mark.parametrize(
        "image_type", [ImageType.LABEL, ImageType.OVERVIEW, ImageType.VOLUME]
    )
    def test_deserialize_label(
        self, dicom_label: Dataset, label: Label, image_type: ImageType
    ):
        # Arrange
        schema = LabelDicomSchema()

        # Act
        deserialized = schema.load(dicom_label)

        # Assert
        assert isinstance(deserialized, Label)
        if image_type == ImageType.LABEL:
            assert deserialized.text == label.text
            assert deserialized.barcode == label.barcode
            assert deserialized.label_is_phi == label.label_is_phi
        elif image_type == ImageType.VOLUME:
            assert deserialized.text is None
            assert deserialized.barcode is None
            assert deserialized.label_in_volume_image == label.label_in_volume_image
            assert deserialized.label_is_phi == label.label_is_phi
        elif image_type == ImageType.OVERVIEW:
            assert deserialized.text is None
            assert deserialized.barcode is None
            assert deserialized.label_in_overview_image == label.label_in_overview_image
            assert deserialized.label_is_phi == label.label_is_phi

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
            serialized.IlluminationTypeCodeSequence, [Defaults.illumination_type]
        )
        assert "OpticalPathDescription" not in serialized
        assert "IlluminationWaveLength" not in serialized
        assert_dicom_code_dataset_equals_code(
            serialized.IlluminationColorCodeSequence[0], Defaults.illumination
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
            expected_samples = [SlideSample(identifier=Defaults.string)]

        # Act
        serialized = schema.dump(slide)

        # Assert
        assert isinstance(serialized, Dataset)
        if slide.identifier is None:
            assert serialized.ContainerIdentifier == Defaults.string
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
            serialized.ContainerTypeCodeSequence[0], Defaults.slide_container_type
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
        expected_samples = [SlideSample(identifier=Defaults.string)]

        # Act
        serialized = schema.dump(slide)

        # Assert
        assert isinstance(serialized, Dataset)
        assert serialized.ContainerIdentifier == Defaults.string
        assert_dicom_code_dataset_equals_code(
            serialized.ContainerTypeCodeSequence[0], Defaults.slide_container_type
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
        image: Image,
        label: Label,
        optical_path: OpticalPath,
        study: Study,
        series: Series,
        patient: Patient,
        image_type: ImageType,
    ):
        # Arrange
        schema = WsiMetadataDicomSchema(context={"image_type": image_type})

        # Act
        serialized = schema.dump(wsi_metadata)

        # Assert
        assert isinstance(serialized, Dataset)
        assert_dicom_equipment_equals_equipment(serialized, equipment)
        assert_dicom_series_equals_series(serialized, series)
        assert_dicom_study_equals_study(serialized, study)
        assert_dicom_image_equals_image(serialized, image)
        assert_dicom_label_equals_label(serialized, label, image_type)
        assert_dicom_optical_path_equals_optical_path(
            serialized.OpticalPathSequence[0], optical_path
        )
        assert_dicom_patient_equals_patient(serialized, patient)

    def test_serialize_wsi_metadata_with_empty_lists(
        self,
        equipment: Equipment,
        image: Image,
        label: Label,
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
            optical_paths=[],
            slide=slide,
            label=label,
            image=image,
        )
        schema = WsiMetadataDicomSchema(context={"image_type": image_type})

        # Act
        serialized = schema.dump(wsi_metadata)

        # Assert
        assert_dicom_optical_path_equals_optical_path(
            serialized.OpticalPathSequence[0], OpticalPath()
        )
        assert "DimensionOrganizationSequence" in serialized
        assert len(serialized.DimensionOrganizationSequence) == 1
        assert (
            serialized.DimensionOrganizationSequence[0].DimensionOrganizationUID
            == wsi_metadata.default_dimension_organization_uids[0]
        )

    def test_deserialize_wsi_metadata(
        self,
        wsi_metadata: WsiMetadata,
        equipment: Equipment,
        image: Image,
        label: Label,
        optical_path: OpticalPath,
        study: Study,
        series: Series,
        patient: Patient,
        image_type: ImageType,
        dicom_wsi_metadata: Dataset,
    ):
        # Arrange
        schema = WsiMetadataDicomSchema()

        # Act
        deserialized = schema.load(dicom_wsi_metadata)

        # Assert
        assert isinstance(deserialized, WsiMetadata)
        assert deserialized.equipment == equipment
        assert deserialized.image == image
        assert deserialized.optical_paths[0] == optical_path
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

    def test_deserialize_wsi_metadata_from_multiple_datasets(
        self,
        wsi_metadata: WsiMetadata,
        equipment: Equipment,
        image: Image,
        label: Label,
        optical_path: OpticalPath,
        study: Study,
        series: Series,
        patient: Patient,
        image_type: ImageType,
        dicom_wsi_metadata: Dataset,
        dicom_label_label: Dataset,
        dicom_overview_label: Dataset,
    ):
        # Arrange
        schema = WsiMetadataDicomSchema()
        datasets = [dicom_wsi_metadata, dicom_label_label, dicom_overview_label]

        # Act
        deserialized = schema.from_datasets(datasets)

        # Assert
        assert isinstance(deserialized, WsiMetadata)
        assert deserialized.equipment == equipment
        assert deserialized.image == image
        assert deserialized.optical_paths[0] == optical_path
        assert deserialized.study == study
        assert deserialized.series == series
        assert deserialized.patient == patient
        assert deserialized.label == label

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
