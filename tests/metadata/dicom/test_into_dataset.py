from pydicom import Dataset
import pytest
from tests.metadata.helpers import bool_to_dicom_literal
from wsidicomizer.metadata.equipment import Equipment
from wsidicomizer.metadata.image import Image, ImageType
from wsidicomizer.metadata.label import Label
from wsidicomizer.metadata.optical_path import OpticalPath
from wsidicom.conceptcode import IlluminationColorCode
from wsidicomizer.metadata.patient import Patient
from pydicom.sr.coding import Code
from wsidicomizer.metadata.series import Series
from wsidicomizer.metadata.study import Study


class TestIntoDataset:
    def test_equipment(self, equipment: Equipment):
        # Arrange
        dataset = Dataset()

        # Act
        equipment.insert_into_dataset(dataset, ImageType.VOLUME)

        # Assert
        assert dataset.Manufacturer == equipment.manufacturer
        assert dataset.ManufacturerModelName == equipment.model_name
        assert dataset.DeviceSerialNumber == equipment.device_serial_number
        assert dataset.SoftwareVersions == equipment.software_versions

    def test_image(self, image: Image):
        # Arrange
        dataset = Dataset()

        # Act
        image.insert_into_dataset(dataset, ImageType.VOLUME)

        # Assert
        assert image.focus_method is not None
        assert image.extended_depth_of_field is not None
        assert image.image_coordinate_system is not None
        assert dataset.AcquisitionDateTime == image.acquisition_datetime
        assert dataset.FocusMethod == image.focus_method.name
        assert dataset.ExtendedDepthOfField == "YES"
        assert (
            dataset.NumberOfFocalPlanes
            == image.extended_depth_of_field.number_of_focal_planes
        )
        assert (
            dataset.DistanceBetweenFocalPlanes
            == image.extended_depth_of_field.distance_between_focal_planes
        )
        assert (
            dataset.TotalPixelMatrixOriginSequence[0].XOffsetInSlideCoordinateSystem
            == image.image_coordinate_system.origin.x
        )
        assert (
            dataset.TotalPixelMatrixOriginSequence[0].YOffsetInSlideCoordinateSystem
            == image.image_coordinate_system.origin.y
        )
        assert dataset.ImageOrientationSlide == [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    @pytest.mark.parametrize(
        "image_type", [ImageType.LABEL, ImageType.OVERVIEW, ImageType.VOLUME]
    )
    def test_label(self, label: Label, image_type: ImageType):
        # Arrange
        dataset = Dataset()

        # Act
        label.insert_into_dataset(dataset, image_type)

        # Assert
        if image_type == ImageType.OVERVIEW:
            if label.label_in_overview_image:
                assert dataset.BurnedInAnnotation == bool_to_dicom_literal(
                    label.label_is_phi
                )
            assert dataset.SpecimenLabelInImage == bool_to_dicom_literal(
                label.label_in_overview_image
            )
        elif image_type == ImageType.VOLUME:
            if label.label_in_overview_image:
                assert dataset.BurnedInAnnotation == bool_to_dicom_literal(
                    label.label_is_phi
                )
            assert dataset.SpecimenLabelInImage == bool_to_dicom_literal(
                label.label_in_volume_image
            )
        else:
            assert dataset.BurnedInAnnotation == bool_to_dicom_literal(
                label.label_is_phi
            )
            assert dataset.SpecimenLabelInImage == "YES"
        assert dataset.LabelText == label.label_text
        assert dataset.BarcodeValue == label.barcode_value

    def test_optical_path(self, optical_path: OpticalPath):
        # Arrange
        dataset = Dataset()

        # Act
        optical_path.insert_into_dataset(dataset, ImageType.VOLUME)

        # Assert
        optical_path_dataset = dataset.OpticalPathSequence[0]
        assert optical_path.illumination_type is not None
        assert (
            optical_path_dataset.IlluminationTypeCodeSequence[0].CodeValue
            == optical_path.illumination_type.value
        )
        assert (
            optical_path_dataset.IlluminationTypeCodeSequence[0].CodingSchemeDesignator
            == optical_path.illumination_type.scheme_designator
        )
        assert (
            optical_path_dataset.IlluminationTypeCodeSequence[0].CodeMeaning
            == optical_path.illumination_type.meaning
        )
        assert optical_path.objective is not None
        if isinstance(optical_path.illumination, float):
            assert (
                optical_path_dataset.IlluminationWaveLength == optical_path.illumination
            )
        elif isinstance(optical_path.illumination, IlluminationColorCode):
            assert (
                optical_path_dataset.IlluminationColorCodeSequence[0].CodeValue
                == optical_path.illumination.value
            )
            assert (
                optical_path_dataset.IlluminationColorCodeSequence[
                    0
                ].CodingSchemeDesignator
                == optical_path.illumination.scheme_designator
            )
            assert (
                optical_path_dataset.IlluminationColorCodeSequence[0].CodeMeaning
                == optical_path.illumination.meaning
            )
        assert optical_path_dataset.OpticalPathIdentifier == optical_path.identifier
        assert optical_path_dataset.OpticalPathDescription == optical_path.description
        assert (
            optical_path_dataset.CondenserLensPower
            == optical_path.objective.condenser_power
        )
        assert (
            optical_path_dataset.ObjectiveLensPower
            == optical_path.objective.objective_power
        )
        assert (
            optical_path_dataset.ObjectiveLensNumericalAperture
            == optical_path.objective.objective_numerical_aperature
        )

    def test_patient(self, patient: Patient):
        # Arrange
        dataset = Dataset()

        # Act
        patient.insert_into_dataset(dataset, ImageType.VOLUME)

        # Assert
        assert patient.sex is not None
        assert patient.de_identification is not None
        assert patient.de_identification.methods is not None
        assert dataset.PatientName == patient.name
        assert dataset.PatientID == patient.identifier
        assert dataset.PatientBirthDate == patient.birth_date
        assert dataset.PatientSex == patient.sex.name
        if isinstance(patient.species_description, Code):
            assert (
                dataset.PatientSpeciesCodeSequence[0].CodeValue
                == patient.species_description.value
            )
            assert (
                dataset.PatientSpeciesCodeSequence[0].CodingSchemeDesignator
                == patient.species_description.scheme_designator
            )
            assert (
                dataset.PatientSpeciesCodeSequence[0].CodeMeaning
                == patient.species_description.meaning
            )
        else:
            assert dataset.PatientSpeciesDescription == patient.species_description
        assert dataset.PatientIdentityRemoved == bool_to_dicom_literal(
            patient.de_identification.identity_removed
        )
        for index, method in enumerate(patient.de_identification.methods):
            if isinstance(method, Code):
                dataset_method = dataset.DeidentificationMethodCodeSequence[index]
                assert dataset_method.CodeValue == method.value
                assert dataset_method.CodingSchemeDesignator == method.scheme_designator
                assert dataset_method.CodeMeaning == method.meaning
            else:
                dataset_method = dataset.DeidentificationMethod
                assert dataset_method == method

    def test_series(self, series: Series):
        # Arrange
        dataset = Dataset()

        # Act
        series.insert_into_dataset(dataset, ImageType.VOLUME)

        # Assert
        assert dataset.SeriesInstanceUID == series.uid
        assert dataset.SeriesNumber == series.number

    def test_study(self, study: Study):
        # Arrange
        dataset = Dataset()

        # Act
        study.insert_into_dataset(dataset, ImageType.VOLUME)

        # Assert
        assert dataset.StudyInstanceUID == study.uid
        assert dataset.StudyID == study.identifier
        assert dataset.StudyDate == study.date
        assert dataset.StudyTime == study.time
        assert dataset.AccessionNumber == study.accession_number
        assert dataset.ReferringPhysicianName == study.referring_physician_name
