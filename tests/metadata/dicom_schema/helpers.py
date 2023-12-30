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


from typing import Sequence, Union
from pydicom import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.sr.coding import Code


from wsidicom.conceptcode import (
    ConceptCode,
    IlluminationColorCode,
)
from wsidicom.instance import ImageType
from wsidicom.metadata import (
    Equipment,
    Image,
    Label,
    OpticalPath,
    Patient,
    Series,
    Study,
)
from wsidicom.metadata.dicom_schema.optical_path import LutDicomParser


def code_to_code_dataset(code: Code):
    dataset = Dataset()
    dataset.CodeValue = code.value
    dataset.CodingSchemeDesignator = code.scheme_designator
    dataset.CodeMeaning = code.meaning
    dataset.CodingSchemeVersion = code.scheme_version

    return dataset


def bool_to_dicom_literal(value: bool) -> str:
    if value:
        return "YES"
    return "NO"


def assert_dicom_bool_equals_bool(dicom_bool: str, expected_bool: bool):
    if expected_bool:
        assert dicom_bool == "YES"
    else:
        assert dicom_bool == "NO"


def assert_dicom_code_dataset_equals_code(
    code_dataset: Dataset, expected_code: Union[Code, ConceptCode]
):
    assert code_dataset.CodeValue == expected_code.value
    assert code_dataset.CodingSchemeDesignator == expected_code.scheme_designator
    assert code_dataset.CodeMeaning == expected_code.meaning
    assert code_dataset.CodingSchemeVersion == expected_code.scheme_version


def assert_dicom_code_sequence_equals_codes(
    code_sequence: Sequence[Dataset], expected_codes: Sequence[Union[Code, ConceptCode]]
):
    assert len(code_sequence) == len(expected_codes)
    for code_dataset, expected_code in zip(code_sequence, expected_codes):
        assert_dicom_code_dataset_equals_code(code_dataset, expected_code)


def assert_dicom_equipment_equals_equipment(
    dicom_equipment: Dataset, equipment: Equipment
):
    assert dicom_equipment.Manufacturer == equipment.manufacturer
    assert dicom_equipment.ManufacturerModelName == equipment.model_name
    assert dicom_equipment.DeviceSerialNumber == equipment.device_serial_number
    if equipment.software_versions is not None:
        if len(equipment.software_versions) == 1:
            assert dicom_equipment.SoftwareVersions == equipment.software_versions[0]
        else:
            assert dicom_equipment.SoftwareVersions == equipment.software_versions


def assert_dicom_image_equals_image(dicom_image: Dataset, image: Image):
    assert dicom_image.AcquisitionDateTime == image.acquisition_datetime
    if image.focus_method is not None:
        assert dicom_image.FocusMethod == image.focus_method.name
    if image.extended_depth_of_field is not None:
        assert dicom_image.ExtendedDepthOfField == bool_to_dicom_literal(True)
        assert (
            dicom_image.NumberOfFocalPlanes
            == image.extended_depth_of_field.number_of_focal_planes
        )
        assert (
            dicom_image.DistanceBetweenFocalPlanes
            == image.extended_depth_of_field.distance_between_focal_planes
        )
    else:
        assert dicom_image.ExtendedDepthOfField == bool_to_dicom_literal(False)
    if image.image_coordinate_system is not None:
        assert (
            dicom_image.TotalPixelMatrixOriginSequence[0].XOffsetInSlideCoordinateSystem
            == image.image_coordinate_system.origin.x
        )
        assert (
            dicom_image.TotalPixelMatrixOriginSequence[0].YOffsetInSlideCoordinateSystem
            == image.image_coordinate_system.origin.y
        )
        assert dicom_image.ImageOrientationSlide == list(
            image.image_coordinate_system.orientation.values
        )


def assert_dicom_label_equals_label(
    dicom_label: Dataset, label: Label, image_type: ImageType
):
    assert dicom_label.LabelText == label.text
    assert dicom_label.BarcodeValue == label.barcode
    if (
        (image_type == ImageType.VOLUME and label.label_in_volume_image)
        or (image_type == ImageType.OVERVIEW and label.label_in_overview_image)
        or image_type == ImageType.LABEL
    ):
        assert dicom_label.SpecimenLabelInImage == bool_to_dicom_literal(True)
        assert dicom_label.BurnedInAnnotation == bool_to_dicom_literal(
            label.label_is_phi
        )
    else:
        assert dicom_label.SpecimenLabelInImage == bool_to_dicom_literal(False)
        assert dicom_label.BurnedInAnnotation == bool_to_dicom_literal(False)


def assert_dicom_optical_path_equals_optical_path(
    dicom_optical_path: Dataset, optical_path: OpticalPath
):
    assert dicom_optical_path.OpticalPathIdentifier == optical_path.identifier
    if optical_path.description is not None:
        assert dicom_optical_path.OpticalPathDescription == optical_path.description
    if optical_path.illumination_types is not None:
        assert_dicom_code_dataset_equals_code(
            dicom_optical_path.IlluminationTypeCodeSequence[0],
            optical_path.illumination_types[0],
        )
    if isinstance(optical_path.illumination, float):
        assert dicom_optical_path.IlluminationWaveLength == optical_path.illumination
    elif isinstance(optical_path.illumination, IlluminationColorCode):
        assert_dicom_code_dataset_equals_code(
            dicom_optical_path.IlluminationColorCodeSequence[0],
            optical_path.illumination,
        )
    if optical_path.light_path_filter is not None:
        assert (
            dicom_optical_path.LightPathFilterPassThroughWavelength
            == optical_path.light_path_filter.nominal
        )
        assert dicom_optical_path.LightPathFilterPassBand == [
            optical_path.light_path_filter.low_pass,
            optical_path.light_path_filter.high_pass,
        ]
        assert_dicom_code_sequence_equals_codes(
            dicom_optical_path.LightPathFilterTypeStackCodeSequence,
            optical_path.light_path_filter.filters,
        )
    if optical_path.image_path_filter is not None:
        assert (
            dicom_optical_path.ImagePathFilterPassThroughWavelength
            == optical_path.image_path_filter.nominal
        )
        assert dicom_optical_path.ImagePathFilterPassBand == [
            optical_path.image_path_filter.low_pass,
            optical_path.image_path_filter.high_pass,
        ]
        assert_dicom_code_sequence_equals_codes(
            dicom_optical_path.ImagePathFilterTypeStackCodeSequence,
            optical_path.image_path_filter.filters,
        )
    if optical_path.objective is not None:
        assert_dicom_code_sequence_equals_codes(
            dicom_optical_path.LensesCodeSequence,
            optical_path.objective.lenses,
        )
        assert (
            dicom_optical_path.CondenserLensPower
            == optical_path.objective.condenser_power
        )
        assert (
            dicom_optical_path.ObjectiveLensPower
            == optical_path.objective.objective_power
        )
        assert (
            dicom_optical_path.ObjectiveLensNumericalAperture
            == optical_path.objective.objective_numerical_aperature
        )
    if optical_path.lut is not None:
        assert "PaletteColorLookupTableSequence" in dicom_optical_path
        assert isinstance(
            dicom_optical_path.PaletteColorLookupTableSequence, DicomSequence
        )
        assert len(dicom_optical_path.PaletteColorLookupTableSequence) == 1
        parsed_lut = LutDicomParser.from_dataset(
            dicom_optical_path.PaletteColorLookupTableSequence
        )
        assert parsed_lut == optical_path.lut
    if optical_path.icc_profile is not None:
        assert "ICCProfile" in dicom_optical_path
        assert isinstance(dicom_optical_path.ICCProfile, bytes)
        assert dicom_optical_path.ICCProfile == optical_path.icc_profile


def assert_dicom_patient_equals_patient(dicom_patient: Dataset, patient: Patient):
    assert dicom_patient.PatientName == patient.name
    assert dicom_patient.PatientID == patient.identifier
    assert dicom_patient.PatientBirthDate == patient.birth_date
    if patient.sex is not None:
        assert dicom_patient.PatientSex == patient.sex.name
    if isinstance(patient.species_description, str):
        assert dicom_patient.PatientSpeciesDescription == patient.species_description
    elif isinstance(patient.species_description, Code):
        assert_dicom_code_dataset_equals_code(
            dicom_patient.PatientSpeciesCodeSequence[0], patient.species_description
        )
    if patient.de_identification is not None:
        assert_dicom_bool_equals_bool(
            dicom_patient.PatientIdentityRemoved,
            patient.de_identification.identity_removed,
        )
        if patient.de_identification.methods is not None:
            string_methods = [
                method
                for method in patient.de_identification.methods
                if isinstance(method, str)
            ]
            if len(string_methods) == 1:
                assert dicom_patient.DeidentificationMethod == string_methods[0]
            elif len(string_methods) > 1:
                assert dicom_patient.DeidentificationMethod == string_methods[0]
            code_methods = [
                method
                for method in patient.de_identification.methods
                if isinstance(method, Code)
            ]
            if len(code_methods) > 1:
                assert_dicom_code_sequence_equals_codes(
                    dicom_patient.DeidentificationMethodCodeSequence, code_methods
                )


def assert_dicom_series_equals_series(dicom_series: Dataset, series: Series):
    assert dicom_series.SeriesInstanceUID == series.uid
    assert dicom_series.SeriesNumber == series.number


def assert_dicom_study_equals_study(dicom_study: Dataset, study: Study):
    assert dicom_study.StudyInstanceUID == study.uid
    assert dicom_study.StudyID == study.identifier
    assert dicom_study.StudyDate == study.date
    assert dicom_study.StudyTime == study.time
    assert dicom_study.AccessionNumber == study.accession_number
    assert dicom_study.ReferringPhysicianName == study.referring_physician_name
