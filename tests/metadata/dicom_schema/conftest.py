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


import pytest
from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.valuerep import DT, DSfloat

from tests.metadata.dicom_schema.helpers import (
    bool_to_dicom_literal,
    code_to_code_dataset,
)
from wsidicom.conceptcode import IlluminationColorCode
from wsidicom.instance import ImageType
from wsidicom.metadata import (
    Equipment,
    Image,
    Label,
    OpticalPath,
    Patient,
    Series,
    Slide,
    Study,
    WsiMetadata,
)
from wsidicom.metadata.schema.dicom.optical_path import LutDicomFormatter
from wsidicom.metadata.schema.dicom.slide import SlideDicomSchema


@pytest.fixture()
def valid_dicom():
    yield True


@pytest.fixture()
def dicom_equipment(equipment: Equipment):
    dataset = Dataset()
    dataset.Manufacturer = equipment.manufacturer
    dataset.ManufacturerModelName = equipment.model_name
    dataset.DeviceSerialNumber = equipment.device_serial_number
    dataset.SoftwareVersions = equipment.software_versions
    yield dataset


@pytest.fixture()
def dicom_image(image: Image, valid_dicom: bool):
    dataset = Dataset()
    dataset.AcquisitionDateTime = DT(image.acquisition_datetime)
    if image.focus_method is not None:
        dataset.FocusMethod = image.focus_method.name
    if image.image_coordinate_system is not None:
        dataset.ImageOrientationSlide = list(
            image.image_coordinate_system.orientation.values
        )
        origin = Dataset()
        origin.XOffsetInSlideCoordinateSystem = image.image_coordinate_system.origin.x
        origin.YOffsetInSlideCoordinateSystem = image.image_coordinate_system.origin.y
        if image.image_coordinate_system.z_offset is not None:
            origin.ZOffsetInSlideCoordinateSystem = (
                image.image_coordinate_system.z_offset
            )

        dataset.TotalPixelMatrixOriginSequence = [origin]
    elif not valid_dicom:
        dataset.ImageOrientationSlide = []
        empty_origin = Dataset()
        dataset.TotalPixelMatrixOriginSequence = [empty_origin]

    if image.extended_depth_of_field is not None:
        dataset.ExtendedDepthOfField = "YES"
        dataset.NumberOfFocalPlanes = (
            image.extended_depth_of_field.number_of_focal_planes
        )
        dataset.DistanceBetweenFocalPlanes = (
            image.extended_depth_of_field.distance_between_focal_planes
        )
    elif valid_dicom:
        dataset.ExtendedDepthOfField = "NO"
    if any(
        item is not None
        for item in [
            image.pixel_spacing,
            image.focal_plane_spacing,
            image.depth_of_field,
        ]
    ):
        pixel_spacing_dataset = Dataset()
        if image.pixel_spacing is not None:
            pixel_spacing_dataset.PixelSpacing = [
                DSfloat(value) for value in image.pixel_spacing.to_tuple()
            ]
        if image.focal_plane_spacing is not None:
            pixel_spacing_dataset.SpacingBetweenSlices = image.focal_plane_spacing
        if image.depth_of_field is not None:
            pixel_spacing_dataset.SliceThickness = image.depth_of_field
        shared_functional_group_dataset = Dataset()
        shared_functional_group_dataset.PixelMeasuresSequence = [pixel_spacing_dataset]
        dataset.SharedFunctionalGroupsSequence = [shared_functional_group_dataset]

    if image.lossy_compressions is not None:
        dataset.LossyImageCompression = "01"
        dataset.LossyImageCompressionMethod = [
            compression.method.value for compression in image.lossy_compressions
        ]
        dataset.LossyImageCompressionRatio = [
            DSfloat(compression.ratio, True) for compression in image.lossy_compressions
        ]
    else:
        dataset.LossyImageCompression = "00"
    if image.comments is not None:
        dataset.ImageComments = image.comments
    yield dataset


@pytest.fixture()
def dicom_label(label: Label, image_type: ImageType):
    return create_dicom_label(label, image_type)


@pytest.fixture()
def dicom_label_label(label: Label):
    return create_dicom_label(label, ImageType.LABEL)


@pytest.fixture()
def dicom_overview_label(label: Label):
    return create_dicom_label(label, ImageType.OVERVIEW)


def create_dicom_label(label: Label, image_type: ImageType):
    dataset = Dataset()
    dataset.ImageType = ["ORIGINAL", "PRIMARY", image_type.value]
    if image_type == ImageType.LABEL:
        dataset.LabelText = label.text
        dataset.BarcodeValue = label.barcode
        dataset.SpecimenLabelInImage = bool_to_dicom_literal(True)
        dataset.BurnedInAnnotation = bool_to_dicom_literal(label.label_is_phi)
    elif image_type == ImageType.VOLUME:
        dataset.SpecimenLabelInImage = bool_to_dicom_literal(
            label.label_in_volume_image
        )
        dataset.BurnedInAnnotation = bool_to_dicom_literal(
            label.label_is_phi and label.label_in_volume_image
        )
    elif image_type == ImageType.OVERVIEW:
        dataset.SpecimenLabelInImage = bool_to_dicom_literal(
            label.label_in_overview_image
        )
        dataset.BurnedInAnnotation = bool_to_dicom_literal(
            label.label_is_phi and label.label_in_overview_image
        )
    else:
        raise ValueError(f"Unknown image type {image_type}.")
    return dataset


@pytest.fixture()
def dicom_optical_path(optical_path: OpticalPath):
    dataset = Dataset()
    dataset.OpticalPathIdentifier = optical_path.identifier
    dataset.OpticalPathDescription = optical_path.description
    if optical_path.illumination_types is not None:
        dataset.IlluminationTypeCodeSequence = [
            illumination_type.to_ds()
            for illumination_type in optical_path.illumination_types
        ]
    if isinstance(optical_path.illumination, (int, float)):
        dataset.IlluminationWaveLength = optical_path.illumination
    elif isinstance(optical_path.illumination, IlluminationColorCode):
        dataset.IlluminationColorCodeSequence = [optical_path.illumination.to_ds()]

    if optical_path.light_path_filter is not None:
        dataset.LightPathFilterPassThroughWavelength = (
            optical_path.light_path_filter.nominal
        )
        dataset.LightPathFilterPassBand = [
            optical_path.light_path_filter.low_pass,
            optical_path.light_path_filter.high_pass,
        ]
        dataset.LightPathFilterTypeStackCodeSequence = [
            filter.to_ds() for filter in optical_path.light_path_filter.filters
        ]
    if optical_path.image_path_filter is not None:
        dataset.ImagePathFilterPassThroughWavelength = (
            optical_path.image_path_filter.nominal
        )
        dataset.ImagePathFilterPassBand = [
            optical_path.image_path_filter.low_pass,
            optical_path.image_path_filter.high_pass,
        ]
        dataset.ImagePathFilterTypeStackCodeSequence = [
            filter.to_ds() for filter in optical_path.image_path_filter.filters
        ]
    if optical_path.objective is not None:
        dataset.LensesCodeSequence = [
            lense.to_ds() for lense in optical_path.objective.lenses
        ]

        dataset.CondenserLensPower = optical_path.objective.condenser_power
        dataset.ObjectiveLensPower = optical_path.objective.objective_power
        dataset.ObjectiveLensNumericalAperture = (
            optical_path.objective.objective_numerical_aperture
        )
    if optical_path.lut is not None:
        dataset.PaletteColorLookupTableSequence = LutDicomFormatter.to_dataset(
            optical_path.lut
        )
    if optical_path.icc_profile is not None:
        dataset.ICCProfile = optical_path.icc_profile

    yield dataset


@pytest.fixture()
def dicom_patient(patient: Patient):
    dataset = Dataset()
    dataset.PatientName = patient.name
    dataset.PatientID = patient.identifier
    dataset.PatientBirthDate = patient.birth_date
    if patient.sex is not None:
        dataset.PatientSex = patient.sex.name
    else:
        dataset.PatientSex = None
    if isinstance(patient.species_description, str):
        dataset.PatientSpeciesDescription = patient.species_description
    elif isinstance(patient.species_description, Code):
        dataset.PatientSpeciesCodeSequence = [
            code_to_code_dataset(patient.species_description)
        ]
    if patient.de_identification is not None:
        dataset.PatientIdentityRemoved = bool_to_dicom_literal(
            patient.de_identification.identity_removed
        )
        if patient.de_identification.methods is not None:
            dataset.DeidentificationMethod = [
                method
                for method in patient.de_identification.methods
                if isinstance(method, str)
            ]
            dataset.DeidentificationMethodCodeSequence = [
                code_to_code_dataset(method)
                for method in patient.de_identification.methods
                if isinstance(method, Code)
            ]
    yield dataset


@pytest.fixture()
def dicom_series(series: Series):
    dataset = Dataset()
    dataset.SeriesInstanceUID = series.uid
    dataset.SeriesNumber = series.number
    yield dataset


@pytest.fixture()
def dicom_study(study: Study):
    dataset = Dataset()
    dataset.StudyInstanceUID = study.uid
    dataset.StudyID = study.identifier
    dataset.StudyDate = study.date
    dataset.StudyTime = study.time
    dataset.AccessionNumber = study.accession_number
    dataset.ReferringPhysicianName = study.referring_physician_name
    if study.description is not None:
        dataset.StudyDescription = study.description
    yield dataset


@pytest.fixture()
def dicom_slide(slide: Slide):
    schema = SlideDicomSchema()
    yield schema.dump(slide)


@pytest.fixture()
def dicom_wsi_metadata(
    wsi_metadata: WsiMetadata,
    dicom_equipment: Dataset,
    dicom_image: Dataset,
    dicom_label: Dataset,
    dicom_slide: Dataset,
    dicom_optical_path: Dataset,
    dicom_study: Dataset,
    dicom_series: Dataset,
    dicom_patient: Dataset,
):
    dataset = Dataset()
    dataset.update(dicom_equipment)
    dataset.update(dicom_image)
    dataset.update(dicom_label)
    dataset.update(dicom_slide)
    dataset.update(dicom_study)
    dataset.update(dicom_series)
    dataset.update(dicom_patient)
    dataset.OpticalPathSequence = [dicom_optical_path]
    dimension_organization_sequence = []
    for uid in wsi_metadata.default_dimension_organization_uids:
        dimension_organization = Dataset()
        dimension_organization.DimensionOrganizationUID = uid
        dimension_organization_sequence.append(dimension_organization)
    dataset.DimensionOrganizationSequence = dimension_organization_sequence
    dataset.FrameOfReferenceUID = wsi_metadata.default_frame_of_reference_uid
    yield dataset
