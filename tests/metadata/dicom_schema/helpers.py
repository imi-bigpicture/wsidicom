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


import datetime
from typing import Iterator, List, Optional, Sequence, Union

from pydicom import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.sr.coding import Code
from pydicom.uid import UID
from pydicom.valuerep import DSfloat

from wsidicom.conceptcode import (
    AnatomicPathologySpecimenTypesCode,
    ConceptCode,
    ContainerTypeCode,
    IlluminationColorCode,
    SpecimenCollectionProcedureCode,
    SpecimenEmbeddingMediaCode,
    SpecimenFixativesCode,
    SpecimenSamplingProcedureCode,
    SpecimenStainsCode,
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
from wsidicom.metadata.sample import (
    Measurement,
    SamplingLocation,
    SpecimenIdentifier,
    SpecimenLocalization,
)
from wsidicom.metadata.schema.dicom.optical_path import LutDicomParser
from wsidicom.metadata.schema.dicom.sample.model import (
    CollectionDicomModel,
    ProcessingDicomModel,
    ReceivingDicomModel,
    SamplingDicomModel,
    SpecimenPreparationStepDicomModel,
    StainingDicomModel,
    StorageDicomModel,
)
from wsidicom.metadata.schema.dicom.sample.schema import SampleCodes


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
    if any(
        item is not None
        for item in [
            image.pixel_spacing,
            image.focal_plane_spacing,
            image.depth_of_field,
        ]
    ):
        assert "SharedFunctionalGroupsSequence" in dicom_image
        shared_functional_group = dicom_image.SharedFunctionalGroupsSequence[0]
        assert "PixelMeasuresSequence" in shared_functional_group
        pixel_measures = shared_functional_group.PixelMeasuresSequence[0]
        if image.pixel_spacing is not None:
            assert pixel_measures.PixelSpacing == [
                DSfloat(image.pixel_spacing.width),
                DSfloat(image.pixel_spacing.height),
            ]
        if image.focal_plane_spacing is not None:
            assert pixel_measures.SpacingBetweenSlices == DSfloat(
                image.focal_plane_spacing
            )
        if image.depth_of_field is not None:
            assert pixel_measures.SliceThickness == DSfloat(image.depth_of_field)


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


def assert_item_name_equals_code(item: Dataset, name: Code):
    assert_code_dataset_equals_code(item.ConceptNameCodeSequence[0], name)


def assert_item_string_equals_value(item: Dataset, value: str):
    assert item.TextValue == value


def assert_item_datetime_equals_value(item: Dataset, value: datetime.datetime):
    assert item.DateTime == value


def assert_item_code_equals_value(item: Dataset, value: Union[Code, ConceptCode]):
    assert_code_dataset_equals_code(item.ConceptCodeSequence[0], value)


def assert_item_measurement_equals_value(item: Dataset, value: Measurement):
    assert item.FloatingPointValue == value.value
    assert_code_dataset_equals_code(
        item.MeasurementUnitsCodeSequence[0],
        value.unit,
    )


def assert_code_dataset_equals_code(item: Dataset, code: Union[Code, ConceptCode]):
    assert item.CodeValue == code.value, (item.CodeMeaning, code.meaning)
    assert item.CodingSchemeDesignator == code.scheme_designator, (
        item.CodeMeaning,
        code.meaning,
    )


def assert_next_item_equals_string(iterator: Iterator[Dataset], name: Code, value: str):
    item = next(iterator)
    assert_item_name_equals_code(item, name)
    assert_item_string_equals_value(item, value)


def assert_next_item_equals_code(
    iterator: Iterator[Dataset], name: Code, value: Union[Code, ConceptCode]
):
    item = next(iterator)
    assert_item_name_equals_code(item, name)
    assert_item_code_equals_value(item, value)


def assert_next_item_equals_datetime(
    iterator: Iterator[Dataset], name: Code, value: datetime.datetime
):
    item = next(iterator)
    assert_item_name_equals_code(item, name)
    assert_item_datetime_equals_value(item, value)


def assert_next_item_equals_measurement(
    iterator: Iterator[Dataset], name: Code, value: Measurement
):
    item = next(iterator)
    assert_item_name_equals_code(item, name)
    assert_item_measurement_equals_value(item, value)


def create_collection_dataset(
    collection_dicom: CollectionDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    if collection_dicom.container is not None:
        items.append(
            create_code_item(SampleCodes.container, collection_dicom.container)
        )
    if collection_dicom.specimen_type is not None:
        items.append(
            create_code_item(SampleCodes.specimen_type, collection_dicom.specimen_type)
        )
    items.append(create_processing_type_item(collection_dicom))
    if collection_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, collection_dicom.date_time
            ),
        )
    if collection_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, collection_dicom.description
            )
        )
    items.append(
        create_code_item(SampleCodes.specimen_collection, collection_dicom.method)
    )
    if collection_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, collection_dicom.fixative))
    if collection_dicom.embedding is not None:
        items.append(
            create_code_item(SampleCodes.embedding, collection_dicom.embedding)
        )
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


def create_sampling_dataset(
    sampling_dicom: SamplingDicomModel, identifier: Union[str, SpecimenIdentifier]
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    if sampling_dicom.container is not None:
        items.append(create_code_item(SampleCodes.container, sampling_dicom.container))
    if sampling_dicom.specimen_type is not None:
        items.append(
            create_code_item(SampleCodes.specimen_type, sampling_dicom.specimen_type)
        )
    items.append(create_processing_type_item(sampling_dicom))
    if sampling_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, sampling_dicom.date_time
            ),
        )
    if sampling_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, sampling_dicom.description
            )
        )
    items.append(create_code_item(SampleCodes.sampling_method, sampling_dicom.method))
    items.append(
        create_string_item(
            SampleCodes.parent_specimen_identifier,
            sampling_dicom.parent_specimen_identifier,
        )
    )
    if sampling_dicom.issuer_of_parent_specimen_identifier is not None:
        items.append(
            create_string_item(
                SampleCodes.issuer_of_parent_specimen_identifier,
                sampling_dicom.issuer_of_parent_specimen_identifier,
            )
        )
    items.append(
        create_code_item(
            SampleCodes.parent_specimen_type, sampling_dicom.parent_specimen_type
        )
    )
    if sampling_dicom.location_reference is not None:
        items.append(
            create_string_item(
                SampleCodes.location_frame_of_reference,
                sampling_dicom.location_reference,
            )
        )
    if sampling_dicom.location_description is not None:
        items.append(
            create_string_item(
                SampleCodes.location_of_sampling_site,
                sampling_dicom.location_description,
            )
        )
    if sampling_dicom.location_x is not None:
        items.append(
            create_measurement_item(
                SampleCodes.location_of_sampling_site_x, sampling_dicom.location_x
            )
        )
    if sampling_dicom.location_y is not None:
        items.append(
            create_measurement_item(
                SampleCodes.location_of_sampling_site_y, sampling_dicom.location_y
            )
        )
    if sampling_dicom.location_z is not None:
        items.append(
            create_measurement_item(
                SampleCodes.location_of_sampling_site_z, sampling_dicom.location_z
            )
        )
    if sampling_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, sampling_dicom.fixative))
    if sampling_dicom.embedding is not None:
        items.append(create_code_item(SampleCodes.embedding, sampling_dicom.embedding))
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


def create_processing_dataset(
    processing_dicom: ProcessingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(processing_dicom))
    if processing_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, processing_dicom.date_time
            ),
        )
    if processing_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, processing_dicom.description
            )
        )
    if processing_dicom.processing is not None:
        items.append(
            create_code_item(
                SampleCodes.processing_description, processing_dicom.processing
            )
        )
    if processing_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, processing_dicom.fixative))
    if processing_dicom.embedding is not None:
        items.append(
            create_code_item(SampleCodes.embedding, processing_dicom.embedding)
        )
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


def create_staining_dataset(
    staining_dicom: StainingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(staining_dicom))
    if staining_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, staining_dicom.date_time
            ),
        )
    if staining_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, staining_dicom.description
            )
        )
    for substance in staining_dicom.substances:
        items.append(
            create_code_item(SampleCodes.using_substance, substance)
            if isinstance(substance, SpecimenStainsCode)
            else create_string_item(SampleCodes.using_substance, substance)
        )
    if staining_dicom.fixative is not None:
        items.append(create_code_item(SampleCodes.fixative, staining_dicom.fixative))
    if staining_dicom.embedding is not None:
        items.append(create_code_item(SampleCodes.embedding, staining_dicom.embedding))
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


def create_receiving_dataset(
    receiving_dicom: ReceivingDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(receiving_dicom))
    if receiving_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, receiving_dicom.date_time
            ),
        )
    if receiving_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, receiving_dicom.description
            )
        )
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


def create_storage_dataset(
    storage_dicom: StorageDicomModel,
    identifier: Union[str, SpecimenIdentifier],
):
    dataset = Dataset()
    items = create_identifier_items(identifier)
    items.append(create_processing_type_item(storage_dicom))
    if storage_dicom.date_time is not None:
        items.append(
            create_datetime_item(
                SampleCodes.datetime_of_processing, storage_dicom.date_time
            ),
        )
    if storage_dicom.description is not None:
        items.append(
            create_string_item(
                SampleCodes.processing_description, storage_dicom.description
            )
        )
    dataset.SpecimenPreparationStepContentItemSequence = items
    return dataset


def create_description_dataset(
    slide_sample_id: str,
    slide_sample_uid: UID,
    specimen_id: str,
    block_id: str,
    specimen_type: AnatomicPathologySpecimenTypesCode,
    collection_method: SpecimenCollectionProcedureCode,
    fixative: SpecimenFixativesCode,
    specimen_sampling_method: SpecimenSamplingProcedureCode,
    embedding_medium: SpecimenEmbeddingMediaCode,
    block_sampling_method: SpecimenSamplingProcedureCode,
    block_type: AnatomicPathologySpecimenTypesCode,
    sample_localization: SpecimenLocalization,
    sampling_location: SamplingLocation,
    primary_anatomic_structures: Sequence[Code],
    stains: Sequence[SpecimenStainsCode],
    short_description: Optional[str] = None,
    detailed_description: Optional[str] = None,
    specimen_container: Optional[ContainerTypeCode] = None,
    block_container: Optional[ContainerTypeCode] = None,
):
    collection = create_collection_dataset(
        CollectionDicomModel(
            identifier=specimen_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=None,
            method=collection_method,
            processing=None,
            container=specimen_container,
            specimen_type=specimen_type,
        ),
        identifier=specimen_id,
    )
    fixation = create_processing_dataset(
        ProcessingDicomModel(
            identifier=specimen_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=fixative,
            embedding=None,
            processing=None,
            container=None,
            specimen_type=None,
        ),
        identifier=specimen_id,
    )
    sampling_to_block = create_sampling_dataset(
        SamplingDicomModel(
            identifier=specimen_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=None,
            method=specimen_sampling_method,
            parent_specimen_identifier=specimen_id,
            issuer_of_parent_specimen_identifier=None,
            parent_specimen_type=specimen_type,
            processing=None,
            location_reference=sampling_location.reference
            if sampling_location is not None
            else None,
            location_description=sampling_location.description
            if sampling_location is not None
            else None,
            location_x=sampling_location.x if sampling_location is not None else None,
            location_y=sampling_location.y if sampling_location is not None else None,
            location_z=sampling_location.z if sampling_location is not None else None,
            container=block_container,
            specimen_type=block_type,
        ),
        identifier=block_id,
    )
    embedding = create_processing_dataset(
        ProcessingDicomModel(
            identifier=block_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=embedding_medium,
            processing=None,
            container=None,
            specimen_type=None,
        ),
        identifier=block_id,
    )
    sampling_to_slide = create_sampling_dataset(
        SamplingDicomModel(
            identifier=block_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            fixative=None,
            embedding=None,
            method=block_sampling_method,
            parent_specimen_identifier=block_id,
            issuer_of_parent_specimen_identifier=None,
            parent_specimen_type=block_type,
            processing=None,
            specimen_type=None,
            container=None,
            location_reference=None,
            location_description=None,
            location_x=None,
            location_y=None,
            location_z=None,
        ),
        identifier=slide_sample_id,
    )
    staining = create_staining_dataset(
        StainingDicomModel(
            identifier=slide_sample_id,
            issuer_of_identifier=None,
            date_time=None,
            description=None,
            substances=list(stains),
            fixative=None,
            embedding=None,
            processing=None,
            container=None,
            specimen_type=None,
        ),
        identifier=slide_sample_id,
    )
    description = Dataset()
    description.SpecimenIdentifier = slide_sample_id
    description.SpecimenUID = slide_sample_uid
    description.SpecimenPreparationSequence = [
        collection,
        fixation,
        sampling_to_block,
        embedding,
        sampling_to_slide,
        staining,
    ]
    description.SpecimenTypeCodeSequence = [
        create_code_dataset(AnatomicPathologySpecimenTypesCode("Slide"))
    ]
    description.PrimaryAnatomicStructureSequence = [
        create_code_dataset(item) for item in primary_anatomic_structures
    ]
    description.SpecimenShortDescription = short_description
    description.SpecimenDetailedDescription = detailed_description
    if sample_localization is not None:
        sample_localization_sequence = []
        if sample_localization.reference is not None:
            sample_localization_sequence.append(
                create_string_item(
                    SampleCodes.location_frame_of_reference,
                    sample_localization.reference,
                )
            )
        if sample_localization.description is not None:
            sample_localization_sequence.append(
                create_string_item(
                    SampleCodes.location_of_specimen, sample_localization.description
                )
            )
        if sample_localization.x is not None:
            sample_localization_sequence.append(
                create_measurement_item(
                    SampleCodes.location_of_specimen_x, sample_localization.x
                )
            )
        if sample_localization.y is not None:
            sample_localization_sequence.append(
                create_measurement_item(
                    SampleCodes.location_of_specimen_x, sample_localization.y
                )
            )
        if sample_localization.y is not None:
            sample_localization_sequence.append(
                create_measurement_item(
                    SampleCodes.location_of_specimen_x, sample_localization.y
                )
            )
        if sample_localization.visual_marking is not None:
            sample_localization_sequence.append(
                create_string_item(
                    SampleCodes.visual_marking_of_specimen,
                    sample_localization.visual_marking,
                )
            )
        description.SpecimenLocalizationContentItemSequence = (
            sample_localization_sequence
        )
    return description


def create_identifier_items(identifier: Union[str, SpecimenIdentifier]):
    identifier_item = create_string_item(
        SampleCodes.identifier,
        identifier if isinstance(identifier, str) else identifier.value,
    )
    if not isinstance(identifier, SpecimenIdentifier) or identifier.issuer is None:
        return [identifier_item]
    issuer_item = create_string_item(
        SampleCodes.issuer_of_identifier,
        identifier.issuer.to_hl7v2(),
    )
    return [identifier_item, issuer_item]


def create_code_dataset(code: Union[Code, ConceptCode]):
    dataset = Dataset()
    dataset.CodeValue = code.value
    dataset.CodingSchemeDesignator = code.scheme_designator
    dataset.CodeMeaning = code.meaning
    dataset.CodingSchemeVersion = code.scheme_version
    return dataset


def create_string_item(name: Code, value: str):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.TextValue = value
    return dataset


def create_datetime_item(name: Code, value: datetime.datetime):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.DateTime = value
    return dataset


def create_code_item(name: Code, value: Union[Code, ConceptCode]):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.ConceptCodeSequence = [create_code_dataset(value)]
    return dataset


def create_measurement_item(name: Code, value: Measurement):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(name)]
    dataset.NumericValue = DSfloat(value.value)
    dataset.FloatingPointValue = value.value
    dataset.MeasurementUnitsCodeSequence = [create_code_dataset(value.unit)]
    return dataset


def create_processing_type_item(step: SpecimenPreparationStepDicomModel):
    dataset = Dataset()
    dataset.ConceptNameCodeSequence = [create_code_dataset(SampleCodes.processing_type)]
    if isinstance(step, CollectionDicomModel):
        processing_type_code = SampleCodes.specimen_collection
    elif isinstance(step, SamplingDicomModel):
        processing_type_code = SampleCodes.sampling_method
    elif isinstance(step, SpecimenPreparationStepDicomModel):
        processing_type_code = SampleCodes.sample_processing
    elif isinstance(step, StainingDicomModel):
        processing_type_code = SampleCodes.staining
    elif isinstance(step, ReceivingDicomModel):
        processing_type_code = SampleCodes.receiving
    elif isinstance(step, StorageDicomModel):
        processing_type_code = SampleCodes.storage
    else:
        raise NotImplementedError()
    dataset.ConceptCodeSequence = [create_code_dataset(processing_type_code)]
    return dataset


def create_specimen_preparation_dataset(step: List[Dataset]):
    dataset = Dataset()
    dataset.SpecimenPreparationStepContentItemSequence = step
    return dataset
