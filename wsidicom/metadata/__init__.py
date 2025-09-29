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

"""Module with metadata models and schemas for serialization to and from DICOM or
json."""

from wsidicom.metadata.equipment import Equipment
from wsidicom.metadata.image import (
    ExtendedDepthOfField,
    FocusMethod,
    Image,
    ImageCoordinateSystem,
    LossyCompression,
)
from wsidicom.metadata.label import Label
from wsidicom.metadata.optical_path import (
    ConstantLutSegment,
    DiscreteLutSegment,
    ImagePathFilter,
    LightPathFilter,
    LinearLutSegment,
    Lut,
    Objectives,
    OpticalPath,
)
from wsidicom.metadata.overview import Overview
from wsidicom.metadata.patient import Patient, PatientDeIdentification, PatientSex
from wsidicom.metadata.pyramid import Pyramid
from wsidicom.metadata.sample import (
    Collection,
    Embedding,
    Fixation,
    LocalIssuerOfIdentifier,
    Measurement,
    Processing,
    Receiving,
    Sample,
    SampleLocalization,
    Sampling,
    SamplingLocation,
    SlideSample,
    Specimen,
    SpecimenIdentifier,
    Staining,
    Storage,
    UniversalIssuerOfIdentifier,
    UniversalIssuerType,
    UnknownSampling,
)
from wsidicom.metadata.series import Series
from wsidicom.metadata.slide import Slide
from wsidicom.metadata.study import Study
from wsidicom.metadata.wsi import WsiMetadata

__all__ = [
    "Equipment",
    "ExtendedDepthOfField",
    "FocusMethod",
    "Image",
    "ImageCoordinateSystem",
    "Label",
    "ConstantLutSegment",
    "DiscreteLutSegment",
    "ImagePathFilter",
    "LightPathFilter",
    "LinearLutSegment",
    "Lut",
    "Objectives",
    "OpticalPath",
    "Overview",
    "Patient",
    "PatientDeIdentification",
    "PatientSex",
    "Pyramid",
    "Collection",
    "Embedding",
    "Fixation",
    "LocalIssuerOfIdentifier",
    "LossyCompression",
    "Measurement",
    "Processing",
    "Receiving",
    "Sample",
    "SampleLocalization",
    "Sampling",
    "SamplingLocation",
    "SlideSample",
    "Specimen",
    "SpecimenIdentifier",
    "Staining",
    "Storage",
    "UniversalIssuerOfIdentifier",
    "UniversalIssuerType",
    "UnknownSampling",
    "Series",
    "Slide",
    "Study",
    "WsiMetadata",
]
