#    Copyright 2021, 2022, 2023 SECTRA AB
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

from pydicom.tag import Tag

PixelDataTag = Tag("PixelData")
ExtendedOffsetTableTag = Tag("ExtendedOffsetTable")
ExtendedOffsetTableLengthsTag = Tag("ExtendedOffsetTableLengths")
LossyImageCompressionRatioTag = Tag("LossyImageCompressionRatio")
LossyImageCompressionMethodTag = Tag("LossyImageCompressionMethod")
OpticalPathIdentificationSequenceTag = Tag("OpticalPathIdentificationSequence")
OpticalPathIdentifierTag = Tag("OpticalPathIdentifier")
PlanePositionSlideSequenceTag = Tag("PlanePositionSlideSequence")
RowPositionInTotalImagePixelMatrixTag = Tag("RowPositionInTotalImagePixelMatrix")
ColumnPositionInTotalImagePixelMatrixTag = Tag("ColumnPositionInTotalImagePixelMatrix")
XOffsetInSlideCoordinateSystemTag = Tag("XOffsetInSlideCoordinateSystem")
YOffsetInSlideCoordinateSystemTag = Tag("YOffsetInSlideCoordinateSystem")
ZOffsetInSlideCoordinateSystemTag = Tag("ZOffsetInSlideCoordinateSystem")
PerFrameFunctionalGroupsSequenceTag = Tag("PerFrameFunctionalGroupsSequence")
SharedFunctionalGroupsSequenceTag = Tag("SharedFunctionalGroupsSequence")
PixelMeasuresSequenceTag = Tag("PixelMeasuresSequence")
TotalPixelMatrixOriginSequenceTag = Tag("TotalPixelMatrixOriginSequence")
OpticalPathSequenceTag = Tag("OpticalPathSequence")
LossyImageCompressionTag = Tag("LossyImageCompression")
InstanceCreationDateTag = Tag("InstanceCreationDate")
InstanceCreationTimeTag = Tag("InstanceCreationTime")

# Instance identity
SOPClassUIDTag = Tag("SOPClassUID")
SOPInstanceUIDTag = Tag("SOPInstanceUID")
StudyInstanceUIDTag = Tag("StudyInstanceUID")
SeriesInstanceUIDTag = Tag("SeriesInstanceUID")
InstanceNumberTag = Tag("InstanceNumber")
FrameOfReferenceUIDTag = Tag("FrameOfReferenceUID")

# Concatenation
ConcatenationUIDTag = Tag("ConcatenationUID")
SOPInstanceUIDOfConcatenationSourceTag = Tag("SOPInstanceUIDOfConcatenationSource")
InConcatenationNumberTag = Tag("InConcatenationNumber")
InConcatenationTotalNumberTag = Tag("InConcatenationTotalNumber")
ConcatenationFrameOffsetNumberTag = Tag("ConcatenationFrameOffsetNumber")

# Image and frame type
ImageTypeTag = Tag("ImageType")
FrameTypeTag = Tag("FrameType")
WholeSlideMicroscopyImageFrameTypeSequenceTag = Tag(
    "WholeSlideMicroscopyImageFrameTypeSequence"
)
NumberOfFramesTag = Tag("NumberOfFrames")
DimensionOrganizationTypeTag = Tag("DimensionOrganizationType")

# Pixel format
RowsTag = Tag("Rows")
ColumnsTag = Tag("Columns")
BitsAllocatedTag = Tag("BitsAllocated")
BitsStoredTag = Tag("BitsStored")
HighBitTag = Tag("HighBit")
SamplesPerPixelTag = Tag("SamplesPerPixel")
PhotometricInterpretationTag = Tag("PhotometricInterpretation")
PixelRepresentationTag = Tag("PixelRepresentation")
PlanarConfigurationTag = Tag("PlanarConfiguration")

# Total pixel matrix
TotalPixelMatrixColumnsTag = Tag("TotalPixelMatrixColumns")
TotalPixelMatrixRowsTag = Tag("TotalPixelMatrixRows")
TotalPixelMatrixFocalPlanesTag = Tag("TotalPixelMatrixFocalPlanes")
NumberOfOpticalPathsTag = Tag("NumberOfOpticalPaths")
ImageOrientationSlideTag = Tag("ImageOrientationSlide")

# Spacing and imaged volume
PixelSpacingTag = Tag("PixelSpacing")
SliceThicknessTag = Tag("SliceThickness")
SpacingBetweenSlicesTag = Tag("SpacingBetweenSlices")
ImagedVolumeWidthTag = Tag("ImagedVolumeWidth")
ImagedVolumeHeightTag = Tag("ImagedVolumeHeight")
ImagedVolumeDepthTag = Tag("ImagedVolumeDepth")

# Focus
FocusMethodTag = Tag("FocusMethod")
ExtendedDepthOfFieldTag = Tag("ExtendedDepthOfField")
NumberOfFocalPlanesTag = Tag("NumberOfFocalPlanes")
DistanceBetweenFocalPlanesTag = Tag("DistanceBetweenFocalPlanes")

# Code
CodeValueTag = Tag("CodeValue")
CodingSchemeDesignatorTag = Tag("CodingSchemeDesignator")
CodeMeaningTag = Tag("CodeMeaning")
CodingSchemeVersionTag = Tag("CodingSchemeVersion")

# Content item
ValueTypeTag = Tag("ValueType")
ConceptNameCodeSequenceTag = Tag("ConceptNameCodeSequence")
ConceptCodeSequenceTag = Tag("ConceptCodeSequence")
TextValueTag = Tag("TextValue")
DateTimeTag = Tag("DateTime")
NumericValueTag = Tag("NumericValue")
FloatingPointValueTag = Tag("FloatingPointValue")
MeasurementUnitsCodeSequenceTag = Tag("MeasurementUnitsCodeSequence")

# Issuer of identifier
UniversalEntityIDTag = Tag("UniversalEntityID")
UniversalEntityIDTypeTag = Tag("UniversalEntityIDType")
LocalNamespaceEntityIDTag = Tag("LocalNamespaceEntityID")

# Palette colour lookup table
SegmentedRedPaletteColorLookupTableDataTag = Tag(
    "SegmentedRedPaletteColorLookupTableData"
)
SegmentedGreenPaletteColorLookupTableDataTag = Tag(
    "SegmentedGreenPaletteColorLookupTableData"
)
SegmentedBluePaletteColorLookupTableDataTag = Tag(
    "SegmentedBluePaletteColorLookupTableData"
)
RedPaletteColorLookupTableDataTag = Tag("RedPaletteColorLookupTableData")
GreenPaletteColorLookupTableDataTag = Tag("GreenPaletteColorLookupTableData")
BluePaletteColorLookupTableDataTag = Tag("BluePaletteColorLookupTableData")

# Character set
SpecificCharacterSetTag = Tag("SpecificCharacterSet")

# File metadata
TransferSyntaxUIDTag = Tag("TransferSyntaxUID")
MediaStorageSOPInstanceUIDTag = Tag("MediaStorageSOPInstanceUID")
MediaStorageSOPClassUIDTag = Tag("MediaStorageSOPClassUID")
