# wsidicom changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.21.3] - 2024-10-28

### Fixed

- Missing `WholeSlideMicroscopyImageFrameTypeSequence` in produced DICOM dataset for some image types.
- Do not insert empty `ContainerComponentSequence` and `PrimaryAnatomicStructureSequence` into produced DICOM dataset.
- Only insert `SpacingBetweenSlices` into produced DICOM dataset if multiple focal planes.
- Prefer use of `ZOffsetInSlideCoordinateSystem` in main DICOM dataset to attribute in `SharedFunctionalGroupsSequence`/`PlanePositionSlideSequence`.

## [0.21.2] - 2024-10-21

### Fixed

- Unpinned requirement for numpy to support numpy 2.

## [0.21.1] - 2024-10-17

### Fixed

- Restricted imagecodecs to exclude version 2024.9.22 due failing bug when encoding jpeg.

## [0.21.0] - 2024-10-08

### Added

- Support for pydicom 3.0. This requires Pyhton >3.10.

### Removed

- Support for Python 3.9.
- Support for pydicom <3.

### Fixed

- Missing support for using fsspec in some methods.

## [0.20.6] - 2024-09-12

### Fixed

- Fixed version of pydicom to <3.0 to avoid breaking changes.

## [0.20.5] - 2024-07-25

### Fixed

- Removed debug printing.

## [0.20.4] - 2024-03-25

### Fixed

- Error when calculating downscaled image size causing `save()` with `add_missing_levels` to fail.
- Missing sort when creating new pyramid levels, causing new levels to be created from a to high resolution level.

## [0.20.3] - 2024-03-20

### Fixed

- Missing handling of pyramid index when creating `WsiInstance` using `create_instance()`.

## [0.20.2] - 2024-03-18

### Fixed

- Fixed handling of string values longer than allowed by value representation.
- Corrected staining substances to be either a single string or a list of codes instead of a list of strings or codes.
- Added `allow_none` for json metadata fields where it was missing.
- Do not add empty `CodingSchemeVersion` to concept code name.

## [0.20.1] - 2024-02-22

### Fixed

- Changed default for parameter `source_owned` in `__init__` of `WsiDicom` to be True, to by default close opened file handles.

## [0.20.0] - 2024-02-21

### Added

- Support for opening files using fsspec.
- Support for reading tile offset from TIFF tags if dual DICOM TIFF file.

### Fixed

- Matching of slide uids between groups when validating opened instances.
- Handling of missing dataset values if no default value.
- Wrong concept name used when serializing and deserializing DICOM specimen samplings.

## [0.19.1] - 2024-02-14

### Fixed

- Correct handling of missing image coordinate system when loading metadata.

## [0.19.0] - 2024-02-12

### Added

- Support for decoding HT-JPEG2000 using Pillow, imagecodecs and/or pylibjpeg-openjpeg.
- Optional codec pyjpegls for JPEG-LS support.

### Fixed

- Handling of non-conformant DICOM Web responses.

## [0.18.3] - 2024-01-22

### Fixed

- Fixed missing `levels` property.

## [0.18.2] - 2024-01-12

### Fixed

- Missing KeyError in exception catch for Pyramid.get(), that made getting image data for non-existing levels (by downscaling) not work.
- Handle loading metadata if `ExtendedDepthOfField` not set.
- Metadata for pyramid is now lazy loaded.

## [0.18.1] - 2024-01-12

### Fixed

- Missing indentation in `Patient` and `Equipment` metadata classes.

## [0.18.0] - 2024-01-12

### Added

- Models for DICOM WSI metadata.
- Serializers from DICOM WSI to and from DICOM and json.
- Support for multiple pyramids within the same slide. A pyramid must have the same image coordinate system and extended depth of field (if any). Use the `pyramid`-parameter to set the pyramid in for example `read_region()`, or use `set_selected_pyramid()` to set the pyramid to use. By default the first detected pyramid is used.
- RLE encoding using image codecs.
- JPEG 2000 encoding of lossless YBR using image codecs.

### Changed

- Levels with different extended depth of fields are no longer considered to be the same pyramid.

### Removed

- `OpticalManager`, replaced by new metadata model.

## [0.17.0] - 2023-12-10

### Added

- Option to not include label and overview in `save()`.

### Changed

- Moved option to change label from `open()` to `save()`.

## [0.16.0] - 2023-12-07

### Added

- Option to transcode when saving.
- Option to limit levels to save.
- Fallback to extended offset table if basic offset table is overflown.
- Setting for what Pillow filter to use when scaling.
- RGB support for Jpeg LS.

### Changed

- Refactored frame index reading (basic, extended, and no offset table and native pixel data) and table writing (basic and extended offset table).
- Refactored methods for getting multiple tiles for `ImageData` to make it easier to implement more efficient methods.

## [0.15.2] - 2023-12-01

### Fixed

- Pillow handling of 16 bit grayscale images as 32 bit instead of 8 bit.
- Using Pillow instead of Codec to decode frames from DICOM web, resulting in `PIL.UnidentifiedImageError` for transfer syntaxes not supported by Pillow.
- Fix decoding of non-square images with `PydicomDecoder` and `PylibjpegRleDecoder`.

## [0.15.1] - 2023-11-30

### Fixed

- Correct order of pixel spacing.
- Correct `photometric_interpretation` for `Jpeg2kSettings`
- `PillowEncoder` encoding Jpeg2000 as `Jpeg2kEncoder`.

## [0.15.0] - 2023-11-30

### Added

- Fallback to EOT when overflowing BOT.
- Use AvailableTransferSyntaxUID if provided to determine transfer syntax to use when opening DICOM Web instances.

### Fixed

- Missing empty BOT when writing EOT.
- Fixed accessing settings for PillowEncoder.

## [0.14.0] - 2023-11-29

### Added

- Support for additional transfer syntaxes both for reading and writing, using pydicom pixel handlers, (optional) image codecs, and (optional) pylibjpeg-rle.
- Additional decoders and encoders to Pillow. Decoder and encoder is selected automatically. Decoder to use can be overridden with the `prefered_decoder`-setting.
- Detection of suitable available transfer syntax when opening slide with DICOM web.
- Support for opening DICOMDIR files using `open_dicomdir()`.

### Changed

- Opening DICOM web instances in parallel. Configurable with `open_web_theads`-setting.
- `open_web()` now takes a list of requested transfer syntaxes to test if available from the server.
- When fetching multiple frames from DICOM web, fetch multiple frames per request instead of making several requests.
- Renamed `OffsetTableType` option `NONE` to `EMPTY`, and added type `NONE` for use with unencapsulated data.

### Removed

- Support for Python 3.8.

### Fixed

- Loading annotation instances using DICOM web.

## [0.13.0] - 2023-11-11

### Added

- Allow a `requests.Session` object to be passed to `WsiDicomWebClient.create_client()`.

### Changed

- Refactored `WsiDicomWebClient.__init__()` to take a `DICOMwebClient` instance, and moved the old `__init__()` method into `WsiDicomWebClient.create_client()`.
- Make some arguments optional for `WsiDicomWebClient.create_client()`.
- Removed `WsiDicomFileClient` since it is no longer needed.
- Fetching multiple frames in one request instead of one request per frame when using DICOM Web.
- Allow multiple series UIDs to be passed to `WsiDicom.open_web()`.
- Loosen UID matching to just `study_instance`.
- Require equality of 'TotalPixelMatrixOriginSequence' when matching datasets.

## [0.12.0] - 2023-10-04

### Changed

- Updated dependency of Pillow to include Pillow 10.
- Tests written as pytests instead of unittest.

## [0.11.0] - 2023-08-09

### Added

- Override image size for labels and overviews if it is likely to be wrong.

### Changed

- Use logger instead of issuing warnings.

### Fixed

- Error in documentation of read_overview() in readme.
- Spellings.

## [0.10.0] - 2023-06-01

### Added

- Support for using opened streams.

### Changed

- Consider image as full tiled if only one tile.

### Removed

- CidConceptCode.from_meaning(), use constructor with only meaning instead.

### Fixed

- Correct formatting of decimal string values when writing dataset.

## [0.9.0] - 2023-03-31

### Added

- Support for opening DICOM WSI using DICOMWeb.
- save() now takes the optional parameter add_missing_levels, that enables adding missing pyramid levels up to the single-tile level.
- read_region(), read_region_mm(), and read_region_mpp() takes an optional parameter threads, that allows multiple threads to be used for stitching together the region.

### Changed

- WsiDicom is now initialized using a Source, that is responsible for provides the instances to view.
- Saving a WsiDicom is now handled by WsiDicomFileTarget.
- Refactoring due to adding support for DICOMWeb and opening instances using Source and saving using Target.
- Frame positions and tiling for levels are now parsed lazily, e.g. on first tile access.

### Removed

- The construct_pyramid() method of the Levels-class has been removed in favour of the add_missing_levels-parameter in save()

## [0.8.0] - 2023-03-21

### Added

- Set TotalPixelMatrixOriginSequence and ImageOrientationSlide from ImageOrgin object.

## [0.7.0] - 2023-02-13

### Added

- Parameter to change label to given image.

### Changed

- Label and overview series can now be None.

## [0.6.0] - 2023-01-24

### Changed

- Raise WsiDicomNotFoundError if no files found for open().
- Added Python 3.11 as supported version.

## [0.5.0] - 2022-12-13

### Added

- Method to center-zoom a Region or RegionMm.

### Changed

- Do not use threads in method WsiDicomFileWriter._write_pixel_data() if only one worker.

## [0.4.0] - 2022-06-30

### Added

- Focal planes are considered equal if configurable within threshold distance, see config.focal_plane_distance_threshold
- Option to read region defined in slide coordinate system for get_region_mm().

### Changed

- Default chunk size for saving is now set to 16 tiles.
- Drop support for python 3.7.

### Fixed

- Focal planes are now written to file in correct order.

## [0.3.2] - 2022-05-08

### Fixed

- Fix version in __init__.py.

## [0.3.2] - 2022-05-08

### Fixed

- Fix version in __init__.py

## [0.3.1] - 2022-05-04

### Fixed

- Order of parameters for ConceptCode matches pydicom Code.

## [0.3.0] - 2022-04-20

### Added

- Simple check for if DICOM WSI is formatted for fast viewing (WsiDicom.ready_for_viewing()).
- __version__ added.

### Changed

- Action for downloading test data.
- Do not set pydicom configuration parameters.

### Fixed

- Use correct data type for 32 bit integer in numpy arrays.
- Fix check for if focal planes are sparse.

## [0.2.0] - 2022-02-14

### Added

- Configuration through global variable settings.
- Relaxed requirements for some DICOM attributes.
- Writing BOT and EOT, and reading EOT.
- Pyramid creation.

### Changed

- save() converts the dataset to TILED_FULL.

### Removed

- Using null path as output path for save().

### Fixed

- Fix NumberOfFrames when saving.

## [0.1.0] - 2021-11-26

### Added

- Initial release of wsidicom

[Unreleased]: https://github.com/imi-bigpicture/wsidicom/compare/v0.21.3..HEAD
[0.21.3]: https://github.com/imi-bigpicture/wsidicom/compare/v0.21.2..v0.21.3
[0.21.2]: https://github.com/imi-bigpicture/wsidicom/compare/v0.21.1..v0.21.2
[0.21.1]: https://github.com/imi-bigpicture/wsidicom/compare/v0.21.0..v0.21.1
[0.21.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.20.6..v0.21.0
[0.20.6]: https://github.com/imi-bigpicture/wsidicom/compare/v0.20.5..v0.20.6
[0.20.5]: https://github.com/imi-bigpicture/wsidicom/compare/v0.20.4..v0.20.5
[0.20.4]: https://github.com/imi-bigpicture/wsidicom/compare/v0.20.3..v0.20.4
[0.20.3]: https://github.com/imi-bigpicture/wsidicom/compare/v0.20.2..v0.20.3
[0.20.2]: https://github.com/imi-bigpicture/wsidicom/compare/v0.20.1..v0.20.2
[0.20.1]: https://github.com/imi-bigpicture/wsidicom/compare/v0.20.0..v0.20.1
[0.20.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.19.1..v0.20.0
[0.19.1]: https://github.com/imi-bigpicture/wsidicom/compare/v0.19.0..v0.19.1
[0.19.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.18.3..v0.19.0
[0.18.3]: https://github.com/imi-bigpicture/wsidicom/compare/v0.18.2..v0.18.3
[0.18.2]: https://github.com/imi-bigpicture/wsidicom/compare/v0.18.1..v0.18.2
[0.18.1]: https://github.com/imi-bigpicture/wsidicom/compare/v0.18.0..v0.18.1
[0.18.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.17.0..v0.18.0
[0.17.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.16.0..v0.17.0
[0.16.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.15.2..v0.16.0
[0.15.2]: https://github.com/imi-bigpicture/wsidicom/compare/v0.15.1..v0.15.2
[0.15.1]: https://github.com/imi-bigpicture/wsidicom/compare/v0.15.0..v0.15.1
[0.15.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.14.0..v0.15.0
[0.14.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.13.0..v0.14.0
[0.13.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.12.0..v0.13.0
[0.12.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.11.0..v0.12.0
[0.11.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.10.0..v0.11.0
[0.10.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.9.0..v0.10.0
[0.9.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.8.0..v0.9.0
[0.8.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.7.0..v0.8.0
[0.7.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.6.0..v0.7.0
[0.6.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.5.0..v0.6.0
[0.5.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.4.0..v0.5.0
[0.4.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.3.2..v0.4.0
[0.3.2]: https://github.com/imi-bigpicture/wsidicom/compare/v0.3.1..v0.3.2
[0.3.1]: https://github.com/imi-bigpicture/wsidicom/compare/v0.3.0..v0.3.1
[0.3.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.2.0..v0.3.0
[0.2.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.1.0..v0.2.0
[0.1.0]: https://github.com/imi-bigpicture/wsidicom/tree/refs/tags/v0.1.0
