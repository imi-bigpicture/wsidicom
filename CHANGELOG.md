# wsidicom changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - ...

## [0.3.2] - 2022-05-08
### Fixed
- Fix version in __init__.py

## [0.3.1] - 2022-05-04
### Fixed
- Order of paramaters for ConceptCode matches pydicom Code.

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

[Unreleased]: https://github.com/imi-bigpicture/wsidicom/compare/0.3.2..HEAD
[0.3.2]: https://github.com/imi-bigpicture/wsidicom/compare/v0.3.1..v0.3.2
[0.3.1]: https://github.com/imi-bigpicture/wsidicom/compare/v0.3.0..v0.3.1
[0.3.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.2.0..v0.3.0
[0.2.0]: https://github.com/imi-bigpicture/wsidicom/compare/v0.1.0..v0.2.0
[0.1.0]: https://github.com/imi-bigpicture/wsidicom/tree/refs/tags/v0.1.0