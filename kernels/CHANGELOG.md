# wsidicom-kernels changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `compose`, pasting tiles onto a canvas in place at integer or sub-pixel
  positions, the edges of sub-pixel tiles blended by coverage so that
  neighbouring tiles meet without seams. Tiles may be row slices of larger
  arrays, and `crops` places rectangles of tiles without copying them out.
- `stitch`, assembling a grid of tiles into one image.
- `downscale`, reducing an image by an integer factor.
- `downscale_grid`, reducing a grid of tiles into one image, optionally into a
  filled frame of a given size.
- `resize`, resampling an image to any size.
- `rgb_to_rgbx`, expanding RGB to 4-byte RGBX.
- `Interpolation`, how `compose` samples tiles at fractional positions, and
  `ResizeFilter`, the antialiasing filters of `downscale` and `resize`.
