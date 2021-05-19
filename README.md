# *wsidicom*
*wsidicom* is a Python package for reading [DICOM WSI](http://dicom.nema.org/Dicom/DICOMWSI/) file sets.

## Installing *wsidicom*
*wsidicom* is **not yet** available on PyPI:
```console
$ python -m pip git+https://github.com/imi-bigpicture/wsidicom.git
```

## Important note
Please note that this is an early release and the API is not frozen yet. Function names and functionality is prone to change.

## Requirements
*wsidicom* uses pydicom, numpy and Pillow (with jpeg and jpeg2000 plugins).

## Limitations
Levels are required to have (close to) 2 factor scale and same tile size.

Only JPEGBaseline8Bit, JPEG2000 and JPEG2000Lossless transfer syntax is supported.

Optical path identifiers needs to be unique across file set.

## Basic usage
***Load a WSI dataset from files in folder.***
```python
from wsidicom import WsiDicom
slide = WsiDicom.open(path_to_folder)
```
***Read a 200x200 px region starting from px 1000, 1000 at level 6.***
 ```python
region = slide.read_region((1000, 1000), 6, (200, 200))
```
***Read 3x3 mm region starting at 0, 0 mm at level 6.***
 ```python
region_mm = slide.read_region_mm((0, 0), 6, (3, 3))
```
***Read 3x3 mm region starting at 0, 0 mm with pixel spacing 0.01 mm/px.***
 ```python
region_mpp = slide.read_region_mpp((0, 0), 0.01, (3, 3))
```
***Read a thumbnail of the whole slide with maximum dimensions 200x200 px.***
 ```python
thumbnail = slide.read_thumbnail(200, 200)
```
***Read an overview image (if available).***
 ```python
overview = slide.read_overview()
```
***Read a label image (if available).***
 ```python
label = slide.read_label()
```
***Read (decoded) tile from position 1, 1 in level 6.***
 ```python
tile = slide.read_tile(6, (1, 1))
```
***Read (encoded) tile from position 1, 1 in level 6.***
 ```python
tile_bytes = slide.read_encoded_tile(6, (1, 1))
```
***Close files***
 ```python
slide.close()
```

## Data structure
A WSI DICOM pyramid is in *wsidicom* represented by a hierarchy of objects of different classes, starting from bottom:
- *WsiDicomFile*, represents a WSI DICOM file. A file can contain one or several focal planes and/or optical paths.
- *WsiDicomInstance*, reperesents a single or a concatenation of WSI DICOM files.
- *WsiDicomLevel*, represents a group of instances with the same image size, i.e. of the same level.
- *WsiDicomLevels*, represents a group of levels, i.e. the pyrimidal structure.
- *WsiDicom*, represents a collection of levels, labels and overviews.

Labels and overviews are structured similarly to levels, but with somewhat different properties and restrictions.

The structure is easiest created using the open() helper functions, e.g. to create a WsiDicom-object:
```python
slide = WsiDicom.open(path_to_folder)
```

But the structure can also be created manually from the bottom:
```python
file = WsiDicomFile(path_to_file)
instance = WsiDicomInstance([file])
level = WsiDicomLevel([instance])
levels = WsiDicomLevels([level])
slide = WsiDicom([levels])
```

## Advanced frame access
It is possible to get direct access to the pydicom filepointer for faster frame readout. To do so, first get the instance of interest:
```python
from wsidicom import WsiDicom
slide = WsiDicom.open(path_to_folder)
instance, z, path = slide.get_instance(level, z, path)
```
And then get the filepointer, frame position and frame lenght for a specific tile, z, and path:
```python
fp, position, lenght = instance.get_filepointer(tile, z, path)
```

## Setup environment for development
Requires poetry and pytest and pytest-watch installed in the virtual environment.

```console
$ git clone https://github.com/imi-bigpicture/wsidicom.git
$ poetry install
```

To watch unit tests use:

```console
$ poetry run pytest-watch -- -m unittest
```

To run integration tests, set the WSIDICOM_TESTDIR environment varible to your test data directory and then use:

```console
$ poetry run pytest -m integration
```

## Acknowledgement
This open source project is part of a project that has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 945358. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA.