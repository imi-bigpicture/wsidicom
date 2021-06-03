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

## Annotation usage
Annotations are currenly handled separarely from the WsiDicom object. Annotations are structured in a hierarchy:
- AnnotationInstance
    Represents a collection of AnnotationGroups. All the groups have the same frame of reference, i.e. annotations are from the same wsi stack.
- AnnotationGroup
    Represents a group of annotations. All annotations in the group are of the same type (e.g. PointAnnotation), have the same label, description and category and type. The category and type are codes that are used to define the annotated feature. A good resource for working with codes is avaiable [here](https://qiicr.gitbook.io/dcmqi-guide/opening/coding_schemes).
- Annotation
    Represents a annotation. An Annotation has a geometry (currently Point, Polyline, Polygon) and an optional list of Measurements.
- Measurement
    Represents a measurement for an Annotation. A Measurement consists of a type-code (e.g. "Area"), a value and a unit-code ("mm")

Codes that are defined in the 222-draft can be created using the create(source, type) function of the ConceptCode-class.

***Load a WSI dataset from files in folder.***
```python
from wsidicom import WsiDicom
slide = WsiDicom.open(path_to_folder)
```
***Create a point annotation at x=10.0, y=20.0 mm.***
```python
from wsidicom import Annotation, Point
point_annotation = Annotation(Point(10.0, 20.0))
```

***Create a point annotation with a measurement.***
```python
from wsidicom import Coder, Measurement
# A measurement is defined by a type code ('Area'), a value (25.0) and a unit code ('Pixels).
area = ConceptCode.measurement('Area')
pixels = ConceptCode.unit('Pixels')
measurement = Measurement(area, 25.0, pixels)
point_annotation_with_measurment = Annotation(Point(10.0, 20.0), [measurement])
```

***Create a group of the annotations.***
```python
from wsidicom import PointAnnotationGroup
# The 222 suplement requires groups to have a label, a category and a type
group = PointAnnotationGroup(
    points=[point_annotation, point_annotation_with_measurment],
    label='group label',
    categorycode=ConceptCode.categorycode('Tissue'),
    typecode=ConceptCode.typecode('Nucleus'),
    description='description'
)
```

***Create a collection of annotation groups.***
```python
from wsidicom import AnnotationInstance
annotations = AnnotationInstance([group], slide.frame_of_reference)
```

***Save the collection to file.***
```python
annotations.save('path_to_dicom_dir/annotation.dcm')
```

***Open the collection and access first annotation in first group.***
```python
annotations.open('path_to_dicom_dir/annotation.dcm')
group = collection[0]
annotation = group[0]
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