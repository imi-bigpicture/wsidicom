# *wsidicom*
*wsidicom* is a Python package for reading [DICOM WSI](http://dicom.nema.org/Dicom/DICOMWSI/) file sets. The aims with the projects are:
- Easy to use interface for reading and writing WSI DICOM images and annotations using the DICOM Media Storage Model.
- Support the latest and upcomming DICOM standards.
- Platform independent installation via PyPI.

## Installing *wsidicom*
*wsidicom* is available on PyPI:
```console
$ pip install wsidicom
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
- *WsiDicomFile*, represents a WSI DICOM file, used for accessing DicomImageData and WsiDataset.
- *DicomImageData*, represents the image data in one or several WSI DICOM files.
- *WsiDataset*, represents the image metadata in one or several WSI DICOM files.
- *WsiInstance*, represents image data and image metadata.
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
instance = WsiInstance(file.dataset, DicomImageData(files))
level = WsiDicomLevel([instance])
levels = WsiDicomLevels([level])
slide = WsiDicom([levels])
```

## Adding support for other file formats.
By subclassing *ImageData* and implementing the required properties (transfer_syntax, image_size, tile_size, and pixel_spacing) and methods (get_tile() and close()) *wsidicom* can be used to access wsi images in other file formats than DICOM. In addition to a ImageData-object, image data, specified in a DICOM dataset, must also be created. For example, assuming a implementation of MyImageData exists that takes a path to a image file as argument and create_dataset() produces a DICOM dataset (see is_wsi_dicom() of WsiDataset for required attributes), WsiInstancees could be created for each pyramidal level, label, or overview:
```python
image_data = MyImageData('path_to_image_file')
dataset = create_dataset()
instance = WsiInstance(dataset, image_data)
```
The created instances can then be arranged into levels etc, and opened as a WsiDicom-object as described in 'Data structure'.

## Annotation usage
Annotations are structured in a hierarchy:
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
    annotations=[point_annotation, point_annotation_with_measurment],
    label='group label',
    categorycode=ConceptCode.category('Tissue'),
    typecode=ConceptCode.type('Nucleus'),
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

***Reopen the slide and access the annotation instance.***
```python
slide = WsiDicom.open(path_to_folder)
annotations = slide.annotations
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
Unfortunately due to data sharing restrictions the default WSI DICOM test files can't be shared. We are working on a solution, please disregard the integration tests at the moment.

## Other DICOM python tools
- [pydicom](https://pydicom.github.io/)
- [highdicom](https://github.com/MGHComputationalPathology/highdicom)

## Contributing
We welcome any contributions to help improve this tool for the WSI DICOM community!

We recommend first creating an issue before creating potential contributions to check that the contribution is in line with the goals of the project. To submit your contribution, please issue a pull request on the imi-bigpicture/wsidicom repository with your changes for review.

Our aim is to provide constructive and positive code reviews for all submissions. The project relies on gradual typing and roughly follows PEP8. However, we are not dogmatic. Most important is that the code is easy to read and understand.

## Acknowledgement
*wsidicom*: Copyright 2021 Sectra AB, licensed under Apache 2.0.

This project is part of a project that has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 945358. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA. IMI website: www.imi.europa.eu