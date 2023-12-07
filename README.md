# *wsidicom*

*wsidicom* is a Python package for reading [DICOM WSI](http://dicom.nema.org/Dicom/DICOMWSI/). The aims with the project are:

- Easy to use interface for reading and writing WSI DICOM images and annotations either from file or through DICOMWeb.
- Support the latest and upcoming DICOM standards.
- Platform independent installation via PyPI.

## Installing *wsidicom*

*wsidicom* is available on PyPI:

```console
pip install wsidicom
```

And through conda:

```console
conda install -c conda-forge wsidicom
```

## Important note

Please note that this is an early release and the API is not frozen yet. Function names and functionality is prone to change.

## Requirements

*wsidicom* uses pydicom, numpy, Pillow, and dicomweb-client. Imagecodecs and pylibjpeg-rle can be installed as optionals to support additional transfer syntaxes.

## Limitations

- Levels are required to have (close to) 2 factor scale and same tile size.

- Only 8 bits per sample is supported for color images, and 8 and 16 bits for grayscale images.

- Without optional dependencies, the following transfer syntaxes are supported:

  - JPEGBaseline8Bit
  - JPEG2000
  - JPEG2000Lossless
  - ImplicitVRLittleEndian
  - ExplicitVRLittleEndian
  - ExplicitVRBigEndian

- With imagecodecs, the following transfer syntaxes are additionally supported:

  - JPEGExtended12Bit
  - JPEGLosslessP14
  - JPEGLosslessSV1
  - JPEGLSLossless
  - JPEGLSNearLossless

- With pylibjpeg-rle RLELossless is additionally supported.

- Optical path identifiers needs to be unique across instances.

- Only one pyramid (i.e. offset from slide corner) per frame of reference is supported.

## Basic usage

***Load a WSI dataset from files in folder.***

```python
from wsidicom import WsiDicom
slide = WsiDicom.open(path_to_folder)
```

***Or load a WSI dataset from opened streams.***

```python
from wsidicom import WsiDicom

slide = WsiDicom.open([file_stream_1, file_stream_2, ... ])
```

***Or load a WSI dataset from a DICOMDIR.***

```python
from wsidicom import WsiDicom

slide = WsiDicom.open_dicomdir(path_to_dicom_dir)
```

***Or load a WSI dataset from DICOMWeb.***

```python
from wsidicom import WsiDicom, WsiDicomWebClient
from requests.auth import HTTPBasicAuth

auth = HTTPBasicAuth('username', 'password')
client = WsiDicomWebClient.create_client(
    'dicom_web_hostname',
    '/qido',
    '/wado,
    auth
)
slide = WsiDicom.open_web(
    client,
    "study uid to open",
    "series uid to open" or ["series uid 1 to open", "series uid 2 to open"]
)
```

Alternatively, if you have already created an instance of
`dicomweb_client.DICOMwebClient`, that may be used to create the
`WsiDicomWebClient` like so:

```python
dw_client = DICOMwebClient(url)
client = WsiDicomWebClient(dw_client)
```

Then proceed to call `WsiDicom.open_web()` with this as in the first example.

***Use as a context manager.***

```python
from wsidicom import WsiDicom
with WsiDicom.open(path_to_folder) as slide:
    ...
```

***Read a 200x200 px region starting from px 1000, 1000 at level 6.***

```python
region = slide.read_region((1000, 1000), 6, (200, 200))
```

***Read a 2000x2000 px region starting from px 1000, 1000 at level 4 using 4 threads.***

```python
region = slide.read_region((1000, 1000), 6, (200, 200), threads=4)
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
thumbnail = slide.read_thumbnail((200, 200))
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

## Saving files

An opened WsiDicom instance can be saved to a new path using the save()-method. The produced files will be:

- Fully tiled. Any sparse tiles will be replaced with a blank tile with color depending on the photometric interpretation.
- Have a basic offset table (or optionally an extended offset table or no offset table).
- Not be concatenated.

By default frames are copied as-is, i.e. without re-compression.

```python
with WsiDicom.open(path_to_folder) as slide:
    slide.save(path_to_output)
```

The output folder must already exists. Be careful to specify a unique folder folder to avoid mixing files from different images.

Optionally frames can be transcoded, either by a encoder setting or an encoder:

```python
from wsidicom.codec import JpegSettings

with WsiDicom.open(path_to_folder) as slide:
    slide.save(path_to_output, transcoding=JpegSettings())
```

## Settings

*wsidicom* can be configured with the settings variable. For example, set the parsing of files to strict:

```python
from wsidicom import settings
settings.strict_uid_check = True
settings._strict_attribute_check = True
```

## Annotation usage

Annotations are structured in a hierarchy:

- AnnotationInstance
    Represents a collection of AnnotationGroups. All the groups have the same frame of reference, i.e. annotations are from the same wsi stack.
- AnnotationGroup
    Represents a group of annotations. All annotations in the group are of the same type (e.g. PointAnnotation), have the same label, description and category and type. The category and type are codes that are used to define the annotated feature. A good resource for working with codes is available [here](https://qiicr.gitbook.io/dcmqi-guide/opening/coding_schemes).
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
from wsidicom import ConceptCode, Measurement
# A measurement is defined by a type code ('Area'), a value (25.0) and a unit code ('Pixels).
area = ConceptCode.measurement('Area')
pixels = ConceptCode.unit('Pixels')
measurement = Measurement(area, 25.0, pixels)
point_annotation_with_measurment = Annotation(Point(10.0, 20.0), [measurement])
```

***Create a group of the annotations.***

```python
from wsidicom import PointAnnotationGroup
# The 222 supplement requires groups to have a label, a category and a type
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
annotations = AnnotationInstance([group], 'volume', slide.uids)
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

Requires poetry installed in the virtual environment.

```console
git clone https://github.com/imi-bigpicture/wsidicom.git
poetry install
```

To watch unit tests use:

```console
poetry run pytest-watch -- -m unittest
```

The integration tests uses test images from nema.org that's needs to be downloaded. The location of the test images can be changed from the default tests\testdata\slides using the environment variable WSIDICOM_TESTDIR. Download the images using the supplied script:

```console
python .\tests\download_test_images.py
```

If the files are already downloaded the script will validate the checksums.

To run integration tests:

```console
poetry run pytest -m integration
```

## Data structure

A WSI DICOM pyramid is in *wsidicom* represented by a hierarchy of objects of different classes, starting from bottom:

- *WsiDicomFile*, represents a WSI DICOM file, used for accessing WsiDicomFileImageData and WsiDataset.
- *WsiDicomFileImageData*, represents the image data in one or several WSI DICOM files.
- *WsiDataset*, represents the image metadata in one or several WSI DICOM files.
- *WsiInstance*, represents image data and image metadata.
- *Level*, represents a group of instances with the same image size, i.e. of the same level.
- *Levels*, represents a group of levels, i.e. the pyrimidal structure.
- *WsiDicom*, represents a collection of levels, labels and overviews.

Labels and overviews are structured similarly to levels, but with somewhat different properties and restrictions. For DICOMWeb the WsiDicomFile\* classes are replaced with WsiDicomWeb\* classes.

A Source is used to create WsiInstances, either from files (*WsiDicomFileSource*) or DICOMWeb (*WsiDicomWebSource*), and can be used to to Initiate a *WsiDicom* object. A source is easiest created with the open() and open_web() helper functions, e.g.:

```python
slide = WsiDicom.open(path_to_folder)
```

## Code structure

- [wsidicom.py](wsidicom/wsidicom.py) - Main class with methods to open DICOM WSI objects.
- [source.py](wsidicom/source.py) - Metaclass Source for serving WsiInstances to WsiDicom.
- [series](wsidicom/series) - Series implementations Levels, Labels, and Overview.
- [group](wsidicom/group) - Group implementations, e.g. Level.
- [instance](wsidicom/instance) - Instance implementations WsiIsntance and WsiDataset, the metaclass ImageData and ImageData implementations WsiDicomImageData and PillowImageData.
- [file](wsidicom/file) - Implementation for reading and writing DICOM WSI files.
- [web](wsidicom/web) - Implementation for reading DICOM WSI from DICOMWeb.
- [graphica_annotations](wsidicom/graphical_annotations.py) - Handling graphical annotations.
- [conceptcode.py](wsidicom/conceptcode.py) - Handling of DICOM concept codes.
- [config.py](wsidicom/config.py) - Handles configuration settings.
- [errors.py](wsidicom/errors.py) - Custom errors.
- [geometry.py](wsidicom/geometry.py) - Classes for geometry handling.
- [optical.py](wsidicom/optical.py) - Handles optical paths.
- [uid.py](wsidicom/uid.py) - Handles DICOM uids.
- [stringprinting.py](wsidicom/stringprinting.py) - For nicer string printing of objects.

## Adding support for other file formats

Support for other formats (or methods to access DICOM data) can be implemented by creating a new Source implementation, that should create WsiInstances for the implemented formats. A format specific implementations of the *ImageData* is likely needed to access the WSI image data. Additionally a WsiDataset needs to be created that returns matching metadata for the WSI.

The implemented Source can then create a instance from the implemented ImageData (and a method returning a WsiDataset):

```python
image_data = MyImageData('path_to_image_file')
dataset = create_dataset_from_image_data(image_data)
instance = WsiInstance(dataset, image_data)
```

The source should arrange the created instances and return them at the level_instances, label_instances, and overview_instances properties. WsiDicom can then open the source object and arrange the instances into levels etc as described in 'Data structure'.

## Other DICOM python tools

- [pydicom](https://pydicom.github.io/)
- [highdicom](https://github.com/MGHComputationalPathology/highdicom)
- [wsidicomizer](https://github.com/imi-bigpicture/wsidicomizer)
- [dicomslide](https://github.com/ImagingDataCommons/dicomslide)

## Contributing

We welcome any contributions to help improve this tool for the WSI DICOM community!

We recommend first creating an issue before creating potential contributions to check that the contribution is in line with the goals of the project. To submit your contribution, please issue a pull request on the imi-bigpicture/wsidicom repository with your changes for review.

Our aim is to provide constructive and positive code reviews for all submissions. The project relies on gradual typing and roughly follows PEP8. However, we are not dogmatic. Most important is that the code is easy to read and understand.

## Acknowledgement

*wsidicom*: Copyright 2021 Sectra AB, licensed under Apache 2.0.

This project is part of a project that has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 945358. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA. IMI website: <www.imi.europa.eu>
