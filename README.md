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

*wsidicom* uses pydicom, numpy, Pillow, marshmallow, fsspec, universal-pathlib, and dicomweb-client. Imagecodecs, pylibjpeg-rle, pyjpegls, and pylibjpeg-openjpeg can be installed as optionals to support additional transfer syntaxes.

## Limitations

- Levels are required to have (close to) 2 factor scale and same tile size.

- Color images must be 8 bits per sample; grayscale images may be 8 or 16 bits.
  16-bit color is not fundamentally unsupported, but no example WSI has been
  found to test and validate it against — please open an issue with a sample if
  you have one.

- Pixel data must be unsigned. The DICOM WSI IOD mandates this
  (`PixelRepresentation` 0000H); signed pixels are non-conformant.

- Color pixel data must be stored interleaved (`PlanarConfiguration` 0).
  Non-interleaved (planar) color only occurs in native uncompressed data and is
  likely not produced by WSI scanners, so it is rejected rather than left
  untested.

- Only the photometric interpretations enumerated by the DICOM WSI IOD
  ([PS3.3 C.8.12.4](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_c.8.12.4.html))
  are supported: `MONOCHROME2`, `RGB`, and the `YBR_*` variants. `MONOCHROME1`
  is not part of that IOD and is not supported.

- Without optional dependencies, the following transfer syntaxes are supported:

  - JPEGBaseline8Bit
  - JPEG2000
  - JPEG2000Lossless
  - HTJ2K
  - HTJ2KLossless
  - HTJ2KLosslessRPCL
  - ImplicitVRLittleEndian
  - ExplicitVRLittleEndian
  - ExplicitVRBigEndian

- With imagecodecs, the following transfer syntaxes are additionally supported:

  - JPEGExtended12Bit
  - JPEGLossless
  - JPEGLosslessSV1
  - JPEGLSLossless
  - JPEGLSNearLossless
  - RLELossless

- With pylibjpeg-rle RLELossless is additionally supported.

- With pyjpegls JPEGLSLossless and JPEGLSNearLossless is additionally supported.

- Optical path identifiers needs to be unique across instances.

## Basic usage

***Load a WSI dataset from files in folder.***

```python
from wsidicom import WsiDicom
slide = WsiDicom.open("path_to_folder")
```

The `files` argument accepts either a path to a folder with DICOM WSI-files or a sequence of paths to DICOM WSI-files.

***Load a WSI from anywhere with [fsspec](https://filesystem-spec.readthedocs.io).***

`WsiDicom.open` takes any fsspec URL, so the same call reads local disk, cloud
storage, and chained/cached protocols. Backend options go through `file_options`:

```python
from wsidicom import WsiDicom

slide = WsiDicom.open("s3://bucket/key", file_options={"s3": {"anon": True}})
slide = WsiDicom.open("simplecache::s3://bucket/key")  # cache remote reads on disk
```

Reads are lazy and seekable: wsidicom pulls the offset tables and only the frames
it decodes, not the whole file — so opening a 5 GB slide over the network doesn't
download 5 GB.

***Read from a source fsspec doesn't know (a database blob, a vault, a custom store).***

Don't buffer the slide into memory — teach fsspec your store by implementing a
small filesystem (see fsspec's
[Developing new filesystems](https://filesystem-spec.readthedocs.io/en/latest/developer.html)).
The example below implements only what wsidicom needs to read a slide: `info` (to
locate the file and get its size), `_open` (one line, wiring up the file class),
and `_fetch_range` (the ranged read for just the bytes it needs):

```python
import fsspec
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from wsidicom import WsiDicom


class BlobFileSystem(AbstractFileSystem):
    protocol = "blob"

    def info(self, path, **kwargs):
        return {"name": path, "size": my_database.byte_size(path), "type": "file"}

    def _open(self, path, mode="rb", **kwargs):
        return BlobFile(self, path, mode, **kwargs)


class BlobFile(AbstractBufferedFile):
    def _fetch_range(self, start, end):
        return my_database.read_bytes(self.path, offset=start, length=end - start)


fsspec.register_implementation("blob", BlobFileSystem)
slide = WsiDicom.open("blob://slide-123")
```

As long as your store supports ranged reads (e.g. Postgres large objects, an
object store, any BLOB API with offset access), wsidicom reads only the bytes it
touches. If you point `open` at a folder-like URL rather than an explicit file,
also implement `ls` so it can enumerate the slide's parts.

***Or load a WSI dataset from a DICOMDIR.***

```python
from wsidicom import WsiDicom

slide = WsiDicom.open_dicomdir("path_to_dicom_dir")
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

Alternatively, if you have already created an instance of `dicomweb_client.DICOMwebClient`, that may be used to create the `WsiDicomWebClient` like so:

```python
dicomweb_client = DICOMwebClient("url")
client = WsiDicomWebClient(dicomweb_client)
```

Then proceed to call `WsiDicom.open_web()` with this as in the first example.

***Use as a context manager.***

```python
from wsidicom import WsiDicom
with WsiDicom.open("path_to_folder") as slide:
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

## API differences between WsiDicom and OpenSlide

The WsiDicom API is similar to OpenSlide, but with some important differences:

- In WsiDicom, the `open`-method (i.e. `WsiDicom.open()`) is used to open a folder with DICOM WSI files, while in OpenSlide a file is opened with the `__init__`-method (e.g. `OpenSlide()`).

- In WsiDicom the `location` parameter in `read_region` is relative to the specified `level`, while in OpenSlide it is relative to the base level.

- In WsiDicom the `level` parameter in `read_region` is the pyramid index, i.e. level 2 always the level with quarter the size of the base level. In OpenSlide it is the index in the list of available levels, and if pyramid levels are missing these will not correspond to pyramid indices.

Conversion between OpenSlide `location` and `level` parameters to WsiDicom can be performed:

```python
with WsiDicom.open("path_to_folder") as wsi:
    level = wsi.levels[openslide_level_index]
    x = openslide_x // 2**(level.level)
    y = openslide_y // 2**(level.level)

```

## Metadata

WsiDicom parses the DICOM metadata in the opened image into easy-to-use dataclasses, see `wsidicom\metadata`.

```python
with WsiDicom.open("path_to_folder") as wsi:
    metadata = wsi.metadata
```

The obtained `WsiMetadata` has child dataclass properties the resembelse the DICOM WSI modules (compare
with the [VL Whole Slide Microscopy Image CIOD](https://dicom.innolitics.com/ciods/vl-whole-slide-microscopy-image)):

- `study`: The study the slide is part of (study identifiers, study date and time, etc.).
- `series`: The series the slide is part of.
- `patient`: Patient information (name, identifier, etc.).
- `equipment`: Scanner information information.
- `optical_paths`: List of optical path descriptions used for imaging the slide.
- `slide`: Slide information, including slide identifier, stainings done on the slide, and samples placed on the slide, see details in [Slide information](#slide-information)
- `label`: Slide label information, such as label text.
- `image`: Image information, including acquisition datetime, pixel spacing, focus method, etc.
- `frame_of_reference_uid`: The unique identifier for the frame of reference for the image.
- `dimension_organization_uids`: List of dimension organization uids.

Note that not all DICOM attributes are represented in the defined metadata model. Instead the full ´pydicom´ Datasets can be accessed per level, for example:

```python
with WsiDicom.open("path_to_folder") as wsi:
    wsi.levels.base_level.datasets[0]
```

If you encounter that some important and/or useful attribute is missing from the model, please make an issue (see [Contributing](#contributing)).

### Slide information

The `Slide` information model models the `Specimen` module has the following properties:

- `identifier`: Identifier for the slide.
- `stainings`: List of stainings done on the slide. Note that the model assumes that the same stainings have been done on all the samples on the slide.
- `samples`: List of samples placed on the slide.

Note that that while the parsing of slide information is designed to be as flexible and permissive as possible, some datasets contains non-standard compliant `Specimen` modules that are (at least currently) not possible to parse. In such cases the `stainings` and `samples` property will be set to `None`. If you have a dataset with a `Specimen` module that you think should be parsable, please make an issue (see [Contributing](#contributing)).

#### SlideSample

Each sample is model with the `SlideSample` dataclass, which represents an item in the DICOM [`Specimen Description Sequence`](https://dicom.innolitics.com/ciods/vl-whole-slide-microscopy-image/specimen/00400560)

- `identifier`: Identifier of the sample.
- `anatomical_sites`: List of codes describing the primary anatomic structures of interest in the sample.
- `sampled_from`: The sampling (of another specimen) that was done to produce the sample (if known). If the sampled specimen also was produced through sampling, this property will give access to the full hierarchy of (known) specimens.
- `uid`: Unique identifier for the sample.
- `localization`: Description of the placement of the sample on the slide. Should be present if more than one sample is placed on the slide.
- `steps`: List of preparation steps performed on the sample.
- `short_description`: Short description of the sample (should not exceed 64 characters).
- `detailed_description`: Unlimited description of the sample.

#### Samplings

The optional `sampled_from` property can either be a `Sampling` or a `UnknownSampling`. Both of these specify a sampled `specimen`, with the difference that the `UnknownSampling` is used when the sampling conditions are not fully know. A `Sampling` is more detailed, and specifies the sampling `method` and optional properties such as sampling `date_time`, `description` and `location`.

#### Specimens

The `specimen` property of a `Sampling` or a `UnknownSampling` links to either a `Specimen` or a `Sample`. A `Specimen` has no known parents (e.g. could be the specimen extracted from a patient), while a `Sample` always is produced from one or more samplings of other `Specimen`s or `Sample`s. The samplings used to produce a `Sample` is given by its `sampled_from`-property. Both `Specimen` and `Sample` contain additional properties describing the specimen:

- identifier: Identifier of the specimen.
- type: Optional anatomic pathology specimen type code (e.g. "tissue specimen"). Should be a specimen type defined in [CID 8103](https://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_8103.html).
- steps: List of processing steps performed on the specimen.
- container: Optional container type code the specimen is placed in. Should be a container type defined in [CID 8101](https://dicom.nema.org/medical/dicom/current/output/chtml/part16/sect_CID_8101.html).

#### Processing and staining steps

The processing steps that can be performed on a sample are:

- `Sampling`: Sampling of the specimen in order to produce new specimen(s). The sampling `method` should be a method defined in [CID 8110](https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8110.html).
- `Collection`: Collection of a specimen from a body. This can only be done on a `Specimen`, i.e. not on a specimen produced by sampling. The collection `method` should be a method defined in [CID 8109](https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8109.html).
- `Processing`: Processing performed on the specimen. The processing `method` should be a method defined in [CID 8113](https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8113.html).
- `Embedding`: Embedding done on the specimen. The embedding medium should be a medium defined in [CID 8115](https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8115.html).
- `Fixation`: Fixation of the specimen. The fixative should be a fixative defined in [CID 8114](https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8114.html).
- `Receiving`: Receiving of the specimen.
- `Storage`: Storage of the specimen.

The `Staining`(s) for a `Slide` contains a list of substances used for staining. The substances used should defined in [CID 8112](https://dicom.nema.org/medical/Dicom/current/output/chtml/part16/sect_CID_8112.html).

Every processing step (including staining) also have the optional properties `date_time` for when the processing was done and `description` for a textual description of the processing.

These steps are parsed from the `SpecimenPreparationSequence` following [`TID 8004`](https://dicom.nema.org/medical/dicom/current/output/chtml/part16/chapter_C.html#sect_TID_8004) for each specimen identifier in the item sequence.

### Exporting to json

The metadata can be exported to json:

```python
from wsidicom.metadata.schema.json import WsiMetadataJsonSchema

with WsiDicom.open("path_to_folder") as wsi:
    metadata = wsi.metadata

schema = WsiMetadataJsonSchema()
metadata_json = schema.dump(metadata)
```

### Settings

The strictness of parsing of DICOM WSI metadata can be configured using the following settings (see [Settings](#settings)):

- `strict_specimen_identifier_check`: Controls how to handle matching between specimen identifiers if one of the identifiers have a issuer of identifier set and the other does not. If `True` the identifiers are considered equal (provided that the identifier value is the same), if `False` the issuer of identifier must always also match. This setting is useful if for example a issuer of identifier is specified in the `Specimen Description Sequence` but steps in the `Specimen Preparation Sequence` lacks the issuer of identifier. The default value is `True`.
- `ignore_specimen_preparation_step_on_validation_error`: Controls how to handle if a step in the `Specimen Preparation Sequence` fails to validate. If `True`, only steps that fails will be ignored. If `False` all steps will be ignored. The default value is `True`.

## Saving files

An opened WsiDicom instance can be saved to a new path using the save()-method. The produced files will be:

- Fully tiled. Any sparse tiles will be replaced with a blank tile with color depending on the photometric interpretation.
- Have a basic offset table (or optionally an extended offset table or no offset table).
- Not be concatenated.

By default frames are copied as-is, i.e. without re-compression.

```python
with WsiDicom.open("path_to_folder") as slide:
    slide.save("path_to_output")
```

The output folder must already exists. Be careful to specify a unique folder folder to avoid mixing files from different images.

Optionally frames can be transcoded, either by a encoder setting or an encoder:

```python
from wsidicom.codec import JpegSettings

with WsiDicom.open("path_to_folder") as slide:
    slide.save("path_to_output", transcoding=JpegSettings())
```

## Settings

Settings are held in the immutable `Settings` object. To change the process-wide default, build a `Settings` and pass it to `set_default_settings`:

```python
from wsidicom import Settings, set_default_settings

set_default_settings(Settings(strict_uid_check=True))
```

To configure a single opened slide instead of the whole process, pass a `Settings` as the keyword-only `settings` argument to `open` (also available on `open_dicomdir` and `open_web`):

```python
from wsidicom import WsiDicom, Settings

slide = WsiDicom.open("path_to_files", settings=Settings(strict_uid_check=True))
```

The given settings stay in effect for that slide's reads and saves, while the default is used for everything else. To apply settings to a block of your own code, use the `use_settings` context manager:

```python
from wsidicom import Settings, use_settings

with use_settings(Settings(resampling_filter="box")):
    ...  # code here (and threads it starts) sees these settings
```

Read the settings in effect with `get_settings()`.

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
slide = WsiDicom.open("path_to_folder")
```

***Create a point annotation at x=10.0, y=20.0 mm.***

```python
from wsidicom import Annotation, Point
point_annotation = Annotation(Point(10.0, 20.0))
```

***Create a point annotation with a measurement.***

```python
from wsidicom import conceptcode, Measurement
# A measurement is defined by a type code ('Area'), a value (25.0) and a unit code ('Pixels).
area = ConceptCode.MeasurementCode("Area")
pixels = ConceptCode.UnitCode("Pixels")
measurement = Measurement(area, 25.0, pixels)
point_annotation_with_measurement = Annotation(Point(10.0, 20.0), [measurement])
```

***Create a group of the annotations.***

```python
from wsidicom import PointAnnotationGroup
# The 222 supplement requires groups to have a label, a category and a type
group = PointAnnotationGroup(
    annotations=[point_annotation, point_annotation_with_measurement],
    label='group label',
    category_code=ConceptCode.AnnotationCategoryCode('Tissue'),
    type_code=ConceptCode.AnnotationTypeCode('Nucleus'),
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
slide = WsiDicom.open("path_to_folder")
annotations = slide.annotations
```

## Setup environment for development

Requires uv installed.

```console
git clone https://github.com/imi-bigpicture/wsidicom.git
uv sync --all-extras
```

To watch unit tests use:

```console
uv run pytest-watch -- -m unittest
```

The integration tests uses test images from nema.org that's needs to be downloaded. The location of the test images can be changed from the default tests\testdata\slides using the environment variable WSIDICOM_TESTDIR. Download the images using the supplied script:

```console
python .\tests\download_test_images.py
```

If the files are already downloaded the script will validate the checksums.

To run integration tests:

```console
uv run pytest -m integration
```

## Data structure

A WSI DICOM pyramid is in *wsidicom* represented by a hierarchy of objects of different classes, starting from bottom:

- *WsiDicomReader*, represents a WSI DICOM file reader, used for accessing WsiDicomFileImageData and WsiDataset.
- *WsiDicomFileImageData*, represents the image data in one or several (in case of concatenation) WSI DICOM files.
- *WsiDataset*, represents the image metadata in one or several (in case of concatenation) WSI DICOM files.
- *WsiInstance*, represents image data and image metadata.
- *Level*, represents a group of instances with the same image size, i.e. of the same level.
- *Pyramid*, represents a group of levels, i.e. the pyrimidal structure.
- *Pyramids*, represents a collection of pyramids, each with different image coordinate system or extended depth of field.
- *WsiDicom*, represents a collection of pyramids, labels and overviews.

Labels and overviews are structured similarly to levels, but with somewhat different properties and restrictions. For DICOMWeb the WsiDicomFile\* classes are replaced with WsiDicomWeb\* classes.

A Source is used to create WsiInstances, either from files (*WsiDicomFileSource*) or DICOMWeb (*WsiDicomWebSource*), and can be used to to Initiate a *WsiDicom* object. A source is easiest created with the open() and open_web() helper functions, e.g.:

```python
slide = WsiDicom.open("path_to_folder")
```

## Code structure

- [codec](wsidicom/codec) - Encoders and decoders for image pixel data.
- [file](wsidicom/file) - Implementation for reading and writing DICOM WSI files.
- [group](wsidicom/group) - Group implementations, e.g. Level.
- [instance](wsidicom/instance) - Instance implementations WsiInstance and WsiDataset, the metaclass ImageData and ImageData implementations WsiDicomImageData and NumpyImageData.
- [metadata](wsidicom/metadata) - Metadata models and schema for serializing and deserializing to DICOM and json.
- [series](wsidicom/series) - Series implementations Levels, Labels, and Overview.
- [web](wsidicom/web) - Implementation for reading DICOM WSI from DICOMWeb.
- [writing](wsidicom/writing) - Pipeline for writing pyramids (tile reading, downsampling, encoding, and tile accumulation).
- [cache.py](wsidicom/cache.py) - Per-instance LRU caching of method return values.
- [conceptcode.py](wsidicom/conceptcode.py) - Handling of DICOM concept codes.
- [config.py](wsidicom/config.py) - Handles configuration settings.
- [downsampler.py](wsidicom/downsampler.py) - Downsampling of image pixels.
- [errors.py](wsidicom/errors.py) - Custom errors.
- [geometry.py](wsidicom/geometry.py) - Classes for geometry handling.
- [graphical_annotations](wsidicom/graphical_annotations.py) - Handling graphical annotations.
- [options.py](wsidicom/options.py) - Option enums for settings.
- [source.py](wsidicom/source.py) - Metaclass Source for serving WsiInstances to WsiDicom.
- [stitcher.py](wsidicom/stitcher.py) - Assembling a grid of tile pixels into one array.
- [stringprinting.py](wsidicom/stringprinting.py) - For nicer string printing of objects.
- [tags.py](wsidicom/tags.py) - Definition of commonly used DICOM tags.
- [target.py](wsidicom/target.py) - Metaclass Target for saving WsiDicom.
- [thread.py](wsidicom/thread.py) - Read executor and cancelable queues for the read and write pipelines.
- [uid.py](wsidicom/uid.py) - Handles DICOM uids.
- [wsidicom.py](wsidicom/wsidicom.py) - Main class with methods to open DICOM WSI objects.

## Pixel pipeline

Pillow was used throughout the pixel pipeline from version 0.1.0, as it was the go-to Python imaging library at the time. As the image-processing ecosystem matured, it became clear that the pixel operations — decoding, stitching tiles, and downsampling — run faster with numpy than with Pillow. Further, converting between the two is not free, so a pipeline that mixes them pays for repeated round-trips.

From version 0.34.0 the pipeline handles pixels as numpy arrays throughout for best performance. For backwards compatibility, a Pillow image is still derived at the `WsiDicom` read boundary, at the cost of a single conversion. Passing `as_array=True` to a read returns the numpy array directly instead of a Pillow image, skipping even that conversion. The write path (`save()`) works entirely in numpy.

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
- [openslide-python](https://openslide.org/api/python/)

## Contributing

We welcome any contributions to help improve this tool for the WSI DICOM community!

We recommend first creating an issue before creating potential contributions to check that the contribution is in line with the goals of the project. To submit your contribution, please issue a pull request on the [imi-bigpicture/wsidicom repository](https://github.com/imi-bigpicture/wsidicom) with your changes for review.

Our aim is to provide constructive and positive code reviews for all submissions. The project relies on gradual typing and roughly follows PEP8. However, we are not dogmatic. Most important is that the code is easy to read and understand.

## Acknowledgement

*wsidicom*: Copyright 2021 Sectra AB, licensed under Apache 2.0.

This project is part of a project that has received funding from the Innovative Medicines Initiative 2 Joint Undertaking under grant agreement No 945358. This Joint Undertaking receives support from the European Union’s Horizon 2020 research and innovation programme and EFPIA. IMI website: <www.imi.europa.eu>
