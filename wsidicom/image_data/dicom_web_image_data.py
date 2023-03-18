import io
from functools import cached_property
from pathlib import Path
from typing import List, Optional

from PIL import Image
from PIL.Image import Image as PILImage
from pydicom.uid import UID

from wsidicom.dataset import TileType
from wsidicom.geometry import Point, Size, SizeMm
from wsidicom.image_data.dicom_image_data import FullTileIndex, SparseTileIndex
from wsidicom.image_data.image_data import ImageData, ImageOrigin
from wsidicom.web.web import DicomWebClient, WsiDicomWeb


class DicomWebImageData(ImageData):
    def __init__(
        self, client: DicomWebClient, study_uid: UID, series_uid: UID, instance_uid: UID
    ):
        self._instance = WsiDicomWeb(client, study_uid, series_uid, instance_uid)
        if self._instance.dataset.tile_type == TileType.FULL:
            self._tile_index = FullTileIndex([self._instance.dataset])
        else:
            self._tile_index = SparseTileIndex([self._instance.dataset])

    @property
    def files(self) -> List[Path]:
        return []

    @property
    def transfer_syntax(self) -> UID:
        return self._instance.transfer_syntax

    @property
    def image_size(self) -> Size:
        return self._instance.dataset.image_size

    @property
    def tile_size(self) -> Size:
        """Should return the pixel tile size of the image, or pixel size of
        the image if not tiled."""
        return self._instance.dataset.tile_size

    @property
    def pixel_spacing(self) -> Optional[SizeMm]:
        """Should return the size of the pixels in mm/pixel."""
        return self._instance.dataset.pixel_spacing

    @property
    def samples_per_pixel(self) -> int:
        """Should return number of samples per pixel (e.g. 3 for RGB."""
        return self._instance.dataset.samples_per_pixel

    @property
    def photometric_interpretation(self) -> str:
        """Should return the photophotometric interpretation of the image
        data."""
        return self._instance.dataset.photometric_interpretation

    @cached_property
    def image_origin(self) -> ImageOrigin:
        """Should return the image origin of the image data."""
        return ImageOrigin.from_dataset(self._instance.dataset)

    def _get_decoded_tile(self, tile: Point, z: float, path: str) -> PILImage:
        """Should return Image for tile defined by tile (x, y), z,
        and optical path."""
        frame = self._get_tile(tile, z, path)
        return Image.open(io.BytesIO(frame))

    def _get_encoded_tile(self, tile: Point, z: float, path: str) -> bytes:
        """Should return image bytes for tile defined by tile (x, y), z,
        and optical path."""
        return self._get_tile(tile, z, path)

    def close(self) -> None:
        """Should close any open files."""
        raise NotImplementedError()

    def _get_tile(self, tile: Point, z: float, path: str) -> bytes:
        tile_index = self._tile_index.get_frame_index(tile, z, path)
        return self._instance.get_tile(tile_index)
