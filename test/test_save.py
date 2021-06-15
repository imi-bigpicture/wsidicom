import glob
import io
import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Sequence, Tuple, TypedDict

from pydicom.dataset import Dataset

import pytest
from PIL import Image, ImageChops
from wsidicom.interface import WsiDicom

wsidicom_test_data_dir = os.environ.get("WSIDICOM_TESTDIR", "C:/temp/wsidicom")
sub_data_dir = "interface"
data_dir = wsidicom_test_data_dir + '/' + sub_data_dir


class WsiFolder(TypedDict):
    path: Path
    wsi_dicom: WsiDicom


@pytest.mark.integration
class WsiDicomSaveTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_folders: Dict[
            Path,
            Tuple[WsiDicom, WsiDicom, Path]
        ]

    @classmethod
    def setUpClass(cls):
        cls.test_folders = {}
        folders = cls._get_folders()
        for folder in folders:
            cls.test_folders[folder] = cls.open(folder)

    @classmethod
    def tearDownClass(cls):
        for (original, restored, tempdir) in cls.test_folders.values():
            original.close()
            restored.close()
            tempdir.cleanup()

    @staticmethod
    def open(path: Path) -> WsiDicom:
        folder = Path(path).joinpath("dcm")
        original = WsiDicom.open(str(folder))
        tempdir = TemporaryDirectory()
        original.save(Path(tempdir.name))
        restored = WsiDicom.open(str(tempdir.name))
        return (original, restored, tempdir)

    @classmethod
    def _get_folders(cls):
        return [
            Path(data_dir).joinpath(item)
            for item in os.listdir(data_dir)
        ]

    def test_read_region(self):
        for folder, (original, restored, tempdir) in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_region/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)

                im = restored.read_region(
                    (region["location"]["x"], region["location"]["y"]),
                    region["level"],
                    (region["size"]["width"], region["size"]["height"])
                )

                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_read_region_mm(self):
        for folder, (original, restored, tempdir) in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_region_mm/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)

                im = restored.read_region_mm(
                    (region["location"]["x"], region["location"]["y"]),
                    region["level"],
                    (region["size"]["width"], region["size"]["height"])
                )

                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_read_region_mpp(self):
        for folder, (original, restored, tempdir) in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_region_mpp/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)

                im = restored.read_region_mpp(
                    (region["location"]["x"], region["location"]["y"]),
                    region["mpp"],
                    (region["size"]["width"], region["size"]["height"])
                )

                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_read_tile(self):
        for folder, (original, restored, tempdir) in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_tile/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)

                im = restored.read_tile(
                    region["level"],
                    (region["location"]["x"], region["location"]["y"])
                )
                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_read_encoded_tile(self):
        for folder, (original, restored, tempdir) in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_encoded_tile/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)
                tile = restored.read_encoded_tile(
                    region["level"],
                    (region["location"]["x"], region["location"]["y"])
                )
                im = Image.open(io.BytesIO(tile))
                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_read_thumbnail(self):
        for folder, (original, restored, tempdir) in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_thumbnail/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)
                im = restored.read_thumbnail(
                    (region["size"]["width"], region["size"]["height"])
                )
                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_optical_module(self):
        for folder, (original, restored, tempdir) in self.test_folders.items():
            original_optical_ds = original.optical.insert_into_ds(Dataset())
            restored_optical_ds = restored.optical.insert_into_ds(Dataset())
            self.assertEqual(
                original_optical_ds.to_json_dict(),
                restored_optical_ds.to_json_dict()
            )
