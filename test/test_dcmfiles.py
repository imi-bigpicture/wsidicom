#    Copyright 2021 SECTRA AB
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import glob
import io
import json
import os
import unittest
from pathlib import Path
from typing import Dict

import pytest
from PIL import Image, ImageChops
from wsidicom import WsiDicom

wsidicom_test_data_dir = os.environ.get("WSIDICOM_TESTDIR", "C:/temp/wsidicom")
sub_data_dir = "interface"
data_dir = wsidicom_test_data_dir + '/' + sub_data_dir


@pytest.mark.integration
class WsiDicomFilesTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_folders: Dict[Path, WsiDicom]

    @classmethod
    def setUpClass(cls):
        cls.test_folders = {}
        folders = cls._get_folders()
        for folder in folders:
            cls.test_folders[folder] = cls.open(folder)

    @classmethod
    def tearDownClass(cls):
        for folder, wsi_dicom in cls.test_folders.items():
            wsi_dicom.close()

    @staticmethod
    def open(path: Path) -> WsiDicom:
        folder = Path(path).joinpath("dcm")
        return WsiDicom.open(str(folder))

    @classmethod
    def _get_folders(cls):
        return [
            Path(data_dir).joinpath(item)
            for item in os.listdir(data_dir)
        ]

    def test_read_region(self):
        for folder, wsi_dicom in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_region/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)

                im = wsi_dicom.read_region(
                    (region["location"]["x"], region["location"]["y"]),
                    region["level"],
                    (region["size"]["width"], region["size"]["height"])
                )

                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_read_region_mm(self):
        for folder, wsi_dicom in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_region_mm/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)

                im = wsi_dicom.read_region_mm(
                    (region["location"]["x"], region["location"]["y"]),
                    region["level"],
                    (region["size"]["width"], region["size"]["height"])
                )

                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_read_region_mpp(self):
        for folder, wsi_dicom in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_region_mpp/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)

                im = wsi_dicom.read_region_mpp(
                    (region["location"]["x"], region["location"]["y"]),
                    region["mpp"],
                    (region["size"]["width"], region["size"]["height"])
                )

                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_read_tile(self):
        for folder, wsi_dicom in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_tile/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)

                im = wsi_dicom.read_tile(
                    region["level"],
                    (region["location"]["x"], region["location"]["y"])
                )
                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_read_encoded_tile(self):
        for folder, wsi_dicom in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_encoded_tile/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)
                tile = wsi_dicom.read_encoded_tile(
                    region["level"],
                    (region["location"]["x"], region["location"]["y"])
                )
                im = Image.open(io.BytesIO(tile))
                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)

    def test_read_thumbnail(self):
        for folder, wsi_dicom in self.test_folders.items():
            json_files = glob.glob(
                str(folder.absolute())+"/read_thumbnail/*.json")

            for json_file in json_files:
                with open(json_file, "rt") as f:
                    region = json.load(f)
                im = wsi_dicom.read_thumbnail(
                    (region["size"]["width"], region["size"]["height"])
                )
                expected_im = Image.open(Path(json_file).with_suffix(".png"))

                diff = ImageChops.difference(im, expected_im)

                bbox = diff.getbbox()
                self.assertIsNone(bbox, msg=json_file)
