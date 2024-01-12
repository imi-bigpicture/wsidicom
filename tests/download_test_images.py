#    Copyright 2022 SECTRA AB
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

import os
from ftplib import FTP
from hashlib import md5
from pathlib import Path
from typing import Any, Dict

FILESERVER = "medical.nema.org"
FILESERVER_SLIDE_PATH = Path("MEDICAL/Dicom/DataSets/WG26")

SLIDES: Dict[str, Dict[str, Any]] = {
    "FULL_WITH_BOT": {
        "name": r"Histech^Samantha [1229631]",
        "parentpath": r"WG26Demo2020_PV",
        "subpath": r"20190104 140000 [Case S - Colon polyps]/Series 000 [SM]",
        "files": {
            "2.25.173648596820938096199028939965251554503.dcm": "865538d55fce37ae6d91d85aebe29029",  # NOQA
            "2.25.181487944453580109633363498147571426374.dcm": "7ff4acf3c71572236ce968e06e79d8db",  # NOQA
            "2.25.191907898033754830752233761154920949936.dcm": "596183ec0444fedaba981de2f94652cb",  # NOQA
            "2.25.209236321826383427842899333369775338594.dcm": "f7fe907553f036ec6d2a68443f006fd4",  # NOQA
            "2.25.222943316513786317622687980099326639180.dcm": "ec1db6ca69c7d6fe8c4dacb04051ff11",  # NOQA
            "2.25.251277550657103455222613143487830679046.dcm": "03399b8332c967a9a97e67bffdd70fb9",  # NOQA
            "2.25.253973508129488054885063915385651983009.dcm": "ac85a6f618ed0ca2bdf921be429fa185",  # NOQA
            "2.25.259312801889857550164526960213815274816.dcm": "b7c64d1ed975b42f2b1b4d917c4ba8c0",  # NOQA
            "2.25.264278491200307498225194538752823371217.dcm": "0cc906dbeb22e10bff65f9b6d8961fa7",  # NOQA
            "2.25.290884199110265475119989552423006928136.dcm": "b8faf60dc44e4cb9aa5e7ab92a706b88",  # NOQA
            "2.25.315427219625170954090644500893748466389.dcm": "84cca002af2e6263f25913b2896e36db",  # NOQA
            "2.25.339652625381363485545547336547695948130.dcm": "52ea8ec41f3f86368d50542c9ed41975",  # NOQA
            "2.25.98447601926716844575918590187306802549.dcm": "382e1b6780404efb7f44c65444694b05",  # NOQA
        },
    },
    "SPARSE_NO_BOT": {
        "name": r"MoticWangjie^Professer [100001]",
        "parentpath": r"WG26Demo2019_PV",
        "subpath": r"20190903 102029 [200001]\Series 000 [SM]",
        "files": {
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753444.778.dcm": "4463e99bea080b14591f0665a6e84559",  # NOQA
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753447.783.dcm": "0b48b76e46c754fc75b13d6fbf682a19",  # NOQA
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753450.788.dcm": "75452d5f811f5c8c8f4475fbac6665bb",  # NOQA
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753451.793.dcm": "3bf4fee344571bda7005d9735b5bd699",  # NOQA
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753451.798.dcm": "84f2b3ba111ec09beb8846d17ddcd9e5",  # NOQA
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753451.803.dcm": "d2e565d600a573685ca1be48d96a0ef4",  # NOQA
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753451.808.dcm": "aedbe1f7ba9f15754078e3c8a9077fee",  # NOQA
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753451.813.dcm": "3533d30ef5e9abe3b8ee6e1c7bc0fb17",  # NOQA
        },
    },
}


def cwd_to_folder(ftp, folder: Path):
    ftp.cwd("/")
    for subfolder in folder.parts:
        ftp.cwd(str(subfolder))


def download_file(ftp: FTP, file: str, filename: Path):
    with open(filename, "wb") as fp:
        ftp.retrbinary(f"RETR {file}", fp.write)


def get_slide_dir() -> Path:
    DEFAULT_DIR = "tests/testdata"
    SLIDE_DIR = "slides"
    test_data_path = os.environ.get("DICOM_TESTDIR")
    if test_data_path is None:
        test_data_dir = Path(DEFAULT_DIR)
        print(
            "Env 'DICOM_TESTDIR' not set, downloading to default folder "
            f"{test_data_dir}."
        )
    else:
        test_data_dir = Path(test_data_path)
        print(f"Downloading to {test_data_dir}")
    return test_data_dir.joinpath(SLIDE_DIR)


def get_or_check_slide(slide_dir: Path, slide: Dict[str, Any], ftp: FTP):
    path = slide_dir.joinpath(slide["name"], slide["subpath"])
    ftp_path = FILESERVER_SLIDE_PATH.joinpath(
        slide["parentpath"], slide["name"], slide["subpath"]
    )
    os.makedirs(path, exist_ok=True)
    cwd_to_folder(ftp, ftp_path)
    for file, checksum in slide["files"].items():
        file_path = path.joinpath(file)
        if not file_path.exists():
            print(
                f"{file} not found, downloading from "
                f"{ftp_path.joinpath(file).as_posix()}"
            )
            download_file(ftp, file, file_path)
        else:
            print(f"{file} found, skipping download")
        check_checksum(file_path, checksum)


def check_checksum(file_path: Path, checksum: str):
    with open(file_path, "rb") as saved_file:
        data = saved_file.read()
        file_checksum = md5(data).hexdigest()
        if checksum != file_checksum:
            raise ValueError(f"Checksum failed for {file_path}")
        else:
            print(f"{file_path} checksum OK")


def main():
    print("Downloading and/or checking testdata from nema.org.")
    slide_dir = get_slide_dir()
    with FTP(FILESERVER) as ftp:
        ftp.login()
        for slide in SLIDES.values():
            get_or_check_slide(slide_dir, slide, ftp)


if __name__ == "__main__":
    main()
