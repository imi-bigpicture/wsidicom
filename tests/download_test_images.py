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
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.parse import quote

import requests

HF_DATASET = "erikgabr/wsi-testdata"
HF_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"


@dataclass
class TestSlideDefinition:
    remote_path: str
    local_path: str
    files: dict[str, str]


SLIDES: dict[str, TestSlideDefinition] = {
    "3DHisthech-1": TestSlideDefinition(
        remote_path=f"{HF_BASE}/dicom/3DHISTECH-1",
        local_path="3DHISTECH-1",
        files={
            "000001.dcm": "fc89d375687ae95dd387864d88b139c7364e8589ce972b2f2b39e5c424f39be5",
            "000002.dcm": "ba63085d29465710c52391f4cb6b5895e3765b2fada0c32d8d0e7faf0e68842c",
            "000003.dcm": "f631fe7d1817ae732aaf650ed575c73b505dbe3c1da12410cf6eba0c7439d95f",
            "000004.dcm": "4e62f7f3b3a71f3720e88904bbd35cc46aa2f1fb5bb458ddf95a876e83d33233",
            "000005.dcm": "6c81c78d0f0220397746ede19b354fd5ea8081af5f65ed01abb3c626c5780a96",
            "000006.dcm": "29cbc3b8f6cfb2c19ffbe1531051be8b459a49bded2ea6f1bf25503a41926e0b",
            "000007.dcm": "5d4a6c129d4fcbac824659bb301de15481a728bb632d650f4066a05488b202b6",
            "000008.dcm": "ad2591e09f61fe119dceec5c619d9dfdfc73eb196ff8b852c2661854a86555c2",
            "000009.dcm": "e12a5b4819dfb984c8701cdeec85a20ef27031d2593bba71023485ecb62f0925",
            "000010.dcm": "3219815e3d14c4d8fe694bc3602d7cd821133efdaf9adb2fbe7ce60e96c9c404",
            "000011.dcm": "5ea855d2eb66b0d11dbfa3d4517a60e72a30e2dbd63288602fa9852309a785ce",
            "000012.dcm": "853c4f226e1b3a17dc36ba0d6f021641534da8886158d2a1c047c4298c9f993e",
            "000013.dcm": "78dd739146225de7905b4b22c5c8f2489c9ca9686ab2627a042d95c36b166c2a",
            "000014.dcm": "2aad7d79f4ab131534a5604e458799e91ede727b0a62ced23939589844b8b6e5",
        },
    ),
    "CMU-1-JP2K-33005": TestSlideDefinition(
        remote_path=f"{HF_BASE}/dicom/CMU-1-JP2K-33005",
        local_path="CMU-1-JP2K-33005",
        files={
            "DCM_0.dcm": "db096bd0ddbd1d61962c03a8c0de9a1f305dc2f8523a7f48b94f22ff4b641d0a",
            "DCM_1.dcm": "7c6e2e114255cfbea9451aaad1ab88b19fbad4b5302538198e536d01b0cc15a2",
            "DCM_2.dcm": "4c0b00f5d3c147ed6b570751bd41a8b9ca57687aeefb4c7d08dc54a4a8789bdc",
            "DCM_3.dcm": "178f190f7fcfe8627f357876d0b253d191068175f6980f077151f35a4726a655",
            "DCM_4.dcm": "9020583673ad0b77e2b45091c83ec45cba4625a23e2c5ecdb0db769031de70a3",
            "DCM_5.dcm": "1f49b277fb9cf1e989a03bdf11b325e241c91025612259b49ec54d1ff1f98212",
        },
    ),
    "JP2K-33003-1": TestSlideDefinition(
        remote_path=f"{HF_BASE}/dicom/JP2K-33003-1",
        local_path="JP2K-33003-1",
        files={
            "DCM_0.dcm": "d33960d767bb6258be0d1d885acceb511818a355ebe4b538098dba79aff722e0",
            "DCM_1.dcm": "d63ccaa66fc4d565d5bfd846f82191af4204e9b162aea6dc311d353614f3210c",
            "DCM_2.dcm": "42184f289044c0050bd3331f074bf3a5690ad0158e98c437b72cdbf968549219",
            "DCM_3.dcm": "a4d93cc8b2425444730d9745582c57763794048478690b0c55ab7c3a36ccede0",
            "DCM_4.dcm": "5544f8e90ec2362d0f047a9a966d023219be87b27115ddc4fec62f12750e61d6",
            "DCM_5.dcm": "b95acc8ee1e330e21a01a1666a88f408b1d91aea3c4c20ce865e5de4ecb17668",
        },
    ),
    "Leica-4": TestSlideDefinition(
        remote_path=f"{HF_BASE}/dicom/Leica-4",
        local_path="Leica-4",
        files={
            "1.3.6.1.4.1.36533.116129230228107214763613716719238114924751.dcm": "8686256901b3403cb185a5c6253e1561b17cf1d13342a6bf8345c412d8c31c30",
            "1.3.6.1.4.1.36533.1881662823325113479691652532302192524914036.dcm": "aee44c12b51d76ca5e63de681dd86490c6ccc572507fcf32e04241de105f57ec",
            "1.3.6.1.4.1.36533.21773233891171386611617621819013191107166.dcm": "334407908d16d629d3f0b2c7e782fdf223bcc3761ab23ec8d810922d50e205fc",
            "1.3.6.1.4.1.36533.2391938919943337319712912711949255392271.dcm": "d88f2d7dac3cd21122f71acaf8efcb25dcd22d940745c3543e56833f0d687702",
            "1.3.6.1.4.1.36533.2411761230176195652241589819186191207215116.dcm": "953927614b4e7698f5fb8ff8a9da099010af205ed091e11c9ca3733db65f08c0",
            "1.3.6.1.4.1.36533.2642199142199497125516614013324167247234250.dcm": "25c901e3556d2f78a88127b536ed9e7c8909e8eb6358db0e0891b7fc3ee112d0",
        },
    ),
    "Hamamatsu-2": TestSlideDefinition(
        remote_path=f"{HF_BASE}/dicom/Hamamatsu-2",
        local_path="Hamamatsu-2",
        files={
            "A1 - 2026-05-29 10.08.06.1.dcm": "5a883359b4d684f1532ae0ca08357a2b9fcb3fa662b01ea39214a02284520e8b",
            "A1 - 2026-05-29 10.08.06.2.dcm": "81691b651af3ebc959c3ced10ca6fe197fa08357ceebfbdba4f235b7d63e6f2b",
            "A1 - 2026-05-29 10.08.06.3.dcm": "b6f28bdfc479511c2f385563c70aeb8252ffd26cded51a7564328d9b7bbaf0d4",
            "A1 - 2026-05-29 10.08.06.4.dcm": "426c6d34b146db00a06a361ec70c6d75ccfa53457341ba6098f2aa3e61c07f32",
            "A1 - 2026-05-29 10.08.06.5.dcm": "a4041247948d0ce392ab7092bc18ec6a443888fd68d09ebc1ff7faaab93403fb",
            "A1 - 2026-05-29 10.08.06.6.dcm": "c8ea3ccb3f089584065a8f8d249bb4ded38ecabaca3fb10c85bdf7ff9dbd6fa6",
            "A1 - 2026-05-29 10.08.06.7.dcm": "ddfbc65094ca5b73fe6cc6729b47a3dccf7990c6477fcbd442478c971814cc0d",
            "A1 - 2026-05-29 10.08.06.8.dcm": "b208c94c06beb0b95aa85969260a7d1b5493a949d1f5a5278f930e36bdba1ebc",
            "A1 - 2026-05-29 10.08.06_label.dcm": "ee484bc5752f64e80912b63c283c8e695f9ec00949c43485cb849a86286504da",
            "A1 - 2026-05-29 10.08.06_slide.dcm": "9bf5c2d18643c636c9f23e52d2caa5ce04f22bf7e2313b17640fd2f0e83b47de",
        },
    ),
    "Hamamatsu-Case-A": TestSlideDefinition(
        remote_path=f"{HF_BASE}/dicom/Hamamatsu-Case-A",
        local_path="Hamamatsu-Case-A",
        files={
            "Breast(Cancer)_HE_T-4um_(normal)_40x.1.dcm": "9bc15b0e5c06fd92e249c72b96f6a261dfd05debc78f112f5e99cf87a83b0833",
            "Breast(Cancer)_HE_T-4um_(normal)_40x.2.dcm": "3059b473d248739369e2c6273ba46215c561d94be9f8db29a720b050a0cd73bd",
            "Breast(Cancer)_HE_T-4um_(normal)_40x.3.dcm": "ed3bf09845b5929be693e1c01e09c28fe8fce3cb11b3c425c69a0fc4933b0bb0",
            "Breast(Cancer)_HE_T-4um_(normal)_40x.4.dcm": "4c92adb86e2817d0afe396efe149a4b3839db7e21f1b0201f325f5b532e8433f",
            "Breast(Cancer)_HE_T-4um_(normal)_40x.5.dcm": "ef8e23b6c245fe740fa0c749ea1e912b9db5077c992055758b7e359f0176e3c1",
            "Breast(Cancer)_HE_T-4um_(normal)_40x_label.dcm": "1141e9c1694bf604b3dfa6ae99fd2c3fa74f0825e78a2fb8a950fe24d23eb18f",
            "Breast(Cancer)_HE_T-4um_(normal)_40x_slide.dcm": "99e8dd49d8fd0f27d8c2e5e0005393d4db304bff083628ba8f13e6d491b48f3b",
        },
    ),
}


def get_slide_dir() -> Path:
    testdata_folder = os.environ.get("WSIDICOM_TESTDIR")
    DEFAULT_TESTDATA_FOLDER = "tests/testdata/slides"
    if testdata_folder is None:
        test_dir_folder = Path(DEFAULT_TESTDATA_FOLDER)
        print(
            "Env 'WSIDICOM_TESTDIR' not set, downloading to default folder "
            f"{test_dir_folder}."
        )
    else:
        test_dir_folder = Path(testdata_folder)
        print(f"Downloading to {test_dir_folder}")
    return test_dir_folder


def check_checksum(file_path: Path, checksum: str):
    sha_256 = sha256()
    with open(file_path, "rb") as file:
        while chunk := file.read(8092):
            sha_256.update(chunk)
        file_checksum = sha_256.hexdigest()
        if checksum != file_checksum:
            raise ValueError(
                f"Checksum failed for {file_path}, was {file_checksum} expected {checksum}"
            )
        else:
            print(f"{file_path} checksum OK")


def download_file(url: str, filename: Path):
    with requests.get(url, stream=True, timeout=30) as request:
        request.raise_for_status()
        with open(filename, "wb") as file:
            for chunk in request.iter_content(chunk_size=1024 * 1024):
                file.write(chunk)


def main():
    print("Downloading and/or checking testdata.")
    slide_dir = get_slide_dir()
    for slide in SLIDES.values():
        full_local_path = slide_dir.joinpath(slide.local_path)
        os.makedirs(full_local_path, exist_ok=True)
        for file, checksum in slide.files.items():
            full_local_file_path = full_local_path.joinpath(file)
            if not full_local_file_path.exists():
                download_file(
                    f"{slide.remote_path}/{quote(file)}", full_local_file_path
                )
            check_checksum(full_local_file_path, checksum)


if __name__ == "__main__":
    main()
