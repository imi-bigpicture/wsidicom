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
import re
from dataclasses import dataclass
from ftplib import Error, error_perm
from hashlib import sha256
from pathlib import Path
from typing import Dict, Sequence

from fsspec import register_implementation
from fsspec.core import url_to_fs
from fsspec.implementations.ftp import FTPFileSystem


@dataclass
class TestSlideDefinition:
    remote_path: str
    local_path: str
    files: Dict[str, str]


SLIDES: Dict[str, TestSlideDefinition] = {
    "FULL_WITH_BOT": TestSlideDefinition(
        remote_path="ftp://medical.nema.org/MEDICAL/Dicom/DataSets/WG26/WG26Demo2020_PV/Histech^Samantha [1229631]/20190104 140000 [Case S - Colon polyps]/Series 000 [SM]",
        local_path="WG26Demo2020_PV/Histech^Samantha [1229631]/20190104 140000 [Case S - Colon polyps]/Series 000 [SM]",
        files={
            "2.25.173648596820938096199028939965251554503.dcm": "46a8e4d9200dbe6d9033c436021717cd8ecfc98c6adaa9c0af5cf154ae5a7258",
            "2.25.181487944453580109633363498147571426374.dcm": "6db30229f5457a8b7c5fc894e94f24300c823ca2c2825d7d97bf5384b63608ef",
            "2.25.191907898033754830752233761154920949936.dcm": "7efcc19596eb280d52ff933caa25503ea60fb6e4bb36d367337e439d092b37e9",
            "2.25.209236321826383427842899333369775338594.dcm": "4441f3dd8cdcd847870c6a589a1564843a257247ded3c14326751656592efaeb",
            "2.25.222943316513786317622687980099326639180.dcm": "1db99f47bbee8b6ea444581b529b40148e1d4b03c07cee220e99c82bef1c7055",
            "2.25.251277550657103455222613143487830679046.dcm": "1e525bd14212490029d4d511ac421fc7cc5788a8c4b76694eca6a889a89ba07f",
            "2.25.253973508129488054885063915385651983009.dcm": "9e4edefcb74325dee2f0f712eadfb8a66279d408285efbf95e2121fe2505b55e",
            "2.25.259312801889857550164526960213815274816.dcm": "228779f94e5d9f173334064b6389b54e530f9332b96873415f3a79ece1270aef",
            "2.25.264278491200307498225194538752823371217.dcm": "125413fbadaac9f171975c09d9f63d07868fd227dba31f5ec5e92e0e6a674b8a",
            "2.25.290884199110265475119989552423006928136.dcm": "f7df6a4b58e9b5842ce4ab52339f5ba7fddf1ad1536a2fa91ae49f21e10e86fa",
            "2.25.315427219625170954090644500893748466389.dcm": "4e5be925331d2cce7171a311d84417fad0674d1dc44680524bd35049c8904662",
            "2.25.339652625381363485545547336547695948130.dcm": "7ba9872adbd32f27839ffa68aac3155451acd6586a6149da119547aa7a797cc3",
            "2.25.98447601926716844575918590187306802549.dcm": "c8f1d0243e570aac30af3c97066bc488b82bd5e334ebdae3577ea5f8db172498",
        },
    ),
    "SPARSE_NO_BOT": TestSlideDefinition(
        remote_path="ftp://medical.nema.org/MEDICAL/Dicom/DataSets/WG26/WG26Demo2019_PV/MoticWangjie^Professer [100001]/20190903 102029 [200001]/Series 000 [SM]",
        local_path="WG26Demo2019_PV/MoticWangjie^Professer [100001]/20190903 102029 [200001]/Series 000 [SM]",
        files={
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753444.778.dcm": "c702db8afea7a18628f21735f7335a6cf3c4e355f7e4fa28db81b0d205ad2799",
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753447.783.dcm": "ad08c959f1e2a6c84676e832274330d6c3a49ce3a1224765469d1de72496b91c",
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753450.788.dcm": "343df2b55a460debeafc02498df70f206b4f13f1b5eacf3f7f76357cd9a3deeb",
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753451.793.dcm": "27880e6d6cf5e0dc4f3d9d913a5881c4c9f65fa8eadab50c3c81a1575c2d60f6",
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753451.798.dcm": "a1d6e4db64dbfbc1805cd9c1c2660312e3c498d2835a87648c276a67488c83de",
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753451.803.dcm": "944c35ac4ac78ebb6119196ed76a5648e6787df3a613257bebbca318f90dba5a",
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753451.808.dcm": "746e14855490cc5f867e07e83a1b32ed8af61346e853d242f1c015f80e7f7371",
            "1.2.276.0.7230010.3.1.4.1145362585.2096.1567753451.813.dcm": "1da91803ab1fca3de83a12cc6ccd3d29263d2d1ddbb81c3953b26077e1ab09fc",
        },
    ),
    "3DHisthech-1": TestSlideDefinition(
        remote_path="zip::https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/3DHISTECH-1.zip",
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
        remote_path="zip::https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/CMU-1-JP2K-33005.zip",
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
        remote_path="zip::https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/JP2K-33003-1.zip",
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
        remote_path="zip::https://openslide.cs.cmu.edu/download/openslide-testdata/DICOM/Leica-4.zip",
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
}


class PatchedFTPFileSystem(FTPFileSystem):
    """Patched to handle case where `dir` does not return a "ls -l" style response."""

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        out = []
        if path not in self.dircache:
            try:
                try:
                    out = [
                        (fn, details)
                        for (fn, details) in self.ftp.mlsd(path)
                        if fn not in [".", ".."]
                        and details["type"] not in ["pdir", "cdir"]
                    ]
                except error_perm:
                    out = self._mlsd2(path)  # Not platform independent
                for fn, details in out:
                    details["name"] = "/".join(
                        ["" if path == "/" else path, fn.lstrip("/")]
                    )
                    if details["type"] == "file":
                        details["size"] = int(details["size"])  # type: ignore
                    else:
                        details["size"] = 0  # type: ignore
                    if details["type"] == "dir":
                        details["type"] = "directory"
                self.dircache[path] = out
            except Error:
                try:
                    info = self.info(path)
                    if info["type"] == "file":  # type: ignore
                        out = [(path, info)]
                except (Error, IndexError) as exc:
                    raise FileNotFoundError(path) from exc
        files = self.dircache.get(path, out)
        if not detail:
            return sorted([fn for fn, details in files])
        return [details for fn, details in files]

    def _mlsd2(self, path="."):
        """
        Patch of the original _mlsd2 function to be platform to handle case where
        `dir` does not return a "ls -l" style response.
        """
        re_linux = re.compile(r"^[0-9]{2}-[0-9]{2}}-[0-9]{2}$")
        re_patched = re.compile(
            r"^([0-9]{2}-[0-9]{2}-[0-9]{2})  ([0-9]{2}:[0-9]{2}[A|P]M) +([0-9]+) (.*)"
        )

        def parse_linux(split_line: Sequence[str]):
            this = (
                split_line[-1],
                {
                    "modify": " ".join(split_line[5:8]),
                    "unix.owner": split_line[2],
                    "unix.group": split_line[3],
                    "unix.mode": split_line[0],
                    "size": split_line[4],
                },
            )
            if "d" == this[1]["unix.mode"][0]:
                this[1]["type"] = "dir"
            else:
                this[1]["type"] = "file"
            return this

        def parse_patched(split_line: Sequence[str]):
            this = (
                " ".join(split_line[3:]),
                {
                    "modify": " ".join(split_line[0:1]),
                },
            )
            if split_line[2] == "<DIR>":
                this[1]["type"] = "dir"
            else:
                this[1]["type"] = "file"
                this[1]["size"] = split_line[2]
            return this

        lines = []
        minfo = []
        self.ftp.dir(path, lines.append)
        for line in lines:
            split_line = line.split()
            if re_linux.match(split_line[0]) and len(split_line) > 8:
                this = parse_linux(split_line)
            elif re_patched.match(line):
                this = parse_patched(split_line)
            else:
                continue
            minfo.append(this)
        return minfo


register_implementation("ftp", PatchedFTPFileSystem, clobber=True)


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


def main():
    print("Downloading and/or checking testdata.")
    slide_dir = get_slide_dir()
    for slide in SLIDES.values():
        full_local_path = slide_dir.joinpath(slide.local_path)
        os.makedirs(full_local_path, exist_ok=True)
        for file, checksum in slide.files.items():
            full_local_file_path = full_local_path.joinpath(file)
            if not full_local_file_path.exists():
                fs, path = url_to_fs(slide.remote_path)  #
                full_remote_file_path = path + "/" + file
                with fs.open(full_remote_file_path, mode="rb") as remote_file:
                    with open(full_local_file_path, "wb") as local_file:
                        local_file.write(remote_file.read())
            check_checksum(full_local_file_path, checksum)


if __name__ == "__main__":
    main()
