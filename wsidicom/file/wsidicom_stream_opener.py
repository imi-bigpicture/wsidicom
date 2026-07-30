#    Copyright 2024 SECTRA AB
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

"""Module for opening WsiDicomIO instances from streams or files."""

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Literal,
    cast,
)

from fsspec.spec import AbstractFileSystem
from pydicom.uid import UID
from upath import UPath

from wsidicom.file.io.wsidicom_io import WsiDicomIO
from wsidicom.paths import as_upath


class WsiDicomStreamOpener:
    def __init__(
        self,
        file_options: dict[str, Any] | None = None,
    ):
        """Create a WsiDicomStreamOpener.

        Parameters
        ----------
        file_options: dict[str, Any] | None = None
            Keyword arguments for opening filesystems.
        """

        self._file_options = file_options or {}

    def open(
        self,
        files: str | Path | UPath | Iterable[str | Path | UPath],
        sop_class_uids: UID | Sequence[UID] | None = None,
    ) -> Iterator[WsiDicomIO]:
        """Open DICOM streams in paths and return WsiDicomIO instances.

        Parameters
        ----------
        files: str | Path | UPath | Iterable[str | Path | UPath],
            Folder, single file, or sequence of files to open.
        sop_class_uids: UID | Sequence[UID] | None = None,
            SOP class uids to filter on.

        Returns
        -------
        Iterator[WsiDicomIO]
            Opened WsiDicomIO instances.
        """
        if isinstance(sop_class_uids, UID):
            sop_class_uids = [sop_class_uids]
        if isinstance(files, (str, Path, UPath)):
            files = [files]
        for file in files:
            for stream, filepath in self._open_streams(file, "rb"):
                try:
                    dicom_io = WsiDicomIO(stream, filepath=filepath)
                    if dicom_io.is_dicom and (
                        sop_class_uids is None
                        or dicom_io.media_storage_sop_class_uid in sop_class_uids
                    ):
                        yield dicom_io
                    else:
                        stream.close()
                except Exception:
                    stream.close()

    def open_for_writing(
        self,
        path: str | Path | UPath,
        mode: Literal["r+b"] | Literal["w+b"],
        transfer_syntax: UID,
    ) -> WsiDicomIO:
        """Open a stream for writing.

        Parameters
        ----------
        path: str | Path | UPath
            Path to open.
        mode: Literal["r+b"] | Literal["w+b"]
            Mode to open in.
        transfer_syntax: UID
            Transfer syntax to use.

        Returns
        -------
        WsiDicomIO
            Opened WsiDicomIO instance.
        """
        fs, fs_path, filepath = self._resolve(path)
        fs.makedirs(filepath.parent.path, exist_ok=True)
        stream = self._open_stream(fs, fs_path, mode)
        return WsiDicomIO(stream, transfer_syntax=transfer_syntax, filepath=filepath)

    def _open_streams(
        self,
        path: str | Path | UPath,
        mode: Literal["rb"] | Literal["r+b"] | Literal["w+b"],
    ) -> Iterator[tuple[BinaryIO, UPath]]:
        """Open streams from path. If path is a directory, open all files in directory.

        Parameters
        ----------
        path: str | Path | UPath,
            Path to open.
        mode: Literal["rb"] | Literal["r+b"] | Literal["w+b"]
            Mode to open in.

        Returns
        -------
        Iterator[tuple[BinaryIO, UPath]]
            Opened streams, each with the path it was opened from.
        """
        fs, fs_path, filepath = self._resolve(path)
        if fs.isdir(fs_path):
            files = filepath.iterdir()
        elif fs.isfile(fs_path):
            files = iter((filepath,))
        else:
            # Neither a directory nor a file, so match it as a glob pattern,
            # from the root of the filesystem it is on.
            root = filepath.parents[-1]
            files = root.glob(str(filepath.relative_to(root)).replace("\\", "/"))
        for file in files:
            if file.is_file():
                yield self._open_stream(fs, file.path, mode), file

    def _resolve(
        self, path: str | Path | UPath
    ) -> tuple[AbstractFileSystem, str, UPath]:
        """Resolve path into the filesystem to use, the path within that
        filesystem, and the path as an `UPath`.

        The path within the filesystem has the protocol stripped, as the
        filesystem holds it instead, and is only usable together with it.

        Parameters
        ----------
        path: str | Path | UPath
            Path to resolve.

        Returns
        -------
        tuple[AbstractFileSystem, str, UPath]
            Filesystem, path within the filesystem, and path as an `UPath`.
        """
        filepath = as_upath(path, self._file_options)
        return filepath.fs, filepath.path, filepath

    def _open_stream(
        self,
        fs: AbstractFileSystem,
        path: str,
        mode: Literal["rb"] | Literal["r+b"] | Literal["w+b"],
    ) -> BinaryIO:
        """Open stream from path.

        Parameters
        ----------
        fs: AbstractFileSystem
            Filesystem to open from.
        path: str
            Path to open.
        mode: Literal["rb"] | Literal["r+b"] | Literal["w+b"]
            Mode to open in.

        Returns
        -------
        BinaryIO
            Opened stream. The type of file object differs by filesystem, e.g.
            `LocalFileOpener` for local files and `AbstractBufferedFile` for
            those read in blocks over a network.
        """
        return cast(BinaryIO, fs.open(path, mode))
