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

from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Dict,
    Iterable,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from fsspec.core import url_to_fs
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from pydicom.uid import UID
from upath import UPath

from wsidicom.file.io.wsidicom_io import WsiDicomIO


class WsiDicomStreamOpener:
    def __init__(
        self,
        file_options: Optional[Dict[str, Any]] = None,
    ):
        """Create a WsiDicomStreamOpener.

        Parameters
        ----------
        file_options: Optional[Dict[str, Any]] = None
            Keyword arguments for opening filesystems and files.
        """

        self._file_options = file_options or {}

    def open(
        self,
        files: Union[str, Path, UPath, Iterable[Union[str, Path, UPath]]],
        sop_class_uids: Optional[Union[UID, Sequence[UID]]] = None,
    ) -> Iterator[WsiDicomIO]:
        """Open DICOM streams in paths and return WsiDicomIO instances.

        Parameters
        ----------
        files: Union[str, Path, UPath, Iterable[Union[str, Path, UPath]]],
            Folder, single file, or sequence of files to open.
        sop_class_uids: Optional[Union[UID, Sequence[UID]]] = None,
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
            for stream in self._open_streams(str(file), "rb"):
                try:
                    if hasattr(stream, "path"):
                        path = UPath(getattr(stream, "path"))
                    else:
                        path = None
                    dicom_io = WsiDicomIO(stream, owned=True, filepath=path)
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
        path: Union[str, Path, UPath],
        mode: Union[Literal["r+b"], Literal["w+b"]],
    ) -> WsiDicomIO:
        """Open a stream for writing.

        Parameters
        ----------
        path: Union[str, Path, UPath]
            Path to open.
        mode: Union[Literal["r+b"], Literal["w+b"]]
            Mode to open in.

        Returns
        -------
        WsiDicomIO
            Opened WsiDicomIO instance.
        """
        fs, path = self._open_filesystem(str(path))
        fs.makedirs(UPath(path).parent, exist_ok=True)
        stream = self._open_stream(fs, path, mode)
        return WsiDicomIO(stream, owned=True, filepath=UPath(path))

    def _open_streams(
        self,
        path: str,
        mode: Union[Literal["rb"], Literal["r+b"], Literal["w+b"]],
    ) -> Iterator[Union[BinaryIO, AbstractBufferedFile]]:
        """Open streams from path. If path is a directory, open all files in directory.

        Parameters
        ----------
        path: str
            Path to open.
        mode: Union[Literal["rb"], Literal["r+b"], Literal["w+b"]]
            Mode to open in.

        Returns
        -------
        Iterator[Union[BinaryIO, AbstractBufferedFile]]
            Opened streams.
        """
        fs, path = self._open_filesystem(path)
        if fs.isdir(str(path)):
            files = (file for file in fs.ls(str(path), detail=False) if fs.isfile(file))
        elif fs.isfile(str(path)):
            files = [str(path)]
        else:
            files = (
                file
                for file in fs.glob(str(path))
                if isinstance(file, str) and fs.isfile(file)
            )
        for file in files:
            yield self._open_stream(fs, file, mode)

    def _open_filesystem(self, path: str) -> Tuple[AbstractFileSystem, str]:
        """Open fsspec filesystem from path.

        Parameters
        ----------
        path: str
            Path to open.

        Returns
        -------
        Tuple[AbstractFileSystem, str]
            Opened filesystem and path.
        """
        fs, path = url_to_fs(path, **self._file_options or {})
        return fs, path  # type: ignore

    def _open_stream(
        self,
        fs: AbstractFileSystem,
        path: str,
        mode: Union[Literal["rb"], Literal["r+b"], Literal["w+b"]],
    ) -> AbstractBufferedFile:
        """Open stream from path.

        Parameters
        ----------
        fs: AbstractFileSystem
            Filesystem to open from.
        path: str
            Path to open.
        mode: Union[Literal["rb"], Literal["r+b"], Literal["w+b"]]
            Mode to open in.

        Returns
        -------
        AbstractBufferedFile
            Opened stream.
        """
        return fs.open(path, mode, **self._file_options or {})  # type: ignore
