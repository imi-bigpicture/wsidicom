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
    Optional,
    Sequence,
    Tuple,
    Union,
)

from fsspec.core import url_to_fs
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from pydicom.uid import UID

from wsidicom.file.io.wsidicom_io import WsiDicomIO


class WsiDicomStreamOpener:
    def __init__(
        self,
        sop_class_uids: Optional[Sequence[UID]] = None,
        storage_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Create a WsiDicomStreamOpener.

        Parameters
        ----------
        sop_class_uids: Optional[Sequence[UID]] = None
            SOP class uids to filter on.
        storage_kwargs: Optional[Dict[str, Any]] = None
            Keyword arguments for opening the file.
        """

        self._sop_class_uids = sop_class_uids
        self._storage_kwargs = storage_kwargs or {}

    def open(
        self,
        files: Union[
            str,
            Path,
            BinaryIO,
            AbstractBufferedFile,
            Iterable[Union[str, Path, BinaryIO, AbstractBufferedFile]],
        ],
    ) -> Iterator[WsiDicomIO]:
        """Open DICOM streams in paths and return WsiDicomIO instances.

        Parameters
        ----------
        files: Union[
            str,
            Path,
            BinaryIO,
            AbstractBufferedFile,
            Iterable[Union[str, Path, BinaryIO, AbstractBufferedFile]],
        ]
            Folder, single file, or sequence of files to open.

        Returns
        -------
        Iterator[WsiDicomIO]
            Opened WsiDicomIO instances.
        """
        if isinstance(files, (str, Path, BinaryIO, AbstractBufferedFile)):
            files = [files]
        streams: Sequence[Tuple[Union[BinaryIO, AbstractBufferedFile], bool]] = []
        for file in files:
            if isinstance(file, (str, Path)):
                streams.extend((stream, True) for stream in self._open_streams(file))
            else:
                streams.append((file, False))
        for stream, owned in streams:
            try:
                path = getattr(stream, "path", None)
                dicom_io = WsiDicomIO(stream, owned=owned, filepath=path)
                if (
                    self._sop_class_uids is None
                    or dicom_io.media_storage_sop_class_uid in self._sop_class_uids
                ):
                    yield dicom_io
                else:
                    stream.close()
            except Exception:
                stream.close()

    def _open_streams(
        self, path: Union[str, Path]
    ) -> Iterator[Union[BinaryIO, AbstractBufferedFile]]:
        """Open streams from path. If path is a directory, open all files in directory.

        Parameters
        ----------
        path: Union[str, Path]
            Path to open.

        Returns
        -------
        Iterator[Union[BinaryIO, AbstractBufferedFile]]
            Opened streams.
        """
        fs, path = self._open_filesystem(path)
        if fs.isdir(path):
            files = (file for file in fs.ls(path) if fs.isfile(file))
        elif fs.isfile(path):
            files = [path]
        else:
            return
        for file in files:
            yield self._open_stream(fs, file)

    def _open_filesystem(
        self, path: Union[str, Path]
    ) -> Tuple[AbstractFileSystem, str]:
        """Open fsspec filesystem from path.

        Parameters
        ----------
        path: Union[str, Path]
            Path to open.

        Returns
        -------
        Tuple[AbstractFileSystem, str]
            Opened filesystem and path.
        """
        fs, path = url_to_fs(str(path), **self._storage_kwargs or {})
        return fs, path  # type: ignore

    def _open_stream(self, fs: AbstractFileSystem, path: str) -> AbstractBufferedFile:
        """Open stream from path.

        Parameters
        ----------
        fs: AbstractFileSystem
            Filesystem to open from.
        path: str
            Path to open.

        Returns
        -------
        AbstractBufferedFile
            Opened stream.
        """
        return fs.open(path, "rb", **self._storage_kwargs or {})  # type: ignore
