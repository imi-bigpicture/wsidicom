#    Copyright 2026 SECTRA AB
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

"""Pool of unbuffered file handles for reading from a local file."""

import os
import queue
import threading
from typing import BinaryIO, cast

_READ_FLAGS = os.O_RDONLY | getattr(os, "O_BINARY", 0)


class FileHandlePool:
    """Reads a local file through a pool of unbuffered handles.

    A read takes a free handle, seeks, reads and returns the handle. Reads on different
    threads therefore do not wait for each other. A new handle is opened only when none
    is free. The number of open handles then follows the peak number of reads at once,
    up to `max_handles`. When that many are open, a read waits for a handle to be
    returned. This keeps many threads reading a slide of many files from opening one
    handle each to every file.

    The handles are unbuffered because frames are read at scattered positions. A
    buffered handle would read a whole buffer for each frame and copy the frame out of
    it. For a frame of a few kilobytes, that costs several times the read itself."""

    def __init__(self, path: str, max_handles: int) -> None:
        """
        Parameters
        ----------
        path: str
            Path of the local file to read.
        max_handles: int
            Most handles to have open at once.
        """
        if max_handles < 1:
            raise ValueError(f"Max handles must be at least 1, got {max_handles}.")
        self._path = path
        self._max_handles = max_handles
        self._available: queue.SimpleQueue[BinaryIO] = queue.SimpleQueue()
        self._handles: list[BinaryIO] = []
        self._lock = threading.Lock()
        self._closed = False

    def __str__(self) -> str:
        return f"{type(self).__name__}({self._path})"

    @property
    def handle_count(self) -> int:
        """Number of handles open, at most `max_handles`."""
        with self._lock:
            return len(self._handles)

    def read(self, position: int, length: int) -> bytes:
        """Return `length` bytes from `position`, or fewer if the file ends before.

        Parameters
        ----------
        position: int
            Position to read from.
        length: int
            Number of bytes to read.

        Returns
        -------
        bytes
            The bytes read.
        """
        try:
            handle = self._available.get_nowait()
        except queue.Empty:
            handle = self._open_or_wait()
        try:
            handle.seek(position)
            data = handle.read(length)
            if len(data) < length:
                # An unbuffered read returns what one system call gives. That may be
                # less than asked for even if the file has not ended.
                data = self._read_rest(handle, data, length)
            return data
        finally:
            self._available.put(handle)

    def close(self) -> None:
        """Close the handles of the pool."""
        with self._lock:
            self._closed = True
            handles, self._handles = self._handles, []
        for handle in handles:
            handle.close()

    @staticmethod
    def _read_rest(handle: BinaryIO, data: bytes, length: int) -> bytes:
        """Keep reading after a short read, until `length` bytes are read or the file
        ends."""
        parts = [data]
        remaining = length - len(data)
        while remaining > 0:
            part = handle.read(remaining)
            if not part:
                break
            parts.append(part)
            remaining -= len(part)
        return b"".join(parts)

    def _open_or_wait(self) -> BinaryIO:
        """Open a new handle, which the pool closes when it is closed. If `max_handles`
        are already open, wait for one of them to be returned instead. A read that is
        waiting when the pool closes gets a closed handle and raises."""
        with self._lock:
            if self._closed:
                raise ValueError(f"{self} is closed.")
            if len(self._handles) < self._max_handles:
                handle = cast(
                    BinaryIO,
                    os.fdopen(os.open(self._path, _READ_FLAGS), "rb", buffering=0),
                )
                self._handles.append(handle)
                return handle
        return self._available.get()
