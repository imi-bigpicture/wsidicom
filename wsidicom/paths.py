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

"""Module for paths to files on any filesystem."""

from pathlib import Path
from typing import Any

from upath import UPath


def as_upath(
    path: str | Path | UPath, file_options: dict[str, Any] | None = None
) -> UPath:
    """Return path as an `UPath` on the filesystem it names.

    An `UPath` names a file on a filesystem and carries the configuration for
    reaching it, so that code handed one can use it without knowing which
    filesystem it is on, and without the file options having to be carried
    alongside it. Use this where a path is taken in.

    A path that is already an `UPath` carries its own configuration and is used
    as given; a path in any other form is configured with `file_options`. A path
    on the local filesystem is made absolute, so that it names the same file
    wherever it is used from.

    Parameters
    ----------
    path: str | Path | UPath
        Path to return as an `UPath`.
    file_options: dict[str, Any] | None = None
        Keyword arguments for opening the filesystem.

    Returns
    -------
    UPath
        Path as an `UPath`.
    """
    if not isinstance(path, UPath):
        path = UPath(path, **(file_options or {}))
    if _is_local(path):
        return path.resolve()
    return path


def as_local_path(
    path: str | Path | UPath, file_options: dict[str, Any] | None = None
) -> Path | None:
    """Return path as a `Path` on the local filesystem, or `None` if it is not
    on it.

    Use for readers that open files by name rather than through a stream. A
    `file://` or `local://` url names a local file and is returned as a `Path`
    to it; a file on another filesystem has no local name. Nor does a chained
    url (a cache, say), as the chain is part of how the file is to be read.

    Parameters
    ----------
    path: str | Path | UPath
        Path to return as a local `Path`.
    file_options: dict[str, Any] | None = None
        Keyword arguments for opening the filesystem.

    Returns
    -------
    Path | None
        Path as a local `Path`, or `None` if it is on another filesystem.
    """
    upath = as_upath(path, file_options)
    if not _is_local(upath):
        return None
    return Path(upath.path)


def _is_local(path: UPath) -> bool:
    """Return True if path names a file on the local filesystem."""
    return path.protocol in ("", "file", "local")
