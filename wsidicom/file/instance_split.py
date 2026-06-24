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

"""Control for how optical paths and focal planes are split across instances."""

from enum import Flag, auto


class InstanceSplit(Flag):
    """Controls how optical paths and focal planes are split across written instances.

    By default all optical paths and focal planes of a pyramid level (or group)
    are combined into a single instance. Flags can be combined, e.g.
    ``InstanceSplit.FOCAL_PLANE | InstanceSplit.OPTICAL_PATH`` writes one instance
    per (focal plane, optical path) pair.
    """

    NONE = 0
    """All optical paths and focal planes in one instance (default)."""
    FOCAL_PLANE = auto()
    """Write a separate instance per focal plane."""
    OPTICAL_PATH = auto()
    """Write a separate instance per optical path."""
