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


import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from wsidicom.options import DecoderOption, DownsamplerOption, ResampleFilterOption


@dataclass(frozen=True)
class Settings:
    """Immutable settings for a WsiDicom.

    Construct with the desired values and pass to ``WsiDicom.open(settings=...)``
    for per-object settings. To change the process-wide default instead, use
    ``set_default_settings(Settings(...))``. The option-enum fields accept their
    string value (e.g. ``"box"``) as well as the enum member.
    """

    strict_uid_check: bool = False
    """If frame of reference uid needs to match."""
    strict_tile_size_check: bool = True
    """If tile size need to match for levels. If `False` the tile size across
    levels are allowed to be non-uniform."""
    strict_specimen_identifier_check: bool = True
    """If `True` the issuer of two specimen identifiers needs to match or both be
    None for the identifiers to match. If `False` the identifiers will match also
    if either issuer is None. Either way the identifier needs to match."""
    focal_plane_distance_threshold: float = 0.000001
    """Threshold in mm for which distances between focal planes are considered to
    be equal. Default is 1 nm, as distance between focal planes are likely larger
    than this."""
    pyramids_origin_threshold: float = 0.02
    """Threshold in mm for the distance between origins of instances to group them
    into the same pyramid. Default is 0.02 mm."""
    preferred_decoder: DecoderOption | None = None
    """Preferred decoder to use."""
    preferred_downsampler: DownsamplerOption | None = None
    """Preferred downsampler to use. None selects the fastest available (opencv
    when installed, else pillow)."""
    open_web_threads: int | None = None
    """Number of threads to use when opening web instances."""
    resampling_filter: ResampleFilterOption = ResampleFilterOption.BILINEAR
    """The resampling filter to use when rescaling images on read (region and
    thumbnail reads). Defaults to BILINEAR."""
    pyramid_resampling_filter: ResampleFilterOption = ResampleFilterOption.BOX
    """The resampling filter to use when generating pyramid levels. Defaults to
    BOX, the standard mipmap construction (each level the exact 2x2 average of the
    one above): it is moiré-free and adds no ringing or overshoot that would
    accumulate from level to level. Its 2x2 footprint also stays within tile
    boundaries, so building a level from the tiles below introduces no seams."""
    ignore_specimen_preparation_step_on_validation_error: bool = True
    """If ignore specimen preparation steps that fails to validate. If false all
    steps will be ignored if one fails to validate."""
    truncate_long_dicom_strings_on_validation_error: bool = False
    """If long DICOM strings should be truncated. This is only used if
    `pydicom.settings.writing_validation_mode` is set to `pydicom.config.RAISE`.
    If set to `True` long strings will be truncated if needed to pass validation."""
    decoded_frame_cache_size: int = 100 * 1024 * 1024
    """Size of the decoded frame cache. Default is 100 MB."""
    encoded_frame_cache_size: int = 100 * 1024 * 1024
    """Size of the encoded frame cache. Default is 100 MB."""
    level_scale_tolerance: float = 1e-2
    """Tolerance for level scale comparison. Default is 1e-2."""

    def __post_init__(self) -> None:
        # Accept the string form of the option enums (e.g. "box") at construction
        # and coerce to the enum member; idempotent when already a member.
        object.__setattr__(
            self, "resampling_filter", ResampleFilterOption(self.resampling_filter)
        )
        object.__setattr__(
            self,
            "pyramid_resampling_filter",
            ResampleFilterOption(self.pyramid_resampling_filter),
        )
        object.__setattr__(
            self, "preferred_decoder", DecoderOption.coerce(self.preferred_decoder)
        )
        object.__setattr__(
            self,
            "preferred_downsampler",
            DownsamplerOption.coerce(self.preferred_downsampler),
        )


_default_settings = Settings()
_active_settings: contextvars.ContextVar[Settings | None] = contextvars.ContextVar(
    "wsidicom_active_settings", default=None
)


def get_settings() -> Settings:
    """The settings in effect: those active in the current context (see
    ``use_settings``), or the process-wide default when none is active.

    Use this for one-off reads (``get_settings().strict_uid_check``). To read
    several fields, ``with use_settings() as settings:`` binds them once.
    """
    return _active_settings.get() or _default_settings


def set_default_settings(new_settings: Settings) -> None:
    """Replace the process-wide default settings.

    Changes what is used when no per-call ``settings`` (see ``WsiDicom.open``)
    and no active scope (see ``use_settings``) override it.

    Parameters
    ----------
    new_settings: Settings
        The new process-wide default settings.
    """
    global _default_settings
    _default_settings = new_settings


@contextmanager
def use_settings(active: Settings | None = None) -> Iterator[Settings]:
    """Activate settings for the current context and yield the settings in effect.

    Use as ``with use_settings(Settings(...)) as settings:`` to apply settings to
    a block (and thread-pool tasks it submits that propagate the context), or
    ``with use_settings() as settings:`` to just read the settings in effect.

    Parameters
    ----------
    active: Settings | None = None
        Settings to activate for the current context. When None, nothing is
        activated and the settings currently in effect are yielded.

    Yields
    ------
    Settings
        The settings in effect within the context.
    """
    if active is None:
        yield get_settings()
        return
    token = _active_settings.set(active)
    try:
        yield active
    finally:
        _active_settings.reset(token)
