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
from dataclasses import dataclass, replace
from typing import cast

from wsidicom.options import DecoderOption, DownsamplerOption, ResampleFilterOption


@dataclass(frozen=True)
class Settings:
    """Immutable settings for a WsiDicom.

    Construct with the desired values and pass to ``WsiDicom.open(settings=...)``
    for per-object settings. To change the process-wide default instead, mutate
    the global ``settings`` (e.g. ``settings.resampling_filter = ...``). The
    option-enum fields accept their string value (e.g. ``"box"``) as well as the
    enum member.
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
        # Accept the string form of the option enums (e.g. "box") and coerce to
        # the enum member; idempotent when already a member.
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


class _SettingsProxy:
    """The mutable process-default surface exposed as the global ``settings``.

    Reads resolve to the settings active in the current context (or the default);
    writes replace the process default, since ``Settings`` is immutable. So
    ``settings.resampling_filter = ...`` keeps configuring the default, while
    reads honor any per-operation ``Settings`` activated by ``use_settings``.
    """

    def __getattr__(self, name: str) -> object:
        return getattr(_active_settings.get() or _default_settings, name)

    def __setattr__(self, name: str, value: object) -> None:
        global _default_settings
        _default_settings = replace(_default_settings, **{name: value})


# Typed as ``Settings`` so attribute access stays fully typed; at runtime it is
# the proxy, which reads the context-active settings and writes the default.
settings: Settings = cast(Settings, _SettingsProxy())
"""Process-wide default settings. Mutate to change the default (e.g.
``settings.resampling_filter = ...``); reads reflect the settings active in the
current context. Pass a ``Settings`` to ``WsiDicom.open(settings=...)`` for
per-call settings instead."""


@contextmanager
def use_settings(active: Settings | None = None) -> Iterator[Settings]:
    """Yield the settings in effect, optionally activating ``active`` first.

    With no argument it is a read: it yields the currently active settings (or
    the default) and changes nothing, so consumers can

        with use_settings() as settings:
            ...  # settings is whatever is in effect

    With an ``active`` ``Settings`` it activates it for the current context and
    for thread-pool tasks submitted within it (executors that propagate the
    context), yields it, and resets on exit:

        with use_settings(Settings(resampling_filter="box")) as settings:
            ...  # settings is active here and for nested reads
    """
    if active is None:
        yield _active_settings.get() or _default_settings
        return
    token = _active_settings.set(active)
    try:
        yield active
    finally:
        _active_settings.reset(token)
