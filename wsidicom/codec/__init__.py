#    Copyright 2023 SECTRA AB
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

"""Module containing image codecs."""
from wsidicom.codec.encoder import Encoder
from wsidicom.codec.settings import (
    Channels,
    JpegSettings,
    JpegLosslessSettings,
    Jpeg2kSettings,
    JpegLsSettings,
    NumpySettings,
    RleSettings,
    Settings,
    Subsampling,
)
from wsidicom.codec.decoder import Decoder
from wsidicom.codec.codec import Codec
from wsidicom.codec.media_types import determine_media_type
