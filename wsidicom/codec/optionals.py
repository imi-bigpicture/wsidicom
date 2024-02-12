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

"""Module with imports for optional codec packages."""


try:
    from rle.utils import decode_frame as rle_decode_frame
    from rle.utils import encode_frame as rle_encode_frame

    PYLIBJPEGRLE_AVAILABLE = True
except ImportError:
    PYLIBJPEGRLE_AVAILABLE = False
    rle_decode_frame = None
    rle_encode_frame = None

try:
    from imagecodecs import (
        JPEG2K,
        JPEG8,
        JPEGLS,
        dicomrle_decode,
        jpeg2k_decode,
        jpeg2k_encode,
        jpeg8_decode,
        jpeg8_encode,
        jpegls_decode,
        jpegls_encode,
        packbits_encode,
    )

    IMAGE_CODECS_AVAILABLE = True

except ImportError:
    IMAGE_CODECS_AVAILABLE = False
    JPEG2K = None
    JPEG8 = None
    JPEGLS = None
    jpeg2k_decode = None
    jpeg8_decode = None
    jpegls_decode = None
    jpeg2k_encode = None
    jpeg8_encode = None
    jpegls_encode = None
    packbits_encode = None
    dicomrle_decode = None

try:
    from jpeg_ls import decode as pylibjpeg_ls_decode
    from jpeg_ls import encode_array as pylibjpeg_ls_encode

    PYLIBJPEGLS_AVAILABLE = True

except ImportError:
    pylibjpeg_ls_decode = None
    pylibjpeg_ls_encode = None
    PYLIBJPEGLS_AVAILABLE = False


try:
    from openjpeg.utils import decode as pylibjpeg_openjpeg_decode
    from openjpeg.utils import encode as pylibjpeg_openjpeg_encode

    PYLIBJPEGOPENJPEG_AVAILABLE = True

except ImportError:
    pylibjpeg_openjpeg_decode = None
    pylibjpeg_openjpeg_encode = None
    PYLIBJPEGOPENJPEG_AVAILABLE = False
