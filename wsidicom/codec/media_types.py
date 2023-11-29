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

from pydicom.uid import (
    RLELossless,
    ExplicitVRLittleEndian,
    UID,
    JPEGTransferSyntaxes,
    JPEGLSTransferSyntaxes,
    JPEG2000TransferSyntaxes,
)


def determine_media_type(transfer_syntax: UID) -> str:
    """Determine media type from transfer syntax.

    Parameters
    ----------
    transfer_syntax: UID
        Transfer syntax to determine media type for.

    Returns
    ----------
    str
        Media type.
    """
    if transfer_syntax in JPEGTransferSyntaxes:
        return "image/jpeg"
    if transfer_syntax in JPEGLSTransferSyntaxes:
        return "image/jls"
    if transfer_syntax in JPEG2000TransferSyntaxes:
        return "image/jp2"
    if transfer_syntax == RLELossless:
        return "image/x-dicom-rle"
    if transfer_syntax == ExplicitVRLittleEndian:
        return "application/octet-stream"
    raise NotImplementedError(f"Unsupported transfer syntax: {transfer_syntax}")
