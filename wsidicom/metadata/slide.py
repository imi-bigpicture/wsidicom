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

"""Slide model."""

from dataclasses import dataclass
from typing import Optional, Sequence, Union

from wsidicom.metadata.sample import SlideSample, SpecimenIdentifier, Staining


@dataclass
class Slide:
    """
    Metadata for a slide.

    A slide has a an identifier and contains one or more samples. All the samples on the
    slide has been stained with the same stainings.

    Corresponds to attributes in the Specimen Module:
    https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.22.html

    Parameters
    ----------
    identifier : Optional[Union[str, SpecimenIdentifier]] = None
        The identifier of the slide (Container Identifier).
    stainings : Optional[Sequence[Staining]] = None
        List of stainings used on the slide.
    samples : Optional[Sequence[SlideSample]] = None
        List of samples on the slide.
    """

    identifier: Optional[Union[str, SpecimenIdentifier]] = None
    stainings: Optional[Sequence[Staining]] = None
    samples: Optional[Sequence[SlideSample]] = None
