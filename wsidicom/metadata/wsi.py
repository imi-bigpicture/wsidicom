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

"""Complete WSI model."""

from collections.abc import Sequence
from dataclasses import dataclass

from pydicom.uid import UID

from wsidicom.metadata.contributing_equipment import ContributingEquipment
from wsidicom.metadata.equipment import Equipment
from wsidicom.metadata.label import Label
from wsidicom.metadata.overview import Overview
from wsidicom.metadata.patient import Patient
from wsidicom.metadata.pyramid import Pyramid
from wsidicom.metadata.series import Series
from wsidicom.metadata.slide import Slide
from wsidicom.metadata.study import Study


@dataclass(frozen=True)
class WsiMetadata:
    """Metadata for a whole slide image.

    Parameters
    ----------
    study: Study
        Metadata of the study the image belongs to.
    series: Series
        Metadata of the series the image belongs to.
    patient: Patient
        Metadata of the patient the related to the imaged slide.
    equipment: Equipment
        Metadata of the scanner equipment used to acquire the image.
    slide: Slide
        Metadata of the imaged slide.
    pyramid: Pyramid
        Metadata of the pyramid of the slide.
    label: Label
        Metadata of the label of the slide.
    overview: Overview | None = None
        Metadata of the overview image of the slide, if present.
    frame_of_reference_uid: UID | None = None
        The frame of reference uid of the image.
    dimension_organization_uids: Sequence[UID] | None = None
        The dimension organization uids of the image.
    contributing_equipment: Sequence[ContributingEquipment] = ()
        Equipment that contributed to the creation of the image, other than the
        acquisition equipment (e.g. a format converter).
    """

    study: Study
    series: Series
    patient: Patient
    equipment: Equipment
    slide: Slide
    pyramid: Pyramid
    label: Label
    overview: Overview | None = None
    frame_of_reference_uid: UID | None = None
    dimension_organization_uids: Sequence[UID] | None = None
    contributing_equipment: Sequence[ContributingEquipment] = ()
