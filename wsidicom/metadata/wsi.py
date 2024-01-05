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

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Sequence

from pydicom.uid import UID, generate_uid

from wsidicom.metadata.equipment import Equipment
from wsidicom.metadata.image import Image
from wsidicom.metadata.label import Label
from wsidicom.metadata.optical_path import OpticalPath
from wsidicom.metadata.patient import Patient
from wsidicom.metadata.series import Series
from wsidicom.metadata.slide import Slide
from wsidicom.metadata.study import Study


@dataclass
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
    optical_paths: Sequence[OpticalPath]
        Metadata of the optical paths used to acquire the image.
    slide: Slide
        Metadata of the imaged slide.
    label: Label
        Metadata of the label of the slide.
    image: Image
        Technical metadata of the image.
    frame_of_reference_uid: Optional[UID] = None
        The frame of reference uid of the image.
    dimension_organization_uids: Optional[Sequence[UID]] = None
        The dimension organization uids of the image.
    """

    study: Study
    series: Series
    patient: Patient
    equipment: Equipment
    optical_paths: Sequence[OpticalPath]
    slide: Slide
    label: Label
    image: Image
    frame_of_reference_uid: Optional[UID] = None
    dimension_organization_uids: Optional[Sequence[UID]] = None

    @cached_property
    def default_frame_of_reference_uid(self) -> UID:
        """Frame of reference uid used if not set."""
        if self.frame_of_reference_uid is not None:
            return self.frame_of_reference_uid
        return generate_uid()

    @cached_property
    def default_dimension_organization_uids(self) -> Sequence[UID]:
        """Dimension organization uids used if not set."""
        if self.dimension_organization_uids is not None:
            return self.dimension_organization_uids
        return [generate_uid()]

    @classmethod
    def merge_image_types(
        cls,
        volume: "WsiMetadata",
        label: Optional["WsiMetadata"],
        overview: Optional["WsiMetadata"],
    ):
        if volume.label is not None:
            if label is not None:
                label_label = label.label
            else:
                label_label = None
            if overview is not None:
                overview_label = overview.label
            else:
                overview_label = None

            merged_label = Label.merge_image_types(
                volume.label, label_label, overview_label
            )
        else:
            merged_label = Label()
        return cls(
            study=volume.study,
            series=volume.series,
            patient=volume.patient,
            equipment=volume.equipment,
            optical_paths=volume.optical_paths,
            slide=volume.slide,
            label=merged_label,
            image=volume.image,
            frame_of_reference_uid=volume.frame_of_reference_uid,
            dimension_organization_uids=volume.dimension_organization_uids,
        )
