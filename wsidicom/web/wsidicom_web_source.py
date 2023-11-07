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

from collections import abc
from typing import Iterable, List, Union

from pydicom.uid import UID

from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.instance import ImageType, WsiDataset, WsiInstance
from wsidicom.source import Source
from wsidicom.web.wsidicom_web_client import WsiDicomWebClient
from wsidicom.web.wsidicom_web_image_data import WsiDicomWebImageData

"""A source for reading WSI DICOM files from DICOMWeb."""


class WsiDicomWebSource(Source):
    """Source reading WSI DICOM instances from DICOMWeb."""

    def __init__(
        self,
        client: WsiDicomWebClient,
        study_uid: Union[str, UID],
        series_uids: Union[str, UID, Iterable[Union[str, UID]]],
        requested_transfer_syntax: UID,
    ):
        """Create a WsiDicomWebSource.

        Parameters
        ----------
        client: WsiDicomWebClient
            Client use for DICOMWeb communication.
        study_uid: Union[str, UID]
            Study UID of DICOM WSI to open.
        series_uids: Union[str, UID, Iterable[Union[str, UID]]]
            Series UIDs of DICOM WSI top open.
        requested_transfer_syntax: UID
            Transfer syntax to request for image data, for example
            UID("1.2.840.10008.1.2.4.50") for JPEGBaseline8Bit.

        """
        if not isinstance(study_uid, UID):
            study_uid = UID(study_uid)

        if not isinstance(series_uids, abc.Iterable):
            series_uids = [series_uids]

        self._level_instances: List[WsiInstance] = []
        self._label_instances: List[WsiInstance] = []
        self._overview_instances: List[WsiInstance] = []
        self._annotation_instances: List[AnnotationInstance] = []

        for series_uid in series_uids:
            if not isinstance(series_uid, UID):
                series_uid = UID(series_uid)

            for instance_uid in client.get_wsi_instances(study_uid, series_uid):
                dataset = client.get_instance(study_uid, series_uid, instance_uid)
                if not WsiDataset.is_supported_wsi_dicom(
                    dataset, requested_transfer_syntax
                ):
                    continue
                dataset = WsiDataset(dataset)
                image_data = WsiDicomWebImageData(
                    client, dataset, requested_transfer_syntax
                )
                instance = WsiInstance(dataset, image_data)
                if instance.image_type == ImageType.VOLUME:
                    self._level_instances.append(instance)
                elif instance.image_type == ImageType.LABEL:
                    self._label_instances.append(instance)
                elif instance.image_type == ImageType.OVERVIEW:
                    self._overview_instances.append(instance)

            for instance_uid in client.get_ann_instances(study_uid, series_uid):
                instance = client.get_instance(study_uid, series_uid, instance_uid)
                annotation_instance = AnnotationInstance.open_dataset(instance)
                self._annotation_instances.append(annotation_instance)

        try:
            self._base_dataset = next(
                instance.dataset
                for instance in sorted(
                    self._level_instances,
                    reverse=True,
                    key=lambda instance: instance.dataset.image_size.width,
                )
            )
        except StopIteration:
            WsiDicomNotFoundError(
                "No level instances found", f"{study_uid}, {series_uid}"
            )

    @property
    def base_dataset(self) -> WsiDataset:
        """The dataset of the one of the level instances."""
        return self._base_dataset

    @property
    def level_instances(self) -> List[WsiInstance]:
        """The level instances parsed from the source."""
        return self._level_instances

    @property
    def label_instances(self) -> List[WsiInstance]:
        """The label instances parsed from the source."""
        return self._label_instances

    @property
    def overview_instances(self) -> List[WsiInstance]:
        """The overview instances parsed from the source."""
        return self._overview_instances

    @property
    def annotation_instances(self) -> List[AnnotationInstance]:
        """The annotation instances parsed from the source."""
        return self._annotation_instances

    def close(self) -> None:
        pass
