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

import logging
from typing import Iterable, List, Optional, Sequence, Union

from pydicom.uid import (
    UID,
    JPEGBaseline8Bit,
    JPEG2000,
    JPEG2000Lossless,
    JPEGLSLossless,
    JPEGLSNearLossless,
    JPEGLosslessSV1,
    JPEGLosslessP14,
    RLELossless,
    ExplicitVRLittleEndian,
)
from wsidicom.codec import Codec

from wsidicom.errors import WsiDicomNotFoundError
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.instance import ImageType, WsiDataset, WsiInstance
from wsidicom.source import Source
from wsidicom.web.wsidicom_web_client import WsiDicomWebClient
from wsidicom.web.wsidicom_web_image_data import WsiDicomWebImageData

"""A source for reading WSI DICOM files from DICOMWeb."""

PREFERED_WEB_TRANSFER_SYNTAXES = [
    JPEGBaseline8Bit,
    JPEG2000,
    JPEG2000Lossless,
    JPEGLSNearLossless,
    JPEGLSLossless,
    JPEGLosslessSV1,
    JPEGLosslessP14,
    RLELossless,
    ExplicitVRLittleEndian,
]


class WsiDicomWebSource(Source):
    """Source reading WSI DICOM instances from DICOMWeb."""

    def __init__(
        self,
        client: WsiDicomWebClient,
        study_uid: Union[str, UID],
        series_uids: Union[str, UID, Iterable[Union[str, UID]]],
        requested_transfer_syntaxes: Optional[Sequence[UID]] = None,
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
        requested_transfer_syntaxes: Optional[Sequence[UID]] = None
            Transfer syntax to request for image data, for example
            UID("1.2.840.10008.1.2.4.50") for JPEGBaseline8Bit.

        """
        if not isinstance(study_uid, UID):
            study_uid = UID(study_uid)

        if isinstance(series_uids, (str, UID)):
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
                if not WsiDataset.is_supported_wsi_dicom(dataset):
                    logging.info(f"Non-supported instance {instance_uid}.")
                    continue
                dataset = WsiDataset(dataset)
                transfer_syntax = self._determine_transfer_syntax(
                    client,
                    dataset,
                    requested_transfer_syntaxes,
                )
                if transfer_syntax is None:
                    logging.info(
                        f"No supported transfer syntax found for instance {instance_uid}."
                    )
                    continue

                image_data = WsiDicomWebImageData(client, dataset, transfer_syntax)
                instance = WsiInstance(dataset, image_data)
                if instance.image_type == ImageType.VOLUME:
                    self._level_instances.append(instance)
                elif instance.image_type == ImageType.LABEL:
                    self._label_instances.append(instance)
                elif instance.image_type == ImageType.OVERVIEW:
                    self._overview_instances.append(instance)

            for instance_uid in client.get_annotation_instances(study_uid, series_uid):
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
                "No level instances found", f"{study_uid}, {series_uids}"
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

    @classmethod
    def _determine_transfer_syntax(
        cls,
        client: WsiDicomWebClient,
        dataset: WsiDataset,
        requested_transfer_syntaxes: Optional[Iterable[UID]] = None,
    ) -> Optional[UID]:
        """Determine transfer syntax to use for image data.

        Parameters
        ----------
        client: WsiDicomWebClient
            Client used for DICOMWeb communication.
        dataset: WsiDataset
            Dataset to determine transfer syntax for.
        requested_transfer_syntaxes: Optional[Iterable[UID]] = None
            Transfer syntaxes to try in order of preference.

        Returns
        ----------
        Optional[UID]
            Transfer syntax to use for image data, or None if no supported transfer
            syntax was found.
        """
        if requested_transfer_syntaxes is None:
            requested_transfer_syntaxes = PREFERED_WEB_TRANSFER_SYNTAXES
        requested_transfer_syntaxes = (
            transfer_syntax
            for transfer_syntax in requested_transfer_syntaxes
            if Codec.is_supported(
                transfer_syntax,
                dataset.samples_per_pixel,
                dataset.bits,
                dataset.photometric_interpretation,
                dataset.pixel_representation,
            )
        )

        return next(
            (
                transfer_syntax
                for transfer_syntax in requested_transfer_syntaxes
                if client.is_transfer_syntax_supported(
                    dataset.uids.slide.study_instance,
                    dataset.uids.slide.series_instance,
                    dataset.uids.instance,
                    transfer_syntax,
                )
            ),
            None,
        )
