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

from typing import List, Union

from pydicom import Dataset
from pydicom.uid import UID

from wsidicom.dataset import ImageType
from wsidicom.graphical_annotations import AnnotationInstance
from wsidicom.instance import WsiInstance
from wsidicom.source import Source
from wsidicom.wsidicom_web.wsidicom_web import WsiDicomWeb, WsiDicomWebClient
from wsidicom.wsidicom_web.wsidicom_web_image_data import WsiDicomWebImageData


class WsiDicomWebSource(Source):
    def __init__(
        self,
        client: WsiDicomWebClient,
        study_uid: Union[str, UID],
        series_uid: Union[str, UID],
    ):
        if not isinstance(study_uid, UID):
            study_uid = UID(study_uid)
        if not isinstance(series_uid, UID):
            series_uid = UID(series_uid)
        self._level_instances: List[WsiInstance] = []
        self._label_instances: List[WsiInstance] = []
        self._overview_instances: List[WsiInstance] = []
        for instance_uid in client.get_wsi_instances(study_uid, series_uid):
            web_instance = WsiDicomWeb(client, study_uid, series_uid, instance_uid)
            image_data = WsiDicomWebImageData(web_instance)
            instance = WsiInstance(web_instance.dataset, image_data)
            if instance.image_type == ImageType.VOLUME:
                self._level_instances.append(instance)
            elif instance.image_type == ImageType.LABEL:
                self._label_instances.append(instance)
            elif instance.image_type == ImageType.OVERVIEW:
                self._overview_instances.append(instance)
        self._annotation_instances: List[AnnotationInstance] = []
        for instance_uid in client.get_ann_instances(study_uid, series_uid):
            instance = client.get_instance(study_uid, series_uid, instance_uid)
            annotation_instance = AnnotationInstance.open_dataset(instance)
            self._annotation_instances.append(annotation_instance)

        self._base_dataset = self._level_instances[0].dataset

    @property
    def base_dataset(self) -> Dataset:
        return self._base_dataset

    @property
    def level_instances(self) -> List[WsiInstance]:
        return self._level_instances

    @property
    def label_instances(self) -> List[WsiInstance]:
        return self._label_instances

    @property
    def overview_instances(self) -> List[WsiInstance]:
        return self._overview_instances

    @property
    def annotation_instances(self) -> List[AnnotationInstance]:
        return self._annotation_instances
