#    Copyright 2021 SECTRA AB
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

from wsidicom.image_data import ImageData
from wsidicom.series import WsiDicomLabels, WsiDicomLevels, WsiDicomOverviews

from .config import settings
from .graphical_annotations import (Annotation, AnnotationGroup,
                                    AnnotationInstance, Measurement, Point,
                                    PointAnnotationGroup, Polygon,
                                    PolygonAnnotationGroup, Polyline,
                                    PolylineAnnotationGroup)
from .instance import WsiDataset, WsiInstance
from .wsidicom import WsiDicom

__version__ = '0.5.0'
