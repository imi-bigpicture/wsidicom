from dataclasses import dataclass, field

from pydicom.uid import UID, generate_uid

from wsidicomizer.model.base import DicomModelBase


@dataclass
class Series(DicomModelBase):
    series_instance_uid: UID = generate_uid()
