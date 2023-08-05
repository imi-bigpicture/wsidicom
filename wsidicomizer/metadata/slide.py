"""Slide model."""
from dataclasses import dataclass
from typing import Dict, Sequence, List, Optional, Sequence
from pydicom import Dataset
from pydicom.sequence import Sequence as DicomSequence
from wsidicom.conceptcode import (
    ContainerComponentTypeCode,
    ContainerTypeCode,
    SpecimenStainsCode,
)
from wsidicom.instance import ImageType

from wsidicomizer.metadata.dicom_attribute import (
    DicomAttribute,
    DicomCodeAttribute,
    DicomSequenceAttribute,
    DicomStringAttribute,
)
from wsidicomizer.metadata.model_base import ModelBase

from wsidicomizer.metadata.sample import SlideSample


@dataclass
class Slide(ModelBase):
    """
    Metadata for a slide.

    A slide has a an identifier and contains one or more samples. The position of the
    samples can be specified using a SampleLocation. All the samples on the slide has
    been stained with the sample list of stainings.
    """

    identifier: Optional[str] = None
    stains: Optional[Sequence[SpecimenStainsCode]] = None
    samples: Optional[Sequence[SlideSample]] = None
    overrides: Optional[Dict[str, bool]] = None

    def insert_into_dataset(self, dataset: Dataset, image_type: ImageType) -> None:
        dicom_attributes: List[DicomAttribute] = [
            DicomStringAttribute(
                "ContainerIdentifier", True, self.identifier, "Unknown"
            ),
            DicomCodeAttribute(
                "ContainerTypeCodeSequence",
                True,
                ContainerTypeCode("Microscope slide").code,
            ),
            DicomSequenceAttribute("IssuerOfTheContainerIdentifierSequence", True, []),
            DicomSequenceAttribute(
                "ContainerComponentSequence",
                False,
                [
                    DicomCodeAttribute(
                        "ContainerComponentTypeCodeSequence",
                        True,
                        ContainerComponentTypeCode("Microscope slide cover slip").code,
                    ),
                    DicomStringAttribute("ContainerComponentMaterial", False, "GLASS"),
                ],
            ),
        ]
        self._insert_dicom_attributes_into_dataset(dataset, dicom_attributes)
        if self.samples is not None:
            dataset.SpecimenDescriptionSequence = DicomSequence(
                [
                    slide_sample.to_description(self.stains)
                    for slide_sample in self.samples
                ]
            )
        else:
            dataset.SpecimenDescriptionSequence = DicomSequence()

    @classmethod
    def from_dataset(cls, dataset: Dataset):
        identifier = dataset.ContainerIdentifier
        samples, stains = SlideSample.from_dataset(dataset)

        return cls(identifier=identifier, stains=stains, samples=samples)
