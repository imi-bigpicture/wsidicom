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

from dataclasses import dataclass
from typing import List, Dict, Optional, Type, TypeVar

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code


ConceptCodeType = TypeVar("ConceptCodeType", bound="ConceptCode")
CidConceptCodeType = TypeVar("CidConceptCodeType", bound="CidConceptCode")


@dataclass
class ConceptCode:
    """Help functions for handling SR codes.
    Provides functions for converting between Code and dicom dataset.
    """

    sequence_name: str

    def __init__(
        self,
        value: str,
        scheme_designator: str,
        meaning: str,
        scheme_version: Optional[str] = None,
    ):
        self.value = value
        self.scheme_designator = scheme_designator
        self.meaning = meaning
        self.scheme_version = scheme_version

    def __hash__(self):
        return hash(
            (self.value, self.scheme_designator, self.meaning, self.scheme_version)
        )

    @property
    def code(self) -> Code:
        return Code(
            value=self.value,
            scheme_designator=self.scheme_designator,
            meaning=self.meaning,
            scheme_version=self.scheme_version,
        )

    @classmethod
    def from_code(cls: Type[ConceptCodeType], code: Code) -> ConceptCodeType:
        return cls(
            code.value, code.scheme_designator, code.meaning, code.scheme_version
        )

    def to_ds(self) -> Dataset:
        """Codes code into DICOM dataset.

        Returns
        ----------
        Dataset
            Dataset of code.

        """
        ds = Dataset()
        ds.CodeValue = self.value
        ds.CodingSchemeDesignator = self.scheme_designator
        ds.CodeMeaning = self.meaning
        if self.scheme_version is not None:
            ds.CodeSchemeVersion = self.scheme_version
        return ds

    def insert_into_ds(self, ds: Dataset) -> Dataset:
        """Codes and insert object into sequence in dataset.

        Parameters
        ----------
        ds: Dataset
           Dataset to insert into.

        Returns
        ----------
        Dataset
            Dataset with object inserted.

        """
        # Append if sequence already set otherwise create
        try:
            sequence = getattr(ds, self.sequence_name)
            sequence.append(self.to_ds())
        except AttributeError:
            setattr(ds, self.sequence_name, DicomSequence([self.to_ds()]))
        return ds

    @classmethod
    def _from_ds(
        cls: Type[ConceptCodeType],
        ds: Dataset,
    ) -> Optional[List[ConceptCodeType]]:
        """Return list of ConceptCode from sequence in dataset.

        Parameters
        ----------
        ds: Dataset
            Datasete containing code sequence.

        Returns
        ----------
        Optional[List[ConceptCodeType]
            Codes created from sequence in dataset.

        """
        if cls.sequence_name not in ds:
            return None
        return [
            cls(
                value=code_ds.CodeValue,
                scheme_designator=code_ds.CodingSchemeDesignator,
                meaning=code_ds.CodeMeaning,
                scheme_version=getattr(code_ds, "CodeSchemeVersion", None),
            )
            for code_ds in getattr(ds, cls.sequence_name)
        ]


class SingleConceptCode(ConceptCode):
    """Code for concepts  # type: ignore that only allow a single item"""

    @classmethod
    def from_ds(cls: Type[ConceptCodeType], ds: Dataset) -> Optional["ConceptCodeType"]:
        """Return measurement code for value. Value can be a code meaning (str)
        or a DICOM dataset containing the code.

        Parameters
        ----------
        value: Dataset
            The dataset for creating the code.

        Returns
        ----------
        Optional[ConceptCodeType]
            Concept code created from value.

        """
        codes = cls._from_ds(ds)
        if codes is None:
            return None
        return codes[0]


class MultipleConceptCode(ConceptCode):
    """Code for concepts  # type: ignore that allow multiple items"""

    @classmethod
    def from_ds(cls: Type[ConceptCodeType], ds: Dataset) -> List[ConceptCodeType]:
        """Return measurement code for value. Value can be a code meaning (str)
        or a DICOM dataset containing the code.

        Parameters
        ----------
        value: Dataset
            The dataset for creating the code.

        Returns
        ----------
        List[ConceptCodeType]
            Concept codes created from dataset.

        """
        codes = cls._from_ds(ds)
        if codes is None:
            return []
        return codes


class CidConceptCode(ConceptCode):
    """Code for concepts  # type: ignore defined in Context groups"""

    cid: Dict[str, Code]

    def __init__(
        self,
        meaning: str,
        value: Optional[str] = None,
        scheme_designator: Optional[str] = None,
        scheme_version: Optional[str] = None,
    ):
        if value is None or scheme_designator is None:
            code = self._from_cid(meaning)
        else:
            code = Code(value, scheme_designator, meaning, scheme_version)
        super().__init__(
            meaning=code.meaning,
            value=code.value,
            scheme_designator=code.scheme_designator,
            scheme_version=code.scheme_version,
        )

    @classmethod
    def _from_cid(cls, meaning: str) -> Code:
        """Return ConceptCode from CID and meaning. For a list of CIDs, see
        http://dicom.nema.org/medical/dicom/current/output/chtml/part16/chapter_B.html # noqa

        Parameters
        ----------
        meaning: str
            Code meaning to get.

        Returns
        ----------
        ConceptCode
            Code created from CID and meaning.

        """
        # The keys are camelcase meanings, dont use.
        for code in cls.cid.values():
            if code.meaning == meaning:
                return code
        raise ValueError(f"Unsupported code with meaning {meaning}.")

    @classmethod
    def list(cls) -> List[str]:
        """Return possible meanings for concept.

        Returns
        ----------
        List[str]
            Possible meanings for concept.

        """
        return [code.meaning for code in cls.cid.values()]

    @classmethod
    def from_code_value(
        cls: Type[CidConceptCodeType], value: str
    ) -> CidConceptCodeType:
        try:
            code = next((code for code in cls.cid.values() if code.value == value))
        except StopIteration:
            raise ValueError("Unsupported code.")
        return cls(
            code.meaning, code.value, code.scheme_designator, code.scheme_version
        )


class UnitCode(SingleConceptCode):
    """Code for concepts representing units according to UCUM scheme"""

    sequence_name = "MeasurementUnitsCodeSequence"

    def __init__(
        self,
        meaning: str,
        value: Optional[str] = None,
        scheme_designator: Optional[str] = None,
        scheme_version: Optional[str] = None,
    ):
        if value is None or scheme_designator is None:
            code = self._from_ucum(meaning)
        else:
            code = Code(value, scheme_designator, meaning, scheme_version)
        super().__init__(
            meaning=code.meaning,
            value=code.value,
            scheme_designator=code.scheme_designator,
            scheme_version=code.scheme_version,
        )

    @classmethod
    def _from_ucum(cls, unit: str) -> Code:
        """Return UCUM scheme ConceptCode.

        Parameters
        ----------
        meaning: str
            Code meaning.

        Returns
        ----------
        ConceptCode
            Code created from meaning.

        """
        return Code(value=unit, scheme_designator="UCUM", meaning=unit)

    @classmethod
    def meanings(cls) -> List[str]:
        """Return possible meanings for concept.

        Returns
        ----------
        List[str]
            Possible meanings for concept.

        """
        return []


class MeasurementCode(CidConceptCode, SingleConceptCode):
    """
    Concept code for measurement type.
    Microscopy Measurement Types
    """

    sequence_name = "ConceptNameCodeSequence"
    cid = {"Area": Code("42798000", "SCT", "Area")}


class AnnotationTypeCode(CidConceptCode, SingleConceptCode):
    """
    Concept code for annotation type.
    Microscopy Annotation Property Types
    """

    sequence_name = "AnnotationPropertyTypeCodeSequence"
    cid = {
        "Nucleus": Code("84640000", "SCT", "Nucleus"),
        "EntireCell": Code("362837007", "SCT", "Entire cell"),
    }


class AnnotationCategoryCode(CidConceptCode, SingleConceptCode):
    """
    Concept code for annotation category.
    From CID 7150 Segmentation Property Categories
    """

    sequence_name = "AnnotationPropertyCategoryCodeSequence"
    cid = codes.cid7150.concepts  # type: ignore


class IlluminationCode(CidConceptCode, MultipleConceptCode):
    """
    Concept code for illumination type.
    From CID 8123 Microscopy Illumination Method
    """

    sequence_name = "IlluminationTypeCodeSequence"
    cid = codes.cid8123.concepts  # type: ignore


class LenseCode(CidConceptCode, MultipleConceptCode):
    """
    Concept code for lense.
    From CID 8121 Microscopy Lens Type
    """

    sequence_name = "LensesCodeSequence"
    cid = codes.cid8121.concepts  # type: ignore


class LightPathFilterCode(CidConceptCode, MultipleConceptCode):
    """
    Concept code for light path filter.
    From CID 8124 Microscopy Filter
    """

    sequence_name = "LightPathFilterTypeStackCodeSequence"
    cid = codes.cid8124.concepts  # type: ignore


class ImagePathFilterCode(CidConceptCode, MultipleConceptCode):
    """
    Concept code for image path filter.
    From CID 8124 Microscopy Filter
    """

    sequence_name = "ImagePathFilterTypeStackCodeSequence"
    cid = codes.cid8124.concepts  # type: ignore


class IlluminationColorCode(CidConceptCode, SingleConceptCode):
    """
    Concept code for illumination color.
    From CID 8122 Microscopy Illuminator and Sensor Color
    """

    sequence_name = "IlluminationColorCodeSequence"
    cid = codes.cid8122.concepts  # type: ignore


class IlluminatorCode(CidConceptCode, SingleConceptCode):
    """
    Concept code for illuminator type.
    From CID 8125 Microscopy Illuminator Type
    """

    sequence_name = "IlluminatorTypeCodeSequence"
    cid = codes.cid8125.concepts  # type: ignore


class ChannelDescriptionCode(CidConceptCode, MultipleConceptCode):
    """
    Concept code for channel descriptor.
    From CID 8122 Microscopy Illuminator and Sensor Color
    """

    sequence_name = "ChannelDescriptionCodeSequence"
    cid = codes.cid8122.concepts  # type: ignore


class SpecimenCollectionProcedureCode(CidConceptCode, SingleConceptCode):
    sequence_name = "ConceptCodeSequence"
    cid = codes.cid8109.concepts  # type: ignore


class SpecimenSamplingProcedureCode(CidConceptCode, SingleConceptCode):
    sequence_name = "ConceptCodeSequence"
    cid = codes.cid8110.concepts  # type: ignore


class SpecimenPreparationProcedureCode(CidConceptCode, SingleConceptCode):
    sequence_name = "ConceptCodeSequence"
    cid = codes.cid8111.concepts  # type: ignore


class SpecimenStainsCode(CidConceptCode, SingleConceptCode):
    sequence_name = "ConceptCodeSequence"
    cid = codes.cid8112.concepts  # type: ignore


class SpecimenPreparationStepsCode(CidConceptCode, SingleConceptCode):
    sequence_name = "ConceptCodeSequence"
    cid = codes.cid8113.concepts  # type: ignore


class SpecimenFixativesCode(CidConceptCode, SingleConceptCode):
    sequence_name = "ConceptCodeSequence"
    cid = codes.cid8114.concepts  # type: ignore


class SpecimenEmbeddingMediaCode(CidConceptCode, SingleConceptCode):
    sequence_name = "ConceptCodeSequence"
    cid = codes.cid8115.concepts  # type: ignore


class AnatomicPathologySpecimenTypesCode(CidConceptCode, SingleConceptCode):
    sequence_name = "ConceptCodeSequence"
    cid = codes.cid8103.concepts  # type: ignore


class ContainerComponentTypeCode(CidConceptCode, SingleConceptCode):
    sequence_name = "ConceptCodeSequence"
    cid = codes.cid8102.concepts  # type: ignore


class ContainerTypeCode(CidConceptCode, SingleConceptCode):
    sequence_name = "ConceptCodeSequence"
    cid = codes.cid8101.concepts  # type: ignore


class ConceptNameCode(SingleConceptCode):
    sequence_name = "ConceptNameCodeSequence"

    @classmethod
    def list(cls) -> List[str]:
        return []
