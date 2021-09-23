from dataclasses import dataclass
from typing import List, Dict, Optional
from abc import ABCMeta, abstractmethod

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code


@dataclass
class ConceptCode(metaclass=ABCMeta):
    """Help functions for handling SR codes.
    Provides functions for converting between Code and dicom dataset.
    For CIDs that are not-yet standardized, functions for creating Code from
    code meaning is provided using the CID definitions in the sup 222 draft.
    For standardized CIDs one can use pydicom.sr.codedict to create codes."""
    sequence_name: str

    def __init__(
        self,
        meaning: str,
        value: str,
        scheme_designator: str,
        scheme_version: str = None
    ):
        self.meaning = meaning
        self.value = value
        self.scheme_designator = scheme_designator
        self.scheme_version = scheme_version

    def __hash__(self):
        return hash((
            self.value,
            self.scheme_designator,
            self.meaning,
            self.scheme_version
        ))

    @classmethod
    @abstractmethod
    def from_ds(cls, ds: Dataset):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def list(cls) -> List[str]:
        raise NotImplementedError

    @property
    def code(self) -> Code:
        return Code(
            value=self.value,
            scheme_designator=self.scheme_designator,
            meaning=self.meaning,
            scheme_version=self.scheme_version
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
        cls,
        ds: Dataset,
    ) -> Optional[List[Code]]:
        """Return list of ConceptCode from sequence in dataset.

        Parameters
        ----------
        ds: Dataset
            Datasete containing code sequence.

        Returns
        ----------
        List[ConceptCode]
            Codes created from sequence in dataset.

        """
        if cls.sequence_name not in ds:
            return None
        return [
            cls(
                value=code_ds.CodeValue,
                scheme_designator=code_ds.CodingSchemeDesignator,
                meaning=code_ds.CodeMeaning,
                scheme_version=getattr(code_ds, 'CodeSchemeVersion', None)
            )
            for code_ds in getattr(ds, cls.sequence_name)
        ]


class SingleConceptCode(ConceptCode):
    """Code for concepts that only allow a single item"""
    @classmethod
    def from_ds(cls, ds: Dataset) -> Optional['ConceptCode']:
        """Return measurement code for value. Value can be a code meaning (str)
        or a DICOM dataset containing the code.

        Parameters
        ----------
        value: Union[str, Dataset]
            The value for creating the code.

        Returns
        ----------
        ConceptCode
            Measurement code created from value.

        """
        codes = cls._from_ds(ds)
        if codes is None:
            return None
        return codes[0]


class MultipleConceptCode(ConceptCode):
    """Code for concepts that allow multiple items"""
    @classmethod
    def from_ds(cls, ds: Dataset) -> Optional[List['ConceptCode']]:
        """Return measurement code for value. Value can be a code meaning (str)
        or a DICOM dataset containing the code.

        Parameters
        ----------
        value: Union[str, Dataset]
            The value for creating the code.

        Returns
        ----------
        ConceptCode
            Measurement code created from value.

        """
        return cls._from_ds(ds)


class CidConceptCode(ConceptCode, metaclass=ABCMeta):
    """Code for concepts defined in Context groups"""
    cid: Dict[str, Code]

    def __init__(
        self,
        meaning: str,
        value: str = None,
        scheme_designator: str = None,
        scheme_version: str = None
    ):
        if value is None or scheme_designator is None:
            code = self._from_cid(meaning)
        else:
            code = Code(value, scheme_designator, meaning, scheme_version)
        super().__init__(
            meaning=code.meaning,
            value=code.value,
            scheme_designator=code.scheme_designator,
            scheme_version=code.scheme_version
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
        raise ValueError("Unsupported code")

    @classmethod
    def list(cls) -> List[str]:
        """Return possible meanings for concept.

        Returns
        ----------
        List[str]
            Possible meanings for concept.

        """
        return [code.meaning for code in cls.cid.values()]


class UnitCode(SingleConceptCode):
    """Code for concepts representing units according to UCUM scheme"""
    sequence_name = 'MeasurementUnitsCodeSequence'

    def __init__(
        self,
        meaning: str,
        value: str = None,
        scheme_designator: str = None,
        scheme_version: str = None
    ):
        if value is None or scheme_designator is None:
            code = self._from_ucum(meaning)
        else:
            code = Code(value, scheme_designator, meaning, scheme_version)
        super().__init__(
            meaning=code.meaning,
            value=code.value,
            scheme_designator=code.scheme_designator,
            scheme_version=code.scheme_version
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
        return Code(
            value=unit,
            scheme_designator='UCUM',
            meaning=unit
        )

    @classmethod
    def measnings(cls) -> List[str]:
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
    sequence_name = 'ConceptNameCodeSequence'
    cid = {'Area': Code('42798000', 'SCT', 'Area')}


class AnnotationTypeCode(CidConceptCode, SingleConceptCode):
    """
    Concept code for annotation type.
    Microscopy Annotation Property Types
    """
    sequence_name = 'AnnotationPropertyTypeCodeSequence'
    cid = {
        'Nucleus': Code('84640000', 'SCT', 'Nucleus'),
        'EntireCell': Code('362837007', 'SCT', 'Entire cell')
    }


class AnnotationCategoryCode(CidConceptCode, SingleConceptCode):
    """
    Concept code for annotation category.
    From CID 7150 Segmentation Property Categories
    """
    sequence_name = 'AnnotationPropertyCategoryCodeSequence'
    cid = codes.cid7150.concepts  # Segmentation Property Categories


class IlluminationCode(CidConceptCode, MultipleConceptCode):
    """
    Concept code for illumination type.
    From CID 8123 Microscopy Illumination Method
    """
    sequence_name = 'IlluminationTypeCodeSequence'
    cid = codes.cid8123.concepts  # Microscopy Illumination Method


class LenseCode(CidConceptCode, MultipleConceptCode):
    """
    Concept code for lense.
    From CID 8121 Microscopy Lens Type
    """
    sequence_name = 'LensesCodeSequence'
    cid = codes.cid8121.concepts  # Microscopy Lens Type


class LightPathFilterCode(CidConceptCode, MultipleConceptCode):
    """
    Concept code for light path filter.
    From CID 8124 Microscopy Filter
    """
    sequence_name = 'LightPathFilterTypeStackCodeSequence'
    cid = codes.cid8124.concepts  # Microscopy Filter


class ImagePathFilterCode(CidConceptCode, MultipleConceptCode):
    """
    Concept code for image path filter.
    From CID 8124 Microscopy Filter
    """
    sequence_name = 'ImagePathFilterTypeStackCodeSequence'
    cid = codes.cid8124.concepts  # Microscopy Filter


class IlluminationColorCode(CidConceptCode, SingleConceptCode):
    """
    Concept code for illumination color.
    From CID 8122 Microscopy Illuminator and Sensor Color
    """
    sequence_name = 'IlluminationColorCodeSequence'
    cid = codes.cid8122.concepts  # Microscopy Illuminator and Sensor Color


class IlluminatorCode(CidConceptCode, SingleConceptCode):
    """
    Concept code for illuminator type.
    From CID 8125 Microscopy Illuminator Type
    """
    sequence_name = 'IlluminatorTypeCodeSequence'
    cid = codes.cid8125.concepts  # Microscopy Illuminator Type


class ChannelDescriptionCode(CidConceptCode, SingleConceptCode):
    """
    Concept code for channel descriptor.
    From CID 8122 Microscopy Illuminator and Sensor Color
    """
    sequence_name = 'ChannelDescriptionCodeSequence'
    cid = codes.cid8122.concepts  # Microscopy Illuminator and Sensor Color


class SpecimenCollectionProcedureCode(CidConceptCode, SingleConceptCode):
    sequence_name = 'ConceptCodeSequence'
    cid = codes.cid8109.concepts


class SpecimenSamplingProcedureCode(CidConceptCode, SingleConceptCode):
    sequence_name = 'ConceptCodeSequence'
    cid = codes.cid8110.concepts


class SpecimenPreparationProcedureCode(CidConceptCode, SingleConceptCode):
    sequence_name = 'ConceptCodeSequence'
    cid = codes.cid8111.concepts


class SpecimenStainsCode(CidConceptCode, SingleConceptCode):
    sequence_name = 'ConceptCodeSequence'
    cid = codes.cid8112.concepts


class SpecimenPreparationStepsCode(CidConceptCode, SingleConceptCode):
    sequence_name = 'ConceptCodeSequence'
    cid = codes.cid8113.concepts


class SpecimenFixativesCode(CidConceptCode, SingleConceptCode):
    sequence_name = 'ConceptCodeSequence'
    cid = codes.cid8114.concepts


class SpecimenEmbeddingMediaCode(CidConceptCode, SingleConceptCode):
    sequence_name = 'ConceptCodeSequence'
    cid = codes.cid8115.concepts


class AnatomicPathologySpecimenTypesCode(CidConceptCode, SingleConceptCode):
    sequence_name = 'ConceptCodeSequence'
    cid = codes.cid8103.concepts


class ConceptNameCode(SingleConceptCode):
    sequence_name = 'ConceptNameCodeSequence'

    @classmethod
    def list(cls) -> List[str]:
        return []
