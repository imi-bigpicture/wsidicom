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
    def meanings(cls) -> List[str]:
        raise NotImplementedError

    @property
    def sequence(self) -> DicomSequence:
        """Return code as DICOM sequence.

        Returns
        ----------
        DicomSequence
            Dicom sequence of dataset containing code.

        """
        ds = Dataset()
        ds.CodeValue = self.value
        ds.CodingSchemeDesignator = self.scheme_designator
        ds.CodeMeaning = self.meaning
        if self.scheme_version is not None:
            ds.CodeSchemeVersion = self.scheme_version
        sequence = DicomSequence([ds])
        return sequence

    def to_ds(self) -> Dataset:
        ds = Dataset()
        ds.CodeValue = self.value
        ds.CodingSchemeDesignator = self.scheme_designator
        ds.CodeMeaning = self.meaning
        if self.scheme_version is not None:
            ds.CodeSchemeVersion = self.scheme_version
        return ds

    def insert_into_ds(self, ds: Dataset) -> Dataset:
        try:
            sequence = getattr(ds, self.sequence_name)
        except AttributeError:
            sequence = DicomSequence([Dataset()])
            setattr(ds, self.sequence_name, sequence)
        sequence.append(self.to_ds())
        return ds

    @classmethod
    def _from_ds(
        cls,
        ds: Dataset,
    ) -> List[Optional[Code]]:
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
            return [None]
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
    @classmethod
    def from_ds(cls, ds: Dataset) -> 'ConceptCode':
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
        if isinstance(ds, Dataset):
            return cls._from_ds(ds)[0]
        raise NotImplementedError(ds)


class MultipleConceptCode(ConceptCode):
    @classmethod
    def from_ds(cls, ds: Dataset) -> List['ConceptCode']:
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
        if isinstance(ds, Dataset):
            return cls._from_ds(ds)
        raise NotImplementedError(ds)


class CidConceptCode(ConceptCode, metaclass=ABCMeta):
    cid: str

    def __init__(
        self,
        meaning: str,
        value: str = None,
        scheme_designator: str = None,
        scheme_version: str = None
    ):
        if value is None or scheme_designator is None:
            code = self._from_cid(value)
        else:
            code = Code(value, scheme_designator, meaning, scheme_version)
        super().__init__(*code)

    @classmethod
    def _get_cid_dict(cls):
        try:
            return getattr(codes, cls.cid)
        except AttributeError:
            raise NotImplementedError("Unsupported cid")

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
        cid_dict = cls._get_cid_dict()
        try:
            return getattr(cid_dict, meaning)
        except AttributeError:
            raise NotImplementedError("Unsupported code")

    @classmethod
    def meanings(cls) -> List[str]:
        cid_dict = cls._get_cid_dict()
        return cid_dict.dir()


class DictConceptCode(ConceptCode, metaclass=ABCMeta):
    code_dict: Dict[str, Code]

    def __init__(
        self,
        meaning: str,
        value: str = None,
        scheme_designator: str = None,
        scheme_version: str = None
    ):
        if value is None or scheme_designator is None:
            code = self._from_dict(meaning)
        else:
            code = Code(value, scheme_designator, meaning, scheme_version)
        super().__init__(*code)

    @classmethod
    def _from_dict(cls, meaning: str) -> Code:
        """Return ConceptCode from dictionary.

        Parameters
        ----------
        meaning: str
            Code meaning of  code to get.

        Returns
        ----------
        ConceptCode
            Code from dictionary.

        """
        try:
            return cls.code_dict[meaning]
        except KeyError:
            raise NotImplementedError("Unsupported code")

    @classmethod
    def meanings(cls) -> List[str]:
        return list(cls.code_dict.keys())


class MeasurementCode(DictConceptCode, SingleConceptCode):
    sequence_name = 'ConceptNameCodeSequence'
    code_dict = {'Area': Code('42798000', 'SCT', 'Area')}  # noqa


class AnnotationTypeCode(DictConceptCode, SingleConceptCode):
    sequence_name = 'AnnotationPropertyTypeCodeSequence'
    code_dict = {
        'Nucleus': Code('84640000', 'SCT', 'Nucleus'),  # noqa
        'Entire cell': Code('362837007', 'SCT', 'Entire cell')  # noqa
    }


class AnnotationCategoryCode(CidConceptCode, SingleConceptCode):
    sequence_name = 'AnnotationPropertyCategoryCodeSequence'
    cid = 'cid7150'


class IlluminationCode(CidConceptCode, MultipleConceptCode):
    sequence_name = 'IlluminationTypeCodeSequence'
    cid = 'cid8123'  # Microscopy Illumination Method


class LenseCode(CidConceptCode, MultipleConceptCode):
    sequence_name = 'LensesCodeSequence'
    cid = 'cid8121'  # Microscopy Lens Type


class LightPathFilterCode(CidConceptCode, MultipleConceptCode):
    sequence_name = 'LightPathFilterTypeStackCodeSequence'
    cid = 'cid8124'  # Microscopy Filter


class ImagePathFilterCode(CidConceptCode, MultipleConceptCode):
    sequence_name = 'ImagePathFilterTypeStackCodeSequence'
    cid = 'cid8124'  # Microscopy Filter


class IlluminationColorCode(DictConceptCode, SingleConceptCode):
    sequence_name = 'IlluminationColorCodeSequence'
    cid = 'cid8122',  # Microscopy Illuminator and Sensor Color


class IlluminatorCode(DictConceptCode, SingleConceptCode):
    sequence_name = 'IlluminatorTypeCodeSequence'
    cid = 'cid8125',  # Microscopy Illuminator Type


class ChannelDescriptionCode(DictConceptCode, SingleConceptCode):
    sequence_name = 'ChannelDescriptionCodeSequence'
    cid = 'cid8122',  # Microscopy Illuminator and Sensor Color


class UnitCode(SingleConceptCode):
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
        super().__init__(*code)

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
        return []
