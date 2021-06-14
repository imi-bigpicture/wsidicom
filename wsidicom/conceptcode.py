from typing import Union

from pydicom.dataset import Dataset
from pydicom.sequence import Sequence as DicomSequence
from pydicom.sr.codedict import codes
from pydicom.sr.coding import Code


class ConceptCode(Code):
    """Help functions for handling SR codes.
    Provides functions for converting between Code and dicom dataset.
    For CIDs that are not-yet standardized, functions for creating Code from
    code meaning is provided using the CID definitions in the sup 222 draft.
    For standardized CIDs one can use pydicom.sr.codedict to create codes."""
    code_dictionaries = {
        'measurement': {'Area': Code('42798000', 'SCT', 'Area')},  # noqa
        'typecode': {
            'Nucleus': Code('84640000', 'SCT', 'Nucleus'),  # noqa
            'Entire cell': Code('362837007', 'SCT', 'Entire cell')  # noqa
        }
    }
    measurement_dictionary = {'Area': Code('42798000', 'SCT', 'Area')}  # noqa

    typecode_dictionary = {
        'Nucleus': Code('84640000', 'SCT', 'Nucleus'),  # noqa
        'Entire cell': Code('362837007', 'SCT', 'Entire cell')  # noqa
    }

    def __hash__(self):
        return hash((
            self.value,
            self.scheme_designator,
            self.meaning,
            self.scheme_version
        ))

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

    @classmethod
    def measurement(cls, value: Union[str, Dataset]) -> 'ConceptCode':
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
        if isinstance(value, str):
            return cls._from_dict('measurement', value)
        elif isinstance(value, Dataset):
            return cls._from_ds(value, 'ConceptNameCodeSequence')
        raise NotImplementedError(value)

    @classmethod
    def type(cls, value: Union[str, Dataset]) -> 'ConceptCode':
        """Return typecode code for value. Value can be a code meaning
        (str) or a DICOM dataset containing the code.

        Parameters
        ----------
        value: Union[str, Dataset]
            The value for creating the code.

        Returns
        ----------
        ConceptCode
            typecode code created from value.

        """
        if isinstance(value, str):
            return cls._from_dict('typecode', value)
        elif isinstance(value, Dataset):
            return cls._from_ds(value, 'AnnotationPropertyTypeCodeSequence')
        raise NotImplementedError(value)

    @classmethod
    def category(cls, value: Union[str, Dataset]) -> 'ConceptCode':
        """Return categorycode code for value. Value can be a code meaning
        (str) or a DICOM dataset containing the code.

        Parameters
        ----------
        value: Union[str, Dataset]
            The value for creating the code.

        Returns
        ----------
        ConceptCode
            categorycode code created from value.

        """
        if isinstance(value, str):
            return cls._from_cid('cid7150', value)
        elif isinstance(value, Dataset):
            return cls._from_ds(
                value,
                'AnnotationPropertyCategoryCodeSequence'
            )
        raise NotImplementedError(value)

    @classmethod
    def unit(cls, value: Union[str, Dataset]) -> 'ConceptCode':
        """Return unit code for value. Value can be a code meaning
        (str) or a DICOM dataset containing the code.

        Parameters
        ----------
        value: Union[str, Dataset]
            The value for creating the code.

        Returns
        ----------
        ConceptCode
            Unit code created from value.

        """
        if isinstance(value, str):
            return cls._from_ucum(value)
        elif isinstance(value, Dataset):
            return cls._from_ds(value, 'MeasurementUnitsCodeSequence')
        raise NotImplementedError(value)

    @classmethod
    def _from_ds(cls, ds: Dataset, sequence_name: str) -> 'ConceptCode':
        """Return ConceptCode from sequence in dataset.

        Parameters
        ----------
        ds: Dataset
            Datasete containing code sequence.
        sequence_name: str
            Name of the sequence containing code.

        Returns
        ----------
        ConceptCode
            Code created from sequence in dataset.

        """
        code_ds = getattr(ds, sequence_name)[0]
        value = code_ds.CodeValue
        scheme = code_ds.CodingSchemeDesignator
        meaning = code_ds.CodeMeaning
        version = getattr(code_ds, 'CodeSchemeVersion', None)
        return cls(
            value=value,
            scheme_designator=scheme,
            meaning=meaning,
            scheme_version=version
        )

    @classmethod
    def _from_dict(cls, dict_name: str, meaning: str) -> 'ConceptCode':
        """Return ConceptCode from dictionary.

        Parameters
        ----------
        dict_name: str
            Dictionary name to get code from.
        meaning: str
            Code meaning of  code to get.

        Returns
        ----------
        ConceptCode
            Code from dictionary.

        """
        try:
            code_dict = cls.code_dictionaries[dict_name]
        except KeyError:
            raise NotImplementedError("Unkown dictionary")
        try:
            code = code_dict[meaning]
        except KeyError:
            raise NotImplementedError("Unsupported code")
        return cls(*code)

    @classmethod
    def _from_ucum(cls, unit: str) -> 'ConceptCode':
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
        return cls(
            value=unit,
            scheme_designator='UCUM',
            meaning=unit
        )

    @classmethod
    def _from_cid(cls, cid: str, meaning: str) -> 'ConceptCode':
        """Return ConceptCode from CID and meaning. For a list of CIDs, see
        http://dicom.nema.org/medical/dicom/current/output/chtml/part16/chapter_B.html # noqa

        Parameters
        ----------
        cid: str
            CID to use.
        meaning: str
            Code meaning to get.

        Returns
        ----------
        ConceptCode
            Code created from CID and meaning.

        """
        try:
            cid = getattr(codes, cid)
        except AttributeError:
            raise NotImplementedError("Unsupported cid")
        try:
            code = getattr(cid, meaning)
        except AttributeError:
            raise NotImplementedError("Unsupported code")
        return cls(*code)
