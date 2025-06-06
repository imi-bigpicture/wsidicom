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

"""DICOM schema for Optical path model."""

import io
import struct
from dataclasses import replace
from enum import Enum
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import numpy as np
from marshmallow import fields, post_load, pre_dump
from pydicom import Dataset
from pydicom.sr.coding import Code
from pydicom.valuerep import VR

from wsidicom.conceptcode import (
    IlluminationCode,
    IlluminationColorCode,
    ImagePathFilterCode,
    LenseCode,
    LightPathFilterCode,
)
from wsidicom.metadata.optical_path import (
    ConstantLutSegment,
    DiscreteLutSegment,
    ImagePathFilter,
    LightPathFilter,
    LinearLutSegment,
    Lut,
    LutDataType,
    LutSegment,
    Objectives,
    OpticalFilter,
    OpticalPath,
)
from wsidicom.metadata.schema.dicom.defaults import defaults
from wsidicom.metadata.schema.dicom.fields import (
    CodeDicomField,
    DefaultingDicomField,
    FlattenOnDumpNestedDicomField,
    FloatDicomField,
    SingleCodeSequenceField,
    StringDicomField,
)
from wsidicom.metadata.schema.dicom.schema import (
    DicomSchema,
    LoadType,
    ModuleDicomSchema,
)


class LutSegmentType(Enum):
    DISCRETE = 0
    LINEAR = 1
    INDIRECT = 2


class LutDicomParser:
    @classmethod
    def from_dataset(cls, lut_sequence: Sequence[Dataset]) -> Optional[Lut]:
        """Read LUT from a DICOM lut sequence.

        Parameters
        ----------
        lut_sequence : Sequence[Dataset]
            DICOM optical path dataset.

        Returns
        -------
        Lut
            Parsed LUT.

        """
        dataset = lut_sequence[0]
        length, red_start, bits = dataset.RedPaletteColorLookupTableDescriptor
        _, green_start, _ = dataset.GreenPaletteColorLookupTableDescriptor
        _, blue_start, _ = dataset.BluePaletteColorLookupTableDescriptor
        if length == 0:
            length = 2**16
        bits = bits
        if bits == 8:
            data_type = np.uint8
        else:
            data_type = np.uint16

        segmented_keys = (
            "SegmentedRedPaletteColorLookupTableData",
            "SegmentedGreenPaletteColorLookupTableData",
            "SegmentedBluePaletteColorLookupTableData",
        )
        non_segmented_keys = (
            "RedPaletteColorLookupTableData",
            "GreenPaletteColorLookupTableData",
            "BluePaletteColorLookupTableData",
        )
        if all(key in dataset for key in segmented_keys):
            red = list(
                cls._parse_segments(
                    dataset.SegmentedRedPaletteColorLookupTableData, data_type
                )
            )
            green = list(
                cls._parse_segments(
                    dataset.SegmentedGreenPaletteColorLookupTableData, data_type
                )
            )
            blue = list(
                cls._parse_segments(
                    dataset.SegmentedBluePaletteColorLookupTableData, data_type
                )
            )

        elif all(key in dataset for key in non_segmented_keys):
            red: List[LutSegment] = [
                cls._parse_single_discrete_segment(
                    dataset.RedPaletteColorLookupTableData, data_type
                )
            ]
            green: List[LutSegment] = [
                cls._parse_single_discrete_segment(
                    dataset.GreenPaletteColorLookupTableData, data_type
                )
            ]
            blue: List[LutSegment] = [
                cls._parse_single_discrete_segment(
                    dataset.BluePaletteColorLookupTableData, data_type
                )
            ]

        else:
            raise ValueError(
                "Lookup table data or segmented lookup table data missing for one "
                "or more components."
            )
        for color, start in (
            (red, red_start),
            (green, green_start),
            (blue, blue_start),
        ):
            cls._add_start_and_end(start, length, color)

        return Lut(
            red=red,
            green=green,
            blue=blue,
            data_type=data_type,
        )

    @classmethod
    def _parse_segments(
        cls, segmented_lut_data: bytes, data_type: LutDataType
    ) -> Iterator[LutSegment]:
        """Parse segments from segmented lut data."""
        previous_segment_type: Optional[LutSegmentType] = None
        previous_segment_end_value: Optional[int] = None
        with io.BytesIO(segmented_lut_data) as buffer:
            data_type = cls._determine_correct_data_type(buffer, data_type)
            next_segment_type = cls._read_next_segment_type(buffer, data_type)
            while next_segment_type is not None:
                segment_type = next_segment_type
                if segment_type == LutSegmentType.DISCRETE:
                    length = cls._read_value(buffer, data_type)
                    values = cls._read_values(buffer, length, data_type)
                    next_segment_type = cls._read_next_segment_type(buffer, data_type)
                    if next_segment_type == LutSegmentType.LINEAR:
                        # If next segment is linear it will take over the last value.
                        previous_segment_end_value = values.pop()
                    if len(values) > 0:
                        yield DiscreteLutSegment(values)
                    previous_segment_type = segment_type

                elif segment_type == LutSegmentType.LINEAR:
                    length = cls._read_value(buffer, data_type)
                    end_value = cls._read_value(buffer, data_type)
                    if previous_segment_end_value is not None:
                        start_value = previous_segment_end_value
                        if previous_segment_type == LutSegmentType.DISCRETE:
                            # If the previous segment was discrete, this segment
                            # takes over its last value.
                            length += 1
                    else:
                        # The standard does allow the first segment to be a linear
                        # segment, but if it happens it is likely that the first value
                        # should be 0
                        start_value = 0
                    if start_value == end_value:
                        yield ConstantLutSegment(start_value, length)
                    else:
                        yield LinearLutSegment(start_value, end_value, length)
                    previous_segment_type = segment_type
                    next_segment_type = cls._read_next_segment_type(buffer, data_type)
                    previous_segment_end_value = end_value
                elif segment_type == LutSegmentType.INDIRECT:
                    raise NotImplementedError(
                        "Indirect segment types are not implemented."
                    )
                else:
                    raise ValueError("Unknown segment type.")

    @classmethod
    def _add_start_and_end(
        cls, start_length: int, total_length: int, segments: List[LutSegment]
    ):
        """Add start and end constant segments if needed."""
        start_segment = cls._create_start_segment(start_length, segments)
        end_segment = cls._create_end_segment(start_length, total_length, segments)
        if start_segment is not None:
            segments.insert(0, start_segment)
        if end_segment is not None:
            segments.append(end_segment)

    @staticmethod
    def _create_start_segment(
        start_length: int, segments: List[LutSegment]
    ) -> Optional[ConstantLutSegment]:
        """Create a start segment if needed."""
        if start_length == 0:
            return None
        first_segment = segments[0]
        if isinstance(first_segment, ConstantLutSegment):
            first_segment = replace(
                first_segment, length=first_segment.length + start_length
            )
            segments[0] = first_segment
            return None
        if isinstance(first_segment, DiscreteLutSegment):
            start_value = first_segment.values[0]
        elif isinstance(first_segment, LinearLutSegment):
            start_value = first_segment.start_value
        else:
            raise ValueError("Unknown segment type.")
        return ConstantLutSegment(start_value, start_length)

    @staticmethod
    def _create_end_segment(
        start_length, total_length: int, segments: List[LutSegment]
    ):
        """Create a end segment if needed."""
        length = start_length
        last_segment = segments[-1]
        length = sum(len(segment) for segment in segments)
        segment_length = total_length - length
        if segment_length < 0:
            raise ValueError("Got a negative length for last segment.")
        if segment_length == 0:
            return None
        if isinstance(last_segment, ConstantLutSegment):
            last_segment = replace(
                last_segment, length=last_segment.length + segment_length
            )
            segments[-1] = last_segment
            return None
        if isinstance(last_segment, DiscreteLutSegment):
            end_value = last_segment.values[-1]
        elif isinstance(last_segment, LinearLutSegment):
            end_value = last_segment.end_value
        else:
            raise ValueError("Unknown segment type.")
        return ConstantLutSegment(end_value, segment_length)

    @classmethod
    def _read_next_segment_type(
        cls, buffer: io.BytesIO, data_type: LutDataType
    ) -> Optional[LutSegmentType]:
        """
        Read next segment type from buffer.

        Return None if not enough data left to read.
        """
        try:
            return LutSegmentType(cls._read_value(buffer, data_type))
        except struct.error:
            return None

    @staticmethod
    def _read_value(buffer: io.BytesIO, data_type: LutDataType) -> int:
        """Read a single value from buffer."""
        if data_type == np.uint8:
            format = "<B"
        else:
            format = "<H"
        return struct.unpack(format, buffer.read(np.dtype(data_type).itemsize))[0]

    @staticmethod
    def _read_values(
        buffer: io.BytesIO, count: int, data_type: LutDataType
    ) -> List[int]:
        """Read multiple values from buffer."""
        if data_type == np.uint8:
            format = f'<{count*"B"}'
        else:
            format = f'<{count*"H"}'
        return list(
            struct.unpack(format, buffer.read(np.dtype(data_type).itemsize * count))
        )

    @classmethod
    def _parse_single_discrete_segment(
        cls, segment_data: bytes, data_type: LutDataType
    ) -> DiscreteLutSegment:
        """Read discrete segment from data."""
        length = len(segment_data) // np.dtype(data_type).itemsize
        if data_type == np.uint8:
            format = f'<{length*"B"}'
        else:
            format = f'<{length*"H"}'
        values = list(struct.unpack(format, segment_data))
        return DiscreteLutSegment(values)

    @classmethod
    def _determine_correct_data_type(
        cls, buffer: io.BytesIO, data_type: LutDataType
    ) -> LutDataType:
        """Determine correct data type for reading segment.

        The segment can either have 8- or 16-bits values. The first value indicates the
        segment type and should be 0, 1, or 2. The second value indicates a length that
        should be larger than 0.

        If reading 8 bits values as 16 bits the segment type will not be 0, 1, or 2, as
        the length value is also read. Then re-try reading as 8 bits.

        If reading 16 bit values as 8 bits the segment length will not be larger than 0,
        as the read value is the second half of the segment type value. Then re-try
        reading as 16 bits.
        """
        buffer.seek(0)
        try:
            cls._read_next_segment_type(buffer, data_type)
        except ValueError as exception:
            # Throws if first read data is not a valid segment type (0, 1, or 2.)
            if data_type == np.uint16:
                # Try reading the segment type as 8 bits
                return cls._determine_correct_data_type(buffer, np.uint8)
            raise ValueError("Failed to parse first segment type from data", exception)

        length = cls._read_value(buffer, data_type)
        if length > 0:
            # Length should be positive for all segment types
            buffer.seek(0)
            return data_type
        if data_type == np.uint8:
            # Try reading the segment as 16 bits.
            return cls._determine_correct_data_type(buffer, np.uint16)
        raise ValueError("Failed to parse first segment length from data.")


class LutDicomFormatter:
    @classmethod
    def to_dataset(cls, lut: Lut) -> List[Dataset]:
        """Convert lut into dataset."""
        dataset = Dataset()
        if lut.length == 2**16:
            length = 0
        else:
            length = lut.length
        red_start, red_data = cls._pack_segments(lut.red, lut.data_type)
        green_start, green_data = cls._pack_segments(lut.green, lut.data_type)
        blue_start, blue_data = cls._pack_segments(lut.blue, lut.data_type)
        dataset.RedPaletteColorLookupTableDescriptor = (length, red_start, lut.bits)
        dataset.GreenPaletteColorLookupTableDescriptor = (length, green_start, lut.bits)
        dataset.BluePaletteColorLookupTableDescriptor = (length, blue_start, lut.bits)
        dataset.SegmentedRedPaletteColorLookupTableData = red_data
        dataset.SegmentedGreenPaletteColorLookupTableData = green_data
        dataset.SegmentedBluePaletteColorLookupTableData = blue_data
        return [dataset]

    @classmethod
    def _pack_segments(
        cls,
        segments: Sequence[LutSegment],
        data_type: LutDataType,
    ) -> Tuple[int, bytes]:
        """Pack segments into bytes.

        Return start position (if constant first segment could be skipped) and bytes.
        """
        if data_type == np.uint8:
            data_format = "B"
        else:
            data_format = "H"
        previous_segment: Optional[LutSegment] = None
        end_index = len(segments) - 1
        start = 0
        with io.BytesIO() as buffer:
            for index, segment in enumerate(segments):
                if isinstance(segment, ConstantLutSegment):
                    if index == 0 and len(segments) > 1:
                        # Starting with constant segment and not only segment, we can
                        # skip it and set start of segmented data to length of segment.
                        start = segment.length
                    elif index != end_index or len(segments) == 1:
                        # Last constant segment can be skipped if not only segment.
                        cls._pack_constant_segment(
                            buffer,
                            segment,
                            previous_segment,
                            data_format,
                        )
                elif isinstance(segment, DiscreteLutSegment):
                    # Get next segment if not last segment.
                    if index < end_index:
                        next_segment = segments[index + 1]
                    else:
                        next_segment = None
                    cls._pack_discrete_segment(
                        buffer, segment, next_segment, data_format
                    )
                elif isinstance(segment, LinearLutSegment):
                    cls._pack_linear_segment(
                        buffer, segment, previous_segment, data_format
                    )
                else:
                    raise NotImplementedError()
                previous_segment = segment
            return start, buffer.getvalue()

    @classmethod
    def _pack_discrete_segment(
        cls,
        buffer: io.BytesIO,
        segment: DiscreteLutSegment,
        next_segment: Optional[LutSegment],
        data_format: str,
    ):
        values = list(segment.values)
        next_segment = next_segment
        if isinstance(next_segment, LinearLutSegment):
            values.append(next_segment.start_value)
        cls._pack_discrete_data(buffer, values, data_format)

    @classmethod
    def _pack_constant_segment(
        cls,
        buffer: io.BytesIO,
        segment: ConstantLutSegment,
        previous_segment: Optional[LutSegment],
        data_format: str,
    ):
        if not isinstance(previous_segment, DiscreteLutSegment):
            cls._pack_discrete_data(
                buffer,
                [segment.value],
                data_format,
            )
        cls._pack_linear_data(buffer, segment.length - 1, segment.value, data_format)

    @classmethod
    def _pack_linear_segment(
        cls,
        buffer: io.BytesIO,
        segment: LinearLutSegment,
        previous_segment: Optional[LutSegment],
        data_format: str,
    ):
        if not isinstance(previous_segment, DiscreteLutSegment):
            cls._pack_discrete_data(
                buffer,
                [segment.start_value],
                data_format,
            )
        cls._pack_linear_data(
            buffer, segment.length - 1, segment.end_value, data_format
        )

    @staticmethod
    def _pack_discrete_data(
        buffer: io.BytesIO,
        values: Sequence[int],
        data_format: str,
    ):
        """Pack discrete segment to buffer."""
        buffer.write(
            struct.pack(
                "<" + 2 * data_format,
                LutSegmentType.DISCRETE.value,
                len(values),
            )
        )
        for value in values:
            buffer.write(struct.pack("<" + data_format, value))

    @staticmethod
    def _pack_linear_data(
        buffer: io.BytesIO,
        length: int,
        end_value: int,
        data_format: str,
    ):
        """Pack linear segment to buffer."""
        buffer.write(
            struct.pack(
                "<" + 3 * data_format,
                LutSegmentType.LINEAR.value,
                length,
                end_value,
            )
        )


class LutDicomField(fields.Field):
    def _serialize(self, value: Optional[Lut], attr: Optional[str], obj: Any, **kwargs):
        if value is None:
            return None
        return LutDicomFormatter.to_dataset(value)

    def _deserialize(
        self,
        value: Optional[Sequence[Dataset]],
        attr: Optional[str],
        data: Optional[Dict[str, Any]],
        **kwargs,
    ) -> Optional[Lut]:
        if value is None or len(value) == 0:
            return None
        return LutDicomParser.from_dataset(value)


class FilterDicomSchema(DicomSchema[LoadType]):
    @pre_dump
    def pre_dump(self, filter: OpticalFilter, **kwargs):
        return {
            "filters": filter.filters,
            "nominal": filter.nominal,
            "filter_band": [filter.low_pass, filter.high_pass],
        }

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs):
        filter_band = data.pop("filter_band", None)
        if filter_band is not None:
            data["low_pass"] = filter_band[0]
            data["high_pass"] = filter_band[1]
        return super().post_load(data, **kwargs)


class LightPathFilterDicomSchema(FilterDicomSchema[LightPathFilter]):
    filters = fields.List(
        CodeDicomField(LightPathFilterCode),
        data_key="LightPathFilterTypeStackCodeSequence",
        allow_none=True,
    )
    nominal = fields.Integer(
        data_key="LightPathFilterPassThroughWavelength", allow_none=True
    )
    low_pass = fields.Integer(load_only=True, allow_none=True)
    high_pass = fields.Integer(load_only=True, allow_none=True)
    filter_band = fields.List(fields.Integer(), data_key="LightPathFilterPassBand")

    @property
    def load_type(self) -> Type[LightPathFilter]:
        return LightPathFilter


class ImagePathFilterDicomSchema(FilterDicomSchema[ImagePathFilter]):
    filters = fields.List(
        CodeDicomField(ImagePathFilterCode),
        data_key="ImagePathFilterTypeStackCodeSequence",
        allow_none=True,
    )
    nominal = fields.Integer(
        data_key="ImagePathFilterPassThroughWavelength", allow_none=True
    )
    low_pass = fields.Integer(load_only=True, allow_none=True)
    high_pass = fields.Integer(load_only=True, allow_none=True)
    filter_band = fields.List(fields.Integer(), data_key="ImagePathFilterPassBand")

    @property
    def load_type(self) -> Type[ImagePathFilter]:
        return ImagePathFilter


class ObjectivesSchema(DicomSchema[Objectives]):
    lenses = fields.List(
        CodeDicomField(LenseCode), data_key="LensesCodeSequence", allow_none=True
    )
    condenser_power = FloatDicomField(data_key="CondenserLensPower", allow_none=True)
    objective_power = FloatDicomField(data_key="ObjectiveLensPower", allow_none=True)
    objective_numerical_aperture = FloatDicomField(
        data_key="ObjectiveLensNumericalAperture", allow_none=True
    )

    @property
    def load_type(self) -> Type[Objectives]:
        return Objectives


class OpticalPathDicomSchema(ModuleDicomSchema[OpticalPath]):
    identifier = DefaultingDicomField(
        StringDicomField(value_representation=VR.SH),
        data_key="OpticalPathIdentifier",
        load_default=None,
        dump_default=defaults.optical_path_identifier,
    )
    description = StringDicomField(
        value_representation=VR.ST, data_key="OpticalPathDescription", load_default=None
    )
    illumination_types = DefaultingDicomField(
        fields.List(CodeDicomField(IlluminationCode)),
        data_key="IlluminationTypeCodeSequence",
        dump_default=[defaults.illumination_type],
    )
    illumination_wavelength = fields.Integer(
        data_key="IlluminationWaveLength", load_default=None
    )
    illumination_color_code = SingleCodeSequenceField(
        IlluminationColorCode,
        data_key="IlluminationColorCodeSequence",
        load_default=None,
    )

    icc_profile = fields.Raw(data_key="ICCProfile", load_default=None)
    lut = LutDicomField(data_key="PaletteColorLookupTableSequence", load_default=None)
    light_path_filter = FlattenOnDumpNestedDicomField(
        LightPathFilterDicomSchema(), load_default=None
    )
    image_path_filter = FlattenOnDumpNestedDicomField(
        ImagePathFilterDicomSchema(), load_default=None
    )
    objective = FlattenOnDumpNestedDicomField(ObjectivesSchema(), load_default=None)

    @property
    def load_type(self) -> Type[OpticalPath]:
        return OpticalPath

    @pre_dump
    def pre_dump(self, optical_path: OpticalPath, **kwargs):
        fields = {
            "identifier": optical_path.identifier,
            "illumination_types": optical_path.illumination_types,
            "light_path_filter": optical_path.light_path_filter,
            "image_path_filter": optical_path.image_path_filter,
            "objective": optical_path.objective,
            "lut": optical_path.lut,
        }
        if optical_path.description is not None:
            fields["description"] = optical_path.description
        if optical_path.icc_profile is not None:
            fields["icc_profile"] = optical_path.icc_profile
        if isinstance(optical_path.illumination, float):
            fields["illumination_wavelength"] = optical_path.illumination
        if isinstance(optical_path.illumination, Code):
            fields["illumination_color_code"] = optical_path.illumination
        else:
            fields["illumination_color_code"] = defaults.illumination
        return fields

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs):
        illumination_wavelength = data.pop("illumination_wavelength", None)
        illumination_color_code = data.pop("illumination_color_code", None)
        if illumination_wavelength is not None:
            data["illumination"] = illumination_wavelength
        elif illumination_color_code is not None:
            data["illumination"] = illumination_color_code
        return super().post_load(data, **kwargs)

    @property
    def module_name(self) -> str:
        return "optical_path"
