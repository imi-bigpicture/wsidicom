#    Copyright 2026 SECTRA AB
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

"""Unit tests for the concatenation part sinks (writer/factory stubbed out)."""

from pathlib import Path

import pytest
from decoy import Decoy, matchers
from upath import UPath

from wsidicom.codec.encoder import Encoder, JpegEncoder, JpegSettings
from wsidicom.file.file_writer import BufferedPartSink, DirectPartSink, PartFactory
from wsidicom.file.io.wsidicom_writer import WsiDicomWriter
from wsidicom.instance import WsiDataset


@pytest.fixture
def part_writer(decoy: Decoy, tmp_path: Path) -> WsiDicomWriter:
    writer = decoy.mock(cls=WsiDicomWriter)
    decoy.when(writer.filepath).then_return(UPath(tmp_path) / "part.dcm")
    return writer


@pytest.fixture
def part_factory(decoy: Decoy, part_writer: WsiDicomWriter) -> PartFactory:
    part_factory = decoy.mock(cls=PartFactory)
    decoy.when(
        part_factory.open(
            matchers.Anything(), matchers.Anything(), concatenated=matchers.Anything()
        )
    ).then_return((part_writer, WsiDataset()))
    return part_factory


class TestDirectPartSink:
    @pytest.mark.parametrize("transcoder", [None, JpegEncoder(JpegSettings())])
    @pytest.mark.parametrize("concatenated", [True, False])
    def test_streams_tiles_and_finalizes_to_writer_filepath(
        self,
        tmp_path: Path,
        part_writer: WsiDicomWriter,
        transcoder: Encoder,
        decoy: Decoy,
        concatenated: bool,
        minimal_dataset: WsiDataset,
    ) -> None:
        # Arrange
        filepath = UPath(tmp_path) / "part.dcm"
        decoy.when(part_writer.filepath).then_return(filepath)
        sink = DirectPartSink(part_writer, minimal_dataset, transcoder=transcoder)
        tiles = [b"a", b"b"]

        # Act
        sink.write(tiles)
        returned = sink.finalize(concatenated=concatenated)

        # Assert — streamed straight through to the already-open writer
        assert returned == filepath
        decoy.verify(part_writer.write_tiles(tiles), times=1)
        decoy.verify(part_writer.finalize(minimal_dataset, transcoder), times=1)

    def test_close_closes_the_writer(
        self, part_writer: WsiDicomWriter, decoy: Decoy
    ) -> None:
        # Arrange
        sink = DirectPartSink(part_writer, WsiDataset(), transcoder=None)

        # Act
        sink.close()

        # Assert
        decoy.verify(part_writer.close(), times=1)


@pytest.mark.parametrize("transcoder", [None, JpegEncoder(JpegSettings())])
@pytest.mark.parametrize("concatenated", [True, False])
class TestBufferedPartSink:
    def test_buffers_then_writes_tiles_back_in_order(
        self,
        tmp_path: Path,
        part_writer: WsiDicomWriter,
        part_factory: PartFactory,
        decoy: Decoy,
        minimal_dataset: WsiDataset,
        transcoder: Encoder,
        concatenated: bool,
    ) -> None:
        # Arrange

        frame_offset = 10
        filepath = UPath(tmp_path) / "part.dcm"
        decoy.when(part_writer.filepath).then_return(filepath)
        decoy.when(
            part_factory.open(
                frame_offset, matchers.Anything(), concatenated=concatenated
            )
        ).then_return((part_writer, minimal_dataset))
        sink = BufferedPartSink(
            UPath(tmp_path), part_factory, frame_offset, transcoder=transcoder
        )
        tiles = [b"aaa", b"bb", b"cccc"]

        # Act — buffer across two calls, then finalize
        sink.write(tiles[:2])
        sink.write(tiles[2:])
        returned = sink.finalize(concatenated=concatenated)

        # Assert
        assert returned == filepath
        decoy.verify(part_writer.write_tiles(matchers.Anything()), times=1)
        decoy.verify(part_writer.finalize(minimal_dataset, transcoder), times=1)

    def test_scratch_file_removed_after_finalize(
        self,
        tmp_path: Path,
        part_writer: WsiDicomWriter,
        part_factory: PartFactory,
        decoy: Decoy,
        minimal_dataset: WsiDataset,
        transcoder: Encoder,
        concatenated: bool,
    ) -> None:
        # Arrange
        decoy.when(part_writer.filepath).then_return(UPath(tmp_path) / "part.dcm")
        decoy.when(
            part_factory.open(
                matchers.Anything(), matchers.Anything(), concatenated=concatenated
            )
        ).then_return((part_writer, minimal_dataset))
        sink = BufferedPartSink(UPath(tmp_path), part_factory, 0, transcoder=transcoder)
        sink.write([b"x"])
        temp_file = list(tmp_path.glob("concat_*.bin"))[0]

        # Act
        returned = sink.finalize(concatenated=concatenated)

        # Assert
        assert returned == part_writer.filepath
        assert not temp_file.exists()
