import pytest
from PIL import Image

from tests.codec import get_encoded_test_file
from wsidicom.codec import Settings as EncoderSettings


@pytest.fixture
def image():
    yield Image.open("tests/testdata/test_tile.png")


@pytest.fixture
def encoded(encoder_settings: EncoderSettings):
    yield get_encoded_test_file(encoder_settings)
