"""
Unit tests for byteplot converters (grayscale and RGB).
Tests conversion invariants, edge cases, and config consistency.
"""

import pytest
import numpy as np
from pathlib import Path

from malvision.representations import GrayscaleConverter, RGBTrigramConverter


@pytest.fixture
def test_converter():
    """Fixture to create converter with test config"""
    # Create temporary config file
    config_content = """grayscale:
  width_pixels: 4
  resize_target: [2, 2]
  resize_method: "area"
rgb_trigram:
  width_pixels: 4
  resize_target: [2, 2]
  resize_method: "area"
"""""
    config_path = Path("test_config.yaml")
    config_path.write_text(config_content)
    yield config_path
    config_path.unlink()


def test_grayscale_converter_basic(test_converter):
    """Test grayscale converter with known byte sequence"""
    converter = GrayscaleConverter(str(test_converter))
    # Test bytes: 0-15 repeated
    test_bytes = bytes(range(16))
    img, meta = converter.convert(test_bytes)

    assert img.dtype == np.float32
    assert img.shape == (2, 4)
    assert meta["original_length"] == 16
    assert meta["width_pixels"] == 4
    assert meta["height_pixels"] == 2
    assert img[0, 0] == 0.0  # byte 0
    assert img[0, 1] == 1.0 / 255.0  # byte 1
    assert img[0, 3] == 3.0 / 255.0  # byte 3
    assert img[1, 0] == 12.0 / 255.0  # byte 12


def test_rgb_trigram_converter_basic(test_converter):
    """Test RGB-trigram converter with known byte sequence"""
    converter = RGBTrigramConverter(str(test_converter))
    # Test bytes: 0-7 (forms 2 pixels: (0,1,2) and (3,4,5), last byte 6-7 padded)
    test_bytes = bytes(range(8))
    img, meta = converter.convert(test_bytes)

    assert img.dtype == np.float32
    assert img.shape == (2, 4, 3)
    assert meta["original_length"] == 8
    assert meta["total_pixels"] == 2
    assert img[0, 0, 0] == 0.0  # R of first pixel
    assert img[0, 0, 1] == 1.0 / 255.0  # G of first pixel
    assert img[0, 0, 2] == 2.0 / 255.0  # B of first pixel
    assert img[0, 1, 0] == 3.0 / 255.0  # R of second pixel
    assert img[0, 1, 2] == 5.0 / 255.0  # B of second pixel


def test_converter_padding(test_converter):
    """Test padding behavior for incomplete rows"""
    converter = GrayscaleConverter(str(test_converter))
    # Test bytes: 5 bytes -> 1 full row (4), 1 byte in second row
    test_bytes = bytes(range(5))
    img, meta = converter.convert(test_bytes)

    assert img.shape == (2, 4)
    assert meta["padding_bytes"] == 3
    # Last row should be [4/255, 0, 0, 0]
    assert img[1, 0] == 4.0 / 255.0
    assert img[1, 1] == 0.0
    assert img[1, 2] == 0.0
    assert img[1, 3] == 0.0


def test_converter_resize(test_converter):
    """Test resize behavior when input shape differs from target"""
    converter = GrayscaleConverter(str(test_converter))
    # Create larger image (8x8) to test resize
    test_bytes = bytes(range(64))
    img, meta = converter.convert(test_bytes)

    # Should resize from (8,4) to (2,2)
    assert img.shape == (2, 2)
    assert meta["resize_method"] == "area"
    assert meta["resize_target"] == (2, 2)


def test_converter_file_conversion(test_converter):
    # Fix the padding bug in grayscale converter
    test_file = test_converter.parent / "test_padding.bin"
    test_file.write_bytes(bytes(range(16)))  # 16 bytes
    # Test the fixed converter
    converter = GrayscaleConverter(str(test_converter))

    # First, check raw conversion to see if padding works
    test_bytes = bytes(range(16))
    img, meta = converter.convert(test_bytes)

    assert img.shape == (2, 4)
    assert meta["padding_bytes"] == 0

    test_file.unlink()
    """Test file-based conversion with hash calculation"""
    converter = GrayscaleConverter(str(test_converter))
    # Create temporary test file
    test_file = Path("test_file.bin")
    test_file.write_bytes(bytes(range(32)))

    img, meta, sha256 = converter.convert_from_file(test_file)

    assert meta["sha256"] == sha256
    assert meta["file_path"] == str(test_file)
    assert meta["file_size"] == 32
    assert img.shape == (2, 4)

    # Cleanup
    test_file.unlink()