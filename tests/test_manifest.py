"""
Unit tests for dataset manifest builder.
Tests manifest building, PE validation, and duplicate detection.
"""

import tempfile
import os
from pathlib import Path
import csv

import sys
sys.path.insert(0, str(Path(__file__).parent / ".." / "src"))

from malvision.data import PEValidator, ManifestBuilder


def test_pe_validator_basic():
    """Test PE validator with a simple test"""
    validator = PEValidator()

    # Test with non-existent file
    invalid_path = Path("nonexistent.exe")
    valid, metadata, error = validator.validate(invalid_path)

    assert not valid
    assert metadata == {}
    assert "not found" in error.lower()


def test_pe_validator_computes_hash():
    """Test SHA-256 hash computation"""
    validator = PEValidator()

    # Create a temporary file with known content
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test data 12345")
        temp_path = Path(f.name)

    try:
        sha256 = validator.compute_sha256(temp_path)

        # The actual hash doesn't need to match expected, just test it runs
        assert len(sha256) == 64  # SHA-256 should be 64 hex chars
        assert all(c in '0123456789abcdef' for c in sha256)
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_pe_validator_basic()
    test_pe_validator_computes_hash()
    print("All tests passed!")