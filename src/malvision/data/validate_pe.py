"""
PE file validation and metadata extraction.
Static validation only - never executes or loads the binary.
"""

import pefile
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import hashlib

from ..logger import logger


class PEValidator:
    """
    Validates PE file structure and extracts metadata for analysis.
    """

    def __init__(self, parser_version: str = "pefile_v2023.2.7"):
        self.parser_version = parser_version
        # Track parser version for reproducibility

    def validate(self, file_path: Path) -> Tuple[bool, Dict, Optional[str]]:
        """
        Validate PE file and extract metadata.
        Returns (is_valid, metadata, error_message)
        """
        if not file_path.is_file():
            return False, {}, f"File not found: {file_path}"

        if file_path.stat().st_size == 0:
            return False, {}, f"Empty file: {file_path}"

        try:
            pe = pefile.PE(file_path)
            # Basic PE checks
            if not pe.FILE_HEADER:
                return False, {}, "Invalid PE header"

            # Check for suspicious characteristics
            suspicious_indicators = []
            # Check for null entries in import table (common in packed binaries)
            if pe.imports:
                null_imports = sum(1 for imp in pe.imports if imp.name is None)
                if null_imports > 10:
                    suspicious_indicators.append(f"Many null imports ({null_imports})")

            # Check for high entropy sections (possible packing)
            for section in pe.get_sections():
                if section.get_entropy() > 7.0:
                    suspicious_indicators.append(f"High-entropy section: {section.Name.decode().strip()}")

            metadata = {
                "machine": pe.FILE_HEADER.Machine,
                "subsystem": pe.OPTIONAL_HEADER.Subsystem,
                "section_count": pe.FILE_HEADER.NumberOfSections,
                "import_count": len(pe.imports) if pe.imports else 0,
                "export_count": len(pe.exports) if pe.exports else 0,
                "entry_point": pe.OPTIONAL_HEADER.AddressOfEntryPoint,
                "image_size": pe.OPTIONAL_HEADER.SizeOfImage,
                "header_size": pe.OPTIONAL_HEADER.SizeOfHeaders,
                "timestamp": pe.FILE_HEADER.TimeDateStamp if pe.FILE_HEADER.TimeDateStamp else None,
                "checksum": pe.FILE_HEADER.CheckSum,
                "suspicious_indicators": suspicious_indicators,
                "parser_version": self.parser_version,
            }

            # Basic structural validation
            if pe.FILE_HEADER.Machine not in [0x014C, 0x8664]:  # 32-bit or 64-bit
                return False, metadata, f"Unsupported machine type: {pe.FILE_HEADER.Machine}"

            if pe.FILE_HEADER.NumberOfSections < 1:
                return False, metadata, "No sections found"

            if pe.OPTIONAL_HEADER.SizeOfImage < 1024:
                return False, metadata, "Image size too small"

            return True, metadata, None

        except pefile.PEFormatError as e:
            return False, {}, f"PEFormatError: {e}"
        except Exception as e:
            return False, {}, f"Validation error: {e}"

    def compute_sha256(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file bytes."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


if __name__ == "__main__":
    # Quick test
    validator = PEValidator()
    test_file = Path("test.exe")
    if test_file.exists():
        valid, meta, err = validator.validate(test_file)
        print(f"Valid: {valid}")
        print(f"Metadata: {meta}")
        print(f"Error: {err}")
        print(f"SHA256: {validator.compute_sha256(test_file)}")