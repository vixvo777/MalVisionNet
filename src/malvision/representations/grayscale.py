"""
Convert raw PE bytes to grayscale byteplot.

The converter follows the specification from PLAN.md Section 5.1:
- Each byte interpreted as 0-255 grayscale intensity
- Reshaped into rows of fixed pixel width (configurable)
- Padding with zeros for incomplete final row
- Returns (numpy array, metadata)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

import yaml
from ..logger import logger
import hashlib

from .. import logger


class GrayscaleConverter:
    """
    Convert raw PE bytes to grayscale byteplot image.
    """

    def __init__(self, config_path: str = "configs/representation.yaml"):
        self.config = self._load_config(config_path)
        self.width_pixels = self.config["grayscale"]["width_pixels"]
        self.resize_target = tuple(self.config["grayscale"]["resize_target"])
        self.resize_method = self.config["grayscale"]["resize_method"]

    def _load_config(self, config_path: str) -> Dict:
        """Load representation configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _pad_row(self, row: np.ndarray, target_width: int) -> np.ndarray:
        """Pad the last row with zeros to reach target width."""
        current_len = len(row)
        if current_len < target_width:
            padding = np.zeros(target_width - current_len, dtype=np.uint8)
            return np.concatenate([row, padding])
        return row

    def convert(self, raw_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convert raw bytes to grayscale byteplot.
        Returns (image_array, metadata)
        """
        # Convert bytes to uint8 array
        try:
            data = np.frombuffer(raw_bytes, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Failed to convert bytes: {e}")
            raise

        # Calculate number of pixels (bytes = pixels for grayscale)
        num_pixels = len(data)
        width = self.width_pixels
        height = int(np.ceil(num_pixels / width))

        # Reshape into rows
        if num_pixels > 0:
            image = data.reshape(height, width)
            # Pad last row if needed
            if image.shape[0] == height and image.shape[1] == width:
                pass  # Already perfect shape
            else:
                # Pad the last row
                image = self._pad_row(image[-1:], width)
                # Keep all previous rows
                image = np.vstack([image[:-1], image[-1:]])
        else:
            # Empty file case
            image = np.zeros((1, width), dtype=np.uint8)

        # Resize to target dimensions if different
        target_h, target_w = self.resize_target
        if image.shape != (target_h, target_w):
            # Use OpenCV for resizing if available, otherwise numpy
            try:
                import cv2
                interpolation = cv2.INTER_AREA if self.resize_method == "area" else cv2.INTER_LINEAR
                image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
                logger.debug(f"Resized using cv2: {image.shape}")
            except Exception as e:
                logger.warning(f"cv2 not available, using numpy resize: {e}")
                # Fallback to numpy resize
                image = np.resize(image, (target_h, target_w))

        metadata = {
            "original_length": num_pixels,
            "width_pixels": width,
            "height_pixels": height,
            "output_shape": image.shape,
            "dtype": str(image.dtype),
            "padding_bytes": max(0, width * height - num_pixels),
            "resize_method": self.resize_method,
            "resize_target": self.resize_target,
            "config_id": "rep_v1_grayscale_256_224area",
            "conversion_time": None,  # Will be set by caller if needed
        }

        return image.astype(np.float32) / 255.0, metadata

    def convert_from_file(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any], str]:
        """Convert file to byteplot with SHA-256 hash."""
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        raw_bytes = file_path.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        image, metadata = self.convert(raw_bytes)
        metadata["sha256"] = sha256
        metadata["file_path"] = str(file_path)
        metadata["file_size"] = file_path.stat().st_size
        return image, metadata, sha256


if __name__ == "__main__":
    # Quick test
    converter = GrayscaleConverter()
    # Test with known byte sequence
    test_bytes = bytes(range(256)) * 2  # Simple test pattern
    img, meta = converter.convert(test_bytes)
    print(f"Grayscale image shape: {img.shape}")
    print(f"Metadata: {meta}")
    print(f"First pixel: {img[0, 0]:.4f}")
    print(f"Last pixel: {img[-1, -1]:.4f}")