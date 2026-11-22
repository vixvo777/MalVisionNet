"""
Convert raw PE bytes to RGB-trigram byteplot.

The converter follows the specification from PLAN.md Section 5.2:
- Three consecutive bytes map to R, G, B channels
- reshaped into rows of fixed pixel width (configurable)
- Pad incomplete final row with zero triplets
- Returns (numpy array, metadata)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
import hashlib
import yaml

from .. import logger


class RGBTrigramConverter:
    """
    Convert raw PE bytes to RGB-trigram byteplot image.
    """

    def __init__(self, config_path: str = "configs/representation.yaml"):
        self.config = self._load_config(config_path)
        self.width_pixels = self.config["rgb_trigram"]["width_pixels"]
        self.resize_target = tuple(self.config["rgb_trigram"]["resize_target"])
        self.resize_method = self.config["rgb_trigram"]["resize_method"]
        self.channel_order = self.config["rgb_trigram"].get("channel_order", "RGB")

    def _load_config(self, config_path: str) -> Dict:
        """Load representation configuration."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def convert(self, raw_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convert raw bytes to RGB-trigram byteplot.
        Returns (image_array, metadata)
        """
        # Convert bytes to uint8 array
        data = np.frombuffer(raw_bytes, dtype=np.uint8)
        num_bytes = len(data)

        # Pad with zeros to be divisible by 3 (R,G,B)
        remainder = num_bytes % 3
        if remainder > 0:
            padding = np.zeros(3 - remainder, dtype=np.uint8)
            data = np.concatenate([data, padding])

        # Group bytes into RGB triplets
        # Plan 5.2: pixels = bytes_array[:usable_length].reshape(-1, 3)
        num_pixels = len(data) // 3
        pixels = data.reshape(-1, 3)

        width = self.width_pixels
        height = int(np.ceil(num_pixels / width))

        # Pad with zero triplets to fill the last row
        total_pixels_needed = height * width
        if num_pixels < total_pixels_needed:
            padding_pixels = total_pixels_needed - num_pixels
            padding_triplets = np.zeros((padding_pixels, 3), dtype=np.uint8)
            pixels = np.vstack([pixels, padding_triplets])

        # Reshape to (height, width, 3)
        image = pixels.reshape(height, width, 3)

        # Apply channel order (e.g. BGR if needed for cv2 compatibility)
        if self.channel_order == "BGR":
            image = image[:, :, ::-1]  # Reverse channels

        # Resize to target dimensions
        target_h, target_w = self.resize_target
        if image.shape[:2] != (target_h, target_w):
            try:
                import cv2
                interpolation = cv2.INTER_AREA if self.resize_method == "area" else cv2.INTER_LINEAR
                image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
            except ImportError:
                logger.warning("cv2 not available, using simple numpy resize")
                # Fallback to simple slice/pad or resize
                image = np.resize(image, (target_h, target_w, 3))

        metadata = {
            "original_length": num_bytes,
            "total_pixels": num_pixels,
            "width_pixels": width,
            "height_pixels": height,
            "output_shape": image.shape,
            "dtype": str(image.dtype),
            "padding_bytes": 3 * total_pixels_needed - num_bytes,
            "resize_method": self.resize_method,
            "resize_target": self.resize_target,
            "channel_order": self.channel_order,
            "config_id": "rep_v1_rgb256_224area",
            "conversion_time": None,
        }

        # Normalize to [0, 1] as float32
        return image.astype(np.float32) / 255.0, metadata

    def convert_from_file(self, file_path: Path) -> Tuple[np.ndarray, Dict[str, Any], str]:
        """Convert file to RGB byteplot with SHA-256 hash."""
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
    converter = RGBTrigramConverter()
    # Test with known byte sequence (R=0, G=1, B=2...)
    test_bytes = bytes(range(255))
    img, meta = converter.convert(test_bytes)
    print(f"RGB image shape: {img.shape}")
    print(f"Metadata: {meta}")
    print(f"First pixel (R,G,B): {img[0, 0, 0]:.4f}, {img[0, 0, 1]:.4f}, {img[0, 0, 2]:.4f}")
    # Expected: 0.0000, 0.0039, 0.0078 (0/255, 1/255, 2/255)
