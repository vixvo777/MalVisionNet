"""
Image resizing utilities for byteplots.
Provides area and bilinear interpolation methods with metadata tracking.
"""

import numpy as np
from typing import Tuple, Dict, Any

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .. import logger


def resize_byteplot(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (224, 224),
    method: str = "area"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resize a byteplot array to target shape (height, width).

    Args:
        image: Source array of shape (H, W) or (H, W, C)
        target_shape: Desired (height, width)
        method: Interpolation method ("area" or "bilinear")

    Returns:
        Resized array, resize_metadata
    """
    orig_shape = image.shape
    target_h, target_w = target_shape

    if orig_shape[:2] == (target_h, target_w):
        return image, {
            "original_shape": orig_shape,
            "target_shape": (target_h, target_w),
            "resized": False,
            "method": "none"
        }

    if HAS_CV2:
        interpolation = cv2.INTER_AREA if method == "area" else cv2.INTER_LINEAR
        resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    else:
        logger.warning("OpenCV not installed; using fallback numpy resizing.")
        if len(orig_shape) == 2:
            resized = np.resize(image, (target_h, target_w))
        else:
            resized = np.resize(image, (target_h, target_w, orig_shape[2]))

    metadata = {
        "original_shape": orig_shape,
        "target_shape": (target_h, target_w),
        "resized": True,
        "method": method,
        "backend": "cv2" if HAS_CV2 else "numpy"
    }

    return resized, metadata
