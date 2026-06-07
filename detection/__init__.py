"""Detection package.

Heavy vision dependencies are imported from detection.detector only when the
backend needs to run YOLO/EasyOCR.
"""

from .plate_utils import (
    PATTERNS,
    classify_plate_type,
    extract_features,
    is_valid_colombian_plate,
    normalize_plate,
)

__all__ = [
    "PATTERNS",
    "classify_plate_type",
    "extract_features",
    "is_valid_colombian_plate",
    "normalize_plate",
]
