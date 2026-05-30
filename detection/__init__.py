"""Módulo de detección de vehículos y reconocimiento de placas."""
from .detector import PlateDetector, Detection
from .tracker import PlateTracker
from .plate_utils import normalize_plate, PLATE_REGEX, VEHICLE_CLASSES

__all__ = [
    "PlateDetector",
    "Detection",
    "PlateTracker",
    "normalize_plate",
    "PLATE_REGEX",
    "VEHICLE_CLASSES",
]
