"""Módulo de captura y streaming de cámara."""
from .manager import CameraManager
from .stream import generate_mjpeg

__all__ = ["CameraManager", "generate_mjpeg"]
