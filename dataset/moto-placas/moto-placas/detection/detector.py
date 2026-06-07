"""
detection/detector.py
PlateDetector — Detecta vehículos con YOLOv8 y lee placas con EasyOCR.

NOTA: Este archivo es el stub/interfaz para el proyecto final.
Reemplazar el contenido del método detect() con tu implementación
existente de Proyecto_Inteligencia/detection/detector.py
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict

# ── Importaciones opcionales (pueden no estar instaladas en dev) ──────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics no instalado. Ejecuta: pip install ultralytics")

try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️  easyocr no instalado. Ejecuta: pip install easyocr")

# Clases YOLO COCO relevantes para el proyecto
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Ruta al modelo YOLO (ajustar según tu proyecto)
MODEL_PATH = Path(__file__).parent.parent / "yolov8n.pt"


class PlateDetector:
    """
    Pipeline completo: YOLO detecta vehículos → recorta ROI de placa → EasyOCR lee texto.
    Compatible con la clase existente en Proyecto_Inteligencia.
    """

    def __init__(self):
        # Cargar YOLO
        if YOLO_AVAILABLE and MODEL_PATH.exists():
            self.yolo = YOLO(str(MODEL_PATH))
            print(f"✅ YOLO cargado desde {MODEL_PATH}")
        else:
            self.yolo = None
            print("⚠️  Corriendo en modo simulación (sin YOLO)")

        # Cargar EasyOCR (español + inglés)
        if OCR_AVAILABLE:
            self.reader = easyocr.Reader(["es", "en"], gpu=False)
            print("✅ EasyOCR cargado")
        else:
            self.reader = None

    def detect(self, image_path: str) -> List[Dict]:
        """
        Procesa una imagen y retorna lista de detecciones.

        Returns:
            Lista de dicts con:
            - vehicle_type: str
            - vehicle_confidence: float
            - plate_text: str
            - ocr_confidence: float
            - bbox_ratio: float (ancho/alto del ROI)
        """
        if self.yolo is None or self.reader is None:
            # Modo demo/simulación: retorna resultado ficticio para testing
            return self._simulate_detection(image_path)

        img = cv2.imread(image_path)
        if img is None:
            return []

        results = self.yolo(img, verbose=False)[0]
        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASSES:
                continue

            vehicle_conf = float(box.conf[0])
            vehicle_type = VEHICLE_CLASSES[class_id]

            # Recortar el vehículo
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = img[y1:y2, x1:x2]

            # Intentar leer placa en el ROI
            plate_text, ocr_conf = self._read_plate(roi)
            bbox_ratio = (x2 - x1) / max((y2 - y1), 1)

            detections.append({
                "vehicle_type": vehicle_type,
                "vehicle_confidence": vehicle_conf,
                "plate_text": plate_text,
                "ocr_confidence": ocr_conf,
                "bbox_ratio": round(bbox_ratio, 2),
            })

        return detections

    def _read_plate(self, roi: np.ndarray) -> tuple:
        """Lee el texto de la placa en un ROI dado."""
        if self.reader is None or roi.size == 0:
            return "", 0.0

        # Preprocesamiento básico
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        ocr_results = self.reader.readtext(binary)
        if not ocr_results:
            return "", 0.0

        # Tomar la detección de mayor confianza
        best = max(ocr_results, key=lambda x: x[2])
        text = best[1].upper().strip()
        confidence = float(best[2])
        return text, confidence

    def _simulate_detection(self, image_path: str) -> List[Dict]:
        """
        Modo simulación: genera detecciones ficticias para testing
        sin necesidad de tener YOLO/EasyOCR instalados.
        Útil para desarrollar y probar el frontend/API.
        """
        import random
        import hashlib

        # Usar hash del nombre de archivo para resultados consistentes
        seed = int(hashlib.md5(image_path.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        vehicle_types = ["motorcycle", "motorcycle", "car", "car", "motorcycle"]
        sample_plates = ["ABC12F", "XYZ34A", "MNP567", "QRS89", "TUV12C", "ABC123"]

        n = rng.randint(1, 2)
        detections = []
        for _ in range(n):
            vtype = rng.choice(vehicle_types)
            plate = rng.choice(sample_plates)
            detections.append({
                "vehicle_type": vtype,
                "vehicle_confidence": round(rng.uniform(0.72, 0.98), 3),
                "plate_text": plate,
                "ocr_confidence": round(rng.uniform(0.55, 0.97), 3),
                "bbox_ratio": round(rng.uniform(2.2, 4.0), 2),
            })
        return detections
