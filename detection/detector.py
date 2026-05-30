"""Detector de vehículos y placas usando YOLO + EasyOCR."""
import cv2
import logging
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO
import easyocr

from .plate_utils import VEHICLE_CLASSES, normalize_plate

logger = logging.getLogger(__name__)

# Ruta al modelo YOLO (relativa a la raíz del proyecto)
_MODEL_PATH = Path(__file__).resolve().parent.parent / "yolov8n.pt"


@dataclass
class Detection:
    """Resultado de una detección individual."""
    vehicle_type: str
    plate_text: str
    confidence: float
    ocr_confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class PlateDetector:
    """Detecta vehículos con YOLO y lee placas con EasyOCR."""

    def __init__(self, model_path: str | None = None):
        path = model_path or str(_MODEL_PATH)
        logger.info(f"🔧 Cargando modelo YOLO desde {path}...")
        self.model = YOLO(path)
        logger.info("🔧 Inicializando EasyOCR...")
        self.ocr = easyocr.Reader(["en"], gpu=False)
        logger.info("✅ Detector listo.")

    def detect(self, frame) -> list[Detection]:
        """Ejecuta detección en un frame y retorna las placas encontradas."""
        detections = []
        results = self.model(frame, verbose=False)

        for r in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            class_id = int(class_id)

            if class_id not in VEHICLE_CLASSES:
                continue

            vehicle_type = VEHICLE_CLASSES[class_id]
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)

            # Recortar región del vehículo
            roi = frame[iy1:iy2, ix1:ix2]
            if roi.size == 0:
                continue

            # Aplicar OCR sobre la región
            ocr_result = self.ocr.readtext(roi)
            if not ocr_result:
                continue

            text = normalize_plate(ocr_result[0][1])
            ocr_conf = ocr_result[0][2] if len(ocr_result[0]) > 2 else 1.0

            if len(text) < 5 or ocr_conf < 0.5:
                continue

            detections.append(Detection(
                vehicle_type=vehicle_type,
                plate_text=text,
                confidence=float(score),
                ocr_confidence=float(ocr_conf),
                bbox=(ix1, iy1, ix2, iy2),
            ))

        return detections

    def annotate_frame(self, frame, detections: list[Detection]):
        """Dibuja bounding boxes y texto de placa sobre el frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 255, 0)

            # Bounding box del vehículo
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Etiqueta con tipo y placa
            label = f"{det.vehicle_type}: {det.plate_text} ({det.ocr_confidence:.0%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color, -1,
            )
            cv2.putText(
                annotated, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )

        return annotated
