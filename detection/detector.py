"""Vehicle detection with YOLO and plate text extraction with EasyOCR."""

import logging
from pathlib import Path

import cv2
import easyocr
from ultralytics import YOLO

from .plate_utils import is_valid_colombian_plate, normalize_plate


logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

MODEL_PATH = Path(__file__).resolve().parent.parent / "yolov8n.pt"
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def _clean_ocr_text(text: str) -> str:
    return "".join(ch for ch in text.upper() if ch.isalnum())


def _prefix_variants(text: str) -> list[str]:
    cleaned = _clean_ocr_text(text)
    variants = [cleaned]
    if cleaned.startswith("I") and len(cleaned) == 3:
        variants.append("W" + cleaned[1:])
    if cleaned.startswith("VV") and len(cleaned) >= 3:
        variants.append("W" + cleaned[2:])
    return list(dict.fromkeys(variants))


def _moto_suffix_variants(text: str) -> list[str]:
    digits = "".join(ch for ch in _clean_ocr_text(text) if ch.isdigit())
    if len(digits) < 2:
        return []
    base = digits[:2]
    return [base + letter for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]


class PlateDetector:
    """Detects vehicles and extracts the most plate-like OCR reading."""

    def __init__(self, model_path: str | None = None):
        path = model_path or str(MODEL_PATH)
        logger.info("Cargando modelo YOLO desde %s", path)
        self.model = YOLO(path)
        logger.info("Inicializando EasyOCR")
        self.ocr = easyocr.Reader(["en"], gpu=True)
        logger.info("Detector listo")

    def _ocr_candidates(self, image) -> list[dict]:
        variants = [image]
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            variants.extend(
                [
                    gray,
                    cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                    cv2.adaptiveThreshold(
                        cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        31,
                        5,
                    ),
                ]
            )
        except cv2.error:
            pass

        all_readings = []
        candidates = []
        for variant in variants:
            all_readings.extend(self._read_ocr(variant))

        for reading in all_readings:
            raw_text = reading["raw_text"]
            confidence = reading["confidence"]
            text = normalize_plate(raw_text)
            if len(text) < 5:
                continue
            candidates.append(
                {
                    "plate_text": text,
                    "ocr_confidence": confidence,
                    "matches_pattern": is_valid_colombian_plate(text),
                    "source": raw_text,
                }
            )

        candidates.extend(self._combined_candidates(all_readings))
        return self._dedupe_candidates(candidates)

    def _read_ocr(self, image) -> list[dict]:
        try:
            readings = self.ocr.readtext(
                image,
                detail=1,
                paragraph=False,
                allowlist=OCR_ALLOWLIST,
            )
        except TypeError:
            readings = self.ocr.readtext(image)

        output = []
        for reading in readings:
            if len(reading) < 2:
                continue
            output.append(
                {
                    "raw_text": _clean_ocr_text(reading[1]),
                    "confidence": float(reading[2]) if len(reading) > 2 else 1.0,
                }
            )
        return output

    @staticmethod
    def _combined_candidates(readings: list[dict]) -> list[dict]:
        candidates = []
        prefixes = [
            reading
            for reading in readings
            if _clean_ocr_text(reading["raw_text"]).isalpha()
            and 2 <= len(_clean_ocr_text(reading["raw_text"])) <= 3
        ]
        numeric_parts = [
            reading
            for reading in readings
            if any(ch.isdigit() for ch in _clean_ocr_text(reading["raw_text"]))
        ]

        for prefix_reading in prefixes:
            for number_reading in numeric_parts:
                confidence = min(prefix_reading["confidence"], number_reading["confidence"])
                for prefix in _prefix_variants(prefix_reading["raw_text"]):
                    if len(prefix) != 3:
                        continue
                    for suffix in _moto_suffix_variants(number_reading["raw_text"]):
                        plate_text = normalize_plate(prefix + suffix)
                        candidates.append(
                            {
                                "plate_text": plate_text,
                                "ocr_confidence": confidence,
                                "matches_pattern": is_valid_colombian_plate(plate_text),
                                "source": f'{prefix_reading["raw_text"]}+{number_reading["raw_text"]}',
                            }
                        )
        return candidates

    @staticmethod
    def _dedupe_candidates(candidates: list[dict]) -> list[dict]:
        by_plate = {}
        for candidate in candidates:
            plate = candidate["plate_text"]
            if plate not in by_plate or candidate["ocr_confidence"] > by_plate[plate]["ocr_confidence"]:
                by_plate[plate] = candidate
        return sorted(
            by_plate.values(),
            key=lambda item: (
                int(item["matches_pattern"]),
                item["ocr_confidence"],
                -abs(len(item["plate_text"]) - 6),
            ),
            reverse=True,
        )

    @staticmethod
    def _best_candidate(candidates: list[dict]) -> dict | None:
        if not candidates:
            return None

        def score(candidate: dict) -> tuple:
            return (
                int(candidate["matches_pattern"]),
                candidate["ocr_confidence"],
                -abs(len(candidate["plate_text"]) - 6),
            )

        best = max(candidates, key=score)
        if best["matches_pattern"] or best["ocr_confidence"] >= 0.45:
            return best
        return None

    def detect(self, source) -> list[dict]:
        """Runs detection on an image path or frame and returns plate readings."""
        frame = cv2.imread(str(source)) if isinstance(source, (str, Path)) else source
        if frame is None:
            return []

        detections = []
        results = self.model(frame, verbose=False)
        vehicle_boxes = []

        for raw_box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = raw_box
            class_id = int(class_id)
            if class_id not in VEHICLE_CLASSES:
                continue
            vehicle_boxes.append(
                {
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "vehicle_type": VEHICLE_CLASSES[class_id],
                    "vehicle_confidence": float(score),
                }
            )

        if not vehicle_boxes:
            height, width = frame.shape[:2]
            vehicle_boxes.append(
                {
                    "bbox": (0, 0, width, height),
                    "vehicle_type": "unknown",
                    "vehicle_confidence": 0.0,
                }
            )

        for vehicle in vehicle_boxes:
            x1, y1, x2, y2 = vehicle["bbox"]
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            candidates = self._ocr_candidates(roi)
            candidates = self._dedupe_candidates(candidates)
            best = self._best_candidate(candidates)
            if best is None:
                continue

            bbox_ratio = (x2 - x1) / max((y2 - y1), 1)
            detections.append(
                {
                    "vehicle_type": vehicle["vehicle_type"],
                    "vehicle_confidence": vehicle["vehicle_confidence"],
                    "plate_text": best["plate_text"],
                    "ocr_confidence": best["ocr_confidence"],
                    "bbox_ratio": bbox_ratio,
                    "bbox": vehicle["bbox"],
                    "ocr_candidates": candidates[:80],
                }
            )

        return detections

    def annotate_frame(self, frame, detections: list[dict]):
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f'{det["vehicle_type"]}: {det["plate_text"]} ({det["ocr_confidence"]:.0%})'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        return annotated
