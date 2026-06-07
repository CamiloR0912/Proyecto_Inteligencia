"""FastAPI backend for the Colombian motorcycle plate prototype."""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np




from detection.detector import PlateDetector
from detection.plate_utils import is_valid_colombian_plate, normalize_plate
from ml.classifier import PlateValidator


app = FastAPI(
    title="API Placas Colombia",
    description="Clasificacion e identificacion de placas de motos colombianas",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).parent
RESULTS_DIR = ROOT_DIR / "data" / "results"
RESULTS_FILE = RESULTS_DIR / "detections.json"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

detector = PlateDetector()
validator = PlateValidator()



def load_results() -> list:
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError:
            return []
    return []


def save_result(result: dict):
    results = load_results()
    results.append(result)
    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def build_features(plate_text: str, detection: dict) -> dict:
    return {
        "text_length": len(plate_text),
        "num_letters": sum(c.isalpha() for c in plate_text),
        "num_digits": sum(c.isdigit() for c in plate_text),
        "ocr_confidence": detection.get("ocr_confidence", 0.0),
        "bbox_ratio": detection.get("bbox_ratio", 1.0),
        "matches_pattern": int(is_valid_colombian_plate(plate_text)),
    }


def process_detections(file_id: str, filename: str, detections: list[dict]) -> list[dict]:
    processed = []
    for detection in detections:
        plate_text = validator.correct_plate(
            detection.get("plate_text", ""),
            detection.get("ocr_candidates", []),
        )
        features = build_features(plate_text, detection)
        rf_valid, rf_confidence = validator.predict(features)

        record = {
            "id": file_id,
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "vehicle_type": detection.get("vehicle_type", "unknown"),
            "vehicle_confidence": round(detection.get("vehicle_confidence", 0.0), 3),
            "plate_text": plate_text,
            "ocr_confidence": round(detection.get("ocr_confidence", 0.0), 3),
            "bbox_ratio": round(detection.get("bbox_ratio", 1.0), 3),
            "rf_valid": bool(rf_valid),
            "rf_confidence": round(float(rf_confidence), 3),
            "matches_pattern": bool(features["matches_pattern"]),
            "dataset_match": validator.lookup_plate(plate_text),
            "ocr_candidates": detection.get("ocr_candidates", []),
        }
        processed.append(record)
        save_result(record)
    return processed


@app.get("/")
def root():
    return {"status": "ok", "message": "API Placas Colombia activa"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imagenes")

    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{file.filename}"

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Imagen no valida o corrupta")

    try:
        detections = detector.detect(frame)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en deteccion: {exc}") from exc

    processed = process_detections(file_id, filename, detections)

    return {
        "file_id": file_id,
        "filename": filename,
        "detections_count": len(processed),
        "detections": processed,
    }


@app.post("/upload-batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    all_results = []
    for file in files[:20]:
        all_results.append(await upload_image(file))
    return {"batch_size": len(all_results), "results": all_results}


@app.get("/results")
def get_results(limit: int = 100):
    results = load_results()
    return {"total": len(results), "results": results[-limit:]}


@app.get("/dataset/summary")
def get_dataset_summary():
    return validator.get_dataset_summary()





@app.get("/stats")
def get_stats():
    results = load_results()
    if not results:
        return {
            "message": "No hay datos aun. Sube imagenes primero.",
            "dataset_summary": validator.get_dataset_summary(),
        }

    vehicle_counts = {}
    ocr_by_vehicle = {}
    for result in results:
        vehicle_type = result.get("vehicle_type", "unknown")
        vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
        ocr_by_vehicle.setdefault(vehicle_type, []).append(
            result.get("ocr_confidence", 0.0)
        )

    valid_count = sum(1 for result in results if result.get("rf_valid"))
    invalid_count = len(results) - valid_count

    return {
        "total_detections": len(results),
        "vehicle_distribution": vehicle_counts,
        "rf_metrics": validator.get_metrics(),
        "ocr_by_vehicle": ocr_by_vehicle,
        "plate_validity": {
            "valid": valid_count,
            "invalid": invalid_count,
        },
        "dataset_summary": validator.get_dataset_summary(),
        "avg_vehicle_confidence": round(
            sum(result.get("vehicle_confidence", 0) for result in results)
            / len(results),
            3,
        ),
    }


@app.delete("/results/clear")
def clear_results():
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
    return {"message": "Resultados limpiados"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
