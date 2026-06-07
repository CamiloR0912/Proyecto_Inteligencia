"""
app.py — API REST con FastAPI
Proyecto Final: Clasificación e Identificación de Placas de Motos (Colombia)
"""

import os
import uuid
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Importar módulos del proyecto
import sys
sys.path.append(str(Path(__file__).parent.parent))
from detection.detector import PlateDetector
from detection.plate_utils import normalize_plate, is_valid_colombian_plate
from ml.classifier import PlateValidator

app = FastAPI(
    title="API Placas Colombia",
    description="Clasificación e identificación de placas de motos colombianas",
    version="1.0.0"
)

# CORS para que React pueda consumir la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directorios
UPLOADS_DIR = Path(__file__).parent.parent / "uploads"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
RESULTS_FILE = RESULTS_DIR / "detections.json"
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Inicializar modelos (se cargan una sola vez)
detector = PlateDetector()
validator = PlateValidator()

def load_results() -> list:
    """Carga los resultados guardados en disco."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_result(result: dict):
    """Agrega un resultado nuevo al historial."""
    results = load_results()
    results.append(result)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ── ENDPOINTS ────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "API Placas Colombia activa"}


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Sube una imagen, ejecuta YOLO + EasyOCR,
    valida con Random Forest y devuelve resultados.
    """
    # Validar tipo de archivo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Solo se aceptan imágenes")

    # Guardar imagen
    file_id = str(uuid.uuid4())[:8]
    filename = f"{file_id}_{file.filename}"
    file_path = UPLOADS_DIR / filename

    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Ejecutar detección
    try:
        detections = detector.detect(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en detección: {str(e)}")

    # Procesar cada detección con el validador RF
    processed = []
    for det in detections:
        plate_text = normalize_plate(det.get("plate_text", ""))
        features = {
            "text_length": len(plate_text),
            "num_letters": sum(c.isalpha() for c in plate_text),
            "num_digits": sum(c.isdigit() for c in plate_text),
            "ocr_confidence": det.get("ocr_confidence", 0.0),
            "bbox_ratio": det.get("bbox_ratio", 1.0),
            "matches_pattern": int(is_valid_colombian_plate(plate_text)),
        }
        rf_valid, rf_confidence = validator.predict(features)

        record = {
            "id": file_id,
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "vehicle_type": det.get("vehicle_type", "unknown"),
            "vehicle_confidence": round(det.get("vehicle_confidence", 0.0), 3),
            "plate_text": plate_text,
            "ocr_confidence": round(det.get("ocr_confidence", 0.0), 3),
            "bbox_ratio": round(det.get("bbox_ratio", 1.0), 3),
            "rf_valid": bool(rf_valid),
            "rf_confidence": round(float(rf_confidence), 3),
            "matches_pattern": bool(features["matches_pattern"]),
        }
        processed.append(record)
        save_result(record)

    return {
        "file_id": file_id,
        "filename": filename,
        "detections_count": len(processed),
        "detections": processed,
    }


@app.post("/upload-batch")
async def upload_batch(files: List[UploadFile] = File(...)):
    """Procesa múltiples imágenes en lote (para demo con dataset)."""
    all_results = []
    for file in files[:20]:  # Límite de 20 imágenes por lote
        result = await upload_image(file)
        all_results.append(result)
    return {"batch_size": len(all_results), "results": all_results}


@app.get("/results")
def get_results(limit: int = 100):
    """Retorna todos los resultados acumulados (para las visualizaciones)."""
    results = load_results()
    return {
        "total": len(results),
        "results": results[-limit:],
    }


@app.get("/stats")
def get_stats():
    """
    Estadísticas agregadas para las visualizaciones del dashboard:
    - Distribución de tipos de vehículo
    - Métricas del clasificador RF
    - Histograma de confianza OCR
    - Placas válidas vs inválidas
    """
    results = load_results()
    if not results:
        return {"message": "No hay datos aún. Sube imágenes primero."}

    # Viz 1: Distribución tipos de vehículo
    vehicle_counts = {}
    for r in results:
        vt = r.get("vehicle_type", "unknown")
        vehicle_counts[vt] = vehicle_counts.get(vt, 0) + 1

    # Viz 2: Métricas del modelo Random Forest
    rf_metrics = validator.get_metrics()

    # Viz 3: Distribución de confianza OCR
    ocr_by_vehicle = {}
    for r in results:
        vt = r.get("vehicle_type", "unknown")
        conf = r.get("ocr_confidence", 0.0)
        if vt not in ocr_by_vehicle:
            ocr_by_vehicle[vt] = []
        ocr_by_vehicle[vt].append(conf)

    # Extra: Placas válidas vs no válidas
    valid_count = sum(1 for r in results if r.get("rf_valid"))
    invalid_count = len(results) - valid_count

    return {
        "total_detections": len(results),
        "vehicle_distribution": vehicle_counts,
        "rf_metrics": rf_metrics,
        "ocr_by_vehicle": ocr_by_vehicle,
        "plate_validity": {
            "valid": valid_count,
            "invalid": invalid_count,
        },
        "avg_vehicle_confidence": round(
            sum(r.get("vehicle_confidence", 0) for r in results) / len(results), 3
        ),
    }


@app.delete("/results/clear")
def clear_results():
    """Limpia todos los resultados (útil para demos)."""
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()
    return {"message": "Resultados limpiados"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
