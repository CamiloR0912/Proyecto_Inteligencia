"""Random Forest validator trained from the Colombian plate CSV dataset."""

import csv
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

from detection.plate_utils import is_valid_colombian_plate, normalize_plate

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn no esta instalado. Ejecuta: pip install -r requirements.txt")


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "dataset" / "dataset_placas_colombia.csv"
MODEL_PATH = Path(__file__).parent / "rf_model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"
METRICS_PATH = Path(__file__).parent / "metrics.json"


def _features_from_plate(
    plate: str,
    ocr_confidence: float,
    bbox_ratio: float,
) -> list[float]:
    normalized = normalize_plate(plate)
    return [
        len(normalized),
        sum(c.isalpha() for c in normalized),
        sum(c.isdigit() for c in normalized),
        float(ocr_confidence),
        float(bbox_ratio),
        int(is_valid_colombian_plate(normalized)),
    ]


def _invalid_variants(plate: str) -> Iterable[str]:
    normalized = normalize_plate(plate)
    if not normalized:
        return []

    variants = {
        normalized[:4],
        normalized + "99",
        normalized[:2] + normalized[3:],
        "X" + normalized[1:4] + "Z",
        normalized.replace("0", "O").replace("1", "I"),
    }
    return [variant for variant in variants if normalize_plate(variant) != normalized]


def _load_dataset_rows() -> list[dict]:
    if not DATASET_PATH.exists():
        return []

    with open(DATASET_PATH, newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def build_training_data(extra_samples: list | None = None):
    """Builds RF features from the CSV plus generated OCR-noise negatives."""
    X, y = [], []
    rows = _load_dataset_rows()

    for index, row in enumerate(rows):
        plate = normalize_plate(row.get("Placa", ""))
        if not is_valid_colombian_plate(plate):
            continue

        is_moto = row.get("Tipo", "").strip().lower() == "moto"
        valid_ratio = 3.2 if is_moto else 2.8
        X.append(_features_from_plate(plate, 0.92, valid_ratio))
        y.append(1)

        if index % 2 == 0:
            for noisy in _invalid_variants(plate)[:1]:
                X.append(_features_from_plate(noisy, 0.35, 1.4))
                y.append(0)

    synthetic_invalid = ["A1", "ABC", "ABCDE12", "12ABC", "PLACA", "A123", ""]
    for noisy in synthetic_invalid:
        X.append(_features_from_plate(noisy, 0.25, 1.0))
        y.append(0)

    if extra_samples:
        for sample in extra_samples:
            X.append([sample[name] for name in PlateValidator.FEATURE_NAMES])
            y.append(int(sample["label"]))

    if not X:
        raise RuntimeError(f"No se encontraron placas validas en {DATASET_PATH}")

    return np.array(X, dtype=float), np.array(y, dtype=int), len(rows)


class PlateValidator:
    """Classifies OCR readings as valid or invalid Colombian plates."""

    FEATURE_NAMES = [
        "text_length",
        "num_letters",
        "num_digits",
        "ocr_confidence",
        "bbox_ratio",
        "matches_pattern",
    ]

    def __init__(self):
        self.model = None
        self.scaler = None
        self._metrics: dict = {}
        self._dataset_index = self._build_dataset_index()

        if MODEL_PATH.exists() and SCALER_PATH.exists():
            self._load()

        if self.model is None or self._metrics.get("dataset_source") != DATASET_PATH.name:
            self._train_and_save()

    def _build_dataset_index(self) -> dict:
        index = {}
        for row in _load_dataset_rows():
            plate = normalize_plate(row.get("Placa", ""))
            if plate:
                index[plate] = {
                    "plate": plate,
                    "type": row.get("Tipo", ""),
                    "service": row.get("Servicio", ""),
                    "registration_location": row.get("Ubicacion_Inscripcion", ""),
                }
        return index

    def _train_and_save(self, extra_samples: list | None = None):
        if not SKLEARN_AVAILABLE:
            return

        X, y, dataset_rows = build_training_data(extra_samples)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(X_train_s, y_train)

        y_pred = self.model.predict(X_test_s)
        self._metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_importances": {
                name: round(float(importance), 4)
                for name, importance in zip(
                    self.FEATURE_NAMES,
                    self.model.feature_importances_,
                )
            },
            "n_estimators": self.model.n_estimators,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "dataset_rows": dataset_rows,
            "dataset_source": DATASET_PATH.name,
        }

        with open(MODEL_PATH, "wb") as file:
            pickle.dump(self.model, file)
        with open(SCALER_PATH, "wb") as file:
            pickle.dump(self.scaler, file)
        with open(METRICS_PATH, "w", encoding="utf-8") as file:
            json.dump(self._metrics, file, indent=2)

        print(f"Random Forest entrenado con {DATASET_PATH.name}")

    def retrain_with_results(self, new_samples: list):
        if not SKLEARN_AVAILABLE or len(new_samples) < 5:
            return False
        self._train_and_save(extra_samples=new_samples)
        return True

    def _load(self):
        with open(MODEL_PATH, "rb") as file:
            self.model = pickle.load(file)
        with open(SCALER_PATH, "rb") as file:
            self.scaler = pickle.load(file)
        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r", encoding="utf-8") as file:
                self._metrics = json.load(file)

    def predict(self, features: Dict) -> Tuple[bool, float]:
        if self.model is None:
            return bool(features.get("matches_pattern")), 0.5

        X = np.array([[features[name] for name in self.FEATURE_NAMES]], dtype=float)
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        prob = self.model.predict_proba(X_scaled)[0][1]
        return bool(pred), float(prob)

    def lookup_plate(self, plate_text: str) -> dict | None:
        return self._dataset_index.get(normalize_plate(plate_text))

    def correct_plate(self, plate_text: str, candidates: list | None = None) -> str:
        normalized = normalize_plate(plate_text)
        if normalized in self._dataset_index:
            return normalized

        for candidate in candidates or []:
            candidate_text = normalize_plate(candidate.get("plate_text", ""))
            if candidate_text in self._dataset_index:
                return candidate_text

        return normalized

    def get_dataset_summary(self) -> dict:
        rows = list(self._dataset_index.values())
        type_counts = {}
        service_counts = {}
        city_counts = {}
        for row in rows:
            type_counts[row["type"]] = type_counts.get(row["type"], 0) + 1
            service_counts[row["service"]] = service_counts.get(row["service"], 0) + 1
            city = row["registration_location"]
            city_counts[city] = city_counts.get(city, 0) + 1
        return {
            "source": DATASET_PATH.name,
            "total_plates": len(rows),
            "type_distribution": type_counts,
            "service_distribution": service_counts,
            "top_locations": dict(
                sorted(city_counts.items(), key=lambda item: item[1], reverse=True)[:10]
            ),
        }

    def get_metrics(self) -> dict:
        return self._metrics
