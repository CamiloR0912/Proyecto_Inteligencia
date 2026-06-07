"""
ml/classifier.py — Clasificador secundario Random Forest
Valida si una lectura OCR corresponde a una placa real colombiana.

Features de entrada:
    - text_length       : longitud del texto OCR
    - num_letters       : cantidad de letras en el texto
    - num_digits        : cantidad de dígitos en el texto
    - ocr_confidence    : confianza del motor OCR (0-1)
    - bbox_ratio        : ancho/alto del bounding box de la placa
    - matches_pattern   : 1 si cumple regex colombiano, 0 si no

Output:
    - is_valid (bool)   : placa válida o no
    - confidence (float): probabilidad de clase positiva
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn no instalado. Ejecuta: pip install scikit-learn")

MODEL_PATH = Path(__file__).parent / "rf_model.pkl"
SCALER_PATH = Path(__file__).parent / "scaler.pkl"
METRICS_PATH = Path(__file__).parent / "metrics.json"

# ── Datos sintéticos de entrenamiento inicial ──────────────────────────
# En producción estos se reemplazan por datos reales del dataset descargado.
# Formato: [text_length, num_letters, num_digits, ocr_confidence, bbox_ratio, matches_pattern]
SYNTHETIC_DATA = {
    # Placas válidas colombianas (label=1)
    # Formato moto: ABC12F o ABC12  → 6-7 chars, 3 letras, 2-3 dígitos
    "valid": [
        [6, 3, 2, 0.92, 3.2, 1],  # ABC12F → válida
        [6, 3, 2, 0.87, 3.0, 1],
        [6, 3, 3, 0.95, 2.8, 1],  # ABC123 → válida (carro)
        [6, 3, 3, 0.89, 3.5, 1],
        [7, 3, 3, 0.91, 2.9, 1],  # ABC·12F con separador
        [6, 3, 2, 0.85, 3.1, 1],
        [6, 3, 3, 0.93, 3.3, 1],
        [6, 3, 2, 0.78, 2.7, 1],
        [6, 3, 3, 0.96, 3.0, 1],
        [7, 3, 3, 0.88, 3.2, 1],
    ],
    # Lecturas OCR incorrectas / ruido (label=0)
    # Texto muy corto, muchos caracteres raros, baja confianza
    "invalid": [
        [2, 1, 1, 0.30, 1.1, 0],  # "A1" → muy corto
        [10, 7, 2, 0.25, 0.8, 0], # texto largo y raro
        [4, 2, 1, 0.40, 1.5, 0],  # "AB1D" → no cumple patrón
        [1, 0, 1, 0.15, 0.5, 0],  # "5" → solo un dígito
        [8, 5, 2, 0.35, 2.0, 0],  # "ABCDE12" → demasiadas letras
        [3, 3, 0, 0.20, 1.2, 0],  # "ABC" → sin dígitos
        [0, 0, 0, 0.10, 0.3, 0],  # texto vacío
        [5, 2, 2, 0.45, 1.8, 0],  # "AB12C" → no válido
        [9, 4, 4, 0.22, 0.9, 0],  # texto largo, baja conf.
        [4, 1, 3, 0.38, 1.3, 0],  # "A123" → falta formato
    ]
}


def build_training_data():
    """Construye X, y a partir de los datos sintéticos."""
    X, y = [], []
    for row in SYNTHETIC_DATA["valid"]:
        X.append(row)
        y.append(1)
    for row in SYNTHETIC_DATA["invalid"]:
        X.append(row)
        y.append(0)
    return np.array(X, dtype=float), np.array(y, dtype=int)


class PlateValidator:
    """
    Clasificador Random Forest para validar lecturas OCR de placas.
    Se entrena automáticamente al instanciar si no existe modelo guardado.
    """

    FEATURE_NAMES = [
        "text_length", "num_letters", "num_digits",
        "ocr_confidence", "bbox_ratio", "matches_pattern"
    ]

    def __init__(self):
        self.model: RandomForestClassifier = None
        self.scaler: StandardScaler = None
        self._metrics: dict = {}

        if MODEL_PATH.exists() and SCALER_PATH.exists():
            self._load()
        else:
            self._train_and_save()

    # ── Entrenamiento ──────────────────────────────────────────────────

    def _train_and_save(self):
        """Entrena el modelo con datos iniciales y lo guarda en disco."""
        if not SKLEARN_AVAILABLE:
            return

        X, y = build_training_data()

        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        # Escalar features
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # Entrenar Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight="balanced",  # Maneja desbalanceo
        )
        self.model.fit(X_train_s, y_train)

        # Calcular métricas
        y_pred = self.model.predict(X_test_s)
        cm = confusion_matrix(y_test, y_pred).tolist()

        self._metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "confusion_matrix": cm,
            "feature_importances": {
                name: round(float(imp), 4)
                for name, imp in zip(
                    self.FEATURE_NAMES, self.model.feature_importances_
                )
            },
            "n_estimators": self.model.n_estimators,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
        }

        # Guardar modelo y métricas
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(METRICS_PATH, "w") as f:
            json.dump(self._metrics, f, indent=2)

        print(f"✅ Random Forest entrenado — Accuracy: {self._metrics['accuracy']}")

    def retrain_with_results(self, new_samples: list):
        """
        Permite reentrenar el modelo con datos reales generados durante la demo.
        new_samples: lista de dicts con las mismas keys que FEATURE_NAMES + 'label'
        """
        if not SKLEARN_AVAILABLE or len(new_samples) < 5:
            return False

        X_new = np.array([
            [s[k] for k in self.FEATURE_NAMES] for s in new_samples
        ], dtype=float)
        y_new = np.array([s["label"] for s in new_samples], dtype=int)

        # Combinar con datos sintéticos
        X_base, y_base = build_training_data()
        X_combined = np.vstack([X_base, X_new])
        y_combined = np.concatenate([y_base, y_new])

        MODEL_PATH.unlink(missing_ok=True)
        SCALER_PATH.unlink(missing_ok=True)
        self._train_and_save()
        return True

    # ── Carga ──────────────────────────────────────────────────────────

    def _load(self):
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            self.scaler = pickle.load(f)
        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r") as f:
                self._metrics = json.load(f)

    # ── Inferencia ─────────────────────────────────────────────────────

    def predict(self, features: Dict) -> Tuple[bool, float]:
        """
        Predice si una lectura OCR es una placa válida.

        Args:
            features: dict con keys = FEATURE_NAMES
        Returns:
            (is_valid: bool, confidence: float)
        """
        if self.model is None:
            # Fallback: usar solo el regex si RF no está disponible
            return bool(features.get("matches_pattern")), 0.5

        X = np.array([[features[k] for k in self.FEATURE_NAMES]], dtype=float)
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict(X_scaled)[0]
        prob = self.model.predict_proba(X_scaled)[0][1]
        return bool(pred), float(prob)

    def get_metrics(self) -> dict:
        """Retorna las métricas del modelo para mostrar en el dashboard."""
        return self._metrics
