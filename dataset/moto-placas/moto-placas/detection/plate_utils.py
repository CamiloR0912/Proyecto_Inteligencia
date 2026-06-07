"""
detection/plate_utils.py
Normalización y validación de placas colombianas.

Formato oficial Colombia:
  Motos:   ABC·12F  (3 letras + 2 dígitos + 1 letra)  ← actual
           ABC·12   (3 letras + 2 dígitos)              ← anterior, aún vigente
  Carros:  ABC·123  (3 letras + 3 dígitos)
  Oficial: OAB·123  (empieza con O)
  Mototaxi privado:  123·ABC
"""

import re

# Patrones de placas colombianas
PATTERNS = {
    "moto_actual":    re.compile(r"^[A-Z]{3}\d{2}[A-Z]$"),       # ABC12F
    "moto_anterior":  re.compile(r"^[A-Z]{3}\d{2}$"),             # ABC12
    "carro":          re.compile(r"^[A-Z]{3}\d{3}$"),             # ABC123
    "oficial":        re.compile(r"^O[A-Z]{2}\d{3}$"),            # OAB123
    "mototaxi":       re.compile(r"^\d{3}[A-Z]{3}$"),             # 123ABC
}

# Correcciones comunes de OCR: caracteres confundidos
OCR_CORRECTIONS = {
    "0": "O",  # cero → O (cuando va en posición de letra)
    "1": "I",  # uno  → I
    "5": "S",
    "8": "B",
    "6": "G",
    "2": "Z",
}

OCR_DIGIT_CORRECTIONS = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "G": "6",
    "Z": "2",
}


def normalize_plate(text: str) -> str:
    """
    Normaliza texto OCR a formato de placa:
    - Convierte a mayúsculas
    - Elimina espacios, puntos y guiones
    - Aplica correcciones de OCR básicas
    """
    if not text:
        return ""

    # Limpiar y mayúsculas
    text = text.upper().strip()
    text = re.sub(r"[\s\.\-·_]", "", text)

    # Intentar aplicar corrección posicional para placas estilo ABC123
    # Posiciones 0-2: letras, posiciones 3-5: dígitos (o 3-4 + 6 para motos)
    corrected = list(text)
    for i, ch in enumerate(corrected):
        if i < 3:
            # Posición de letra: corregir dígitos que parecen letras
            corrected[i] = OCR_CORRECTIONS.get(ch, ch)
        elif i >= 3:
            # Posición de dígito: corregir letras que parecen dígitos
            if not corrected[i].isdigit():
                corrected[i] = OCR_DIGIT_CORRECTIONS.get(ch, ch)

    return "".join(corrected)


def is_valid_colombian_plate(text: str) -> bool:
    """
    Retorna True si el texto normalizado coincide con algún formato
    de placa vehicular colombiana.
    """
    if not text or len(text) < 5:
        return False
    return any(pattern.match(text) for pattern in PATTERNS.values())


def classify_plate_type(text: str) -> str:
    """
    Clasifica el tipo de placa según el patrón.
    Returns: 'moto_actual' | 'moto_anterior' | 'carro' | 'oficial' | 'mototaxi' | 'desconocido'
    """
    for plate_type, pattern in PATTERNS.items():
        if pattern.match(text):
            return plate_type
    return "desconocido"


def extract_features(text: str, ocr_confidence: float = 0.0, bbox_ratio: float = 1.0) -> dict:
    """
    Extrae features para el clasificador Random Forest.
    """
    normalized = normalize_plate(text)
    return {
        "text_length": len(normalized),
        "num_letters": sum(c.isalpha() for c in normalized),
        "num_digits": sum(c.isdigit() for c in normalized),
        "ocr_confidence": ocr_confidence,
        "bbox_ratio": bbox_ratio,
        "matches_pattern": int(is_valid_colombian_plate(normalized)),
    }
