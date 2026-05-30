"""Utilidades para normalización y validación de placas vehiculares."""
import re

# Formato de placa colombiana: 3 letras + 3 números (ej: ABC123)
PLATE_REGEX = re.compile(r"^[A-Z]{3}\d{3}$")

# Clases YOLO (COCO dataset) correspondientes a vehículos
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Mapeo de errores comunes de OCR por posición
_LETTER_FIXES = {"0": "O", "1": "I", "5": "S"}
_DIGIT_FIXES = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8"}


def normalize_plate(text: str) -> str:
    """Normaliza una placa colombiana corrigiendo errores comunes de OCR.

    Aplica correcciones posicionales:
    - Primeros 3 caracteres deben ser letras
    - Últimos 3 caracteres deben ser números
    """
    if not text:
        return ""

    text = text.upper().replace(" ", "")
    text = re.sub(r"[^A-Z0-9]", "", text)

    if len(text) < 6:
        return text

    corrected = ""
    for i, char in enumerate(text[:6]):
        if i < 3:  # Parte de letras
            corrected += _LETTER_FIXES.get(char, char)
        else:  # Parte de números
            corrected += _DIGIT_FIXES.get(char, char)

    return corrected + text[6:]


def is_valid_plate(text: str) -> bool:
    """Verifica si el texto tiene formato de placa colombiana válida."""
    return bool(PLATE_REGEX.match(text))
