"""Utilities for Colombian license plate normalization and validation."""

import re


PATTERNS = {
    "moto_actual": re.compile(r"^[A-Z]{3}\d{2}[A-Z]$"),
    "moto_anterior": re.compile(r"^[A-Z]{3}\d{2}$"),
    "carro": re.compile(r"^[A-Z]{3}\d{3}$"),
    "oficial": re.compile(r"^O[A-Z]{2}\d{3}$"),
    "mototaxi": re.compile(r"^\d{3}[A-Z]{3}$"),
}

OCR_LETTER_CORRECTIONS = {
    "0": "O",
    "1": "I",
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


def _as_letter(ch: str) -> str:
    return OCR_LETTER_CORRECTIONS.get(ch, ch)


def _as_digit(ch: str) -> str:
    return OCR_DIGIT_CORRECTIONS.get(ch, ch)


def _candidate(text: str, roles: str) -> str:
    chars = []
    for ch, role in zip(text, roles):
        chars.append(_as_letter(ch) if role == "L" else _as_digit(ch))
    return "".join(chars)


def normalize_plate(text: str) -> str:
    """Normalize OCR text while respecting Colombian moto plate formats."""
    if not text:
        return ""

    cleaned = text.upper().strip()
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)
    if not cleaned:
        return ""

    role_options = {
        5: ["LLLDD"],
        6: ["LLLDDL", "LLLDDD", "DDDLLL"],
    }.get(len(cleaned), [])

    candidates = [_candidate(cleaned, roles) for roles in role_options]
    candidates.append(cleaned)

    # Añadir variantes quitando la "I" inicial y el "1" final (artefactos del borde)
    if cleaned.startswith("I"):
        candidates.append(cleaned[1:])
    if cleaned.endswith("1"):
        candidates.append(cleaned[:-1])
    if cleaned.startswith("I") and cleaned.endswith("1"):
        candidates.append(cleaned[1:-1])

    for candidate in candidates:
        if is_valid_colombian_plate(candidate):
            return candidate

    return candidates[0] if candidates else cleaned


def is_valid_colombian_plate(text: str) -> bool:
    if not text or len(text) < 5:
        return False
    return any(pattern.match(text) for pattern in PATTERNS.values())


def classify_plate_type(text: str) -> str:
    normalized = normalize_plate(text)
    for plate_type, pattern in PATTERNS.items():
        if pattern.match(normalized):
            return plate_type
    return "desconocido"


def extract_features(
    text: str,
    ocr_confidence: float = 0.0,
    bbox_ratio: float = 1.0,
) -> dict:
    normalized = normalize_plate(text)
    return {
        "text_length": len(normalized),
        "num_letters": sum(c.isalpha() for c in normalized),
        "num_digits": sum(c.isdigit() for c in normalized),
        "ocr_confidence": ocr_confidence,
        "bbox_ratio": bbox_ratio,
        "matches_pattern": int(is_valid_colombian_plate(normalized)),
    }
