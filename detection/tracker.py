"""Rastreo temporal de placas con consenso por votación y cooldown."""
import time
import logging
from collections import Counter
from difflib import SequenceMatcher
from dataclasses import dataclass, field

from .plate_utils import normalize_plate, is_valid_colombian_plate as is_valid_plate

logger = logging.getLogger(__name__)

# Configuración de detección
COOLDOWN = 60        # Segundos entre envíos de la misma placa
MIN_VOTES = 3        # Votos mínimos para consenso directo
MIN_FRAMES = 5       # Frames mínimos antes de evaluar consenso
MAX_HISTORY = 10     # Límite de frames en historial
RESET_THRESHOLD = 3  # Frames de placa diferente para resetear


@dataclass
class ConfirmedPlate:
    """Resultado de una placa confirmada por consenso."""
    plate: str
    vehicle_type: str
    confidence: float
    timestamp: str


@dataclass
class _VehicleState:
    """Estado interno de tracking por tipo de vehículo."""
    history: list = field(default_factory=list)
    last_sent_time: float = 0.0
    last_sent_plate: str = ""
    last_detected: str = ""


class PlateTracker:
    """Acumula lecturas de OCR y confirma placas por consenso."""

    def __init__(self):
        self._states: dict[str, _VehicleState] = {}

    def _get_state(self, vehicle_type: str) -> _VehicleState:
        if vehicle_type not in self._states:
            self._states[vehicle_type] = _VehicleState()
        return self._states[vehicle_type]

    def add_reading(self, vehicle_type: str, raw_text: str, confidence: float) -> ConfirmedPlate | None:
        """Agrega una lectura de OCR y retorna ConfirmedPlate si hay consenso.

        Returns:
            ConfirmedPlate si se confirmó una placa nueva, None si no.
        """
        text = normalize_plate(raw_text)
        if len(text) < 5 or confidence < 0.5:
            return None

        state = self._get_state(vehicle_type)
        now = time.time()

        # Si está en cooldown y es la misma placa, ignorar
        if (state.last_sent_plate
                and now - state.last_sent_time < COOLDOWN
                and normalize_plate(text) == state.last_sent_plate):
            return None

        # Detectar cambio de vehículo por diferencia de placas
        if state.history:
            recent = [normalize_plate(p) for p in state.history[-RESET_THRESHOLD:]]
            similarities = [SequenceMatcher(None, text, p).ratio() for p in recent]
            if similarities and max(similarities) < 0.5:
                logger.info(f"🔄 Nuevo vehículo ({vehicle_type}). Reseteando historial.")
                state.history.clear()

        state.history.append(text)
        state.last_detected = text

        # Limitar tamaño del historial
        if len(state.history) > MAX_HISTORY:
            state.history = state.history[-MAX_HISTORY:]

        logger.info(
            f"📸 {vehicle_type}: '{text}' (conf={confidence:.2f}) "
            f"[{len(state.history)} frames]"
        )

        # Evaluar consenso si hay suficientes frames
        if len(state.history) < MIN_FRAMES:
            return None

        consensus = self._get_consensus(state.history)
        if not consensus:
            logger.warning(f"❌ Sin consenso para {vehicle_type}. Reseteando.")
            state.history.clear()
            return None

        # Verificar cooldown
        if (state.last_sent_plate == consensus
                and now - state.last_sent_time < COOLDOWN):
            state.history.clear()
            return None

        # ¡Placa confirmada!
        state.last_sent_time = now
        state.last_sent_plate = consensus
        state.history.clear()

        from datetime import datetime
        confirmed = ConfirmedPlate(
            plate=consensus,
            vehicle_type=vehicle_type,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
        )
        logger.info(f"✅ Placa confirmada: {consensus} ({vehicle_type})")
        return confirmed

    @staticmethod
    def _get_consensus(plates: list[str]) -> str | None:
        """Obtiene la placa más confiable por votación + similitud."""
        normed = [normalize_plate(p) for p in plates if len(p) >= 5]
        if not normed:
            return None

        valid = [p for p in normed if len(p) == 6] or normed
        cnt = Counter(valid)
        best, count = cnt.most_common(1)[0]

        # Consenso directo por votación
        if count >= MIN_VOTES and is_valid_plate(best):
            logger.info(f"🎯 Consenso por votación: {best} ({count} votos)")
            return best

        # Fallback: consenso por similitud
        for candidate, _ in cnt.most_common(3):
            if len(candidate) != 6:
                continue
            ratios = [SequenceMatcher(None, candidate, o).ratio() for o in valid]
            avg_sim = sum(ratios) / len(ratios)
            if avg_sim > 0.8 and is_valid_plate(candidate):
                logger.info(f"🎯 Consenso por similitud: {candidate} (sim={avg_sim:.2f})")
                return candidate

        return None
