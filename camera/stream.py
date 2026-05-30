"""Generador de stream MJPEG para transmisión HTTP."""
import time

import cv2


def generate_mjpeg(get_frame_fn) -> bytes:
    """Genera un stream MJPEG continuo.

    Args:
        get_frame_fn: Callable que retorna un frame (numpy array) o None.
    """
    while True:
        try:
            frame = get_frame_fn()
            if frame is None:
                time.sleep(0.05)  # Evitar busy-wait
                continue
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
            time.sleep(0.033)  # ~30 FPS máximo
        except Exception:
            break
