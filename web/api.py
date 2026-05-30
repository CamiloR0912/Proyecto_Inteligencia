"""API FastAPI — endpoints REST, WebSocket y loop de detección."""
import asyncio
import json
import logging
import threading
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from camera import CameraManager, generate_mjpeg
from detection import PlateDetector, PlateTracker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Instancias globales ──────────────────────────────────────────────
camera_mgr = CameraManager()
detector: PlateDetector | None = None  # Se inicializa en startup
tracker = PlateTracker()

# Frame compartido entre hilos (captura y stream)
_latest_frame = None
_frame_lock = threading.Lock()

# Historial de placas confirmadas (últimas 50)
plate_log: list[dict] = []
MAX_LOG = 50

# Clientes WebSocket conectados
_ws_clients: set[WebSocket] = set()
_event_loop: asyncio.AbstractEventLoop | None = None  # Event loop principal

# ── App FastAPI ──────────────────────────────────────────────────────
app = FastAPI(title="SmartPark AI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Lifecycle ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global detector, _event_loop

    # Capturar el event loop para usarlo desde hilos background
    _event_loop = asyncio.get_running_loop()

    # Iniciar cámara
    try:
        camera_mgr.add_camera("entrada", 0)
        logger.info("📷 Cámara 'entrada' lista.")
    except Exception as e:
        logger.error(f"⚠️ No se pudo abrir la cámara: {e}")
        return  # No iniciar hilos si no hay cámara

    # Cargar modelos de IA
    detector = PlateDetector()

    # Hilo 1: Captura rápida de frames para el stream
    t1 = threading.Thread(target=_capture_loop, daemon=True)
    t1.start()
    logger.info("🎥 Loop de captura iniciado (~30 FPS).")

    # Hilo 2: Detección de placas (más lenta, en paralelo)
    t2 = threading.Thread(target=_detection_loop, daemon=True)
    t2.start()
    logger.info("🚀 Loop de detección iniciado.")


@app.on_event("shutdown")
def shutdown():
    camera_mgr.release()
    logger.info("👋 Servidor finalizado.")


# ── Captura rápida de frames (hilo 1) ───────────────────────────────
def _capture_loop():
    """Lee frames de la cámara a ~30 FPS para un stream fluido."""
    global _latest_frame

    while True:
        try:
            frame = camera_mgr.get_frame("entrada")
            with _frame_lock:
                _latest_frame = frame
        except Exception as e:
            logger.error(f"Error capturando frame: {e}")
            time.sleep(1)
            continue
        time.sleep(0.033)  # ~30 FPS


# ── Detección de placas (hilo 2, más lento) ─────────────────────────
_latest_detections = []
_detections_lock = threading.Lock()


def _detection_loop():
    """Toma frames, ejecuta IA y envía resultados por WebSocket."""
    global _latest_detections

    while True:
        # Tomar el frame más reciente
        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            time.sleep(0.1)
            continue

        # Ejecutar YOLO + EasyOCR (puede tardar varios segundos en CPU)
        try:
            detections = detector.detect(frame.copy())
        except Exception as e:
            logger.error(f"Error en detección: {e}")
            time.sleep(0.5)
            continue

        # Guardar detecciones para anotación
        with _detections_lock:
            _latest_detections = detections

        # Procesar cada detección en el tracker
        for det in detections:
            confirmed = tracker.add_reading(
                det.vehicle_type, det.plate_text, det.ocr_confidence
            )
            if confirmed:
                entry = {
                    "plate": confirmed.plate,
                    "vehicle_type": confirmed.vehicle_type,
                    "confidence": round(confirmed.confidence, 2),
                    "timestamp": confirmed.timestamp,
                }
                plate_log.append(entry)
                if len(plate_log) > MAX_LOG:
                    plate_log.pop(0)

                # Broadcast a todos los clientes WebSocket
                _broadcast(entry)

        # Pequeña pausa para no saturar CPU
        time.sleep(0.1)


def _broadcast(data: dict):
    """Envía datos a todos los clientes WebSocket conectados."""
    if not _event_loop or not _ws_clients:
        return

    message = json.dumps(data, ensure_ascii=False)

    async def _send_all():
        disconnected = set()
        for ws in _ws_clients.copy():
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        _ws_clients -= disconnected

    asyncio.run_coroutine_threadsafe(_send_all(), _event_loop)


# ── Endpoints ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    """Sirve la interfaz web."""
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/stream")
def video_stream():
    """Stream MJPEG del video con anotaciones de detección."""
    def get_annotated_frame():
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            return None
        # Anotar con las últimas detecciones disponibles
        with _detections_lock:
            dets = _latest_detections
        if dets and detector:
            return detector.annotate_frame(frame, dets)
        return frame

    return StreamingResponse(
        generate_mjpeg(get_annotated_frame),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/cameras")
def list_cameras():
    """Lista las cámaras registradas."""
    return {"cameras": camera_mgr.list_cameras()}


@app.get("/api/plates")
def get_plates():
    """Retorna el historial de placas detectadas."""
    return {"plates": plate_log}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket para recibir detecciones en tiempo real."""
    await ws.accept()
    _ws_clients.add(ws)
    logger.info(f"🔌 Cliente WebSocket conectado. Total: {len(_ws_clients)}")
    try:
        while True:
            # Mantener la conexión viva esperando mensajes del cliente
            await ws.receive_text()
    except WebSocketDisconnect:
        _ws_clients.discard(ws)
        logger.info(f"🔌 Cliente desconectado. Total: {len(_ws_clients)}")
