"""Administrador de cámaras con soporte para múltiples fuentes."""
import cv2
from threading import Lock


class CameraManager:
    """Gestiona cámaras (webcam local, RTSP, HTTP) de forma thread-safe."""

    def __init__(self):
        self.cameras: dict[str, cv2.VideoCapture] = {}
        self._locks: dict[str, Lock] = {}

    def add_camera(self, cam_id: str, source: int | str) -> None:
        """Registra una cámara. `source` puede ser 0, 1 o una URL RTSP/HTTP."""
        import logging
        cap = None
        
        def try_open(src, backend=None):
            c = cv2.VideoCapture(src, backend) if backend else cv2.VideoCapture(src)
            if c.isOpened():
                # Leer algunos frames para verificar que la cámara sí entrega imagen
                for _ in range(3):
                    ret, _ = c.read()
                    if ret: 
                        return c
                c.release()
            return None

        # En Windows, intentar varios backends
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            source_idx = int(source)
            cap = try_open(source_idx, cv2.CAP_DSHOW)
            if not cap:
                logging.warning(f"⚠️ Falló DSHOW para cámara {source_idx}, intentando MSMF...")
                cap = try_open(source_idx, cv2.CAP_MSMF)
            if not cap:
                logging.warning(f"⚠️ Falló MSMF para cámara {source_idx}, intentando default...")
                cap = try_open(source_idx)
        else:
            cap = try_open(source)
            
        if not cap:
            raise ValueError(f"No se pudo abrir la cámara o no entrega frames: {source}. Revisa permisos o si otra app la usa.")
            
        self.cameras[cam_id] = cap
        self._locks[cam_id] = Lock()

    def get_frame(self, cam_id: str):
        """Obtiene el frame actual de una cámara registrada."""
        if cam_id not in self.cameras:
            raise ValueError(f"Cámara '{cam_id}' no registrada.")
        with self._locks[cam_id]:
            ret, frame = self.cameras[cam_id].read()
        if not ret:
            raise RuntimeError(f"No se pudo leer frame de cámara '{cam_id}'")
        return frame

    def list_cameras(self) -> list[str]:
        """Retorna los IDs de cámaras registradas."""
        return list(self.cameras.keys())

    def release(self) -> None:
        """Libera todas las cámaras."""
        for cap in self.cameras.values():
            cap.release()
        self.cameras.clear()
        self._locks.clear()
