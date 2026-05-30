"""SmartPark AI — Punto de entrada único.

Ejecuta:
    python main.py

Abre http://localhost:8000 en tu navegador.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "web.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
