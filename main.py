"""PlacasCO - punto de entrada principal.

Ejecuta:
    python main.py

Abre http://localhost:8000/docs para probar la API.
"""

import uvicorn


if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
