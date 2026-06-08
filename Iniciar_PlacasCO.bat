@echo off
chcp 65001 >nul 2>&1
echo ============================================
echo   PlacasCO - Iniciando servidor...
echo ============================================
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: No se encontro el entorno virtual.
    echo Ejecuta primero: Instalar_PlacasCO.bat
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo Cargando modelos de IA (YOLOv8 + EasyOCR)...
echo Esto puede tardar unos segundos la primera vez.
echo.

start /b python main.py

echo Esperando a que el servidor inicie...
timeout /t 8 /nobreak >nul

echo Abriendo navegador en http://localhost:8000
start http://localhost:8000

echo.
echo ============================================
echo   PlacasCO esta corriendo!
echo   No cierres esta ventana.
echo   Para detener, presiona Ctrl+C o cierra esta ventana.
echo ============================================
echo.

python main.py
