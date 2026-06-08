@echo off
chcp 65001 >nul 2>&1
echo ============================================
echo   PlacasCO - Instalador (Windows)
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python no esta instalado.
        echo Descargalo en https://www.python.org/downloads/
        pause
        exit /b 1
    )
)
echo       Python encontrado.

echo [2/3] Creando entorno virtual...
if not exist "venv" (
    python -m venv venv 2>nul || py -m venv venv
)
echo       Entorno virtual listo.

echo [3/3] Instalando dependencias (esto puede tardar unos minutos)...
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo.
echo ============================================
echo   Instalacion completada exitosamente!
echo   Ahora ejecuta: Iniciar_PlacasCO.bat
echo ============================================
pause
