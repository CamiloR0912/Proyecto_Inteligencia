#!/bin/bash
echo "============================================"
echo "  PlacasCO - Instalador (Mac/Linux)"
echo "============================================"
echo ""

cd "$(dirname "$0")"

echo "[1/3] Verificando Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 no esta instalado."
    echo "Instala Python desde https://www.python.org/downloads/"
    read -p "Presiona Enter para salir..."
    exit 1
fi
echo "      Python3 encontrado."

echo "[2/3] Creando entorno virtual..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
echo "      Entorno virtual listo."

echo "[3/3] Instalando dependencias (esto puede tardar unos minutos)..."
source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "============================================"
echo "  Instalacion completada exitosamente!"
echo "  Ahora ejecuta: Iniciar_PlacasCO.command"
echo "============================================"
read -p "Presiona Enter para salir..."
