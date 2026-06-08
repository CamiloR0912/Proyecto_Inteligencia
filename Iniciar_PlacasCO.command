#!/bin/bash
echo "============================================"
echo "  PlacasCO - Iniciando servidor..."
echo "============================================"
echo ""

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "ERROR: No se encontro el entorno virtual."
    echo "Ejecuta primero: Instalar_PlacasCO.command"
    read -p "Presiona Enter para salir..."
    exit 1
fi

source venv/bin/activate

echo "Cargando modelos de IA (YOLOv8 + EasyOCR)..."
echo "Esto puede tardar unos segundos la primera vez."
echo ""

# Abrir el navegador tras 8 segundos
(sleep 8 && open http://localhost:8000) &

echo ""
echo "============================================"
echo "  PlacasCO esta corriendo!"
echo "  No cierres esta ventana."
echo "  Para detener, presiona Ctrl+C."
echo "============================================"
echo ""

python3 main.py
