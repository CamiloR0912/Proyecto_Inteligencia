# 🏍️🚗 PlacasCO — Sistema de Clasificación e Identificación de Placas

PlacasCO es un sistema basado en Inteligencia Computacional diseñado para detectar, clasificar y leer placas vehiculares colombianas (enfocado en motocicletas y automóviles). El sistema procesa imágenes mediante un pipeline de Deep Learning, las valida semánticamente, y presenta los resultados en un Dashboard analítico interactivo.

## 🏗️ Arquitectura del Sistema

El proyecto unifica el backend y el frontend en un solo servidor para facilitar su ejecución:

1. **Pipeline de Visión (Inteligencia Computacional):**
   - **YOLOv8:** Detección y clasificación multi-clase de vehículos (Motocicleta, Carro, Bus, Camión), aislando el ROI (Region of Interest).
   - **EasyOCR:** Modelo CRNN para extraer ópticamente el texto de la placa desde el recorte del vehículo.
   - **Normalización y Validación:** Algoritmo de corrección con reglas RegEx que corrige errores ópticos comunes (ej. `0` por `O`, bordes confundidos con `I` o `1`) asegurando el patrón oficial colombiano.

2. **Clasificador Random Forest:**
   - Modelo de Machine Learning clásico que valida si las características extraídas (longitud, letras, dígitos, confianza OCR, ratio del bounding box) corresponden a una placa real o a un falso positivo.

3. **Backend + Frontend unificado (FastAPI + React):**
   - Servidor que expone la lógica computacional a través de endpoints REST (`/upload`, `/stats`, `/results`) y al mismo tiempo sirve la interfaz web compilada.
   - Dashboard interactivo con 3 visualizaciones analíticas (Recharts): distribución vehicular, métricas del clasificador RF, y dispersión de confianza OCR vs YOLO.

## 🚀 Requisitos Previos

- **Python 3.10 o superior.**
- Opcional: Tarjeta gráfica compatible con CUDA para acelerar la inferencia de YOLOv8.

> **Nota:** No es necesario tener Node.js instalado. El frontend ya viene pre-compilado en la carpeta `frontend/dist/`.

## ⚙️ Instalación y Ejecución

### Opción 1: Un Clic (Recomendada)

#### Windows
1. Hacer doble clic en **`Instalar_PlacasCO.bat`** (solo la primera vez).
2. Hacer doble clic en **`Iniciar_PlacasCO.bat`**.
3. El navegador se abrirá automáticamente con la aplicación.

#### Mac / Linux
1. Hacer doble clic en **`Instalar_PlacasCO.command`** (solo la primera vez).
2. Hacer doble clic en **`Iniciar_PlacasCO.command`**.
3. El navegador se abrirá automáticamente con la aplicación.

> Si los archivos `.command` no se ejecutan, abrir una terminal y ejecutar: `chmod +x *.command`

### Opción 2: Manual (Terminal)

```bash
# 1. Crear y activar el entorno virtual
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar el servidor
python main.py
```

Abrir `http://localhost:8000` en el navegador.

## 📊 Uso del Dashboard

1. Ve a la pestaña **📤 Subir Imágenes**.
2. Arrastra una foto o múltiples fotos de vehículos (carros o motos).
3. El pipeline aislará la placa y la leerá en segundos.
4. Cambia a la pestaña **📊 Visualizaciones** para ver cómo tus detecciones alimentan los gráficos en tiempo real.
5. Revisa la pestaña **📋 Resultados** para ver el historial detallado de todas las detecciones.

## 📁 Estructura del Proyecto

```
Proyecto_Inteligencia/
├── main.py                  # Servidor FastAPI (API + Frontend)
├── requirements.txt         # Dependencias de Python
├── yolov8n.pt               # Pesos del modelo YOLOv8
├── Instalar_PlacasCO.bat    # Instalador (Windows)
├── Instalar_PlacasCO.command# Instalador (Mac/Linux)
├── Iniciar_PlacasCO.bat     # Ejecutar app (Windows)
├── Iniciar_PlacasCO.command # Ejecutar app (Mac/Linux)
├── detection/               # Módulo de detección (YOLO + OCR)
│   ├── detector.py
│   └── plate_utils.py
├── ml/                      # Clasificador Random Forest
│   ├── classifier.py
│   ├── rf_model.pkl
│   └── scaler.pkl
├── frontend/                # Interfaz web (React)
│   ├── dist/                # Frontend compilado (producción)
│   └── src/                 # Código fuente React
└── data/results/            # Historial de detecciones (JSON)
```
