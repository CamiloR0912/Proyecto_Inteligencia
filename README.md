# 🏍️🚗 PlacasCO — Sistema de Clasificación e Identificación de Placas

PlacasCO es un sistema integral basado en Inteligencia Computacional diseñado para detectar, clasificar y leer placas vehiculares (enfocado en motocicletas y automóviles colombianos). El sistema consta de un pipeline de Deep Learning que procesa imágenes en tiempo real, validándolas semánticamente, y un Dashboard analítico para su monitoreo.

## 🏗️ Arquitectura del Sistema

El proyecto está construido bajo una arquitectura de microservicios separando el procesamiento pesado de la interfaz gráfica:

1. **Inteligencia Computacional (Pipeline de Visión):**
   - **YOLOv8:** Utilizado para la detección y clasificación multi-clase (distingue entre Motocicleta, Carro, Bus, etc.) aislando el ROI (Region of Interest).
   - **EasyOCR:** Modelo CRNN empleado para extraer ópticamente el texto de la placa del vehículo recortado.
   - **Módulo PLN (Procesamiento de Lenguaje Natural):** Algoritmo de normalización y reglas RegEx que corrige errores ópticos comunes (ej. `0` por `O`) asegurando el patrón oficial de Colombia.

2. **Backend (FastAPI):**
   - Servidor asíncrono que expone la lógica computacional a través de endpoints REST (`/upload`, `/stats`, `/results`).
   - Almacena el historial de detecciones para visualizaciones históricas.

3. **Frontend (React + Vite):**
   - Dashboard interactivo y dinámico que consume el API en tiempo real.
   - Incorpora 3 visualizaciones analíticas con Recharts: Distribución vehicular, métricas de validez de OCR y dispersión de confianza estadística.

## 🚀 Requisitos Previos

- Python 3.10 o superior (Testeado con entornos hasta 3.14 Alfa).
- Node.js (v18 o superior) y npm.
- Opcional: Tarjeta gráfica compatible con CUDA para acelerar la inferencia de YOLOv8.

## ⚙️ Instalación y Ejecución

Debes levantar ambos servicios (Backend y Frontend) en terminales separadas.

### 1. Levantar el Backend (FastAPI)
Abre una terminal en la raíz del proyecto (`Proyecto_Inteligencia/`):

```bash
# 1. Activar el entorno virtual (opcional pero recomendado)
.\venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar el servidor API
python api\app.py
```
*El backend quedará corriendo en `http://localhost:8000`.*

> **Nota sobre scikit-learn:** Si estás corriendo una versión de Python muy reciente (ej. 3.14 Alfa) para la cual no existen binarios de scikit-learn, el sistema automáticamente hará un fallback a la validación estricta por reglas (PLN/RegEx), por lo que la aplicación seguirá funcionando sin interrupciones.

### 2. Levantar el Frontend (React)
Abre una **nueva terminal** y navega a la carpeta `frontend/`:

```bash
# 1. Entrar al directorio
cd frontend

# 2. Instalar dependencias de Node
npm install

# 3. Iniciar el servidor de desarrollo
npm run dev
```
*El dashboard quedará corriendo en `http://localhost:5173`. Abre este enlace en tu navegador.*

## 📊 Uso del Dashboard
1. Ve a la pestaña **📤 Subir Imágenes**.
2. Arrastra una foto o múltiples fotos de vehículos (carros o motos).
3. El pipeline aislará la placa y la leerá en fracciones de segundo.
4. Cambia a la pestaña **📊 Visualizaciones** para ver cómo tus detecciones alimentan los gráficos en tiempo real.
