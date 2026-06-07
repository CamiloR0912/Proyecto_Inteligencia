# 🏍️ PlacasCO — Guía de Instalación y Ejecución

## Estructura del proyecto

```
moto-placas/
├── detection/
│   ├── detector.py       ← YOLOv8 + EasyOCR (reemplazar con tu código)
│   └── plate_utils.py    ← Normalización placas colombianas
├── ml/
│   └── classifier.py     ← Random Forest (se entrena automáticamente)
├── api/
│   └── app.py            ← FastAPI REST API
├── frontend/
│   ├── src/
│   │   ├── App.jsx       ← Dashboard React con 3 visualizaciones
│   │   ├── index.css     ← Estilos
│   │   └── main.jsx      ← Entry point
│   ├── package.json
│   └── vite.config.js
├── data/results/         ← JSON con detecciones (se crea automático)
├── uploads/              ← Imágenes subidas (se crea automático)
├── yolov8n.pt            ← Copiar tu modelo YOLO aquí
└── requirements.txt
```

---

## Paso 1 — Instalar dependencias Python

```bash
# En la raíz del proyecto
pip install -r requirements.txt
```

> ⚠️ Si `easyocr` o `ultralytics` tardan, son pesados (~1-2GB). Normal.

---

## Paso 2 — Integrar tu detector existente

Copia el contenido de tu `Proyecto_Inteligencia/detection/detector.py` al método
`detect()` de `detection/detector.py`. El archivo ya tiene la estructura compatible.

Si ya tienes un `PlateDetector` funcionando, simplemente copia el archivo completo
y ajusta los imports.

---

## Paso 3 — Dataset (Roboflow)

### Dataset recomendado: Placas Colombia (ITM)
URL: https://universe.roboflow.com/itm-mprof/placas-colombia

```python
# Instalar CLI de Roboflow
pip install roboflow

# Descargar (necesitas cuenta gratuita en roboflow.com)
from roboflow import Roboflow
rf = Roboflow(api_key="TU_API_KEY")  # Gratis en roboflow.com
project = rf.workspace("itm-mprof").project("placas-colombia")
dataset = project.version(1).download("yolov8")
```

### Dataset secundario: Motorcycle License Plate Detection
URL: https://universe.roboflow.com/motorcycle-9gyny/motorcycle-license-plate-detection

Descarga las imágenes a `data/dataset/` para usarlas en la demo de subida por lotes.

---

## Paso 4 — Iniciar el backend

```bash
# Desde la raíz del proyecto
cd api
uvicorn app:app --reload --port 8000
```

El API estará en: http://localhost:8000
Documentación automática: http://localhost:8000/docs

---

## Paso 5 — Iniciar el frontend

```bash
cd frontend
npm install
npm run dev
```

El dashboard estará en: http://localhost:5173

---

## Uso del prototipo

1. Abre http://localhost:5173
2. Ve a la pestaña **Subir Imágenes**
3. Arrastra fotos de vehículos (JPG/PNG)
4. El sistema procesa con YOLO + OCR + Random Forest en tiempo real
5. Ve a **Visualizaciones** para ver los 3 gráficos con análisis
6. Ve a **Resultados** para la tabla de detecciones

---

## Endpoints del API

| Método | Ruta            | Descripción                              |
|--------|-----------------|------------------------------------------|
| GET    | /               | Estado del API                           |
| POST   | /upload         | Subir una imagen y analizar              |
| POST   | /upload-batch   | Subir múltiples imágenes                 |
| GET    | /results        | Obtener historial de detecciones         |
| GET    | /stats          | Estadísticas para las visualizaciones    |
| DELETE | /results/clear  | Limpiar historial (útil en demos)        |

Documentación interactiva: http://localhost:8000/docs

---

## Visualizaciones incluidas

| # | Tipo    | Qué muestra                                          | Análisis |
|---|---------|------------------------------------------------------|---------|
| 1 | Barras  | Distribución de tipos de vehículo (YOLO)             | Capacidad de clasificación multi-clase |
| 2 | Pie + Métricas | Placas válidas vs inválidas + matriz de confusión RF | Evaluación del clasificador secundario |
| 3 | Scatter | Confianza OCR vs confianza YOLO por detección        | Correlación calidad imagen - lectura placa |

---

## Modelo Random Forest — Detalles técnicos

**Features de entrada (6):**
- `text_length` — longitud del texto OCR
- `num_letters` — cantidad de letras
- `num_digits` — cantidad de dígitos
- `ocr_confidence` — confianza del motor OCR (0-1)
- `bbox_ratio` — ancho/alto del bounding box
- `matches_pattern` — cumple regex colombiano (0/1)

**Hiperparámetros:**
- `n_estimators=100`
- `max_depth=5`
- `class_weight="balanced"`
- `random_state=42`

**Patrón regex Colombia:**
- Moto actual: `^[A-Z]{3}\d{2}[A-Z]$` → ej: ABC12F
- Moto anterior: `^[A-Z]{3}\d{2}$` → ej: ABC12
- Carro: `^[A-Z]{3}\d{3}$` → ej: ABC123

---

## Para la sustentación

Puntos clave a explicar:
1. **YOLOv8**: CNN pre-entrenada en COCO, fine-tuned para detección de vehículos → clasificación multi-clase
2. **EasyOCR**: red CRNN (CNN + RNN) para reconocimiento de secuencias de caracteres
3. **Random Forest**: ensemble de árboles de decisión, interpretable, evita overfitting
4. **Pipeline**: Imagen → YOLO (clasifica) → recorta ROI → OCR (lee texto) → RF (valida placa)
5. **Dataset**: Placas Colombia (ITM/Roboflow) + Motorcycle License Plate Detection (957 imgs)
