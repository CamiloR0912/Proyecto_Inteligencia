# Plan de Implementación: App Local ("Un Clic")

Para entregar un proyecto robusto al docente que no dependa de servidores ni consola manual, unificaremos el sistema para que se comporte como una aplicación de escritorio nativa usando un archivo ejecutable `.bat`.

## Cambios Propuestos

### 1. Compilación del Frontend (React/Vite)
Se ejecutará el comando `npm run build` dentro de la carpeta `frontend/`. Esto tomará todo el código de React y lo comprimirá en archivos estáticos puros (HTML, CSS, JS) dentro de una nueva carpeta llamada `frontend/dist/`.

### 2. Unificación en el Backend
**[MODIFY] [main.py](file:///d:/Universidad/Semestres/7mo/Inteligencia/Proyecto_Inteligencia/main.py)**
- Se modificará el código de FastAPI para que, además de procesar las imágenes, actúe como servidor web de la página. 
- Usaremos `StaticFiles` y `FileResponse` para que al entrar a `localhost:8000` se muestre el `index.html` compilado de React.
- Se eliminará la antigua ruta de prueba `/` que devolvía `{"status": "ok"}`.

### 3. Scripts de Instalación (Windows y Mac)
**[NEW] Instalar_PlacasCO.bat (Windows)**
**[NEW] Instalar_PlacasCO.command (Mac)**
- Crear el entorno virtual (`venv`).
- Instalar dependencias usando `pip install -r requirements.txt`.

### 4. Scripts de Inicio (Windows y Mac)
**[NEW] Iniciar_PlacasCO.bat (Windows)**
**[NEW] Iniciar_PlacasCO.command (Mac)**
- Activar el entorno virtual automáticamente.
- Ejecutar el servidor Python en segundo plano (`main.py`).
- Esperar 5 segundos y abrir `http://localhost:8000` en el navegador web predeterminado.

## Verificación
1. Ejecutar el archivo `Iniciar_PlacasCO.bat`.
2. Verificar que el navegador se abra automáticamente y muestre la interfaz de PlacasCO.
3. Verificar que al subir una imagen, la API siga respondiendo correctamente (la conexión Frontend-Backend no debe romperse al estar en el mismo puerto).
