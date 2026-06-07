// frontend/src/App.jsx
// Dashboard principal — Clasificación de Placas de Motos Colombia

import { useState, useEffect, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  ScatterChart, Scatter, ZAxis,
} from "recharts";

const API = "http://localhost:8000";

// ── Colores del tema ───────────────────────────────────────────────────
const COLORS = {
  moto: "#F59E0B",
  car: "#3B82F6",
  bus: "#8B5CF6",
  truck: "#EF4444",
  unknown: "#6B7280",
  valid: "#10B981",
  invalid: "#EF4444",
};

const VEHICLE_LABELS = {
  motorcycle: "Motocicleta",
  car: "Automóvil",
  bus: "Bus",
  truck: "Camión",
  unknown: "Desconocido",
};

// ── Hook para datos del API ─────────────────────────────────────────────
function useStats() {
  const [stats, setStats] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [statsRes, resultsRes] = await Promise.all([
        fetch(`${API}/stats`),
        fetch(`${API}/results?limit=50`),
      ]);
      if (statsRes.ok) setStats(await statsRes.json());
      if (resultsRes.ok) {
        const data = await resultsRes.json();
        setResults(data.results || []);
      }
    } catch (e) {
      console.error("Error cargando datos:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 5000); // Auto-refresh cada 5s
    return () => clearInterval(interval);
  }, [refresh]);

  return { stats, results, loading, refresh };
}

// ── Componente: Upload de imágenes ──────────────────────────────────────
function UploadZone({ onUpload }) {
  const [uploading, setUploading] = useState(false);
  const [lastResult, setLastResult] = useState(null);
  const [error, setError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles) => {
    if (!acceptedFiles.length) return;
    setUploading(true);
    setError(null);
    setLastResult(null);

    try {
      const formData = new FormData();
      // Si son múltiples archivos, usar batch
      if (acceptedFiles.length > 1) {
        acceptedFiles.forEach((file) => formData.append("files", file));
        const res = await fetch(`${API}/upload-batch`, { method: "POST", body: formData });
        const data = await res.json();
        setLastResult({ batch: true, ...data });
      } else {
        formData.append("file", acceptedFiles[0]);
        const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
        const data = await res.json();
        setLastResult(data);
      }
      onUpload();
    } catch (e) {
      setError("Error conectando con la API. ¿Está corriendo el backend?");
    } finally {
      setUploading(false);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".jpg", ".jpeg", ".png", ".webp"] },
    multiple: true,
  });

  return (
    <div className="upload-section">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? "active" : ""} ${uploading ? "uploading" : ""}`}
      >
        <input {...getInputProps()} />
        <div className="dropzone-content">
          {uploading ? (
            <>
              <div className="spinner" />
              <p>Analizando imagen con YOLO + OCR...</p>
            </>
          ) : (
            <>
              <span className="upload-icon">📷</span>
              <p className="upload-title">
                {isDragActive ? "Suelta la imagen aquí" : "Arrastra imágenes de vehículos"}
              </p>
              <p className="upload-sub">o haz clic para seleccionar · JPG, PNG, WEBP</p>
              <p className="upload-sub">Puedes subir varias imágenes a la vez</p>
            </>
          )}
        </div>
      </div>

      {error && <div className="error-banner">⚠️ {error}</div>}

      {lastResult && !lastResult.batch && lastResult.detections?.length > 0 && (
        <div className="detection-result">
          <h4>Resultado — {lastResult.filename}</h4>
          {lastResult.detections.map((d, i) => (
            <div key={i} className="detection-card">
              <span className={`vehicle-badge ${d.vehicle_type}`}>
                {VEHICLE_LABELS[d.vehicle_type] || d.vehicle_type}
              </span>
              <span className="plate-text">{d.plate_text || "Sin placa"}</span>
              <span className={`validity-badge ${d.rf_valid ? "valid" : "invalid"}`}>
                {d.rf_valid ? "✅ Placa válida" : "❌ No válida"}
              </span>
              <span className="confidence">
                YOLO: {(d.vehicle_confidence * 100).toFixed(0)}% ·
                OCR: {(d.ocr_confidence * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {lastResult?.batch && (
        <div className="detection-result">
          <h4>✅ Lote procesado — {lastResult.batch_size} imágenes</h4>
        </div>
      )}

      {lastResult && !lastResult.batch && lastResult.detections_count === 0 && (
        <div className="detection-result">
          <p>No se detectaron vehículos en la imagen.</p>
        </div>
      )}
    </div>
  );
}

// ── Visualización 1: Distribución de tipos de vehículo (Barras) ─────────
function VehicleDistributionChart({ stats }) {
  if (!stats?.vehicle_distribution) return null;

  const data = Object.entries(stats.vehicle_distribution).map(([type, count]) => ({
    name: VEHICLE_LABELS[type] || type,
    count,
    fill: COLORS[type] || COLORS.unknown,
  }));

  return (
    <div className="chart-card">
      <div className="chart-header">
        <h3>📊 Viz 1 — Distribución de Vehículos Detectados</h3>
        <p className="chart-analysis">
          <strong>Análisis:</strong> El gráfico muestra cuántos vehículos de cada tipo ha
          clasificado el modelo YOLOv8. Las motocicletas aparecen diferenciadas de los
          automóviles, validando la capacidad de clasificación del modelo CNN subyacente.
          Un dominio de motos confirma que el dataset está balanceado hacia el caso de uso principal.
        </p>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="name" tick={{ fill: "#CBD5E1", fontSize: 12 }} />
          <YAxis tick={{ fill: "#CBD5E1", fontSize: 12 }} />
          <Tooltip
            contentStyle={{ background: "#1E293B", border: "1px solid #334155", borderRadius: 8 }}
            labelStyle={{ color: "#F8FAFC" }}
          />
          <Bar dataKey="count" name="Vehículos">
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Visualización 2: Placas válidas vs inválidas (Pie) + Métricas RF ────
function PlateValidityChart({ stats }) {
  if (!stats?.plate_validity) return null;

  const pieData = [
    { name: "Placas Válidas", value: stats.plate_validity.valid, color: COLORS.valid },
    { name: "No Válidas / Ruido", value: stats.plate_validity.invalid, color: COLORS.invalid },
  ];

  const metrics = stats.rf_metrics || {};
  const cm = metrics.confusion_matrix || [[0, 0], [0, 0]];

  return (
    <div className="chart-card">
      <div className="chart-header">
        <h3>🔵 Viz 2 — Clasificador Random Forest: Validación de Placas</h3>
        <p className="chart-analysis">
          <strong>Análisis:</strong> El clasificador Random Forest usa 6 features
          (longitud del texto, letras, dígitos, confianza OCR, ratio del bounding box,
          cumplimiento del patrón colombiano) para distinguir placas reales de ruido OCR.
          {metrics.accuracy && (
            <> La exactitud alcanzada es del <strong>{(metrics.accuracy * 100).toFixed(1)}%</strong>,
              con F1-score de <strong>{(metrics.f1_score * 100).toFixed(1)}%</strong>.</>
          )}
        </p>
      </div>

      <div className="viz2-grid">
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={pieData} cx="50%" cy="50%" outerRadius={80} dataKey="value" label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`} labelLine={false} fontSize={11}>
              {pieData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
            </Pie>
            <Tooltip contentStyle={{ background: "#1E293B", border: "1px solid #334155", borderRadius: 8 }} />
          </PieChart>
        </ResponsiveContainer>

        <div className="metrics-panel">
          <h4>Métricas del modelo</h4>
          {[
            { label: "Accuracy", value: metrics.accuracy, color: "#10B981" },
            { label: "Precision", value: metrics.precision, color: "#3B82F6" },
            { label: "Recall", value: metrics.recall, color: "#F59E0B" },
            { label: "F1-Score", value: metrics.f1_score, color: "#8B5CF6" },
          ].map(({ label, value, color }) => (
            <div key={label} className="metric-row">
              <span className="metric-label">{label}</span>
              <div className="metric-bar-bg">
                <div className="metric-bar-fill" style={{ width: `${(value || 0) * 100}%`, background: color }} />
              </div>
              <span className="metric-value" style={{ color }}>{value ? `${(value * 100).toFixed(1)}%` : "—"}</span>
            </div>
          ))}

          {metrics.feature_importances && (
            <div className="feature-importance">
              <h5>Importancia de features</h5>
              {Object.entries(metrics.feature_importances)
                .sort(([, a], [, b]) => b - a)
                .map(([feat, imp]) => (
                  <div key={feat} className="feat-row">
                    <span>{feat.replace(/_/g, " ")}</span>
                    <div className="feat-bar-bg">
                      <div className="feat-bar-fill" style={{ width: `${imp * 100}%` }} />
                    </div>
                    <span>{(imp * 100).toFixed(1)}%</span>
                  </div>
                ))}
            </div>
          )}
        </div>
      </div>

      {/* Matriz de Confusión */}
      <div className="confusion-matrix">
        <h4>Matriz de Confusión</h4>
        <table>
          <thead>
            <tr><th></th><th>Pred. Inválida</th><th>Pred. Válida</th></tr>
          </thead>
          <tbody>
            <tr><td>Real Inválida</td><td className="cm-tn">{cm[0]?.[0] ?? 0}</td><td className="cm-fp">{cm[0]?.[1] ?? 0}</td></tr>
            <tr><td>Real Válida</td><td className="cm-fn">{cm[1]?.[0] ?? 0}</td><td className="cm-tp">{cm[1]?.[1] ?? 0}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── Visualización 3: Confianza OCR por tipo de vehículo (Scatter) ───────
function OcrConfidenceChart({ results }) {
  if (!results?.length) return null;

  const data = results.map((r) => ({
    vehicle: VEHICLE_LABELS[r.vehicle_type] || r.vehicle_type,
    ocr: Math.round(r.ocr_confidence * 100),
    yolo: Math.round(r.vehicle_confidence * 100),
    plate: r.plate_text || "—",
    valid: r.rf_valid,
    fill: r.rf_valid ? COLORS.valid : COLORS.invalid,
  }));

  return (
    <div className="chart-card">
      <div className="chart-header">
        <h3>🔬 Viz 3 — Confianza OCR vs Confianza YOLO por Detección</h3>
        <p className="chart-analysis">
          <strong>Análisis:</strong> Cada punto representa una detección. El eje X es la
          confianza del modelo YOLO al clasificar el vehículo; el eje Y es la confianza
          del OCR al leer la placa. Los puntos verdes son placas clasificadas como válidas
          por el Random Forest; los rojos son ruido o lecturas incorrectas. Una correlación
          positiva indicaría que imágenes de mejor calidad producen mejores lecturas de placa.
        </p>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="yolo" name="Confianza YOLO" unit="%" tick={{ fill: "#CBD5E1", fontSize: 11 }} label={{ value: "Confianza YOLO (%)", position: "insideBottom", offset: -5, fill: "#94A3B8", fontSize: 11 }} />
          <YAxis dataKey="ocr" name="Confianza OCR" unit="%" tick={{ fill: "#CBD5E1", fontSize: 11 }} label={{ value: "Confianza OCR (%)", angle: -90, position: "insideLeft", fill: "#94A3B8", fontSize: 11 }} />
          <ZAxis range={[60, 60]} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{ background: "#1E293B", border: "1px solid #334155", borderRadius: 8 }}
            content={({ payload }) => {
              if (!payload?.length) return null;
              const d = payload[0]?.payload;
              return (
                <div style={{ padding: 8, fontSize: 12 }}>
                  <p><strong>{d.vehicle}</strong></p>
                  <p>Placa: {d.plate}</p>
                  <p>YOLO: {d.yolo}% · OCR: {d.ocr}%</p>
                  <p style={{ color: d.valid ? COLORS.valid : COLORS.invalid }}>
                    {d.valid ? "✅ Válida" : "❌ No válida"}
                  </p>
                </div>
              );
            }}
          />
          <Scatter data={data} name="Detecciones">
            {data.map((d, i) => (
              <Cell key={i} fill={d.fill} fillOpacity={0.8} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
      <div className="scatter-legend">
        <span style={{ color: COLORS.valid }}>● Placa válida (RF)</span>
        <span style={{ color: COLORS.invalid }}>● No válida / Ruido</span>
      </div>
    </div>
  );
}

// ── Tabla de resultados recientes ───────────────────────────────────────
function ResultsTable({ results }) {
  if (!results?.length) return null;
  const recent = [...results].reverse().slice(0, 15);

  return (
    <div className="table-card">
      <h3>📋 Últimas Detecciones</h3>
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Archivo</th>
              <th>Vehículo</th>
              <th>Placa</th>
              <th>YOLO %</th>
              <th>OCR %</th>
              <th>RF Válida</th>
              <th>Patrón CO</th>
            </tr>
          </thead>
          <tbody>
            {recent.map((r, i) => (
              <tr key={i}>
                <td className="filename">{r.filename?.split("_").slice(1).join("_") || r.filename}</td>
                <td>
                  <span className={`badge-small ${r.vehicle_type}`}>
                    {VEHICLE_LABELS[r.vehicle_type] || r.vehicle_type}
                  </span>
                </td>
                <td className="plate-mono">{r.plate_text || "—"}</td>
                <td>{(r.vehicle_confidence * 100).toFixed(0)}%</td>
                <td>{(r.ocr_confidence * 100).toFixed(0)}%</td>
                <td>{r.rf_valid ? "✅" : "❌"}</td>
                <td>{r.matches_pattern ? "✅" : "❌"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── App principal ───────────────────────────────────────────────────────
export default function App() {
  const { stats, results, loading, refresh } = useStats();
  const [activeTab, setActiveTab] = useState("upload");

  const totalDetections = stats?.total_detections ?? 0;
  const motos = stats?.vehicle_distribution?.motorcycle ?? 0;
  const validPlates = stats?.plate_validity?.valid ?? 0;
  const avgConf = stats?.avg_vehicle_confidence ?? 0;

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="logo">🏍️</div>
          <div>
            <h1>PlacasCO</h1>
            <p>Clasificación e Identificación de Placas — Colombia</p>
          </div>
        </div>
        <div className="header-right">
          <div className="status-dot" />
          <span>API {loading ? "sincronizando…" : "conectada"}</span>
        </div>
      </header>

      {/* KPIs */}
      <div className="kpis">
        {[
          { label: "Total Detecciones", value: totalDetections, icon: "🔍" },
          { label: "Motocicletas", value: motos, icon: "🏍️" },
          { label: "Placas Válidas (RF)", value: validPlates, icon: "✅" },
          { label: "Conf. YOLO Prom.", value: `${(avgConf * 100).toFixed(0)}%`, icon: "📡" },
        ].map(({ label, value, icon }) => (
          <div key={label} className="kpi-card">
            <span className="kpi-icon">{icon}</span>
            <span className="kpi-value">{value}</span>
            <span className="kpi-label">{label}</span>
          </div>
        ))}
      </div>

      {/* Navegación */}
      <nav className="tabs">
        {[
          { id: "upload", label: "📤 Subir Imágenes" },
          { id: "charts", label: "📊 Visualizaciones" },
          { id: "table", label: "📋 Resultados" },
        ].map(({ id, label }) => (
          <button
            key={id}
            className={`tab ${activeTab === id ? "active" : ""}`}
            onClick={() => setActiveTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      {/* Contenido */}
      <main className="main-content">
        {activeTab === "upload" && (
          <div>
            <div className="section-intro">
              <h2>Carga de imágenes en tiempo real</h2>
              <p>
                Sube fotos de vehículos. El sistema ejecutará <strong>YOLOv8</strong> para
                detectar y clasificar el tipo de vehículo, <strong>EasyOCR</strong> para
                leer la placa, y el <strong>clasificador Random Forest</strong> para validar
                si la lectura corresponde a una placa colombiana real.
              </p>
            </div>
            <UploadZone onUpload={refresh} />
          </div>
        )}

        {activeTab === "charts" && (
          <div className="charts-grid">
            {!stats || stats.message ? (
              <div className="empty-state">
                <p>📭 No hay datos aún. Ve a <strong>Subir Imágenes</strong> y carga algunas fotos de vehículos.</p>
              </div>
            ) : (
              <>
                <VehicleDistributionChart stats={stats} />
                <PlateValidityChart stats={stats} />
                <OcrConfidenceChart results={results} />
              </>
            )}
          </div>
        )}

        {activeTab === "table" && (
          results.length ? <ResultsTable results={results} /> : (
            <div className="empty-state">
              <p>📭 No hay detecciones aún. Sube imágenes primero.</p>
            </div>
          )
        )}
      </main>

      {/* Footer con info del modelo */}
      <footer className="app-footer">
        <span>YOLOv8 + EasyOCR + Random Forest</span>
      </footer>
    </div>
  );
}
