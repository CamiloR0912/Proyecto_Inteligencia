/**
 * SmartPark AI — Frontend Logic
 * WebSocket connection, detection rendering, and UI updates.
 */

// ── State ────────────────────────────────────────────────────────────
const state = {
    ws: null,
    plates: [],
    stats: { total: 0, car: 0, motorcycle: 0, bus: 0, truck: 0 },
    reconnectDelay: 1000,
    maxReconnectDelay: 10000,
    lastLivePlate: "",
};

// ── DOM References ───────────────────────────────────────────────────
const dom = {
    status: document.getElementById("connection-status"),
    statusDot: null,
    statusText: null,
    detectionList: document.getElementById("detection-list"),
    emptyState: document.getElementById("empty-state"),
    plateCount: document.getElementById("plate-count"),
    statTotal: document.getElementById("stat-total"),
    lastDetection: document.getElementById("last-detection"),
    lastPlate: document.getElementById("last-plate"),
    lastType: document.getElementById("last-type"),
    videoStream: document.getElementById("video-stream"),
    videoOverlay: document.getElementById("video-overlay"),
    livePlateBar: document.getElementById("live-plate-bar"),
    livePlateText: document.getElementById("live-plate-text"),
    livePlateType: document.getElementById("live-plate-type"),
    _livePlateTimer: null,
};

// Initialize status dot and text references
dom.statusDot = dom.status.querySelector(".status-dot");
dom.statusText = dom.status.querySelector(".status-text");

// ── Vehicle Icons ────────────────────────────────────────────────────
const VEHICLE_ICONS = {
    car: "🚗",
    motorcycle: "🏍️",
    bus: "🚌",
    truck: "🚛",
};

const VEHICLE_LABELS = {
    car: "Carro",
    motorcycle: "Moto",
    bus: "Bus",
    truck: "Camión",
};

// ── WebSocket Connection ─────────────────────────────────────────────
function connectWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    state.ws = new WebSocket(wsUrl);

    state.ws.onopen = () => {
        console.log("✅ WebSocket conectado");
        setConnectionStatus(true);
        state.reconnectDelay = 1000;
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === "live") {
                updateLivePlate(data);
            } else {
                handleDetection(data);
            }
        } catch (e) {
            console.error("Error parsing WebSocket message:", e);
        }
    };

    state.ws.onclose = () => {
        console.log("🔌 WebSocket desconectado. Reconectando...");
        setConnectionStatus(false);
        scheduleReconnect();
    };

    state.ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        state.ws.close();
    };
}

function scheduleReconnect() {
    setTimeout(() => {
        connectWebSocket();
    }, state.reconnectDelay);
    state.reconnectDelay = Math.min(
        state.reconnectDelay * 1.5,
        state.maxReconnectDelay
    );
}

function setConnectionStatus(connected) {
    dom.statusDot.className = connected
        ? "status-dot status-dot--connected"
        : "status-dot status-dot--disconnected";
    dom.statusText.textContent = connected ? "Conectado" : "Desconectado";
}

// ── Detection Handler ────────────────────────────────────────────────
function handleDetection(data) {
    // Add to state
    state.plates.unshift(data);
    if (state.plates.length > 50) state.plates.pop();

    // Update stats
    state.stats.total++;
    const type = data.vehicle_type || "car";
    if (type in state.stats) state.stats[type]++;

    updateStats();
    renderDetection(data);
    updateLastDetection(data);
    hideEmptyState();
}

function updateStats() {
    dom.statTotal.textContent = state.stats.total;
    dom.plateCount.textContent = state.stats.total;
}

// ── Render Detection Card ────────────────────────────────────────────
function renderDetection(data) {
    const card = document.createElement("div");
    card.className = "detection-card new";

    const time = formatTime(data.timestamp);

    card.innerHTML = `
        <div class="detection-card__info">
            <div class="detection-card__plate">${escapeHtml(data.plate)}</div>
        </div>
        <span class="detection-card__time">${time}</span>
    `;

    // Insert at top
    if (dom.detectionList.firstChild) {
        dom.detectionList.insertBefore(card, dom.detectionList.firstChild);
    } else {
        dom.detectionList.appendChild(card);
    }

    // Remove 'new' highlight after animation
    setTimeout(() => card.classList.remove("new"), 2000);

    // Limit displayed cards
    while (dom.detectionList.children.length > 50) {
        dom.detectionList.removeChild(dom.detectionList.lastChild);
    }
}

function updateLastDetection(data) {
    const type = data.vehicle_type || "car";
    dom.lastPlate.textContent = data.plate;
    dom.lastType.textContent = VEHICLE_LABELS[type] || type;
    dom.lastDetection.hidden = false;

    // Re-trigger animation
    dom.lastDetection.style.animation = "none";
    dom.lastDetection.offsetHeight; // force reflow
    dom.lastDetection.style.animation = "";
}

function updateLivePlate(data) {
    const type = data.vehicle_type || "car";

    // Actualizar barra debajo de la cámara
    dom.livePlateBar.hidden = false;
    dom.livePlateText.textContent = data.plate || "---";

    clearTimeout(dom._livePlateTimer);
    dom._livePlateTimer = setTimeout(() => {
        dom.livePlateBar.hidden = true;
    }, 3000);

    // Agregar a la lista de detecciones si la placa cambió
    if (data.plate && data.plate !== state.lastLivePlate) {
        state.lastLivePlate = data.plate;
        const entry = { ...data, timestamp: new Date().toISOString() };
        state.plates.unshift(entry);
        if (state.plates.length > 50) state.plates.pop();
        state.stats.total++;
        if (type in state.stats) state.stats[type]++;
        updateStats();
        renderDetection(entry);
        updateLastDetection(entry);
        hideEmptyState();
    }
}

function hideEmptyState() {
    if (dom.emptyState) {
        dom.emptyState.remove();
    }
}

// ── Utilities ────────────────────────────────────────────────────────
function formatTime(isoString) {
    if (!isoString) return "--:--";
    try {
        const date = new Date(isoString);
        return date.toLocaleTimeString("es-CO", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: false,
        });
    } catch {
        return "--:--";
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ── Video Stream Setup ───────────────────────────────────────────────
function setupVideoStream() {
    dom.videoStream.onload = () => {
        dom.videoOverlay.classList.add("hidden");
    };

    dom.videoStream.onerror = () => {
        dom.videoOverlay.classList.remove("hidden");
        // Retry stream after delay
        setTimeout(() => {
            dom.videoStream.src = "/stream?" + Date.now();
        }, 3000);
    };
}

// ── Load History ─────────────────────────────────────────────────────
async function loadHistory() {
    try {
        const res = await fetch("/api/plates");
        const data = await res.json();
        if (data.plates && data.plates.length > 0) {
            data.plates.forEach((plate) => handleDetection(plate));
        }
    } catch (e) {
        console.log("No se pudo cargar historial:", e);
    }
}

// ── Polling de detección en vivo ─────────────────────────────────────
async function pollDetection() {
    try {
        const res = await fetch("/api/latest-detection");
        const data = await res.json();
        if (data.plate) {
            updateLivePlate(data);
        }
    } catch (e) {
        // servidor no disponible aún, reintentar
    }
    setTimeout(pollDetection, 500);
}

// ── Init ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    setupVideoStream();
    loadHistory();
    connectWebSocket();
    pollDetection();
});
