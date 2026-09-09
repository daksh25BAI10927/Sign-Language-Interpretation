/* ============================================================
   SIGNAL — frontend logic
   Talks to the FastAPI sign-language-interpreter backend over
   REST + WebSocket. See the client layer at the bottom of this
   file (checkHealth / getStatus / startInterpreter / stopInterpreter
   / connectStatusSocket) — everything else is UI wiring.
   ============================================================ */

(() => {
  "use strict";

  const SENTINELS = new Set(["No sign detected", "Unknown", "Low confidence"]);
  const HEALTH_POLL_MS = 5000;
  const MAX_RECONNECT_DELAY_MS = 15000;
  const MAX_LOG_ROWS = 300;

  /* ---------------- config ---------------- */

  let apiBase = "http://127.0.0.1:8000";

  function wsUrlFromApiBase(base) {
    return base.replace(/^http/i, "ws").replace(/\/+$/, "") + "/interpreter/ws";
  }

  /* ---------------- backend client layer ---------------- */

  async function checkHealth() {
    const res = await fetch(`${apiBase}/`, { cache: "no-store" });
    if (!res.ok) throw new Error(`Health check failed (${res.status})`);
    return res.json();
  }

  async function getStatus() {
    const res = await fetch(`${apiBase}/interpreter/status`, { cache: "no-store" });
    if (!res.ok) throw new Error(`Status request failed (${res.status})`);
    return res.json();
  }

  async function startInterpreter() {
    const res = await fetch(`${apiBase}/interpreter/start`, { method: "POST" });
    const body = await safeJson(res);
    if (!res.ok) throw new HttpError(res.status, body);
    return body;
  }

  async function stopInterpreter() {
    const res = await fetch(`${apiBase}/interpreter/stop`, { method: "POST" });
    const body = await safeJson(res);
    if (!res.ok) throw new HttpError(res.status, body);
    return body;
  }

  let socket = null;
  let reconnectAttempt = 0;
  let reconnectTimer = null;
  let intentionalClose = false;
  let socketGeneration = 0;

  function connectStatusSocket(onStatus, onError, onClose) {
    const generation = ++socketGeneration;
    intentionalClose = false;
    setWsState("connecting");
    const nextSocket = new WebSocket(wsUrlFromApiBase(apiBase));
    socket = nextSocket;

    nextSocket.onopen = () => {
      if (generation !== socketGeneration) return;
      reconnectAttempt = 0;
      setWsState("open");
    };

    nextSocket.onmessage = (event) => {
      if (generation !== socketGeneration) return;
      try {
        const status = JSON.parse(event.data);
        onStatus(status);
      } catch (err) {
        console.error("Malformed status payload", err);
      }
    };

    nextSocket.onerror = () => {
      if (generation !== socketGeneration) return;
      onError();
    };

    nextSocket.onclose = () => {
      if (generation !== socketGeneration) return;
      setWsState("closed");
      onClose();
      if (!intentionalClose) scheduleReconnect(onStatus, onError, onClose);
    };
  }

  function scheduleReconnect(onStatus, onError, onClose) {
    reconnectAttempt += 1;
    const delay = Math.min(1000 * 2 ** reconnectAttempt, MAX_RECONNECT_DELAY_MS);
    setWsState("reconnecting");
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(() => connectStatusSocket(onStatus, onError, onClose), delay);
  }

  function closeStatusSocket() {
    intentionalClose = true;
    socketGeneration += 1;
    clearTimeout(reconnectTimer);
    if (socket) {
      socket.close();
      socket = null;
    }
  }

  class HttpError extends Error {
    constructor(status, body) {
      super((body && body.detail) || (body && body.message) || `Request failed (${status})`);
      this.status = status;
      this.body = body;
    }
  }

  async function safeJson(res) {
    try { return await res.json(); } catch { return null; }
  }

  /* ---------------- DOM refs ---------------- */

  const el = (id) => document.getElementById(id);

  const apiLed         = document.querySelector('[data-led="api"]');
  const wsLed          = document.querySelector('[data-led="ws"]');
  const interpreterLed = document.querySelector('[data-led="interpreter"]');
  const cameraLed      = document.querySelector('[data-led="camera"]');
  const handLed        = document.querySelector('[data-led="hand"]');

  const apiText  = el("api-text");
  const wsText   = el("ws-text");

  const settingsBtn   = el("settings-btn");
  const settingsPanel = el("settings-panel");
  const apiBaseInput  = el("api-base-input");
  const settingsApply = el("settings-apply");

  const video          = el("preview-video");
  const monitorEmpty   = el("monitor-empty");
  const enableCameraBtn= el("enable-camera");
  const captionLabel   = el("caption-label");
  const captionConf    = el("caption-confidence");

  const startBtn = el("start-btn");
  const stopBtn  = el("stop-btn");
  const heroStartBtn = el("hero-start-btn");
  const interpreterStateText = el("interpreter-state-text");

  const errorBanner = el("error-banner");
  const errorText   = el("error-text");
  const retryBtn    = el("retry-btn");

  const interpreterValue = el("interpreter-value");
  const cameraValue      = el("camera-value");
  const handValue        = el("hand-value");
  const handSubmeta      = el("hand-submeta");
  const confidenceFill   = el("confidence-fill");
  const confidenceValue  = el("confidence-value");
  const fpsValue         = el("fps-value");

  const logBody  = el("log-body");
  const logEmpty = el("log-empty");
  const exportLogBtn = el("export-log");

  const aboutToggle = el("about-toggle");
  const aboutPanel  = el("about-panel");

  const mobileMenuBtn = el("mobile-menu-btn");
  const navLinks      = el("nav-links");

  /* ---------------- UI state helpers ---------------- */

  function setLed(node, color) {
    if (node) node.setAttribute("data-on", color);
  }

  function setApiState(state) {
    apiText.textContent = state;
    setLed(apiLed, state === "online" ? "green" : state === "checking" ? "amber" : "red");
  }

  function setWsState(state) {
    wsText.textContent = state;
    setLed(wsLed, state === "open" ? "green" : state === "connecting" ? "amber" : state === "reconnecting" ? "amber" : "red");
  }

  let interpreterPending = false; // true while a start/stop request is in flight

  function renderInterpreterState(running, pendingLabel) {
    let label = pendingLabel || (running ? "running" : "stopped");
    interpreterValue.textContent = label;
    interpreterStateText.textContent = label[0].toUpperCase() + label.slice(1);

    let color = "grey";
    if (pendingLabel) color = "amber";
    else if (running) color = "red";
    setLed(interpreterLed, color);

    startBtn.disabled = interpreterPending || running;
    stopBtn.disabled  = interpreterPending || !running;
    enableCameraBtn.disabled = interpreterPending || running;

    if (running) {
      monitorEmpty.querySelector("p").textContent =
        "Browser preview is paused while the interpreter uses this camera.";
    } else if (!mediaStream) {
      monitorEmpty.querySelector("p").textContent = "No camera preview active";
    }
  }

  function renderCameraState(connected) {
    const label = connected === true ? "connected" : connected === false ? "disconnected" : "unknown";
    cameraValue.textContent = label;
    setLed(cameraLed, connected === true ? "green" : connected === false ? "red" : "grey");
  }

  function renderHandState(handDetected, handCount, handedness) {
    handValue.textContent = handDetected ? "hand detected" : "no hand detected";
    setLed(handLed, handDetected ? "cyan" : "grey");
    handSubmeta.textContent = handDetected ? `count ${handCount ?? "—"} · ${handedness || "Unknown"}` : "";
  }

  function renderPrediction(prediction, confidence) {
    const label = prediction || "—";
    captionLabel.textContent = label;
    const pct = Math.round((confidence || 0) * 100);
    captionConf.textContent = prediction ? `${pct}%` : "";
    confidenceValue.textContent = `${pct}%`;
    confidenceFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
    confidenceFill.style.background =
      pct >= 70 ? "var(--cyan)" : pct >= 40 ? "var(--amber)" : "var(--grey)";
  }

  function renderFps(fps) {
    fpsValue.textContent = typeof fps === "number" ? `${fps.toFixed(1)} fps` : "— fps";
  }

  function showError(message) {
    errorText.textContent = message;
    errorBanner.hidden = false;
  }
  function hideError() {
    errorBanner.hidden = true;
  }

  function resetPredictionDisplay() {
    captionLabel.textContent = "Interpreter is not running";
    captionConf.textContent = "";
    confidenceValue.textContent = "0%";
    confidenceFill.style.width = "0%";
    fpsValue.textContent = "— fps";
    renderHandState(false, 0, null);
  }

  /* ---------------- prediction log ---------------- */

  const history = [];
  let lastLoggedLabel = null;

  function maybeLogPrediction(status) {
    const label = status.prediction;
    if (!label || label === lastLoggedLabel) return;
    lastLoggedLabel = label;

    const entry = {
      time: new Date(),
      label,
      confidence: status.confidence || 0,
      handedness: status.handedness || "Unknown",
      sentinel: SENTINELS.has(label),
    };
    history.unshift(entry);
    if (history.length > MAX_LOG_ROWS) history.pop();
    renderLog();
  }

  function renderLog() {
    if (history.length === 0) {
      logEmpty.hidden = false;
      return;
    }
    logEmpty.hidden = true;
    logBody.innerHTML = history
      .map((e) => {
        const time = e.time.toLocaleTimeString([], { hour12: false });
        const pct = Math.round(e.confidence * 100);
        const labelClass = e.sentinel ? "log__label log__label--sentinel" : "log__label";
        return `<div class="log__row">
          <span class="log__time">${time}</span>
          <span class="${labelClass}">${escapeHtml(e.label)}</span>
          <span class="log__confidence">${e.sentinel ? "—" : pct + "%"}</span>
          <span class="log__hand">${escapeHtml(e.handedness)}</span>
        </div>`;
      })
      .join("");
  }

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function exportHistoryCsv() {
    const rows = [["time", "prediction", "confidence", "handedness"]];
    [...history].reverse().forEach((e) => {
      rows.push([e.time.toISOString(), e.label, e.confidence.toFixed(2), e.handedness]);
    });
    const csv = rows.map((r) => r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `signal-prediction-log-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  /* ---------------- local camera preview ---------------- */

  let mediaStream = null;

  async function enableCameraPreview() {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = mediaStream;
      monitorEmpty.hidden = true;
    } catch (err) {
      monitorEmpty.querySelector("p").textContent =
        err.name === "NotAllowedError"
          ? "Camera access was denied. Allow it in your browser's site settings, then try again."
          : "Couldn't access a camera on this device.";
    }
  }

  function teardownCameraPreview() {
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
  }

  /* ---------------- status handling ---------------- */

  function handleStatus(status) {
    setApiState("online");
    hideError();

    renderInterpreterState(status.running, interpreterPending ? interpreterStateText.textContent.toLowerCase() : null);
    renderCameraState(status.camera_connected);
    renderHandState(status.hand_detected, status.hand_count, status.handedness);
    renderPrediction(status.prediction, status.confidence);
    renderFps(status.fps);

    if (status.error) showError(status.error);
    if (!status.running) {
      // Backend resets on stop — mirror that instead of showing a stale label.
      if (!status.prediction) resetPredictionDisplay();
    }

    maybeLogPrediction(status);
  }

  /* ---------------- transport actions ---------------- */

  async function handleStart() {
    interpreterPending = true;
    hideError();
    renderInterpreterState(false, "starting");
    try {
      // Desktop browsers and OpenCV commonly cannot share one webcam. Release
      // the browser stream before asking the backend to open the device.
      if (mediaStream) {
        teardownCameraPreview();
        monitorEmpty.hidden = false;
        await new Promise((resolve) => setTimeout(resolve, 250));
      }
      const result = await startInterpreter();
      if (!result.success) showError(result.message);
    } catch (err) {
      showError(err.message || "Couldn't start the interpreter.");
    } finally {
      interpreterPending = false;
      refreshStatusOnce();
    }
  }

  async function handleStop() {
    interpreterPending = true;
    renderInterpreterState(true, "stopping");
    try {
      const result = await stopInterpreter();
      if (!result.success) showError(result.message);
      lastLoggedLabel = null;
      resetPredictionDisplay();
    } catch (err) {
      showError(err.message || "Couldn't stop the interpreter.");
    } finally {
      interpreterPending = false;
      refreshStatusOnce();
    }
  }

  async function refreshStatusOnce() {
    try {
      const status = await getStatus();
      handleStatus(status);
    } catch {
      /* WebSocket will pick it back up; health polling reports connectivity. */
    }
  }

  /* ---------------- health polling ---------------- */

  let healthTimer = null;

  async function pollHealth() {
    try {
      await checkHealth();
      setApiState("online");
    } catch {
      setApiState("offline");
    }
  }

  function startHealthPolling() {
    clearInterval(healthTimer);
    pollHealth();
    healthTimer = setInterval(pollHealth, HEALTH_POLL_MS);
  }

  /* ---------------- boot / reconnect wiring ---------------- */

  function bootBackendConnection() {
    startHealthPolling();
    refreshStatusOnce();
    closeStatusSocket();
    connectStatusSocket(
      handleStatus,
      () => {},
      () => {}
    );
  }

  /* ---------------- smooth scroll helper ---------------- */

  function smoothScrollTo(targetId) {
    const target = document.querySelector(targetId);
    if (target) {
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  /* ---------------- event wiring ---------------- */

  // Settings toggle
  settingsBtn.addEventListener("click", () => {
    const isHidden = settingsPanel.hidden;
    settingsPanel.hidden = !isHidden;
    settingsBtn.setAttribute("aria-expanded", String(isHidden));
  });

  settingsApply.addEventListener("click", () => {
    const value = apiBaseInput.value.trim().replace(/\/+$/, "");
    if (!value) return;
    apiBase = value;
    setApiState("checking");
    setWsState("connecting");
    bootBackendConnection();
  });

  // Camera & transport
  enableCameraBtn.addEventListener("click", enableCameraPreview);
  startBtn.addEventListener("click", handleStart);
  stopBtn.addEventListener("click", handleStop);
  retryBtn.addEventListener("click", () => {
    hideError();
    handleStart();
  });
  exportLogBtn.addEventListener("click", exportHistoryCsv);

  // Hero CTA — scrolls to dashboard and starts interpreter
  if (heroStartBtn) {
    heroStartBtn.addEventListener("click", () => {
      smoothScrollTo("#dashboard");
      handleStart();
    });
  }

  // About toggle
  aboutToggle.addEventListener("click", () => {
    const isHidden = aboutPanel.hidden;
    aboutPanel.hidden = !isHidden;
    aboutToggle.setAttribute("aria-expanded", String(isHidden));
  });

  // Mobile nav toggle
  if (mobileMenuBtn && navLinks) {
    mobileMenuBtn.addEventListener("click", () => {
      navLinks.classList.toggle("open");
    });

    // Close mobile nav when a link is clicked
    navLinks.querySelectorAll(".navbar__link").forEach((link) => {
      link.addEventListener("click", () => {
        navLinks.classList.remove("open");
      });
    });
  }

  // Smooth scroll for nav links
  document.querySelectorAll('.navbar__link[href^="#"]').forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      const target = link.getAttribute("href");
      smoothScrollTo(target);
    });
  });

  // Navbar background on scroll
  const navbar = el("navbar");
  if (navbar) {
    window.addEventListener("scroll", () => {
      navbar.classList.toggle("navbar--scrolled", window.scrollY > 20);
    }, { passive: true });
  }

  window.addEventListener("beforeunload", () => {
    teardownCameraPreview();
    closeStatusSocket();
  });

  /* ---------------- init ---------------- */

  apiBaseInput.value = apiBase;
  resetPredictionDisplay();
  renderInterpreterState(false, null);
  bootBackendConnection();

})();
