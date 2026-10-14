/* ═══════════════════════════════════════════════════════════════════════════
   PawsAI — app.js  ·  Front-end logic
   ═══════════════════════════════════════════════════════════════════════════ */

(() => {
  "use strict";

  // ── Elements ──────────────────────────────────────────────────────────────
  const dropZone        = document.getElementById("drop-zone");
  const fileInput       = document.getElementById("file-input");
  const dzIdle          = document.getElementById("dz-idle");
  const dzPreview       = document.getElementById("dz-preview");
  const previewImg      = document.getElementById("preview-img");
  const clearBtn        = document.getElementById("clear-btn");
  const predictBtn      = document.getElementById("predict-btn");
  const spinner         = document.getElementById("spinner");
  const btnLabel        = predictBtn.querySelector(".btn-label");
  const resultCard      = document.getElementById("result-card");
  const resultAnimal    = document.getElementById("result-animal");
  const resultLabel     = document.getElementById("result-label");
  const resultConfTxt   = document.getElementById("result-confidence-text");
  const confidenceFill  = document.getElementById("confidence-fill");
  const resultRaw       = document.getElementById("result-raw");
  const tryAgainBtn     = document.getElementById("try-again-btn");
  const errorCard       = document.getElementById("error-card");
  const errorMsg        = document.getElementById("error-msg");
  const errorDismiss    = document.getElementById("error-dismiss-btn");
  const modelBadge      = document.getElementById("model-badge");
  const badgeText       = modelBadge.querySelector(".badge-text");

  let currentFile = null;

  // ── Model status check ────────────────────────────────────────────────────
  async function checkModelStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.model_ready) {
        modelBadge.className = "model-badge model-ready";
        badgeText.textContent = "Model Ready";
      } else {
        modelBadge.className = "model-badge model-missing";
        badgeText.textContent = "No Model";
      }
    } catch {
      modelBadge.className = "model-badge model-missing";
      badgeText.textContent = "Offline";
    }
  }
  checkModelStatus();

  // ── Drag & drop ───────────────────────────────────────────────────────────
  dropZone.addEventListener("click", (e) => {
    // Only trigger file picker if clicking on the idle zone (not preview/clear)
    if (currentFile) return;
    fileInput.click();
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) loadFile(file);
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files[0]) loadFile(fileInput.files[0]);
  });

  // ── Load file ─────────────────────────────────────────────────────────────
  function loadFile(file) {
    if (!file.type.startsWith("image/")) {
      showError("Please upload an image file (JPG, PNG, WebP, etc.).");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError("Image is too large (max 10 MB).");
      return;
    }
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      dzIdle.style.display = "none";
      dzPreview.style.display = "flex";
      dzPreview.style.alignItems = "center";
      predictBtn.disabled = false;
      hideResult();
      hideError();
    };
    reader.readAsDataURL(file);
  }

  // ── Clear ─────────────────────────────────────────────────────────────────
  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetDropZone();
  });

  function resetDropZone() {
    currentFile = null;
    fileInput.value = "";
    previewImg.src = "";
    dzPreview.style.display = "none";
    dzIdle.style.display = "flex";
    predictBtn.disabled = true;
    hideResult();
    hideError();
  }

  // ── Try again ─────────────────────────────────────────────────────────────
  tryAgainBtn.addEventListener("click", resetDropZone);
  errorDismiss.addEventListener("click", hideError);

  // ── Predict ───────────────────────────────────────────────────────────────
  predictBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    setLoading(true);
    hideResult();
    hideError();

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok || data.error) {
        throw new Error(data.error || `Server error (${res.status})`);
      }

      showResult(data);
    } catch (err) {
      showError(err.message || "Prediction failed. Is the Flask server running?");
    } finally {
      setLoading(false);
    }
  });

  // ── UI helpers ────────────────────────────────────────────────────────────
  function setLoading(on) {
    predictBtn.disabled = on;
    spinner.style.display  = on ? "block" : "none";
    btnLabel.textContent   = on ? "Classifying…" : "Classify Image";
  }

  function showResult({ label, confidence, raw_score }) {
    const isCat = label === "cat";
    resultAnimal.textContent = isCat ? "🐱" : "🐶";
    resultLabel.textContent  = label.toUpperCase();
    resultConfTxt.textContent = `Confidence: ${confidence}%`;
    resultRaw.textContent     = `raw sigmoid score: ${raw_score} (0 = cat, 1 = dog)`;

    // Bar position: raw_score is 0..1. 0=cat on left, 1=dog on right.
    const barPct = raw_score * 100;
    confidenceFill.style.width = `${barPct}%`;

    resultCard.style.display = "block";
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function hideResult() {
    resultCard.style.display  = "none";
    confidenceFill.style.width = "0%";
  }

  function showError(msg) {
    errorMsg.textContent       = msg;
    errorCard.style.display    = "block";
  }

  function hideError() {
    errorCard.style.display = "none";
  }

})();
