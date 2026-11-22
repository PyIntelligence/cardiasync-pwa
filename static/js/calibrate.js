/**
 * CardiaSync — Fluxo de Calibração
 */
(() => {
  const panel1 = document.getElementById("panelStep1");
  const panel2 = document.getElementById("panelStep2");
  const panel3 = document.getElementById("panelStep3");
  const step1 = document.getElementById("step1");
  const step2 = document.getElementById("step2");
  const step3 = document.getElementById("step3");

  const video = document.getElementById("videoCal");
  const overlay = document.getElementById("overlayCal");
  const ctx = overlay.getContext("2d");
  const cameraStatus = document.getElementById("cameraStatusCal");
  const guideCircle = document.querySelector("#fingerGuideCal .guide-circle");
  const guideText = document.querySelector("#fingerGuideCal .guide-text");
  const btnGo = document.getElementById("btnGoToCamera");
  const btnStart = document.getElementById("btnStartCal");
  const progressArea = document.getElementById("progressAreaCal");
  const progressFill = document.getElementById("progressFillCal");
  const countdownEl = document.getElementById("countdownCal");
  const qualityBadge = document.getElementById("qualityBadgeCal");
  const errorBox = document.getElementById("errorBoxCal");
  const errorText = document.getElementById("errorTextCal");

  const ecgFileInput = document.getElementById("ecgFileInputCal");
  const ecgRawInput = document.getElementById("ecgRawInputCal");
  const ecgFormatSelect = document.getElementById("ecgFormatSelectCal");
  const ecgPreviewInfo = document.getElementById("ecgPreviewInfoCal");
  const ecgStatusBadge = document.getElementById("ecgStatusBadgeCal");
  const clearEcgBtn = document.getElementById("clearEcgBtnCal");
  const ecgWaveformCanvas = document.getElementById("ecgWaveformCanvasCal");
  const btnSampleEcgCal = document.getElementById("btnSampleEcgCal");

  let stream = null;
  let capturing = false;
  let redSignal = [];
  let animationId = null;
  const CAPTURE_SECONDS = 15;
  const TARGET_FPS = 30;

  let refSystolic = null;
  let refDiastolic = null;

  fetch("/health", { cache: "no-store" }).catch(() => {});

  function showError(msg) {
    errorBox.hidden = false;
    errorText.textContent = msg;
  }
  function hideError() {
    errorBox.hidden = true;
  }

  async function fetchJsonWithRetry(url, options = {}, { retries = 2, timeoutMs = 25000, onRetry = null } = {}) {
    let lastErr = null;
    for (let attempt = 0; attempt <= retries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);
      try {
        const res = await fetch(url, {
          ...options,
          signal: controller.signal,
          cache: "no-store",
        });
        clearTimeout(timer);

        const rawText = await res.text();
        let data = null;
        try {
          data = rawText ? JSON.parse(rawText) : {};
        } catch {
          if (res.status >= 502 && res.status <= 504 && attempt < retries) {
            if (onRetry) onRetry(attempt + 1, retries);
            await new Promise((r) => setTimeout(r, 1800 * (attempt + 1)));
            continue;
          }
          throw new Error(`Resposta inválida do servidor (HTTP ${res.status}). Tente novamente.`);
        }

        return { res, data };
      } catch (err) {
        clearTimeout(timer);
        lastErr = err;
        if (attempt < retries) {
          if (onRetry) onRetry(attempt + 1, retries);
          await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
          continue;
        }
      }
    }
    if (lastErr && lastErr.name === "AbortError") {
      throw new Error("O servidor demorou para responder. Verifique sua conexão e tente novamente.");
    }
    throw new Error(
      (lastErr && lastErr.message && lastErr.message !== "Load failed" && lastErr.message !== "Failed to fetch")
        ? lastErr.message
        : "Não foi possível conectar ao servidor. Verifique sua conexão com a internet e tente novamente."
    );
  }

  function updateEcgPreview() {
    if (!ecgRawInput || !window.CardiaSyncECG) return;
    const parsed = window.CardiaSyncECG.parseAndAnalyzeEcg(
      ecgRawInput.value,
      ecgFormatSelect?.value || "auto"
    );
    if (!parsed) {
      ecgPreviewInfo.textContent = "Nenhum dado de ECG carregado.";
      ecgStatusBadge.textContent = "Sem ECG";
      ecgStatusBadge.className = "badge medium";
      clearEcgBtn.hidden = true;
      window.CardiaSyncECG.renderEcgWaveformCanvas(ecgWaveformCanvas, null);
      return;
    }
    if (parsed.error) {
      ecgPreviewInfo.textContent = `⚠️ ${parsed.error}`;
      ecgStatusBadge.textContent = "ECG Inválido";
      ecgStatusBadge.className = "badge bad";
      clearEcgBtn.hidden = false;
      window.CardiaSyncECG.renderEcgWaveformCanvas(ecgWaveformCanvas, parsed);
      return;
    }
    ecgPreviewInfo.textContent = `✅ ${parsed.summary}`;
    ecgStatusBadge.textContent = `ECG Pronto (${parsed.integrityScore}%)`;
    ecgStatusBadge.className = "badge good";
    clearEcgBtn.hidden = false;
    window.CardiaSyncECG.renderEcgWaveformCanvas(ecgWaveformCanvas, parsed);
  }

  btnSampleEcgCal?.addEventListener("click", () => {
    if (!window.CardiaSyncECG) return;
    ecgRawInput.value = window.CardiaSyncECG.SAMPLE_NORMAL_RR.join(", ");
    if (ecgFormatSelect) ecgFormatSelect.value = "rr_ms";
    updateEcgPreview();
  });

  ecgFileInput?.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      ecgRawInput.value = e.target.result || "";
      updateEcgPreview();
    };
    reader.readAsText(file);
  });

  ecgRawInput?.addEventListener("input", updateEcgPreview);
  ecgFormatSelect?.addEventListener("change", updateEcgPreview);
  clearEcgBtn?.addEventListener("click", () => {
    ecgRawInput.value = "";
    if (ecgFileInput) ecgFileInput.value = "";
    updateEcgPreview();
  });

  setTimeout(updateEcgPreview, 50);

  function setStep(n) {
    [step1, step2, step3].forEach((el, i) => {
      el.classList.remove("active", "done");
      if (i + 1 < n) el.classList.add("done");
      if (i + 1 === n) el.classList.add("active");
    });
    panel1.hidden = n !== 1;
    panel2.hidden = n !== 2;
    panel3.hidden = n !== 3;
  }

  btnGo.addEventListener("click", async () => {
    hideError();
    const s = parseFloat(document.getElementById("refSystolic").value);
    const d = parseFloat(document.getElementById("refDiastolic").value);

    if (isNaN(s) || isNaN(d) || s < 70 || s > 220 || d < 40 || d > 130) {
      showError("Informe valores válidos de pressão sistólica (70-220) e diastólica (40-130).");
      return;
    }
    if (d >= s) {
      showError("A diastólica deve ser menor que a sistólica.");
      return;
    }

    const parsedEcg = window.CardiaSyncECG
      ? window.CardiaSyncECG.parseAndAnalyzeEcg(ecgRawInput?.value || "", ecgFormatSelect?.value || "auto")
      : null;
    if (parsedEcg && parsedEcg.error) {
      showError("Corrija ou limpe o campo de ECG antes de continuar: " + parsedEcg.error);
      return;
    }

    refSystolic = s;
    refDiastolic = d;
    setStep(2);
    await initCamera();
  });

  async function initCamera() {
    if (!window.isSecureContext && location.hostname !== "localhost") {
      showError("HTTPS necessário para acessar a câmera.");
      return;
    }

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: TARGET_FPS },
        },
      });
      video.srcObject = stream;
      await video.play();

      const track = stream.getVideoTracks()[0];
      const caps = track.getCapabilities?.() || {};
      if (caps.torch) {
        try {
          await track.applyConstraints({ advanced: [{ torch: true }] });
          cameraStatus.textContent = "Câmera + flash ativos";
        } catch {
          cameraStatus.textContent = "Câmera ativa";
        }
      } else {
        cameraStatus.textContent = "Câmera ativa";
      }

      overlay.width = video.videoWidth || 640;
      overlay.height = video.videoHeight || 480;
      btnStart.disabled = false;
      startPreviewLoop();
    } catch (err) {
      console.error(err);
      showError("Erro ao acessar a câmera: " + (err.message || err.name));
    }
  }

  function startPreviewLoop() {
    const loop = () => {
      if (!video.videoWidth) {
        animationId = requestAnimationFrame(loop);
        return;
      }
      ctx.drawImage(video, 0, 0, overlay.width, overlay.height);

      const cx = Math.floor(overlay.width / 2);
      const cy = Math.floor(overlay.height / 2);
      const size = Math.min(80, Math.floor(overlay.width * 0.2));
      const imageData = ctx.getImageData(cx - size / 2, cy - size / 2, size, size);
      const data = imageData.data;

      let sumR = 0, sumG = 0, sumB = 0;
      const pixels = data.length / 4;
      for (let i = 0; i < data.length; i += 4) {
        sumR += data[i];
        sumG += data[i + 1];
        sumB += data[i + 2];
      }
      const meanR = sumR / pixels;
      const meanG = sumG / pixels;
      const meanB = sumB / pixels;

      const fingerLikely = meanR > meanG + 15 && meanR > meanB + 15 && meanR < 180 && meanR > 25;

      if (fingerLikely) {
        guideCircle.classList.add("good");
        guideCircle.classList.remove("bad");
        guideText.textContent = "Dedo detectado — mantenha estável";
      } else {
        guideCircle.classList.remove("good");
        guideCircle.classList.add("bad");
        guideText.textContent = "Cubra a câmera + flash com o dedo";
      }

      if (capturing) redSignal.push(meanR);
      animationId = requestAnimationFrame(loop);
    };
    animationId = requestAnimationFrame(loop);
  }

  btnStart.addEventListener("click", () => {
    if (capturing) return;
    hideError();
    redSignal = [];
    capturing = true;
    btnStart.disabled = true;
    progressArea.hidden = false;

    const start = performance.now();
    const duration = CAPTURE_SECONDS * 1000;

    const tick = () => {
      const elapsed = performance.now() - start;
      const remaining = Math.max(0, CAPTURE_SECONDS - elapsed / 1000);
      progressFill.style.width = Math.min(100, (elapsed / duration) * 100) + "%";
      countdownEl.textContent = remaining.toFixed(0);

      if (redSignal.length > 10) {
        const recent = redSignal.slice(-30);
        const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
        const std = Math.sqrt(recent.reduce((a, b) => a + (b - mean) ** 2, 0) / recent.length);
        let label = "Baixa", cls = "bad";
        if (std > 3 && mean > 30 && mean < 180) { label = "Boa"; cls = "good"; }
        else if (std > 1.5) { label = "Média"; cls = "medium"; }
        qualityBadge.textContent = "Qualidade: " + label;
        qualityBadge.className = "badge " + cls;
      }

      if (elapsed < duration) requestAnimationFrame(tick);
      else finish();
    };
    requestAnimationFrame(tick);
  });

  async function finish() {
    capturing = false;
    btnStart.disabled = false;
    progressArea.hidden = true;

    if (redSignal.length < 30) {
      showError("Sinal insuficiente. Tente novamente.");
      return;
    }

    const payload = {
      ref_systolic: refSystolic,
      ref_diastolic: refDiastolic,
      red_signal: redSignal,
      fps: TARGET_FPS,
      condition: document.getElementById("conditionCal").value,
      user_name: document.getElementById("userNameCal").value.trim() || "Anônimo",
      notes: document.getElementById("notesCal").value || "",
      age: document.getElementById("ageCal")?.value || null,
      sex: document.getElementById("sexCal")?.value || null,
      skin_tone: document.getElementById("skinCal")?.value || null,
    };

    const parsedEcg = window.CardiaSyncECG
      ? window.CardiaSyncECG.parseAndAnalyzeEcg(ecgRawInput?.value || "", ecgFormatSelect?.value || "auto")
      : null;
    if (parsedEcg && !parsedEcg.error) {
      payload[parsedEcg.field] = parsedEcg.data;
    }

    try {
      cameraStatus.textContent = "Salvando calibração…";
      const { res, data } = await fetchJsonWithRetry(
        "/api/calibrate",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
        {
          retries: 2,
          timeoutMs: 25000,
          onRetry: (n, max) => {
            cameraStatus.textContent = `Reconectando ao servidor (${n}/${max})…`;
          },
        }
      );

      if (!res.ok || !data.ok) {
        showError(data?.error || `Erro ao salvar calibração (HTTP ${res.status}).`);
        return;
      }

      const conf = data.bp_confidence || {};
      const ecgBadgeRes = document.getElementById("resEcgBadgeCal");
      if (ecgBadgeRes) {
        if (data.ecg_validated) {
          ecgBadgeRes.textContent = conf.label || "⚡ Validado por ECG";
          ecgBadgeRes.className = "badge " + (conf.badge_class || "good");
        } else {
          ecgBadgeRes.textContent = "Sem ECG (Apenas PPG)";
          ecgBadgeRes.className = "badge medium";
        }
      }

      let msg = data.message || "";
      if (data.model_trained) {
        msg += " ✅ Modelo de regressão linear atualizado automaticamente.";
      }
      document.getElementById("calMsg").textContent = msg;
      document.getElementById("resRef").textContent = `${refSystolic}/${refDiastolic}`;
      document.getElementById("resHRCal").textContent = data.features?.heart_rate ?? "—";
      document.getElementById("resQualityCal").textContent =
        data.features?.signal_quality != null
          ? Math.round(data.features.signal_quality * 100) + "%"
          : "—";

      setStep(3);

      if (window.CardiaSyncECG) {
        const calSafety = document.getElementById("calSafetyContainer");
        if (calSafety) {
          calSafety.innerHTML = window.CardiaSyncECG.buildSafetyHtml(conf, data.variability);
        }
        const calCmpCanvas = document.getElementById("calCompareCanvas");
        if (calCmpCanvas) {
          window.CardiaSyncECG.renderEcgVsPpgComparisonCanvas(calCmpCanvas, data.variability, conf);
        }
      }

      if (stream) stream.getTracks().forEach((t) => t.stop());
    } catch (err) {
      console.error(err);
      showError(err.message || "Falha de comunicação com o servidor.");
    }
  }

  window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (animationId) cancelAnimationFrame(animationId);
  });
})();
