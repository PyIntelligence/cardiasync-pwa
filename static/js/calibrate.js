/**
 * CardiaSync — Fluxo de Calibração
 * Passo 1: valores do aparelho + ECG opcional → Passo 2: captura câmera → Passo 3: resultado
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

  // Controles opcionais de ECG
  const ecgFileInput = document.getElementById("ecgFileInputCal");
  const ecgRawInput = document.getElementById("ecgRawInputCal");
  const ecgFormatSelect = document.getElementById("ecgFormatSelectCal");
  const ecgPreviewInfo = document.getElementById("ecgPreviewInfoCal");
  const ecgStatusBadge = document.getElementById("ecgStatusBadgeCal");
  const clearEcgBtn = document.getElementById("clearEcgBtnCal");

  let stream = null;
  let capturing = false;
  let redSignal = [];
  let animationId = null;
  const CAPTURE_SECONDS = 15;
  const TARGET_FPS = 30;

  let refSystolic = null;
  let refDiastolic = null;

  function showError(msg) {
    errorBox.hidden = false;
    errorText.textContent = msg;
  }
  function hideError() {
    errorBox.hidden = true;
  }

  function parseEcgInput(rawText, mode = "auto") {
    if (!rawText || !rawText.trim()) return null;

    const values = rawText
      .split(/[\s,;|\t\r\n]+/)
      .map((t) => t.trim())
      .filter((t) => t !== "" && !isNaN(Number(t)))
      .map(Number)
      .filter((n) => isFinite(n) && n >= 0);

    if (values.length < 3) {
      return { error: "Insira pelo menos 3 batimentos/picos válidos de ECG." };
    }

    let detected = mode;
    if (mode === "auto") {
      const strictlyIncreasing = values.every((v, i) => i === 0 || v > values[i - 1]);
      const median = [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
      detected = strictlyIncreasing && median < 250 ? "peaks_sec" : "rr_ms";
    }

    if (detected === "rr_ms") {
      const validRR = values.filter((v) => v >= 300 && v <= 2000);
      if (validRR.length < 3) {
        return { error: "Intervalos RR fora da faixa fisiológica (300–2000 ms)." };
      }
      const meanRR = validRR.reduce((a, b) => a + b, 0) / validRR.length;
      return {
        field: "ecg_rr_ms",
        data: validRR,
        summary: `${validRR.length} intervalos RR (~${Math.round(60000 / meanRR)} bpm)`,
      };
    }

    if (!values.every((v, i) => i === 0 || v > values[i - 1])) {
      return { error: "Tempos de pico R (s) devem estar em ordem crescente." };
    }
    return {
      field: "ecg_peaks_sec",
      data: values,
      summary: `${values.length} picos R (${values[values.length - 1].toFixed(1)}s)`,
    };
  }

  function updateEcgPreview() {
    if (!ecgRawInput) return;
    const parsed = parseEcgInput(ecgRawInput.value, ecgFormatSelect?.value || "auto");
    if (!parsed) {
      ecgPreviewInfo.textContent = "Nenhum dado de ECG carregado.";
      ecgStatusBadge.textContent = "Sem ECG";
      ecgStatusBadge.className = "badge medium";
      clearEcgBtn.hidden = true;
      return;
    }
    if (parsed.error) {
      ecgPreviewInfo.textContent = `⚠️ ${parsed.error}`;
      ecgStatusBadge.textContent = "ECG Inválido";
      ecgStatusBadge.className = "badge bad";
      clearEcgBtn.hidden = false;
      return;
    }
    ecgPreviewInfo.textContent = `✅ ${parsed.summary}`;
    ecgStatusBadge.textContent = "ECG Pronto";
    ecgStatusBadge.className = "badge good";
    clearEcgBtn.hidden = false;
  }

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

  // ─── Passo 1 → 2 ───
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

    const parsedEcg = parseEcgInput(ecgRawInput?.value || "", ecgFormatSelect?.value || "auto");
    if (parsedEcg && parsedEcg.error) {
      showError("Corrija ou limpe o campo de ECG antes de continuar: " + parsedEcg.error);
      return;
    }

    refSystolic = s;
    refDiastolic = d;
    setStep(2);
    await initCamera();
  });

  // ─── Câmera ───
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

  // ─── Captura ───
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

    const parsedEcg = parseEcgInput(ecgRawInput?.value || "", ecgFormatSelect?.value || "auto");
    if (parsedEcg && !parsedEcg.error) {
      payload[parsedEcg.field] = parsedEcg.data;
    }

    try {
      cameraStatus.textContent = "Salvando calibração…";
      const res = await fetch("/api/calibrate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!data.ok) {
        showError(data.error || "Erro ao salvar calibração.");
        return;
      }

      const ecgBadgeRes = document.getElementById("resEcgBadgeCal");
      if (ecgBadgeRes) {
        if (data.ecg_validated) {
          ecgBadgeRes.textContent = "⚡ Validado por ECG";
          ecgBadgeRes.className = "badge good";
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

      if (stream) stream.getTracks().forEach((t) => t.stop());
    } catch (err) {
      console.error(err);
      showError("Falha de comunicação com o servidor.");
    }
  }

  window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (animationId) cancelAnimationFrame(animationId);
  });
})();
