/**
 * CardiaSync — Tela de Medição
 * - Verificação obrigatória de calibração antes de medir
 * - Medição 100% automática ao encostar o dedo na câmera + flash (sem precisar tocar em botão)
 * - Visualização gráfica em tempo real do ECG (P-QRS-T) + comparativo ECG × PPG + Segurança Clínica
 */
(() => {
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const cameraStatus = document.getElementById("cameraStatus");
  const fingerGuide = document.getElementById("fingerGuide");
  const guideCircle = fingerGuide.querySelector(".guide-circle");
  const guideText = fingerGuide.querySelector(".guide-text");
  const btnStart = document.getElementById("btnStart");
  const btnRearmAuto = document.getElementById("btnRearmAuto");
  const autoMeasureToggle = document.getElementById("autoMeasureToggle");
  const progressArea = document.getElementById("progressArea");
  const progressFill = document.getElementById("progressFill");
  const countdownEl = document.getElementById("countdown");
  const qualityBadge = document.getElementById("qualityBadge");
  const resultBox = document.getElementById("resultBox");
  const errorBox = document.getElementById("errorBox");
  const errorText = document.getElementById("errorText");
  const userNameInput = document.getElementById("userName");

  const calibrationBanner = document.getElementById("calibrationBanner");
  const calBannerIcon = document.getElementById("calBannerIcon");
  const calBannerTitle = document.getElementById("calBannerTitle");
  const calBannerText = document.getElementById("calBannerText");
  const btnGoCalibrate = document.getElementById("btnGoCalibrate");

  const ecgFileInput = document.getElementById("ecgFileInput");
  const ecgRawInput = document.getElementById("ecgRawInput");
  const ecgFormatSelect = document.getElementById("ecgFormatSelect");
  const ecgPreviewInfo = document.getElementById("ecgPreviewInfo");
  const ecgStatusBadge = document.getElementById("ecgStatusBadge");
  const clearEcgBtn = document.getElementById("clearEcgBtn");
  const ecgWaveformCanvas = document.getElementById("ecgWaveformCanvas");
  const btnSampleEcgNormal = document.getElementById("btnSampleEcgNormal");
  const btnSampleEcgEctopic = document.getElementById("btnSampleEcgEctopic");

  let stream = null;
  let cameraReady = false;
  let isCalibrated = false;
  let calibrationChecked = false;
  let redSignal = [];
  let animationId = null;

  let autoState = "WAITING_FINGER";
  let fingerStableSince = null;
  let fingerLostSince = null;
  let cooldownLiftSince = null;
  let captureStartTime = 0;

  const STABILIZE_MS = 1500;
  const FINGER_LOST_ABORT_MS = 1200;
  const COOLDOWN_LIFT_MS = 1800;
  const CAPTURE_SECONDS = 15;
  const TARGET_FPS = 30;

  function showError(msg) {
    errorBox.hidden = false;
    errorText.textContent = msg;
  }

  function hideError() {
    errorBox.hidden = true;
  }

  function isSecureContext() {
    return window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1";
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
      throw new Error("O servidor demorou para responder (tempo esgotado). Verifique sua internet e tente novamente.");
    }
    throw new Error(
      (lastErr && lastErr.message && lastErr.message !== "Load failed" && lastErr.message !== "Failed to fetch")
        ? lastErr.message
        : "Não foi possível conectar ao servidor. Verifique sua conexão com a internet e tente novamente."
    );
  }

  async function checkCalibrationStatus() {
    const userName = (userNameInput?.value || "").trim() || "Anônimo";
    try {
      const { res, data } = await fetchJsonWithRetry(
        `/api/calibration-status?user_name=${encodeURIComponent(userName)}`,
        { method: "GET" },
        {
          retries: 2,
          timeoutMs: 15000,
          onRetry: (n, max) => {
            if (calBannerText) {
              calBannerText.textContent = `Acordando servidor… tentativa ${n}/${max}`;
            }
          },
        }
      );

      if (!res.ok || !data.ok) {
        throw new Error(data?.error || `Erro HTTP ${res.status}`);
      }

      calibrationChecked = true;
      isCalibrated = !!data.is_calibrated;

      if (isCalibrated) {
        calibrationBanner.className = "cal-banner ok";
        calBannerIcon.textContent = "✅";
        calBannerTitle.textContent =
          data.model_scope === "personal"
            ? `Calibrado (Modelo Pessoal · ${data.personal_calibrations} calibrações)`
            : `Calibrado (Modelo Global · ${data.global_calibrations} amostras)`;
        calBannerText.textContent =
          "Encoste o dedo sobre a câmera e o flash — a medição iniciará automaticamente.";
        btnGoCalibrate.hidden = true;
        if (cameraReady) btnStart.disabled = false;
      } else {
        calibrationBanner.className = "cal-banner blocked";
        calBannerIcon.textContent = "⚠️";
        calBannerTitle.textContent = "Calibração necessária antes de medir";
        calBannerText.textContent =
          `Você tem ${data.personal_calibrations}/${data.min_personal} calibrações pessoais ` +
          `(e ${data.global_calibrations}/${data.min_global} globais). ` +
          `Faça a calibração com aparelho de pressão primeiro.`;
        btnGoCalibrate.hidden = false;
        btnStart.disabled = true;
        resetAutoFlow("WAITING_FINGER");
      }
    } catch (err) {
      calibrationChecked = false;
      isCalibrated = false;
      calibrationBanner.className = "cal-banner blocked";
      calBannerIcon.textContent = "🔌";
      calBannerTitle.textContent = "Sem resposta do servidor";
      calBannerText.textContent = err.message;
      btnGoCalibrate.hidden = false;
      btnStart.disabled = true;
    }
  }

  let userNameDebounce = null;
  userNameInput?.addEventListener("input", () => {
    clearTimeout(userNameDebounce);
    userNameDebounce = setTimeout(checkCalibrationStatus, 500);
  });

  function updateEcgPreview() {
    if (!ecgRawInput || !window.CardiaSyncECG) return;
    const parsed = window.CardiaSyncECG.parseAndAnalyzeEcg(
      ecgRawInput.value,
      ecgFormatSelect?.value || "auto"
    );

    if (!parsed) {
      ecgPreviewInfo.textContent = "Nenhum ECG carregado.";
      ecgStatusBadge.textContent = "Sem ECG (Segurança 48%)";
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
    ecgStatusBadge.textContent = `ECG Ativo (Integridade ${parsed.integrityScore}%)`;
    ecgStatusBadge.className = "badge good";
    clearEcgBtn.hidden = false;
    window.CardiaSyncECG.renderEcgWaveformCanvas(ecgWaveformCanvas, parsed);
  }

  btnSampleEcgNormal?.addEventListener("click", () => {
    if (!window.CardiaSyncECG) return;
    ecgRawInput.value = window.CardiaSyncECG.SAMPLE_NORMAL_RR.join(", ");
    if (ecgFormatSelect) ecgFormatSelect.value = "rr_ms";
    updateEcgPreview();
  });

  btnSampleEcgEctopic?.addEventListener("click", () => {
    if (!window.CardiaSyncECG) return;
    ecgRawInput.value = window.CardiaSyncECG.SAMPLE_ECTOPIC_RR.join(", ");
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

  function resetAutoFlow(newState = "WAITING_FINGER") {
    autoState = newState;
    fingerStableSince = null;
    fingerLostSince = null;
    cooldownLiftSince = null;
    redSignal = [];
    progressArea.hidden = true;
    progressFill.style.width = "0%";
    countdownEl.textContent = String(CAPTURE_SECONDS);
    if (isCalibrated && cameraReady) {
      btnStart.disabled = false;
      btnStart.textContent = "Iniciar Manualmente";
    }
  }

  function beginCapture() {
    if (!isCalibrated) {
      showError("Calibração necessária antes de medir. Acesse a aba Calibrar primeiro.");
      return;
    }
    if (autoState === "CAPTURING" || autoState === "PROCESSING") return;

    hideError();
    resultBox.hidden = true;
    redSignal = [];
    fingerLostSince = null;
    autoState = "CAPTURING";
    captureStartTime = performance.now();

    progressArea.hidden = false;
    progressFill.style.width = "0%";
    countdownEl.textContent = String(CAPTURE_SECONDS);
    qualityBadge.textContent = "Qualidade: Analisando…";
    qualityBadge.className = "badge";

    btnStart.disabled = false;
    btnStart.textContent = "Cancelar Captura";
  }

  btnStart.addEventListener("click", () => {
    if (autoState === "CAPTURING") {
      resetAutoFlow("COOLDOWN");
      cameraStatus.textContent = "Captura cancelada — retire e recoloque o dedo";
      return;
    }
    beginCapture();
  });

  btnRearmAuto?.addEventListener("click", () => {
    hideError();
    resultBox.hidden = true;
    resetAutoFlow("WAITING_FINGER");
    cameraStatus.textContent = "Pronto — encoste o dedo para medir automaticamente";
  });

  async function initCamera() {
    hideError();

    if (!isSecureContext()) {
      showError("Erro ao acessar a câmera. O site precisa estar em HTTPS (ou localhost).");
      cameraStatus.textContent = "HTTPS necessário";
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("Seu navegador não suporta acesso à câmera.");
      return;
    }

    try {
      const constraints = {
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: TARGET_FPS },
        },
      };

      stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = stream;
      await video.play();

      const track = stream.getVideoTracks()[0];
      const capabilities = track.getCapabilities?.() || {};
      if (capabilities.torch) {
        try {
          await track.applyConstraints({ advanced: [{ torch: true }] });
          cameraStatus.textContent = "Câmera + flash ativos — encoste o dedo";
        } catch {
          cameraStatus.textContent = "Câmera ativa — encoste o dedo";
        }
      } else {
        cameraStatus.textContent = "Câmera ativa — encoste o dedo";
      }

      overlay.width = video.videoWidth || 640;
      overlay.height = video.videoHeight || 480;
      cameraReady = true;

      if (isCalibrated) {
        btnStart.disabled = false;
      }
      startPreviewLoop();
    } catch (err) {
      console.error(err);
      let msg = "Não foi possível acessar a câmera.";
      if (err.name === "NotAllowedError") {
        msg = "Permissão de câmera negada. Permita o acesso nas configurações do navegador/site.";
      } else if (err.name === "NotFoundError") {
        msg = "Nenhuma câmera encontrada neste dispositivo.";
      } else if (err.name === "NotReadableError") {
        msg = "A câmera está sendo usada por outro aplicativo.";
      }
      showError(msg);
      cameraStatus.textContent = "Erro de câmera";
    }
  }

  function startPreviewLoop() {
    const processFrame = () => {
      if (!video.videoWidth) {
        animationId = requestAnimationFrame(processFrame);
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

      const isRedDominant = meanR > meanG + 12 && meanR > meanB + 12;
      const isDarkEnough = meanR < 195 && meanR > 20;
      const channelSpread = Math.max(meanR, meanG, meanB) - Math.min(meanR, meanG, meanB);
      const uniformEnough = channelSpread < 90;
      const fingerLikely = isRedDominant && isDarkEnough && uniformEnough;

      const now = performance.now();
      const autoEnabled = !autoMeasureToggle || autoMeasureToggle.checked;

      if (!isCalibrated) {
        guideCircle.classList.remove("good", "stabilizing");
        guideCircle.classList.add("bad");
        guideText.textContent = calibrationChecked
          ? "Calibre o aparelho antes de medir"
          : "Verificando calibração…";
        cameraStatus.textContent = calibrationChecked
          ? "Bloqueado: faça a calibração primeiro"
          : "Conectando ao servidor…";
        animationId = requestAnimationFrame(processFrame);
        return;
      }

      if (autoState === "WAITING_FINGER") {
        if (fingerLikely) {
          guideCircle.classList.add("good", "stabilizing");
          guideCircle.classList.remove("bad");
          if (autoEnabled) {
            autoState = "STABILIZING";
            fingerStableSince = now;
            guideText.textContent = "Dedo detectado — estabilizando…";
            cameraStatus.textContent = "Estabilizando sinal…";
          } else {
            guideText.textContent = "Dedo detectado — toque em Iniciar";
            cameraStatus.textContent = "Sinal bom — pronto para medir";
          }
        } else if (meanR < 15) {
          guideCircle.classList.remove("good", "stabilizing");
          guideCircle.classList.add("bad");
          guideText.textContent = "Muito escuro — alivie um pouco o dedo";
          cameraStatus.textContent = "Dedo muito apertado ou pouca luz";
        } else {
          guideCircle.classList.remove("good", "stabilizing");
          guideCircle.classList.add("bad");
          guideText.textContent = "Encoste o dedo na câmera + flash";
          cameraStatus.textContent = "Aguardando dedo na câmera…";
        }
      } else if (autoState === "STABILIZING") {
        if (!fingerLikely) {
          resetAutoFlow("WAITING_FINGER");
        } else {
          const heldMs = now - fingerStableSince;
          const remSec = Math.max(0, (STABILIZE_MS - heldMs) / 1000).toFixed(1);
          guideCircle.classList.add("good", "stabilizing");
          guideText.textContent = `Iniciando automaticamente em ${remSec}s…`;
          cameraStatus.textContent = "Mantenha o dedo imóvel…";
          if (heldMs >= STABILIZE_MS) {
            beginCapture();
          }
        }
      } else if (autoState === "CAPTURING") {
        redSignal.push(meanR);

        const elapsed = now - captureStartTime;
        const durationMs = CAPTURE_SECONDS * 1000;
        const remaining = Math.max(0, CAPTURE_SECONDS - elapsed / 1000);
        const pct = Math.min(100, (elapsed / durationMs) * 100);

        countdownEl.textContent = remaining.toFixed(0);
        progressFill.style.width = pct + "%";

        if (fingerLikely) {
          fingerLostSince = null;
          guideCircle.classList.add("good");
          guideCircle.classList.remove("bad", "stabilizing");
          guideText.textContent = `Medindo automaticamente… (${remaining.toFixed(0)}s)`;
          cameraStatus.textContent = "Capturando pulso — não mova o dedo";
        } else {
          if (!fingerLostSince) fingerLostSince = now;
          guideCircle.classList.remove("good");
          guideCircle.classList.add("bad");
          guideText.textContent = "Dedo saiu da câmera — recoloque agora";
          cameraStatus.textContent = "Sinal interrompido!";

          if (now - fingerLostSince > FINGER_LOST_ABORT_MS) {
            resetAutoFlow("WAITING_FINGER");
            showError("O dedo saiu da câmera durante a leitura. Encoste o dedo novamente para medir automaticamente.");
            animationId = requestAnimationFrame(processFrame);
            return;
          }
        }

        if (redSignal.length > 10) {
          const recent = redSignal.slice(-30);
          const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
          const variance = recent.reduce((a, b) => a + (b - mean) ** 2, 0) / recent.length;
          const std = Math.sqrt(variance);

          let qLabel = "Baixa";
          let qClass = "bad";
          if (std > 3 && mean > 30 && mean < 180) {
            qLabel = "Boa";
            qClass = "good";
          } else if (std > 1.5) {
            qLabel = "Média";
            qClass = "medium";
          }
          qualityBadge.textContent = "Qualidade: " + qLabel;
          qualityBadge.className = "badge " + qClass;
        }

        if (elapsed >= durationMs) {
          finishCapture();
        }
      } else if (autoState === "PROCESSING") {
        guideCircle.classList.remove("stabilizing");
        guideText.textContent = "Calculando pressão e HRV…";
      } else if (autoState === "COOLDOWN") {
        if (fingerLikely) {
          cooldownLiftSince = null;
          guideCircle.classList.add("good");
          guideCircle.classList.remove("bad", "stabilizing");
          guideText.textContent = "Concluído! Retire o dedo para nova medição";
          cameraStatus.textContent = "Retire o dedo para rearmar";
        } else {
          if (!cooldownLiftSince) cooldownLiftSince = now;
          guideCircle.classList.remove("good", "stabilizing");
          guideCircle.classList.add("bad");
          guideText.textContent = "Encoste o dedo para nova medição";
          cameraStatus.textContent = "Aguardando novo posicionamento…";
          if (now - cooldownLiftSince >= COOLDOWN_LIFT_MS) {
            resetAutoFlow("WAITING_FINGER");
          }
        }
      }

      animationId = requestAnimationFrame(processFrame);
    };

    animationId = requestAnimationFrame(processFrame);
  }

  async function finishCapture() {
    autoState = "PROCESSING";
    btnStart.disabled = true;
    progressArea.hidden = true;
    progressFill.style.width = "0%";

    if (redSignal.length < 30) {
      showError("Sinal muito curto. Encoste o dedo firmemente na câmera e no flash.");
      resetAutoFlow("COOLDOWN");
      return;
    }

    const payload = {
      red_signal: redSignal,
      fps: TARGET_FPS,
      condition: document.getElementById("condition").value,
      user_name: (userNameInput?.value || "").trim() || "Anônimo",
      email: document.getElementById("email")?.value?.trim() || null,
      phone: document.getElementById("phone")?.value?.trim() || null,
      feeding: document.getElementById("feeding").value || null,
      position: document.getElementById("position").value || null,
      arm: document.getElementById("arm").value || null,
      pulse_height: document.getElementById("pulseHeight").value || null,
      observation: document.getElementById("observation").value || null,
      send_email: !!document.getElementById("sendEmail")?.checked,
      send_whatsapp: !!document.getElementById("sendWhatsapp")?.checked,
    };

    const parsedEcg = window.CardiaSyncECG
      ? window.CardiaSyncECG.parseAndAnalyzeEcg(ecgRawInput?.value || "", ecgFormatSelect?.value || "auto")
      : null;
    if (parsedEcg && !parsedEcg.error) {
      payload[parsedEcg.field] = parsedEcg.data;
    }

    try {
      cameraStatus.textContent = "Processando medição…";
      const { res, data } = await fetchJsonWithRetry(
        "/api/measure",
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
        if (data?.requires_calibration) {
          isCalibrated = false;
          await checkCalibrationStatus();
        }
        showError(data?.error || `Erro ao processar medição (HTTP ${res.status}).`);
        cameraStatus.textContent = "Medição não concluída";
        resetAutoFlow("COOLDOWN");
        return;
      }

      const conf = data.bp_confidence || {};
      const confBadge = document.getElementById("resConfidenceBadge");
      if (confBadge) {
        confBadge.textContent = conf.label || "Sem ECG — Precisão Reduzida";
        confBadge.className = "badge " + (conf.badge_class || "medium");
      }

      document.getElementById("resHR").textContent = data.heart_rate ?? "—";
      if (data.estimated_systolic && data.estimated_diastolic) {
        const margin = data.sbp_std ? ` (±${Math.round(data.sbp_std)})` : "";
        document.getElementById("resBP").textContent =
          `${data.estimated_systolic}/${data.estimated_diastolic}${margin}`;
      } else {
        document.getElementById("resBP").textContent = "N/A";
      }
      document.getElementById("resQuality").textContent =
        data.signal_quality != null ? Math.round(data.signal_quality * 100) + "%" : "—";
      document.getElementById("resMsg").textContent = data.message || "";

      const prv = (data.variability && data.variability.prv) || {};
      const elRm = document.getElementById("resRmssd");
      const elSd = document.getElementById("resSdnn");
      const elPn = document.getElementById("resPnn50");
      const elNote = document.getElementById("resHrvNote");
      const elEcg = document.getElementById("resEcgNote");
      if (elRm) elRm.textContent = prv.rmssd_ms != null ? prv.rmssd_ms : "—";
      if (elSd) elSd.textContent = prv.sdnn_ms != null ? prv.sdnn_ms : "—";
      if (elPn) elPn.textContent = prv.pnn50_pct != null ? prv.pnn50_pct : "—";
      if (elNote) {
        elNote.textContent = prv.n_intervals
          ? `PRV (PPG): ${prv.n_intervals} intervalos · ${prv.reliable ? "confiável para janela curta" : "poucos batimentos — experimental"}`
          : "PRV indisponível (poucos picos detectados).";
      }
      if (elEcg) {
        if (data.variability && data.variability.ecg_provided) {
          const cmp = data.variability.comparison || {};
          const hrvEcg = data.variability.hrv_ecg || {};
          elEcg.style.color = conf.level === "high" ? "#4ade80" : "#fbbf24";
          elEcg.textContent =
            `ECG Referência: RMSSD ${hrvEcg.rmssd_ms ?? "—"} ms · SDNN ${hrvEcg.sdnn_ms ?? "—"} ms` +
            (cmp.delta_rmssd_ms != null ? ` · ΔRMSSD ${cmp.delta_rmssd_ms} ms` : "") +
            (cmp.delta_mean_hr_bpm != null ? ` · ΔFC ${cmp.delta_mean_hr_bpm} bpm` : "") +
            (conf.warning ? ` — ⚠️ ${conf.warning}` : "");
        } else {
          elEcg.style.color = "#fbbf24";
          elEcg.textContent =
            conf.warning || "Sem ECG — pressão calculada apenas via câmera (menor segurança clínica).";
        }
      }

      resultBox.hidden = false;
      if (window.CardiaSyncECG) {
        const safetyContainer = document.getElementById("ecgSafetyContainer");
        if (safetyContainer) {
          safetyContainer.innerHTML = window.CardiaSyncECG.buildSafetyHtml(conf, data.variability);
        }
        const compareCanvas = document.getElementById("ecgPpgCompareCanvas");
        if (compareCanvas) {
          window.CardiaSyncECG.renderEcgVsPpgComparisonCanvas(compareCanvas, data.variability, conf);
        }
      }

      const btnPdf = document.getElementById("btnPdf");
      if (btnPdf && data.measurement_id) {
        btnPdf.href = `/api/measurement/${data.measurement_id}/pdf`;
        btnPdf.style.display = "inline-flex";
      }

      const sendStatus = document.getElementById("sendStatus");
      if (sendStatus) {
        const parts = [];
        if (data.email) parts.push(data.email.ok ? "📧 E-mail enviado" : "📧 E-mail: " + data.email.message);
        if (data.whatsapp) parts.push(data.whatsapp.ok ? "💬 WhatsApp enviado" : "💬 WhatsApp: " + data.whatsapp.message);
        sendStatus.textContent = parts.join(" · ");
      }

      cameraStatus.textContent = "Medição concluída — retire o dedo";
      resetAutoFlow("COOLDOWN");
    } catch (err) {
      console.error(err);
      showError(err.message || "Falha de comunicação com o servidor. Verifique a conexão.");
      cameraStatus.textContent = "Falha de envio — retire e recoloque o dedo";
      resetAutoFlow("COOLDOWN");
    }
  }

  checkCalibrationStatus();
  initCamera();

  window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (animationId) cancelAnimationFrame(animationId);
  });
})();
