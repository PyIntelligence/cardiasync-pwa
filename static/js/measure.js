/**
 * CardiaSync — Tela de Medição
 * Preview ao vivo + overlay de guia + qualidade de sinal + contador
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
  const progressArea = document.getElementById("progressArea");
  const progressFill = document.getElementById("progressFill");
  const countdownEl = document.getElementById("countdown");
  const qualityBadge = document.getElementById("qualityBadge");
  const resultBox = document.getElementById("resultBox");
  const errorBox = document.getElementById("errorBox");
  const errorText = document.getElementById("errorText");

  let stream = null;
  let capturing = false;
  let redSignal = [];
  let animationId = null;
  const CAPTURE_SECONDS = 15;
  const TARGET_FPS = 30;

  // ─── Utilitários ───
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

  // ─── Inicializar câmera ───
  async function initCamera() {
    hideError();

    if (!isSecureContext()) {
      showError(
        "Erro ao acessar a câmera. O site precisa estar em HTTPS (ou localhost). " +
        "No Render isso já é padrão. Se estiver testando localmente por IP, use um túnel HTTPS (ngrok / Cloudflare Tunnel)."
      );
      cameraStatus.textContent = "HTTPS necessário";
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showError("Seu navegador não suporta acesso à câmera.");
      return;
    }

    try {
      // Preferência: câmera traseira
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

      // Tentar ligar o flash/torch (nem todos os dispositivos permitem)
      const track = stream.getVideoTracks()[0];
      const capabilities = track.getCapabilities?.() || {};
      if (capabilities.torch) {
        try {
          await track.applyConstraints({ advanced: [{ torch: true }] });
          cameraStatus.textContent = "Câmera + flash ativos";
        } catch {
          cameraStatus.textContent = "Câmera ativa (flash não disponível)";
        }
      } else {
        cameraStatus.textContent = "Câmera ativa";
      }

      // Ajusta canvas
      overlay.width = video.videoWidth || 640;
      overlay.height = video.videoHeight || 480;

      btnStart.disabled = false;
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

  // ─── Loop de preview + análise de qualidade ───
  function startPreviewLoop() {
    const processFrame = () => {
      if (!video.videoWidth) {
        animationId = requestAnimationFrame(processFrame);
        return;
      }

      // Desenha frame no canvas (para análise)
      ctx.drawImage(video, 0, 0, overlay.width, overlay.height);

      // Região central (onde o dedo deve estar)
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

      // Heurística melhorada de “dedo presente”
      const isRedDominant = meanR > meanG + 12 && meanR > meanB + 12;
      const isDarkEnough = meanR < 195 && meanR > 20;
      // Uniformidade: desvio entre canais não deve ser extremo demais
      const channelSpread = Math.max(meanR, meanG, meanB) - Math.min(meanR, meanG, meanB);
      const uniformEnough = channelSpread < 90;
      const fingerLikely = isRedDominant && isDarkEnough && uniformEnough;

      if (fingerLikely) {
        guideCircle.classList.add("good");
        guideCircle.classList.remove("bad");
        guideText.textContent = "Dedo detectado — mantenha estável";
        cameraStatus.textContent = "Sinal bom — pronto para medir";
      } else if (meanR < 15) {
        guideCircle.classList.remove("good");
        guideCircle.classList.add("bad");
        guideText.textContent = "Muito escuro — ajuste a pressão do dedo";
        cameraStatus.textContent = "Cobertura insuficiente ou dedo muito apertado";
      } else {
        guideCircle.classList.remove("good");
        guideCircle.classList.add("bad");
        guideText.textContent = "Cubra a câmera + flash com o dedo";
        cameraStatus.textContent = "Aguardando posicionamento do dedo";
      }

      // Se estiver capturando, acumula o canal vermelho médio
      if (capturing) {
        redSignal.push(meanR);
      }

      animationId = requestAnimationFrame(processFrame);
    };

    animationId = requestAnimationFrame(processFrame);
  }

  // ─── Iniciar captura ───
  btnStart.addEventListener("click", async () => {
    if (capturing) return;
    hideError();
    resultBox.hidden = true;
    redSignal = [];
    capturing = true;
    btnStart.disabled = true;
    progressArea.hidden = false;

    const startTime = performance.now();
    const durationMs = CAPTURE_SECONDS * 1000;

    const tick = () => {
      const elapsed = performance.now() - startTime;
      const remaining = Math.max(0, CAPTURE_SECONDS - elapsed / 1000);
      const pct = Math.min(100, (elapsed / durationMs) * 100);

      countdownEl.textContent = remaining.toFixed(0);
      progressFill.style.width = pct + "%";

      // Qualidade aproximada durante a captura
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

      if (elapsed < durationMs) {
        requestAnimationFrame(tick);
      } else {
        finishCapture();
      }
    };

    requestAnimationFrame(tick);
  });

  async function finishCapture() {
    capturing = false;
    btnStart.disabled = false;
    progressArea.hidden = true;
    progressFill.style.width = "0%";

    if (redSignal.length < 30) {
      showError("Sinal muito curto. Tente novamente mantendo o dedo firme.");
      return;
    }

    // Envia para o backend
    const payload = {
      red_signal: redSignal,
      fps: TARGET_FPS,
      condition: document.getElementById("condition").value,
      user_name: document.getElementById("userName").value.trim() || "Anônimo",
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

    try {
      cameraStatus.textContent = "Processando…";
      const res = await fetch("/api/measure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!data.ok) {
        showError(data.error || "Erro ao processar medição.");
        return;
      }

      // Mostra resultado
      document.getElementById("resHR").textContent = data.heart_rate ?? "—";
      if (data.estimated_systolic && data.estimated_diastolic) {
        document.getElementById("resBP").textContent =
          `${data.estimated_systolic}/${data.estimated_diastolic}`;
      } else {
        document.getElementById("resBP").textContent = "N/A";
      }
      document.getElementById("resQuality").textContent =
        data.signal_quality != null ? Math.round(data.signal_quality * 100) + "%" : "—";
      document.getElementById("resMsg").textContent = data.message || "";

      // HRV / PRV (PPG) e validação ECG opcional
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
          elEcg.textContent =
            (cmp.message || "Comparação com ECG") +
            (cmp.delta_rmssd_ms != null ? ` · ΔRMSSD ${cmp.delta_rmssd_ms} ms` : "") +
            (cmp.delta_mean_hr_bpm != null ? ` · ΔFC ${cmp.delta_mean_hr_bpm} bpm` : "");
        } else {
          elEcg.textContent = "Sem ECG — validação por eletrocardiograma omitida (modo normal).";
        }
      }

      // PDF
      const btnPdf = document.getElementById("btnPdf");
      if (btnPdf && data.measurement_id) {
        btnPdf.href = `/api/measurement/${data.measurement_id}/pdf`;
        btnPdf.style.display = "inline-flex";
      }

      // Status de envio
      const sendStatus = document.getElementById("sendStatus");
      if (sendStatus) {
        const parts = [];
        if (data.email) parts.push(data.email.ok ? "📧 E-mail enviado" : "📧 E-mail: " + data.email.message);
        if (data.whatsapp) parts.push(data.whatsapp.ok ? "💬 WhatsApp enviado" : "💬 WhatsApp: " + data.whatsapp.message);
        sendStatus.textContent = parts.join(" · ");
      }

      resultBox.hidden = false;
      cameraStatus.textContent = "Medição concluída";
    } catch (err) {
      console.error(err);
      showError("Falha de comunicação com o servidor. Verifique a conexão.");
    }
  }

  // Inicia tudo
  initCamera();

  // Limpeza ao sair
  window.addEventListener("beforeunload", () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (animationId) cancelAnimationFrame(animationId);
  });
})();
