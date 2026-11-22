/**
 * CardiaSync — Visualizador Gráfico de ECG, Tradução Neural PPG-to-ECG,
 * Triagem de Irregularidade de Pulso e Segurança Clínica
 */
window.CardiaSyncECG = (() => {
  const SAMPLE_NORMAL_RR = [
    812, 824, 808, 818, 830, 815, 805, 822, 828, 810,
    802, 819, 825, 814, 809, 821, 816, 807, 823, 812
  ];

  const SAMPLE_ECTOPIC_RR = [
    815, 822, 810, 825, 510, 1110, 818, 808, 824, 812,
    820, 806, 828, 814, 819, 811, 825, 816
  ];

  function parseAndAnalyzeEcg(rawText, mode = "auto") {
    if (!rawText || !rawText.trim()) return null;

    const values = rawText
      .split(/[\s,;|	
]+/)
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

    let rawRR = [];
    let field = "ecg_rr_ms";
    let payloadData = values;

    if (detected === "rr_ms") {
      rawRR = values;
      field = "ecg_rr_ms";
      payloadData = values;
    } else {
      if (!values.every((v, i) => i === 0 || v > values[i - 1])) {
        return { error: "Tempos de pico R (s) devem estar em ordem crescente." };
      }
      field = "ecg_peaks_sec";
      payloadData = values;
      for (let i = 1; i < values.length; i++) {
        rawRR.push(Math.round((values[i] - values[i - 1]) * 1000 * 10) / 10);
      }
    }

    const inRange = rawRR.filter((v) => v >= 300 && v <= 2000);
    if (inRange.length < 3) {
      return { error: "Intervalos RR fora da faixa fisiológica (300–2000 ms)." };
    }

    const sorted = [...inRange].sort((a, b) => a - b);
    const medianRR = sorted[Math.floor(sorted.length / 2)];

    const cleanRR = [];
    const rejectedIndices = [];
    rawRR.forEach((rr, idx) => {
      if (rr < 300 || rr > 2000 || Math.abs(rr - medianRR) / medianRR > 0.20) {
        rejectedIndices.push(idx);
      } else {
        cleanRR.push(rr);
      }
    });

    if (cleanRR.length < 3) {
      return { error: "Excesso de artefatos/ectopia no ECG (menos de 3 batimentos normais)." };
    }

    const meanRR = cleanRR.reduce((a, b) => a + b, 0) / cleanRR.length;
    const meanHR = 60000 / meanRR;
    const sdnn = Math.sqrt(
      cleanRR.reduce((acc, v) => acc + (v - meanRR) ** 2, 0) / Math.max(1, cleanRR.length - 1)
    );
    let sqDiffSum = 0;
    for (let i = 1; i < cleanRR.length; i++) {
      sqDiffSum += (cleanRR[i] - cleanRR[i - 1]) ** 2;
    }
    const rmssd = cleanRR.length > 1 ? Math.sqrt(sqDiffSum / (cleanRR.length - 1)) : 0;

    const validRatio = cleanRR.length / rawRR.length;
    const integrityScore = Math.max(60, Math.min(99, Math.round(validRatio * 98)));

    const ectopicNote =
      rejectedIndices.length > 0
        ? ` · ${rejectedIndices.length} ectopia/artefato filtrado`
        : " · Ritmo sinusal regular";

    return {
      field,
      data: payloadData,
      rawRR,
      cleanRR,
      rejectedIndices,
      medianRR: Math.round(medianRR),
      meanHR: Math.round(meanHR * 10) / 10,
      sdnn: Math.round(sdnn * 10) / 10,
      rmssd: Math.round(rmssd * 10) / 10,
      integrityScore,
      summary: `${cleanRR.length}/${rawRR.length} batimentos válidos (~${Math.round(meanHR)} bpm)${ectopicNote}`,
    };
  }

  function renderEcgWaveformCanvas(canvas, analysis) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(280, Math.floor(rect.width || canvas.width || 460));
    const height = Math.max(150, Math.floor(rect.height || canvas.height || 175));

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#060d1a";
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "rgba(34, 197, 94, 0.08)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 20) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    if (!analysis || analysis.error || !analysis.rawRR || !analysis.rawRR.length) {
      ctx.fillStyle = "#64748b";
      ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Carregue um arquivo ECG ou clique em 'Exemplo ECG' para visualizar o traçado", width / 2, height / 2);
      return;
    }

    const rawRR = analysis.rawRR;
    const rejectedSet = new Set(analysis.rejectedIndices || []);

    const rTimes = [350];
    for (let i = 0; i < rawRR.length; i++) {
      rTimes.push(rTimes[rTimes.length - 1] + rawRR[i]);
    }
    const totalMs = rTimes[rTimes.length - 1] + 400;
    const windowMs = Math.min(totalMs, 11000);
    const startMs = Math.max(0, totalMs - windowMs);

    const baselineY = height * 0.64;
    const ampScale = height * 0.44;

    function pqrstValue(dt, isEctopic) {
      const p = !isEctopic ? 0.13 * Math.exp(-((dt + 165) ** 2) / (2 * 24 ** 2)) : 0;
      const q = -0.14 * Math.exp(-((dt + 26) ** 2) / (2 * 9 ** 2));
      const rHeight = isEctopic ? 0.78 : 1.0;
      const rWidth = isEctopic ? 16 : 11;
      const r = rHeight * Math.exp(-(dt ** 2) / (2 * rWidth ** 2));
      const s = -0.24 * Math.exp(-((dt - 28) ** 2) / (2 * 11 ** 2));
      const t = (isEctopic ? -0.18 : 0.24) * Math.exp(-((dt - 210) ** 2) / (2 * 42 ** 2));
      return p + q + r + s + t;
    }

    const samples = Math.min(900, width * 2);
    ctx.lineWidth = 2;

    for (let i = 1; i <= samples; i++) {
      const t0 = startMs + ((i - 1) / samples) * windowMs;
      const t1 = startMs + (i / samples) * windowMs;

      let nearestIdx = 0;
      let minAbsDt = Infinity;
      for (let k = 0; k < rTimes.length; k++) {
        const d = t1 - rTimes[k];
        if (Math.abs(d) < minAbsDt) {
          minAbsDt = Math.abs(d);
          nearestIdx = k;
        }
      }
      const isBeatRejected =
        (nearestIdx > 0 && rejectedSet.has(nearestIdx - 1)) ||
        (nearestIdx < rawRR.length && rejectedSet.has(nearestIdx) && Math.abs(t1 - rTimes[nearestIdx]) < 140);

      let v0 = 0;
      let v1 = 0;
      for (let k = 0; k < rTimes.length; k++) {
        const beatEctopic = k > 0 && rejectedSet.has(k - 1);
        const dt0 = t0 - rTimes[k];
        const dt1 = t1 - rTimes[k];
        if (Math.abs(dt0) < 350) v0 += pqrstValue(dt0, beatEctopic);
        if (Math.abs(dt1) < 350) v1 += pqrstValue(dt1, beatEctopic);
      }

      const x0 = ((t0 - startMs) / windowMs) * width;
      const y0 = baselineY - v0 * ampScale;
      const x1 = ((t1 - startMs) / windowMs) * width;
      const y1 = baselineY - v1 * ampScale;

      ctx.strokeStyle = isBeatRejected ? "#ef4444" : "#22c55e";
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }

    ctx.font = "10px -apple-system, BlinkMacSystemFont, monospace";
    ctx.textAlign = "center";
    for (let k = 0; k < rTimes.length; k++) {
      const rt = rTimes[k];
      if (rt < startMs || rt > startMs + windowMs) continue;
      const x = ((rt - startMs) / windowMs) * width;
      const isEctopic = k > 0 && rejectedSet.has(k - 1);
      const yPeak = baselineY - (isEctopic ? 0.78 : 1.0) * ampScale;

      ctx.fillStyle = isEctopic ? "#ef4444" : "#4ade80";
      ctx.beginPath();
      ctx.arc(x, yPeak, 3, 0, Math.PI * 2);
      ctx.fill();

      if (k > 0) {
        const rrVal = Math.round(rawRR[k - 1]);
        const prevX = ((rTimes[k - 1] - startMs) / windowMs) * width;
        const midX = (prevX + x) / 2;
        if (midX > 18 && midX < width - 18) {
          ctx.fillStyle = isEctopic ? "#fca5a5" : "#94a3b8";
          ctx.fillText(isEctopic ? `⚠️${rrVal}` : `${rrVal}`, midX, height - 8);
        }
      }
    }

    ctx.textAlign = "left";
    ctx.fillStyle = "#4ade80";
    ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillText(`ECG Derivação Sintética (RR) · FC Média: ${analysis.meanHR} bpm`, 10, 16);

    ctx.textAlign = "right";
    ctx.fillStyle = rejectedSet.size > 0 ? "#fbbf24" : "#38bdf8";
    ctx.fillText(
      `RMSSD: ${analysis.rmssd} ms · SDNN: ${analysis.sdnn} ms`,
      width - 10,
      16
    );
  }

  /**
   * Renderiza o Osciloscópio Duplo de Tradução Neural PPG-to-ECG:
   * - Canal Superior: Traçado P-QRS-T reconstruído pela Rede Neural MLP
   * - Canal Inferior: Pulso Óptico (PPG) sincronizado da câmera
   */
  function renderNeuralEcgPredictionCanvas(canvas, neuralPred) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(290, Math.floor(rect.width || canvas.width || 480));
    const height = Math.max(210, Math.floor(rect.height || canvas.height || 230));

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    // Fundo escuro clínico
    ctx.fillStyle = "#050b16";
    ctx.fillRect(0, 0, width, height);

    // Grade milimetrada dupla (fina e principal)
    ctx.strokeStyle = "rgba(16, 185, 129, 0.06)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 12) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 12) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    ctx.strokeStyle = "rgba(16, 185, 129, 0.14)";
    for (let x = 0; x < width; x += 60) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 60) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const pts = (neuralPred && neuralPred.waveform_points) || [];
    if (pts.length < 4) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Aguardando captura PPG para reconstrução neural de ECG…", width / 2, height / 2);
      return;
    }

    const splitY = Math.floor(height * 0.60);

    // Divisor entre canal superior (ECG Neural) e canal inferior (PPG Câmera)
    ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, splitY);
    ctx.lineTo(width, splitY);
    ctx.stroke();

    const padL = 12;
    const padR = 12;
    const plotW = width - padL - padR;

    const ecgVals = pts.map((p) => Number(p.ecg_mv || 0));
    const ppgVals = pts.map((p) => Number(p.ppg_norm || 0));

    const ecgMin = Math.min(...ecgVals);
    const ecgMax = Math.max(...ecgVals);
    const ecgSpan = Math.max(0.45, ecgMax - ecgMin);
    const ecgTop = 26;
    const ecgBot = splitY - 12;
    const ecgH = ecgBot - ecgTop;

    const ppgMin = Math.min(...ppgVals);
    const ppgMax = Math.max(...ppgVals);
    const ppgSpan = Math.max(0.25, ppgMax - ppgMin);
    const ppgTop = splitY + 22;
    const ppgBot = height - 14;
    const ppgH = ppgBot - ppgTop;

    // 1. Canal Inferior: Onda PPG Óptica da Câmera (com preenchimento suave)
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = padL + (i / Math.max(1, pts.length - 1)) * plotW;
      const y = ppgBot - ((Number(p.ppg_norm || 0) - ppgMin) / ppgSpan) * ppgH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.8;
    ctx.stroke();

    ctx.lineTo(padL + plotW, ppgBot);
    ctx.lineTo(padL, ppgBot);
    ctx.closePath();
    ctx.fillStyle = "rgba(56, 189, 248, 0.12)";
    ctx.fill();

    // 2. Canal Superior: Traçado ECG Previsto por Rede Neural (segmentado por regularidade)
    ctx.lineWidth = 2.1;
    for (let i = 1; i < pts.length; i++) {
      const x0 = padL + ((i - 1) / Math.max(1, pts.length - 1)) * plotW;
      const y0 = ecgBot - ((Number(pts[i - 1].ecg_mv || 0) - ecgMin) / ecgSpan) * ecgH;
      const x1 = padL + (i / Math.max(1, pts.length - 1)) * plotW;
      const y1 = ecgBot - ((Number(pts[i].ecg_mv || 0) - ecgMin) / ecgSpan) * ecgH;

      const isIrreg = Boolean(pts[i].irreg || pts[i - 1].irreg);
      ctx.strokeStyle = isIrreg ? "#ef4444" : "#10b981";
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }

    // Marca picos R previstos e episódios irregulares
    for (let i = 2; i < pts.length - 2; i++) {
      const v = Number(pts[i].ecg_mv || 0);
      if (
        v > ecgMin + 0.65 * ecgSpan &&
        v >= Number(pts[i - 1].ecg_mv || 0) &&
        v > Number(pts[i + 1].ecg_mv || 0)
      ) {
        const x = padL + (i / Math.max(1, pts.length - 1)) * plotW;
        const y = ecgBot - ((v - ecgMin) / ecgSpan) * ecgH;
        const isIrreg = Boolean(pts[i].irreg);
        ctx.fillStyle = isIrreg ? "#ef4444" : "#34d399";
        ctx.beginPath();
        ctx.arc(x, y, isIrreg ? 3.5 : 2.5, 0, Math.PI * 2);
        ctx.fill();
        if (isIrreg) {
          ctx.font = "bold 9px -apple-system, BlinkMacSystemFont, sans-serif";
          ctx.textAlign = "center";
          ctx.fillStyle = "#fca5a5";
          ctx.fillText("⚠️", x, Math.max(14, y - 6));
        }
      }
    }

    // Cabeçalhos dos dois canais
    ctx.textAlign = "left";
    ctx.font = "bold 10.5px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillStyle = "#34d399";
    ctx.fillText(
      `🧠 ECG Reconstruído (Rede Neural PPG→ECG · Experimental) · FC Prev: ${neuralPred.predicted_hr_bpm ?? "—"} bpm`,
      10,
      15
    );

    ctx.textAlign = "right";
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px monospace";
    const intervals = neuralPred.intervals_ms || {};
    ctx.fillText(
      `PAT: ${neuralPred.predicted_pat_ms ?? "—"}ms · QTc: ${intervals.qtc_ms ?? "—"}ms`,
      width - 10,
      15
    );

    ctx.textAlign = "left";
    ctx.font = "bold 10px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillStyle = "#7dd3fc";
    ctx.fillText("📷 Pulso Óptico Capturado pela Câmera (PPG Sincronizado)", 10, splitY + 14);
  }

  function buildNeuralTelemetryHtml(neuralPred) {
    if (!neuralPred || !neuralPred.available) return "";
    const iv = neuralPred.intervals_ms || {};
    const isTrained = neuralPred.model_mode === "trained_neural";
    const modeBadgeClass = isTrained ? "good" : "medium";
    const modeText = isTrained
      ? `Rede Neural MLP Treinada (${neuralPred.n_paired_recordings} pares PPG+ECG)`
      : "Rede Neural PPG→ECG (Base Fisiológica)";

    return `
      <div class="neural-telemetry-box">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;flex-wrap:wrap;margin-bottom:0.55rem;">
          <span class="badge ${modeBadgeClass}">${modeText}</span>
          <span style="font-size:0.74rem;color:#94a3b8;">${neuralPred.architecture || "MLP PPG→ECG"}</span>
        </div>
        <div class="neural-telemetry-grid">
          <div class="nt-item">
            <span class="nt-lbl">FC ECG Prev.</span>
            <span class="nt-val">${neuralPred.predicted_hr_bpm ?? "—"} <small>bpm</small></span>
          </div>
          <div class="nt-item">
            <span class="nt-lbl">PAT Estimado</span>
            <span class="nt-val">${neuralPred.predicted_pat_ms ?? "—"} <small>ms</small></span>
          </div>
          <div class="nt-item">
            <span class="nt-lbl">Intervalo PR</span>
            <span class="nt-val">${iv.pr_ms ?? "—"} <small>ms</small></span>
          </div>
          <div class="nt-item">
            <span class="nt-lbl">Complexo QRS</span>
            <span class="nt-val">${iv.qrs_ms ?? "—"} <small>ms</small></span>
          </div>
          <div class="nt-item">
            <span class="nt-lbl">QTc (Bazett)</span>
            <span class="nt-val">${iv.qtc_ms ?? "—"} <small>ms</small></span>
          </div>
          <div class="nt-item">
            <span class="nt-lbl">RMSSD Prev.</span>
            <span class="nt-val">${neuralPred.predicted_rmssd_ms ?? "—"} <small>ms</small></span>
          </div>
        </div>
        <p style="font-size:0.76rem;color:#94a3b8;margin-top:0.55rem;line-height:1.4;">
          ℹ️ <strong>Previsão Experimental:</strong> ${neuralPred.disclaimer || ""}
        </p>
      </div>
    `;
  }

  function buildIrregularityHtml(irregularity) {
    if (!irregularity) return "";
    const status = irregularity.status || "regular";
    const score = irregularity.regularity_score != null ? irregularity.regularity_score : 85;
    let barColor = "#22c55e";
    let icon = "✅";
    if (status === "irregular") {
      barColor = "#ef4444";
      icon = "🚨";
    } else if (status === "borderline") {
      barColor = "#f59e0b";
      icon = "⚠️";
    } else if (status === "insufficient") {
      barColor = "#64748b";
      icon = "ℹ️";
    }

    return `
      <div class="ecg-safety-card level-irreg-${status}">
        <div class="ecg-safety-top">
          <strong>${icon} Triagem de Ritmo: ${irregularity.label || "Ritmo Regular"}</strong>
          <span class="ecg-safety-score" style="color:${barColor};">${score}%</span>
        </div>
        <div class="ecg-safety-bar-bg">
          <div class="ecg-safety-bar-fill" style="width:${score}%;background:${barColor};"></div>
        </div>
        <ul class="ecg-safety-pillars" style="margin-bottom:0.45rem;">
          <li><span>Intervalos Irregulares:</span> <strong>${irregularity.irregular_count ?? 0} de ${irregularity.total_beats ?? 0} (${irregularity.irregularity_pct ?? 0}%)</strong></li>
          <li><span>Padrão Extrassístole + Pausa:</span> <strong>${irregularity.premature_pairs ?? 0} episódio(s)</strong></li>
        </ul>
        <p style="font-size:0.8rem;color:var(--text);line-height:1.45;margin:0;">
          ${irregularity.clinical_summary || ""}
        </p>
      </div>
    `;
  }

  function renderEcgVsPpgComparisonCanvas(canvas, variability, bpConfidence) {
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(280, Math.floor(rect.width || canvas.width || 460));
    const height = Math.max(165, Math.floor(rect.height || canvas.height || 185));

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    ctx.fillStyle = "#070f1e";
    ctx.fillRect(0, 0, width, height);

    const ppgIbi = (variability && variability.ppg_ibi_ms) || [];
    const ecgSeries = (variability && variability.ecg_series) || {};
    const ecgCleanRR = ecgSeries.clean_rr_ms || [];
    const ecgRawRR = ecgSeries.raw_rr_ms || [];
    const rejectedSet = new Set(ecgSeries.rejected_indices || []);
    const hasEcg = Boolean(variability && variability.ecg_provided && ecgCleanRR.length >= 2);

    const ppgHr = ppgIbi.filter((v) => v > 0).map((v) => 60000 / v);
    const ecgHr = ecgRawRR.filter((v) => v > 0).map((v) => 60000 / v);

    const allHr = [...ppgHr, ...ecgHr];
    if (allHr.length < 2) {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Série batimento-a-batimento insuficiente para plotagem.", width / 2, height / 2);
      return;
    }

    const padL = 38;
    const padR = 14;
    const padT = 28;
    const padB = 26;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    const minHr = Math.max(35, Math.floor(Math.min(...allHr) - 8));
    const maxHr = Math.min(200, Math.ceil(Math.max(...allHr) + 8));
    const spanHr = Math.max(15, maxHr - minHr);

    const yForHr = (hr) => padT + plotH - ((hr - minHr) / spanHr) * plotH;

    ctx.strokeStyle = "rgba(148, 163, 184, 0.14)";
    ctx.lineWidth = 1;
    ctx.font = "10px monospace";
    ctx.fillStyle = "#64748b";
    ctx.textAlign = "right";

    const ticks = 4;
    for (let t = 0; t <= ticks; t++) {
      const hrVal = minHr + (spanHr * t) / ticks;
      const y = yForHr(hrVal);
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(width - padR, y);
      ctx.stroke();
      ctx.fillText(`${Math.round(hrVal)}`, padL - 5, y + 3);
    }

    if (hasEcg && ecgHr.length >= 2) {
      ctx.fillStyle = "rgba(34, 197, 94, 0.10)";
      ctx.beginPath();
      ecgHr.forEach((hr, i) => {
        const x = padL + (i / Math.max(1, ecgHr.length - 1)) * plotW;
        const yTop = yForHr(Math.min(maxHr, hr + 5));
        if (i === 0) ctx.moveTo(x, yTop);
        else ctx.lineTo(x, yTop);
      });
      for (let i = ecgHr.length - 1; i >= 0; i--) {
        const x = padL + (i / Math.max(1, ecgHr.length - 1)) * plotW;
        const yBot = yForHr(Math.max(minHr, ecgHr[i] - 5));
        ctx.lineTo(x, yBot);
      }
      ctx.closePath();
      ctx.fill();
    }

    if (ppgHr.length >= 2) {
      ctx.save();
      ctx.strokeStyle = "#60a5fa";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ppgHr.forEach((hr, i) => {
        const x = padL + (i / Math.max(1, ppgHr.length - 1)) * plotW;
        const y = yForHr(hr);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();

      ppgHr.forEach((hr, i) => {
        const x = padL + (i / Math.max(1, ppgHr.length - 1)) * plotW;
        const y = yForHr(hr);
        ctx.fillStyle = "#60a5fa";
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    if (hasEcg && ecgHr.length >= 2) {
      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ecgHr.forEach((hr, i) => {
        const x = padL + (i / Math.max(1, ecgHr.length - 1)) * plotW;
        const y = yForHr(hr);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ecgHr.forEach((hr, i) => {
        const x = padL + (i / Math.max(1, ecgHr.length - 1)) * plotW;
        const y = yForHr(hr);
        const isRej = rejectedSet.has(i);
        ctx.fillStyle = isRej ? "#ef4444" : "#22c55e";
        ctx.beginPath();
        ctx.arc(x, y, isRej ? 4 : 2.8, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    ctx.textAlign = "left";
    ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillStyle = "#93c5fd";
    ctx.fillText("┄ PPG Câmera (bpm)", padL, 16);

    if (hasEcg) {
      ctx.fillStyle = "#4ade80";
      ctx.fillText("━ ECG Referência (±5 bpm)", padL + 132, 16);
    } else {
      ctx.fillStyle = "#fbbf24";
      ctx.fillText("⚠️ Sem ECG externo anexado", padL + 132, 16);
    }

    ctx.textAlign = "center";
    ctx.fillStyle = "#64748b";
    ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.fillText("Sequência de Batimentos Capturados (15s)", padL + plotW / 2, height - 6);
  }

  function buildSafetyHtml(bpConfidence, variability) {
    const conf = bpConfidence || {};
    const level = conf.level || "moderate_no_ecg";
    const score = conf.safety_score != null ? conf.safety_score : (level === "high" ? 92 : level === "moderate_mismatch" ? 65 : 48);
    const hasEcg = Boolean(variability && variability.ecg_provided);
    const cmp = (variability && variability.comparison) || {};
    const ecgSeries = (variability && variability.ecg_series) || {};

    let barColor = "#f59e0b";
    let title = "Segurança Moderada — Apenas Câmera (PPG)";
    if (level === "high") {
      barColor = "#22c55e";
      title = "Segurança Alta — Sincronia ECG × Câmera Confirmada";
    } else if (level === "moderate_mismatch") {
      barColor = "#eab308";
      title = "Atenção — Divergência entre ECG e Câmera";
    }

    const deltaHrTxt =
      hasEcg && cmp.delta_mean_hr_bpm != null
        ? `${cmp.delta_mean_hr_bpm > 0 ? "+" : ""}${cmp.delta_mean_hr_bpm} bpm`
        : "Não medido";

    const ectopicCount = ecgSeries.ectopic_count || 0;
    const ectopicTxt = hasEcg
      ? ectopicCount > 0
        ? `${ectopicCount} batimento(s) atípico(s) filtrado(s)`
        : "Ritmo sinusal regular (0 ectopias)"
      : "Triagem óptica via rede neural PPG→ECG";

    const marginTxt =
      level === "high"
        ? "Margem nominal do modelo calibrado"
        : level === "moderate_mismatch"
        ? "Margem ampliada em 1,5× por divergência"
        : "Margem ampliada em 1,4× (mín. ±6 mmHg) por ausência de ECG";

    return `
      <div class="ecg-safety-card level-${level}">
        <div class="ecg-safety-top">
          <strong>🛡️ ${title}</strong>
          <span class="ecg-safety-score" style="color:${barColor};">${score}%</span>
        </div>
        <div class="ecg-safety-bar-bg">
          <div class="ecg-safety-bar-fill" style="width:${score}%;background:${barColor};"></div>
        </div>
        <ul class="ecg-safety-pillars">
          <li><span>1. Validação Elétrica (ECG):</span> <strong>${hasEcg ? "✅ Ativa" : "⚠️ Ausente (somente óptico)"}</strong></li>
          <li><span>2. Divergência FC (PPG − ECG):</span> <strong>${deltaHrTxt}</strong></li>
          <li><span>3. Auditoria de Ritmo:</span> <strong>${ectopicTxt}</strong></li>
          <li><span>4. Faixa de Incerteza PA:</span> <strong>${marginTxt}</strong></li>
        </ul>
      </div>
    `;
  }

  return {
    SAMPLE_NORMAL_RR,
    SAMPLE_ECTOPIC_RR,
    parseAndAnalyzeEcg,
    renderEcgWaveformCanvas,
    renderNeuralEcgPredictionCanvas,
    buildNeuralTelemetryHtml,
    buildIrregularityHtml,
    renderEcgVsPpgComparisonCanvas,
    buildSafetyHtml,
  };
})();
