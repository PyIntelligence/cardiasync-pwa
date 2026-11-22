"""
Geração de PDF de relatório de medição / calibração com suporte a:
- Validação ECG de referência
- Triagem automática de irregularidade de pulso (suspeita de arritmia/ectopia)
- Traçado vetorial reconstruído por Rede Neural (PPG-to-ECG · Experimental)
"""

import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.graphics.shapes import Drawing, Rect, Line, PolyLine, String


def _build_neural_ecg_drawing(neural_pred: dict) -> Drawing:
    """Desenha o gráfico vetorial ECG Previsto (PPG-to-ECG) + Onda PPG no PDF."""
    w = 170 * mm
    h = 46 * mm
    d = Drawing(w, h)

    # Fundo escuro estilo monitor clínico
    d.add(Rect(0, 0, w, h, fillColor=HexColor("#071122"), strokeColor=HexColor("#1e293b"), strokeWidth=0.8))

    # Grade milimetrada sutil
    step = 6 * mm
    x = step
    while x < w:
        d.add(Line(x, 0, x, h, strokeColor=HexColor("#0f233c"), strokeWidth=0.35))
        x += step
    y = step
    while y < h:
        d.add(Line(0, y, w, y, strokeColor=HexColor("#0f233c"), strokeWidth=0.35))
        y += step

    # Linha divisória entre canal superior (ECG Neural) e inferior (PPG Óptico)
    mid_y = h * 0.42
    d.add(Line(0, mid_y, w, mid_y, strokeColor=HexColor("#1e3a5f"), strokeWidth=0.5))

    d.add(String(4 * mm, h - 4.2 * mm, "ECG Sintético Previsto por Rede Neural (PPG-to-ECG · Experimental)",
                 fontName="Helvetica-Bold", fontSize=6.8, fillColor=HexColor("#34d399")))
    d.add(String(4 * mm, mid_y - 3.8 * mm, "Pulso Óptico Capturado pela Câmera (PPG Sincronizado)",
                 fontName="Helvetica-Bold", fontSize=6.5, fillColor=HexColor("#60a5fa")))

    pts = (neural_pred or {}).get("waveform_points") or []
    if len(pts) < 4:
        d.add(String(w / 2, h / 2, "Série insuficiente para traçado gráfico.",
                     fontName="Helvetica", fontSize=8, fillColor=HexColor("#94a3b8"), textAnchor="middle"))
        return d

    pad_l = 4 * mm
    pad_r = 4 * mm
    plot_w = w - pad_l - pad_r

    ecg_vals = [float(p.get("ecg_mv", 0.0)) for p in pts]
    ppg_vals = [float(p.get("ppg_norm", 0.0)) for p in pts]

    ecg_min, ecg_max = min(ecg_vals), max(ecg_vals)
    ecg_span = max(0.4, ecg_max - ecg_min)
    ecg_y0 = mid_y + 3 * mm
    ecg_h = (h - mid_y) - 8 * mm

    ppg_min, ppg_max = min(ppg_vals), max(ppg_vals)
    ppg_span = max(0.2, ppg_max - ppg_min)
    ppg_y0 = 2.5 * mm
    ppg_h = mid_y - 7.5 * mm

    n = len(pts)
    ecg_coords = []
    ppg_coords = []
    for i, p in enumerate(pts):
        px = pad_l + (i / max(1, n - 1)) * plot_w
        ey = ecg_y0 + ((float(p.get("ecg_mv", 0.0)) - ecg_min) / ecg_span) * ecg_h
        py = ppg_y0 + ((float(p.get("ppg_norm", 0.0)) - ppg_min) / ppg_span) * ppg_h
        ecg_coords.extend([px, ey])
        ppg_coords.extend([px, py])

    d.add(PolyLine(ecg_coords, strokeColor=HexColor("#22c55e"), strokeWidth=1.15))
    d.add(PolyLine(ppg_coords, strokeColor=HexColor("#60a5fa"), strokeWidth=1.0))

    # Destaca segmentos irregulares em vermelho
    irreg_seg = []
    for i, p in enumerate(pts):
        px = pad_l + (i / max(1, n - 1)) * plot_w
        ey = ecg_y0 + ((float(p.get("ecg_mv", 0.0)) - ecg_min) / ecg_span) * ecg_h
        if p.get("irreg"):
            irreg_seg.extend([px, ey])
        else:
            if len(irreg_seg) >= 4:
                d.add(PolyLine(irreg_seg, strokeColor=HexColor("#ef4444"), strokeWidth=1.5))
            irreg_seg = []
    if len(irreg_seg) >= 4:
        d.add(PolyLine(irreg_seg, strokeColor=HexColor("#ef4444"), strokeWidth=1.5))

    return d


def build_measurement_pdf(data: dict) -> bytes:
    """
    data keys esperadas:
      user_name, created_at, heart_rate, estimated_systolic, estimated_diastolic,
      sbp_std, dbp_std, signal_quality, condition, feeding, position, arm,
      pulse_height, observation, status, ecg_validation, bp_confidence,
      pulse_irregularity, neural_ecg_prediction
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleBR",
        parent=styles["Heading1"],
        fontSize=17,
        textColor=HexColor("#1e40af"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "SecTitle",
        parent=styles["Heading2"],
        fontSize=11.5,
        textColor=HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=5,
    )
    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontSize=9,
        textColor=HexColor("#64748b"),
    )
    value_style = ParagraphStyle(
        "Value",
        parent=styles["Normal"],
        fontSize=9.5,
        textColor=black,
        spaceAfter=4,
        leading=13,
    )
    warn_style = ParagraphStyle(
        "Warn",
        parent=styles["Normal"],
        fontSize=8,
        textColor=HexColor("#b45309"),
        alignment=TA_CENTER,
        spaceBefore=10,
    )

    story = []
    story.append(Paragraph("CardiaSync — Relatório Cardiovascular de Triagem", title_style))
    story.append(Paragraph("Fotopletismografia (PPG), Triagem de Ritmo e Tradução Neural PPG-to-ECG", subtitle))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#3b82f6")))
    story.append(Spacer(1, 8))

    created = data.get("created_at") or datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    rows = [
        ["Paciente / Usuário", data.get("user_name") or "—"],
        ["Data e hora", str(created)],
        ["Condição / Posição", f"{data.get('condition') or '—'} · {data.get('position') or '—'}"],
        ["Braço / Altura Sensor", f"{data.get('arm') or '—'} · {data.get('pulse_height') or '—'}"],
    ]

    table = Table(rows, colWidths=[65 * mm, 105 * mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#475569")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, HexColor("#e2e8f0")),
    ]))
    story.append(table)
    story.append(Spacer(1, 10))

    # 1. Resultados principais de Pressão e Pulso
    hr = data.get("heart_rate")
    sbp = data.get("estimated_systolic")
    dbp = data.get("estimated_diastolic")
    sbp_std = data.get("sbp_std")
    dbp_std = data.get("dbp_std")
    quality = data.get("signal_quality")
    q_pct = f"{int(quality * 100)}%" if quality is not None else "—"

    ecg_val = data.get("ecg_validation") or {}
    bp_conf = data.get("bp_confidence") or ecg_val.get("bp_confidence") or {}
    ecg_provided = bool(ecg_val.get("ecg_provided"))

    conf_label = bp_conf.get("label") or (
        "Validado por ECG (Alta Confiança)" if ecg_provided else "Sem ECG — Precisão Reduzida"
    )

    sbp_str = f"{sbp} mmHg" + (f" (±{sbp_std})" if sbp and sbp_std else "") if sbp else "Não calibrado"
    dbp_str = f"{dbp} mmHg" + (f" (±{dbp_std})" if dbp and dbp_std else "") if dbp else "Não calibrado"

    result_data = [
        ["Frequência Cardíaca (PPG)", f"{hr} bpm" if hr else "—"],
        ["Pressão Sistólica (estimada)", sbp_str],
        ["Pressão Diastólica (estimada)", dbp_str],
        ["Qualidade do sinal óptico", q_pct],
        ["Validação ECG / Segurança", conf_label],
    ]

    if ecg_provided:
        hrv_ecg = ecg_val.get("hrv_ecg") or {}
        cmp = ecg_val.get("comparison") or {}
        if hrv_ecg.get("rmssd_ms") is not None:
            result_data.append(["HRV ECG Real (RMSSD / SDNN)", f"{hrv_ecg.get('rmssd_ms')} ms / {hrv_ecg.get('sdnn_ms') or '—'} ms"])
        if cmp.get("delta_mean_hr_bpm") is not None:
            result_data.append(["Divergência FC (PPG - ECG)", f"{cmp.get('delta_mean_hr_bpm')} bpm"])

    res_table = Table(result_data, colWidths=[65 * mm, 105 * mm])
    box_color = HexColor("#16a34a") if ecg_provided and bp_conf.get("level") == "high" else HexColor("#3b82f6")
    bg_color = HexColor("#f0fdf4") if ecg_provided and bp_conf.get("level") == "high" else HexColor("#f0f9ff")

    res_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_color),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#1e3a8a")),
        ("BOX", (0, 0), (-1, -1), 1, box_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(res_table)

    # 2. Triagem Automática de Irregularidade de Pulso (para levar ao cardiologista)
    irreg = data.get("pulse_irregularity") or {}
    if irreg:
        story.append(Paragraph("Triagem de Ritmo e Irregularidade de Pulso", section_style))
        irreg_status = irreg.get("status", "regular")
        irreg_bg = (
            HexColor("#fef2f2") if irreg_status == "irregular"
            else HexColor("#fffbeb") if irreg_status == "borderline"
            else HexColor("#f0fdf4")
        )
        irreg_border = (
            HexColor("#dc2626") if irreg_status == "irregular"
            else HexColor("#d97706") if irreg_status == "borderline"
            else HexColor("#16a34a")
        )
        irreg_rows = [
            ["Classificação de Ritmo", irreg.get("label") or "—"],
            ["Índice de Regularidade", f"{irreg.get('regularity_score', '—')}% ({irreg.get('irregular_count', 0)}/{irreg.get('total_beats', 0)} intervalos atípicos)"],
            ["Pares Precoces + Pausa", f"{irreg.get('premature_pairs', 0)} episódio(s) sugestivo(s) de extrassístole"],
        ]
        irreg_table = Table(irreg_rows, colWidths=[65 * mm, 105 * mm])
        irreg_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), irreg_bg),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOX", (0, 0), (-1, -1), 0.9, irreg_border),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(irreg_table)
        if irreg.get("clinical_summary"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<b>Parecer de Triagem:</b> {irreg['clinical_summary']}", value_style))

    # 3. Previsão Neural de ECG (PPG-to-ECG Translation) + Gráfico Vetorial
    neural_pred = data.get("neural_ecg_prediction") or {}
    if neural_pred.get("available"):
        story.append(Paragraph("Previsão Neural de ECG (PPG-to-ECG · Experimental)", section_style))
        intervals = neural_pred.get("intervals_ms") or {}
        neural_rows = [
            ["Arquitetura / Modo", neural_pred.get("model_label") or "Rede Neural PPG→ECG"],
            ["FC ECG Prevista / PAT Est.", f"{neural_pred.get('predicted_hr_bpm', '—')} bpm · PAT {neural_pred.get('predicted_pat_ms', '—')} ms"],
            ["Intervalos Estimados", f"PR: {intervals.get('pr_ms', '—')} ms · QRS: {intervals.get('qrs_ms', '—')} ms · QTc: {intervals.get('qtc_ms', '—')} ms"],
            ["HRV Neural Prevista", f"RMSSD: {neural_pred.get('predicted_rmssd_ms', '—')} ms · SDNN: {neural_pred.get('predicted_sdnn_ms', '—')} ms"],
        ]
        n_table = Table(neural_rows, colWidths=[65 * mm, 105 * mm])
        n_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#f8fafc")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#0f172a")),
            ("BOX", (0, 0), (-1, -1), 0.8, HexColor("#6366f1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.35, HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(n_table)
        story.append(Spacer(1, 6))
        story.append(_build_neural_ecg_drawing(neural_pred))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            f"<i>Nota sobre o traçado neural: {neural_pred.get('disclaimer', '')}</i>",
            ParagraphStyle("NeuralDisc", parent=styles["Normal"], fontSize=8, textColor=HexColor("#475569")),
        ))

    if data.get("observation"):
        story.append(Spacer(1, 6))
        story.append(Paragraph("<b>Observação do Usuário:</b> " + str(data["observation"]), value_style))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#cbd5e1")))
    story.append(Paragraph(
        "AVISO MÉDICO: Este relatório é gerado por um sistema experimental de fotopletismografia (PPG) e tradução neural. "
        "Não é um dispositivo médico certificado e o traçado ECG previsto não substitui um eletrocardiograma real de 12 derivações. "
        "Utilize os alertas de irregularidade de pulso como apoio para consulta com seu cardiologista.",
        warn_style,
    ))
    story.append(Paragraph(
        f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} — CardiaSync © 2026",
        ParagraphStyle("Foot", parent=styles["Normal"], fontSize=7, textColor=HexColor("#94a3b8"), alignment=TA_CENTER, spaceBefore=6),
    ))

    doc.build(story)
    return buffer.getvalue()
