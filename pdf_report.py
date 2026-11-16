"""
Geração de PDF de relatório de medição / calibração.
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
from reportlab.lib.enums import TA_CENTER, TA_LEFT


def build_measurement_pdf(data: dict) -> bytes:
    """
    data keys esperadas:
      user_name, created_at, heart_rate, estimated_systolic, estimated_diastolic,
      signal_quality, condition, feeding, position, arm, pulse_height,
      observation, status, calibration_count (opcional)
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleBR",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=HexColor("#1e40af"),
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    subtitle = ParagraphStyle(
        "Sub",
        parent=styles["Normal"],
        fontSize=10,
        textColor=HexColor("#64748b"),
        alignment=TA_CENTER,
        spaceAfter=12,
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
        fontSize=12,
        textColor=black,
        spaceAfter=4,
    )
    warn_style = ParagraphStyle(
        "Warn",
        parent=styles["Normal"],
        fontSize=8,
        textColor=HexColor("#b45309"),
        alignment=TA_CENTER,
        spaceBefore=16,
    )

    story = []
    story.append(Paragraph("CardiaSync — Relatório de Medição", title_style))
    story.append(Paragraph("Sistema experimental de PPG por câmera", subtitle))
    story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#3b82f6")))
    story.append(Spacer(1, 10))

    created = data.get("created_at") or datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    rows = [
        ["Paciente / Usuário", data.get("user_name") or "—"],
        ["Data e hora", str(created)],
        ["Condição", data.get("condition") or "—"],
        ["Alimentação", data.get("feeding") or "—"],
        ["Posição", data.get("position") or "—"],
        ["Braço", data.get("arm") or "—"],
        ["Altura do sensor", data.get("pulse_height") or "—"],
    ]

    table = Table(rows, colWidths=[70 * mm, 100 * mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#475569")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, HexColor("#e2e8f0")),
    ]))
    story.append(table)
    story.append(Spacer(1, 14))

    # Resultados principais
    hr = data.get("heart_rate")
    sbp = data.get("estimated_systolic")
    dbp = data.get("estimated_diastolic")
    quality = data.get("signal_quality")
    q_pct = f"{int(quality * 100)}%" if quality is not None else "—"

    result_data = [
        ["Frequência Cardíaca", f"{hr} bpm" if hr else "—"],
        ["Pressão Sistólica (estimada)", f"{sbp} mmHg" if sbp else "Não calibrado"],
        ["Pressão Diastólica (estimada)", f"{dbp} mmHg" if dbp else "Não calibrado"],
        ["Qualidade do sinal", q_pct],
        ["Status do modelo", data.get("status") or "—"],
    ]
    res_table = Table(result_data, colWidths=[70 * mm, 100 * mm])
    res_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#f0f9ff")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#1e40af")),
        ("BOX", (0, 0), (-1, -1), 1, HexColor("#3b82f6")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, HexColor("#bfdbfe")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(res_table)

    if data.get("observation"):
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Observação</b>", label_style))
        story.append(Paragraph(str(data["observation"]), value_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#cbd5e1")))
    story.append(Paragraph(
        "AVISO: Este relatório é gerado por um sistema experimental de fotopletismografia (PPG). "
        "Não é um dispositivo médico certificado. As estimativas de pressão arterial não devem ser "
        "utilizadas para diagnóstico, tratamento ou tomada de decisão clínica. Consulte sempre um profissional de saúde.",
        warn_style,
    ))
    story.append(Paragraph(
        f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} — CardiaSync PWA © 2026",
        ParagraphStyle("Foot", parent=styles["Normal"], fontSize=7, textColor=HexColor("#94a3b8"), alignment=TA_CENTER, spaceBefore=8),
    ))

    doc.build(story)
    return buffer.getvalue()
