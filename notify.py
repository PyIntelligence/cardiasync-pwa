"""
Envio de PDF por e-mail e WhatsApp.

Configuração via variáveis de ambiente (Render):

E-mail (SMTP genérico ou serviços como SendGrid/Resend/Gmail):
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM

WhatsApp (Twilio):
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM
  (ex: whatsapp:+14155238886)
"""

import os
import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional

logger = logging.getLogger(__name__)


def send_email_with_pdf(
    to_email: str,
    subject: str,
    body_text: str,
    pdf_bytes: bytes,
    filename: str = "relatorio_cardiasync.pdf",
) -> tuple[bool, str]:
    """Envia e-mail com PDF anexado. Retorna (sucesso, mensagem)."""
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    from_addr = os.getenv("SMTP_FROM") or user

    if not all([host, user, password, from_addr]):
        return False, "SMTP não configurado (SMTP_HOST, SMTP_USER, SMTP_PASSWORD, SMTP_FROM)"

    if not to_email or "@" not in to_email:
        return False, "E-mail de destino inválido"

    try:
        msg = MIMEMultipart()
        msg["From"] = from_addr
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body_text, "plain", "utf-8"))

        part = MIMEBase("application", "pdf")
        part.set_payload(pdf_bytes)
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{filename}"')
        msg.attach(part)

        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(from_addr, [to_email], msg.as_string())

        return True, "E-mail enviado com sucesso"
    except Exception as e:
        logger.exception("Falha ao enviar e-mail")
        return False, f"Erro no envio de e-mail: {e}"


def send_whatsapp_notification(
    to_phone: str,
    message: str,
    media_url: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Envia mensagem WhatsApp via Twilio.
    to_phone deve estar no formato internacional, ex: +5583999999999
    media_url: URL pública do PDF (opcional; Twilio precisa de URL acessível).
    """
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_wa = os.getenv("TWILIO_WHATSAPP_FROM")  # whatsapp:+14155238886

    if not all([sid, token, from_wa]):
        return False, "Twilio WhatsApp não configurado (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM)"

    phone = to_phone.strip().replace(" ", "").replace("-", "")
    if not phone.startswith("+"):
        # Assume Brasil se não tiver +
        if phone.startswith("55"):
            phone = "+" + phone
        else:
            phone = "+55" + phone

    try:
        from twilio.rest import Client
        client = Client(sid, token)
        kwargs = {
            "from_": from_wa,
            "to": f"whatsapp:{phone}",
            "body": message,
        }
        if media_url:
            kwargs["media_url"] = [media_url]
        client.messages.create(**kwargs)
        return True, "Mensagem WhatsApp enviada"
    except ImportError:
        return False, "Biblioteca twilio não instalada (pip install twilio)"
    except Exception as e:
        logger.exception("Falha WhatsApp")
        return False, f"Erro WhatsApp: {e}"


def send_email_plain(
    to_email: str,
    subject: str,
    body_text: str,
) -> tuple[bool, str]:
    """Envia e-mail simples em texto (ex.: código de recuperação de senha)."""
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    from_addr = os.getenv("SMTP_FROM") or user

    if not all([host, user, password, from_addr]):
        return False, "E-mail (SMTP) ainda não configurado no servidor."

    if not to_email or "@" not in to_email:
        return False, "E-mail de destino inválido."

    try:
        msg = MIMEMultipart()
        msg["From"] = from_addr
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body_text, "plain", "utf-8"))

        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(from_addr, [to_email], msg.as_string())

        return True, "Código enviado para o e-mail cadastrado."
    except Exception as e:
        logger.exception("Falha ao enviar e-mail de recuperação")
        return False, f"Erro ao enviar e-mail: {e}"
