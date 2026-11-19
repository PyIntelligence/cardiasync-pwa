"""
Métricas de variabilidade cardíaca / de pulso (HRV / PRV).

A partir do PPG calculamos PRV (pulse rate variability). Quando houver
intervalos RR de ECG, calculamos HRV de referência e comparamos.

Sem ECG o fluxo roda normalmente — só não há bloco de validação ECG.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def ibi_from_peak_indices(peaks: Sequence[int], fps: float) -> np.ndarray:
    """Converte índices de pico em intervalos entre batimentos (ms)."""
    if len(peaks) < 2 or fps <= 0:
        return np.array([], dtype=float)
    intervals_s = np.diff(np.asarray(peaks, dtype=float)) / float(fps)
    return intervals_s * 1000.0  # ms


def clean_ibi(ibi_ms: np.ndarray, min_ms: float = 333.0, max_ms: float = 1500.0) -> np.ndarray:
    """Remove intervalos fisiologicamente improváveis (~40–180 bpm)."""
    if ibi_ms is None or len(ibi_ms) == 0:
        return np.array([], dtype=float)
    arr = np.asarray(ibi_ms, dtype=float)
    return arr[(arr >= min_ms) & (arr <= max_ms)]


def compute_hrv_metrics(ibi_ms: Sequence[float], label: str = "prv") -> Dict[str, Any]:
    """
    Métricas clássicas de domínio do tempo a partir de IBI em ms.

    label: "prv" (PPG) ou "hrv" (ECG) — só para documentação no JSON.
    """
    empty = {
        "source": label,
        "n_intervals": 0,
        "mean_ibi_ms": None,
        "mean_hr_bpm": None,
        "sdnn_ms": None,
        "rmssd_ms": None,
        "sdsd_ms": None,
        "pnn50_pct": None,
        "nn50": None,
        "cv_pct": None,
        "reliable": False,
        "note": "Poucos intervalos para HRV/PRV estável (ideal ≥ 20–30 s de sinal limpo).",
    }

    ibi = clean_ibi(np.asarray(list(ibi_ms), dtype=float))
    n = len(ibi)
    if n < 3:
        empty["n_intervals"] = n
        return empty

    mean_ibi = float(np.mean(ibi))
    sdnn = float(np.std(ibi, ddof=1)) if n > 1 else 0.0
    diff = np.diff(ibi)
    rmssd = float(np.sqrt(np.mean(diff ** 2))) if len(diff) else None
    sdsd = float(np.std(diff, ddof=1)) if len(diff) > 1 else None
    nn50 = int(np.sum(np.abs(diff) > 50.0)) if len(diff) else 0
    pnn50 = (100.0 * nn50 / len(diff)) if len(diff) else None
    cv = (100.0 * sdnn / mean_ibi) if mean_ibi > 0 else None

    # Confiabilidade grosseira: mais intervalos + menos CV extremo
    reliable = n >= 8 and mean_ibi > 0 and (cv is None or cv < 40)

    return {
        "source": label,
        "n_intervals": n,
        "mean_ibi_ms": round(mean_ibi, 2),
        "mean_hr_bpm": round(60000.0 / mean_ibi, 1) if mean_ibi > 0 else None,
        "sdnn_ms": round(sdnn, 2),
        "rmssd_ms": round(rmssd, 2) if rmssd is not None else None,
        "sdsd_ms": round(sdsd, 2) if sdsd is not None else None,
        "pnn50_pct": round(pnn50, 2) if pnn50 is not None else None,
        "nn50": nn50,
        "cv_pct": round(cv, 2) if cv is not None else None,
        "reliable": reliable,
        "note": (
            "PRV a partir de PPG — correlaciona com HRV mas não é idêntico ao ECG."
            if label == "prv"
            else "HRV a partir de intervalos RR de ECG."
        ),
    }


def hrv_from_ppg_peaks(peaks: Sequence[int], fps: float) -> Dict[str, Any]:
    ibi = ibi_from_peak_indices(peaks, fps)
    return compute_hrv_metrics(ibi, label="prv")


def parse_ecg_input(
    ecg_rr_ms: Optional[Sequence[float]] = None,
    ecg_peaks_sec: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """
    Aceita:
      - lista de intervalos RR em ms, ou
      - timestamps de picos R em segundos
    Retorna IBI em ms (pode ser vazio).
    """
    if ecg_rr_ms is not None and len(list(ecg_rr_ms)) > 0:
        return clean_ibi(np.asarray(list(ecg_rr_ms), dtype=float))
    if ecg_peaks_sec is not None and len(list(ecg_peaks_sec)) >= 2:
        t = np.asarray(list(ecg_peaks_sec), dtype=float)
        t = np.sort(t)
        return clean_ibi(np.diff(t) * 1000.0)
    return np.array([], dtype=float)


def compare_prv_hrv(prv: Dict[str, Any], hrv: Dict[str, Any]) -> Dict[str, Any]:
    """Compara métricas PPG vs ECG quando ambas existem."""
    if not prv or not hrv or prv.get("n_intervals", 0) < 3 or hrv.get("n_intervals", 0) < 3:
        return {
            "available": False,
            "message": "ECG não fornecido ou intervalos insuficientes — validação ECG omitida.",
        }

    def delta(key: str) -> Optional[float]:
        a, b = prv.get(key), hrv.get(key)
        if a is None or b is None:
            return None
        return round(float(a) - float(b), 2)

    return {
        "available": True,
        "message": "Comparação PRV (PPG) vs HRV (ECG).",
        "delta_mean_hr_bpm": delta("mean_hr_bpm"),
        "delta_sdnn_ms": delta("sdnn_ms"),
        "delta_rmssd_ms": delta("rmssd_ms"),
        "delta_pnn50_pct": delta("pnn50_pct"),
        "abs_delta_hr": abs(delta("mean_hr_bpm") or 0),
        "agreement_hr_within_5bpm": (
            abs(delta("mean_hr_bpm") or 99) <= 5
        ),
        "note": (
            "Diferenças são esperadas: PRV ≠ HRV (atraso de pulso / PAT). "
            "Use ECG como referência quando disponível."
        ),
    }


def build_variability_report(
    peaks: Sequence[int],
    fps: float,
    ecg_rr_ms: Optional[Sequence[float]] = None,
    ecg_peaks_sec: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Relatório completo: PRV sempre; HRV+comparação só se ECG vier no request.
    """
    prv = hrv_from_ppg_peaks(peaks, fps)
    ecg_ibi = parse_ecg_input(ecg_rr_ms=ecg_rr_ms, ecg_peaks_sec=ecg_peaks_sec)
    if len(ecg_ibi) >= 3:
        hrv = compute_hrv_metrics(ecg_ibi, label="hrv")
        comparison = compare_prv_hrv(prv, hrv)
    else:
        hrv = None
        comparison = {
            "available": False,
            "message": "Sem dados de ECG — sistema operando só com PPG/PRV.",
        }

    return {
        "prv": prv,
        "hrv_ecg": hrv,
        "comparison": comparison,
        "ecg_provided": hrv is not None,
    }
