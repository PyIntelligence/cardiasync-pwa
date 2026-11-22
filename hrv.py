"""
Métricas de variabilidade cardíaca / de pulso (HRV / PRV).

A partir do PPG calculamos PRV (pulse rate variability). Quando houver
intervalos RR de ECG, calculamos HRV de referência, filtramos artefatos/ectopia
e avaliamos o nível de confiança clínica da medição de pressão arterial.

Sem ECG o fluxo roda normalmente — a pressão arterial continua sendo estimada,
porém com margem de incerteza ampliada e sinalização de menor segurança clínica.
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


def analyze_ecg_rr_detailed(
    ibi_ms: np.ndarray,
    min_ms: float = 300.0,
    max_ms: float = 2000.0,
    ectopic_threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Limpeza profissional de intervalos RR de ECG com rastreamento de batimentos
    aceitos vs. rejeitados (ectopia/artefato) para visualização clínica.
    """
    if ibi_ms is None or len(ibi_ms) == 0:
        return {
            "clean": np.array([], dtype=float),
            "status": "missing",
            "raw_rr_ms": [],
            "clean_rr_ms": [],
            "rejected_indices": [],
            "ectopic_count": 0,
            "total_beats": 0,
        }

    arr = np.asarray(ibi_ms, dtype=float)
    arr = arr[np.isfinite(arr)]
    raw_list = [round(float(x), 1) for x in arr]

    in_range_mask = (arr >= min_ms) & (arr <= max_ms)
    valid_range = arr[in_range_mask]
    if len(valid_range) < 3:
        return {
            "clean": np.array([], dtype=float),
            "status": "invalid_range",
            "raw_rr_ms": raw_list,
            "clean_rr_ms": [],
            "rejected_indices": list(range(len(raw_list))),
            "ectopic_count": len(raw_list),
            "total_beats": len(raw_list),
        }

    median_rr = float(np.median(valid_range))
    if median_rr <= 0:
        return {
            "clean": np.array([], dtype=float),
            "status": "invalid_range",
            "raw_rr_ms": raw_list,
            "clean_rr_ms": [],
            "rejected_indices": list(range(len(raw_list))),
            "ectopic_count": len(raw_list),
            "total_beats": len(raw_list),
        }

    rejected_indices: List[int] = []
    clean_vals: List[float] = []
    for idx, val in enumerate(arr):
        if val < min_ms or val > max_ms or abs(val - median_rr) / median_rr > ectopic_threshold:
            rejected_indices.append(idx)
        else:
            clean_vals.append(float(val))

    clean_arr = np.asarray(clean_vals, dtype=float)
    if len(clean_arr) < 3:
        return {
            "clean": np.array([], dtype=float),
            "status": "noisy_ecg",
            "raw_rr_ms": raw_list,
            "clean_rr_ms": [round(float(x), 1) for x in clean_arr],
            "rejected_indices": rejected_indices,
            "ectopic_count": len(rejected_indices),
            "total_beats": len(raw_list),
        }

    return {
        "clean": clean_arr,
        "status": "validated",
        "raw_rr_ms": raw_list,
        "clean_rr_ms": [round(float(x), 1) for x in clean_arr],
        "rejected_indices": rejected_indices,
        "ectopic_count": len(rejected_indices),
        "total_beats": len(raw_list),
    }


def clean_ecg_rr(
    ibi_ms: np.ndarray,
    min_ms: float = 300.0,
    max_ms: float = 2000.0,
    ectopic_threshold: float = 0.20,
) -> Tuple[np.ndarray, str]:
    detail = analyze_ecg_rr_detailed(ibi_ms, min_ms, max_ms, ectopic_threshold)
    return detail["clean"], detail["status"]


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

    if label == "hrv":
        ibi, _ = clean_ecg_rr(np.asarray(list(ibi_ms), dtype=float))
    else:
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
            else "HRV de referência a partir de intervalos RR de ECG."
        ),
    }


def hrv_from_ppg_peaks(peaks: Sequence[int], fps: float) -> Dict[str, Any]:
    ibi = ibi_from_peak_indices(peaks, fps)
    return compute_hrv_metrics(ibi, label="prv")


def parse_ecg_input(
    ecg_rr_ms: Optional[Sequence[float]] = None,
    ecg_peaks_sec: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, str]:
    detail = parse_ecg_input_detailed(ecg_rr_ms=ecg_rr_ms, ecg_peaks_sec=ecg_peaks_sec)
    return detail["clean"], detail["status"]


def parse_ecg_input_detailed(
    ecg_rr_ms: Optional[Sequence[float]] = None,
    ecg_peaks_sec: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    if ecg_rr_ms is not None and len(list(ecg_rr_ms)) > 0:
        return analyze_ecg_rr_detailed(np.asarray(list(ecg_rr_ms), dtype=float))
    if ecg_peaks_sec is not None and len(list(ecg_peaks_sec)) >= 2:
        t = np.asarray(list(ecg_peaks_sec), dtype=float)
        t = t[np.isfinite(t)]
        if len(t) < 2 or not np.all(np.diff(t) > 0):
            return {
                "clean": np.array([], dtype=float),
                "status": "invalid_order",
                "raw_rr_ms": [],
                "clean_rr_ms": [],
                "rejected_indices": [],
                "ectopic_count": 0,
                "total_beats": 0,
            }
        return analyze_ecg_rr_detailed(np.diff(t) * 1000.0)
    return {
        "clean": np.array([], dtype=float),
        "status": "missing",
        "raw_rr_ms": [],
        "clean_rr_ms": [],
        "rejected_indices": [],
        "ectopic_count": 0,
        "total_beats": 0,
    }


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


def assess_bp_confidence(
    ecg_status: str,
    comparison: Optional[Dict[str, Any]] = None,
    base_sbp_std: Optional[float] = None,
    base_dbp_std: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Define o nível de segurança clínica da medição de pressão arterial sem
    interromper o cálculo quando o ECG está ausente.
    """
    if ecg_status == "validated":
        abs_delta_hr = float((comparison or {}).get("abs_delta_hr") or 0.0)
        if abs_delta_hr <= 5.0:
            safety_score = max(85, min(98, int(round(98 - abs_delta_hr * 2.2))))
            return {
                "level": "high",
                "safety_score": safety_score,
                "badge_class": "good",
                "label": "Validado por ECG (Alta Confiança)",
                "adjusted_sbp_std": round(float(base_sbp_std), 1) if base_sbp_std is not None else None,
                "adjusted_dbp_std": round(float(base_dbp_std), 1) if base_dbp_std is not None else None,
                "warning": None,
            }
        safety_score = max(55, min(74, int(round(78 - abs_delta_hr * 1.2))))
        return {
            "level": "moderate_mismatch",
            "safety_score": safety_score,
            "badge_class": "medium",
            "label": "Divergência ECG × Câmera (Confiança Reduzida)",
            "adjusted_sbp_std": round(float(base_sbp_std) * 1.5, 1) if base_sbp_std is not None else None,
            "adjusted_dbp_std": round(float(base_dbp_std) * 1.5, 1) if base_dbp_std is not None else None,
            "warning": (
                f"Diferença de {abs_delta_hr:.1f} bpm entre o ECG e o pulso óptico. "
                "A pressão foi estimada, mas recomenda-se repetir mantendo o dedo estável."
            ),
        }

    reason_map = {
        "missing": "Sem dados de ECG enviados.",
        "invalid_range": "Intervalos RR fora da faixa fisiológica (300–2000 ms).",
        "invalid_order": "Tempos de pico R fora de ordem crescente.",
        "noisy_ecg": "Sinal de ECG com excesso de artefatos/ectopia.",
    }
    reason_txt = reason_map.get(ecg_status, "ECG indisponível.")

    return {
        "level": "moderate_no_ecg",
        "safety_score": 48,
        "badge_class": "medium",
        "label": "Sem ECG — Precisão Reduzida",
        "adjusted_sbp_std": round(max(float(base_sbp_std) * 1.4, 6.0), 1) if base_sbp_std is not None else None,
        "adjusted_dbp_std": round(max(float(base_dbp_std) * 1.4, 4.5), 1) if base_dbp_std is not None else None,
        "warning": (
            f"{reason_txt} A pressão arterial foi calculada apenas pela câmera (PPG), "
            "com menor segurança clínica — utilize apenas como triagem."
        ),
    }


def build_variability_report(
    peaks: Sequence[int],
    fps: float,
    ecg_rr_ms: Optional[Sequence[float]] = None,
    ecg_peaks_sec: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Relatório completo: PRV sempre; HRV + comparação + séries batimento-a-batimento quando enviado.
    """
    raw_ppg_ibi = ibi_from_peak_indices(peaks, fps)
    clean_ppg_ibi = clean_ibi(raw_ppg_ibi)
    prv = compute_hrv_metrics(raw_ppg_ibi, label="prv")

    ecg_detail = parse_ecg_input_detailed(ecg_rr_ms=ecg_rr_ms, ecg_peaks_sec=ecg_peaks_sec)
    ecg_ibi = ecg_detail["clean"]
    ecg_status = ecg_detail["status"]

    if len(ecg_ibi) >= 3:
        hrv = compute_hrv_metrics(ecg_ibi, label="hrv")
        comparison = compare_prv_hrv(prv, hrv)
    else:
        hrv = None
        comparison = {
            "available": False,
            "message": "Sem dados válidos de ECG — sistema operando apenas com PPG/PRV.",
        }

    return {
        "prv": prv,
        "hrv_ecg": hrv,
        "comparison": comparison,
        "ecg_provided": hrv is not None,
        "ecg_status": ecg_status,
        "ppg_ibi_ms": [round(float(x), 1) for x in clean_ppg_ibi],
        "ecg_series": {
            "raw_rr_ms": ecg_detail["raw_rr_ms"],
            "clean_rr_ms": ecg_detail["clean_rr_ms"],
            "rejected_indices": ecg_detail["rejected_indices"],
            "ectopic_count": ecg_detail["ectopic_count"],
            "total_beats": ecg_detail["total_beats"],
        },
    }
