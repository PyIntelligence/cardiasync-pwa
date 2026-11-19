"""
Validação de precisão do modelo global + detecção de outliers em PPG/calibrações.

Validação:
  - K-fold (ou LOO se n pequeno)
  - MAE, RMSE, bias, R²
  - % dentro de 5 / 10 / 15 mmHg (estilo AAMI/BHS simplificado)

Outliers:
  - Sinal bruto: amplitude extrema, qualidade baixa, FC fora de faixa, instabilidade
  - Features de calibração: z-score multivariado aproximado (por feature) + residual do modelo
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from calibration_engine import (
    FEATURE_NAMES,
    HAS_SKLEARN,
    _build_feature_matrix,
    _g,
    estimate_blood_pressure,
    extract_basic_features,
    train_polynomial_global_model,
    train_simple_linear_model,
)

try:
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
except ImportError:
    KFold = None
    r2_score = None


# ──────────────────────────────────────────────
# Outliers no sinal PPG (durante captura / pós-features)
# ──────────────────────────────────────────────

def detect_signal_outliers(features: Dict, red_signal: Optional[list] = None) -> Dict[str, Any]:
    """
    Avalia se um sinal/feature set é outlier.
    Retorna flags, score 0–1 (1 = mais anômalo) e motivos.
    """
    reasons: List[str] = []
    score = 0.0

    hr = features.get("heart_rate")
    q = features.get("signal_quality") or 0.0
    std = features.get("std_red") or 0.0
    mean_r = features.get("mean_red") or 0.0
    stability = features.get("stability") or 0.0
    snr = features.get("snr_approx") or 0.0
    n_peaks = features.get("n_peaks") or 0

    if q < 0.35:
        reasons.append("qualidade do sinal baixa")
        score += 0.25
    if hr is None:
        reasons.append("frequência cardíaca não detectada")
        score += 0.25
    elif hr < 40 or hr > 180:
        reasons.append(f"FC fora da faixa fisiológica ({hr} bpm)")
        score += 0.3
    if mean_r < 15 or mean_r > 240:
        reasons.append("intensidade média atípica (dedo/iluminação)")
        score += 0.15
    if std < 0.8:
        reasons.append("variância muito baixa (pouco pulso)")
        score += 0.15
    if std > 80:
        reasons.append("variância excessiva (movimento/ruído)")
        score += 0.2
    if stability < 0.25 and n_peaks >= 3:
        reasons.append("intervalos entre batimentos instáveis")
        score += 0.15
    if snr < 1.2 and snr > 0:
        reasons.append("SNR baixo")
        score += 0.1

    if red_signal and len(red_signal) >= 30:
        arr = np.asarray(red_signal, dtype=float)
        # Saltos bruscos (movimento)
        diff = np.abs(np.diff(arr))
        if len(diff) and float(np.percentile(diff, 95)) > max(25.0, 3 * float(np.std(arr))):
            reasons.append("saltos bruscos no sinal (possível movimento)")
            score += 0.2
        # Saturação
        sat = float(np.mean((arr < 5) | (arr > 250)))
        if sat > 0.15:
            reasons.append("muitos frames saturados/escuros")
            score += 0.15

    score = min(1.0, score)
    is_outlier = score >= 0.45 or q < 0.3 or hr is None

    return {
        "is_outlier": is_outlier,
        "outlier_score": round(score, 3),
        "reasons": reasons,
        "usable_for_training": (not is_outlier) and q >= 0.35 and hr is not None,
    }


def detect_calibration_outliers(
    calibrations: list,
    z_thresh: float = 3.0,
) -> Dict[str, Any]:
    """
    Detecta outliers no conjunto de calibrações (espaço de features + PA de referência).
    Usa z-score univariado por coluna; marca amostra se ≥ 2 features forem extremas
    ou se a própria PA de referência for absurda.
    """
    X, y_s, y_d, meta = _build_feature_matrix(calibrations, min_samples=1)
    if X is None or len(X) < 5:
        return {
            "ok": False,
            "n": 0 if X is None else len(X),
            "outlier_indices": [],
            "details": [],
            "message": "Poucos dados para análise de outliers de calibração.",
        }

    # Inclui referências no espaço de detecção
    full = np.column_stack([X, y_s, y_d])
    names = FEATURE_NAMES + ["ref_systolic", "ref_diastolic"]
    mu = np.mean(full, axis=0)
    sigma = np.std(full, axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    z = np.abs((full - mu) / sigma)

    outlier_indices = []
    details = []
    for i in range(len(full)):
        extreme_cols = [names[j] for j in range(len(names)) if z[i, j] >= z_thresh]
        # Regras fisiológicas hard
        hard = []
        if y_s[i] < 70 or y_s[i] > 220:
            hard.append("ref_systolic fora da faixa")
        if y_d[i] < 40 or y_d[i] > 130:
            hard.append("ref_diastolic fora da faixa")
        if y_d[i] >= y_s[i]:
            hard.append("diastólica ≥ sistólica")
        if X[i, 0] < 40 or X[i, 0] > 180:
            hard.append("FC atípica")

        is_out = len(extreme_cols) >= 2 or len(hard) > 0
        if is_out:
            outlier_indices.append(i)
            details.append({
                "index": i,
                "z_extreme_features": extreme_cols,
                "hard_rules": hard,
                "hr": float(X[i, 0]),
                "ref_systolic": float(y_s[i]),
                "ref_diastolic": float(y_d[i]),
                "max_z": round(float(np.max(z[i])), 2),
            })

    return {
        "ok": True,
        "n": int(len(X)),
        "n_outliers": len(outlier_indices),
        "outlier_fraction": round(len(outlier_indices) / len(X), 3),
        "outlier_indices": outlier_indices,
        "details": details,
        "z_threshold": z_thresh,
    }


def filter_calibrations_remove_outliers(
    calibrations: list,
    z_thresh: float = 3.0,
) -> Tuple[list, Dict]:
    """Remove outliers de calibração antes do treino. Mantém ordem relativa."""
    report = detect_calibration_outliers(calibrations, z_thresh=z_thresh)
    if not report.get("ok"):
        return calibrations, report

    # Reconstrói lista só com pontos que passam no mesmo filtro do matrix builder
    usable = []
    for c in calibrations:
        hr = _g(c, "heart_rate")
        amp = _g(c, "peak_amplitude")
        std = _g(c, "std_red")
        ref_s = _g(c, "ref_systolic")
        ref_d = _g(c, "ref_diastolic")
        q = _g(c, "signal_quality")
        if None in (hr, amp, std, ref_s, ref_d):
            continue
        if q is not None and q < 0.3:
            continue
        usable.append(c)

    drop = set(report.get("outlier_indices") or [])
    cleaned = [c for i, c in enumerate(usable) if i not in drop]
    report["n_kept"] = len(cleaned)
    report["n_removed"] = len(usable) - len(cleaned)
    return cleaned, report


# ──────────────────────────────────────────────
# Validação de precisão do modelo global
# ──────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    out = {
        "n": int(len(y_true)),
        "mae": round(float(np.mean(abs_err)), 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "bias": round(float(np.mean(err)), 2),
        "std_error": round(float(np.std(err)), 2),
        "pct_within_5": round(float(np.mean(abs_err <= 5) * 100), 1),
        "pct_within_10": round(float(np.mean(abs_err <= 10) * 100), 1),
        "pct_within_15": round(float(np.mean(abs_err <= 15) * 100), 1),
    }
    if r2_score is not None and len(y_true) >= 3:
        try:
            out["r2"] = round(float(r2_score(y_true, y_pred)), 4)
        except Exception:
            out["r2"] = None
    else:
        out["r2"] = None
    return out


def validate_global_model(
    calibrations: list,
    degree: int = 2,
    n_folds: int = 5,
    remove_outliers: bool = True,
) -> Dict[str, Any]:
    """
    Validação cruzada do modelo polinomial global.

    Se sklearn indisponível, faz validação hold-out simples 80/20.
    """
    cleaned, outlier_report = (
        filter_calibrations_remove_outliers(calibrations)
        if remove_outliers
        else (calibrations, {"ok": False, "n_removed": 0})
    )

    X, y_s, y_d, _ = _build_feature_matrix(cleaned, min_samples=8)
    if X is None:
        return {
            "ok": False,
            "error": "Dados insuficientes para validação (mín. 8 após filtros).",
            "outlier_report": outlier_report,
        }

    n = len(X)
    degree = max(1, min(3, int(degree)))
    folds = min(n_folds, n) if n >= 10 else max(2, min(n, 5))

    pred_s = np.full(n, np.nan)
    pred_d = np.full(n, np.nan)

    if HAS_SKLEARN and KFold is not None and n >= 8:
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        # índices no cleaned list alinhados com X
        # reconstruir lista cleaned já filtrada pelo matrix — usar índices 0..n-1 sobre X
        # Treina em folds usando train_polynomial sobre subset de dicts sintéticos
        class Row:
            pass

        rows = []
        for i in range(n):
            r = Row()
            r.heart_rate, r.peak_amplitude, r.std_red = X[i, 0], X[i, 1], X[i, 2]
            r.mean_red, r.pulse_width = X[i, 3], X[i, 4]
            r.ref_systolic, r.ref_diastolic = y_s[i], y_d[i]
            r.signal_quality = 0.8
            rows.append(r)

        for train_idx, test_idx in kf.split(X):
            train_rows = [rows[i] for i in train_idx]
            coeffs = train_polynomial_global_model(
                train_rows, degree=degree, min_samples=max(5, len(train_rows) // 2)
            )
            if not coeffs:
                # fallback linear no fold
                coeffs = train_simple_linear_model(train_rows)
            if not coeffs:
                continue
            for i in test_idx:
                feats = {
                    "heart_rate": X[i, 0],
                    "peak_amplitude": X[i, 1],
                    "std_red": X[i, 2],
                    "mean_red": X[i, 3],
                    "pulse_width": X[i, 4],
                }
                ps, pd, st, _, _ = estimate_blood_pressure(feats, coeffs)
                if st == "calibrado" and ps is not None:
                    pred_s[i] = ps
                    pred_d[i] = pd
    else:
        # Hold-out 80/20 sem sklearn CV
        idx = np.arange(n)
        np.random.default_rng(42).shuffle(idx)
        split = max(1, int(n * 0.8))
        train_i, test_i = idx[:split], idx[split:]
        if len(test_i) == 0:
            test_i = idx[-1:]
            train_i = idx[:-1]

        class Row:
            pass

        train_rows = []
        for i in train_i:
            r = Row()
            r.heart_rate, r.peak_amplitude, r.std_red = X[i, 0], X[i, 1], X[i, 2]
            r.mean_red, r.pulse_width = X[i, 3], X[i, 4]
            r.ref_systolic, r.ref_diastolic = float(y_s[i]), float(y_d[i])
            r.signal_quality = 0.8
            train_rows.append(r)
        coeffs = train_polynomial_global_model(train_rows, degree=degree, min_samples=5)
        if not coeffs:
            coeffs = train_simple_linear_model(train_rows)
        if coeffs:
            for i in test_i:
                feats = {
                    "heart_rate": X[i, 0],
                    "peak_amplitude": X[i, 1],
                    "std_red": X[i, 2],
                    "mean_red": X[i, 3],
                    "pulse_width": X[i, 4],
                }
                ps, pd, st, _, _ = estimate_blood_pressure(feats, coeffs)
                if st == "calibrado" and ps is not None:
                    pred_s[i] = ps
                    pred_d[i] = pd
        folds = 1

    mask = ~np.isnan(pred_s)
    if not np.any(mask):
        return {
            "ok": False,
            "error": "Não foi possível gerar previsões na validação (sklearn/dados).",
            "outlier_report": outlier_report,
        }

    m_s = _metrics(y_s[mask], pred_s[mask])
    m_d = _metrics(y_d[mask], pred_d[mask])

    # Interpretação simples estilo BHS/AAMI (muito simplificada)
    grade_s = _grade_bhs(m_s)
    grade_d = _grade_bhs(m_d)

    return {
        "ok": True,
        "n_samples": int(n),
        "n_evaluated": int(np.sum(mask)),
        "n_folds": folds,
        "degree": degree,
        "outliers_removed": outlier_report.get("n_removed", 0),
        "outlier_report": outlier_report,
        "systolic": m_s,
        "diastolic": m_d,
        "bhs_grade_systolic": grade_s,
        "bhs_grade_diastolic": grade_d,
        "interpretation": _interpret(m_s, m_d, grade_s, grade_d),
    }


def _grade_bhs(m: Dict) -> str:
    """
    Aproximação das classes BHS com base em % dentro de 5/10/15 mmHg.
    A: ≥60% ≤5, ≥85% ≤10, ≥95% ≤15
    B: ≥50, ≥75, ≥90
    C: ≥40, ≥65, ≥85
    D: pior
    """
    p5, p10, p15 = m.get("pct_within_5", 0), m.get("pct_within_10", 0), m.get("pct_within_15", 0)
    if p5 >= 60 and p10 >= 85 and p15 >= 95:
        return "A"
    if p5 >= 50 and p10 >= 75 and p15 >= 90:
        return "B"
    if p5 >= 40 and p10 >= 65 and p15 >= 85:
        return "C"
    return "D"


def _interpret(m_s, m_d, g_s, g_d) -> str:
    parts = [
        f"SBP: MAE {m_s['mae']} mmHg, bias {m_s['bias']:+} (grau BHS≈{g_s}).",
        f"DBP: MAE {m_d['mae']} mmHg, bias {m_d['bias']:+} (grau BHS≈{g_d}).",
    ]
    if g_s in ("A", "B") and g_d in ("A", "B"):
        parts.append("Precisão promissora para protótipo experimental — ainda não equivale a dispositivo clínico validado.")
    elif g_s == "D" or g_d == "D":
        parts.append("Precisão insuficiente: colete mais calibrações diversas e revise outliers/qualidade do sinal.")
    else:
        parts.append("Precisão intermediária: o modelo global ajuda, mas continue calibrando e filtrando outliers.")
    return " ".join(parts)


def residual_outliers_with_model(
    calibrations: list,
    coeffs: Optional[Dict],
    residual_z: float = 3.0,
) -> Dict[str, Any]:
    """Pontos cujo residual |pred − ref| é extremo em relação ao RMSE do conjunto."""
    if not coeffs:
        return {"ok": False, "error": "Modelo ausente."}

    X, y_s, y_d, _ = _build_feature_matrix(calibrations, min_samples=1)
    if X is None:
        return {"ok": False, "error": "Sem dados."}

    errs_s, errs_d, rows = [], [], []
    for i in range(len(X)):
        feats = {
            "heart_rate": X[i, 0],
            "peak_amplitude": X[i, 1],
            "std_red": X[i, 2],
            "mean_red": X[i, 3],
            "pulse_width": X[i, 4],
        }
        ps, pd, st, _, _ = estimate_blood_pressure(feats, coeffs)
        if st != "calibrado" or ps is None:
            continue
        es, ed = float(ps - y_s[i]), float(pd - y_d[i])
        errs_s.append(es)
        errs_d.append(ed)
        rows.append({
            "index": i,
            "ref_systolic": float(y_s[i]),
            "ref_diastolic": float(y_d[i]),
            "pred_systolic": float(ps),
            "pred_diastolic": float(pd),
            "err_systolic": round(es, 2),
            "err_diastolic": round(ed, 2),
        })

    if len(errs_s) < 5:
        return {"ok": False, "error": "Poucos resíduos."}

    es = np.array(errs_s)
    ed = np.array(errs_d)
    zs = np.abs((es - es.mean()) / (es.std() + 1e-9))
    zd = np.abs((ed - ed.mean()) / (ed.std() + 1e-9))

    flagged = []
    for i, row in enumerate(rows):
        if zs[i] >= residual_z or zd[i] >= residual_z:
            row["z_systolic"] = round(float(zs[i]), 2)
            row["z_diastolic"] = round(float(zd[i]), 2)
            flagged.append(row)

    return {
        "ok": True,
        "n": len(rows),
        "n_flagged": len(flagged),
        "residual_z": residual_z,
        "flagged": flagged,
        "rmse_systolic": round(float(np.sqrt(np.mean(es ** 2))), 2),
        "rmse_diastolic": round(float(np.sqrt(np.mean(ed ** 2))), 2),
    }
