"""
Motor de calibração PPG:
- Features (picos + FFT)
- Ridge linear (pessoal)
- Regressão polinomial global (grau 2, Ridge)
- Estimativa com intervalo (± residual_std)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

FEATURE_NAMES = ["hr", "amplitude", "std", "mean_red", "pulse_width"]
MIN_PERSONAL = 5
MIN_GLOBAL = 12  # mínimo para modelo populacional polinomial


def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return signal
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def extract_basic_features(red_signal: list, fps: float = 30.0) -> Dict:
    empty = {
        "heart_rate": None,
        "signal_quality": 0.0,
        "mean_red": 0.0,
        "std_red": 0.0,
        "peak_amplitude": 0.0,
        "pulse_width": 0.0,
        "snr_approx": 0.0,
        "n_peaks": 0,
        "stability": 0.0,
    }

    if not red_signal or len(red_signal) < int(fps * 3):
        return empty

    signal = np.asarray(red_signal, dtype=float)
    mean_red = float(np.mean(signal))
    std_red = float(np.std(signal))

    if std_red < 0.4:
        empty["mean_red"] = round(mean_red, 2)
        return empty

    long_w = max(5, int(fps * 1.0))
    trend = _smooth(signal, long_w)
    detrended = signal - trend
    short_w = max(3, int(fps * 0.06))
    centered = _smooth(detrended, short_w)

    amp = float(np.std(centered))
    if amp < 1e-6:
        empty.update({"mean_red": round(mean_red, 2), "std_red": round(std_red, 2)})
        return empty

    abs_max = float(np.max(np.abs(centered)))
    threshold = 0.30 * abs_max
    min_distance = max(3, int(fps * 0.35))

    peaks = []
    last = -min_distance
    for i in range(1, len(centered) - 1):
        if (
            centered[i] > centered[i - 1]
            and centered[i] > centered[i + 1]
            and centered[i] > threshold
            and (i - last) >= min_distance
        ):
            peaks.append(i)
            last = i

    heart_rate = None
    peak_amplitude = 0.0
    pulse_width = 0.0
    stability = 0.0

    if len(peaks) >= 3:
        intervals = np.diff(peaks) / fps
        med = float(np.median(intervals))
        good = intervals[(intervals > med * 0.55) & (intervals < med * 1.55)]
        if len(good) >= 2:
            mean_iv = float(np.mean(good))
            if 0.38 <= mean_iv <= 1.6:
                heart_rate = 60.0 / mean_iv
            peak_amplitude = float(np.mean([centered[p] for p in peaks]))
            pulse_width = mean_iv * 0.32
            cv = float(np.std(good) / (np.mean(good) + 1e-9))
            stability = max(0.0, min(1.0, 1.0 - cv * 2.2))

    n = len(centered)
    if n >= int(fps * 4):
        windowed = centered * np.hanning(n)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        mask = (freqs >= 0.7) & (freqs <= 3.0)
        if np.any(mask):
            idx = np.argmax(spectrum[mask])
            hr_fft = float(freqs[mask][idx]) * 60.0
            if heart_rate is None and 42 <= hr_fft <= 180:
                heart_rate = hr_fft
            elif heart_rate is not None and abs(heart_rate - hr_fft) < 15:
                heart_rate = 0.6 * heart_rate + 0.4 * hr_fft

    residual = signal - _smooth(signal, 5)
    noise_std = float(np.std(residual)) + 1e-9
    snr_approx = float(amp / noise_std)

    quality = 0.0
    if 35 < mean_red < 200:
        quality += 0.15
    elif 20 < mean_red < 230:
        quality += 0.08
    if std_red > 4.0:
        quality += 0.15
    elif std_red > 2.0:
        quality += 0.08
    if heart_rate is not None and 45 <= heart_rate <= 160:
        quality += 0.25
    elif heart_rate is not None:
        quality += 0.10
    if len(peaks) >= 6:
        quality += 0.15
    elif len(peaks) >= 3:
        quality += 0.08
    quality += 0.15 * stability
    if snr_approx > 3.0:
        quality += 0.15
    elif snr_approx > 1.5:
        quality += 0.08
    quality = round(min(1.0, quality), 3)

    return {
        "heart_rate": round(float(heart_rate), 1) if heart_rate else None,
        "signal_quality": quality,
        "mean_red": round(mean_red, 2),
        "std_red": round(std_red, 2),
        "peak_amplitude": round(peak_amplitude, 2),
        "pulse_width": round(pulse_width, 3),
        "snr_approx": round(snr_approx, 2),
        "n_peaks": len(peaks),
        "stability": round(stability, 3),
        # índices de pico para HRV/PRV (não serializar em massa no cliente se não precisar)
        "peak_indices": peaks,
    }


def _g(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _build_feature_matrix(calibrations: list, min_samples: int = 5):
    rows, sbp, dbp, meta = [], [], [], []
    for c in calibrations:
        hr = _g(c, "heart_rate")
        amp = _g(c, "peak_amplitude")
        std = _g(c, "std_red")
        mean_r = _g(c, "mean_red")
        pw = _g(c, "pulse_width")
        ref_s = _g(c, "ref_systolic")
        ref_d = _g(c, "ref_diastolic")
        q = _g(c, "signal_quality")

        if None in (hr, amp, std, ref_s, ref_d):
            continue
        if q is not None and q < 0.3:
            continue

        rows.append([float(hr), float(amp), float(std), float(mean_r or 0), float(pw or 0)])
        sbp.append(float(ref_s))
        dbp.append(float(ref_d))
        meta.append({
            "age": _g(c, "age"),
            "sex": _g(c, "sex"),
            "skin_tone": _g(c, "skin_tone"),
            "condition": _g(c, "condition"),
        })

    if len(rows) < min_samples:
        return None, None, None, None
    return np.array(rows), np.array(sbp), np.array(dbp), meta


def train_simple_linear_model(calibrations: list) -> Optional[Dict]:
    """Ridge linear pessoal (≥ 5 amostras)."""
    if not HAS_SKLEARN:
        return None
    X, y_sbp, y_dbp, _ = _build_feature_matrix(calibrations, MIN_PERSONAL)
    if X is None:
        return None

    model_s = Ridge(alpha=1.0)
    model_d = Ridge(alpha=1.0)
    model_s.fit(X, y_sbp)
    model_d.fit(X, y_dbp)
    pred_s = model_s.predict(X)
    pred_d = model_d.predict(X)

    return {
        "feature_names": FEATURE_NAMES,
        "model_type": "ridge",
        "degree": 1,
        "scope": "personal",
        "systolic": {
            "coef": [round(float(c), 6) for c in model_s.coef_],
            "intercept": round(float(model_s.intercept_), 4),
        },
        "diastolic": {
            "coef": [round(float(c), 6) for c in model_d.coef_],
            "intercept": round(float(model_d.intercept_), 4),
        },
        "r2_systolic": round(float(model_s.score(X, y_sbp)), 4),
        "r2_diastolic": round(float(model_d.score(X, y_dbp)), 4),
        "residual_std_sbp": round(float(np.std(y_sbp - pred_s)), 3),
        "residual_std_dbp": round(float(np.std(y_dbp - pred_d)), 3),
        "n_samples": int(len(X)),
    }


def train_polynomial_global_model(
    calibrations: list,
    degree: int = 2,
    min_samples: int = MIN_GLOBAL,
) -> Optional[Dict]:
    """
    Regressão polinomial global (Pipeline: StandardScaler → PolynomialFeatures → Ridge).

    Coeficientes são salvos no espaço das features polinomiais expandido.
    Para prever, reconstrói o mesmo pipeline com os parâmetros guardados.
    """
    if not HAS_SKLEARN:
        return None

    X, y_sbp, y_dbp, meta = _build_feature_matrix(calibrations, min_samples)
    if X is None:
        return None

    degree = max(1, min(3, int(degree)))  # limita grau 1–3

    def fit_one(y):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("ridge", Ridge(alpha=2.0)),
        ])
        pipe.fit(X, y)
        pred = pipe.predict(X)
        r2 = float(pipe.score(X, y))
        resid_std = float(np.std(y - pred))
        ridge = pipe.named_steps["ridge"]
        scaler = pipe.named_steps["scaler"]
        return {
            "coef": [round(float(c), 8) for c in ridge.coef_],
            "intercept": round(float(ridge.intercept_), 4),
            "scaler_mean": [round(float(m), 6) for m in scaler.mean_],
            "scaler_scale": [round(float(s), 6) for s in scaler.scale_],
            "r2": round(r2, 4),
            "residual_std": round(resid_std, 3),
        }

    s_pack = fit_one(y_sbp)
    d_pack = fit_one(y_dbp)

    # Nomes das features polinomiais (para auditoria)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly.fit(X)
    try:
        poly_names = poly.get_feature_names_out(FEATURE_NAMES).tolist()
    except Exception:
        poly_names = [f"f{i}" for i in range(len(s_pack["coef"]))]

    return {
        "feature_names": FEATURE_NAMES,
        "poly_feature_names": poly_names,
        "model_type": "polynomial",
        "degree": degree,
        "scope": "global",
        "systolic": {
            "coef": s_pack["coef"],
            "intercept": s_pack["intercept"],
            "scaler_mean": s_pack["scaler_mean"],
            "scaler_scale": s_pack["scaler_scale"],
        },
        "diastolic": {
            "coef": d_pack["coef"],
            "intercept": d_pack["intercept"],
            "scaler_mean": d_pack["scaler_mean"],
            "scaler_scale": d_pack["scaler_scale"],
        },
        "r2_systolic": s_pack["r2"],
        "r2_diastolic": d_pack["r2"],
        "residual_std_sbp": s_pack["residual_std"],
        "residual_std_dbp": d_pack["residual_std"],
        "n_samples": int(len(X)),
    }


def _predict_poly(pack: Dict, x_raw: np.ndarray, degree: int) -> float:
    """Aplica scaler + polynomial features + coeficientes salvos."""
    mean = np.array(pack["scaler_mean"], dtype=float)
    scale = np.array(pack["scaler_scale"], dtype=float)
    scale = np.where(scale < 1e-9, 1.0, scale)
    x_std = (x_raw - mean) / scale

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    # fit com shape correto
    poly.fit(np.zeros((1, len(x_std))))
    x_poly = poly.transform(x_std.reshape(1, -1))[0]

    coef = np.array(pack["coef"], dtype=float)
    if len(coef) != len(x_poly):
        # fallback: corta ou preenche
        n = min(len(coef), len(x_poly))
        return float(np.dot(coef[:n], x_poly[:n]) + pack["intercept"])
    return float(np.dot(coef, x_poly) + pack["intercept"])


def estimate_blood_pressure(
    features: Dict,
    calibration_coeffs: Optional[Dict] = None,
) -> Tuple[Optional[float], Optional[float], str, Optional[float], Optional[float]]:
    """
    Retorna: sbp, dbp, status, sbp_std, dbp_std
    """
    if calibration_coeffs is None:
        return None, None, "sem_calibracao", None, None

    try:
        names = calibration_coeffs.get("feature_names", FEATURE_NAMES)
        mapping = {
            "hr": features.get("heart_rate") or 70.0,
            "amplitude": features.get("peak_amplitude") or 0.0,
            "std": features.get("std_red") or 0.0,
            "mean_red": features.get("mean_red") or 100.0,
            "pulse_width": features.get("pulse_width") or 0.3,
        }
        x = np.array([mapping.get(n, 0.0) for n in names], dtype=float)

        model_type = calibration_coeffs.get("model_type", "ridge")
        degree = int(calibration_coeffs.get("degree", 1))

        if model_type == "polynomial" and degree >= 2:
            sbp = _predict_poly(calibration_coeffs["systolic"], x, degree)
            dbp = _predict_poly(calibration_coeffs["diastolic"], x, degree)
        else:
            s = calibration_coeffs["systolic"]
            d = calibration_coeffs["diastolic"]
            sbp = float(np.dot(s["coef"], x) + s["intercept"])
            dbp = float(np.dot(d["coef"], x) + d["intercept"])

        sbp = max(70.0, min(220.0, sbp))
        dbp = max(40.0, min(130.0, dbp))
        if dbp >= sbp:
            dbp = sbp - 20.0

        s_std = calibration_coeffs.get("residual_std_sbp")
        d_std = calibration_coeffs.get("residual_std_dbp")

        return round(sbp), round(dbp), "calibrado", s_std, d_std
    except Exception:
        return None, None, "erro_modelo", None, None


def compute_mae_on_calibrations(calibrations: list, coeffs: Dict) -> Dict:
    """MAE global do modelo sobre o conjunto de calibração."""
    X, y_s, y_d, _ = _build_feature_matrix(calibrations, min_samples=1)
    if X is None:
        return {}
    errs_s, errs_d = [], []
    for i in range(len(X)):
        feats = {
            "heart_rate": X[i, 0],
            "peak_amplitude": X[i, 1],
            "std_red": X[i, 2],
            "mean_red": X[i, 3],
            "pulse_width": X[i, 4],
        }
        ps, pd, st, _, _ = estimate_blood_pressure(feats, coeffs)
        if st == "calibrado" and ps is not None:
            errs_s.append(abs(ps - y_s[i]))
            errs_d.append(abs(pd - y_d[i]))
    if not errs_s:
        return {}
    return {
        "mae_systolic": round(float(np.mean(errs_s)), 2),
        "mae_diastolic": round(float(np.mean(errs_d)), 2),
        "n": len(errs_s),
    }
