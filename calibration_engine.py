"""
Motor de calibração e estimativa de pressão arterial por PPG.

- Extração robusta de features do sinal (picos + FFT)
- Treino real de regressão Ridge (sklearn) quando ≥ 5 calibrações
- Estimativa de PA usando o modelo treinado
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    from sklearn.linear_model import Ridge
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _smooth(signal: np.ndarray, window: int) -> np.ndarray:
    if window < 3:
        return signal
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def extract_basic_features(red_signal: list, fps: float = 30.0) -> Dict:
    """
    Extrai features do sinal PPG com detecção de qualidade melhorada.
    Combina detecção de picos no tempo + estimativa por FFT.
    """
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

    # Remove tendência lenta e suaviza levemente
    long_w = max(5, int(fps * 1.0))
    trend = _smooth(signal, long_w)
    detrended = signal - trend
    short_w = max(3, int(fps * 0.06))
    centered = _smooth(detrended, short_w)

    amp = float(np.std(centered))
    if amp < 1e-6:
        empty.update({"mean_red": round(mean_red, 2), "std_red": round(std_red, 2)})
        return empty

    # ── Picos no domínio do tempo ──
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

    # ── Fallback / reforço por FFT ──
    n = len(centered)
    if n >= int(fps * 4):
        windowed = centered * np.hanning(n)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, d=1.0 / fps)
        # Faixa cardíaca 0.7–3.0 Hz (42–180 bpm)
        mask = (freqs >= 0.7) & (freqs <= 3.0)
        if np.any(mask):
            idx = np.argmax(spectrum[mask])
            peak_freq = float(freqs[mask][idx])
            hr_fft = peak_freq * 60.0
            if heart_rate is None and 42 <= hr_fft <= 180:
                heart_rate = hr_fft
            elif heart_rate is not None:
                # Consenso: média se próximos
                if abs(heart_rate - hr_fft) < 15:
                    heart_rate = 0.6 * heart_rate + 0.4 * hr_fft

    # SNR aproximado
    residual = signal - _smooth(signal, 5)
    noise_std = float(np.std(residual)) + 1e-9
    snr_approx = float(amp / noise_std)

    # ── Qualidade (0–1) ──
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
    }


def _build_feature_matrix(calibrations: list):
    rows, sbp, dbp = [], [], []
    for c in calibrations:
        def g(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        hr = g(c, "heart_rate")
        amp = g(c, "peak_amplitude")
        std = g(c, "std_red")
        mean_r = g(c, "mean_red")
        pw = g(c, "pulse_width")
        ref_s = g(c, "ref_systolic")
        ref_d = g(c, "ref_diastolic")
        q = g(c, "signal_quality")

        if None in (hr, amp, std, ref_s, ref_d):
            continue
        if q is not None and q < 0.3:
            continue

        rows.append([float(hr), float(amp), float(std), float(mean_r or 0), float(pw or 0)])
        sbp.append(float(ref_s))
        dbp.append(float(ref_d))

    if len(rows) < 5:
        return None, None, None
    return np.array(rows), np.array(sbp), np.array(dbp)


def train_simple_linear_model(calibrations: list) -> Optional[Dict]:
    """Treina Ridge quando há ≥ 5 calibrações válidas."""
    if not HAS_SKLEARN:
        return None

    X, y_sbp, y_dbp = _build_feature_matrix(calibrations)
    if X is None:
        return None

    model_s = Ridge(alpha=1.0)
    model_d = Ridge(alpha=1.0)
    model_s.fit(X, y_sbp)
    model_d.fit(X, y_dbp)

    return {
        "feature_names": ["hr", "amplitude", "std", "mean_red", "pulse_width"],
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
        "n_samples": int(len(X)),
        "model_type": "ridge",
    }


def estimate_blood_pressure(
    features: Dict,
    calibration_coeffs: Optional[Dict] = None,
) -> Tuple[Optional[float], Optional[float], str]:
    if calibration_coeffs is None:
        return None, None, "sem_calibracao"

    try:
        names = calibration_coeffs.get(
            "feature_names", ["hr", "amplitude", "std", "mean_red", "pulse_width"]
        )
        mapping = {
            "hr": features.get("heart_rate") or 70.0,
            "amplitude": features.get("peak_amplitude") or 0.0,
            "std": features.get("std_red") or 0.0,
            "mean_red": features.get("mean_red") or 100.0,
            "pulse_width": features.get("pulse_width") or 0.3,
        }
        x = np.array([mapping.get(n, 0.0) for n in names], dtype=float)

        s = calibration_coeffs["systolic"]
        d = calibration_coeffs["diastolic"]
        sbp = float(np.dot(s["coef"], x) + s["intercept"])
        dbp = float(np.dot(d["coef"], x) + d["intercept"])

        sbp = max(70.0, min(220.0, sbp))
        dbp = max(40.0, min(130.0, dbp))
        if dbp >= sbp:
            dbp = sbp - 20.0

        return round(sbp), round(dbp), "calibrado"
    except Exception:
        return None, None, "erro_modelo"
