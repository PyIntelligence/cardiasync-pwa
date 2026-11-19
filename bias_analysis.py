"""
Análise de viés demográfico em PPG / estimativa de pressão.

Agrupa calibrações por sexo, faixa etária e tom de pele (quando informados),
calcula erro médio (bias), MAE e desvio padrão do residual por grupo.

Importante:
- skin_tone aqui é um proxy grosseiro e auto-declarado — não substitui
  classificação científica de fototipo.
- Com poucos dados por grupo os números são instáveis; o relatório indica n.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from calibration_engine import estimate_blood_pressure, _g


def _age_bucket(age: Optional[int]) -> str:
    if age is None:
        return "não informado"
    try:
        a = int(age)
    except (TypeError, ValueError):
        return "não informado"
    if a < 30:
        return "18–29"
    if a < 45:
        return "30–44"
    if a < 60:
        return "45–59"
    return "60+"


def _norm_label(value: Optional[str], fallback: str = "não informado") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback


def _group_stats(errors: List[float]) -> Dict[str, Any]:
    if not errors:
        return {"n": 0, "bias": None, "mae": None, "std": None}
    arr = np.array(errors, dtype=float)
    return {
        "n": int(len(arr)),
        "bias": round(float(np.mean(arr)), 2),       # média do (pred - real); >0 superestima
        "mae": round(float(np.mean(np.abs(arr))), 2),
        "std": round(float(np.std(arr)), 2),
    }


def analyze_demographic_bias(
    calibrations: list,
    coeffs: Optional[Dict],
) -> Dict[str, Any]:
    """
    Para cada calibração com referência, estima PA com o modelo e agrega erros
    por sexo, faixa etária e tom de pele.
    """
    report: Dict[str, Any] = {
        "ok": True,
        "model_available": coeffs is not None,
        "total_used": 0,
        "overall": {},
        "by_sex": {},
        "by_age_group": {},
        "by_skin_tone": {},
        "warnings": [],
        "notes": [
            "Bias = média (estimado − referência). Positivo = modelo superestima.",
            "MAE = erro absoluto médio em mmHg.",
            "Grupos com n < 5 são pouco confiáveis.",
            "Tom de pele é auto-declarado e apenas um proxy grosseiro.",
        ],
    }

    if coeffs is None:
        report["ok"] = False
        report["warnings"].append("Nenhum modelo ativo para calcular resíduos.")
        return report

    err_s_all: List[float] = []
    err_d_all: List[float] = []
    buckets = {
        "sex": defaultdict(lambda: {"s": [], "d": []}),
        "age": defaultdict(lambda: {"s": [], "d": []}),
        "skin": defaultdict(lambda: {"s": [], "d": []}),
    }

    for c in calibrations:
        ref_s = _g(c, "ref_systolic")
        ref_d = _g(c, "ref_diastolic")
        hr = _g(c, "heart_rate")
        q = _g(c, "signal_quality")
        if ref_s is None or ref_d is None or hr is None:
            continue
        if q is not None and q < 0.3:
            continue

        feats = {
            "heart_rate": hr,
            "peak_amplitude": _g(c, "peak_amplitude") or 0.0,
            "std_red": _g(c, "std_red") or 0.0,
            "mean_red": _g(c, "mean_red") or 0.0,
            "pulse_width": _g(c, "pulse_width") or 0.0,
        }
        pred_s, pred_d, status, _, _ = estimate_blood_pressure(feats, coeffs)
        if status != "calibrado" or pred_s is None:
            continue

        es = float(pred_s) - float(ref_s)
        ed = float(pred_d) - float(ref_d)
        err_s_all.append(es)
        err_d_all.append(ed)

        sex = _norm_label(_g(c, "sex"))
        age_g = _age_bucket(_g(c, "age"))
        skin = _norm_label(_g(c, "skin_tone"))

        buckets["sex"][sex]["s"].append(es)
        buckets["sex"][sex]["d"].append(ed)
        buckets["age"][age_g]["s"].append(es)
        buckets["age"][age_g]["d"].append(ed)
        buckets["skin"][skin]["s"].append(es)
        buckets["skin"][skin]["d"].append(ed)

    report["total_used"] = len(err_s_all)
    report["overall"] = {
        "systolic": _group_stats(err_s_all),
        "diastolic": _group_stats(err_d_all),
    }

    def pack(group_dict):
        out = {}
        for label, ed in group_dict.items():
            out[label] = {
                "systolic": _group_stats(ed["s"]),
                "diastolic": _group_stats(ed["d"]),
            }
        return out

    report["by_sex"] = pack(buckets["sex"])
    report["by_age_group"] = pack(buckets["age"])
    report["by_skin_tone"] = pack(buckets["skin"])

    # Alertas de viés grosseiro
    for dim_name, dim in [
        ("sexo", report["by_sex"]),
        ("faixa etária", report["by_age_group"]),
        ("tom de pele", report["by_skin_tone"]),
    ]:
        biases = []
        for label, stats in dim.items():
            n = stats["systolic"]["n"]
            b = stats["systolic"]["bias"]
            if n >= 5 and b is not None:
                biases.append((label, b, n))
        if len(biases) >= 2:
            biases.sort(key=lambda x: x[1])
            lo, hi = biases[0], biases[-1]
            gap = abs(hi[1] - lo[1])
            if gap >= 5:
                report["warnings"].append(
                    f"Possível viés por {dim_name}: diferença de bias sistólico "
                    f"≈ {gap:.1f} mmHg entre '{lo[0]}' ({lo[1]:+.1f}, n={lo[2]}) "
                    f"e '{hi[0]}' ({hi[1]:+.1f}, n={hi[2]})."
                )

    if report["total_used"] < 12:
        report["warnings"].append(
            "Poucas calibrações válidas para análise de viés estável (recomendado ≥ 30)."
        )

    # Cobertura demográfica
    missing = []
    if sum(1 for v in report["by_sex"].values() if v["systolic"]["n"] > 0 and "não informado" not in str(v)) == 0:
        pass
    known_sex = sum(v["systolic"]["n"] for k, v in report["by_sex"].items() if k != "não informado")
    known_age = sum(v["systolic"]["n"] for k, v in report["by_age_group"].items() if k != "não informado")
    known_skin = sum(v["systolic"]["n"] for k, v in report["by_skin_tone"].items() if k != "não informado")
    if known_sex < report["total_used"] * 0.5:
        missing.append("sexo")
    if known_age < report["total_used"] * 0.5:
        missing.append("idade")
    if known_skin < report["total_used"] * 0.5:
        missing.append("tom de pele")
    if missing:
        report["warnings"].append(
            "Dados demográficos incompletos (" + ", ".join(missing) +
            "). Peça preenchimento na calibração para análise de viés mais útil."
        )

    return report
