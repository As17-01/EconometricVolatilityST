"""Проверка стилизованных фактов финансовых временных рядов."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


def stylized_facts(returns: pd.Series | np.ndarray, name: str = "series") -> dict:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"name": name, "n": 0}

    abs_r = np.abs(r)
    sq_r = r ** 2

    def acf(x, lag):
        x = x - x.mean()
        denom = np.sum(x * x)
        return float(np.sum(x[:-lag] * x[lag:]) / denom) if denom > 0 else float("nan")

    # Ljung-Box на автокорреляцию доходностей и квадратов
    lb_r = acorr_ljungbox(r, lags=[10], return_df=True)
    lb_sq = acorr_ljungbox(sq_r, lags=[10], return_df=True)

    return {
        "name": name,
        "n": int(len(r)),
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)),
        "skew": float(stats.skew(r)),
        "kurtosis_excess": float(stats.kurtosis(r)),  # excess
        "min": float(r.min()),
        "max": float(r.max()),
        "acf1_returns": acf(r, 1),
        "acf5_returns": acf(r, 5),
        "acf1_abs_returns": acf(abs_r, 1),
        "acf5_abs_returns": acf(abs_r, 5),
        "acf10_abs_returns": acf(abs_r, 10),
        "acf1_sq_returns": acf(sq_r, 1),
        "acf5_sq_returns": acf(sq_r, 5),
        "ljungbox_r_p": float(lb_r["lb_pvalue"].iloc[0]),
        "ljungbox_sq_p": float(lb_sq["lb_pvalue"].iloc[0]),
    }
