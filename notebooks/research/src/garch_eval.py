"""GARCH(1,1): обучение, fixed-parameter walk-forward, метрики прогноза волатильности."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

# Доходности приходят в долях (например, 0.01 = 1%). Внутри arch удобнее работать
# в процентах для устойчивости численной оптимизации.
SCALE = 100.0


@dataclass
class GarchParams:
    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float | None = None
    dist: str = "normal"

    def to_dict(self) -> dict:
        return asdict(self)


def fit_garch_const_mean(
    returns: pd.Series, dist: str = "normal"
) -> tuple[GarchParams, object]:
    """Обучает GARCH(1,1) с постоянным средним. Возвращает параметры и result-объект arch."""
    y = returns.dropna() * SCALE
    am = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist=dist)
    res = am.fit(disp="off")
    p = res.params
    nu = float(p["nu"]) if "nu" in p.index else None
    params = GarchParams(
        mu=float(p["mu"]),
        omega=float(p["omega"]),
        alpha=float(p["alpha[1]"]),
        beta=float(p["beta[1]"]),
        nu=nu,
        dist=dist,
    )
    return params, res


def _unconditional_var(p: GarchParams) -> float:
    persistence = p.alpha + p.beta
    if persistence < 0.999:
        return p.omega / (1.0 - persistence)
    return p.omega / 0.001


def walk_forward_fixed(
    params: GarchParams,
    history: pd.Series,
    test: pd.Series,
) -> pd.DataFrame:
    """One-step-ahead волатильность на test при зафиксированных параметрах GARCH.

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}, где ε_t = r_t - μ.
    История (history) используется только для прогрева состояния σ²; параметры заморожены.
    Все returns ВНУТРИ функции в процентах.
    """
    h = history.dropna() * SCALE
    t = test.dropna() * SCALE

    eps_hist = (h.values - params.mu)
    sigma2 = _unconditional_var(params)
    for e in eps_hist:
        sigma2 = params.omega + params.alpha * e * e + params.beta * sigma2

    sigmas = np.empty(len(t))
    eps_t = t.values - params.mu
    for i in range(len(t)):
        sigmas[i] = np.sqrt(sigma2)
        e = eps_t[i]
        sigma2 = params.omega + params.alpha * e * e + params.beta * sigma2

    out = pd.DataFrame(
        {
            "r_pct": t.values,
            "sigma_pct": sigmas,
        },
        index=t.index,
    )
    return out


def var_forecast(
    df: pd.DataFrame,
    params: GarchParams,
    alphas: Iterable[float] = (0.05, 0.01),
) -> pd.DataFrame:
    """Считает one-step VaR (отрицательное число — потенциальный убыток в %).

    Если dist='t' и есть nu, используем стандартизованное t-квантиль.
    """
    out = df.copy()
    for a in alphas:
        if params.dist == "t" and params.nu is not None and params.nu > 2:
            nu = params.nu
            q = stats.t.ppf(a, df=nu)
            std_q = q * np.sqrt((nu - 2) / nu)
        else:
            std_q = stats.norm.ppf(a)
        out[f"VaR_{int(a*100)}"] = params.mu + std_q * out["sigma_pct"]
    return out


def realized_proxy(returns_pct: pd.Series, window: int = 5) -> pd.Series:
    """Прокси реализованной волатильности: rolling std по абсолютным доходностям."""
    return returns_pct.rolling(window).std()


def kupiec_pof(violations: int, n: int, alpha: float) -> tuple[float, float]:
    """Kupiec PoF (Proportion of Failures) test.
    Возвращает (LR, p-value). H0: фактическая частота нарушений = alpha.
    """
    if n == 0:
        return float("nan"), float("nan")
    pi_hat = violations / n
    if pi_hat in (0.0, 1.0):
        # Логарифм 0 — берём предельное значение
        if pi_hat == 0.0:
            num = 0.0
        else:
            num = 0.0
        denom = violations * np.log(alpha) + (n - violations) * np.log(1 - alpha)
        lr = -2.0 * (denom - num)
    else:
        num = violations * np.log(pi_hat) + (n - violations) * np.log(1 - pi_hat)
        denom = violations * np.log(alpha) + (n - violations) * np.log(1 - alpha)
        lr = -2.0 * (denom - num)
    p_value = 1.0 - stats.chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


def evaluate_forecast(
    df: pd.DataFrame, params: GarchParams, alphas=(0.05, 0.01)
) -> dict:
    """Сводные метрики прогноза волатильности (df: r_pct, sigma_pct)."""
    r = df["r_pct"].values
    s = df["sigma_pct"].values
    s2 = s * s
    eps2 = (r - params.mu) ** 2

    rmse_proxy = float(np.sqrt(np.mean((np.abs(r) - s) ** 2)))
    mae_proxy = float(np.mean(np.abs(np.abs(r) - s)))

    # QLIKE: log(σ²) + r²/σ² (известна устойчивостью к шуму прокси r²)
    qlike = float(np.mean(np.log(s2) + eps2 / s2))

    # MZ-регрессия: r² = a + b·σ² + u, идеально a=0, b=1
    X = np.column_stack([np.ones_like(s2), s2])
    coef, *_ = np.linalg.lstsq(X, eps2, rcond=None)
    a_mz, b_mz = float(coef[0]), float(coef[1])
    yhat = X @ coef
    ss_res = float(np.sum((eps2 - yhat) ** 2))
    ss_tot = float(np.sum((eps2 - eps2.mean()) ** 2))
    r2_mz = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    var_res = {}
    df_var = var_forecast(df, params, alphas=alphas)
    for a in alphas:
        col = f"VaR_{int(a*100)}"
        violations = int((r < df_var[col].values).sum())
        n = len(r)
        rate = violations / n
        lr, p = kupiec_pof(violations, n, a)
        var_res[f"VaR{int(a*100)}_violations"] = violations
        var_res[f"VaR{int(a*100)}_rate"] = rate
        var_res[f"VaR{int(a*100)}_expected_rate"] = a
        var_res[f"VaR{int(a*100)}_kupiec_LR"] = lr
        var_res[f"VaR{int(a*100)}_kupiec_p"] = p

    return {
        "n": int(len(r)),
        "RMSE_abs_r": rmse_proxy,
        "MAE_abs_r": mae_proxy,
        "QLIKE": qlike,
        "MZ_a": a_mz,
        "MZ_b": b_mz,
        "MZ_R2": r2_mz,
        **var_res,
    }
