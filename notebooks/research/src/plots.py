"""Графики для отчёта."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ART = Path(__file__).resolve().parents[1] / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _autocorr(x: np.ndarray, lags: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.sum(x * x)
    out = np.empty(lags + 1)
    out[0] = 1.0
    for k in range(1, lags + 1):
        out[k] = float(np.sum(x[:-k] * x[k:]) / denom) if denom > 0 else 0.0
    return out


def fig_returns_overview(returns: pd.Series, train_end="2019-12-31") -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    r_pct = returns * 100
    axes[0].plot(r_pct.index, r_pct.values, lw=0.6, color="#1f4e79")
    axes[0].axvline(pd.Timestamp(train_end), color="red", ls="--", alpha=0.6, label="train/test split")
    axes[0].set_ylabel("Доходность, %")
    axes[0].set_title("S&P 500 — дневные log-доходности (2010–2024)")
    axes[0].legend()
    axes[1].plot(r_pct.index, r_pct.values.cumsum(), lw=1.0, color="#2e7d32")
    axes[1].axvline(pd.Timestamp(train_end), color="red", ls="--", alpha=0.6)
    axes[1].set_ylabel("Накопл. лог-доход, %")
    axes[1].set_xlabel("Дата")
    fig.tight_layout()
    p = FIG / "01_returns_overview.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_real_vs_synth_series(real_pct, synth_tg, synth_ct) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, x, name, c in zip(
        axes,
        [real_pct, synth_tg, synth_ct],
        ["Real (train, 2010–2019)", "TimeGAN-synth", "CTGAN-synth"],
        ["#1f4e79", "#c0504d", "#7030a0"],
    ):
        ax.plot(x, lw=0.6, color=c)
        ax.set_ylabel("r, %")
        ax.set_title(name, loc="left", fontsize=10)
    axes[-1].set_xlabel("Шаг (день)")
    fig.suptitle("Реальные vs синтетические доходности", y=1.0)
    fig.tight_layout()
    p = FIG / "02_real_vs_synth_series.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_distributions(real_pct, synth_tg, synth_ct) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-7, 7, 80)
    ax.hist(real_pct, bins=bins, density=True, alpha=0.5, color="#1f4e79", label="Real train")
    ax.hist(synth_tg, bins=bins, density=True, alpha=0.5, color="#c0504d", label="TimeGAN")
    ax.hist(synth_ct, bins=bins, density=True, alpha=0.5, color="#7030a0", label="CTGAN")
    ax.set_xlabel("Доходность, %")
    ax.set_ylabel("Плотность")
    ax.set_title("Распределения дневных доходностей: real vs synth")
    ax.legend()
    fig.tight_layout()
    p = FIG / "03_distributions.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_acf_compare(real_pct, synth_tg, synth_ct, lags=20) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for col, fn_name, label in [
        (0, lambda x: x, "ACF доходностей"),
        (1, lambda x: np.abs(x), "ACF |доходностей|  (волатильная кластеризация)"),
    ]:
        ax = axes[col]
        for x, name, c in [
            (real_pct, "Real", "#1f4e79"),
            (synth_tg, "TimeGAN", "#c0504d"),
            (synth_ct, "CTGAN", "#7030a0"),
        ]:
            transformed = np.abs(x) if col == 1 else np.asarray(x)
            a = _autocorr(transformed, lags=lags)
            ax.plot(range(lags + 1), a, "o-", ms=3, label=name, color=c)
        ax.axhline(0, color="black", lw=0.5)
        n = len(real_pct)
        ci = 1.96 / np.sqrt(n)
        ax.axhspan(-ci, ci, color="grey", alpha=0.15, label="95% CI ≈ ±1.96/√N")
        ax.set_title(label)
        ax.set_xlabel("lag"); ax.set_ylabel("ACF")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "04_acf_compare.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def _load_forecast(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def fig_volatility_forecasts(test_returns_pct: pd.Series) -> Path:
    branches = {
        "B1 (real)":      ("forecasts_B1.csv",         "#1f4e79"),
        "B2-TimeGAN":     ("forecasts_B2_timegan.csv", "#c0504d"),
        "B2-CTGAN":       ("forecasts_B2_ctgan.csv",   "#7030a0"),
        "B3 (augmented)": ("forecasts_B3.csv",         "#2e7d32"),
    }
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.bar(test_returns_pct.index, np.abs(test_returns_pct.values),
           color="lightgrey", width=1.0, label="|r_t| (proxy)")
    for name, (fname, c) in branches.items():
        df = _load_forecast(ART / fname)
        ax.plot(df.index, df["sigma_pct"].values, lw=1.0, label=name, color=c)
    ax.set_ylabel("σ̂_t, %"); ax.set_xlabel("Дата")
    ax.set_title("Прогноз условной волатильности на test (2020–2024)")
    ax.legend(ncol=5, fontsize=8, loc="upper right")
    fig.tight_layout()
    p = FIG / "05_volatility_forecasts.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_var_violations(test_returns_pct: pd.Series, alpha: float = 0.05) -> Path:
    branches = {
        "B1 (real)":      ("forecasts_B1.csv",         "#1f4e79"),
        "B2-TimeGAN":     ("forecasts_B2_timegan.csv", "#c0504d"),
        "B2-CTGAN":       ("forecasts_B2_ctgan.csv",   "#7030a0"),
        "B3 (augmented)": ("forecasts_B3.csv",         "#2e7d32"),
    }
    fig, axes = plt.subplots(len(branches), 1, figsize=(11, 8), sharex=True)
    col = f"VaR_{int(alpha*100)}"
    for ax, (name, (fname, c)) in zip(axes, branches.items()):
        df = _load_forecast(ART / fname)
        r = df["r_pct"].values
        var = df[col].values
        viol = r < var
        ax.plot(df.index, r, color="lightgrey", lw=0.6, label="r_t")
        ax.plot(df.index, var, color=c, lw=1.0, label=f"VaR({int(alpha*100)}%)")
        ax.scatter(df.index[viol], r[viol], color="red", s=10, zorder=3,
                   label=f"violation (n={viol.sum()})")
        ax.set_ylabel("%"); ax.set_title(name, loc="left", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
    axes[-1].set_xlabel("Дата")
    fig.suptitle(f"VaR({int(alpha*100)}%) one-step-ahead — прогноз и нарушения (test)", y=1.0)
    fig.tight_layout()
    p = FIG / f"06_var_violations_{int(alpha*100)}.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_metrics_bars() -> Path:
    summary = pd.read_csv(ART / "metrics_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    palette = {"B1_real": "#1f4e79", "B2_timegan": "#c0504d",
               "B2_ctgan": "#7030a0", "B3_aug": "#2e7d32"}
    colors = [palette[b] for b in summary["branch"]]

    axes[0].bar(summary["branch"], summary["RMSE_abs_r"], color=colors)
    axes[0].set_title("RMSE( σ̂, |r| ) — меньше лучше")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(summary["branch"], summary["QLIKE"], color=colors)
    axes[1].set_title("QLIKE — меньше лучше")
    axes[1].tick_params(axis="x", rotation=20)

    width = 0.35
    x = np.arange(len(summary))
    axes[2].bar(x - width/2, summary["VaR5_rate"]*100, width=width, label="VaR 5% rate",
                color=[palette[b] for b in summary["branch"]], alpha=0.6)
    axes[2].bar(x + width/2, summary["VaR1_rate"]*100, width=width, label="VaR 1% rate",
                color=[palette[b] for b in summary["branch"]])
    axes[2].axhline(5, color="red", ls="--", lw=0.8, label="5% / 1% target")
    axes[2].axhline(1, color="red", ls="--", lw=0.8)
    axes[2].set_xticks(x); axes[2].set_xticklabels(summary["branch"], rotation=20)
    axes[2].set_ylabel("rate, %"); axes[2].set_title("Доли нарушений VaR")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    p = FIG / "07_metrics_bars.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def make_all(returns: pd.Series, train: pd.Series, test: pd.Series, SCALE: float = 100.0) -> dict:
    real_pct = train.values * SCALE
    synth_tg = pd.read_csv(ART / "synth_timegan.csv")["synth_pct"].values
    synth_ct = pd.read_csv(ART / "synth_ctgan.csv")["synth_pct"].values
    test_pct = test * SCALE

    paths = {
        "returns_overview": fig_returns_overview(returns),
        "real_vs_synth_series": fig_real_vs_synth_series(real_pct, synth_tg, synth_ct),
        "distributions": fig_distributions(real_pct, synth_tg, synth_ct),
        "acf_compare": fig_acf_compare(real_pct, synth_tg, synth_ct, lags=20),
        "volatility_forecasts": fig_volatility_forecasts(test_pct),
        "var5_violations": fig_var_violations(test_pct, alpha=0.05),
        "var1_violations": fig_var_violations(test_pct, alpha=0.01),
        "metrics_bars": fig_metrics_bars(),
    }
    return paths


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src import data as dm
    prices, returns = dm.load_or_download()
    train, test = dm.train_test_split(returns)
    paths = make_all(returns, train, test)
    for k, v in paths.items():
        print(f"  {k:25s} -> {v}")
