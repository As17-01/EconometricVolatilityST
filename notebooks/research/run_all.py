"""Главный оркестратор экспериментов.

Запускает 4 ветки:
    B1            : GARCH на реальном train, walk-forward на real test
    B2-TimeGAN    : TimeGAN на real train -> synth(2515), GARCH на synth, walk-forward на real test
    B2-CTGAN      : CTGAN на real train  -> synth(2515), GARCH на synth, walk-forward на real test
    B3            : GARCH на (synth_TimeGAN(2515) + real train), walk-forward на real test

Сохраняет всё в notebooks/research/artifacts/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SRC.parent))  # чтобы заработал `from src import ...`

from src import data as data_mod  # noqa: E402
from src import garch_eval as ge  # noqa: E402
from src import stylized as sf  # noqa: E402
from src import timegan as tg  # noqa: E402
from src import ctgan_wrapper as cgw  # noqa: E402

SCALE = ge.SCALE
SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_forecasts(df: pd.DataFrame, params: ge.GarchParams, path: Path) -> None:
    out = ge.var_forecast(df, params, alphas=(0.05, 0.01))
    out.to_csv(path)


def run() -> None:
    t0 = time.time()
    set_seed()

    # ---------- 1. Данные ----------
    prices, returns = data_mod.load_or_download(force=False)
    train, test = data_mod.train_test_split(returns)

    print(f"[DATA] train: {len(train)}  test: {len(test)}")
    print(f"       train: {train.index.min().date()} -> {train.index.max().date()}")
    print(f"       test : {test.index.min().date()} -> {test.index.max().date()}")

    # сразу сохраняем split-описание
    save_json(
        {
            "n_total": int(len(returns)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_start": str(train.index.min().date()),
            "train_end": str(train.index.max().date()),
            "test_start": str(test.index.min().date()),
            "test_end": str(test.index.max().date()),
        },
        ART / "split_info.json",
    )

    # стилизованные факты для real train (в процентах)
    sf_real = sf.stylized_facts(train.values * SCALE, name="real_train")

    # ---------- 2. B1 — Real-only baseline ----------
    print("\n[B1] Fitting GARCH(1,1)-t on real train ...")
    params_real, res_real = ge.fit_garch_const_mean(train, dist="t")
    print(f"    params: {params_real.to_dict()}")
    save_json(params_real.to_dict(), ART / "garch_params_real.json")

    df_b1 = ge.walk_forward_fixed(params_real, history=train, test=test)
    metrics_b1 = ge.evaluate_forecast(df_b1, params_real)
    save_forecasts(df_b1, params_real, ART / "forecasts_B1.csv")
    print(f"    B1 metrics: {metrics_b1}")

    # ---------- 3. TimeGAN -> синтетика для B2 и B3 ----------
    print("\n[B2-TimeGAN] Training TimeGAN on real train returns ...")
    cfg = tg.TimeGANConfig(
        seq_len=24,
        hidden_dim=24,
        num_layers=2,
        batch_size=128,
        ae_epochs=400,
        sup_epochs=300,
        joint_epochs=400,
        lr=1e-3,
    )
    print(f"    device: {cfg.device}")
    real_pct = train.values.astype(np.float32) * SCALE
    set_seed()
    synth_tg, hist_tg, model_tg = tg.fit_and_sample(
        real_pct, n_target=len(train), cfg=cfg, verbose=True
    )
    pd.Series(synth_tg, name="synth_pct").to_csv(ART / "synth_timegan.csv", index=False)
    save_json(hist_tg, ART / "timegan_history.json")
    torch.save(
        {
            "embedder": model_tg.embedder.state_dict(),
            "recovery": model_tg.recovery.state_dict(),
            "generator": model_tg.generator.state_dict(),
            "supervisor": model_tg.supervisor.state_dict(),
            "discriminator": model_tg.discriminator.state_dict(),
            "cfg": cfg.__dict__,
        },
        ART / "timegan_model.pt",
    )

    sf_tg = sf.stylized_facts(synth_tg, name="synth_timegan")

    # GARCH на синтетике (она уже в %, делим на SCALE чтоб однообразно)
    synth_tg_series = pd.Series(synth_tg / SCALE, index=train.index, name="synth")
    params_synth_tg, _ = ge.fit_garch_const_mean(synth_tg_series, dist="t")
    print(f"    GARCH on TimeGAN synth: {params_synth_tg.to_dict()}")
    save_json(params_synth_tg.to_dict(), ART / "garch_params_synth_timegan.json")

    df_b2tg = ge.walk_forward_fixed(params_synth_tg, history=train, test=test)
    metrics_b2tg = ge.evaluate_forecast(df_b2tg, params_synth_tg)
    save_forecasts(df_b2tg, params_synth_tg, ART / "forecasts_B2_timegan.csv")
    print(f"    B2-TimeGAN metrics: {metrics_b2tg}")

    # ---------- 4. CTGAN -> синтетика для B2-CTGAN ----------
    print("\n[B2-CTGAN] Training CTGAN on real train returns (windows) ...")
    set_seed()
    synth_ct, _ = cgw.fit_and_sample_ctgan(
        real_pct, n_target=len(train), seq_len=24, epochs=300, verbose=False
    )
    pd.Series(synth_ct, name="synth_pct").to_csv(ART / "synth_ctgan.csv", index=False)
    sf_ct = sf.stylized_facts(synth_ct, name="synth_ctgan")

    synth_ct_series = pd.Series(synth_ct / SCALE, index=train.index, name="synth")
    try:
        params_synth_ct, _ = ge.fit_garch_const_mean(synth_ct_series, dist="t")
    except Exception as e:
        print(f"    CTGAN GARCH-t failed ({e}); fallback to normal")
        params_synth_ct, _ = ge.fit_garch_const_mean(synth_ct_series, dist="normal")
    print(f"    GARCH on CTGAN synth: {params_synth_ct.to_dict()}")
    save_json(params_synth_ct.to_dict(), ART / "garch_params_synth_ctgan.json")

    df_b2ct = ge.walk_forward_fixed(params_synth_ct, history=train, test=test)
    metrics_b2ct = ge.evaluate_forecast(df_b2ct, params_synth_ct)
    save_forecasts(df_b2ct, params_synth_ct, ART / "forecasts_B2_ctgan.csv")
    print(f"    B2-CTGAN metrics: {metrics_b2ct}")

    # ---------- 5. B3 — Augmentation: synth_TimeGAN + real train ----------
    print("\n[B3] Augmentation: GARCH on (TimeGAN-synth + real train) ...")
    aug_values = np.concatenate([synth_tg / SCALE, train.values])
    aug_index = pd.date_range(end=train.index.max(), periods=len(aug_values), freq="B")
    aug_series = pd.Series(aug_values, index=aug_index, name="aug")

    params_aug, _ = ge.fit_garch_const_mean(aug_series, dist="t")
    print(f"    GARCH params (aug): {params_aug.to_dict()}")
    save_json(params_aug.to_dict(), ART / "garch_params_aug.json")

    df_b3 = ge.walk_forward_fixed(params_aug, history=train, test=test)
    metrics_b3 = ge.evaluate_forecast(df_b3, params_aug)
    save_forecasts(df_b3, params_aug, ART / "forecasts_B3.csv")
    print(f"    B3 metrics: {metrics_b3}")

    # ---------- 6. Сводки ----------
    rows = []
    for tag, m, p in [
        ("B1_real", metrics_b1, params_real),
        ("B2_timegan", metrics_b2tg, params_synth_tg),
        ("B2_ctgan", metrics_b2ct, params_synth_ct),
        ("B3_aug", metrics_b3, params_aug),
    ]:
        row = {"branch": tag, **p.to_dict(), **m}
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(ART / "metrics_summary.csv", index=False)
    print("\n[SUMMARY]")
    cols_show = [
        "branch", "alpha", "beta", "RMSE_abs_r", "QLIKE",
        "MZ_a", "MZ_b", "MZ_R2",
        "VaR5_violations", "VaR5_rate", "VaR5_kupiec_p",
        "VaR1_violations", "VaR1_rate", "VaR1_kupiec_p",
    ]
    with pd.option_context("display.float_format", "{:0.4f}".format, "display.width", 200):
        print(summary[cols_show].to_string(index=False))

    sf_table = pd.DataFrame([sf_real, sf_tg, sf_ct])
    sf_table.to_csv(ART / "stylized_facts.csv", index=False)
    print("\n[STYLIZED FACTS]")
    print(sf_table.to_string(index=False))

    print(f"\n[DONE] elapsed: {(time.time() - t0):.1f}s")


if __name__ == "__main__":
    run()
