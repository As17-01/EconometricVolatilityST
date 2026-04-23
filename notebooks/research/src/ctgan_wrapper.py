"""Обёртка над CTGAN для генерации финансового ряда из окон.

CTGAN изначально для табличных данных. Мы намеренно используем его как
КОНТРАСТ к TimeGAN: режем ряд на окна длины T, рассматриваем каждое окно
как строку из T непрерывных признаков, обучаем CTGAN, потом сэмплим окна
и склеиваем встык. Это должно ломать временную структуру (отсутствие
кластеров волатильности, отсутствие ACF в |r| и т.д.).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_and_sample_ctgan(
    series_real: np.ndarray,
    n_target: int,
    seq_len: int = 24,
    epochs: int = 300,
    verbose: bool = False,
) -> tuple[np.ndarray, "CTGAN"]:
    from ctgan import CTGAN

    s = np.asarray(series_real, dtype=np.float32).reshape(-1)

    # окна (M, T)
    idx = np.arange(0, len(s) - seq_len + 1)
    windows = np.stack([s[i : i + seq_len] for i in idx], axis=0)
    cols = [f"t{i}" for i in range(seq_len)]
    df = pd.DataFrame(windows, columns=cols)

    model = CTGAN(epochs=epochs, verbose=verbose)
    model.fit(df, discrete_columns=[])

    n_windows = int(np.ceil(n_target / seq_len)) + 1
    synth_df = model.sample(n_windows)
    synth = synth_df[cols].to_numpy(dtype=np.float32)

    flat = synth.reshape(-1)[:n_target]
    return flat, model
