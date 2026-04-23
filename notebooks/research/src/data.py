"""Загрузка и подготовка данных S&P 500 (^GSPC) для GARCH/GAN экспериментов."""
from __future__ import annotations

from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2].parent / "data" / "research"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PRICES_CSV = DATA_DIR / "sp500_prices.csv"
RETURNS_CSV = DATA_DIR / "sp500_log_returns.csv"

TRAIN_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2024-12-31"


def download_sp500(start: str = "2010-01-01", end: str | None = None) -> pd.DataFrame:
    """Скачивает дневные цены закрытия S&P 500 через yfinance.

    Возвращает DataFrame с одним столбцом 'close' и DatetimeIndex.
    """
    import yfinance as yf

    if end is None:
        end = date.today().isoformat()

    df = yf.download("^GSPC", start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("yfinance вернул пустой DataFrame")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    out = df[["Close"]].copy()
    out.columns = ["close"]
    out.index.name = "date"
    out.index = pd.to_datetime(out.index)
    out = out.dropna()
    return out


def compute_log_returns(prices: pd.DataFrame) -> pd.Series:
    r = np.log(prices["close"]).diff().dropna()
    r.name = "log_return"
    return r


def load_or_download(force: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    if not force and PRICES_CSV.exists() and RETURNS_CSV.exists():
        prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
        returns = pd.read_csv(RETURNS_CSV, index_col=0, parse_dates=True)["log_return"]
        return prices, returns

    prices = download_sp500()
    returns = compute_log_returns(prices)
    prices.to_csv(PRICES_CSV)
    returns.to_frame().to_csv(RETURNS_CSV)
    return prices, returns


def train_test_split(
    series: pd.Series,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
) -> tuple[pd.Series, pd.Series]:
    train = series.loc[:train_end].copy()
    test = series.loc[test_start:test_end].copy()
    return train, test


if __name__ == "__main__":
    prices, returns = load_or_download(force=True)
    train, test = train_test_split(returns)
    print(f"prices: {len(prices)} rows, {prices.index.min().date()} -> {prices.index.max().date()}")
    print(f"returns: {len(returns)} rows")
    print(f"train: {len(train)} ({train.index.min().date()} -> {train.index.max().date()})")
    print(f"test:  {len(test)} ({test.index.min().date()} -> {test.index.max().date()})")
    print(f"saved to: {DATA_DIR}")
