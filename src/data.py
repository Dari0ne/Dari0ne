from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf


Frequency = Literal["daily", "weekly"]


@dataclass
class DataBundle:
    prices: pd.DataFrame
    returns: pd.DataFrame
    annualization_factor: int


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    frequency: Frequency,
) -> pd.DataFrame:
    interval = "1d" if frequency == "daily" else "1wk"
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError("Nessun dato scaricato da yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        key = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
        prices = raw[key].copy()
    else:
        key = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[key]].rename(columns={key: tickers[0]})

    prices = prices.sort_index().ffill().dropna(how="all")
    prices = prices.dropna(axis=1, how="all").ffill().dropna(how="any")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"Ticker mancanti nei dati scaricati: {missing}")
    return prices[tickers]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    if returns.empty:
        raise ValueError("Rendimenti vuoti: controlla intervallo date o ticker.")
    return returns


def load_data(config: dict) -> DataBundle:
    data_cfg = config["data"]
    ann_factor = (
        data_cfg.get("trading_days_daily", 252)
        if data_cfg["frequency"] == "daily"
        else data_cfg.get("trading_days_weekly", 52)
    )
    prices = download_prices(
        tickers=data_cfg["tickers"],
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        frequency=data_cfg["frequency"],
    )
    returns = compute_log_returns(prices)
    return DataBundle(prices=prices, returns=returns, annualization_factor=ann_factor)
