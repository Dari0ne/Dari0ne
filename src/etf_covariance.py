"""ETF covariance analysis script.

This module downloads five years of daily adjusted close prices for selected
ETFs, computes logarithmic returns, calculates an annualized covariance matrix,
and saves the result to ``data/covariance_matrix.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


TICKERS = ["SPY", "QQQ", "EFA", "IEMG"]
TRADING_DAYS_PER_YEAR = 252
OUTPUT_PATH = Path("data/covariance_matrix.csv")


def download_adjusted_close(tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Download adjusted close prices and return a clean DataFrame.

    Missing values are handled by forward filling internal gaps and dropping any
    remaining rows that still contain missing data.
    """
    data = yf.download(tickers, period=period, interval="1d", auto_adjust=False, progress=False)

    if data.empty:
        raise ValueError("No data returned from yfinance.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" not in data.columns.get_level_values(0):
            raise ValueError("Adjusted close data is unavailable in yfinance response.")
        prices = data["Adj Close"].copy()
    else:
        if "Adj Close" in data.columns:
            prices = data[["Adj Close"]].rename(columns={"Adj Close": tickers[0]}).copy()
        elif "Close" in data.columns:
            prices = data[["Close"]].rename(columns={"Close": tickers[0]}).copy()
        else:
            raise ValueError("Neither adjusted close nor close data found in yfinance response.")

    prices = prices.sort_index().ffill().dropna(how="any")

    missing_tickers = [ticker for ticker in tickers if ticker not in prices.columns]
    if missing_tickers:
        raise ValueError(f"Missing expected ticker columns: {missing_tickers}")

    return prices[tickers]


def compute_annualized_covariance(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized covariance matrix from daily log returns."""
    log_returns = np.log(prices / prices.shift(1)).dropna(how="any")
    daily_covariance = log_returns.cov()
    annualized_covariance = daily_covariance * TRADING_DAYS_PER_YEAR
    return annualized_covariance


def main() -> None:
    """Run ETF covariance analysis and persist output."""
    prices = download_adjusted_close(TICKERS, period="5y")
    covariance_matrix = compute_annualized_covariance(prices)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    covariance_matrix.to_csv(OUTPUT_PATH)

    print("Annualized covariance matrix (252 trading days):")
    print(covariance_matrix)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
