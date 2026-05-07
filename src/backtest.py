from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    common = [c for c in returns.columns if c in weights.index]
    return returns[common].mul(weights[common], axis=1).sum(axis=1)


def compute_summary_stats(port_ret: pd.Series, ann_factor: int, rf: float) -> dict[str, float]:
    eq = np.exp(port_ret.cumsum())
    cagr = float(eq.iloc[-1] ** (ann_factor / len(eq)) - 1)
    vol = float(port_ret.std(ddof=1) * np.sqrt(ann_factor))
    sharpe = float((port_ret.mean() * ann_factor - rf) / (vol + 1e-12))
    dd = eq / eq.cummax() - 1
    mdd = float(dd.min())
    return {"CAGR": cagr, "Volatility": vol, "Sharpe": sharpe, "MaxDrawdown": mdd}


def turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    idx = prev_w.index.union(new_w.index)
    p = prev_w.reindex(idx).fillna(0)
    n = new_w.reindex(idx).fillna(0)
    return float(np.abs(n - p).sum())
