from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def _negative_sharpe(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray, rf: float) -> float:
    ret = float(weights @ mu)
    vol = _portfolio_vol(weights, cov)
    return -(ret - rf) / (vol + 1e-12)


def _bounds(n: int, wmin: float, wmax: float) -> list[tuple[float, float]]:
    return [(wmin, wmax)] * n


def _constraint_sum_to_one() -> dict:
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}


def min_variance(mu: pd.Series, cov: pd.DataFrame, wmin: float, wmax: float) -> pd.Series:
    n = len(mu)
    x0 = np.full(n, 1.0 / n)
    res = minimize(
        _portfolio_vol,
        x0,
        args=(cov.values,),
        method="SLSQP",
        bounds=_bounds(n, wmin, wmax),
        constraints=[_constraint_sum_to_one()],
    )
    if not res.success:
        raise RuntimeError(f"Ottimizzazione min variance fallita: {res.message}")
    return pd.Series(res.x, index=mu.index, name="min_variance")


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float, wmin: float, wmax: float) -> pd.Series:
    n = len(mu)
    x0 = np.full(n, 1.0 / n)
    res = minimize(
        _negative_sharpe,
        x0,
        args=(mu.values, cov.values, rf),
        method="SLSQP",
        bounds=_bounds(n, wmin, wmax),
        constraints=[_constraint_sum_to_one()],
    )
    if not res.success:
        raise RuntimeError(f"Ottimizzazione max sharpe fallita: {res.message}")
    return pd.Series(res.x, index=mu.index, name="max_sharpe")


def risk_parity(cov: pd.DataFrame, wmin: float, wmax: float) -> pd.Series:
    n = len(cov)
    x0 = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov.values @ w
        mrc = cov.values @ w
        rc = w * mrc / (np.sqrt(port_var) + 1e-12)
        target = np.full(n, rc.mean())
        return float(((rc - target) ** 2).sum())

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=_bounds(n, wmin, wmax),
        constraints=[_constraint_sum_to_one()],
    )
    if not res.success:
        raise RuntimeError(f"Ottimizzazione risk parity fallita: {res.message}")
    return pd.Series(res.x, index=cov.index, name="risk_parity")


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    points: int,
    wmin: float,
    wmax: float,
) -> pd.DataFrame:
    target_rets = np.linspace(mu.min(), mu.max(), points)
    rows = []
    n = len(mu)
    for tr in target_rets:
        cons = [
            _constraint_sum_to_one(),
            {"type": "eq", "fun": lambda w, tr=tr: float(w @ mu.values) - tr},
        ]
        res = minimize(
            _portfolio_vol,
            np.full(n, 1 / n),
            args=(cov.values,),
            method="SLSQP",
            bounds=_bounds(n, wmin, wmax),
            constraints=cons,
        )
        if res.success:
            vol = _portfolio_vol(res.x, cov.values)
            rows.append({"target_return": tr, "volatility": vol})
    return pd.DataFrame(rows)
