from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


def build_features(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    benchmark: str | None,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series]:
    feats = []
    targets = []
    for ticker in returns.columns:
        r = returns[ticker]
        df = pd.DataFrame(index=returns.index)
        df["ticker"] = ticker
        df["mom_1m"] = prices[ticker].pct_change(4)
        df["mom_3m"] = prices[ticker].pct_change(12)
        df["mom_6m"] = prices[ticker].pct_change(24)
        df["vol_1m"] = r.rolling(4).std()
        df["vol_3m"] = r.rolling(12).std()
        eq = np.exp(r.cumsum())
        dd = eq / eq.cummax() - 1
        df["rolling_drawdown"] = dd.rolling(12).min()

        if benchmark and benchmark in returns.columns:
            df["corr_benchmark"] = r.rolling(12).corr(returns[benchmark])
        else:
            df["corr_benchmark"] = np.nan

        y = prices[ticker].shift(-horizon) / prices[ticker] - 1
        y = y.reindex(df.index)

        valid = df.drop(columns=["ticker"]).notna().all(axis=1) & y.notna()
        feats.append(df.loc[valid])
        targets.append(y.loc[valid])

    X = pd.concat(feats).sort_index()
    y_all = pd.concat(targets).sort_index()
    return X, y_all


def train_models(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict[str, Any]:
    feature_cols = [c for c in X.columns if c != "ticker"]
    Xn = X[feature_cols]
    tscv = TimeSeriesSplit(n_splits=cfg["ml"].get("tscv_splits", 5))

    models: dict[str, Any] = {}
    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=cfg["ml"].get("ridge_alphas", [1.0])[1])),
    ])
    rf_cfg = cfg["ml"]["random_forest"]
    rf = RandomForestRegressor(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        random_state=rf_cfg["random_state"],
        n_jobs=-1,
    )

    for name, model in {"ridge": ridge, "random_forest": rf}.items():
        oos_preds = pd.Series(index=y.index, dtype=float)
        for tr_idx, te_idx in tscv.split(Xn):
            model.fit(Xn.iloc[tr_idx], y.iloc[tr_idx])
            oos_preds.iloc[te_idx] = model.predict(Xn.iloc[te_idx])
        model.fit(Xn, y)
        models[name] = {"model": model, "oos_preds": oos_preds}

    if cfg["ml"]["xgboost"].get("enabled", True):
        try:
            from xgboost import XGBRegressor

            xcfg = cfg["ml"]["xgboost"]
            xgb = XGBRegressor(
                n_estimators=xcfg["n_estimators"],
                max_depth=xcfg["max_depth"],
                learning_rate=xcfg["learning_rate"],
                subsample=xcfg["subsample"],
                colsample_bytree=xcfg["colsample_bytree"],
                random_state=xcfg["random_state"],
                objective="reg:squarederror",
            )
            oos_preds = pd.Series(index=y.index, dtype=float)
            for tr_idx, te_idx in tscv.split(Xn):
                xgb.fit(Xn.iloc[tr_idx], y.iloc[tr_idx])
                oos_preds.iloc[te_idx] = xgb.predict(Xn.iloc[te_idx])
            xgb.fit(Xn, y)
            models["xgboost"] = {"model": xgb, "oos_preds": oos_preds}
        except Exception:
            pass
    return models


def expected_returns_from_model(
    model_obj: Any,
    latest_features: pd.DataFrame,
) -> pd.Series:
    feature_cols = [c for c in latest_features.columns if c != "ticker"]
    preds = model_obj.predict(latest_features[feature_cols])
    out = pd.Series(preds, index=latest_features["ticker"].values, name="expected_return_ml")
    return out
