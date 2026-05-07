from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtest import compute_summary_stats, portfolio_returns
from src.data import load_data
from src.metrics import annualized_stats, build_metrics_table, correlation_analysis, save_correlation_heatmap
from src.ml import build_features, expected_returns_from_model, train_models
from src.optimization import efficient_frontier, max_sharpe, min_variance, risk_parity
from src.reporting import save_efficient_frontier_plot, save_equity_curves, write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    Path("outputs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    db = load_data(cfg)
    metrics = build_metrics_table(db.returns, db.annualization_factor, cfg["data"]["risk_free_rate"])
    mean_ret, _, cov = annualized_stats(db.returns, db.annualization_factor)
    corr = db.returns.corr()
    duplicates = correlation_analysis(corr, threshold=0.85)
    save_correlation_heatmap(corr, "outputs/correlation_heatmap.png")

    wmin = cfg["optimization"]["weight_min"]
    wmax = cfg["optimization"]["weight_max"]
    w_eq = pd.Series(1 / len(mean_ret), index=mean_ret.index, name="equal_weight")
    w_minv = min_variance(mean_ret, cov, wmin, wmax)
    w_maxs = max_sharpe(mean_ret, cov, cfg["data"]["risk_free_rate"], wmin, wmax)
    w_rp = risk_parity(cov, wmin, wmax)

    frontier = efficient_frontier(mean_ret, cov, points=40, wmin=wmin, wmax=wmax)
    save_efficient_frontier_plot(frontier, "outputs/efficient_frontier.png")

    horizon = cfg["ml"]["forecast_horizon_periods"]
    X, y = build_features(db.prices, db.returns, cfg["data"].get("benchmark"), horizon)
    models = train_models(X, y, cfg)

    latest_rows = X.groupby("ticker").tail(1)
    expected_ml = expected_returns_from_model(models["random_forest"]["model"], latest_rows)
    expected_ml = expected_ml.reindex(mean_ret.index).fillna(mean_ret.mean())
    w_ml = max_sharpe(expected_ml, cov, cfg["data"]["risk_free_rate"], wmin, wmax)
    w_ml.name = "max_sharpe_ml"

    weights_df = pd.concat([w_eq, w_minv, w_maxs, w_rp, w_ml], axis=1)
    weights_df.to_csv("outputs/optimal_weights.csv")
    metrics.to_csv("outputs/metrics.csv")

    strategy_returns = {
        "equal_weight": portfolio_returns(db.returns, w_eq),
        "min_variance": portfolio_returns(db.returns, w_minv),
        "max_sharpe_hist": portfolio_returns(db.returns, w_maxs),
        "max_sharpe_ml": portfolio_returns(db.returns, w_ml),
    }

    cost = cfg["backtest"]["transaction_cost_bps"] / 10000.0
    strategy_returns = {k: v - cost / 12 for k, v in strategy_returns.items()}

    equities = {k: np.exp(v.cumsum()) for k, v in strategy_returns.items()}
    save_equity_curves(equities, "outputs/equity_curves.png")

    summary = pd.DataFrame(
        {
            k: compute_summary_stats(v, db.annualization_factor, cfg["data"]["risk_free_rate"])
            for k, v in strategy_returns.items()
        }
    ).T
    summary.to_csv("outputs/backtest_summary.csv")

    ml_rows = []
    for name, payload in models.items():
        pred = payload["oos_preds"].dropna()
        common = y.loc[pred.index]
        mse = float(((common - pred) ** 2).mean())
        ml_rows.append({"model": name, "oos_mse": mse})
    ml_scores = pd.DataFrame(ml_rows).sort_values("oos_mse")
    ml_scores.to_csv("outputs/ml_oos_scores.csv", index=False)

    write_markdown_report("reports/final_report.md", metrics, duplicates, summary, ml_scores)
    print("Pipeline completata. Report in reports/final_report.md")


if __name__ == "__main__":
    main()
