from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def annualized_stats(returns: pd.DataFrame, ann_factor: int) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    mean_ret = returns.mean() * ann_factor
    vol = returns.std(ddof=1) * np.sqrt(ann_factor)
    cov = returns.cov() * ann_factor
    return mean_ret, vol, cov


def max_drawdown_from_returns(returns: pd.Series) -> float:
    equity = np.exp(returns.cumsum())
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def build_metrics_table(
    returns: pd.DataFrame, ann_factor: int, risk_free_rate: float
) -> pd.DataFrame:
    mean_ret, vol, _ = annualized_stats(returns, ann_factor)
    sharpe = (mean_ret - risk_free_rate) / vol.replace(0, np.nan)
    mdd = returns.apply(max_drawdown_from_returns)
    out = pd.DataFrame(
        {
            "annual_return": mean_ret,
            "annual_volatility": vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": mdd,
        }
    )
    return out.sort_values("sharpe_ratio", ascending=False)


def correlation_analysis(corr: pd.DataFrame, threshold: float = 0.85) -> list[tuple[str, str, float]]:
    duplicates: list[tuple[str, str, float]] = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = float(corr.iloc[i, j])
            if value > threshold:
                duplicates.append((cols[i], cols[j], value))
    return sorted(duplicates, key=lambda x: x[2], reverse=True)


def save_correlation_heatmap(corr: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.index)
    ax.set_title("ETF Correlation Heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
