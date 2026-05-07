from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_efficient_frontier_plot(frontier: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(frontier["volatility"], frontier["target_return"], color="tab:blue")
    ax.set_xlabel("Volatilità")
    ax.set_ylabel("Rendimento atteso")
    ax.set_title("Efficient Frontier")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_equity_curves(equities: dict[str, pd.Series], output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, series in equities.items():
        ax.plot(series.index, series.values, label=name)
    ax.set_title("Equity Curves")
    ax.set_ylabel("Crescita cumulata")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_markdown_report(
    path: str,
    metrics: pd.DataFrame,
    duplicates: list[tuple[str, str, float]],
    backtest_stats: pd.DataFrame,
    ml_scores: pd.DataFrame,
) -> None:
    lines = [
        "# Report Analisi ETF",
        "",
        "## Disclaimer",
        "Questo documento è esclusivamente a fini educativi/quantitativi e non costituisce consulenza finanziaria personalizzata.",
        "",
        "## Metriche ETF",
        metrics.to_markdown(),
        "",
        "## ETF potenzialmente ridondanti (corr > 0.85)",
    ]
    if duplicates:
        lines.extend([f"- {a} / {b}: {c:.3f}" for a, b, c in duplicates])
    else:
        lines.append("- Nessuna coppia oltre soglia.")

    lines.extend([
        "",
        "## Confronto Backtest Strategie",
        backtest_stats.to_markdown(),
        "",
        "## Performance out-of-sample ML",
        ml_scores.to_markdown(index=False),
        "",
        "## Limiti e assunzioni",
        "- Sensibilità ai dati storici e ai parametri scelti.",
        "- Costi di transazione modellati in modo semplificato.",
        "- Modelli ML non garantiscono capacità predittiva futura.",
    ])

    Path(path).write_text("\n".join(lines), encoding="utf-8")
