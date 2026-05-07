# ETF Portfolio Lab

Progetto Python 3.11+ per analizzare e ottimizzare un portafoglio ETF (6–10 strumenti), con confronto tra approcci classici e machine learning.

## Disclaimer
Questo progetto è esclusivamente educativo e di analisi quantitativa. **Non fornisce consulenza finanziaria personalizzata**.

## Funzionalità principali
- Download prezzi storici Adj Close da `yfinance`
- Calcolo rendimenti logaritmici, metriche rischio/rendimento, correlazioni, covarianza annualizzata
- Identificazione ETF potenzialmente ridondanti (correlazione > 0.85)
- Ottimizzazione classica: Minimum Variance, Maximum Sharpe, Risk Parity, Efficient Frontier
- Pipeline ML (Ridge, Random Forest, XGBoost opzionale) con `TimeSeriesSplit`
- Backtest con ribilanciamento mensile e costi di transazione
- Report finale Markdown + grafici + CSV output

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Esecuzione
```bash
python -m src.main --config config.yaml
```

## Output attesi
- `reports/final_report.md`
- `outputs/metrics.csv`
- `outputs/optimal_weights.csv`
- `outputs/correlation_heatmap.png`
- `outputs/efficient_frontier.png`
- `outputs/equity_curves.png`

## Note su limiti/assunzioni
- Dati dipendono da disponibilità/qualità feed `yfinance`.
- Costi e liquidità reali non sono modellati in modo completo.
- Feature ML sono puramente tecniche; regime shifts possono ridurre robustezza out-of-sample.
- Ottimizzazione media-varianza è sensibile all'errore di stima dei rendimenti attesi.
