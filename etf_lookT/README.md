# Portfolio ETF Analysis

Tools to analyze an ETF portfolio, extract ETF holdings, estimate equity and bond exposures, and download current or historical market data where available.

## Scripts

- `analyze_portfolio.py`: reads a portfolio CSV, downloads ETF holdings from configured providers, and generates equity/bond exposure reports.
- `download_market_prices.py`: reads the generated reports and creates current-price CSVs for top equity holdings and top bond holdings.
- `download_historical_timeseries.py`: downloads daily historical time series for top equity holdings and top bond holdings using provider symbols extracted from ETF holdings and optional mappings.

## Install

```powershell
pip install -r requirements.txt
```

## Expected Portfolio CSV

The portfolio CSV must contain:

```text
Strumenti, Asset Class, Controvalore, ISIN
```

## Usage

Analyze the ETF portfolio:

```powershell
python analyze_portfolio.py --portfolio "portfolio-composition-Portafoglio_1.csv"
```

Download current prices using provider-holdings prices as fallback:

```powershell
python download_market_prices.py --providers holdings
```

Download daily historical time series:

```powershell
python download_historical_timeseries.py --start 2020-01-01 --end 2026-06-07
```

Use Nasdaq Data Link / Quandl:

```powershell
python download_historical_timeseries.py --providers nasdaq,yahoo,stooq --nasdaq-api-key "YOUR_API_KEY"
```

## Optional Mapping Files

Create these files only when automatic symbols are not enough:

- `input/equity_ticker_mapping.csv`
- `input/bond_price_mapping.csv`

Columns:

```csv
Holding,Ticker,StooqSymbol,NasdaqDataset
```

For bonds:

```csv
ISIN_holding,Ticker,StooqSymbol,NasdaqDataset
```

## Privacy

Do not commit personal portfolio CSVs, generated output, API keys, or market data downloads unless you intentionally want them in the repository.
