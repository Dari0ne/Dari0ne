import argparse
import io
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = BASE_DIR / "output"
DEFAULT_OUTPUT_DIR = BASE_DIR / "historical_prices"
DEFAULT_TICKER_MAP = BASE_DIR / "input" / "equity_ticker_mapping.csv"
DEFAULT_BOND_MAP = BASE_DIR / "input" / "bond_price_mapping.csv"

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
STOOQ_DAILY_URL = "https://stooq.com/q/d/l/"
NASDAQ_DATA_URL = "https://data.nasdaq.com/api/v3/datasets/{dataset}/data.json"


def parse_args():
    default_start = date.today().replace(year=date.today().year - 5).isoformat()
    p = argparse.ArgumentParser(description="Download daily historical time series for top ETF holdings.")
    p.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--ticker-map", default=str(DEFAULT_TICKER_MAP))
    p.add_argument("--bond-map", default=str(DEFAULT_BOND_MAP))
    p.add_argument("--providers", default="yahoo,stooq,nasdaq")
    p.add_argument("--nasdaq-api-key", default=os.getenv("NASDAQ_DATA_LINK_API_KEY") or os.getenv("QUANDL_API_KEY"))
    p.add_argument("--start", default=default_start)
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--top-equities", type=int, default=50)
    p.add_argument("--top-bonds", type=int, default=100)
    p.add_argument("--sleep", type=float, default=0.15)
    return p.parse_args()


def get_json(url, params):
    response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=45)
    response.raise_for_status()
    return response.json()


def get_csv(url, params):
    response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=45)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def clean_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def normalize_providers(value):
    valid = {"yahoo", "stooq", "nasdaq", "quandl"}
    providers = [p.strip().lower() for p in value.split(",") if p.strip()]
    unknown = [p for p in providers if p not in valid]
    if unknown:
        raise ValueError("Unsupported providers: " + ", ".join(unknown))
    return ["nasdaq" if p == "quandl" else p for p in providers]


def load_mapping(path, required_key):
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if required_key not in df.columns:
        raise ValueError(f"Mapping must contain {required_key}")
    out = {}
    for _, row in df.dropna(subset=[required_key]).iterrows():
        key = str(row[required_key]).strip().upper()
        out[key] = {
            "Ticker": clean_value(row.get("Ticker")),
            "StooqSymbol": clean_value(row.get("StooqSymbol")),
            "NasdaqDataset": clean_value(row.get("NasdaqDataset")),
        }
    return out


def parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def yahoo_period(value):
    return int(time.mktime(datetime.combine(value, datetime.min.time()).timetuple()))


def default_stooq_symbol(symbol):
    if not symbol:
        return None
    return f"{symbol}.us" if "." not in symbol else symbol


def load_top_equities(analysis_dir, limit):
    path = Path(analysis_dir) / "holdings_azionarie_dettaglio.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run analyze_portfolio.py first.")
    df = pd.read_csv(path).dropna(subset=["Holding"])
    for col in ["Ticker", "ISIN_holding", "Prezzo_holding"]:
        if col not in df.columns:
            df[col] = None
    grouped = df.groupby(["Holding", "Ticker", "ISIN_holding"], dropna=False).agg({"Peso_effettivo_portafoglio": "sum", "Prezzo_holding": "first"}).reset_index()
    return grouped.sort_values("Peso_effettivo_portafoglio", ascending=False).head(limit)


def load_top_bonds(analysis_dir, limit):
    path = Path(analysis_dir) / "holdings_obbligazionarie_dettaglio.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run analyze_portfolio.py first.")
    df = pd.read_csv(path).dropna(subset=["ISIN_holding"])
    for col in ["Ticker", "Prezzo_holding", "Scadenza", "Cedola", "Rating", "Paese", "ETF", "Obligor"]:
        if col not in df.columns:
            df[col] = None
    grouped = df.groupby(["ISIN_holding", "Holding", "Obligor", "Ticker"], dropna=False).agg({
        "Peso_effettivo_portafoglio": "sum", "Prezzo_holding": "first", "Scadenza": "first",
        "Cedola": "first", "Rating": "first", "Paese": "first", "ETF": "first",
    }).reset_index()
    return grouped.sort_values("Peso_effettivo_portafoglio", ascending=False).head(limit)


def maturity_year(value):
    value = clean_value(value)
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    return digits[:4] if len(digits) >= 4 else None


def add_unique(items, value):
    value = clean_value(value)
    if value and value not in items:
        items.append(value)


def bond_candidate_symbols(row, mapping_row):
    candidates = []
    add_unique(candidates, mapping_row.get("Ticker"))
    add_unique(candidates, row.get("Ticker"))
    add_unique(candidates, row.get("ISIN_holding"))
    ticker = clean_value(row.get("Ticker"))
    year = maturity_year(row.get("Scadenza"))
    if ticker and year:
        add_unique(candidates, f"{ticker}{year}")
        add_unique(candidates, f"{ticker}-{year}")
        add_unique(candidates, f"{ticker} {year}")
    holding = clean_value(row.get("Holding"))
    if holding:
        cleaned = re.sub(r"\b(REGS|REG S|144A|MTN|GMTN|NOTE|NOTES|BOND|BONDS|FRN)\b", "", holding, flags=re.I)
        add_unique(candidates, re.sub(r"\s+", " ", cleaned).strip())
    return candidates


def yahoo_daily(symbol, start_date, end_date):
    data = get_json(YAHOO_CHART_URL.format(symbol=symbol), {
        "period1": yahoo_period(start_date),
        "period2": yahoo_period(end_date + timedelta(days=1)),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    })
    result = data.get("chart", {}).get("result", [])
    if not result:
        return pd.DataFrame()
    result = result[0]
    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    adjclose = result.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
    if not timestamps:
        return pd.DataFrame()
    df = pd.DataFrame({
        "Data": [datetime.utcfromtimestamp(ts).date().isoformat() for ts in timestamps],
        "Open": quote.get("open", []),
        "High": quote.get("high", []),
        "Low": quote.get("low", []),
        "Close": quote.get("close", []),
        "Adj_Close": adjclose,
        "Volume": quote.get("volume", []),
    })
    df["Provider"] = "yahoo"
    df["Provider_symbol"] = symbol
    return df


def stooq_daily(symbol, start_date, end_date):
    if not symbol:
        return pd.DataFrame()
    df = get_csv(STOOQ_DAILY_URL, {"s": symbol.lower(), "d1": start_date.strftime("%Y%m%d"), "d2": end_date.strftime("%Y%m%d"), "i": "d"})
    if df.empty or "Close" not in df.columns or str(df.iloc[0].get("Close")).upper() == "N/D":
        return pd.DataFrame()
    df = df.rename(columns={"Date": "Data"})
    if "Adj_Close" not in df.columns:
        df["Adj_Close"] = df["Close"]
    df["Provider"] = "stooq"
    df["Provider_symbol"] = symbol
    return df[["Data", "Open", "High", "Low", "Close", "Adj_Close", "Volume", "Provider", "Provider_symbol"]]


def first_matching_column(df, names):
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def nasdaq_daily(dataset, start_date, end_date, api_key):
    if not dataset:
        return pd.DataFrame()
    params = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat(), "collapse": "daily"}
    if api_key:
        params["api_key"] = api_key
    data = get_json(NASDAQ_DATA_URL.format(dataset=dataset), params)
    ds = data.get("dataset_data", {})
    rows = ds.get("data", [])
    cols = ds.get("column_names", [])
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows, columns=cols)
    date_col = first_matching_column(raw, ["Date", "date"])
    close_col = first_matching_column(raw, ["Close", "Last", "Value", "Price", "Mid", "Ask", "Bid"])
    if not date_col or not close_col:
        return pd.DataFrame()
    out = pd.DataFrame()
    out["Data"] = raw[date_col]
    out["Open"] = raw[first_matching_column(raw, ["Open"])] if first_matching_column(raw, ["Open"]) else None
    out["High"] = raw[first_matching_column(raw, ["High"])] if first_matching_column(raw, ["High"]) else None
    out["Low"] = raw[first_matching_column(raw, ["Low"])] if first_matching_column(raw, ["Low"]) else None
    out["Close"] = raw[close_col]
    out["Adj_Close"] = raw[close_col]
    volume_col = first_matching_column(raw, ["Volume"])
    out["Volume"] = raw[volume_col] if volume_col else None
    out["Provider"] = "nasdaq"
    out["Provider_symbol"] = dataset
    return out


def fetch_daily(symbol, mapping_row, providers, start_date, end_date, api_key):
    errors = []
    for provider in providers:
        try:
            if provider == "yahoo" and symbol:
                df = yahoo_daily(symbol, start_date, end_date)
            elif provider == "stooq":
                df = stooq_daily(mapping_row.get("StooqSymbol") or default_stooq_symbol(symbol), start_date, end_date)
            elif provider == "nasdaq":
                df = nasdaq_daily(mapping_row.get("NasdaqDataset"), start_date, end_date, api_key)
            else:
                df = pd.DataFrame()
            if not df.empty:
                return df, None
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
    return pd.DataFrame(), " | ".join(errors) if errors else "nessuna serie trovata"


def add_instrument_columns(df, asset_type, row, symbol):
    df = df.copy()
    df.insert(0, "Asset_type", asset_type)
    df.insert(1, "Holding", row.get("Holding"))
    df.insert(2, "ISIN_holding", row.get("ISIN_holding"))
    df.insert(3, "Ticker", symbol)
    df["Peso_effettivo_portafoglio"] = row.get("Peso_effettivo_portafoglio")
    df["Prezzo_holding_corrente"] = row.get("Prezzo_holding")
    if asset_type == "bond":
        for col in ["Obligor", "ETF", "Scadenza", "Cedola", "Rating", "Paese"]:
            df[col] = row.get(col)
    return df


def download_rows(rows, asset_type, mapping, providers, start_date, end_date, api_key, sleep_seconds):
    series_rows = []
    status_rows = []
    for _, row in rows.iterrows():
        if asset_type == "equity":
            key = str(row.get("Holding")).strip().upper()
            mapping_row = mapping.get(key, {})
            candidates = [mapping_row.get("Ticker") or clean_value(row.get("Ticker"))]
        else:
            key = str(row.get("ISIN_holding")).strip().upper()
            mapping_row = mapping.get(key, {})
            candidates = bond_candidate_symbols(row, mapping_row)
        found = pd.DataFrame()
        used_symbol = None
        error = None
        for symbol in [c for c in candidates if c]:
            found, error = fetch_daily(symbol, mapping_row, providers, start_date, end_date, api_key)
            if not found.empty:
                used_symbol = symbol
                break
        if found.empty:
            status_rows.append({"Asset_type": asset_type, "Holding": row.get("Holding"), "ISIN_holding": row.get("ISIN_holding"), "Ticker": used_symbol, "Candidati_provati": "; ".join([c for c in candidates if c]), "Stato": "serie non trovata", "Errore": error})
        else:
            series_rows.append(add_instrument_columns(found, asset_type, row, used_symbol))
            status_rows.append({"Asset_type": asset_type, "Holding": row.get("Holding"), "ISIN_holding": row.get("ISIN_holding"), "Ticker": used_symbol, "Candidati_provati": "; ".join([c for c in candidates if c]), "Stato": "scaricato", "Righe": len(found), "Provider": found["Provider"].iloc[0], "Provider_symbol": found["Provider_symbol"].iloc[0]})
        time.sleep(sleep_seconds)
    return (pd.concat(series_rows, ignore_index=True) if series_rows else pd.DataFrame(), pd.DataFrame(status_rows))


def build_bond_universe(bonds, bond_map):
    rows = []
    for _, row in bonds.iterrows():
        key = str(row.get("ISIN_holding")).strip().upper()
        mapping_row = bond_map.get(key, {})
        rows.append({
            "ISIN_holding": row.get("ISIN_holding"), "Holding": row.get("Holding"), "Obligor": row.get("Obligor"),
            "Ticker_provider_etf": row.get("Ticker"), "Prezzo_holding_corrente": row.get("Prezzo_holding"),
            "Scadenza": row.get("Scadenza"), "Cedola": row.get("Cedola"), "Rating": row.get("Rating"),
            "Paese": row.get("Paese"), "ETF": row.get("ETF"), "Peso_effettivo_portafoglio": row.get("Peso_effettivo_portafoglio"),
            "Ticker_mapping": mapping_row.get("Ticker"), "StooqSymbol_mapping": mapping_row.get("StooqSymbol"),
            "NasdaqDataset_mapping": mapping_row.get("NasdaqDataset"), "Candidati_symbol": "; ".join(bond_candidate_symbols(row, mapping_row)),
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    providers = normalize_providers(args.providers)
    start_date = parse_date(args.start)
    end_date = parse_date(args.end)
    equity_map = load_mapping(args.ticker_map, "Holding")
    bond_map = load_mapping(args.bond_map, "ISIN_holding")
    equities = load_top_equities(args.analysis_dir, args.top_equities)
    bonds = load_top_bonds(args.analysis_dir, args.top_bonds)
    equity_series, equity_status = download_rows(equities, "equity", equity_map, providers, start_date, end_date, args.nasdaq_api_key, args.sleep)
    bond_series, bond_status = download_rows(bonds, "bond", bond_map, providers, start_date, end_date, args.nasdaq_api_key, args.sleep)
    equity_series.to_csv(output_dir / "serie_storiche_azioni_daily.csv", index=False)
    bond_series.to_csv(output_dir / "serie_storiche_bond_daily.csv", index=False)
    pd.concat([equity_status, bond_status], ignore_index=True).to_csv(output_dir / "serie_storiche_status.csv", index=False)
    build_bond_universe(bonds, bond_map).to_csv(output_dir / "anagrafica_bond_estratta_da_etf.csv", index=False)
    print("Generated historical reports in", output_dir)


if __name__ == "__main__":
    main()
