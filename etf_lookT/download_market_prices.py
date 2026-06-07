import argparse
import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ANALYSIS_DIR = BASE_DIR / "output"
DEFAULT_PRICES_DIR = BASE_DIR / "market_prices"
DEFAULT_TICKER_MAP = BASE_DIR / "input" / "equity_ticker_mapping.csv"
DEFAULT_BOND_MAP = BASE_DIR / "input" / "bond_price_mapping.csv"

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
STOOQ_QUOTE_URL = "https://stooq.com/q/l/"
NASDAQ_DATA_URL = "https://data.nasdaq.com/api/v3/datasets/{dataset}/data.json"


def parse_args():
    p = argparse.ArgumentParser(description="Download current market prices for top ETF holdings.")
    p.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_PRICES_DIR))
    p.add_argument("--ticker-map", default=str(DEFAULT_TICKER_MAP))
    p.add_argument("--bond-map", default=str(DEFAULT_BOND_MAP))
    p.add_argument("--providers", default="yahoo,stooq,nasdaq")
    p.add_argument("--nasdaq-api-key", default=os.getenv("NASDAQ_DATA_LINK_API_KEY") or os.getenv("QUANDL_API_KEY"))
    p.add_argument("--top-equities", type=int, default=50)
    p.add_argument("--top-bonds", type=int, default=100)
    p.add_argument("--sleep", type=float, default=0.15)
    return p.parse_args()


def get_json(url, params):
    response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    return response.json()


def get_csv(url, params):
    response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    return pd.read_csv(io.StringIO(response.text))


def normalize_providers(value):
    valid = {"holdings", "yahoo", "stooq", "nasdaq", "quandl"}
    providers = [p.strip().lower() for p in value.split(",") if p.strip()]
    unknown = [p for p in providers if p not in valid]
    if unknown:
        raise ValueError("Unsupported providers: " + ", ".join(unknown))
    return ["nasdaq" if p == "quandl" else p for p in providers]


def clean_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


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


def yahoo_search(query, preferred_types):
    data = get_json(YAHOO_SEARCH_URL, {"q": query, "quotesCount": 8, "newsCount": 0})
    quotes = data.get("quotes", [])
    preferred = [q for q in quotes if q.get("symbol") and q.get("quoteType") in preferred_types]
    return (preferred or quotes or [None])[0]


def yahoo_quote(symbols):
    symbols = [s for s in symbols if s]
    if not symbols:
        return {}
    data = get_json(YAHOO_QUOTE_URL, {"symbols": ",".join(symbols)})
    return {r.get("symbol"): r for r in data.get("quoteResponse", {}).get("result", []) if r.get("symbol")}


def stooq_quote(symbol):
    if not symbol:
        return None
    df = get_csv(STOOQ_QUOTE_URL, {"s": symbol.lower(), "f": "sd2t2ohlcv", "h": "", "e": "csv"})
    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    if str(row.get("Close")).upper() == "N/D":
        return None
    return {
        "symbol": row.get("Symbol"),
        "shortName": row.get("Symbol"),
        "quoteType": "EQUITY_OR_BOND",
        "fullExchangeName": "Stooq",
        "regularMarketPrice": row.get("Close"),
        "provider": "stooq",
        "provider_dataset": symbol,
        "provider_date": row.get("Date"),
        "provider_time": row.get("Time"),
    }


def nasdaq_quote(dataset, api_key):
    if not dataset:
        return None
    params = {"limit": 1}
    if api_key:
        params["api_key"] = api_key
    data = get_json(NASDAQ_DATA_URL.format(dataset=dataset), params)
    ds = data.get("dataset_data", {})
    rows = ds.get("data", [])
    cols = ds.get("column_names", [])
    if not rows:
        return None
    row = dict(zip(cols, rows[0]))
    price = next((row[c] for c in ["Close", "Last", "Value", "Price", "Mid", "Ask", "Bid"] if c in row and pd.notna(row[c])), None)
    return {
        "symbol": dataset,
        "shortName": ds.get("name") or dataset,
        "quoteType": "DATASET",
        "fullExchangeName": "Nasdaq Data Link",
        "regularMarketPrice": price,
        "provider": "nasdaq",
        "provider_dataset": dataset,
        "provider_date": row.get("Date") or row.get("date"),
    }


def default_stooq_symbol(symbol):
    if not symbol:
        return None
    return f"{symbol}.us" if "." not in symbol else symbol


def price_with_providers(symbol, mapping_row, providers, api_key):
    errors = []
    for provider in providers:
        try:
            if provider == "holdings":
                return None, "prezzo da holdings provider"
            if provider == "yahoo" and symbol:
                quote = yahoo_quote([symbol]).get(symbol)
                if quote:
                    quote["provider"] = "yahoo"
                    quote["provider_dataset"] = symbol
                    return quote, "prezzo scaricato"
            if provider == "stooq":
                quote = stooq_quote(mapping_row.get("StooqSymbol") or default_stooq_symbol(symbol))
                if quote:
                    return quote, "prezzo scaricato"
            if provider == "nasdaq":
                quote = nasdaq_quote(mapping_row.get("NasdaqDataset"), api_key)
                if quote:
                    return quote, "prezzo scaricato"
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
    return None, "prezzo non trovato" + ("; " + " | ".join(errors) if errors else "")


def quote_row(base, quote, status, symbol_source):
    market_time = quote.get("regularMarketTime") if quote else None
    date_value = datetime.fromtimestamp(market_time, tz=timezone.utc).isoformat() if market_time else (quote or {}).get("provider_date")
    market_price = quote.get("regularMarketPrice") if quote else None
    provider_price = base.get("Prezzo_provider")
    final_price = market_price if pd.notna(market_price) else provider_price
    source = (quote or {}).get("provider") if pd.notna(market_price) else ("provider_holdings" if pd.notna(provider_price) else None)
    return {
        **base,
        "Ticker": (quote or {}).get("symbol") or base.get("Ticker"),
        "Ticker_source": symbol_source,
        "Provider": (quote or {}).get("provider"),
        "Provider_dataset": (quote or {}).get("provider_dataset"),
        "Nome_mercato": (quote or {}).get("shortName") or (quote or {}).get("longName"),
        "Tipo_mercato": (quote or {}).get("quoteType"),
        "Exchange": (quote or {}).get("fullExchangeName") or (quote or {}).get("exchange"),
        "Valuta": (quote or {}).get("currency"),
        "Prezzo_mercato": market_price,
        "Prezzo_finale": final_price,
        "Fonte_prezzo_finale": source,
        "Variazione_pct": (quote or {}).get("regularMarketChangePercent"),
        "Market_cap": (quote or {}).get("marketCap"),
        "Data_prezzo": date_value,
        "Ora_prezzo": (quote or {}).get("provider_time"),
        "Stato": status,
    }


def load_top_equities(analysis_dir, limit):
    path = Path(analysis_dir) / "holdings_azionarie_dettaglio.csv"
    if not path.exists():
        path = Path(analysis_dir) / "top_societa_azionarie.csv"
    df = pd.read_csv(path).dropna(subset=["Holding"])
    for col in ["Ticker", "ISIN_holding", "Prezzo_holding"]:
        if col not in df.columns:
            df[col] = None
    grouped = df.groupby(["Holding", "Ticker", "ISIN_holding"], dropna=False).agg({"Peso_effettivo_portafoglio": "sum", "Prezzo_holding": "first"}).reset_index()
    return grouped.sort_values("Peso_effettivo_portafoglio", ascending=False).head(limit)


def load_top_bonds(analysis_dir, limit):
    path = Path(analysis_dir) / "holdings_obbligazionarie_dettaglio.csv"
    df = pd.read_csv(path).dropna(subset=["ISIN_holding"])
    for col in ["Ticker", "Prezzo_holding", "Scadenza", "Cedola", "Rating", "Paese", "ETF", "Obligor"]:
        if col not in df.columns:
            df[col] = None
    grouped = df.groupby(["ISIN_holding", "Holding", "Obligor"], dropna=False).agg({
        "Peso_effettivo_portafoglio": "sum", "Prezzo_holding": "first", "Scadenza": "first",
        "Cedola": "first", "Rating": "first", "Paese": "first", "ETF": "first", "Ticker": "first",
    }).reset_index()
    return grouped.sort_values("Peso_effettivo_portafoglio", ascending=False).head(limit)


def price_equities(rows, mapping, providers, api_key, sleep_seconds):
    out = []
    for _, row in rows.iterrows():
        holding = str(row["Holding"]).strip()
        m = mapping.get(holding.upper(), {})
        symbol = m.get("Ticker") or clean_value(row.get("Ticker"))
        source = "ticker_map" if m.get("Ticker") else "holdings_ticker"
        if not symbol and "yahoo" in providers:
            result = yahoo_search(holding, {"EQUITY"})
            symbol = result.get("symbol") if result else None
            source = "yahoo_search" if symbol else "not_found"
        base = {"Holding": holding, "ISIN_holding": row.get("ISIN_holding"), "Ticker": symbol, "Peso_effettivo_portafoglio": row.get("Peso_effettivo_portafoglio"), "Prezzo_provider": row.get("Prezzo_holding")}
        quote, status = price_with_providers(symbol, m, providers, api_key)
        out.append(quote_row(base, quote or {}, status, source))
        time.sleep(sleep_seconds)
    return pd.DataFrame(out)


def price_bonds(rows, mapping, providers, api_key, sleep_seconds):
    out = []
    for _, row in rows.iterrows():
        isin = str(row["ISIN_holding"]).strip()
        m = mapping.get(isin.upper(), {})
        symbol = m.get("Ticker") or clean_value(row.get("Ticker"))
        base = {"ISIN_holding": isin, "Holding": row.get("Holding"), "Obligor": row.get("Obligor"), "ETF": row.get("ETF"), "Ticker": symbol, "Peso_effettivo_portafoglio": row.get("Peso_effettivo_portafoglio"), "Prezzo_provider": row.get("Prezzo_holding"), "Scadenza": row.get("Scadenza"), "Cedola": row.get("Cedola"), "Rating": row.get("Rating"), "Paese": row.get("Paese")}
        quote, status = price_with_providers(symbol, m, providers, api_key)
        out.append(quote_row(base, quote or {}, status, "bond_map" if m.get("Ticker") else "holdings_ticker"))
        time.sleep(sleep_seconds)
    return pd.DataFrame(out)


def main():
    args = parse_args()
    providers = normalize_providers(args.providers)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    equity_map = load_mapping(args.ticker_map, "Holding")
    bond_map = load_mapping(args.bond_map, "ISIN_holding")
    equities = load_top_equities(args.analysis_dir, args.top_equities)
    bonds = load_top_bonds(args.analysis_dir, args.top_bonds)
    price_equities(equities, equity_map, providers, args.nasdaq_api_key, args.sleep).to_csv(output_dir / "prezzi_azioni_top_holdings.csv", index=False)
    price_bonds(bonds, bond_map, providers, args.nasdaq_api_key, args.sleep).to_csv(output_dir / "prezzi_bond_top_holdings.csv", index=False)
    print("Generated price reports in", output_dir)


if __name__ == "__main__":
    main()
