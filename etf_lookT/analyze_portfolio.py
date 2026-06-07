import argparse
import io
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PORTFOLIO = BASE_DIR / "portfolio-composition-Portafoglio_1.csv"
DEFAULT_OUTPUT = BASE_DIR / "output"

# Add or edit ETF holdings URLs here. BlackRock/iShares product pages are supported.
HOLDINGS_URLS = {
    "IE00B4L5Y983": "https://www.ishares.com/uk/individual/en/products/251882/ishares-msci-world-ucits-etf-acc-fund?siteEntryPassthrough=true&switchLocale=y",
    "IE00BP3QZB59": "https://www.ishares.com/uk/individual/en/products/270048/ishares-msci-world-value-factor-ucits-etf?siteEntryPassthrough=true&switchLocale=y",
    "IE00BKM4GZ66": "https://www.ishares.com/uk/individual/en/products/264659/ishares-core-msci-em-imi-ucits-etf?siteEntryPassthrough=true&switchLocale=y",
    "IE00BSKRJX20": "https://www.blackrock.com/uk/individual/products/272122/",
    "IE00BYZTVV78": "https://www.blackrock.com/be/intermediaries/fr/products/280851/ishares-sustainable-euro-corporate-bond-0-3yr-ucits-etf",
    "IE00BDZVH966": "https://www.blackrock.com/fr/intermediaries/products/295696/ishares-tips-ucits-etf",
    "IE000VSFIC94": "https://www.blackrock.com/es/particulares/productos/331364/ishares-broad-high-yield-corp-bond-ucits-etf",
    "IE00B9M6RS56": "https://www.blackrock.com/be/intermediaries/nl/products/254531/ishares-j-p-morgan-em-bond-eur-hedged-ucits-etf",
}


def parse_args():
    p = argparse.ArgumentParser(description="Analyze ETF portfolio holdings.")
    p.add_argument("--portfolio", default=str(DEFAULT_PORTFOLIO))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def get_url(url, **kwargs):
    headers = kwargs.pop("headers", {})
    headers.setdefault("User-Agent", "Mozilla/5.0")
    return requests.get(url, headers=headers, timeout=60, **kwargs)


def is_blackrock_page(url):
    host = urlparse(url).netloc.lower()
    return "blackrock.com" in host or "ishares.com" in host


def scalar(value):
    return value.get("raw", value.get("display")) if isinstance(value, dict) else value


def blackrock_json_to_frame(data):
    rows = data.get("aaData", [])
    if not rows:
        return pd.DataFrame()
    max_cols = max(len(r) for r in rows)
    cols = [
        "Ticker", "Name", "Sector", "Asset Class", "Market Value", "Weight (%)",
        "Notional Value", "Nominal", "Par Value", "ISIN", "Price", "Location",
        "Exchange", "Duration", "Maturity", "Coupon", "Market Currency", "Effective Date",
    ] if max_cols >= 18 else [
        "Ticker", "Name", "Sector", "Asset Class", "Market Value", "Weight (%)",
        "Notional Value", "Nominal", "ISIN", "Price", "Location", "Exchange",
        "Market Currency",
    ]
    out = []
    for row in rows:
        values = [scalar(v) for v in row]
        values.extend([None] * (len(cols) - len(values)))
        out.append(values[:len(cols)])
    return pd.DataFrame(out, columns=cols)


def read_blackrock_page(url):
    page = get_url(url)
    page.raise_for_status()
    matches = re.findall(r'data-ajaxUri="([^"]+tab=all[^"]*fileType=json[^"]*)"', page.text)
    if not matches:
        matches = re.findall(r'data-ajaxUri="([^"]+fileType=json[^"]*)"', page.text)
    if not matches:
        raise ValueError("BlackRock holdings endpoint not found")
    holdings_url = urljoin(url, matches[0].replace("&amp;", "&"))
    r = get_url(holdings_url)
    r.raise_for_status()
    return blackrock_json_to_frame(json.loads(r.text.lstrip("\ufeff")))


def read_provider_file(url):
    if is_blackrock_page(url) and not url.lower().endswith((".csv", ".xls", ".xlsx")):
        return read_blackrock_page(url)
    r = get_url(url)
    r.raise_for_status()
    content = io.BytesIO(r.content)
    if url.lower().endswith(".csv"):
        return pd.read_csv(content)
    if url.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(content)
    tables = pd.read_html(r.text)
    if not tables:
        raise ValueError("No table found")
    return max(tables, key=len)


def find_col(df, patterns):
    for col in df.columns:
        c = str(col).lower()
        if any(re.search(p, c) for p in patterns):
            return col
    return None


def clean_text(value):
    if pd.isna(value):
        return None
    text = str(value).strip()
    return None if not text or text.lower() in {"nan", "none", "-"} else text


def normalize_obligor(value):
    text = clean_text(value)
    if not text:
        return None
    if re.fullmatch(r"TREASURY\s*(\(CPI\))?\s*(NOTE|NOTES)?", text, flags=re.I):
        return "UNITED STATES TREASURY"
    text = re.sub(r"\b(REGS|REG S|144A|MTN|GMTN|NOTE|NOTES|BOND|BONDS|FRN)\b", "", text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip(" -") or None


def infer_obligor(row):
    explicit = clean_text(row.get("Obligor"))
    if explicit:
        return normalize_obligor(explicit)
    name = clean_text(row.get("Holding"))
    if not name:
        return None
    name = re.sub(r"\b\d{1,2}([.,]\d+)?\s*%\b.*$", "", name).strip()
    name = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b.*$", "", name).strip()
    name = re.sub(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b.*$", "", name).strip()
    return normalize_obligor(name)


def normalize_holdings(raw):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = {
        "Holding": find_col(df, ["^name$", "security", "holding", "issuer", "emittente", "titolo"]),
        "ISIN_holding": find_col(df, ["^isin$", "isin"]),
        "Ticker": find_col(df, ["^ticker$", "ticker"]),
        "Obligor": find_col(df, ["issuer", "obligor", "emittente"]),
        "Peso_holding": find_col(df, ["weight", "peso", "%", "percent"]),
        "Prezzo_holding": find_col(df, ["^price$", "prezzo"]),
        "Settore": find_col(df, ["sector", "settore", "industry"]),
        "Paese": find_col(df, ["country", "paese", "location"]),
        "Scadenza": find_col(df, ["maturity", "scadenza"]),
        "Cedola": find_col(df, ["coupon", "cedola"]),
        "Rating": find_col(df, ["rating"]),
        "Tipo": find_col(df, ["asset class", "type", "tipo"]),
    }
    out = pd.DataFrame()
    for new_col, old_col in cols.items():
        out[new_col] = df[old_col] if old_col else None
    out["Peso_holding"] = pd.to_numeric(
        out["Peso_holding"].astype(str).str.replace("%", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    ) / 100
    out["Obligor"] = out.apply(infer_obligor, axis=1)
    return out.dropna(how="all")


def euro_to_float(value):
    if pd.isna(value) or value == "-":
        return 0.0
    return float(str(value).replace("€", "").replace("\xa0", "").replace(".", "").replace(",", ".").strip())


def load_portfolio(path):
    df = pd.read_csv(path)
    required = {"Strumenti", "Asset Class", "Controvalore", "ISIN"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing portfolio columns: " + ", ".join(sorted(missing)))
    df["Controvalore_num"] = df["Controvalore"].apply(euro_to_float)
    df["Peso_portafoglio"] = df["Controvalore_num"] / df["Controvalore_num"].sum()
    return df


def classify_etfs(portfolio):
    equity = portfolio[portfolio["Asset Class"].str.contains("Azionario", case=False, na=False)]
    bonds = portfolio[portfolio["Asset Class"].str.contains("Bond|Inflazione", case=False, na=False)]
    return equity, bonds


def enrich(portfolio, holdings):
    rows = []
    for _, etf in portfolio.iterrows():
        h = holdings.get(etf["ISIN"])
        if h is None or h.empty:
            continue
        h = h.copy()
        h.insert(0, "ISIN_ETF", etf["ISIN"])
        h.insert(1, "ETF", etf["Strumenti"])
        h.insert(2, "Asset_class_ETF", etf["Asset Class"])
        h["Peso_ETF_portafoglio"] = etf["Peso_portafoglio"]
        h["Peso_effettivo_portafoglio"] = h["Peso_holding"] * etf["Peso_portafoglio"]
        rows.append(h)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio = load_portfolio(Path(args.portfolio).expanduser())
    equity_etfs, bond_etfs = classify_etfs(portfolio)

    holdings = {}
    sources = []
    portfolio_isins = set(portfolio["ISIN"].astype(str))
    for isin, url in HOLDINGS_URLS.items():
        if isin not in portfolio_isins or not url:
            continue
        try:
            raw = read_provider_file(url)
            holdings[isin] = normalize_holdings(raw)
            sources.append({"ISIN": isin, "URL": url, "Stato": "caricato", "Righe": len(holdings[isin])})
        except Exception as exc:
            sources.append({"ISIN": isin, "URL": url, "Stato": f"errore: {exc}", "Righe": 0})

    equity = enrich(equity_etfs, holdings)
    bonds = enrich(bond_etfs, holdings)

    top_equity = pd.DataFrame(columns=["Holding", "Peso_effettivo_portafoglio"])
    if not equity.empty:
        top_equity = equity.dropna(subset=["Holding"]).groupby("Holding")["Peso_effettivo_portafoglio"].sum().sort_values(ascending=False).reset_index()

    top_obligors = pd.DataFrame(columns=["Obligor", "Peso_effettivo_portafoglio", "Numero_posizioni"])
    if not bonds.empty:
        weights = bonds.dropna(subset=["Obligor"]).groupby("Obligor")["Peso_effettivo_portafoglio"].sum().reset_index()
        counts = bonds.dropna(subset=["Obligor"]).groupby("Obligor")["Holding"].count().reset_index().rename(columns={"Holding": "Numero_posizioni"})
        top_obligors = weights.merge(counts, on="Obligor", how="left").sort_values("Peso_effettivo_portafoglio", ascending=False)

    top_equity.to_csv(output_dir / "top_societa_azionarie.csv", index=False)
    equity.to_csv(output_dir / "holdings_azionarie_dettaglio.csv", index=False)
    bonds.to_csv(output_dir / "holdings_obbligazionarie_dettaglio.csv", index=False)
    top_obligors.to_csv(output_dir / "top_obligor_obbligazionari.csv", index=False)
    pd.DataFrame(sources).to_csv(output_dir / "fonti_holdings.csv", index=False)

    print("Generated reports in", output_dir)


if __name__ == "__main__":
    main()
