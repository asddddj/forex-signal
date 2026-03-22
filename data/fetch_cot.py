import sqlite3
import pandas as pd
import requests
import zipfile
import io
import os
import sys
from datetime import datetime

sys.path.insert(0, '.')
from data.config import DB_PATH

# -------------------------------------------------------
# CFTC Financial Futures — Disaggregated Format
#
# Smart money in this report = Asset Managers
# (large institutional funds: pension funds, mutual funds)
# These are the equivalent of "commercials" in legacy COT.
#
# Leveraged Money = hedge funds (also useful)
# Dealer         = banks/intermediaries
# NonRept        = small retail traders
# -------------------------------------------------------

COT_ANNUAL_URLS = {
    2022: "https://www.cftc.gov/files/dea/history/fut_fin_txt_2022.zip",
    2023: "https://www.cftc.gov/files/dea/history/fut_fin_txt_2023.zip",
    2024: "https://www.cftc.gov/files/dea/history/fut_fin_txt_2024.zip",
    2025: "https://www.cftc.gov/files/dea/history/fut_fin_txt_2025.zip",
}

# Exact market names confirmed from CFTC file
COT_MARKETS = {
    "EUR": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "GBP": "BRITISH POUND STERLING - CHICAGO MERCANTILE EXCHANGE",
    "JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "CHF": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",
    "CAD": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "AUD": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "NZD": "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "USD": "U.S. DOLLAR INDEX - ICE FUTURES U.S.",
}

# Exact column names confirmed from CFTC file
LONG_COL   = "Asset_Mgr_Positions_Long_All"
SHORT_COL  = "Asset_Mgr_Positions_Short_All"
LEVL_COL   = "Lev_Money_Positions_Long_All"
LEVS_COL   = "Lev_Money_Positions_Short_All"
DATE_COL   = "As_of_Date_In_Form_YYMMDD"
MARKET_COL = "Market_and_Exchange_Names"
OI_COL     = "Open_Interest_All"


def download_cot_year(year, url):
    """Download and filter one year of CFTC Financial Futures COT data."""
    print(f"  Downloading {year}...")
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  {year}: HTTP {resp.status_code} — skipping")
            return pd.DataFrame()

        z   = zipfile.ZipFile(io.BytesIO(resp.content))
        csv = z.read(z.namelist()[0]).decode("utf-8", errors="ignore")
        df  = pd.read_csv(io.StringIO(csv), low_memory=False)

        # Filter to currency markets only
        mask = df[MARKET_COL].isin(COT_MARKETS.values())
        df   = df[mask].copy()
        print(f"  {year}: {len(df)} currency rows found")
        return df

    except Exception as e:
        print(f"  {year}: Error — {e}")
        return pd.DataFrame()


def parse_cot(df):
    """
    Parse raw CFTC data into clean signal table.

    Asset Managers = institutional smart money (pension funds,
    sovereign wealth funds, large mutual funds). When they are
    heavily net long a currency → strong bullish institutional bias.

    Leveraged Money = hedge funds. These are trend followers —
    useful as a confirmation signal.

    Net position = longs - shorts
    52-week percentile = where current net sits vs past year
      100% = most bullish ever seen
        0% = most bearish ever seen
    """
    d = df.copy()

    # Parse date
    d["date"] = pd.to_datetime(
        d[DATE_COL].astype(str), format="%y%m%d", errors="coerce"
    )
    d = d.dropna(subset=["date"]).sort_values("date")

    # Convert numeric columns
    for col in [LONG_COL, SHORT_COL, LEVL_COL, LEVS_COL, OI_COL]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)

    # Compute net positions
    d["asset_mgr_net"] = d[LONG_COL]  - d[SHORT_COL]   # institutional net
    d["lev_money_net"] = d[LEVL_COL]  - d[LEVS_COL]    # hedge fund net

    # Combined smart money net (weighted: 70% asset mgr + 30% lev money)
    d["smart_money_net"] = (d["asset_mgr_net"] * 0.7 +
                            d["lev_money_net"] * 0.3).round(0).astype(int)

    # Map market name → currency code
    market_to_currency = {v: k for k, v in COT_MARKETS.items()}
    d["currency"] = d[MARKET_COL].map(market_to_currency)
    d = d.dropna(subset=["currency"])

    # 52-week rolling percentile per currency
    def rolling_percentile(series, window=52):
        out = []
        for i in range(len(series)):
            start = max(0, i - window + 1)
            w     = series.iloc[start:i+1]
            mn, mx = w.min(), w.max()
            if mx == mn:
                out.append(50.0)
            else:
                out.append(round((w.iloc[-1] - mn) / (mx - mn) * 100, 1))
        return out

    result_parts = []
    for currency in d["currency"].unique():
        cd = d[d["currency"] == currency].copy().sort_values("date")
        cd["comm_net"]     = cd["asset_mgr_net"]   # keep as comm_net for compatibility
        cd["comm_net_pct"] = rolling_percentile(cd["asset_mgr_net"])
        cd["lev_net_pct"]  = rolling_percentile(cd["lev_money_net"])
        result_parts.append(cd)

    if not result_parts:
        return pd.DataFrame()

    result = pd.concat(result_parts, ignore_index=True)

    # Select final columns
    keep = [
        "date", "currency", MARKET_COL,
        LONG_COL, SHORT_COL, "asset_mgr_net", "comm_net_pct",
        LEVL_COL, LEVS_COL, "lev_money_net", "lev_net_pct",
        "smart_money_net", OI_COL,
        "comm_net",   # alias for dashboard compatibility
    ]
    keep   = [c for c in keep if c in result.columns]
    result = result[keep].copy()
    result = result.rename(columns={MARKET_COL: "market", OI_COL: "open_interest"})
    result["date"]       = result["date"].dt.strftime("%Y-%m-%d")
    result["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return result


def get_cot_signal(currency, cot_df):
    """
    Convert latest COT percentile into a score -2 to +2.
    Uses comm_net_pct (asset manager 52-week percentile).

    +2 = above 75th percentile (heavy institutional longs)
    +1 = 55th–75th
     0 = 45th–55th (neutral)
    -1 = 25th–45th
    -2 = below 25th (heavy institutional shorts)
    """
    latest = cot_df[cot_df["currency"] == currency].sort_values("date").tail(1)
    if latest.empty or "comm_net_pct" not in latest.columns:
        return 0, "No COT data"

    pct   = float(latest.iloc[0]["comm_net_pct"])
    net   = int(latest.iloc[0].get("comm_net", 0))
    dated = latest.iloc[0]["date"]

    if pct >= 75:   score, label = +2, "Strong bullish — institutions heavily long"
    elif pct >= 55: score, label = +1, "Mild bullish"
    elif pct >= 45: score, label =  0, "Neutral"
    elif pct >= 25: score, label = -1, "Mild bearish"
    else:           score, label = -2, "Strong bearish — institutions heavily short"

    return score, f"{label} | Net: {net:,} | Percentile: {pct:.0f}% [{dated}]"


def save_to_db(df, table_name):
    if df.empty:
        print(f"  No data to save for {table_name}")
        return
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"  Saved {len(df)} rows → '{table_name}'")


def main():
    print("=" * 55)
    print("  Forex Signals — COT Data Pipeline")
    print("  Source: CFTC Financial Futures (Disaggregated)")
    print("  Smart money = Asset Managers (institutional funds)")
    print("=" * 55)

    all_years = []
    for year, url in COT_ANNUAL_URLS.items():
        df = download_cot_year(year, url)
        if not df.empty:
            all_years.append(df)

    if not all_years:
        print("\nERROR: No data downloaded.")
        return

    combined = pd.concat(all_years, ignore_index=True)
    print(f"\nTotal raw rows: {len(combined)}")

    parsed = parse_cot(combined)
    if parsed.empty:
        print("ERROR: Parsing returned empty.")
        return

    print(f"Parsed rows:    {len(parsed)}")
    print(f"Currencies:     {sorted(parsed['currency'].unique().tolist())}")
    print(f"Columns:        {parsed.columns.tolist()}")

    save_to_db(parsed, "cot_data")

    # Print signals
    print("\n" + "=" * 55)
    print("  Latest COT signals (institutional positioning):")
    print("=" * 55)
    for currency in sorted(parsed["currency"].unique()):
        score, label = get_cot_signal(currency, parsed)
        arrow = "▲" if score > 0 else ("▼" if score < 0 else "─")
        print(f"  {currency}  {arrow}  Score: {score:+d}  |  {label}")

    print(f"\n  Saved to: {DB_PATH} → table: cot_data")
    print("  Next: python models/signal_engine.py")


if __name__ == "__main__":
    main()
