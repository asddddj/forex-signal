import sqlite3
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import sys

sys.path.insert(0, '.')
from data.config import DB_PATH, PAIRS, TIMEFRAMES, STRENGTH_PAIRS

# -------------------------------------------------------
# STEP 1 — Download OHLCV price data
# -------------------------------------------------------

def download_pair(ticker, interval, years_back=2):
    """
    Download historical OHLCV candles for one pair + timeframe.
    yfinance limits: 1h = last 730 days, 4h/1d = much longer.
    We loop in chunks to get maximum history.
    """
    all_chunks = []
    end_date   = datetime.now()

    # Chunk sizes by timeframe (yfinance limits)
    chunk_map  = {"1h": 59, "4h": 59, "1d": 365}
    chunk_days = chunk_map.get(interval, 59)
    start_date = end_date - timedelta(days=years_back * 365)
    current_end = end_date

    while current_end > start_date:
        current_start = max(current_end - timedelta(days=chunk_days), start_date)
        try:
            df = yf.download(
                ticker,
                start=current_start.strftime("%Y-%m-%d"),
                end=current_end.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )
            if not df.empty:
                all_chunks.append(df)
        except Exception as e:
            print(f"    Chunk error: {e}")
        current_end = current_start
        time.sleep(1)

    if not all_chunks:
        return pd.DataFrame()

    combined = pd.concat(all_chunks)
    combined = combined[~combined.index.duplicated(keep="first")].sort_index()

    # Flatten MultiIndex columns (yfinance sometimes returns these)
    if isinstance(combined.columns, pd.MultiIndex):
        combined.columns = combined.columns.get_level_values(0)

    combined = combined.rename(columns={
        "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
    })
    combined.index.name = "datetime"
    combined = combined.reset_index()
    combined["datetime"] = pd.to_datetime(combined["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    combined["ticker"]   = ticker
    combined["interval"] = interval

    for col in ["open","high","low","close"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").round(5)

    return combined[["datetime","ticker","interval","open","high","low","close","volume"]]


# -------------------------------------------------------
# STEP 2 — Compute technical indicators per candle
# -------------------------------------------------------

def compute_indicators(df):
    """
    Add technical indicators as extra columns.
    These are used as features in the Phase 3 ML model
    and as supporting context in the dashboard charts.
    """
    d = df.sort_values("datetime").copy()
    for col in ["open","high","low","close","volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    # Trend
    d["ema_8"]    = ta.ema(d["close"], length=8)
    d["ema_21"]   = ta.ema(d["close"], length=21)
    d["ema_50"]   = ta.ema(d["close"], length=50)
    d["ema_200"]  = ta.ema(d["close"], length=200)
    d["ema_diff"] = d["ema_8"] - d["ema_21"]

    # Momentum
    d["rsi"]      = ta.rsi(d["close"], length=14)
    macd          = ta.macd(d["close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        d["macd"]      = macd.iloc[:,0]
        d["macd_sig"]  = macd.iloc[:,2]
        d["macd_hist"] = macd.iloc[:,1]

    # Volatility
    d["atr"]      = ta.atr(d["high"], d["low"], d["close"], length=14)
    bb            = ta.bbands(d["close"], length=20, std=2)
    if bb is not None and not bb.empty:
        d["bb_upper"] = bb.iloc[:,0]
        d["bb_mid"]   = bb.iloc[:,1]
        d["bb_lower"] = bb.iloc[:,2]
        d["bb_pct"]   = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"])

    # Candle features
    d["body"]     = abs(d["close"] - d["open"])
    d["candle_range"] = d["high"] - d["low"]
    d["direction"]= (d["close"] >= d["open"]).astype(int)

    # Returns
    d["return_1c"]  = d["close"].pct_change(1)
    d["return_4c"]  = d["close"].pct_change(4)
    d["return_24c"] = d["close"].pct_change(24)

    # Price position in recent range
    d["high_20"]  = d["high"].rolling(20).max()
    d["low_20"]   = d["low"].rolling(20).min()
    d["pos_20"]   = ((d["close"] - d["low_20"]) /
                     (d["high_20"] - d["low_20"]).replace(0, float("nan")))

    return d


# -------------------------------------------------------
# STEP 3 — Detect Fair Value Gaps (FVG)
# -------------------------------------------------------

def detect_fvgs(df, lookback=200):
    """
    Fair Value Gap = a 3-candle pattern where price moves so
    fast it leaves an imbalance (gap) in the chart.

    Bullish FVG: candle[i-2] high < candle[i] low
      → gap between top of 2-candles-ago and bottom of current
    Bearish FVG: candle[i-2] low > candle[i] high
      → gap between bottom of 2-candles-ago and top of current

    These gaps act as magnets — price almost always returns to fill them.
    We store unmitigated (unfilled) FVGs as entry zones.
    """
    d    = df.sort_values("datetime").reset_index(drop=True)
    fvgs = []

    # Only scan the most recent N candles for efficiency
    start_idx = max(2, len(d) - lookback)

    for i in range(start_idx, len(d)):
        c_prev2 = d.iloc[i-2]   # 2 candles ago
        c_curr  = d.iloc[i]     # current candle

        # Bullish FVG — gap above candle[i-2] high
        if c_prev2["high"] < c_curr["low"]:
            fvg_top    = c_curr["low"]
            fvg_bottom = c_prev2["high"]
            fvgs.append({
                "datetime":   c_curr["datetime"],
                "ticker":     c_curr["ticker"],
                "interval":   c_curr["interval"],
                "type":       "BULLISH",
                "fvg_top":    round(fvg_top, 5),
                "fvg_bottom": round(fvg_bottom, 5),
                "fvg_size":   round((fvg_top - fvg_bottom) * 10000, 1),  # in pips
                "mitigated":  False,
            })

        # Bearish FVG — gap below candle[i-2] low
        elif c_prev2["low"] > c_curr["high"]:
            fvg_top    = c_prev2["low"]
            fvg_bottom = c_curr["high"]
            fvgs.append({
                "datetime":   c_curr["datetime"],
                "ticker":     c_curr["ticker"],
                "interval":   c_curr["interval"],
                "type":       "BEARISH",
                "fvg_top":    round(fvg_top, 5),
                "fvg_bottom": round(fvg_bottom, 5),
                "fvg_size":   round((fvg_top - fvg_bottom) * 10000, 1),
                "mitigated":  False,
            })

    # Mark FVGs as mitigated if current price has passed through them
    if fvgs and not d.empty:
        current_price = d.iloc[-1]["close"]
        for fvg in fvgs:
            if fvg["type"] == "BULLISH" and current_price <= fvg["fvg_bottom"]:
                fvg["mitigated"] = True
            elif fvg["type"] == "BEARISH" and current_price >= fvg["fvg_top"]:
                fvg["mitigated"] = True

    return pd.DataFrame(fvgs)


# -------------------------------------------------------
# STEP 4 — Detect Liquidity Levels
# -------------------------------------------------------

def detect_liquidity_levels(df):
    """
    Liquidity levels are price zones where retail stop losses cluster.
    Institutions sweep these before reversing.

    We detect:
    - Previous Day High / Low  (PDH / PDL)
    - Previous Week High / Low (PWH / PWL)
    - Equal Highs / Equal Lows (within 5 pips of each other)
    """
    d = df.sort_values("datetime").copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    levels = []

    ticker   = d["ticker"].iloc[0]
    interval = d["interval"].iloc[0]

    # ── Previous Day High / Low ────────────────────────
    d["date"] = d["datetime"].dt.date
    daily     = d.groupby("date").agg(
        day_high=("high","max"),
        day_low=("low","min")
    ).reset_index()

    if len(daily) >= 2:
        prev_day = daily.iloc[-2]
        levels.append({
            "ticker": ticker, "interval": interval,
            "type": "PDH", "price": round(float(prev_day["day_high"]), 5),
            "description": "Previous Day High",
        })
        levels.append({
            "ticker": ticker, "interval": interval,
            "type": "PDL", "price": round(float(prev_day["day_low"]), 5),
            "description": "Previous Day Low",
        })

    # ── Previous Week High / Low ───────────────────────
    d["week"] = d["datetime"].dt.isocalendar().week
    d["year"] = d["datetime"].dt.year
    weekly    = d.groupby(["year","week"]).agg(
        week_high=("high","max"),
        week_low=("low","min")
    ).reset_index()

    if len(weekly) >= 2:
        prev_week = weekly.iloc[-2]
        levels.append({
            "ticker": ticker, "interval": interval,
            "type": "PWH", "price": round(float(prev_week["week_high"]), 5),
            "description": "Previous Week High",
        })
        levels.append({
            "ticker": ticker, "interval": interval,
            "type": "PWL", "price": round(float(prev_week["week_low"]), 5),
            "description": "Previous Week Low",
        })

    # ── Equal Highs / Equal Lows (swing points) ────────
    # Find swing highs and lows in recent 100 candles
    recent = d.tail(100).reset_index(drop=True)
    pip    = 0.0001 if "JPY" not in ticker else 0.01
    zone   = pip * 5   # 5 pips tolerance for "equal"

    for i in range(2, len(recent)-2):
        # Swing high: higher than 2 candles each side
        if (recent.iloc[i]["high"] > recent.iloc[i-1]["high"] and
            recent.iloc[i]["high"] > recent.iloc[i-2]["high"] and
            recent.iloc[i]["high"] > recent.iloc[i+1]["high"] and
            recent.iloc[i]["high"] > recent.iloc[i+2]["high"]):
            # Check if another swing high is within zone
            for j in range(max(0,i-20), i):
                if (recent.iloc[j]["high"] > recent.iloc[max(0,j-1)]["high"] and
                    abs(recent.iloc[j]["high"] - recent.iloc[i]["high"]) <= zone):
                    levels.append({
                        "ticker": ticker, "interval": interval,
                        "type": "EQH",
                        "price": round(float(recent.iloc[i]["high"]), 5),
                        "description": "Equal Highs (Buy-side Liquidity)",
                    })
                    break

        # Swing low: lower than 2 candles each side
        if (recent.iloc[i]["low"] < recent.iloc[i-1]["low"] and
            recent.iloc[i]["low"] < recent.iloc[i-2]["low"] and
            recent.iloc[i]["low"] < recent.iloc[i+1]["low"] and
            recent.iloc[i]["low"] < recent.iloc[i+2]["low"]):
            for j in range(max(0,i-20), i):
                if (recent.iloc[j]["low"] < recent.iloc[max(0,j-1)]["low"] and
                    abs(recent.iloc[j]["low"] - recent.iloc[i]["low"]) <= zone):
                    levels.append({
                        "ticker": ticker, "interval": interval,
                        "type": "EQL",
                        "price": round(float(recent.iloc[i]["low"]), 5),
                        "description": "Equal Lows (Sell-side Liquidity)",
                    })
                    break

    return pd.DataFrame(levels) if levels else pd.DataFrame()


# -------------------------------------------------------
# STEP 5 — Save to SQLite
# -------------------------------------------------------

def save_to_db(df, table_name, if_exists="replace"):
    if df is None or df.empty:
        print(f"  No data to save for {table_name}")
        return
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.close()
    print(f"  Saved {len(df)} rows → '{table_name}'")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    print("="*55)
    print("  Forex Signals — Phase 2: Price Data Pipeline")
    print(f"  Pairs: {list(PAIRS.keys())}")
    print(f"  Timeframes: 1H, 4H, Daily")
    print("="*55)

    all_candles  = []
    all_fvgs     = []
    all_liquidity = []

    for pair_name, ticker in PAIRS.items():
        print(f"\n[{pair_name}]")

        for tf_name in ["1h", "4h", "1d"]:
            print(f"  Downloading {tf_name}...")
            years = 2 if tf_name == "1h" else 3
            raw   = download_pair(ticker, tf_name, years_back=years)

            if raw.empty:
                print(f"  No data for {pair_name} {tf_name} — skipping")
                continue

            # Add indicators
            enriched = compute_indicators(raw)
            all_candles.append(enriched)
            print(f"  {tf_name}: {len(enriched)} candles with indicators")

            # Detect FVGs on 1H and 4H only
            if tf_name in ["1h", "4h"]:
                fvgs = detect_fvgs(enriched)
                if not fvgs.empty:
                    all_fvgs.append(fvgs)
                    unmitigated = fvgs[~fvgs["mitigated"]]
                    print(f"  FVGs: {len(fvgs)} total, "
                          f"{len(unmitigated)} unmitigated")

            # Detect liquidity levels on 4H only
            if tf_name == "4h":
                liq = detect_liquidity_levels(enriched)
                if not liq.empty:
                    all_liquidity.append(liq)
                    print(f"  Liquidity levels: {len(liq)} detected")

        time.sleep(2)  # pause between pairs

    # Save all data
    print("\nSaving to database...")
    if all_candles:
        candles_df = pd.concat(all_candles, ignore_index=True)
        save_to_db(candles_df, "candles")
        print(f"  Total candles saved: {len(candles_df)}")

    if all_fvgs:
        fvgs_df = pd.concat(all_fvgs, ignore_index=True)
        save_to_db(fvgs_df, "fair_value_gaps")
        unmitigated = fvgs_df[~fvgs_df["mitigated"]]
        print(f"  Total FVGs: {len(fvgs_df)} "
              f"({len(unmitigated)} unmitigated)")

    if all_liquidity:
        liq_df = pd.concat(all_liquidity, ignore_index=True)
        save_to_db(liq_df, "liquidity_levels")

    # Print database summary
    print("\n" + "="*55)
    print("  Database summary:")
    conn   = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    for t in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  • {t}: {count} rows")
    conn.close()
    print("\n  Next step: python data/fetch_cot.py")


if __name__ == "__main__":
    main()
