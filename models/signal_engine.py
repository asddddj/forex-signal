import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import os
import sys

sys.path.insert(0, '.')
from data.config import (
    DB_PATH, PAIRS, CURRENCIES, STRENGTH_PAIRS,
    NY_TIMEZONE, MIDNIGHT_HOUR, LIQUIDITY_PIP_ZONE,
    CONFLUENCE_LONG_THRESHOLD, CONFLUENCE_SHORT_THRESHOLD
)

# -------------------------------------------------------
# HELPERS
# -------------------------------------------------------

def load_table(table_name):
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def save_to_db(df, table_name, if_exists="replace"):
    if df is None or df.empty:
        print(f"  No data to save for {table_name}")
        return
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    conn.close()
    print(f"  Saved {len(df)} rows → '{table_name}'")


def get_pip_size(ticker):
    """JPY pairs use 0.01 pip, all others use 0.0001."""
    return 0.01 if "JPY" in ticker else 0.0001


# ═══════════════════════════════════════════════════════
# SIGNAL 1 — NY MIDNIGHT OPEN BIAS
# ═══════════════════════════════════════════════════════

def compute_midnight_open_bias(ticker):
    """
    The NY Midnight Open (00:00 New York time) is the reference
    price that defines the daily directional bias.

    Price ABOVE midnight open → institutional bias is BULLISH
    Price BELOW midnight open → institutional bias is BEARISH

    This is the primary filter — all other signals must align
    with midnight open bias to produce a valid trade signal.

    Score:
      +2 = strongly above (>20 pips)
      +1 = above (5–20 pips)
       0 = within 5 pips (no clear bias)
      -1 = below (5–20 pips)
      -2 = strongly below (>20 pips)
    """
    candles = load_table("candles")
    if candles.empty:
        return 0, "No candle data", None, None

    pair_candles = candles[
        (candles["ticker"] == ticker) &
        (candles["interval"] == "1h")
    ].copy()

    if pair_candles.empty:
        return 0, "No 1H candles for this pair", None, None

    pair_candles["datetime"] = pd.to_datetime(pair_candles["datetime"])

    # Convert to NY timezone
    ny_tz = pytz.timezone(NY_TIMEZONE)

    try:
        pair_candles["datetime_ny"] = pair_candles["datetime"].dt.tz_localize(
            "UTC", ambiguous="NaT", nonexistent="NaT"
        ).dt.tz_convert(ny_tz)
    except Exception:
        pair_candles["datetime_ny"] = pair_candles["datetime"].dt.tz_localize(
            ny_tz, ambiguous="NaT", nonexistent="NaT"
        )

    pair_candles = pair_candles.dropna(subset=["datetime_ny"])

    # Find today's midnight open (00:00 NY time)
    now_ny   = datetime.now(ny_tz)
    today_ny = now_ny.date()

    midnight_candles = pair_candles[
        (pair_candles["datetime_ny"].dt.date == today_ny) &
        (pair_candles["datetime_ny"].dt.hour == MIDNIGHT_HOUR) &
        (pair_candles["datetime_ny"].dt.minute == 0)
    ]

    # If today's midnight not found, use most recent midnight
    if midnight_candles.empty:
        yesterday_ny = today_ny - timedelta(days=1)
        midnight_candles = pair_candles[
            (pair_candles["datetime_ny"].dt.date == yesterday_ny) &
            (pair_candles["datetime_ny"].dt.hour == MIDNIGHT_HOUR)
        ]

    if midnight_candles.empty:
        return 0, "Midnight open candle not found", None, None

    midnight_open  = float(midnight_candles.iloc[0]["open"])
    current_price  = float(pair_candles.sort_values("datetime").iloc[-1]["close"])
    pip            = get_pip_size(ticker)
    diff_pips      = (current_price - midnight_open) / pip

    if diff_pips > 20:    score, bias = +2, "Strongly bullish"
    elif diff_pips > 5:   score, bias = +1, "Bullish"
    elif diff_pips > -5:  score, bias = 0,  "Neutral — within 5 pips of midnight open"
    elif diff_pips > -20: score, bias = -1, "Bearish"
    else:                 score, bias = -2, "Strongly bearish"

    detail = (f"{bias} | Midnight open: {midnight_open:.5f} | "
              f"Current: {current_price:.5f} | Diff: {diff_pips:+.1f} pips")

    return score, detail, midnight_open, current_price


# ═══════════════════════════════════════════════════════
# SIGNAL 2 — CURRENCY STRENGTH METER
# ═══════════════════════════════════════════════════════

def compute_currency_strength():
    """
    Measures the relative strength of each currency (USD, EUR, GBP, JPY etc.)
    independently across all 28 major pairs.

    Method:
    1. For each of the 28 pairs, calculate % change over last 24 candles (1H)
    2. For each currency, average its performance across all pairs it appears in
       (positive when it's the base currency, negative when it's the quote)
    3. Normalise scores to 0–100 range

    Score 70+ = strong currency
    Score 30- = weak currency

    For a pair signal:
    Base currency score - Quote currency score = net directional strength
    """
    candles = load_table("candles")
    if candles.empty:
        return {}, pd.DataFrame()

    h1 = candles[candles["interval"] == "1h"].copy()
    h1["datetime"] = pd.to_datetime(h1["datetime"])

    strength_scores = {c: [] for c in CURRENCIES}

    for ticker in STRENGTH_PAIRS:
        pair_data = h1[h1["ticker"] == ticker].sort_values("datetime").tail(25)
        if len(pair_data) < 2:
            continue

        # % change over last 24 hours
        old_price = float(pair_data.iloc[0]["close"])
        new_price = float(pair_data.iloc[-1]["close"])
        if old_price == 0:
            continue
        pct_change = (new_price - old_price) / old_price * 100

        # Clean ticker name (remove =X suffix)
        clean = ticker.replace("=X", "")
        if len(clean) != 6:
            continue
        base  = clean[:3]
        quote = clean[3:]

        # Base currency strengthens when pair goes up
        # Quote currency weakens when pair goes up
        if base in strength_scores:
            strength_scores[base].append(pct_change)
        if quote in strength_scores:
            strength_scores[quote].append(-pct_change)

    # Average scores per currency
    raw_scores = {}
    for currency, scores in strength_scores.items():
        raw_scores[currency] = np.mean(scores) if scores else 0.0

    # Normalise to 0–100
    values    = list(raw_scores.values())
    min_val   = min(values)
    max_val   = max(values)
    val_range = max_val - min_val if max_val != min_val else 1

    normalised = {
        c: round((raw_scores[c] - min_val) / val_range * 100, 1)
        for c in CURRENCIES
    }

    # Build a readable dataframe
    strength_df = pd.DataFrame([
        {"currency": c, "strength_score": normalised[c], "raw_score": round(raw_scores[c], 4)}
        for c in sorted(normalised, key=normalised.get, reverse=True)
    ])
    strength_df["computed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return normalised, strength_df


def get_strength_signal(ticker, strength_scores):
    """
    For a given pair, compute directional signal from currency strength.
    e.g. EURUSD: EUR_score - USD_score = net signal

    Score:
      +2 = base much stronger than quote (>30 point gap)
      +1 = base stronger (15–30 gap)
       0 = roughly equal (<15 gap)
      -1 = quote stronger (15–30 gap)
      -2 = quote much stronger (>30 gap)
    """
    clean = ticker.replace("=X", "")
    if len(clean) != 6:
        return 0, "Cannot parse pair"

    base  = clean[:3]
    quote = clean[3:]

    base_score  = strength_scores.get(base, 50)
    quote_score = strength_scores.get(quote, 50)
    diff        = base_score - quote_score

    if diff > 30:    score, label = +2, f"{base} much stronger than {quote}"
    elif diff > 15:  score, label = +1, f"{base} stronger than {quote}"
    elif diff > -15: score, label = 0,  f"{base} and {quote} roughly equal"
    elif diff > -30: score, label = -1, f"{quote} stronger than {base}"
    else:            score, label = -2, f"{quote} much stronger than {base}"

    detail = (f"{label} | {base}: {base_score:.0f} | "
              f"{quote}: {quote_score:.0f} | Diff: {diff:+.0f}")
    return score, detail


# ═══════════════════════════════════════════════════════
# SIGNAL 3 — FAIR VALUE GAP PROXIMITY
# ═══════════════════════════════════════════════════════

def compute_fvg_signal(ticker):
    """
    Checks if current price is inside or near an unmitigated FVG.

    Bullish FVG below current price = support → bullish signal
    Bearish FVG above current price = resistance → bearish signal

    FVGs on 4H timeframe are weighted more than 1H (more significant).

    Score:
      +2 = inside or within 5 pips of bullish FVG (4H)
      +1 = near bullish FVG (1H) or moderately close to 4H
       0 = no significant FVG nearby
      -1 = near bearish FVG (1H)
      -2 = inside or within 5 pips of bearish FVG (4H)
    """
    fvgs    = load_table("fair_value_gaps")
    candles = load_table("candles")

    if fvgs.empty or candles.empty:
        return 0, "No FVG data"

    # Get current price
    pair_candles = candles[
        (candles["ticker"] == ticker) &
        (candles["interval"] == "1h")
    ].sort_values("datetime")

    if pair_candles.empty:
        return 0, "No candle data"

    current_price = float(pair_candles.iloc[-1]["close"])
    pip           = get_pip_size(ticker)
    proximity_pip = 10   # pips to consider "near" an FVG

    # Filter unmitigated FVGs for this pair
    pair_fvgs = fvgs[
        (fvgs["ticker"] == ticker) &
        (~fvgs["mitigated"].astype(bool))
    ].copy()

    if pair_fvgs.empty:
        return 0, "No unmitigated FVGs found"

    best_score  = 0
    best_detail = "No significant FVG nearby"

    for _, fvg in pair_fvgs.iterrows():
        top    = float(fvg["fvg_top"])
        bottom = float(fvg["fvg_bottom"])
        tf     = fvg["interval"]
        ftype  = fvg["type"]
        weight = 2 if tf == "4h" else 1

        # Distance from current price to FVG in pips
        if ftype == "BULLISH":
            # Bullish FVG below price = support
            if current_price > bottom:
                dist_pips = (current_price - top) / pip
                if dist_pips <= 0:           # inside FVG
                    score = +2 if weight == 2 else +1
                    detail = f"Price INSIDE bullish FVG ({tf}) — strong support"
                elif dist_pips <= proximity_pip:
                    score = +1
                    detail = f"Near bullish FVG ({tf}) — {dist_pips:.1f} pips above"
                else:
                    continue
                if abs(score) > abs(best_score):
                    best_score, best_detail = score, detail

        elif ftype == "BEARISH":
            # Bearish FVG above price = resistance
            if current_price < top:
                dist_pips = (bottom - current_price) / pip
                if dist_pips <= 0:           # inside FVG
                    score = -2 if weight == 2 else -1
                    detail = f"Price INSIDE bearish FVG ({tf}) — strong resistance"
                elif dist_pips <= proximity_pip:
                    score = -1
                    detail = f"Near bearish FVG ({tf}) — {dist_pips:.1f} pips below"
                else:
                    continue
                if abs(score) > abs(best_score):
                    best_score, best_detail = score, detail

    return best_score, best_detail


# ═══════════════════════════════════════════════════════
# SIGNAL 4 — LIQUIDITY LEVEL PROXIMITY
# ═══════════════════════════════════════════════════════

def compute_liquidity_signal(ticker):
    """
    Detects if price is near a liquidity pool (PDH/PDL/PWH/PWL/EQH/EQL).

    Liquidity pools are where retail stop losses cluster.
    Institutions sweep these levels before reversing.

    Logic:
    - Price near a HIGH liquidity level + bullish bias
      → institutions likely to sweep UP then reverse down
      → bearish signal (counter-intuitive but correct)

    - Price near a LOW liquidity level + bearish bias
      → institutions likely to sweep DOWN then reverse up
      → bullish signal

    - After a sweep (price has just passed through the level)
      → strong reversal signal in opposite direction

    Score:
      +2 = just swept a low liquidity level (reversal up expected)
      +1 = approaching low liquidity (buy-side likely to be targeted)
       0 = no significant liquidity nearby
      -1 = approaching high liquidity (sell-side likely to be targeted)
      -2 = just swept a high liquidity level (reversal down expected)
    """
    liq     = load_table("liquidity_levels")
    candles = load_table("candles")

    if liq.empty or candles.empty:
        return 0, "No liquidity data"

    pair_liq     = liq[liq["ticker"] == ticker].copy()
    pair_candles = candles[
        (candles["ticker"] == ticker) &
        (candles["interval"] == "4h")
    ].sort_values("datetime")

    if pair_liq.empty or pair_candles.empty:
        return 0, "No data for this pair"

    current_price = float(pair_candles.iloc[-1]["close"])
    prev_price    = float(pair_candles.iloc[-2]["close"]) if len(pair_candles) > 1 else current_price
    pip           = get_pip_size(ticker)
    zone_pips     = LIQUIDITY_PIP_ZONE

    best_score  = 0
    best_detail = "No significant liquidity nearby"

    for _, level in pair_liq.iterrows():
        liq_price = float(level["price"])
        liq_type  = level["type"]
        liq_desc  = level["description"]
        dist_pips = abs(current_price - liq_price) / pip

        # High liquidity levels (PDH, PWH, EQH) — above price
        if liq_type in ["PDH", "PWH", "EQH"] and liq_price > current_price:
            if dist_pips <= zone_pips:
                # Check if price just swept through (previous candle was above)
                if prev_price > liq_price:
                    score  = -2
                    detail = f"Just swept {liq_desc} at {liq_price:.5f} — reversal down likely"
                else:
                    score  = -1
                    detail = f"Approaching {liq_desc} at {liq_price:.5f} ({dist_pips:.1f} pips away)"
                if abs(score) > abs(best_score):
                    best_score, best_detail = score, detail

        # Low liquidity levels (PDL, PWL, EQL) — below price
        elif liq_type in ["PDL", "PWL", "EQL"] and liq_price < current_price:
            if dist_pips <= zone_pips:
                if prev_price < liq_price:
                    score  = +2
                    detail = f"Just swept {liq_desc} at {liq_price:.5f} — reversal up likely"
                else:
                    score  = +1
                    detail = f"Approaching {liq_desc} at {liq_price:.5f} ({dist_pips:.1f} pips away)"
                if abs(score) > abs(best_score):
                    best_score, best_detail = score, detail

    return best_score, best_detail


# ═══════════════════════════════════════════════════════
# SIGNAL 5 — COT INSTITUTIONAL POSITIONING
# ═══════════════════════════════════════════════════════

def compute_cot_signal(ticker):
    """
    Uses CFTC Commitment of Traders data to determine
    institutional (smart money) directional bias.

    For a pair like EURUSD:
    - We look at EUR commercial positioning AND USD commercial positioning
    - EUR commercials net long AND USD commercials net short → strong long signal
    - Combined score = EUR_score - USD_score

    Score ranges from -2 to +2 per currency,
    combined pair score = average of (base_score - quote_score) / 2
    """
    cot = load_table("cot_data")
    if cot.empty:
        return 0, "No COT data available"

    clean = ticker.replace("=X", "")
    if len(clean) != 6:
        return 0, "Cannot parse pair"

    base  = clean[:3]
    quote = clean[3:]

    def currency_score(currency):
        latest = cot[cot["currency"] == currency].sort_values("date").tail(1)
        if latest.empty or "comm_net_pct" not in latest.columns:
            return 0, 50.0
        pct = float(latest.iloc[0]["comm_net_pct"])
        if pct >= 75:   return +2, pct
        elif pct >= 55: return +1, pct
        elif pct >= 45: return  0, pct
        elif pct >= 25: return -1, pct
        else:           return -2, pct

    base_score,  base_pct  = currency_score(base)
    quote_score, quote_pct = currency_score(quote)

    # Net signal: base bullish and quote bearish = long signal
    net = base_score - quote_score

    if net >= 3:    score, label = +2, "Strong institutional long bias"
    elif net >= 1:  score, label = +1, "Mild institutional long bias"
    elif net == 0:  score, label =  0, "Neutral institutional positioning"
    elif net >= -2: score, label = -1, "Mild institutional short bias"
    else:           score, label = -2, "Strong institutional short bias"

    detail = (f"{label} | {base} COT: {base_pct:.0f}% | "
              f"{quote} COT: {quote_pct:.0f}%")
    return score, detail


# ═══════════════════════════════════════════════════════
# CONFLUENCE ENGINE — combines all 5 signals
# ═══════════════════════════════════════════════════════

def compute_confluence(ticker, strength_scores):
    """
    Combines all 5 signal scores into one confluence score
    and generates a final directional signal.

    Weights:
      NY Midnight Open:   30% (primary bias filter)
      Currency Strength:  25%
      FVG Proximity:      20%
      Liquidity:          15%
      COT Positioning:    10%

    Final score range: -2 to +2
    Long signal:  score >= +1.5
    Short signal: score <= -1.5
    No trade:     between -1.5 and +1.5
    """
    print(f"\n  [{ticker}] Computing signals...")

    # Compute each signal
    mbo_score,  mbo_detail,  mbo_open, mbo_price = compute_midnight_open_bias(ticker)
    str_score,  str_detail  = get_strength_signal(ticker, strength_scores)
    fvg_score,  fvg_detail  = compute_fvg_signal(ticker)
    liq_score,  liq_detail  = compute_liquidity_signal(ticker)
    cot_score,  cot_detail  = compute_cot_signal(ticker)

    print(f"    NY Midnight Open:   {mbo_score:+d}  {mbo_detail[:60]}")
    print(f"    Currency Strength:  {str_score:+d}  {str_detail[:60]}")
    print(f"    FVG Proximity:      {fvg_score:+d}  {fvg_detail[:60]}")
    print(f"    Liquidity:          {liq_score:+d}  {liq_detail[:60]}")
    print(f"    COT Positioning:    {cot_score:+d}  {cot_detail[:60]}")

    # Weighted confluence score
    weights = {
        "midnight_open":     0.30,
        "currency_strength": 0.25,
        "fvg":               0.20,
        "liquidity":         0.15,
        "cot":               0.10,
    }
    scores = {
        "midnight_open":     mbo_score,
        "currency_strength": str_score,
        "fvg":               fvg_score,
        "liquidity":         liq_score,
        "cot":               cot_score,
    }

    confluence = sum(scores[k] * weights[k] for k in weights)
    confluence = round(confluence, 3)

    # Count how many signals agree with direction
    if confluence > 0:
        agreeing = sum(1 for s in scores.values() if s > 0)
    else:
        agreeing = sum(1 for s in scores.values() if s < 0)

    # Generate signal
    if confluence >= CONFLUENCE_LONG_THRESHOLD:
        signal    = "LONG"
        confidence = min(100, int((confluence / 2) * 100))
    elif confluence <= CONFLUENCE_SHORT_THRESHOLD:
        signal    = "SHORT"
        confidence = min(100, int((abs(confluence) / 2) * 100))
    else:
        signal    = "NO TRADE"
        confidence = 0

    # Estimate pip target based on ATR
    candles = load_table("candles")
    pip_target = 0
    if not candles.empty:
        pair_4h = candles[
            (candles["ticker"] == ticker) &
            (candles["interval"] == "4h") &
            (candles["atr"].notna())
        ].sort_values("datetime").tail(10)
        if not pair_4h.empty:
            avg_atr  = float(pair_4h["atr"].mean())
            pip      = get_pip_size(ticker)
            pip_target = round(avg_atr / pip * 1.5, 0)  # 1.5× ATR target

    print(f"\n    ── Confluence: {confluence:+.3f} ──")
    print(f"    ── Signal: {signal} | Confidence: {confidence}% ──")
    if pip_target > 0:
        print(f"    ── Pip target: ~{pip_target:.0f} pips ──")

    # Build result dict
    result = {
        "computed_at":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker":               ticker,
        "signal":               signal,
        "confluence_score":     confluence,
        "confidence_pct":       confidence,
        "pip_target":           pip_target,
        "signals_agreeing":     agreeing,
        # Individual scores
        "midnight_open_score":  mbo_score,
        "midnight_open_detail": mbo_detail,
        "midnight_open_price":  mbo_open,
        "current_price":        mbo_price,
        "strength_score":       str_score,
        "strength_detail":      str_detail,
        "fvg_score":            fvg_score,
        "fvg_detail":           fvg_detail,
        "liquidity_score":      liq_score,
        "liquidity_detail":     liq_detail,
        "cot_score":            cot_score,
        "cot_detail":           cot_detail,
    }
    return result


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("="*55)
    print("  Forex Signal Engine — Phase 3")
    print(f"  Pairs: {list(PAIRS.keys())}")
    print(f"  Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*55)

    # Compute currency strength once (used by all pairs)
    print("\nComputing currency strength...")
    strength_scores, strength_df = compute_currency_strength()

    if strength_scores:
        save_to_db(strength_df, "currency_strength")
        print("  Strength ranking:")
        for _, row in strength_df.iterrows():
            bar = "█" * int(row["strength_score"] / 10)
            print(f"    {row['currency']}  {bar:<10}  {row['strength_score']:.0f}")

    # Compute confluence for each pair
    all_signals = []
    for pair_name, ticker in PAIRS.items():
        result = compute_confluence(ticker, strength_scores)
        all_signals.append(result)

    # Save signals to database
    signals_df = pd.DataFrame(all_signals)
    save_to_db(signals_df, "signals")

    # Print final summary
    print("\n" + "="*55)
    print("  SIGNAL SUMMARY")
    print("="*55)
    for row in all_signals:
        pair   = row["ticker"].replace("=X","")
        sig    = row["signal"]
        conf   = row["confidence_pct"]
        score  = row["confluence_score"]
        agree  = row["signals_agreeing"]
        target = row["pip_target"]

        if sig == "LONG":
            arrow = "▲ LONG "
        elif sig == "SHORT":
            arrow = "▼ SHORT"
        else:
            arrow = "─ NO TRADE"

        print(f"\n  {pair}  {arrow}")
        print(f"  Confluence: {score:+.3f} | Confidence: {conf}% | "
              f"{agree}/5 signals agree | Target: ~{target:.0f} pips")
        print(f"  NY Open:  {row['midnight_open_detail'][:55]}")
        print(f"  Strength: {row['strength_detail'][:55]}")
        print(f"  FVG:      {row['fvg_detail'][:55]}")
        print(f"  Liq:      {row['liquidity_detail'][:55]}")
        print(f"  COT:      {row['cot_detail'][:55]}")

    print("\n" + "="*55)
    print(f"  Signals saved to database → 'signals' table")
    print("  Next step: python data/fetch_news.py")
    print("  Then:      streamlit run app/main.py")


if __name__ == "__main__":
    main()
