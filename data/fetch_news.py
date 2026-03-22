import sqlite3
import pandas as pd
import requests
import feedparser
import pytz
import os
import sys
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

sys.path.insert(0, '.')
from data.config import (
    DB_PATH, FOREX_FACTORY_RSS,
    HIGH_IMPACT_SUPPRESS_HOURS, NEWS_WARNING_HOURS
)

NY_TZ  = pytz.timezone("America/New_York")
UTC_TZ = pytz.utc

# -------------------------------------------------------
# CURRENCIES AFFECTED BY EACH PAIR
# -------------------------------------------------------

PAIR_CURRENCIES = {
    "EURUSD=X": ["EUR", "USD"],
    "GBPUSD=X": ["GBP", "USD"],
    "USDJPY=X": ["USD", "JPY"],
}

# High-impact event keywords — these always suppress signals
HIGH_IMPACT_KEYWORDS = [
    "NFP", "Non-Farm", "FOMC", "Fed Rate", "Federal Reserve",
    "CPI", "Inflation", "BOE Rate", "ECB Rate", "BOJ Rate",
    "GDP", "Employment Change", "Unemployment Rate",
    "Retail Sales", "Interest Rate Decision",
    "Central Bank", "Press Conference", "Powell", "Lagarde", "Bailey",
]

# Medium-impact keywords — warn but don't suppress
MEDIUM_IMPACT_KEYWORDS = [
    "PMI", "ISM", "Trade Balance", "Current Account",
    "Consumer Confidence", "Business Confidence",
    "Manufacturing", "Services", "Housing",
    "Durable Goods", "Factory Orders",
]


# ═══════════════════════════════════════════════════════
# SOURCE 1 — ForexFactory RSS Feed
# ═══════════════════════════════════════════════════════

def fetch_forexfactory_rss():
    """
    Fetch economic calendar events from ForexFactory RSS.
    Returns a list of event dicts with time, currency, title, impact.
    """
    print("  Fetching ForexFactory RSS...")
    events = []

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36"
        }
        resp = requests.get(FOREX_FACTORY_RSS, headers=headers, timeout=15)

        if resp.status_code != 200:
            print(f"  ForexFactory RSS: HTTP {resp.status_code}")
            return events

        feed = feedparser.parse(resp.content)

        for entry in feed.entries:
            try:
                title    = entry.get("title", "")
                summary  = entry.get("summary", "")
                link     = entry.get("link", "")

                # Parse time from title or published field
                pub_time = None
                if hasattr(entry, "published"):
                    try:
                        pub_time = pd.to_datetime(
                            entry.published, utc=True
                        ).to_pydatetime()
                    except Exception:
                        pass

                # Extract currency from title (e.g. "USD - NFP")
                currency = "ALL"
                for curr in ["USD","EUR","GBP","JPY","AUD","CAD","CHF","NZD"]:
                    if curr in title.upper():
                        currency = curr
                        break

                # Determine impact level
                impact = classify_impact(title + " " + summary)

                events.append({
                    "source":   "ForexFactory",
                    "datetime": pub_time.strftime("%Y-%m-%d %H:%M:%S")
                              if pub_time else None,
                    "currency": currency,
                    "title":    title[:200],
                    "impact":   impact,
                    "link":     link,
                })
            except Exception:
                continue

        print(f"  ForexFactory: {len(events)} events parsed")

    except Exception as e:
        print(f"  ForexFactory RSS error: {e}")

    return events


# ═══════════════════════════════════════════════════════
# SOURCE 2 — Fallback: Build event list from known schedule
# ═══════════════════════════════════════════════════════

def get_scheduled_events():
    """
    Fallback scheduled events for the next 7 days.
    Based on known recurring economic releases.
    Used when RSS feed is unavailable.

    This covers the most market-moving events for EUR/USD, GBP/USD, USD/JPY.
    """
    now    = datetime.now(UTC_TZ)
    events = []

    # US Events (affect USD pairs)
    us_events = [
        # Day, Hour (UTC), Title, Impact
        (4,  13, 30, "USD", "Non-Farm Payrolls",        "HIGH"),
        (4,  13, 30, "USD", "Unemployment Rate",        "HIGH"),
        (2,  14,  0, "USD", "ISM Manufacturing PMI",    "MEDIUM"),
        (3,  18,  0, "USD", "FOMC Meeting Minutes",     "HIGH"),
        (3,  13, 30, "USD", "ADP Employment Change",    "MEDIUM"),
        (4,  14,  0, "USD", "ISM Services PMI",         "MEDIUM"),
        (2,  13, 30, "USD", "CPI Month-over-Month",     "HIGH"),
        (4,  13, 30, "USD", "Core PCE Price Index",     "HIGH"),
        (3,  13, 30, "USD", "GDP Growth Rate",          "HIGH"),
        (3,  13, 30, "USD", "Retail Sales",             "HIGH"),
    ]

    # EU Events (affect EUR pairs)
    eu_events = [
        (1,  9,  0, "EUR", "ECB Interest Rate Decision","HIGH"),
        (1,  9, 30, "EUR", "ECB Press Conference",      "HIGH"),
        (1,  9,  0, "EUR", "CPI Flash Estimate",        "HIGH"),
        (2,  9,  0, "EUR", "Manufacturing PMI",         "MEDIUM"),
        (3,  9,  0, "EUR", "Services PMI",              "MEDIUM"),
        (4,  9,  0, "EUR", "GDP Growth Rate",           "HIGH"),
        (3, 10,  0, "EUR", "German Ifo Business Climate","MEDIUM"),
    ]

    # UK Events (affect GBP pairs)
    uk_events = [
        (4, 12,  0, "GBP", "BOE Interest Rate Decision","HIGH"),
        (4, 12, 30, "GBP", "BOE Press Conference",      "HIGH"),
        (2,  7,  0, "GBP", "CPI Year-over-Year",        "HIGH"),
        (2,  7,  0, "GBP", "Manufacturing PMI",         "MEDIUM"),
        (3,  7,  0, "GBP", "GDP Month-over-Month",      "HIGH"),
        (4,  7,  0, "GBP", "Retail Sales",              "HIGH"),
    ]

    # JP Events (affect JPY pairs)
    jp_events = [
        (1,  3,  0, "JPY", "BOJ Interest Rate Decision","HIGH"),
        (2, 23, 30, "JPY", "CPI National",              "HIGH"),
        (2, 23, 30, "JPY", "Manufacturing PMI",         "MEDIUM"),
        (3,  0, 50, "JPY", "GDP Growth Rate",           "HIGH"),
    ]

    all_scheduled = us_events + eu_events + uk_events + jp_events

    for weekday, hour, minute, currency, title, impact in all_scheduled:
        # Find next occurrence of this weekday
        days_ahead = weekday - now.weekday()
        if days_ahead < 0:
            days_ahead += 7
        event_date = now + timedelta(days=days_ahead)
        event_time = event_date.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        events.append({
            "source":   "Scheduled",
            "datetime": event_time.strftime("%Y-%m-%d %H:%M:%S"),
            "currency": currency,
            "title":    title,
            "impact":   impact,
            "link":     "",
        })

    return events


def classify_impact(text):
    """Classify event as HIGH, MEDIUM, or LOW impact."""
    text_upper = text.upper()
    for kw in HIGH_IMPACT_KEYWORDS:
        if kw.upper() in text_upper:
            return "HIGH"
    for kw in MEDIUM_IMPACT_KEYWORDS:
        if kw.upper() in text_upper:
            return "MEDIUM"
    return "LOW"


# ═══════════════════════════════════════════════════════
# COMBINE AND SAVE NEWS
# ═══════════════════════════════════════════════════════

def fetch_all_news():
    """
    Fetch from all sources and combine into one clean event list.
    Deduplicates by title + date.
    """
    all_events = []

    # Try ForexFactory RSS first
    ff_events = fetch_forexfactory_rss()
    all_events.extend(ff_events)

    # Always add scheduled events as backup coverage
    scheduled = get_scheduled_events()
    all_events.extend(scheduled)

    if not all_events:
        return pd.DataFrame()

    df = pd.DataFrame(all_events)

    # Only keep HIGH and MEDIUM impact
    df = df[df["impact"].isin(["HIGH","MEDIUM"])].copy()

    # Parse datetimes
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    # Only keep events in next 7 days
    now         = pd.Timestamp.now(tz="UTC")
    week_ahead  = now + pd.Timedelta(days=7)
    df["datetime"] = df["datetime"].dt.tz_localize("UTC", ambiguous="NaT",
                                                    nonexistent="NaT")
    df = df.dropna(subset=["datetime"])
    df = df[(df["datetime"] >= now) & (df["datetime"] <= week_ahead)]

    # Deduplicate
    df = df.drop_duplicates(subset=["currency","title"]).sort_values("datetime")

    # Convert to NY time for display
    ny_tz = pytz.timezone("America/New_York")
    df["datetime_ny"] = df["datetime"].dt.tz_convert(ny_tz).dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return df


# ═══════════════════════════════════════════════════════
# SIGNAL FILTER — check news impact on each signal
# ═══════════════════════════════════════════════════════

def check_news_impact(ticker, news_df):
    """
    For a given pair, check if any upcoming high/medium impact
    news event should suppress or warn the signal.

    Returns:
      status = "SUPPRESSED" | "WARNING" | "CLEAR"
      reason = explanation string
      events = list of conflicting events
    """
    if news_df.empty:
        return "CLEAR", "No news data available", []

    currencies   = PAIR_CURRENCIES.get(ticker, [])
    now_utc      = datetime.now(UTC_TZ)
    news_df_copy = news_df.copy()

    # Parse datetimes back
    news_df_copy["datetime"] = pd.to_datetime(
        news_df_copy["datetime"], utc=True, errors="coerce"
    )
    news_df_copy = news_df_copy.dropna(subset=["datetime"])

    # Filter to events affecting this pair's currencies
    relevant = news_df_copy[
        news_df_copy["currency"].isin(currencies + ["ALL"])
    ].copy()

    if relevant.empty:
        return "CLEAR", "No relevant news in next 48 hours", []

    conflicting = []
    for _, event in relevant.iterrows():
        event_time  = event["datetime"]
        hours_until = (event_time - now_utc).total_seconds() / 3600

        if hours_until < 0:
            continue  # event already passed

        if hours_until <= HIGH_IMPACT_SUPPRESS_HOURS and event["impact"] == "HIGH":
            conflicting.append({
                "event":   event["title"],
                "currency":event["currency"],
                "time_ny": event.get("datetime_ny",""),
                "impact":  event["impact"],
                "hours":   round(hours_until, 1),
                "action":  "SUPPRESS",
            })
        elif hours_until <= NEWS_WARNING_HOURS:
            conflicting.append({
                "event":   event["title"],
                "currency":event["currency"],
                "time_ny": event.get("datetime_ny",""),
                "impact":  event["impact"],
                "hours":   round(hours_until, 1),
                "action":  "WARN" if event["impact"] == "HIGH" else "INFO",
            })

    if not conflicting:
        return "CLEAR", "No conflicting news in next 48 hours", []

    # Determine worst status
    actions = [e["action"] for e in conflicting]
    if "SUPPRESS" in actions:
        status = "SUPPRESSED"
        reason = (f"HIGH impact news within {HIGH_IMPACT_SUPPRESS_HOURS}h: "
                  f"{conflicting[0]['event']} ({conflicting[0]['currency']}) "
                  f"at {conflicting[0]['time_ny']} NY")
    elif "WARN" in actions:
        status = "WARNING"
        reason = (f"High impact news in {conflicting[0]['hours']:.1f}h: "
                  f"{conflicting[0]['event']} ({conflicting[0]['currency']}) "
                  f"at {conflicting[0]['time_ny']} NY")
    else:
        status = "INFO"
        reason = (f"Medium impact news upcoming: "
                  f"{conflicting[0]['event']} ({conflicting[0]['currency']})")

    return status, reason, conflicting


# ═══════════════════════════════════════════════════════
# APPLY NEWS FILTER TO SAVED SIGNALS
# ═══════════════════════════════════════════════════════

def apply_news_filter(news_df):
    """
    Load the latest signals from the database,
    apply the news filter to each one,
    and update the signals table with news status.
    """
    try:
        conn    = sqlite3.connect(DB_PATH)
        signals = pd.read_sql("SELECT * FROM signals", conn)
        conn.close()
    except Exception:
        print("  No signals table found — run signal_engine.py first")
        return

    if signals.empty:
        print("  No signals to filter")
        return

    news_statuses = []
    for _, row in signals.iterrows():
        ticker = row["ticker"]
        status, reason, events = check_news_impact(ticker, news_df)

        # If signal was LONG or SHORT but news suppresses it
        original_signal = row["signal"]
        final_signal    = original_signal

        if status == "SUPPRESSED" and original_signal in ["LONG","SHORT"]:
            final_signal = f"{original_signal} ⚠ SUPPRESSED"

        news_statuses.append({
            "ticker":         ticker,
            "original_signal":original_signal,
            "final_signal":   final_signal,
            "news_status":    status,
            "news_reason":    reason,
            "news_events":    str([e["event"] for e in events]),
        })

    news_df_status = pd.DataFrame(news_statuses)

    # Merge back into signals
    signals = signals.merge(
        news_df_status[["ticker","final_signal","news_status","news_reason","news_events"]],
        on="ticker", how="left"
    )

    # Save updated signals
    conn = sqlite3.connect(DB_PATH)
    signals.to_sql("signals", conn, if_exists="replace", index=False)
    conn.close()
    print("  Signals table updated with news filter")

    return signals


def save_to_db(df, table_name):
    if df is None or df.empty:
        print(f"  No data for {table_name}")
        return
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    print(f"  Saved {len(df)} rows → '{table_name}'")


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════

def main():
    print("="*55)
    print("  Forex Signals — Phase 4: News Filter")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("="*55)

    # Fetch all news
    print("\nFetching economic calendar...")
    news_df = fetch_all_news()

    if news_df.empty:
        print("  No news events found — using scheduled fallback")
        news_df = pd.DataFrame(get_scheduled_events())

    print(f"\n  Found {len(news_df)} upcoming HIGH/MEDIUM impact events")

    # Save to database
    if not news_df.empty:
        save_to_db(news_df, "news_events")

    # Print upcoming events
    print("\n  Next 48 hours — High impact events:")
    print("  " + "─"*51)
    if not news_df.empty:
        now        = pd.Timestamp.now(tz="UTC")
        near_news  = news_df.copy()
        near_news["datetime"] = pd.to_datetime(
            near_news["datetime"], utc=True, errors="coerce"
        )
        near_news  = near_news.dropna(subset=["datetime"])
        near_news  = near_news[
            near_news["datetime"] <= now + pd.Timedelta(hours=48)
        ].sort_values("datetime")

        for _, event in near_news.iterrows():
            impact_icon = "🔴" if event["impact"] == "HIGH" else "🟡"
            hours_until = max(0, (event["datetime"] - now).total_seconds() / 3600)
            print(f"  {impact_icon} {event.get('datetime_ny','')[:16]} NY  "
                  f"[{event['currency']}]  {event['title'][:40]}  "
                  f"({hours_until:.1f}h)")

    # Apply filter to existing signals
    print("\nApplying news filter to signals...")
    updated_signals = apply_news_filter(news_df)

    if updated_signals is not None and not updated_signals.empty:
        print("\n" + "="*55)
        print("  SIGNAL STATUS AFTER NEWS FILTER:")
        print("="*55)
        for _, row in updated_signals.iterrows():
            pair   = row["ticker"].replace("=X","")
            signal = row.get("final_signal", row["signal"])
            status = row.get("news_status","CLEAR")

            if status == "SUPPRESSED":
                icon = "⛔"
            elif status == "WARNING":
                icon = "⚠️ "
            else:
                icon = "✅"

            print(f"\n  {icon} {pair}  →  {signal}")
            if row.get("news_reason"):
                print(f"     {row['news_reason'][:70]}")

    print(f"\n  News events saved → 'news_events' table")
    print("  Signals updated  → 'signals' table")
    print("\n  Next step: streamlit run app/main.py")


if __name__ == "__main__":
    main()
