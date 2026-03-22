# ── Database ──────────────────────────────
DB_PATH = "data/forex_signals.db"

# ── Currency pairs ────────────────────────
PAIRS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
}

# ── Timeframes to fetch ───────────────────
TIMEFRAMES = {
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1d",
}

# ── Currency strength — all 28 pairs ──────
STRENGTH_PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X",
    "AUDUSD=X","USDCAD=X","NZDUSD=X","EURGBP=X",
    "EURJPY=X","GBPJPY=X","AUDJPY=X","EURAUD=X",
    "EURCHF=X","GBPCHF=X","AUDNZD=X","AUDCAD=X",
    "AUDCHF=X","GBPAUD=X","GBPCAD=X","GBPNZD=X",
    "EURCAD=X","EURNZD=X","CADCHF=X","CADJPY=X",
    "CHFJPY=X","NZDCAD=X","NZDCHF=X","NZDJPY=X",
]

CURRENCIES = ["USD","EUR","GBP","JPY","AUD","CAD","CHF","NZD"]

# ── Signal thresholds ─────────────────────
CONFLUENCE_LONG_THRESHOLD  =  1.5   # score above this = Long signal
CONFLUENCE_SHORT_THRESHOLD = -1.5   # score below this = Short signal

# ── NY Midnight Open ──────────────────────
NY_TIMEZONE   = "America/New_York"
MIDNIGHT_HOUR = 0   # 00:00 NY time

# ── Liquidity settings ────────────────────
LIQUIDITY_PIP_ZONE = 10   # pips within a liquidity level to flag it

# ── FVG settings ──────────────────────────
FVG_LOOKBACK = 200   # how many candles back to scan for FVGs

# ── COT data ──────────────────────────────
COT_URL = "https://www.cftc.gov/dea/newcot/f_disagg.htm"
COT_CURRENCIES = {
    "EUR": "EURO FX",
    "GBP": "BRITISH POUND STERLING",
    "JPY": "JAPANESE YEN",
    "USD": "USD INDEX",
}

# ── News ──────────────────────────────────
FOREX_FACTORY_RSS = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
HIGH_IMPACT_SUPPRESS_HOURS = 4    # suppress signal within 4h of high impact news
NEWS_WARNING_HOURS         = 24   # warn but don't suppress within 24h