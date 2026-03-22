import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
import os
import sys

sys.path.insert(0, '.')
from data.config import DB_PATH, PAIRS, CURRENCIES

st.set_page_config(
    page_title="Forex Signal System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .signal-card{border-radius:12px;padding:18px;border:1.5px solid;margin-bottom:4px}
  .card-long{background:#f0fdf4;border-color:#86efac}
  .card-short{background:#fff1f2;border-color:#fca5a5}
  .card-notrade{background:#f8fafc;border-color:#e2e8f0}
  .card-suppressed{background:#fff7ed;border-color:#fed7aa}
  .sig-pair{font-size:22px;font-weight:700;margin:0 0 2px;color:#1e293b}
  .sig-signal{font-size:15px;font-weight:600;margin:0 0 10px}
  .sig-long{color:#16a34a}.sig-short{color:#dc2626}
  .sig-none{color:#64748b}.sig-warn{color:#d97706}
  .sig-row{display:flex;justify-content:space-between;margin-bottom:5px}
  .sig-key{font-size:12px;color:#64748b}
  .sig-val{font-size:12px;font-weight:500;color:#334155}
  .conf-bar-wrap{background:#e2e8f0;border-radius:6px;height:7px;margin:8px 0}
  .conf-bar-fill{height:100%;border-radius:6px}
  .bar-long{background:#22c55e}
  .bar-short{background:#ef4444}
  .bar-none{background:#94a3b8}
  .section-hdr{font-size:12px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.07em;margin:0 0 10px}
  .news-row{display:flex;gap:10px;align-items:center;padding:7px 0;border-bottom:1px solid #f1f5f9}
  .news-row:last-child{border-bottom:none}
  .badge-high{background:#fee2e2;color:#991b1b;padding:1px 7px;border-radius:8px;font-size:11px;font-weight:600;flex-shrink:0}
  .badge-med{background:#fef3c7;color:#92400e;padding:1px 7px;border-radius:8px;font-size:11px;font-weight:600;flex-shrink:0}
  .news-time{font-size:12px;color:#64748b;min-width:95px;flex-shrink:0}
  .news-curr{font-size:12px;font-weight:700;color:#3b82f6;min-width:30px}
  .news-title{font-size:12px;color:#334155}
  div[data-testid="metric-container"]{background:#f8fafc;border-radius:10px;border:1px solid #e2e8f0;padding:12px}
</style>
""", unsafe_allow_html=True)

NY_TZ = pytz.timezone("America/New_York")

# ── Data loaders ──────────────────────────────────────

@st.cache_data(ttl=60)
def load_table(name):
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql(f"SELECT * FROM {name}", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_candles(ticker, interval, limit=200):
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql(
            f"SELECT * FROM candles WHERE ticker=? AND interval=? "
            f"ORDER BY datetime DESC LIMIT {limit}",
            conn, params=(ticker, interval)
        )
        conn.close()
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

# ── Header ────────────────────────────────────────────

now_ny = datetime.now(NY_TZ)
c_title, c_btn = st.columns([8, 1])
with c_title:
    st.markdown(
        f"## 📈 Forex Signal System"
        f"<span style='font-size:13px;color:#94a3b8;margin-left:14px'>"
        f"NY: {now_ny.strftime('%a %d %b %Y  %H:%M')}</span>",
        unsafe_allow_html=True
    )
with c_btn:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

st.divider()

# ── Load data ─────────────────────────────────────────

signals_df  = load_table("signals")
strength_df = load_table("currency_strength")
fvg_df      = load_table("fair_value_gaps")
liq_df      = load_table("liquidity_levels")
cot_df      = load_table("cot_data")
news_df     = load_table("news_events")

# ═══════════════════════════════════════════════════════
# SIGNAL CARDS
# ═══════════════════════════════════════════════════════

st.markdown('<p class="section-hdr">Live Signals</p>', unsafe_allow_html=True)

if signals_df.empty:
    st.warning("No signals — run `python models/signal_engine.py` then `python data/fetch_news.py`")
else:
    card_cols = st.columns(len(PAIRS))

    def dot(s):
        s = int(s) if s else 0
        return ("●●" if s>=2 else "●○" if s==1 else "○○" if s==0 else "○●" if s==-1 else "●●")

    def dot_col(s):
        s = int(s) if s else 0
        return "#16a34a" if s>0 else ("#dc2626" if s<0 else "#94a3b8")

    for col, (pair_name, ticker) in zip(card_cols, PAIRS.items()):
        r = signals_df[signals_df["ticker"]==ticker]
        if r.empty:
            with col: st.info(f"{pair_name} — no signal")
            continue
        r = r.iloc[0]

        sig    = r.get("final_signal", r.get("signal","NO TRADE"))
        conf   = float(r.get("confluence_score",0))
        pct    = int(r.get("confidence_pct",0))
        target = float(r.get("pip_target",0))
        ns     = r.get("news_status","CLEAR")
        cur_p  = r.get("current_price")
        mid_p  = r.get("midnight_open_price")

        if "LONG" in str(sig) and "SUPPRESS" not in str(sig):
            cc,sc,arrow,bc = "card-long","sig-long","▲ LONG","bar-long"
        elif "SHORT" in str(sig) and "SUPPRESS" not in str(sig):
            cc,sc,arrow,bc = "card-short","sig-short","▼ SHORT","bar-short"
        elif "SUPPRESS" in str(sig):
            cc,sc,arrow,bc = "card-suppressed","sig-warn","⛔ SUPPRESSED","bar-none"
        else:
            cc,sc,arrow,bc = "card-notrade","sig-none","─ NO TRADE","bar-none"

        n_icon = {"CLEAR":"✅","WARNING":"⚠️","SUPPRESSED":"⛔","INFO":"ℹ️"}.get(ns,"✅")

        mbo = r.get("midnight_open_score",0)
        st_ = r.get("strength_score",0)
        fv  = r.get("fvg_score",0)
        lq  = r.get("liquidity_score",0)
        ct  = r.get("cot_score",0)

        cp  = f"{float(cur_p):.5f}" if cur_p else "—"
        mp  = f"{float(mid_p):.5f}" if mid_p else "—"

        with col:
            st.markdown(f"""
<div class="signal-card {cc}">
  <p class="sig-pair">{pair_name}</p>
  <p class="sig-signal {sc}">{arrow}</p>
  <div class="conf-bar-wrap">
    <div class="conf-bar-fill {bc}" style="width:{pct}%"></div>
  </div>
  <div class="sig-row"><span class="sig-key">Confidence</span><span class="sig-val">{pct}%</span></div>
  <div class="sig-row"><span class="sig-key">Confluence</span><span class="sig-val">{conf:+.3f}</span></div>
  <div class="sig-row"><span class="sig-key">Pip target</span><span class="sig-val">~{target:.0f} pips</span></div>
  <div class="sig-row"><span class="sig-key">Price</span><span class="sig-val">{cp}</span></div>
  <div class="sig-row"><span class="sig-key">NY Midnight</span><span class="sig-val">{mp}</span></div>
  <hr style="border:none;border-top:1px solid #e2e8f0;margin:8px 0">
  <div class="sig-row"><span class="sig-key">NY Open</span><span style="font-size:12px;font-weight:600;color:{dot_col(mbo)}">{dot(mbo)} {int(mbo):+d}</span></div>
  <div class="sig-row"><span class="sig-key">Strength</span><span style="font-size:12px;font-weight:600;color:{dot_col(st_)}">{dot(st_)} {int(st_):+d}</span></div>
  <div class="sig-row"><span class="sig-key">FVG</span><span style="font-size:12px;font-weight:600;color:{dot_col(fv)}">{dot(fv)} {int(fv):+d}</span></div>
  <div class="sig-row"><span class="sig-key">Liquidity</span><span style="font-size:12px;font-weight:600;color:{dot_col(lq)}">{dot(lq)} {int(lq):+d}</span></div>
  <div class="sig-row"><span class="sig-key">COT</span><span style="font-size:12px;font-weight:600;color:{dot_col(ct)}">{dot(ct)} {int(ct):+d}</span></div>
  <hr style="border:none;border-top:1px solid #e2e8f0;margin:8px 0">
  <div class="sig-row"><span class="sig-key">News</span><span class="sig-val">{n_icon} {ns}</span></div>
</div>""", unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💱 Currency Strength",
    "📊 FVG + Liquidity Chart",
    "🏦 COT Report",
    "📅 Economic Calendar",
    "📋 Signal Detail",
])

# ── TAB 1: CURRENCY STRENGTH ─────────────────────────

with tab1:
    st.markdown('<p class="section-hdr">Currency Strength Meter</p>', unsafe_allow_html=True)
    if strength_df.empty:
        st.info("No strength data. Run signal_engine.py")
    else:
        ss = strength_df.sort_values("strength_score", ascending=False).reset_index(drop=True)
        colors = ["#16a34a" if s>=65 else "#84cc16" if s>=50 else "#f59e0b" if s>=35 else "#ef4444"
                  for s in ss["strength_score"]]
        fig = go.Figure(go.Bar(
            x=ss["currency"], y=ss["strength_score"].round(1),
            marker_color=colors,
            text=ss["strength_score"].round(1),
            textposition="outside",
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="#94a3b8", line_width=1,
                      annotation_text="Neutral 50")
        fig.update_layout(height=300, margin=dict(l=0,r=0,t=20,b=0),
            yaxis=dict(range=[0,115]), plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Pair opportunity table
        sc_map = ss.set_index("currency")["strength_score"].to_dict()
        rows   = []
        for pn, tk in PAIRS.items():
            clean = tk.replace("=X","")
            if len(clean)!=6: continue
            b, q  = clean[:3], clean[3:]
            bs, qs = sc_map.get(b,50), sc_map.get(q,50)
            d = bs - qs
            rows.append({"Pair":pn,"Base":f"{b} ({bs:.0f})","Quote":f"{q} ({qs:.0f})",
                         "Diff":f"{d:+.0f}","Direction":"▲ LONG" if d>10 else ("▼ SHORT" if d<-10 else "─ NEUTRAL")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── TAB 2: FVG CHART ─────────────────────────────────

with tab2:
    st.markdown('<p class="section-hdr">Fair Value Gaps + Liquidity Levels</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: sel_pair = st.selectbox("Pair", list(PAIRS.keys()), key="fp")
    with c2: sel_tf   = st.selectbox("Timeframe", ["1h","4h","1d"], key="ft")
    ticker  = PAIRS[sel_pair]
    candles = load_candles(ticker, sel_tf, 150)

    if candles.empty:
        st.info("No candle data. Run fetch_prices.py")
    else:
        candles["datetime"] = pd.to_datetime(candles["datetime"])
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=candles["datetime"], open=candles["open"], high=candles["high"],
            low=candles["low"], close=candles["close"], name=sel_pair,
            increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
            increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef4444",
        ))

        # FVG zones
        if not fvg_df.empty:
            pfvg = fvg_df[(fvg_df["ticker"]==ticker)&(fvg_df["interval"]==sel_tf)&(~fvg_df["mitigated"].astype(bool))].tail(15)
            for _, f in pfvg.iterrows():
                color = "rgba(34,197,94,0.15)" if f["type"]=="BULLISH" else "rgba(239,68,68,0.15)"
                lc    = "#22c55e" if f["type"]=="BULLISH" else "#ef4444"
                fig.add_shape(type="rect",
                    x0=pd.to_datetime(f["datetime"]), x1=candles["datetime"].iloc[-1],
                    y0=f["fvg_bottom"], y1=f["fvg_top"],
                    fillcolor=color, line=dict(color=lc, width=0.5), layer="below")

        # Liquidity levels
        if not liq_df.empty:
            lc_map = {"PDH":"#f59e0b","PDL":"#f59e0b","PWH":"#8b5cf6","PWL":"#8b5cf6","EQH":"#3b82f6","EQL":"#3b82f6"}
            for _, lv in liq_df[liq_df["ticker"]==ticker].iterrows():
                fig.add_hline(y=float(lv["price"]),
                    line=dict(color=lc_map.get(lv["type"],"#94a3b8"), width=1, dash="dash"),
                    annotation_text=f"{lv['type']} {float(lv['price']):.5f}",
                    annotation_position="left", annotation_font=dict(size=10))

        # NY Midnight Open
        if not signals_df.empty:
            sr = signals_df[signals_df["ticker"]==ticker]
            if not sr.empty and sr.iloc[0].get("midnight_open_price"):
                mp = float(sr.iloc[0]["midnight_open_price"])
                fig.add_hline(y=mp,
                    line=dict(color="#0ea5e9", width=1.5, dash="dot"),
                    annotation_text=f"NY Open {mp:.5f}",
                    annotation_position="right", annotation_font=dict(size=10, color="#0ea5e9"))

        fig.update_layout(height=480, margin=dict(l=0,r=80,t=10,b=0),
            xaxis_rangeslider_visible=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)"), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
<div style="display:flex;gap:18px;font-size:11px;color:#64748b">
  <span><span style="background:rgba(34,197,94,0.3);padding:1px 6px;border-radius:3px">Green</span> Bullish FVG</span>
  <span><span style="background:rgba(239,68,68,0.3);padding:1px 6px;border-radius:3px">Red</span> Bearish FVG</span>
  <span><span style="color:#f59e0b">━━</span> PDH/PDL</span>
  <span><span style="color:#8b5cf6">━━</span> PWH/PWL</span>
  <span><span style="color:#3b82f6">━━</span> Equal H/L</span>
  <span><span style="color:#0ea5e9">┅┅</span> NY Midnight Open</span>
</div>""", unsafe_allow_html=True)

# ── TAB 3: COT REPORT ────────────────────────────────

with tab3:
    st.markdown('<p class="section-hdr">COT Institutional Positioning (CFTC)</p>', unsafe_allow_html=True)
    st.caption("Commercial traders = banks and institutions (smart money). Updated every Friday.")

    if cot_df.empty:
        st.info("No COT data. Run fetch_cot.py")
    else:
        cot_df["date"] = pd.to_datetime(cot_df["date"])
        avail = [c for c in ["EUR","GBP","JPY","USD","AUD","CAD","CHF","NZD"] if c in cot_df["currency"].values]
        sel_curr = st.selectbox("Currency", avail, key="cot_c")
        cd = cot_df[cot_df["currency"]==sel_curr].sort_values("date")

        if not cd.empty:
            latest = cd.iloc[-1]
            net = int(latest.get("comm_net", 0)) if "comm_net" in latest.index else 0
            pct    = float(latest.get("comm_net_pct",50))
            dated  = str(latest["date"])[:10]

            c1,c2,c3 = st.columns(3)
            c1.metric("Commercial Net", f"{net:,}")
            c2.metric("52-Week Percentile", f"{pct:.0f}%")
            c3.metric("Signal",
                "Strong Bullish" if pct>=75 else "Mild Bullish" if pct>=55 else
                "Neutral" if pct>=45 else "Mild Bearish" if pct>=25 else "Strong Bearish",
                delta=f"As of {dated}")

            st.divider()
            fig2 = go.Figure()
            if "comm_net" not in cd.columns:
             st.warning("COT net position data not available. Re-run `python data/fetch_cot.py`")
    st.stop()
q75  = cd["comm_net"].quantile(0.75)
q25  = cd["comm_net"].quantile(0.25)
ymx  = cd["comm_net"].max()*1.1; ymn = cd["comm_net"].min()*1.1
fig2.add_hrect(y0=q75,y1=float(ymx),fillcolor="rgba(34,197,94,0.08)",line_width=0,
                annotation_text="Bullish extreme",annotation_position="top right",
                annotation_font=dict(size=10,color="#16a34a"))
fig2.add_hrect(y0=float(ymn),y1=q25,fillcolor="rgba(239,68,68,0.08)",line_width=0,
                annotation_text="Bearish extreme",annotation_position="bottom right",
                annotation_font=dict(size=10,color="#dc2626"))
if "noncomm_net" in cd.columns:
                fig2.add_trace(go.Scatter(x=cd["date"],y=cd["noncomm_net"],name="Non-commercial (retail)",
                    line=dict(color="#94a3b8",width=1,dash="dot"),opacity=0.5))
fig2.add_trace(go.Scatter(x=cd["date"],y=cd["comm_net"],name="Commercial (smart money)",
                line=dict(color="#3b82f6",width=2.5),fill="tozeroy",fillcolor="rgba(59,130,246,0.06)"))
fig2.add_hline(y=0,line_dash="dash",line_color="#94a3b8",line_width=1)
fig2.update_layout(height=340,margin=dict(l=0,r=0,t=10,b=0),
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
                legend=dict(orientation="h",yanchor="bottom",y=1.02))
st.plotly_chart(fig2, use_container_width=True)
fig3 = go.Figure(go.Indicator(mode="gauge+number",value=pct,
                gauge={"axis":{"range":[0,100]},"bar":{"color":"#3b82f6"},
                    "steps":[{"range":[0,25],"color":"rgba(239,68,68,0.2)"},
                              {"range":[25,75],"color":"rgba(148,163,184,0.1)"},
                              {"range":[75,100],"color":"rgba(34,197,94,0.2)"}]},
                title={"text":f"{sel_curr} Comm. Percentile"},number={"suffix":"%"}))
fig3.update_layout(height=240,margin=dict(l=20,r=20,t=40,b=20),paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig3, use_container_width=True)

# ── TAB 4: ECONOMIC CALENDAR ─────────────────────────

with tab4:
    st.markdown('<p class="section-hdr">Economic Calendar — Next 7 Days (NY Time)</p>', unsafe_allow_html=True)

    if news_df.empty:
        st.info("No news data. Run fetch_news.py")
    else:
        nd = news_df.copy()
        nd["datetime"] = pd.to_datetime(nd["datetime"], errors="coerce", utc=True)
        nd = nd.dropna(subset=["datetime"]).sort_values("datetime")

        impact_sel = st.multiselect("Impact", ["HIGH","MEDIUM"], default=["HIGH","MEDIUM"], key="nf")
        nd = nd[nd["impact"].isin(impact_sel)]

        ny = pytz.timezone("America/New_York")
        nd["dt_ny"]  = nd["datetime"].dt.tz_convert(ny)
        nd["day_ny"] = nd["dt_ny"].dt.strftime("%A %d %b")
        now_u = pd.Timestamp.now(tz="UTC")

        for day, devs in nd.groupby("day_ny", sort=False):
            is_today = devs["datetime"].min().date() == now_u.date()
            st.markdown(f"**📅 {day}**" + (" &nbsp;`Today`" if is_today else ""), unsafe_allow_html=True)
            html = ""
            for _, ev in devs.iterrows():
                past  = ev["datetime"] < now_u
                op    = "opacity:0.4;" if past else ""
                badge = f'<span class="badge-high">HIGH</span>' if ev["impact"]=="HIGH" else f'<span class="badge-med">MED</span>'
                html += f"""<div class="news-row" style="{op}">
  <span class="news-time">{ev['dt_ny'].strftime('%H:%M')} NY</span>
  <span class="news-curr">{ev.get('currency','—')}</span>
  {badge}
  <span class="news-title">{str(ev['title'])[:65]}</span>
</div>"""
            st.markdown(html, unsafe_allow_html=True)
            st.markdown("")

# ── TAB 5: SIGNAL DETAIL ─────────────────────────────

with tab5:
    st.markdown('<p class="section-hdr">Full Signal Breakdown</p>', unsafe_allow_html=True)

    if signals_df.empty:
        st.info("No signals yet.")
    else:
        sel_d  = st.selectbox("Pair", list(PAIRS.keys()), key="dp")
        ticker = PAIRS[sel_d]
        r      = signals_df[signals_df["ticker"]==ticker]

        if r.empty:
            st.info("No signal for this pair.")
        else:
            r = r.iloc[0]
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Signal",     r.get("final_signal",r.get("signal","—")))
            c2.metric("Confluence", f"{float(r.get('confluence_score',0)):+.3f}")
            c3.metric("Confidence", f"{int(r.get('confidence_pct',0))}%")
            c4.metric("Pip Target", f"~{float(r.get('pip_target',0)):.0f} pips")
            st.divider()

            details = [
                ("NY Midnight Open","midnight_open_score","midnight_open_detail","30%"),
                ("Currency Strength","strength_score","strength_detail","25%"),
                ("FVG Proximity","fvg_score","fvg_detail","20%"),
                ("Liquidity","liquidity_score","liquidity_detail","15%"),
                ("COT Positioning","cot_score","cot_detail","10%"),
            ]
            for label, sk, dk, wt in details:
                score  = int(r.get(sk,0) or 0)
                detail = str(r.get(dk,"—"))[:120]
                color  = "#16a34a" if score>0 else ("#dc2626" if score<0 else "#94a3b8")
                bw     = int((score+2)/4*100)
                st.markdown(f"""
<div style="background:#f8fafc;border-radius:8px;padding:12px;margin-bottom:8px;border:1px solid #e2e8f0">
  <div style="display:flex;justify-content:space-between;margin-bottom:5px">
    <span style="font-size:13px;font-weight:600;color:#334155">{label} <span style="color:#94a3b8;font-weight:400">({wt})</span></span>
    <span style="font-size:15px;font-weight:700;color:{color}">{score:+d}</span>
  </div>
  <div style="background:#e2e8f0;border-radius:4px;height:5px;margin-bottom:7px">
    <div style="background:{color};width:{bw}%;height:100%;border-radius:4px"></div>
  </div>
  <span style="font-size:12px;color:#64748b">{detail}</span>
</div>""", unsafe_allow_html=True)

            ns = r.get("news_status","CLEAR")
            nr = r.get("news_reason","")
            ni = {"CLEAR":"✅","WARNING":"⚠️","SUPPRESSED":"⛔"}.get(ns,"✅")
            st.divider()
            st.markdown(f"**News:** {ni} {ns}")
            if nr: st.caption(nr)
            st.caption(f"Computed: {r.get('computed_at','—')}")
            st.divider()
            st.markdown("**Refresh commands:**")
            st.code("python data/fetch_prices.py\npython models/signal_engine.py\npython data/fetch_news.py", language="bash")
