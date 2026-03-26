"""
dashboard/app.py — GeoShock v2 Real-Time Dashboard
─────────────────────────────────────────────────────────────────────────────
Run:  PYTHONPATH=. PYTHONWARNINGS=ignore streamlit run dashboard/app.py 2>/dev/null

Sections
────────
  00  LIVE ALERT BAR    Layer 0 real-time event signal (GDELT + CAMEO + AIS)
  01  GPR MONITOR       GPR time series, regime distribution, rolling metrics
  02  GIPI PANEL        Geopolitical Inflation Pressure Index + channel breakdown
  03  GROWTH-AT-RISK    Fan chart, predictive distribution, tail probability
  04  LOCAL PROJECTIONS IRF by outcome and regime
  05  VAR / FEVD        Forecast error variance decomposition, Granger table
  06  INFLATION CHANNELS Breakeven, import prices, energy, food, Arab Light spread
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

# ── Suppress all warnings before any library import ──────────────────────────
import warnings
import logging
import os

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

for _noisy in [
    "streamlit", "watchdog", "urllib3", "matplotlib",
    "PIL", "pyarrow", "numba", "scipy", "sklearn",
    "statsmodels", "pandas", "plotly", "yfinance",
    "peewee", "fsevents",
]:
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)
import streamlit as st

# Re-apply after streamlit's own import (st can reset warning filters)
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoShock — Geopolitical Risk Monitor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Mono', monospace !important; }
  .stApp           { background-color: #07091a; }
  .stSidebar       { background-color: #0c1122; }
  .section-hdr {
    font-size: 10px; letter-spacing: 3px; color: #475569;
    text-transform: uppercase; padding: 6px 0 3px;
    border-bottom: 1px solid #1e2d4a; margin: 12px 0 8px;
  }
  .alert-calm    { background:#0d2218;border-left:4px solid #10b981;
                   border-radius:6px;padding:10px 16px;margin:6px 0; }
  .alert-elevated{ background:#221a08;border-left:4px solid #e8a020;
                   border-radius:6px;padding:10px 16px;margin:6px 0; }
  .alert-crisis  { background:#220c0c;border-left:4px solid #ef4444;
                   border-radius:6px;padding:10px 16px;margin:6px 0; }
  .ais-box       { background:#0f1628;border:1px solid #1e2d4a;
                   border-radius:6px;padding:8px 14px;margin:4px 0;font-size:11px; }
  div[data-testid="stMetricValue"] { color:#e8a020!important;font-size:26px!important; }
  div[data-testid="stMetricLabel"] { color:#64748b!important;font-size:10px!important; }
</style>
""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG, PANEL, BORDER = "#07091a", "#0c1122", "#1e2d4a"
AMBER, BLUE, GREEN, RED  = "#e8a020", "#3b82f6", "#10b981", "#ef4444"
PURPLE, CYAN, MUTED, TEXT = "#a855f7", "#06b6d4", "#64748b", "#e2e8f0"
ORANGE = "#f97316"

REGIME_COLOR = {"calm": GREEN, "elevated": AMBER, "crisis": RED, "CALM": GREEN,
                "ELEVATED": AMBER, "CRISIS": RED}

BASE_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=PANEL,
    font=dict(family="IBM Plex Mono,monospace", color=TEXT, size=11),
    margin=dict(l=48, r=18, t=38, b=36),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
)

Q_COLORS = {0.05: RED, 0.10: ORANGE, 0.25: AMBER,
            0.50: BLUE, 0.75: GREEN, 0.90: CYAN, 0.95: PURPLE}


# ═══════════════════════════════════════════════════════════════════════════════
#  CACHED LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Loading macro data …")
def load_data(refresh: bool = False) -> pd.DataFrame:
    from data.pipeline import DataPipeline
    dp = DataPipeline()
    return dp.build(use_cache=not refresh)


@st.cache_data(ttl=300, show_spinner="Running Layer 0 detection …")
def load_event_signal(
    lookback_hours: int = 48,
    use_llm: bool = True,
    _key: str = "",
) -> dict:
    """Run Layer 0 and return signal dict.
    Falls back to outputs/event_signal.json if live run fails."""
    try:
        from data.event_detector import EventDetector
        ed  = EventDetector(anthropic_key=_key or None)
        sig = ed.detect(lookback_hours=lookback_hours,
                        use_llm=use_llm, use_ais=True)
        result = sig.to_dict()
        # If nowcast came back as 0, try to preserve last known good value
        _out = Path(__file__).parent.parent / "outputs" / "event_signal.json"
        if result.get("gpr_nowcast", 0.0) == 0.0 and _out.exists():
            try:
                import json as _j
                _prev = _j.loads(_out.read_text())
                if _prev.get("gpr_nowcast", 0.0) != 0.0:
                    result["gpr_nowcast"] = _prev["gpr_nowcast"]
            except Exception:
                pass
        # persist for the status bar reader
        _out.parent.mkdir(exist_ok=True)
        import json as _j
        _out.write_text(_j.dumps(result, default=str))
        return result
    except Exception as e:
        # Try to load last saved signal before returning zeros
        _out = Path(__file__).parent.parent / "outputs" / "event_signal.json"
        if _out.exists():
            try:
                import json as _j
                cached = _j.loads(_out.read_text())
                cached["_from_cache"] = True
                cached["_cache_error"] = str(e)
                return cached
            except Exception:
                pass
        return {"error": str(e), "regime": "UNKNOWN",
                "severity_score": 0.0, "gpr_nowcast": 0.0,
                "cameo_codes": [], "ais_anomaly": False,
                "top_headlines": [], "article_count": 0,
                "location_hit_rate": 0.0, "llm_used": False,
                "llm_summary": None, "ais_signal": None}


@st.cache_data(ttl=7200, show_spinner="Estimating Growth-at-Risk …")
def compute_gar(df_json: str, outcome: str, horizon: int,
                use_gipi: bool = True) -> dict:
    df = pd.read_json(df_json)
    df.index = pd.to_datetime(df.index)
    from models.quantile_risk import GrowthAtRisk
    gipi_ok = use_gipi and ("gipi" in df.columns and df["gipi"].notna().sum() >= 36)
    gar = GrowthAtRisk(df=df, outcome=outcome, gpr_col="gpr_z",
                       gipi_col="gipi" if gipi_ok else None,
                       use_interaction=gipi_ok)
    gar.fit(horizons=[horizon], verbose=False)
    r = gar.results_[horizon]

    def _s(key: float) -> list:
        s = r.fitted.get(key, pd.Series(dtype=float))
        return s.values.tolist()

    idx = r.fitted.get(0.50, pd.Series(dtype=float)).index
    return {
        "gar_5":      r.gar_5,
        "gar_25":     r.gar_25,
        "median":     r.median_forecast,
        "prob_neg":   r.prob_neg_growth,
        "prob_rec":   r.prob_recession,
        "skewness":   r.conditional_skewness,
        "nowcast":    {str(k): v for k, v in r.nowcast.items()},
        "dates":      idx.astype(str).tolist(),
        "q05":        _s(0.05), "q10": _s(0.10),
        "q25":        _s(0.25), "q50": _s(0.50),
        "q75":        _s(0.75), "q90": _s(0.90),
        "q95":        _s(0.95),
    }


@st.cache_data(ttl=7200, show_spinner="Estimating Local Projections …")
def compute_lp(df_json: str, outcome: str, shock: str,
               horizon: int, n_bs: int) -> dict:
    df = pd.read_json(df_json)
    df.index = pd.to_datetime(df.index)
    from models.local_projections import LocalProjections
    lp = LocalProjections(df, shock=shock, outcome=outcome,
                          horizon=horizon, bootstrap_reps=n_bs)
    r  = lp.fit(regime="full", verbose=False)
    return {
        "betas":    r.betas.tolist(),
        "ci68_lo":  r.ci_68[:, 0].tolist(),
        "ci68_hi":  r.ci_68[:, 1].tolist(),
        "ci90_lo":  r.ci_90[:, 0].tolist(),
        "ci90_hi":  r.ci_90[:, 1].tolist(),
        "horizons": r.horizons.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def fig_gpr_series(df: pd.DataFrame, n: int = 120) -> go.Figure:
    sub = df.dropna(subset=["gpr_level"]).iloc[-n:]
    fig = go.Figure()

    # Regime shading
    if "regime" in sub.columns:
        for rg, col in [("calm", GREEN), ("elevated", AMBER), ("crisis", RED)]:
            mask = sub["regime"] == rg
            if mask.any():
                chg = mask.astype(int).diff().fillna(0)
                starts = sub.index[chg == 1].tolist()
                ends   = sub.index[chg == -1].tolist()
                if mask.iloc[0]:  starts = [sub.index[0]] + starts
                if mask.iloc[-1]: ends   = ends + [sub.index[-1]]
                for s, e in zip(starts, ends):
                    fig.add_vrect(x0=s, x1=e, fillcolor=col,
                                  opacity=0.07, layer="below", line_width=0)

    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["gpr_level"],
        line=dict(color=AMBER, width=2), name="GPR",
        hovertemplate="%{x|%b %Y}<br>GPR=%{y:.0f}<extra></extra>",
    ))
    if "gpr_z" in sub.columns:
        fig.add_trace(go.Scatter(
            x=sub.index, y=sub["gpr_z"] * sub["gpr_level"].mean() / sub["gpr_z"].std(),
            line=dict(color=BLUE, width=1, dash="dot"), name="GPR z (scaled)",
            opacity=0.5,
        ))

    fig.update_layout(**BASE_LAYOUT, height=260,
                      title=dict(text="Geopolitical Risk Index",
                                 font=dict(size=11, color=MUTED), x=0.01),
                      legend=dict(orientation="h", y=1.1, font=dict(size=9),
                                  bgcolor="rgba(0,0,0,0)"))
    return fig


def fig_gipi(df: pd.DataFrame, n: int = 120) -> go.Figure:
    """GIPI time series with component overlay."""
    if "gipi" not in df.columns:
        return go.Figure()

    sub = df.dropna(subset=["gipi"]).iloc[-n:]
    fig = go.Figure()

    # Component lines (light)
    comp_map = {
        "gscpi":            ("GSCPI",       CYAN,   0.4),
        "import_price_yoy": ("Import P YoY",BLUE,   0.4),
        "global_energy_yoy":("Energy YoY",  ORANGE, 0.4),
        "global_food_yoy":  ("Food YoY",    GREEN,  0.4),
        "d_breakeven_5y":   ("ΔBreakeven",  PURPLE, 0.4),
    }
    for col, (label, color, alpha) in comp_map.items():
        if col in df.columns:
            s = df[col].reindex(sub.index)
            s_z = (s - s.mean()) / (s.std() + 1e-9)
            fig.add_trace(go.Scatter(
                x=sub.index, y=s_z,
                line=dict(color=color, width=1), opacity=alpha,
                name=label, visible="legendonly",
            ))

    # GIPI main line
    fig.add_trace(go.Scatter(
        x=sub.index, y=sub["gipi"],
        line=dict(color=AMBER, width=2.5), name="GIPI (PC1)",
        hovertemplate="%{x|%b %Y}<br>GIPI=%{y:.2f}<extra></extra>",
    ))
    # Zero line + thresholds
    fig.add_hline(y=0,   line=dict(color=MUTED, width=0.8, dash="dot"))
    fig.add_hline(y=1.5, line=dict(color=AMBER,  width=0.8, dash="dash"),
                  annotation_text="elevated", annotation_font_size=8)
    fig.add_hline(y=2.5, line=dict(color=RED,    width=0.8, dash="dash"),
                  annotation_text="crisis", annotation_font_size=8)

    fig.update_layout(**BASE_LAYOUT, height=260,
                      title=dict(
                          text="GIPI — Geopolitical Inflation Pressure Index",
                          font=dict(size=11, color=MUTED), x=0.01),
                      legend=dict(orientation="h", y=1.12,
                                  font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
    return fig


def fig_gar_fan(gar_data: dict, horizon: int) -> go.Figure:
    dates = pd.to_datetime(gar_data["dates"])
    fig   = go.Figure()

    # Confidence bands
    band_pairs = [
        ("q05", "q95", 0.12, "5–95th pctile"),
        ("q10", "q90", 0.18, "10–90th pctile"),
        ("q25", "q75", 0.28, "25–75th pctile"),
    ]
    for lo, hi, alpha, _ in band_pairs:
        lo_v = gar_data.get(lo, [])
        hi_v = gar_data.get(hi, [])
        if lo_v and hi_v:
            fig.add_trace(go.Scatter(
                x=list(dates) + list(dates[::-1]),
                y=lo_v + hi_v[::-1],
                fill="toself", fillcolor=f"rgba(232,160,32,{alpha})",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))

    q50 = gar_data.get("q50", [])
    if q50:
        fig.add_trace(go.Scatter(
            x=dates, y=q50, line=dict(color=AMBER, width=2),
            name="Median", hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
        ))
    q05 = gar_data.get("q05", [])
    if q05:
        fig.add_trace(go.Scatter(
            x=dates, y=q05, line=dict(color=RED, width=1.5, dash="dash"),
            name="GaR₅", hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color=MUTED, width=0.8, dash="dot"))
    fig.add_hline(y=-2, line=dict(color=RED, width=0.6, dash="dot"),
                  annotation_text="recession", annotation_font_size=8)

    fig.update_layout(**BASE_LAYOUT, height=300,
                      title=dict(
                          text=f"Growth-at-Risk Fan Chart  (h = {horizon} months)",
                          font=dict(size=11, color=MUTED), x=0.01),
                      yaxis_title="IP YoY Growth (%)",
                      legend=dict(orientation="h", y=1.12,
                                  font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
    return fig


def fig_gar_dist(gar_data: dict) -> go.Figure:
    """Kernel density of forecast distribution."""
    from scipy.stats import gaussian_kde
    nowcast = gar_data.get("nowcast", {})
    qs = sorted([(float(k), float(v)) for k, v in nowcast.items()])
    if len(qs) < 3:
        return go.Figure()
    vals = [v for _, v in qs]
    lo, hi = min(vals) - 3, max(vals) + 3
    x = np.linspace(lo, hi, 300)
    try:
        kde_val = gaussian_kde(vals)(x)
    except Exception:
        return go.Figure()

    fig = go.Figure()
    # Recession zone
    fig.add_vrect(x0=lo, x1=-2, fillcolor=RED, opacity=0.07,
                  layer="below", line_width=0)
    # KDE
    fig.add_trace(go.Scatter(
        x=x, y=kde_val, fill="tozeroy",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color=BLUE, width=2), name="Forecast density",
    ))
    # Quantile markers
    for q, v in qs:
        color = Q_COLORS.get(q, MUTED)
        fig.add_vline(x=v, line=dict(color=color, width=1, dash="dot"),
                      annotation_text=f"τ={q:.2f}", annotation_font_size=7)

    fig.update_layout(**BASE_LAYOUT, height=250,
                      title=dict(text="Predictive Distribution (nowcast)",
                                 font=dict(size=11, color=MUTED), x=0.01),
                      xaxis_title="IP YoY Growth (%)", yaxis_title="Density",
                      showlegend=False)
    return fig


def fig_lp_irf(lp_data: dict, outcome: str) -> go.Figure:
    h  = lp_data["horizons"]
    b  = lp_data["betas"]
    fig = go.Figure()

    # 90% CI
    fig.add_trace(go.Scatter(
        x=h + h[::-1],
        y=lp_data["ci90_hi"] + lp_data["ci90_lo"][::-1],
        fill="toself", fillcolor="rgba(232,160,32,0.10)",
        line=dict(width=0), showlegend=False,
    ))
    # 68% CI
    fig.add_trace(go.Scatter(
        x=h + h[::-1],
        y=lp_data["ci68_hi"] + lp_data["ci68_lo"][::-1],
        fill="toself", fillcolor="rgba(232,160,32,0.22)",
        line=dict(width=0), showlegend=False,
    ))
    # Point estimates
    fig.add_trace(go.Scatter(
        x=h, y=b, line=dict(color=AMBER, width=2),
        mode="lines+markers", marker=dict(size=4),
        name="IRF", hovertemplate="h=%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=MUTED, width=0.8, dash="dot"))

    fig.update_layout(**BASE_LAYOUT, height=240,
                      title=dict(text=f"LP-IRF: GPR Shock → {outcome}",
                                 font=dict(size=10, color=MUTED), x=0.01))
    return fig


def fig_inflation_channels(df: pd.DataFrame, n: int = 120) -> go.Figure:
    """Multi-panel: key inflation channel series."""
    channels = [
        ("cpi_inflation",      "CPI YoY (%)",           AMBER),
        ("breakeven_5y",       "5Y Breakeven (%)",       BLUE),
        ("import_price_yoy",   "Import Price YoY (%)",   CYAN),
        ("global_energy_yoy",  "Global Energy YoY (%)",  ORANGE),
        ("global_food_yoy",    "Global Food YoY (%)",    GREEN),
        ("arab_wti_spread",    "Arab Light–WTI Spread ($)", RED),
    ]
    available = [(col, lbl, clr) for col, lbl, clr in channels if col in df.columns]
    if not available:
        return go.Figure()

    n_plots = len(available)
    rows    = (n_plots + 1) // 2
    fig     = make_subplots(rows=rows, cols=2,
                            subplot_titles=[lbl for _, lbl, _ in available],
                            vertical_spacing=0.08)
    for i, (col, lbl, clr) in enumerate(available):
        r, c = divmod(i, 2)
        sub = df[col].dropna().iloc[-n:]
        fig.add_trace(
            go.Scatter(x=sub.index, y=sub.values, line=dict(color=clr, width=1.5),
                       name=lbl, showlegend=False,
                       hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>"),
            row=r+1, col=c+1,
        )

    fig.update_layout(
        **{k: v for k, v in BASE_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        height=max(300, rows * 200),
        title=dict(text="Inflation Transmission Channels",
                   font=dict(size=11, color=MUTED), x=0.01),
        showlegend=False,
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("""
        <div style='padding:4px 0 12px'>
          <div style='font-size:9px;letter-spacing:3px;color:#475569'>GeoShock</div>
          <div style='font-size:16px;font-weight:700;color:#f1f5f9'>🛰 Risk Monitor</div>
        </div>""", unsafe_allow_html=True)

        # ── Layer 0 ──────────────────────────────────────────────────────────
        st.markdown("<div class='section-hdr'>LAYER 0 — EVENT DETECTION</div>",
                    unsafe_allow_html=True)
        run_l0      = st.button("🔄 Refresh Event Signal", use_container_width=True)
        lookback    = st.slider("GDELT lookback (hours)", 12, 168, 48, step=12)
        use_llm     = st.checkbox("Use LLM CAMEO coding", value=True,
                                  help="Requires ANTHROPIC_API_KEY in .env")
        show_raw_hl = st.checkbox("Show raw headlines", value=False)

        # ── Data ─────────────────────────────────────────────────────────────
        st.markdown("<div class='section-hdr'>DATA</div>", unsafe_allow_html=True)
        refresh     = st.button("🔄 Refresh Macro Data", use_container_width=True)
        n_history   = st.slider("Chart history (months)", 36, 480, 120)

        # ── Models ───────────────────────────────────────────────────────────
        st.markdown("<div class='section-hdr'>MODELS</div>", unsafe_allow_html=True)
        outcome     = st.selectbox("Outcome variable",
                                   ["ip_yoy", "cpi_inflation", "gdp_growth",
                                    "unemp", "sp500_return", "ip_growth"],
                                   index=0)
        gar_horizon = st.select_slider("GaR horizon (months)", [3, 6, 12], value=6)
        lp_horizon  = st.select_slider("LP horizon (months)",  [12, 18, 24], value=18)
        n_bootstrap = st.select_slider("Bootstrap reps", [100, 250, 500], value=250)
        use_gipi    = st.checkbox("Include GIPI in GaR", value=True)

        # ── Display ──────────────────────────────────────────────────────────
        st.markdown("<div class='section-hdr'>DISPLAY</div>", unsafe_allow_html=True)
        show_gipi   = st.checkbox("Show GIPI Panel", value=True)
        show_inf    = st.checkbox("Show Inflation Channels", value=True)
        show_var    = st.checkbox("Show VAR / FEVD", value=False)
        show_granger= st.checkbox("Show Granger Tests", value=True)

        st.divider()
        st.markdown(
            "<div style='font-size:8px;color:#334155;line-height:1.7'>"
            "Data: FRED · Yahoo Finance · Iacoviello GPR · NY Fed GSCPI<br>"
            "       EIA Arab Light · FAO Food Price Index<br>"
            "Layer 0: GDELT · Anthropic Claude · AIS Tanker Proxy<br>"
            "Models: LP (Jordà 2005) · GaR (Adrian+ 2019) · VAR<br>"
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            "</div>", unsafe_allow_html=True)

    return dict(
        refresh=refresh, run_l0=run_l0, lookback=lookback, use_llm=use_llm,
        show_raw_hl=show_raw_hl, n_history=n_history,
        outcome=outcome, gar_horizon=gar_horizon, lp_horizon=lp_horizon,
        n_bootstrap=n_bootstrap, use_gipi=use_gipi,
        show_gipi=show_gipi, show_inf=show_inf,
        show_var=show_var, show_granger=show_granger,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 0 PANEL
# ═══════════════════════════════════════════════════════════════════════════════

def render_layer0_panel(opts: dict) -> None:
    """Full Layer 0 real-time event detection panel."""
    st.markdown("<div class='section-hdr'>00 — LAYER 0: REAL-TIME EVENT DETECTION</div>",
                unsafe_allow_html=True)

    from config import cfg
    with st.spinner("Running Layer 0 (GDELT + CAMEO + AIS) …"):
        sig = load_event_signal(
            lookback_hours=opts["lookback"],
            use_llm=opts["use_llm"],
            _key=cfg.ANTHROPIC_KEY,
        )

    if "error" in sig and not sig.get("_from_cache"):
        st.warning(f"⚠ Layer 0 live run failed: `{sig['error']}`")
        return

    regime = sig.get("regime", "UNKNOWN")
    sev    = sig.get("severity_score", 0.0)
    gpr_z  = sig.get("gpr_nowcast", 0.0)
    codes  = sig.get("cameo_codes", [])
    ais_an = sig.get("ais_anomaly", False)
    n_art  = sig.get("article_count", 0)
    hr     = sig.get("location_hit_rate", 0.0)
    llm    = sig.get("llm_used", False)
    summ   = sig.get("llm_summary", "")
    headlines = sig.get("top_headlines", [])
    ais_s  = sig.get("ais_signal") or {}

    # ── Alert banner ─────────────────────────────────────────────────────────
    icon = {"CRISIS": "🔴", "ELEVATED": "🟡", "CALM": "🟢"}.get(regime, "⚪")
    css_cls = f"alert-{regime.lower()}"
    col_r = REGIME_COLOR.get(regime, AMBER)

    st.markdown(f"""
    <div class="{css_cls}">
      <span style='font-size:18px'>{icon}</span>
      <span style='color:{col_r};font-weight:700;font-size:13px;
                   letter-spacing:2px;margin-left:8px'>{regime}</span>
      <span style='color:#64748b;font-size:10px;margin-left:16px'>
        Severity {sev:.1f}/10 &nbsp;|&nbsp;
        GPR nowcast z = {gpr_z:.2f} &nbsp;|&nbsp;
        CAMEO: {', '.join(codes) if codes else 'none'} &nbsp;|&nbsp;
        {n_art} articles ({hr:.0%} ME) &nbsp;|&nbsp;
        {'LLM-coded' if llm else 'Rule-based CAMEO'}
      </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Severity", f"{sev:.1f}/10",
              delta="⚠ HIGH" if sev >= 5 else None,
              delta_color="inverse" if sev >= 5 else "off")
    c2.metric("GPR Nowcast (z)", f"{gpr_z:.2f}")
    c3.metric("CAMEO Codes", ", ".join(codes[:2]) if codes else "none")
    c4.metric("AIS Anomaly", "⚠ YES" if ais_an else "NO",
              delta="Hormuz signal" if ais_an else None,
              delta_color="inverse" if ais_an else "off")
    c5.metric("Articles", f"{n_art}  ({hr:.0%} ME)")

    # ── Two-column detail ─────────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("**Top headlines**")
        for i, h in enumerate(headlines[:6], 1):
            st.markdown(
                f"<div style='font-size:10px;color:#94a3b8;"
                f"padding:2px 0;border-bottom:1px solid #1e2d4a'>"
                f"{i}. {h[:120]}</div>", unsafe_allow_html=True)
        if summ:
            st.markdown(
                f"<div class='ais-box'>"
                f"<span style='color:{MUTED};font-size:9px'>LLM SUMMARY</span><br>"
                f"<span style='font-size:10px;color:{TEXT}'>{summ}</span></div>",
                unsafe_allow_html=True)

    with col_b:
        # AIS tanker proxy detail
        tanker_z  = ais_s.get("tanker_z", 0.0) if ais_s else 0.0
        bw_spread = ais_s.get("brent_wti_spread", 0.0) if ais_s else 0.0
        bw_z      = ais_s.get("brent_wti_z", 0.0) if ais_s else 0.0
        tickers   = ais_s.get("tickers_used", []) if ais_s else []

        ais_color = RED if ais_an else GREEN
        st.markdown(f"""
        <div class='ais-box'>
          <div style='color:{MUTED};font-size:9px;letter-spacing:2px'>
            AIS TANKER PROXY — STRAIT OF HORMUZ</div>
          <div style='margin-top:8px'>
            <div style='color:{ais_color};font-weight:700;font-size:12px'>
              {'⚠  ANOMALY DETECTED' if ais_an else '✓  Normal'}</div>
            <div style='font-size:10px;color:{TEXT};margin-top:6px'>
              Tanker basket z: <b>{tanker_z:+.2f}</b><br>
              Brent–WTI spread: <b>${bw_spread:.1f}</b>  (z={bw_z:+.2f})<br>
              Tickers: {', '.join(tickers) if tickers else 'N/A'}<br>
              <span style='color:{MUTED};font-size:9px'>
                Anomaly = tanker |z| > 1.5 AND Brent–WTI z > 1.5</span>
            </div>
          </div>
        </div>
        <div class='ais-box' style='margin-top:6px'>
          <div style='color:{MUTED};font-size:9px;letter-spacing:2px'>
            CAMEO SEVERITY SCALE</div>
          <div style='font-size:9px;color:{TEXT};margin-top:4px'>
            0–3  Diplomatic tensions / protests<br>
            3–5  Military posturing / coercion<br>
            5–7  Strikes, clashes, blockades<br>
            7–9  Major attacks, naval action<br>
            9–10 WMD / critical infrastructure
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Severity gauge (plotly)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sev,
            gauge=dict(
                axis=dict(range=[0, 10],
                          tickfont=dict(color=MUTED, size=9)),
                bar=dict(color=REGIME_COLOR.get(regime, AMBER), thickness=0.35),
                bgcolor=PANEL,
                borderwidth=0,
                steps=[
                    dict(range=[0, 3.3], color="#0f1628"),
                    dict(range=[3.3, 6.6], color="#1a1505"),
                    dict(range=[6.6, 10],  color="#1a0505"),
                ],
                threshold=dict(line=dict(color=RED, width=2),
                               thickness=0.6, value=7.0),
            ),
            number=dict(font=dict(color=REGIME_COLOR.get(regime, AMBER), size=28)),
            title=dict(text="Severity", font=dict(color=MUTED, size=9)),
        ))
        fig_g.update_layout(
            paper_bgcolor=BG, plot_bgcolor=BG,
            height=160, margin=dict(l=10, r=10, t=20, b=5),
        )
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

    # ── Raw headlines expander ────────────────────────────────────────────────
    if opts.get("show_raw_hl") and headlines:
        with st.expander("All headlines"):
            for h in headlines:
                st.markdown(f"<div style='font-size:9px;color:#94a3b8'>{h}</div>",
                            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    opts = render_sidebar()

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown("""
    <div style='padding:2px 0 14px'>
      <div style='font-size:9px;letter-spacing:3px;color:#475569'>REAL-TIME TAIL RISK</div>
      <h1 style='font-size:20px;font-weight:700;color:#f1f5f9;margin:3px 0'>
        GeoShock — Geopolitical Risk → US Macro Forecasting
      </h1>
      <p style='font-size:10px;color:#475569;margin:0'>
        Layer 0 Event Detection · Local Projections (Jordà 2005) ·
        Growth-at-Risk + GIPI (Adrian+ 2019) · Cholesky VAR
      </p>
    </div>""", unsafe_allow_html=True)

    # ── Load macro data ───────────────────────────────────────────────────────
    with st.status("Loading macro dataset …", expanded=False) as data_status:
        try:
            df = load_data(refresh=opts["refresh"])
            data_status.update(label="Macro data loaded", state="complete")
        except Exception as e:
            st.error(f"Data load error: {e}")
            from data.pipeline import _synthetic_gpr, _synthetic_fred, engineer_features
            gpr = _synthetic_gpr().to_frame("gpr")
            fred = _synthetic_fred({})
            raw  = gpr.join(fred, how="outer")
            df   = engineer_features(raw)
            data_status.update(label="Macro data loaded (synthetic fallback)", state="complete")

    if df is None or df.empty:
        st.error("No data available.")
        return

    latest = df.dropna(subset=["gpr_level"]).iloc[-1]
    prev   = df.dropna(subset=["gpr_level"]).iloc[-2]
    last_dt = df.dropna(subset=["gpr_level"]).index[-1]

    # ── Top KPI row ───────────────────────────────────────────────────────────
    gpr_v  = latest.get("gpr_level", np.nan)
    gpr_z  = latest.get("gpr_z", np.nan)
    regime = str(latest.get("regime", "N/A"))
    rc     = REGIME_COLOR.get(regime, AMBER)

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("GPR Index", f"{gpr_v:.0f}" if not np.isnan(gpr_v) else "N/A",
              f"{gpr_v - prev.get('gpr_level',gpr_v):+.0f} MoM" if not np.isnan(gpr_v) else None)
    c2.metric("GPR Z-Score", f"{gpr_z:.2f}" if not np.isnan(gpr_z) else "N/A")
    c3.metric("Regime", regime.upper(), delta_color="off")
    c4.metric("VIX",    f"{latest.get('vix',np.nan):.1f}" if 'vix' in latest.index else "N/A",
              f"{latest.get('vix_change',np.nan):+.1f}" if 'vix_change' in latest.index else None)
    c5.metric("WTI",    f"${latest.get('oil_price',np.nan):.0f}" if 'oil_price' in latest.index else "N/A",
              f"{latest.get('oil_return',np.nan):+.1f}% MoM" if 'oil_return' in latest.index else None)
    c6.metric("HY Spread", f"{latest.get('hy_spread',np.nan):.0f}bps" if 'hy_spread' in latest.index else "N/A")

    gipi_v = latest.get("gipi", np.nan)
    c7.metric("GIPI", f"{gipi_v:.2f}" if not np.isnan(gipi_v) else "N/A",
              help="Geopolitical Inflation Pressure Index (z-score)")

    # Load Layer 0 nowcast for display (non-blocking — uses cached value if available)
    try:
        import json as _json, pathlib as _pl
        _root = _pl.Path(__file__).parent.parent
        _sig_path = _root / "outputs" / "event_signal.json"
        _l0 = _json.loads(_sig_path.read_text()) if _sig_path.exists() else {}
        _nowcast_z   = float(_l0.get("gpr_nowcast", float("nan")))
        _l0_regime   = str(_l0.get("regime", "")).upper()
        _l0_sev      = float(_l0.get("severity_score", float("nan")))
        _l0_articles = int(_l0.get("article_count", 0))
        _l0_ok = not (isinstance(_nowcast_z, float) and _nowcast_z != _nowcast_z)
    except Exception:
        _l0_ok = False

    import datetime as _dt
    _now_month = _dt.datetime.now().strftime("%B %Y")
    _lag_months = ((_dt.datetime.now().year - last_dt.year) * 12
                   + _dt.datetime.now().month - last_dt.month)

    if _l0_ok:
        _nc_color = "#ef4444" if _nowcast_z >= 2.5 else "#f59e0b" if _nowcast_z >= 1.5 else "#10b981"
        _nowcast_html = (
            f"&nbsp;&nbsp;⚡ <b>Nowcast ({_now_month}):</b> "
            f"<span style='color:{_nc_color}'>z={_nowcast_z:.2f} &nbsp;{_l0_regime}</span>"
            f"&nbsp; <span style='color:#64748b;font-size:9px'>"
            f"[bridges {_lag_months}-month GPR lag · {_l0_articles} articles]</span>"
        )
    else:
        _nowcast_html = "&nbsp;&nbsp;<span style='color:#64748b'>⚡ Nowcast: run Layer 0 to update</span>"

    st.markdown(f"""
    <div style='background:#0c1122;border:1px solid #1e2d4a;border-left:4px solid {rc};
    border-radius:5px;padding:6px 14px;margin:6px 0;font-size:10px;color:{rc}'>
      📅 <b>Official GPR: {last_dt.strftime('%B %Y')}</b> &nbsp;|&nbsp;
      GPR={gpr_v:.0f} (z={gpr_z:.2f}) &nbsp;|&nbsp; Regime: <b>{regime.upper()}</b>
      {_nowcast_html}
    </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Section 00: Layer 0 Event Detection
    # ─────────────────────────────────────────────────────────────────────────
    if opts.get("run_l0", False):
        load_event_signal.clear()

    with st.status("00 — Loading Layer 0: Event Detection …", expanded=True) as s00:
        render_layer0_panel(opts)
        s00.update(label="00 — Layer 0: Event Detection  ✓", state="complete", expanded=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Section 01: GPR Monitor
    # ─────────────────────────────────────────────────────────────────────────
    with st.status("01 — Loading GPR Monitor …", expanded=True) as s01:
        c_a, c_b = st.columns([3, 1])
        with c_a:
            st.plotly_chart(fig_gpr_series(df, n=opts["n_history"]), use_container_width=True, config={"displayModeBar": False})
        with c_b:
            if "regime" in df.columns:
                rc_cnt = df["regime"].value_counts()
                fig_rc = go.Figure(go.Bar(
                    x=rc_cnt.index.tolist(), y=rc_cnt.values,
                    marker_color=[REGIME_COLOR.get(r, AMBER) for r in rc_cnt.index],
                    opacity=0.8,
                ))
                _lr = {**BASE_LAYOUT, "margin": dict(l=36, r=8, t=28, b=28)}
                fig_rc.update_layout(**_lr, height=180, showlegend=False,
                                     title=dict(text="Regime distribution",
                                                font=dict(size=9, color=MUTED), x=0.01))
                st.plotly_chart(fig_rc, use_container_width=True, config={"displayModeBar": False})

            if "gpr_z" in df.columns:
                sub = df.dropna(subset=["gpr_z"]).iloc[-opts["n_history"]:]
                fig_z = go.Figure(go.Scatter(
                    x=sub.index, y=sub["gpr_z"],
                    line=dict(color=BLUE, width=1.5), fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.08)",
                    hovertemplate="%{x|%b %Y}: z=%{y:.2f}<extra></extra>",
                ))
                fig_z.add_hline(y=1.5, line=dict(color=AMBER, width=0.8, dash="dash"))
                fig_z.add_hline(y=2.5, line=dict(color=RED,   width=0.8, dash="dash"))
                _lz = {**BASE_LAYOUT, "margin": dict(l=36, r=8, t=28, b=28)}
                fig_z.update_layout(**_lz, height=180,
                                    title=dict(text="GPR Z-Score",
                                               font=dict(size=9, color=MUTED), x=0.01))
                st.plotly_chart(fig_z, use_container_width=True, config={"displayModeBar": False})
        s01.update(label="01 — GPR Monitor  ✓", state="complete", expanded=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Section 02: GIPI Panel
    # ─────────────────────────────────────────────────────────────────────────
    if opts.get("show_gipi") and "gipi" in df.columns:
        with st.status("02 — Loading GIPI Panel …", expanded=True) as s02:
            g1, g2 = st.columns([3, 1])
            with g1:
                st.plotly_chart(fig_gipi(df, n=opts["n_history"]), use_container_width=True, config={"displayModeBar": False})
            with g2:
                gipi_last = df["gipi"].dropna().iloc[-1] if df["gipi"].notna().any() else np.nan
                gipi_3m   = df["gipi"].dropna().iloc[-4:-1].mean() if df["gipi"].notna().sum() > 3 else np.nan
                gipi_rg   = ("CRISIS" if gipi_last >= 2.5
                             else "ELEVATED" if gipi_last >= 1.5 else "CALM")
                gcol          = REGIME_COLOR.get(gipi_rg, MUTED)
                gipi_last_str = f"{gipi_last:.2f}" if not np.isnan(gipi_last) else "N/A"
                gipi_3m_str   = f"{gipi_3m:.2f}"   if not np.isnan(gipi_3m)   else "N/A"
                st.markdown(f"""
                <div style='background:#0c1122;border:1px solid #1e2d4a;
                            border-left:4px solid {gcol};border-radius:6px;
                            padding:14px;margin-top:8px'>
                  <div style='color:{MUTED};font-size:9px'>GIPI NOW</div>
                  <div style='color:{gcol};font-size:26px;font-weight:700'>
                    {gipi_last_str}</div>
                  <div style='color:{MUTED};font-size:10px;margin-top:4px'>
                    3-month avg: {gipi_3m_str}<br>
                    Regime: <b style='color:{gcol}'>{gipi_rg}</b>
                  </div>
                </div>
                <div style='background:#0c1122;border:1px solid #1e2d4a;
                            border-radius:6px;padding:12px;margin-top:8px;
                            font-size:9px;color:{MUTED}'>
                  <b style='color:{TEXT}'>GIPI = PC1 of:</b><br>
                  · NY Fed GSCPI<br>
                  · Import Price YoY<br>
                  · Global Energy Price YoY<br>
                  · Global Food Price YoY<br>
                  · ΔBreakeven 5Y<br><br>
                  Higher GIPI = more geopolitical<br>inflation transmission pressure.
                </div>
                """, unsafe_allow_html=True)
            s02.update(label="02 — GIPI Panel  ✓", state="complete", expanded=True)

    # ── Resolve outcome — use fallback if selected column missing from df ─────
    _pref = [opts["outcome"], "ip_yoy", "ip_growth", "cpi_inflation",
             "gdp_growth", "unemp", "sp500_return"]
    safe_outcome = next(
        (c for c in _pref if c in df.columns and df[c].notna().sum() >= 24),
        None,
    )
    if safe_outcome is None:
        st.error("❌ No outcome variable available. Delete `data/cache/` and re-run `python run.py`.")
        return
    # silently use fallback — no info banner shown to user

    df_json = df.to_json()

    # ─────────────────────────────────────────────────────────────────────────
    # Section 03: Growth-at-Risk
    # ─────────────────────────────────────────────────────────────────────────
    with st.status("03 — Estimating Growth-at-Risk …", expanded=True) as s03:
        try:
            gar_data = compute_gar(df_json, safe_outcome,
                                   opts["gar_horizon"], opts["use_gipi"])

            r1, r2 = st.columns([3, 1])
            with r1:
                st.plotly_chart(fig_gar_fan(gar_data, opts["gar_horizon"]), use_container_width=True, config={"displayModeBar": False})
            with r2:
                st.plotly_chart(fig_gar_dist(gar_data), use_container_width=True, config={"displayModeBar": False})
                gk1, gk2 = st.columns(2)
                gk1.metric("GaR₅",    f"{gar_data['gar_5']:+.1f}%")
                gk2.metric("Median",   f"{gar_data['median']:+.1f}%")
                gk1.metric("P(neg)",   f"{gar_data['prob_neg']:.1%}")
                gk2.metric("P(rec<-2%)",f"{gar_data['prob_rec']:.1%}")
                st.metric("Cond. Skewness", f"{gar_data['skewness']:+.2f}",
                          help="+ve = upside risk dominates; -ve = downside tail heavy")
            s03.update(label="03 — Growth-at-Risk  ✓", state="complete", expanded=True)
        except Exception as e:
            logging.error(f"GaR: {e}")
            s03.update(label="03 — Growth-at-Risk  ✗", state="error", expanded=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Section 04: Local Projections
    # ─────────────────────────────────────────────────────────────────────────
    with st.status("04 — Estimating Local Projections …", expanded=True) as s04:
        try:
            lp_data = compute_lp(df_json, safe_outcome, "gpr_shock",
                                 opts["lp_horizon"], opts["n_bootstrap"])
            lc1, lc2 = st.columns([2, 1])
            with lc1:
                st.plotly_chart(fig_lp_irf(lp_data, opts["outcome"]), use_container_width=True, config={"displayModeBar": False})
            with lc2:
                peak_h = int(np.argmax(np.abs(lp_data["betas"])))
                peak_b = lp_data["betas"][peak_h]
                st.markdown(f"""
                <div style='background:#0c1122;border:1px solid #1e2d4a;
                            border-radius:6px;padding:14px;margin-top:8px'>
                  <div style='color:{MUTED};font-size:9px'>LP SUMMARY</div>
                  <div style='font-size:11px;color:{TEXT};margin-top:8px'>
                    Peak response at h={lp_data['horizons'][peak_h]}<br>
                    Peak β = {peak_b:.3f}<br>
                    Horizon = {opts['lp_horizon']} months<br>
                    Bootstrap reps = {opts['n_bootstrap']}
                  </div>
                </div>""", unsafe_allow_html=True)
            s04.update(label="04 — Local Projections  ✓", state="complete", expanded=True)
        except Exception as e:
            logging.error(f"LP: {e}")
            s04.update(label="04 — Local Projections  ✗", state="error", expanded=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Section 05: VAR / FEVD
    # ─────────────────────────────────────────────────────────────────────────
    if opts.get("show_var"):
        with st.status("05 — Loading VAR / FEVD …", expanded=True) as s05:
            fevd_path    = Path(__file__).parent.parent / "outputs" / "fevd.png"
            granger_path = Path(__file__).parent.parent / "outputs" / "granger_causality.csv"

            if fevd_path.exists():
                vc1, vc2 = st.columns([2, 1])
                with vc1:
                    st.image(str(fevd_path), caption="FEVD — GPR share of forecast error variance")
                with vc2:
                    if opts.get("show_granger") and granger_path.exists():
                        gc = pd.read_csv(granger_path)
                        st.markdown("**Granger causality: GPR → X**")
                        for _, row in gc.iterrows():
                            pval = row.get("p_value", row.get("pvalue", np.nan))
                            sig  = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                            col  = RED if pval < 0.05 else TEXT
                            st.markdown(
                                f"<div style='font-size:10px;color:{col}'>"
                                f"{row.get('variable','')}: p={pval:.3f} {sig}</div>",
                                unsafe_allow_html=True)
                s05.update(label="05 — VAR / FEVD  ✓", state="complete", expanded=True)
            else:
                st.info("Run `python run.py` first to generate VAR outputs.")
                s05.update(label="05 — VAR / FEVD (no data)", state="complete", expanded=False)

    # ─────────────────────────────────────────────────────────────────────────
    # Section 06: Inflation Channels
    # ─────────────────────────────────────────────────────────────────────────
    if opts.get("show_inf"):
        with st.status("06 — Loading Inflation Channels …", expanded=True) as s06:
            ic_fig = fig_inflation_channels(df, n=opts["n_history"])
            if ic_fig.data:
                st.plotly_chart(ic_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Inflation channel data not yet fetched. Run `python run.py` first.")

            # Channel data table
            channel_cols = [c for c in [
                "cpi_inflation", "core_inflation", "breakeven_5y", "breakeven_10y",
                "import_price_yoy", "import_xfuel_yoy", "global_energy_yoy",
                "global_food_yoy", "fao_food_yoy", "arab_wti_spread", "gipi",
            ] if c in df.columns]
            if channel_cols:
                recent = df[channel_cols].dropna(how="all").tail(12)
                st.markdown("**Last 12 months — key inflation transmission variables**")
                st.dataframe(
                    recent.style.format("{:.2f}", na_rep="—")
                           .background_gradient(cmap="RdYlGn_r", axis=0),
                    use_container_width=True,
                )
            s06.update(label="06 — Inflation Channels  ✓", state="complete", expanded=True)


if __name__ == "__main__":
    main()
