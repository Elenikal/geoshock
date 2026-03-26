"""
data/pipeline.py
─────────────────────────────────────────────────────────────────────────────
End-to-end data pipeline for the Geopolitical Shock → US Macro Model.

Sources
-------
1. FRED API          — macro, financial, uncertainty series (free key required)
2. Yahoo Finance     — VIX, S&P 500, ETFs  (no key required)
3. Matteo Iacoviello — GPR Index Excel file (direct download, no key)
4. NY Fed            — GSCPI CSV (direct download, no key)

All data is cached locally in data/cache/ as Parquet files.
Call pipeline.build() to refresh everything.

Usage
-----
  from data.pipeline import DataPipeline
  dp = DataPipeline()
  df = dp.build()          # fetches + merges all sources
  df = dp.load_cached()    # loads from disk without refetching
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import io
import os
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _end_date() -> str:
    """Return cfg.END_DATE, defaulting to today if None."""
    if cfg.END_DATE:
        return str(cfg.END_DATE)
    return pd.Timestamp.today().strftime("%Y-%m-%d")


# ── Optional imports (graceful fallback) ─────────────────────────────────────
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    log.warning("fredapi not installed — install with: pip install fredapi")

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    log.warning("yfinance not installed — install with: pip install yfinance")

try:
    import sys
    _proj_root = str(Path(__file__).resolve().parent.parent)
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)
    from config import cfg
    log.info(f"[DEBUG] config.py loaded OK — FRED_KEY len={len(cfg.FRED_KEY)}, "
             f"FRED_CORE keys={len(getattr(cfg, 'FRED_CORE', {}))}")
except Exception as _cfg_err:
    log.error(f"[DEBUG] config.py import FAILED: {_cfg_err}")
    class cfg:
        FRED_KEY = os.getenv("FRED_API_KEY", "")
        START_DATE = "1985-01-01"
        END_DATE = "2025-12-31"
        DATA_DIR = Path("data/cache")
        FRED_SERIES = {}
        FRED_CORE = {}
        FRED_INFLATION = {}
        YF_TICKERS = {}


# ═══════════════════════════════════════════════════════════════════════════════
#  GPR INDEX  (Caldara & Iacoviello 2022, Fed Board of Governors)
# ═══════════════════════════════════════════════════════════════════════════════

GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
GSCPI_URL = "https://www.newyorkfed.org/medialibrary/media/research/gscpi/downloads/GSCPI_data.xlsx"


def fetch_gpr_index() -> pd.Series:
    """
    Download the Geopolitical Risk (GPR) Index from Iacoviello's website.
    Falls back to the FRED GEPUCURRENT series if download fails.

    Returns monthly pd.Series from 1985-01-01, name='gpr'.
    """
    log.info("Fetching GPR index from Iacoviello website …")
    try:
        resp = requests.get(GPR_URL, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0)

        # Locate date and US GPR columns (structure varies slightly by version)
        date_col = [c for c in df.columns if "year" in str(c).lower() or "date" in str(c).lower()]
        gpr_col = [c for c in df.columns if "gprc_us" in str(c).lower() or
                   ("gpr" in str(c).lower() and "us" not in str(c).lower() and
                    "threat" not in str(c).lower() and "act" not in str(c).lower())]

        if not date_col or not gpr_col:
            # Fallback: assume first col = year/month, one of the later cols is overall GPR
            df.columns = [str(c).strip() for c in df.columns]
            year_col = df.columns[0]
            month_col = df.columns[1] if len(df.columns) > 1 else None
            gpr_col_name = df.columns[2]
        else:
            year_col = date_col[0]
            gpr_col_name = gpr_col[0]
            month_col = None

        # Build date index
        if month_col:
            df["date"] = pd.to_datetime(
                df[year_col].astype(int).astype(str) + "-" +
                df[month_col].astype(int).astype(str).str.zfill(2) + "-01"
            )
        else:
            df["date"] = pd.to_datetime(df[year_col])

        s = df.set_index("date")[gpr_col_name].astype(float)
        s.name = "gpr"
        s = s[s.index >= cfg.START_DATE]
        log.info(f"  GPR: {len(s)} observations ({s.index[0].date()} → {s.index[-1].date()})")
        return s

    except Exception as e:
        log.warning(f"  GPR direct download failed ({e}). Trying FRED fallback …")
        return _fetch_gpr_fred()


def _fetch_gpr_fred() -> pd.Series:
    """Fallback: fetch GPR from FRED series GEPUCURRENT."""
    if not FRED_AVAILABLE:
        log.error("fredapi not available either. Returning synthetic GPR for demo.")
        return _synthetic_gpr()
    fred = Fred(api_key=cfg.FRED_KEY)
    s = fred.get_series("GEPUCURRENT", observation_start=cfg.START_DATE,
                        observation_end=_end_date())
    s.name = "gpr"
    return s.resample("MS").mean()


def _synthetic_gpr() -> pd.Series:
    """Generate synthetic GPR for testing when no internet access."""
    import numpy as np
    np.random.seed(42)
    dates = pd.date_range(cfg.START_DATE, _end_date(), freq="MS")
    # Mean-reverting process with crisis spikes matching known episodes
    base = np.random.normal(100, 20, len(dates))
    base = pd.Series(base, index=dates)
    # Add known crisis spikes
    spikes = {
        "1990-08": 250, "1990-09": 280, "1991-01": 260,   # Gulf War I
        "2001-09": 350, "2001-10": 320,                     # 9/11
        "2003-03": 200, "2003-04": 180,                     # Iraq War
        "2011-02": 160, "2011-03": 175,                     # Arab Spring
        "2014-07": 165, "2014-08": 170,                     # Ukraine/ISIS
        "2019-09": 185,                                      # Abqaiq
        "2020-01": 175,                                      # Soleimani
        "2022-02": 300, "2022-03": 290,                     # Ukraine
        "2023-10": 185, "2023-11": 175,                     # Gaza
        "2024-04": 160, "2024-10": 155,                     # Iran-Israel
    }
    for ym, val in spikes.items():
        try:
            base[ym] = val
        except Exception:
            pass
    base = base.clip(lower=20)
    base.name = "gpr"
    return base


# ═══════════════════════════════════════════════════════════════════════════════
#  FRED  DATA
# ═══════════════════════════════════════════════════════════════════════════════

FRED_CORE = {
    "INDPRO":            "ip",
    "UNRATE":            "unemp",
    "CPIAUCSL":          "cpi",
    "CPILFESL":          "core_cpi",
    "A191RL1Q225SBEA":   "gdp_growth",   # quarterly → interpolated
    "GDPC1":             "gdp_level",    # quarterly
    "BAMLH0A0HYM2":      "hy_spread",
    "T10Y2Y":            "term_spread",
    "GS10":              "gs10",
    "FEDFUNDS":          "fedfunds",
    "DCOILWTICO":        "wti",
    "UMCSENT":           "umcsent",
    "GEPUCURRENT":       "gpr_fred",     # backup GPR
    "WTISPLC":           "wti_spot",
    "IPG211111CS":       "ip_oil",
}

# ── NEW: Inflation transmission channel series ────────────────────────────────
FRED_INFLATION_CHANNELS = {
    # Breakeven inflation expectations (market-implied)
    "T5YIE":             "breakeven_5y",     # 5-Year TIPS breakeven inflation rate
    "T10YIE":            "breakeven_10y",    # 10-Year TIPS breakeven inflation rate
    # Import prices (key cost-push transmission channel)
    "IR":                "import_price_all",  # Import Price Index: All Commodities
    "IREXFUELS":         "import_price_xfuel",# Import Price Index excl. Fuels
    "IR10":              "import_price_fuel", # Import Price Index: Fuels & Lubricants
    # Energy — broader than WTI alone
    "MHHNGSP":           "henry_hub",         # Henry Hub Natural Gas Spot Price
    "PNRGINDEXM":        "global_energy_idx", # IMF Global Price of Energy Index
    "CPIENGSL":          "cpi_energy",        # CPI Energy Component (SA)
    "PPIDES":            "ppi_energy",        # PPI Final Demand: Energy
    # Food prices
    "PFOODINDEXM":       "global_food_idx",   # IMF Global Price of Food Index
    "CUSR0000SAF11":     "cpi_food_home",     # CPI Food at Home (SA)
    # Strategic Petroleum Reserve (supply buffer signal)
    "WCSSTUS":           "spr_stocks",        # US SPR Weekly Stocks (thousand barrels)
    # OPEC crude oil production (EIA/STEO via FRED)
    # Spare capacity proxy = 36-month rolling max minus current production
    "PAPR_OPEC":         "opec_production",   # EIA STEO OPEC Crude Production (mb/d)
    # USD index (pass-through amplifier)
    "DTWEXBGS":          "usd_index",         # Nominal Broad USD Index
}


def fetch_fred_series(series_map: dict | None = None) -> pd.DataFrame:
    """
    Fetch all FRED series and resample to monthly frequency.
    Now fetches both core macro series AND the six inflation transmission channels.
    """
    # Merge core + inflation channel series
    combined = {**FRED_CORE, **FRED_INFLATION_CHANNELS}
    if series_map is None:
        series_map = combined

    if not FRED_AVAILABLE:
        log.warning("fredapi unavailable — generating synthetic FRED data for demo.")
        return _synthetic_fred(series_map)

    fred = Fred(api_key=cfg.FRED_KEY)
    frames = {}
    for fred_id, col in series_map.items():
        try:
            s = fred.get_series(
                fred_id,
                observation_start=cfg.START_DATE,
                observation_end=_end_date(),
            )
            # Resample to month-start
            s = s.resample("MS").last().rename(col)
            frames[col] = s
            log.info(f"  FRED {fred_id:20s} → {col}: {len(s)} obs")
        except Exception as e:
            log.warning(f"  FRED {fred_id} failed: {e}")

    return pd.DataFrame(frames)


def _synthetic_fred(series_map: dict) -> pd.DataFrame:
    """Generate plausible synthetic FRED data for offline testing."""
    np.random.seed(123)
    dates = pd.date_range(cfg.START_DATE, _end_date(), freq="MS")
    n = len(dates)
    df = pd.DataFrame(index=dates)

    for col in series_map.values():
        if col == "ip":
            df[col] = 80 + np.cumsum(np.random.normal(0.2, 1.0, n))
        elif col == "unemp":
            df[col] = np.clip(5 + np.random.normal(0, 1, n) + np.sin(np.arange(n)/60), 2, 15)
        elif col == "cpi":
            df[col] = 120 * np.exp(np.cumsum(np.random.normal(0.002, 0.003, n)))
        elif col == "core_cpi":
            df[col] = 118 * np.exp(np.cumsum(np.random.normal(0.002, 0.002, n)))
        elif col == "gdp_growth":
            df[col] = np.random.normal(2.5, 2.0, n)
        elif col == "gdp_level":
            df[col] = 12000 * np.exp(np.cumsum(np.random.normal(0.006, 0.005, n)))
        elif col == "hy_spread":
            df[col] = np.clip(4 + np.random.normal(0, 1.5, n), 1.5, 20)
        elif col == "term_spread":
            df[col] = np.random.normal(1.0, 0.8, n)
        elif col in ("gs10", "fedfunds"):
            df[col] = np.clip(3 + np.random.normal(0, 1.5, n), 0, 20)
        elif col in ("wti", "wti_spot"):
            df[col] = np.clip(50 + np.cumsum(np.random.normal(0, 3, n)), 10, 140)
        elif col == "umcsent":
            df[col] = np.clip(85 + np.random.normal(0, 8, n), 40, 120)
        elif col == "ip_oil":
            df[col] = 80 + np.random.normal(0, 5, n)
        # --- new inflation channel synthetics ---
        elif col in ("breakeven_5y", "breakeven_10y"):
            df[col] = np.clip(2.0 + np.random.normal(0, 0.4, n), 0.5, 4.5)
        elif col in ("import_price_all", "import_price_fuel"):
            df[col] = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.008, n)))
        elif col == "import_price_xfuel":
            df[col] = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.004, n)))
        elif col == "henry_hub":
            df[col] = np.clip(3 + np.random.lognormal(0, 0.5, n), 1, 15)
        elif col in ("global_energy_idx", "global_food_idx"):
            df[col] = 100 * np.exp(np.cumsum(np.random.normal(0.002, 0.012, n)))
        elif col in ("cpi_energy", "ppi_energy"):
            df[col] = 200 * np.exp(np.cumsum(np.random.normal(0.002, 0.010, n)))
        elif col == "cpi_food_home":
            df[col] = 150 * np.exp(np.cumsum(np.random.normal(0.002, 0.004, n)))
        elif col == "spr_stocks":
            df[col] = np.clip(700000 - np.arange(n) * 1000 + np.random.normal(0, 20000, n), 300000, 800000)
        elif col == "usd_index":
            df[col] = np.clip(100 + np.cumsum(np.random.normal(0, 0.5, n)), 70, 130)
        else:
            df[col] = np.random.normal(100, 10, n)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  EIA: Arab Light Crude Landed Cost (Geopolitical Oil Premium Proxy)
# ═══════════════════════════════════════════════════════════════════════════════

EIA_ARAB_LIGHT_URL = (
    "https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx"
    "?n=PET&s=ISA4990008&f=M"
)


def fetch_arab_light_eia() -> pd.Series:
    """
    Fetch EIA monthly US landed cost of Saudi Arabian Light crude oil.
    Used to construct the Arab Light–WTI geopolitical premium spread.
    Free, no API key required.
    Returns monthly pd.Series, name='arab_light'.
    """
    log.info("Fetching EIA Arab Light crude landed cost …")
    try:
        resp = requests.get(EIA_ARAB_LIGHT_URL, timeout=20)
        resp.raise_for_status()
        # EIA returns a tab-separated text file
        from io import StringIO
        lines = resp.text.strip().split("\n")
        # Find data lines: "YYYY-MMM  value"
        data_rows = []
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                try:
                    date = pd.to_datetime(parts[0].strip(), format="%Y-%b")
                    val  = float(parts[1].strip())
                    data_rows.append((date, val))
                except (ValueError, IndexError):
                    continue
        if data_rows:
            s = pd.Series(
                {d: v for d, v in data_rows},
                name="arab_light"
            ).sort_index()
            s.index = s.index.to_period("M").to_timestamp("MS")
            s = s[s.index >= cfg.START_DATE]
            log.info(f"  EIA Arab Light: {len(s)} obs ({s.index[0].date()} → {s.index[-1].date()})")
            return s
        raise ValueError("No parseable data rows found")
    except Exception as e:
        log.warning(f"  EIA Arab Light fetch failed ({e}) — generating synthetic.")
        dates = pd.date_range(cfg.START_DATE, _end_date(), freq="MS")
        np.random.seed(1001)
        synthetic_wti = np.clip(50 + np.cumsum(np.random.normal(0, 3, len(dates))), 10, 140)
        # Arab Light typically trades ~$1–3 above WTI; spikes during ME crises
        spread = np.clip(np.random.normal(2.0, 1.5, len(dates)), -5, 15)
        s = pd.Series(synthetic_wti + spread, index=dates, name="arab_light")
        return s


# ═══════════════════════════════════════════════════════════════════════════════
#  FAO Food Price Index (alternative to FRED PFOODINDEXM)
# ═══════════════════════════════════════════════════════════════════════════════

FAO_URL = "https://www.fao.org/fileadmin/templates/worldfood/Reports_and_docs/Food_price_indices_data_jul24.xls"


def fetch_fao_food_index() -> pd.DataFrame:
    """
    Fetch FAO Food Price Index and sub-indices from FAO website.
    Sub-indices: Cereals, Oils, Dairy, Meat, Sugar.
    Falls back to FRED PFOODINDEXM if direct download fails.
    Returns monthly DataFrame with columns [fao_food, fao_cereals, fao_oils, fao_dairy, fao_meat, fao_sugar].
    """
    log.info("Fetching FAO Food Price Index …")
    try:
        resp = requests.get(FAO_URL, timeout=30)
        resp.raise_for_status()
        df_raw = pd.read_excel(io.BytesIO(resp.content), sheet_name="Monthly", skiprows=2,
                               engine="xlrd")
        # FAO format: Year | Month | Food | Cereals | Oils | Dairy | Meat | Sugar
        df_raw.columns = [str(c).strip().lower().replace(" ", "_") for c in df_raw.columns]
        # Find date columns
        year_col = [c for c in df_raw.columns if "year" in c][0]
        month_col = [c for c in df_raw.columns if "month" in c][0]
        df_raw = df_raw.dropna(subset=[year_col, month_col])
        df_raw["date"] = pd.to_datetime(
            df_raw[year_col].astype(int).astype(str) + "-" +
            df_raw[month_col].astype(int).astype(str).str.zfill(2) + "-01"
        )
        df_raw = df_raw.set_index("date")
        # Rename to standard names
        rename = {}
        for c in df_raw.columns:
            if "food" in c and "cereals" not in c: rename[c] = "fao_food"
            elif "cereal" in c: rename[c] = "fao_cereals"
            elif "oil" in c:    rename[c] = "fao_oils"
            elif "dairy" in c:  rename[c] = "fao_dairy"
            elif "meat" in c:   rename[c] = "fao_meat"
            elif "sugar" in c:  rename[c] = "fao_sugar"
        df_raw = df_raw.rename(columns=rename)
        cols = [c for c in ["fao_food","fao_cereals","fao_oils","fao_dairy","fao_meat","fao_sugar"]
                if c in df_raw.columns]
        df_out = df_raw[cols].apply(pd.to_numeric, errors="coerce")
        df_out = df_out.resample("MS").last()
        df_out = df_out[df_out.index >= cfg.START_DATE]
        log.info(f"  FAO Food Index: {len(df_out)} obs, cols: {list(df_out.columns)}")
        return df_out
    except Exception as e:
        log.warning(f"  FAO direct fetch failed ({e}) — using FRED PFOODINDEXM fallback.")
        return _fao_fallback_fred()


def _fao_fallback_fred() -> pd.DataFrame:
    """Use FRED IMF food price index as FAO fallback."""
    if not FRED_AVAILABLE:
        dates = pd.date_range(cfg.START_DATE, _end_date(), freq="MS")
        np.random.seed(2002)
        df = pd.DataFrame(index=dates)
        df["fao_food"]    = 100 * np.exp(np.cumsum(np.random.normal(0.002, 0.012, len(dates))))
        df["fao_cereals"] = 100 * np.exp(np.cumsum(np.random.normal(0.002, 0.015, len(dates))))
        df["fao_oils"]    = 100 * np.exp(np.cumsum(np.random.normal(0.002, 0.018, len(dates))))
        return df
    try:
        fred = Fred(api_key=cfg.FRED_KEY)
        s = fred.get_series("PFOODINDEXM",
                            observation_start=cfg.START_DATE,
                            observation_end=_end_date())
        s = s.resample("MS").last()
        df = pd.DataFrame({"fao_food": s, "fao_cereals": s * 0.9, "fao_oils": s * 1.1})
        log.info(f"  FAO fallback via FRED PFOODINDEXM: {len(df)} obs")
        return df
    except Exception as e:
        log.warning(f"  FRED PFOODINDEXM also failed: {e}")
        dates = pd.date_range(cfg.START_DATE, _end_date(), freq="MS")
        return pd.DataFrame({"fao_food": np.nan, "fao_cereals": np.nan}, index=dates)




# ═══════════════════════════════════════════════════════════════════════════════
#  YAHOO FINANCE
# ═══════════════════════════════════════════════════════════════════════════════

YF_MAP = {
    "^VIX":  "vix",
    "^GSPC": "sp500",
    "ITA":   "defense_etf",
    "XLE":   "energy_etf",
    "GLD":   "gold_etf",
    "TLT":   "tlt",
}


def fetch_yahoo_data(tickers: dict | None = None) -> pd.DataFrame:
    """
    Fetch adjusted close prices from Yahoo Finance and resample to monthly.
    """
    if tickers is None:
        tickers = YF_MAP

    if not YF_AVAILABLE:
        log.warning("yfinance unavailable — generating synthetic market data.")
        return _synthetic_yahoo(tickers)

    frames = {}
    for ticker, col in tickers.items():
        try:
            raw = yf.download(
                ticker,
                start=cfg.START_DATE,
                end=_end_date(),
                progress=False,
                auto_adjust=True,
            )
            # yfinance >=0.2.x may return MultiIndex columns — flatten to Series
            if isinstance(raw.columns, pd.MultiIndex):
                raw = raw.xs("Close", axis=1, level=0)
                if isinstance(raw, pd.DataFrame):
                    raw = raw.iloc[:, 0]
            else:
                raw = raw["Close"]
            s = raw.resample("MS").last().rename(col)
            frames[col] = s
            log.info(f"  Yahoo {ticker:10s} → {col}: {len(s)} obs")
        except Exception as e:
            log.warning(f"  Yahoo {ticker} failed: {e}")

    return pd.DataFrame(frames)


def _synthetic_yahoo(tickers: dict) -> pd.DataFrame:
    np.random.seed(456)
    dates = pd.date_range(cfg.START_DATE, _end_date(), freq="MS")
    n = len(dates)
    df = pd.DataFrame(index=dates)
    for col in tickers.values():
        if col == "vix":
            df[col] = np.clip(15 + np.random.lognormal(0, 0.5, n), 9, 80)
        elif col == "sp500":
            df[col] = 500 * np.exp(np.cumsum(np.random.normal(0.007, 0.04, n)))
        elif col in ("defense_etf", "energy_etf", "gold_etf", "tlt"):
            df[col] = 50 * np.exp(np.cumsum(np.random.normal(0.005, 0.03, n)))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  NY FED  GSCPI
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_gscpi() -> pd.Series:
    """
    Fetch NY Fed Global Supply Chain Pressure Index (starts 1998).

    Priority order:
      1. Local cache file  data/cache/gscpi_data.xls  (hand-downloaded from NY Fed)
      2. Live NY Fed URL   (CSV format, current as of 2025)
      3. Synthetic fallback
    """
    log.info("Fetching GSCPI from NY Fed …")

    def _parse_gscpi_df(df: pd.DataFrame) -> pd.Series:
        df.columns = [str(c).strip() for c in df.columns]
        date_col = df.columns[0]
        val_col  = df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        s = df.set_index(date_col)[val_col].resample("MS").last()
        s.name = "gscpi"
        return s

    # ── 1. Local file (most reliable) ────────────────────────────────────────
    local_paths = [
        Path("data/cache/gscpi_data.xls"),
        Path("data/cache/gscpi_data.xlsx"),
        Path("data/cache/gscpi_data.csv"),
    ]
    for local in local_paths:
        if local.exists():
            try:
                if local.suffix == ".csv":
                    df = pd.read_csv(local)
                elif local.suffix == ".xls":
                    df = pd.read_excel(local, engine="xlrd")
                else:
                    df = pd.read_excel(local, engine="openpyxl")
                s = _parse_gscpi_df(df)
                log.info(f"  GSCPI (local {local.name}): {len(s)} obs "
                         f"({s.index[0].date()} → {s.index[-1].date()})")
                return s
            except Exception as e:
                log.warning(f"  GSCPI local read failed ({e}) — trying URL …")

    # ── 2. Live URL (CSV, current NY Fed format) ──────────────────────────────
    for url in [
        "https://www.newyorkfed.org/medialibrary/media/research/gscpi/downloads/gscpi_data.csv",
        "https://www.newyorkfed.org/medialibrary/media/research/gscpi/downloads/GSCPI_data.xlsx",
    ]:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            if b"<!DOCTYPE" in resp.content[:200]:
                continue   # HTML error page
            if url.endswith(".csv"):
                df = pd.read_csv(io.StringIO(resp.content.decode("utf-8")))
            else:
                df = pd.read_excel(io.BytesIO(resp.content), engine="openpyxl")
            s = _parse_gscpi_df(df)
            log.info(f"  GSCPI (URL): {len(s)} obs "
                     f"({s.index[0].date()} → {s.index[-1].date()})")
            # Save to cache for next run
            s.to_csv("data/cache/gscpi_data.csv", header=True)
            return s
        except Exception as e:
            log.debug(f"  GSCPI URL {url} failed: {e}")

    # ── 3. Synthetic fallback ─────────────────────────────────────────────────
    log.warning("  GSCPI fetch failed — generating synthetic (results unreliable).")
    dates = pd.date_range("1998-01-01", _end_date(), freq="MS")
    np.random.seed(789)
    s = pd.Series(np.random.normal(0, 1, len(dates)), index=dates, name="gscpi")
    s["2021-06":"2022-03"] += 3.0
    return s



def _build_gipi(gipi_df: pd.DataFrame, n_components: int = 1) -> pd.Series:
    """
    Build the Geopolitical Inflation Pressure Index (GIPI) via PCA.

    GIPI = PC1 of [GSCPI, Import Price YoY, Global Energy YoY,
                   Global Food YoY, ΔBreakeven 5Y]

    All inputs are standardised before PCA. The sign convention is fixed
    so that higher GIPI = more inflationary pressure (positive loadings
    on GSCPI, energy, food; sign-flipped if needed).

    Returns monthly pd.Series named 'gipi', standardised to z-scores.
    Falls back to equal-weighted mean if fewer than 2 inputs available.
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        log.warning("scikit-learn not available — GIPI will use equal-weighted mean")
        valid = gipi_df.dropna(how="all", axis=1)
        if valid.empty:
            return pd.Series(np.nan, index=gipi_df.index, name="gipi")
        mean_z = valid.apply(lambda s: (s - s.mean()) / (s.std() + 1e-9)).mean(axis=1)
        return mean_z.rename("gipi")

    # Need at least 2 columns with sufficient overlap
    valid_cols = [c for c in gipi_df.columns
                  if gipi_df[c].notna().sum() >= 36]   # need 3 years minimum

    if len(valid_cols) < 2:
        log.warning(f"  GIPI: only {len(valid_cols)} valid inputs — using mean fallback")
        if len(valid_cols) == 1:
            s = gipi_df[valid_cols[0]]
            return ((s - s.mean()) / (s.std() + 1e-9)).rename("gipi")
        return pd.Series(np.nan, index=gipi_df.index, name="gipi")

    sub = gipi_df[valid_cols].copy()
    # Use only rows where >= half the columns have data
    sub = sub.dropna(thresh=max(2, len(valid_cols) // 2))

    scaler = StandardScaler()
    X = scaler.fit_transform(sub.fillna(sub.mean()))

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X).ravel()
    loadings = pca.components_[0]

    # Sign convention: ensure positive loading on energy/food/supply-chain pressure
    energy_like = [i for i, c in enumerate(valid_cols)
                   if any(k in c for k in ("energy", "food", "gscpi", "import"))]
    if energy_like and loadings[energy_like[0]] < 0:
        pc1 = -pc1
        loadings = -loadings

    gipi = pd.Series(pc1, index=sub.index, name="gipi")
    gipi = gipi.reindex(gipi_df.index)    # restore full index (NaN where no data)

    var_explained = pca.explained_variance_ratio_[0]
    log.info(f"  GIPI: PCA variance explained = {var_explained:.1%}, "
             f"inputs = {valid_cols}, loadings = {dict(zip(valid_cols, loadings.round(3)))}")

    return gipi


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw series into model-ready features:

    Levels → growth rates, spreads, standardised shocks.
    All series converted to stationary transformations.

    Returns
    -------
    pd.DataFrame with engineered features, monthly frequency.
    """
    out = pd.DataFrame(index=df.index)

    # ── GPR shock: standardised monthly change ────────────────────────────────
    if "gpr" in df.columns:
        out["gpr_level"] = df["gpr"]
        out["gpr_yoy"] = df["gpr"].pct_change(12) * 100
        out["gpr_shock"] = (df["gpr"] - df["gpr"].rolling(24).mean()) / df["gpr"].rolling(24).std()
        out["gpr_z"] = (df["gpr"] - df["gpr"].mean()) / df["gpr"].std()

    # ── Oil ───────────────────────────────────────────────────────────────────
    oil_col = "wti" if "wti" in df.columns else "wti_spot"
    if oil_col in df.columns:
        out["oil_price"] = df[oil_col]
        out["oil_return"] = np.log(df[oil_col]).diff() * 100        # MoM log return
        out["oil_yoy"] = df[oil_col].pct_change(12) * 100           # YoY %

    # ── Industrial Production ─────────────────────────────────────────────────
    if "ip" in df.columns:
        out["ip_growth"] = np.log(df["ip"]).diff() * 100            # MoM log return
        out["ip_yoy"] = df["ip"].pct_change(12) * 100               # YoY %

    # ── CPI / Inflation ───────────────────────────────────────────────────────
    if "cpi" in df.columns:
        out["cpi_inflation"] = df["cpi"].pct_change(12) * 100       # YoY CPI inflation
        out["cpi_mom"] = np.log(df["cpi"]).diff() * 100             # MoM
    if "core_cpi" in df.columns:
        out["core_inflation"] = df["core_cpi"].pct_change(12) * 100

    # ── GDP ───────────────────────────────────────────────────────────────────
    if "gdp_growth" in df.columns:
        out["gdp_growth"] = df["gdp_growth"]
    if "gdp_level" in df.columns:
        out["gdp_yoy"] = df["gdp_level"].pct_change(4) * 100        # YoY (quarterly)

    # ── Financial conditions ─────────────────────────────────────────────────
    if "hy_spread" in df.columns:
        out["hy_spread"] = df["hy_spread"]
        out["d_hy_spread"] = df["hy_spread"].diff()
    if "term_spread" in df.columns:
        out["term_spread"] = df["term_spread"]
    if "fedfunds" in df.columns:
        out["fedfunds"] = df["fedfunds"]
        out["d_fedfunds"] = df["fedfunds"].diff()

    # ── VIX ───────────────────────────────────────────────────────────────────
    if "vix" in df.columns:
        out["vix"] = df["vix"]
        out["vix_change"] = df["vix"].diff()
        out["vix_log"] = np.log(df["vix"])

    # ── S&P 500 ───────────────────────────────────────────────────────────────
    if "sp500" in df.columns:
        out["sp500_return"] = np.log(df["sp500"]).diff() * 100
        out["sp500_yoy"] = df["sp500"].pct_change(12) * 100

    # ── Unemployment ─────────────────────────────────────────────────────────
    if "unemp" in df.columns:
        out["unemp"] = df["unemp"]
        out["d_unemp"] = df["unemp"].diff()

    # ── GSCPI ─────────────────────────────────────────────────────────────────
    if "gscpi" in df.columns:
        out["gscpi"] = df["gscpi"]

    # ── NEW: Inflation Transmission Channels ──────────────────────────────────

    # 1. Breakeven inflation (market expectations channel)
    if "breakeven_5y" in df.columns:
        out["breakeven_5y"] = df["breakeven_5y"]
        out["d_breakeven_5y"] = df["breakeven_5y"].diff()
    if "breakeven_10y" in df.columns:
        out["breakeven_10y"] = df["breakeven_10y"]

    # 2. Import prices (cost-push / supply chain channel)
    if "import_price_all" in df.columns:
        out["import_price_yoy"]      = df["import_price_all"].pct_change(12) * 100
        out["import_price_mom"]      = df["import_price_all"].pct_change(1) * 100
    if "import_price_xfuel" in df.columns:
        out["import_xfuel_yoy"]      = df["import_price_xfuel"].pct_change(12) * 100

    # 3. Broader energy prices (gas + coal + oil, not just WTI)
    if "global_energy_idx" in df.columns:
        out["global_energy_yoy"]     = df["global_energy_idx"].pct_change(12) * 100
    if "henry_hub" in df.columns:
        out["natgas_return"]         = np.log(df["henry_hub"]).diff() * 100

    # 4. Global food prices (Black Sea / MENA food channel)
    if "global_food_idx" in df.columns:
        out["global_food_yoy"]       = df["global_food_idx"].pct_change(12) * 100
    if "fao_food" in df.columns:
        out["fao_food_yoy"]          = df["fao_food"].pct_change(12) * 100
    if "fao_cereals" in df.columns:
        out["fao_cereals_yoy"]       = df["fao_cereals"].pct_change(12) * 100

    # 5. USD index (pass-through amplifier — stronger dollar → lower import costs)
    if "usd_index" in df.columns:
        out["usd_return"]            = np.log(df["usd_index"]).diff() * 100
        out["usd_yoy"]               = df["usd_index"].pct_change(12) * 100

    # 6. Arab Light–WTI spread (geopolitical oil premium)
    if "arab_light" in df.columns and ("wti" in df.columns or "wti_spot" in df.columns):
        wti_col = "wti" if "wti" in df.columns else "wti_spot"
        out["arab_wti_spread"]       = df["arab_light"] - df[wti_col]
        # Standardise the spread for use as a regressor
        _spread = out["arab_wti_spread"].dropna()
        out["arab_wti_spread_z"]     = (
            (out["arab_wti_spread"] - _spread.rolling(24).mean()) /
            (_spread.rolling(24).std() + 1e-9)
        )

    # 7. Energy CPI & PPI components (inflation pass-through measurement)
    if "cpi_energy" in df.columns:
        out["cpi_energy_yoy"]        = df["cpi_energy"].pct_change(12) * 100
    if "ppi_energy" in df.columns:
        out["ppi_energy_yoy"]        = df["ppi_energy"].pct_change(12) * 100

    # 8. SPR stocks change (supply buffer — drawdown = geopolitical response)
    if "spr_stocks" in df.columns:
        out["spr_change"]  = df["spr_stocks"].diff()    # thousand barrels/week
        out["spr_stocks"]  = df["spr_stocks"]           # level — needed by IV module

    # 9. OPEC spare capacity proxy (IV instrument for GIPI)
    #    Spare cap = rolling 36-month max production − current production
    #    High spare cap → OPEC can buffer a GPR shock → GIPI stays low
    #    Predetermined at 6-month lag, exogenous to US IP demand shocks
    if "opec_production" in df.columns:
        prod = df["opec_production"].resample("MS").mean()  # ensure monthly
        rolling_max = prod.rolling(36, min_periods=12).max()
        spare_cap = rolling_max - prod
        spare_cap.name = "opec_spare_cap"
        out["opec_spare_cap"] = spare_cap
        log.info(f"  OPEC spare cap constructed: "
                 f"{spare_cap.notna().sum()} obs, "
                 f"mean={spare_cap.mean():.2f} mb/d")

    # ── GIPI: Geopolitical Inflation Pressure Index (PCA composite) ───────────
    # Canonical 5-series specification (Section 3.2 of paper).
    # Do NOT add fallback series — they change the PCA loadings and shift the
    # index value, breaking comparability with paper results (GIPI=-0.32 Mar 2026).
    _GIPI_CANONICAL = [
        "gscpi",              # NY Fed Global Supply Chain Pressure Index
        "import_price_yoy",   # Import Price Index YoY
        "global_energy_yoy",  # IMF Global Energy Price YoY
        "global_food_yoy",    # IMF Global Food Price YoY
        "d_breakeven_5y",     # Delta 5Y TIPS Breakeven
    ]
    gipi_inputs = {}
    for col in _GIPI_CANONICAL:
        if col in out.columns and out[col].notna().sum() >= 12:
            gipi_inputs[col] = out[col]

    if gipi_inputs:
        gipi_df = pd.DataFrame(gipi_inputs)
        out["gipi"] = _build_gipi(gipi_df)
        log.info(f"  GIPI built from: {list(gipi_inputs.keys())}")
    else:
        log.warning("  GIPI: no valid inputs found — column will be NaN. "
                    "Delete data/cache/ and re-run to force fresh fetch.")
        out["gipi"] = np.nan

    # ── Consumer sentiment ────────────────────────────────────────────────────
    if "umcsent" in df.columns:
        out["umcsent"] = df["umcsent"]
        out["umcsent_change"] = df["umcsent"].diff()

    # ── Regime classification from GPR ───────────────────────────────────────
    if "gpr_z" in out.columns:
        # Use integer codes (0=calm,1=elevated,2=crisis) — Categorical breaks StandardScaler
        out["regime"] = pd.cut(
            out["gpr_z"],
            bins=[-np.inf, 1.5, 2.5, np.inf],
            labels=[0, 1, 2],
        ).astype(float)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class DataPipeline:
    """
    Orchestrates all data fetching, merging, and feature engineering.

    Methods
    -------
    build(use_cache=False)
        Full rebuild from APIs. Pass use_cache=True to skip download if fresh.
    load_cached()
        Load the last saved Parquet without refetching.
    get_model_df(variables)
        Return subset of columns with no NaN rows (list-wise deletion).
    """

    CACHE_FILE = Path("data/cache/master_dataset.parquet")
    RAW_FILE   = Path("data/cache/raw_merged.parquet")
    # Bump this when adding new columns — forces cache rebuild automatically
    CACHE_VERSION = "v2.1"
    CACHE_VERSION_FILE = Path("data/cache/.cache_version")

    def __init__(self):
        self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self.raw_: pd.DataFrame | None = None
        self.features_: pd.DataFrame | None = None

    def _cache_version_ok(self) -> bool:
        """Return True only if cached data matches current code version."""
        if not self.CACHE_VERSION_FILE.exists():
            return False
        return self.CACHE_VERSION_FILE.read_text().strip() == self.CACHE_VERSION

    def _write_cache_version(self):
        self.CACHE_VERSION_FILE.write_text(self.CACHE_VERSION)

    # ─────────────────────────────────────────────────────────────────────────
    def build(self, use_cache: bool = False) -> pd.DataFrame:
        """
        Fetch all data, merge, engineer features, cache to disk.
        Automatically ignores stale cache if version has changed or
        required columns are missing.
        """
        REQUIRED_COLS = ["gpr_level", "gpr_z", "ip_yoy", "cpi_inflation",
                         "vix_log", "hy_spread", "gscpi"]

        if use_cache and self._cache_is_fresh() and self._cache_version_ok():
            log.info("Cache is fresh — loading from disk.")
            cached = self.load_cached()
            # Verify required columns exist in cache
            missing = [c for c in REQUIRED_COLS if c not in cached.columns]
            if not missing:
                return cached
            log.warning(f"Cache missing columns {missing} — rebuilding.")

        log.info("=" * 60)
        log.info("BUILDING DATA PIPELINE (v2 — Enhanced Inflation Channels)")
        log.info("=" * 60)

        # 1. Fetch raw data
        gpr    = fetch_gpr_index().to_frame("gpr")
        fred   = fetch_fred_series()          # now includes inflation channels
        yahoo  = fetch_yahoo_data()
        gscpi  = fetch_gscpi().to_frame("gscpi")
        alight = fetch_arab_light_eia().to_frame("arab_light")
        fao    = fetch_fao_food_index()       # fao_food, fao_cereals, fao_oils

        # 2. Merge to common monthly index
        frames = [gpr, fred, yahoo, gscpi, alight, fao]
        raw = frames[0]
        for f in frames[1:]:
            raw = raw.join(f, how="outer")

        raw = raw.sort_index()
        raw = raw[raw.index >= cfg.START_DATE]
        raw = raw[raw.index <= pd.Timestamp(_end_date())]

        self.raw_ = raw
        raw.to_parquet(self.RAW_FILE)
        log.info(f"\nRaw merged: {raw.shape[0]} rows × {raw.shape[1]} cols")

        # 3. Engineer features
        features = engineer_features(raw)
        features = features.sort_index()

        self.features_ = features
        features.to_parquet(self.CACHE_FILE)
        self._write_cache_version()

        log.info(f"Features: {features.shape[0]} rows × {features.shape[1]} cols")
        log.info(f"Date range: {features.index[0].date()} → {features.index[-1].date()}")
        log.info(f"Cached to: {self.CACHE_FILE}")

        return features

    # ─────────────────────────────────────────────────────────────────────────
    def load_cached(self) -> pd.DataFrame:
        if not self.CACHE_FILE.exists():
            raise FileNotFoundError("No cache found. Run DataPipeline().build() first.")
        df = pd.read_parquet(self.CACHE_FILE)
        self.features_ = df
        return df

    # ─────────────────────────────────────────────────────────────────────────
    def get_model_df(
        self,
        variables: list[str] | None = None,
        dropna: bool = True,
    ) -> pd.DataFrame:
        """Return a clean subset of feature columns, optionally dropping NaN rows."""
        if self.features_ is None:
            self.load_cached()
        df = self.features_
        if variables:
            available = [v for v in variables if v in df.columns]
            missing = [v for v in variables if v not in df.columns]
            if missing:
                log.warning(f"Columns not found: {missing}")
            df = df[available]
        if dropna:
            df = df.dropna()
        return df

    # ─────────────────────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        """Return descriptive statistics of the feature dataset."""
        if self.features_ is None:
            self.load_cached()
        numeric = self.features_.select_dtypes(include=[np.number])
        return numeric.describe().T.round(3)

    # ─────────────────────────────────────────────────────────────────────────
    def _cache_is_fresh(self, max_age_hours: int = 24) -> bool:
        if not self.CACHE_FILE.exists():
            return False
        age = datetime.now().timestamp() - self.CACHE_FILE.stat().st_mtime
        return age < max_age_hours * 3600


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(use_cache: bool = False) -> pd.DataFrame:
    """One-liner: fetch, merge, engineer, return DataFrame."""
    dp = DataPipeline()
    return dp.build(use_cache=use_cache)


if __name__ == "__main__":
    df = build_dataset(use_cache=False)
    print("\nDataset preview:")
    print(df.tail(3).to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")