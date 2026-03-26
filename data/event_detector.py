"""
data/event_detector.py - Layer 0: Real-Time Geopolitical Event Detection
-----------------------------------------------------------------------------
Three signal sources, all free / low-cost:

SOURCE 1 - GDELT 2.0 Doc API
  Queries the last N hours of global news for Middle East conflict events.
  No API key. Returns article titles, dates, tones, source countries.

SOURCE 2 - CAMEO Coding via LLM (Anthropic / Claude)
  Sends article headlines to Claude which classifies events into CAMEO
  (Conflict and Mediation Event Observations) codes and estimates a
  severity score 0-10. Falls back to rule-based regex if no key.

SOURCE 3 - AIS Tanker Equity Proxy (Strait of Hormuz signal)
  Uses free Yahoo Finance data on listed VLCC tanker operators (INSW, TK,
  TRMD, FRO) plus the Brent-WTI spread as a real-time proxy for shipping
  disruption risk in the Strait of Hormuz.
  Rationale:
    * When Hormuz is threatened, tanker stocks fall sharply on disruption risk
    * Brent-WTI spread widens because Brent is more exposed to ME supply
    * Simultaneous z > 1.5 in both -> anomaly flag
  Note: Real AIS data (MarineTraffic, Kpler, VesselsValue) would be more
  precise for production. This proxy is fully free.

Output: EventSignal dataclass with severity_score, regime, cameo_codes,
        ais_anomaly, gpr_nowcast, headlines, llm_summary.

Usage
-----
  from data.event_detector import EventDetector
  ed = EventDetector()
  sig = ed.detect(lookback_hours=48, use_llm=True)
  print(sig.regime, sig.severity_score, sig.cameo_codes)

  # CLI:
  python data/event_detector.py 48
-----------------------------------------------------------------------------
"""

from __future__ import annotations

import io
import os
import re
import json
import time
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False


# ===============================================================================
#  CAMEO reference - conflict subset (codes 13-20)
# ===============================================================================

CAMEO_SEVERITY: dict[str, float] = {
    "13": 2.0,   # Threaten
    "14": 1.5,   # Protest
    "15": 3.5,   # Military posture
    "17": 3.5,   # Coerce
    "18": 5.5,   # Assault
    "19": 6.5,   # Fight
    "190": 7.0,  # Conventional military force
    "191": 6.5,  # Blockade / restrict movement
    "192": 6.5,  # Occupy territory
    "193": 6.5,  # Fight - small arms
    "194": 7.5,  # Fight - artillery / tanks
    "195": 8.0,  # Aerial weapons
    "196": 5.0,  # Ceasefire violation
    "20": 9.0,   # Mass violence
    "200": 10.0, # WMD use
    "201": 8.5,  # Mass expulsion
    "202": 9.0,  # Mass killings
    "203": 9.5,  # Ethnic cleansing
    "204": 9.0,  # Assassination
    "2041": 9.5, # Successful assassination
}

# Rule-based keyword -> (CAMEO code, severity)
_KW: list[tuple[str, str, float]] = [
    (r"\b(strike|struck|attack|attacked|bomb|bombing|missile|rocket|drone strike)\b", "195", 8.0),
    (r"\b(explosion|blast|detonation|detonated)\b",                                  "193", 7.0),
    (r"\b(assassinat|killed|shot|executed)\b",                                       "204", 8.5),
    (r"\b(invasion|invade|ground troops|armoured|tanks)\b",                          "192", 7.5),
    (r"\b(blockade|naval blockade|strait|chokepoint)\b",                             "191", 7.0),
    (r"\b(nuclear|radiolog|chemical weapon|dirty bomb|sarin|enrichment)\b",          "200", 10.0),
    (r"\b(ceasefire|truce|de-escalat|withdrawal|withdraw)\b",                        "16", -2.0),  # de-esc
    (r"\b(sanction|embargo|restrict|ban)\b",                                          "17",  3.5),
    (r"\b(threat|warn|ultimatum|ultimatums)\b",                                       "13",  2.5),
    (r"\b(protest|demonstration|riot)\b",                                             "14",  1.5),
    (r"\b(military buildup|deploy|reinforc|warship|carrier group)\b",                "15",  3.5),
    (r"\b(airstrike|air strike|bombing run)\b",                                      "195", 8.5),
    (r"\b(hostage|kidnap|abduct)\b",                                                  "18",  5.5),
]

ME_LOCATIONS: set[str] = {
    "iran", "israel", "gaza", "palestine", "west bank",
    "lebanon", "hezbollah", "hamas", "islamic jihad",
    "houthi", "houthis", "yemen", "ansar allah",
    "strait of hormuz", "hormuz", "persian gulf",
    "red sea", "bab al-mandeb", "suez", "suez canal",
    "saudi arabia", "aramco", "abqaiq", "riyadh",
    "iraq", "baghdad", "syria", "damascus",
    "jordan", "tehran", "tel aviv", "jerusalem",
    "oman", "bahrain", "qatar", "uae", "dubai",
    "irgc", "revolutionary guard", "mossad",
    "hezbullah", "hizbollah",
}


# ===============================================================================
#  DATA CLASSES
# ===============================================================================

@dataclass
class AISSignal:
    tanker_z:     float
    brent_wti_spread: float
    brent_wti_z:  float
    freight_z:    float          # BWET TD3C VLCC freight proxy z-score
    anomaly:      bool
    tickers_used: list[str]
    timestamp:    datetime

    def to_dict(self) -> dict:
        return {
            "tanker_z":         round(self.tanker_z, 3),
            "brent_wti_spread": round(self.brent_wti_spread, 2),
            "brent_wti_z":      round(self.brent_wti_z, 3),
            "freight_z":        round(self.freight_z, 3),
            "anomaly":          self.anomaly,
            "tickers_used":     self.tickers_used,
        }


@dataclass
class EventSignal:
    timestamp:        datetime
    severity_score:   float          # 0-10 composite
    cameo_codes:      list[str]
    ais_anomaly:      bool
    gpr_nowcast:      float          # estimated GPR z-score
    top_headlines:    list[str]
    article_count:    int
    location_hit_rate: float
    raw_articles:     list[dict]
    ais_signal:       Optional[AISSignal]
    llm_used:         bool
    llm_summary:      Optional[str]

    @property
    def regime(self) -> str:
        if self.gpr_nowcast >= 2.5:
            return "CRISIS"
        elif self.gpr_nowcast >= 1.5:
            return "ELEVATED"
        return "CALM"

    @property
    def icon(self) -> str:
        return {"CRISIS": "?", "ELEVATED": "?", "CALM": "?"}[self.regime]

    def to_dict(self) -> dict:
        return {
            "timestamp":        self.timestamp.isoformat(),
            "severity_score":   round(self.severity_score, 2),
            "gpr_nowcast":      round(self.gpr_nowcast, 2),
            "regime":           self.regime,
            "cameo_codes":      self.cameo_codes,
            "ais_anomaly":      self.ais_anomaly,
            "ais_signal":       self.ais_signal.to_dict() if self.ais_signal else None,
            "article_count":    self.article_count,
            "location_hit_rate": round(self.location_hit_rate, 3),
            "top_headlines":    self.top_headlines[:5],
            "llm_used":         self.llm_used,
            "llm_summary":      self.llm_summary,
        }


# ===============================================================================
#  GDELT FETCHER
# ===============================================================================

_GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"

_GDELT_QUERY = (
    "(Iran OR Israel OR Hamas OR Houthi OR Yemen OR Hormuz OR Syria OR Iraq) "
    "AND (military OR strike OR attack OR missile OR war OR nuclear OR blockade)"
)


def _gdelt_timespan(hours: int) -> str:
    return f"{hours}h" if hours <= 24 else f"{hours // 24}d"


# -- SSL fix for Mac (expired Python certs) -----------------------------------
import ssl
import urllib.request

def _get_ssl_context():
    """Return an unverified SSL context as fallback for Mac cert issues."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def _safe_get(url: str, params: dict | None = None, timeout: int = 20) -> requests.Response:
    """
    requests.get with automatic SSL fallback.
    Tries verified first, then unverified (Mac expired cert workaround).
    """
    try:
        return requests.get(url, params=params, timeout=timeout)
    except requests.exceptions.SSLError:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        return requests.get(url, params=params, timeout=timeout, verify=False)


def fetch_gdelt_articles(lookback_hours: int = 48, max_records: int = 50) -> list[dict]:
    log.info(f"GDELT: querying last {lookback_hours}h ...")
    try:
        r = _safe_get(
            _GDELT_API,
            params={
                "query":      _GDELT_QUERY,
                "mode":       "artlist",
                "maxrecords": max_records,
                "timespan":   _gdelt_timespan(lookback_hours),
                "format":     "json",
                "sort":       "datedesc",
            },
            timeout=20,
        )
        r.raise_for_status()
        articles = r.json().get("articles", [])
        log.info(f"  -> {len(articles)} articles returned")
        return articles
    except Exception as e:
        log.warning(f"  GDELT failed ({e}) - using synthetic fallback")
        return _synthetic_articles()


def _synthetic_articles() -> list[dict]:
    now = datetime.now(timezone.utc)
    fmt = "%Y%m%dT%H%M%SZ"
    return [
        {"title": "[DEMO] Iran military exercise near Strait of Hormuz",
         "url": "https://demo/1", "domain": "demo.news",
         "seendate": (now - timedelta(hours=3)).strftime(fmt), "tone": "-3.5"},
        {"title": "[DEMO] US carrier group repositioned to Persian Gulf",
         "url": "https://demo/2", "domain": "demo.news",
         "seendate": (now - timedelta(hours=8)).strftime(fmt), "tone": "-5.1"},
        {"title": "[DEMO] Israel conducts airstrikes in Lebanon",
         "url": "https://demo/3", "domain": "demo.news",
         "seendate": (now - timedelta(hours=12)).strftime(fmt), "tone": "-6.2"},
    ]


# ===============================================================================
#  RULE-BASED CAMEO CODER (fallback)
# ===============================================================================

def _is_me_location(text: str) -> bool:
    t = text.lower()
    return any(loc in t for loc in ME_LOCATIONS)


def rule_based_cameo(title: str) -> tuple[list[str], float]:
    """Regex-based CAMEO coding. Returns (codes, severity)."""
    text = title.lower()
    codes: list[str] = []
    severity = 0.0

    for pattern, code, sev in _KW:
        if re.search(pattern, text, re.I):
            codes.append(code)
            severity = severity + sev if sev < 0 else max(severity, sev)

    severity = max(0.0, min(10.0, severity))
    if _is_me_location(text):
        severity = min(10.0, severity * 1.25)

    return (list(set(codes)) or ["--"]), round(severity, 2)


# ===============================================================================
#  LLM CAMEO CODER - Anthropic Claude
# ===============================================================================

_LLM_SYSTEM = """You are a geopolitical risk analyst specialising in CAMEO event coding.

Given news headlines, identify Middle East conflict events and return JSON only:
{
  "overall_severity":     <float 0-10>,
  "dominant_cameo_codes": [<CAMEO codes as strings, e.g. "195", "191">],
  "gpr_z_estimate":       <float: estimated GPR z-score, 0=normal, 3=crisis>,
  "key_events":           [<brief event descriptions>],
  "summary":              "<one sentence>",
  "de_escalation_signals": <bool>
}

CAMEO severity guide:
  0   = no conflict relevance
  3   = diplomatic tensions / military posturing
  5   = low-level clashes or strikes
  7   = significant strikes, naval blockades, infrastructure attacks
  9   = major war escalation, assassination of senior leader
  10  = WMD use, nuclear facility attack, oil-field destruction

Relevant CAMEO codes for this task:
  13=Threaten, 15=Military posture, 191=Blockade, 192=Occupy,
  193=Small arms, 194=Artillery/tanks, 195=Aerial weapons,
  200=WMD, 204=Assassination

Return ONLY the JSON object. No markdown fences."""
_LLM_SYSTEM = _LLM_SYSTEM.encode("ascii", "ignore").decode("ascii")


def _clean_text(text: str) -> str:
    """Normalise Unicode and strip non-ASCII. Never raises."""
    import unicodedata
    _MAP = {
        "\u2018": "'",  "\u2019": "'",  # curly single quotes
        "\u201c": '"',  "\u201d": '"',  # curly double quotes
        "\u2013": "-",   "\u2014": "--",  # en/em dash
        "\u2026": "...", "\u00b7": ".",   # ellipsis, middle dot
        "\u2032": "'",  "\u00b4": "'",  # prime, acute
    }
    for uni, asc in _MAP.items():
        text = text.replace(uni, asc)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", errors="ignore").decode("ascii")


def llm_cameo_code(
    headlines: list[str],
    api_key: Optional[str] = None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """LLM CAMEO coding via isolated subprocess (Unicode-safe)."""
    if not headlines:
        return _rule_based_batch(headlines)
    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return _rule_based_batch(headlines)
    try:
        import subprocess, sys, unicodedata
        def _a(s):
            return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
        clean = [_a(h) for h in headlines[:20]]
        inp = json.dumps({
            "key":    key,
            "model":  model,
            "system": _a(_LLM_SYSTEM),
            "text":   _a("\n".join("- " + h for h in clean)),
        }, ensure_ascii=True).encode("ascii")
        helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_llm_cameo_helper.py")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, helper],
            input=inp, capture_output=True, timeout=40, env=env,
        )
        out = proc.stdout.decode("utf-8", errors="replace").strip()
        if not out:
            raise ValueError("empty subprocess output")
        result = json.loads(out)
        if result.get("llm_used"):
            log.info("  LLM CAMEO: severity=%s gpr_z=%s",
                     result.get("overall_severity"), result.get("gpr_z_estimate"))
            return result
        log.warning("  LLM subprocess: %s - rule-based fallback", result.get("error", "?"))
        return _rule_based_batch(headlines)
    except Exception as e:
        emsg = "".join(c if ord(c) < 128 else "?" for c in str(e))
        log.warning("  LLM error (%s) - rule-based fallback", emsg)
        return _rule_based_batch(headlines)

def _rule_based_batch(headlines: list[str]) -> dict:
    if not headlines:
        return {"overall_severity": 0.0, "dominant_cameo_codes": [],
                "gpr_z_estimate": 0.0, "key_events": [],
                "summary": "No headlines.", "de_escalation_signals": False,
                "llm_used": False}

    all_codes: list[str] = []
    sevs: list[float] = []
    for h in headlines:
        c, s = rule_based_cameo(h)
        all_codes.extend(c)
        sevs.append(s)

    severity = float(np.percentile(sevs, 75)) if sevs else 0.0
    codes = list({c for c in all_codes if c != "--"})[:5]
    return {
        "overall_severity":     round(severity, 2),
        "dominant_cameo_codes": codes,
        "gpr_z_estimate":       round(min(3.0, severity / 3.33), 2),
        "key_events":           headlines[:3],
        "summary":              "Rule-based CAMEO (LLM unavailable).",
        "de_escalation_signals": False,
        "llm_used":             False,
    }


# ===============================================================================
#  AIS TANKER PROXY - Strait of Hormuz Disruption Signal
# ===============================================================================
#
#  Economic rationale:
#    * ~20% of global oil transits the Strait of Hormuz via VLCC tankers.
#    * Disruption risk -> tanker equity selloff (insurance + routing uncertainty)
#    * Brent-WTI spread widens because Brent reflects ME supply premium.
#    * Both signals firing simultaneously = rare, high-conviction anomaly.
#
#  Limitation: This is a financial market PROXY for AIS vessel tracking.
#  For production use, integrate real AIS APIs: MarineTraffic, VesselsValue,
#  Kpler, or Spire Maritime. The proxy works as a free, real-time alternative.
#
#  Tickers: INSW (International Seaways), TK (Teekay), TRMD (TORM),
#           FRO (Frontline) - all own VLCC fleets with Hormuz exposure.
# ===============================================================================

_TANKER_TICKERS = ["INSW", "TK", "TRMD", "FRO"]
_FREIGHT_TICKER = "BWET"   # Breakwave Tanker ETF — 90% TD3C VLCC futures (MEG→Asia)
_BRENT_TICKER   = "BZ=F"
_WTI_TICKER     = "CL=F"


def _rolling_z(series: pd.Series, window: int = 90) -> pd.Series:
    mu  = series.rolling(window, min_periods=10).mean()
    sig = series.rolling(window, min_periods=10).std()
    return ((series - mu) / sig.clip(lower=1e-6)).fillna(0.0)


def fetch_ais_proxy(
    lookback_days: int = 90,
    z_threshold:   float = 1.5,
    tickers:       list[str] | None = None,
) -> AISSignal:
    """
    Compute AIS proxy signal from tanker equities + Brent-WTI spread.

    Anomaly = True when EITHER:
      1. abs(tanker_basket_z) > threshold AND brent_wti_z > threshold
      2. tanker_z > threshold (surging = disruption) AND brent_wti_z > 0.5
    Uses 90-day rolling window to avoid normalizing sustained disruptions.
    """
    now = datetime.now(timezone.utc)
    if not YF_AVAILABLE:
        log.warning("yfinance unavailable - AIS proxy defaulting to no-anomaly")
        return AISSignal(0.0, 0.0, 0.0, 0.0, False, [], now)

    tickers = tickers or _TANKER_TICKERS
    start   = (now - timedelta(days=lookback_days + 5)).strftime("%Y-%m-%d")

    # -- Tanker basket ---------------------------------------------------------
    basket_rets: list[pd.Series] = []
    used: list[str] = []

    for t in tickers:
        try:
            raw = yf.download(t, start=start, progress=False, auto_adjust=True)
            if raw.empty:
                continue
            close = (raw["Close"].iloc[:, 0]
                     if isinstance(raw.columns, pd.MultiIndex)
                     else raw["Close"])
            ret = close.pct_change().dropna()
            basket_rets.append(ret.rename(t))
            used.append(t)
        except Exception as e:
            log.debug(f"  Tanker {t}: {e}")

    tanker_z_val = 0.0
    if basket_rets:
        basket = pd.concat(basket_rets, axis=1).mean(axis=1)
        tanker_z_val = float(_rolling_z(basket).iloc[-1])

    # -- BWET freight proxy (TD3C VLCC futures) --------------------------------
    freight_z_val = 0.0
    try:
        fw = yf.download(_FREIGHT_TICKER, start=start, progress=False, auto_adjust=True)
        if not fw.empty:
            fc = (fw["Close"].iloc[:, 0] if isinstance(fw.columns, pd.MultiIndex)
                  else fw["Close"])
            freight_ret = fc.pct_change().dropna()
            if len(freight_ret) > 10:
                freight_z_val = float(_rolling_z(freight_ret).iloc[-1])
                used.append(_FREIGHT_TICKER)
    except Exception as e:
        log.debug(f"  BWET freight: {e}")

    # -- Brent-WTI spread ------------------------------------------------------
    spread_val   = 0.0
    bwz_val      = 0.0

    try:
        br = yf.download(_BRENT_TICKER, start=start, progress=False, auto_adjust=True)
        wt = yf.download(_WTI_TICKER,   start=start, progress=False, auto_adjust=True)
        if not br.empty and not wt.empty:
            bc = (br["Close"].iloc[:,0] if isinstance(br.columns, pd.MultiIndex)
                  else br["Close"])
            wc = (wt["Close"].iloc[:,0] if isinstance(wt.columns, pd.MultiIndex)
                  else wt["Close"])
            spread = (bc - wc).dropna()
            spread_val = float(spread.iloc[-1]) if not spread.empty else 0.0
            bwz_val    = float(_rolling_z(spread).iloc[-1]) if not spread.empty else 0.0
    except Exception as e:
        log.debug(f"  Brent-WTI: {e}")

    # Anomaly triggers if ANY of:
    #   1. Tanker basket + Brent-WTI spread both extreme
    #   2. Tanker stocks surging (disruption profit) + any spread elevation
    #   3. Freight rates spiking (BWET z > threshold) — most direct signal
    anomaly = (
        (abs(tanker_z_val) > z_threshold and bwz_val > z_threshold) or
        (tanker_z_val > z_threshold and bwz_val > 0.5) or
        (freight_z_val > z_threshold)
    )

    log.info(f"  AIS proxy: tanker_z={tanker_z_val:+.2f}  freight_z={freight_z_val:+.2f}  "
             f"brent_wti_spread=${spread_val:.1f}  brent_wti_z={bwz_val:+.2f}  "
             f"anomaly={anomaly}")

    return AISSignal(
        tanker_z=round(tanker_z_val, 3),
        brent_wti_spread=round(spread_val, 2),
        brent_wti_z=round(bwz_val, 3),
        freight_z=round(freight_z_val, 3),
        anomaly=anomaly,
        tickers_used=used,
        timestamp=now,
    )


# ===============================================================================
#  COMPOSITE SEVERITY SCORE
# ===============================================================================

def _composite_severity(
    articles:   list[dict],
    llm_result: dict,
    ais:        Optional[AISSignal],
) -> float:
    """
    Weighted composite:
      50% LLM/rule-based article severity
      30% 75th-percentile of per-article rule scores
      20% AIS proxy severity
    """
    llm_sev = float(llm_result.get("overall_severity", 0.0))

    per_article = [rule_based_cameo(a.get("title",""))[1] for a in articles]
    art_sev = float(np.percentile(per_article, 75)) if per_article else 0.0

    ais_sev = 0.0
    if ais:
        ais_sev = min(10.0, (abs(ais.tanker_z) + max(0, ais.brent_wti_z)) * 2.0)

    return round(min(10.0, max(0.0, 0.50*llm_sev + 0.30*art_sev + 0.20*ais_sev)), 2)


# ===============================================================================
#  MAIN EVENT DETECTOR
# ===============================================================================

class EventDetector:
    """
    Layer 0 real-time event detection pipeline.

    Parameters
    ----------
    anthropic_key : str, optional
        Anthropic API key. If None, reads ANTHROPIC_API_KEY env var.
        Falls back to rule-based CAMEO if unavailable.
    """

    def __init__(self, anthropic_key: Optional[str] = None):
        self.key = anthropic_key or os.getenv("ANTHROPIC_API_KEY", "")

    def detect(
        self,
        lookback_hours:  int  = 48,
        use_llm:         bool = True,
        use_ais:         bool = True,
        max_articles:    int  = 50,
        ais_tickers:     list[str] | None = None,
        ais_z_threshold: float = 1.5,
    ) -> EventSignal:
        """
        Run full Layer 0 detection.

        Parameters
        ----------
        lookback_hours  : Hours of news to scan
        use_llm         : Use Claude for CAMEO coding
        use_ais         : Compute AIS tanker proxy signal
        max_articles    : Max GDELT articles to fetch
        ais_tickers     : VLCC operator tickers (default: INSW, TK, TRMD, FRO)
        ais_z_threshold : Sigma threshold for AIS anomaly flag

        Returns
        -------
        EventSignal
        """
        log.info(f"\n{'='*55}")
        log.info(f"  LAYER 0 EVENT DETECTION  ({lookback_hours}h lookback)")
        log.info(f"{'='*55}")
        ts = datetime.now(timezone.utc)

        # 1. Fetch GDELT news
        raw = fetch_gdelt_articles(lookback_hours, max_articles)

        # 2. Score location relevance
        me_hits  = [a for a in raw if _is_me_location(a.get("title",""))]
        all_h    = [a.get("title","") for a in raw if a.get("title")]
        me_h     = [a.get("title","") for a in me_hits if a.get("title")]
        hit_rate = len(me_hits) / len(raw) if raw else 0.0

        # 3. CAMEO coding
        target_headlines = me_h if me_h else all_h
        if use_llm and self.key:
            llm_res = llm_cameo_code(target_headlines, api_key=self.key)
        else:
            llm_res = _rule_based_batch(target_headlines)

        # 4. AIS proxy
        ais = (fetch_ais_proxy(
                   z_threshold=ais_z_threshold,
                   tickers=ais_tickers)
               if use_ais else None)

        # 5. Composite severity + GPR nowcast
        severity = _composite_severity(raw, llm_res, ais)

        # gpr_z: prefer LLM estimate; fall back to CAMEO-based score
        if llm_res.get("gpr_z_estimate") is not None:
            gpr_z = float(llm_res["gpr_z_estimate"])
        else:
            # Rule-based: derive z from max CAMEO severity in detected codes
            detected_codes = llm_res.get("dominant_cameo_codes", [])
            if detected_codes:
                max_sev = max(
                    CAMEO_SEVERITY.get(str(c), 0.0) for c in detected_codes
                )
                # map 0-10 CAMEO severity → GPR z-score (calibrated: sev=8 ≈ z=2.5)
                gpr_z = round(max_sev * 0.35, 2)
            else:
                gpr_z = round(severity / 3.33, 2)

        if ais and ais.anomaly:
            gpr_z = min(4.0, gpr_z + 0.5)
            log.info("  ?  AIS anomaly -> GPR z boosted +0.5")

        sig = EventSignal(
            timestamp=ts,
            severity_score=severity,
            cameo_codes=llm_res.get("dominant_cameo_codes", []),
            ais_anomaly=bool(ais and ais.anomaly),
            gpr_nowcast=round(gpr_z, 2),
            top_headlines=all_h[:10],
            article_count=len(raw),
            location_hit_rate=round(hit_rate, 3),
            raw_articles=raw[:20],
            ais_signal=ais,
            llm_used=bool(llm_res.get("llm_used")),
            llm_summary=llm_res.get("summary"),
        )

        log.info(f"  {sig.icon} REGIME={sig.regime}  "
                 f"severity={sig.severity_score:.1f}  "
                 f"gpr_z={sig.gpr_nowcast:.2f}  "
                 f"ais_anomaly={sig.ais_anomaly}")
        return sig


# ===============================================================================
#  HISTORICAL EVENT FLAG DATAFRAME (for model validation)
# ===============================================================================

HISTORICAL_EPISODES: dict[str, tuple] = {
    "Gulf War I":          ("1990-08", "1991-03", "190", 9.0),
    "9/11 Attacks":        ("2001-09", "2001-10", "200", 10.0),
    "Iraq War":            ("2003-03", "2003-05", "190", 8.5),
    "Arab Spring Libya":   ("2011-02", "2011-10", "190", 7.5),
    "Abqaiq Attack":       ("2019-09", "2019-10", "195", 8.5),
    "Soleimani Strike":    ("2020-01", "2020-02", "204", 9.0),
    "Russia-Ukraine":      ("2022-02", "2022-06", "190", 9.5),
    "Hamas-Israel":        ("2023-10", "2024-03", "190", 8.0),
    "Israel-Iran 2024":    ("2024-04", "2024-10", "195", 8.5),
    "US-Israel-Iran 2026": ("2026-03", "2026-03", "195", 9.0),
}


def build_episode_flags(gpr: pd.Series | None = None) -> pd.DataFrame:
    """Build monthly episode flag dataframe, optionally joined with GPR."""
    rows = []
    for name, (start, end, code, sev) in HISTORICAL_EPISODES.items():
        for p in pd.period_range(start=start, end=end, freq="M"):
            rows.append({"date": p.to_timestamp(), "episode": name,
                         "cameo": code, "severity": sev})
    df = pd.DataFrame(rows).set_index("date")
    if gpr is not None:
        df = df.join(gpr.rename("gpr_observed"), how="left")
    return df


# ===============================================================================
#  CLI ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    ed  = EventDetector()
    sig = ed.detect(lookback_hours=hours, use_llm=True, use_ais=True)

    print(f"\n{'?'*55}")
    print(f"  {sig.icon} REGIME:     {sig.regime}")
    print(f"  Severity:    {sig.severity_score:.1f} / 10")
    print(f"  GPR z:       {sig.gpr_nowcast:.2f}")
    print(f"  CAMEO codes: {', '.join(sig.cameo_codes) or 'none'}")
    print(f"  AIS anomaly: {sig.ais_anomaly}")
    if sig.ais_signal:
        print(f"  Tanker z:    {sig.ais_signal.tanker_z:+.2f}")
        print(f"  Brent-WTI:   ${sig.ais_signal.brent_wti_spread:.1f}  "
              f"(z={sig.ais_signal.brent_wti_z:+.2f})")
    print(f"  Articles:    {sig.article_count}  "
          f"({sig.location_hit_rate:.0%} ME relevance)")
    if sig.llm_summary:
        print(f"  Summary:     {sig.llm_summary}")
    print("\n  Top headlines:")
    for h in sig.top_headlines[:5]:
        print(f"    * {h[:100]}")
    print(f"{'?'*55}\n")
