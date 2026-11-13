"""
scripts/build_dashboard_data.py — data for the GitHub Pages dashboard (docs/).

Fetches live data (FRED, Yahoo, NY Fed GSCPI, Caldara–Iacoviello GPR), runs
Layer 0, re-estimates the paper's models on the latest vintage and writes

  docs/data/dashboard.json   everything the page displays
  docs/data/signals.json     append-only archive of Layer 0 signals

Run locally:   GEOSHOCK_STRICT=1 python scripts/build_dashboard_data.py
In CI:         .github/workflows/dashboard.yml (daily)
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("dashboard")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import data.pipeline as pl                                   # noqa: E402
from models.local_projections import LocalProjections        # noqa: E402
from models.quantile_risk import GrowthAtRisk                # noqa: E402

DOCS = ROOT / "docs" / "data"
H_GAR = [3, 6, 12]
H_LP = 18
HIST_START = "2010-01-01"


def _r(x, d=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), d)


def _series(s: pd.Series, d=3) -> list:
    return [_r(v, d) for v in s.values]


def gpr_subindices() -> pd.DataFrame:
    r = requests.get(pl.GPR_URL, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    x = pl.read_gpr_xls(r.content)["1985":]
    out = pd.DataFrame(index=x.index)
    for c in ["GPRT", "GPRA"]:
        s = x[c].astype(float).dropna()
        out[c + "_z"] = (s - s.mean()) / s.std()
        out["sh_" + c] = s.diff() / s.diff().std()
    return out


def gaussian_tail(nc: dict, above: float | None = None, below: float | None = None) -> float:
    sig = max((nc[0.90] - nc[0.10]) / (norm.ppf(0.90) - norm.ppf(0.10)), 0.1)
    if above is not None:
        return float(1 - norm.cdf(above, nc[0.50], sig))
    return float(norm.cdf(below, nc[0.50], sig))


def tail_block(df, outcome, l0_z=None):
    out = {}
    specs = [("baseline", None, None), ("enhanced", "gipi", None)]
    if l0_z is not None:
        specs.append(("enhanced_l0", "gipi", {"gpr_z": l0_z}))
    for name, gc, ov in specs:
        g = GrowthAtRisk(df, outcome=outcome, gpr_col="gpr_z", gipi_col=gc,
                         use_interaction=gc is not None)
        res = g.fit(H_GAR, verbose=False, nowcast_overrides=ov)
        out[name] = {}
        for h, r in res.items():
            nc = r.nowcast
            d = {"origin": str(r.origin.date()),
                 "q": {f"{q:.2f}": _r(v) for q, v in nc.items()},
                 "bowley": _r(r.conditional_skewness)}
            if outcome == "cpi_inflation":
                d["p_gt3"] = _r(gaussian_tail(nc, above=3.0))
                d["p_gt4"] = _r(gaussian_tail(nc, above=4.0))
            else:
                d["p_lt0"] = _r(gaussian_tail(nc, below=0.0))
                d["p_ltm2"] = _r(gaussian_tail(nc, below=-2.0))
            out[name][str(h)] = d
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="use the pinned April-2026 raw data instead of live sources (testing)")
    args = ap.parse_args()
    built = datetime.now(timezone.utc)
    pl.cfg.END_DATE = None                       # live: up to today
    if args.offline:
        df = pl.DataPipeline().rebuild_from_raw(ROOT / "data" / "raw_merged_2026-04-07.parquet")
    else:
        df = pl.DataPipeline().build(use_cache=False)
    df = df.loc[:pd.Timestamp.today().normalize()]

    required = ["gpr_z", "gpr_shock", "gipi", "cpi_inflation", "core_inflation", "ip_yoy",
                "vix_log", "hy_spread", "term_spread", "d_fedfunds", "sp500_return", "gscpi",
                "oil_return", "fedfunds", "vix_change"]
    missing = [c for c in required if c not in df or df[c].notna().sum() < 120]
    if missing:
        raise RuntimeError(f"required series missing: {missing}")

    sub = gpr_subindices()
    df = df.join(sub, how="left")

    # ── Layer 0 ───────────────────────────────────────────────────────────────
    from data.event_detector import EventDetector
    import time
    anchor = float(df["gpr_z"].dropna().iloc[-1])
    l0, err = None, None
    for attempt in range(4):                      # GDELT rate-limits (HTTP 429)
        try:
            sig = EventDetector(os.getenv("ANTHROPIC_API_KEY") or None).detect(
                lookback_hours=48, use_llm=bool(os.getenv("ANTHROPIC_API_KEY")),
                gpr_anchor_z=anchor, archive=False)
            l0 = sig.to_dict()
            l0.update(ok=True, stale=False, gpr_anchor_z=_r(anchor),
                      anchor_month=str(df["gpr_z"].dropna().index[-1].date()))
            break
        except Exception as e:
            err = str(e)[:200]
            log.warning(f"Layer 0 attempt {attempt + 1} failed: {err}")
            time.sleep(15 * (attempt + 1))
    if l0 is None:
        prev = DOCS / "dashboard.json"
        old = json.loads(prev.read_text()).get("layer0", {}) if prev.exists() else {}
        l0 = dict(old, ok=bool(old.get("ok")), stale=True, error=err)
    l0_z = l0.get("gpr_nowcast") if l0.get("ok") else None

    # ── Channel state ─────────────────────────────────────────────────────────
    g = df["gipi"].dropna()
    med = float(df["gipi"].shift(1).median())
    run = 0
    for v in g.values[::-1]:
        if (v >= med) == (g.iloc[-1] >= med):
            run += 1
        else:
            break
    state = {"gipi": _r(g.iloc[-1]), "month": str(g.index[-1].date()), "median": _r(med),
             "open": bool(g.iloc[-1] >= med), "months_in_state": run}

    # ── State-dependent CPI response (live re-estimate) ───────────────────────
    st = df["gipi"].shift(1)
    D = (st >= st.median()).where(st.notna())
    lp = {}
    for out in ["cpi_inflation", "core_inflation", "ip_yoy"]:
        o = LocalProjections(df, shock="sh_GPRT", outcome=out, horizon=H_LP).fit_state_dependent(D)
        lp[out] = {k: [_r(v, 4) for v in np.asarray(o[k])] for k in
                   ["b_hi", "se_hi", "b_lo", "se_lo", "diff", "se_diff", "p_diff"]}

    # ── Tail risk ─────────────────────────────────────────────────────────────
    dft = df.loc["1998-02-01":]
    iar = tail_block(dft, "cpi_inflation", l0_z)
    gar = tail_block(dft, "ip_yoy", l0_z)

    # ── History for charts ────────────────────────────────────────────────────
    h = df.loc[HIST_START:]
    cols = {"gprt_z": "GPRT_z", "gpra_z": "GPRA_z", "gipi": "gipi", "cpi": "cpi_inflation",
            "core": "core_inflation", "ip": "ip_yoy", "energy": "global_energy_yoy",
            "food": "global_food_yoy", "imports": "import_price_yoy", "gscpi": "gscpi",
            "d_be5": "d_breakeven_5y"}
    hist = {"dates": [d.strftime("%Y-%m") for d in h.index]}
    hist.update({k: _series(h[c]) if c in h else [] for k, c in cols.items()})

    last = {k: {"value": _r(df[c].dropna().iloc[-1]), "month": df[c].dropna().index[-1].strftime("%Y-%m")}
            for k, c in cols.items() if c in df and df[c].notna().any()}

    notes = []
    if "credit_spread_source" in df.columns:
        notes.append("FRED now publishes only three years of ICE BofA high-yield spreads; "
                     "the financial-conditions index and LP controls use Moody's Baa–10Y spread instead.")
    out = {"built_at": built.isoformat(timespec="seconds"), "llm_used": bool(l0.get("llm_used")),
           "notes": notes,
           "offline_test": bool(args.offline),
           "layer0": l0, "state": state, "latest": last, "lp_state": lp,
           "iar": iar, "gar": gar, "history": hist}
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "dashboard.json").write_text(json.dumps(out, default=str))

    # Append-only Layer 0 archive
    arch_p = DOCS / "signals.json"
    arch = json.loads(arch_p.read_text()) if arch_p.exists() else []
    if l0.get("ok") and not l0.get("stale") and not args.offline:
        arch.append({k: l0.get(k) for k in ["timestamp", "gpr_nowcast", "severity_score", "regime",
                                            "cameo_codes", "ais_anomaly", "article_count",
                                            "llm_used", "gpr_anchor_z", "anchor_month"]}
                    | {"headlines": (l0.get("top_headlines") or [])[:5]})
    arch_p.write_text(json.dumps(arch, indent=1, default=str))
    log.info(f"dashboard.json written; GIPI {state['gipi']} ({'open' if state['open'] else 'closed'}); "
             f"Layer 0 ẑ={l0_z}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:                      # surface the reason in the public run summary
        if os.getenv("GITHUB_ACTIONS"):
            msg = f"{type(e).__name__}: {e}".replace("\n", " ")[:500]
            print(f"::error title=Dashboard build failed::{msg}")
        raise
