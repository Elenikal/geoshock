"""
run.py — GeoShock v2 Master Pipeline
─────────────────────────────────────────────────────────────────────────────
Executes all four layers in sequence:

  Layer 0  Real-time event detection (GDELT + LLM CAMEO + AIS proxy)
  Layer 1  Data pipeline (FRED + Yahoo + GPR + GSCPI + Arab Light + FAO)
           → 6 new inflation transmission channels + GIPI composite
  Layer 2  Econometric models (Local Projections · GaR v2 · VAR)
  Layer 3  Export (Parquet + CSVs + figures)

Usage
─────
  python run.py                    Full run
  python run.py --layer0-only      Only event detection, then exit
  python run.py --skip-layer0      Skip event detection
  python run.py --use-cache        Skip API data fetch
  python run.py --no-llm           Rule-based CAMEO (no Anthropic key needed)
  python run.py --skip-var         Skip VAR (faster iteration)
  python run.py --lookback 72      GDELT lookback hours (default 48)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys, io
# Force UTF-8 on stdout/stderr on Python 3.9 / macOS (default is ASCII/latin-1)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer'):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import io as _io
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=_io.TextIOWrapper(open('/dev/stderr', 'wb'), encoding='utf-8', errors='replace'),
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import cfg


# ─────────────────────────────────────────────────────────────────────────────
def run_layer0(use_llm: bool = True, lookback_hours: int = 48) -> dict:
    log.info("\n" + "━" * 58)
    log.info("  LAYER 0 — REAL-TIME EVENT DETECTION")
    log.info("━" * 58)
    try:
        from data.event_detector import EventDetector
        ed  = EventDetector(anthropic_key=cfg.ANTHROPIC_KEY or None)
        sig = ed.detect(
            lookback_hours=lookback_hours,
            use_llm=use_llm,
            use_ais=True,
            max_articles=cfg.L0_MAX_ARTICLES,
            ais_tickers=cfg.L0_TANKER_TICKERS,
            ais_z_threshold=cfg.L0_AIS_Z_THRESHOLD,
        )
        result = sig.to_dict()

        # Save to disk
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(cfg.OUTPUT_DIR / "event_signal.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info(f"  Event signal → {cfg.OUTPUT_DIR / 'event_signal.json'}")

        return result
    except Exception as e:
        log.error(f"  Layer 0 error: {e}")
        return {"error": str(e), "regime": "UNKNOWN", "severity_score": 0.0,
                "gpr_nowcast": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
def run_layer1(use_cache: bool = False) -> pd.DataFrame:
    log.info("\n" + "━" * 58)
    log.info("  LAYER 1 — DATA PIPELINE  (v2: +Inflation Channels +GIPI)")
    log.info("━" * 58)

    from data.pipeline import DataPipeline
    dp = DataPipeline()
    df = dp.build(use_cache=use_cache)

    inflation_cols = [c for c in df.columns if any(
        k in c for k in ("breakeven", "import_price", "energy_yoy",
                         "food_yoy", "arab_wti", "gipi", "fao_",
                         "usd_", "natgas_", "spr_")
    )]
    log.info(f"  New inflation channel features ({len(inflation_cols)}): {inflation_cols}")

    if "gipi" in df.columns:
        n = df["gipi"].notna().sum()
        last = df["gipi"].dropna().iloc[-1] if n > 0 else np.nan
        log.info(f"  GIPI: {n} obs, latest = {last:.2f}")

    log.info(f"  Dataset: {df.shape[0]} rows × {df.shape[1]} cols  "
             f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def run_local_projections(df: pd.DataFrame) -> dict:
    log.info("\n" + "━" * 58)
    log.info("  LAYER 2A — LOCAL PROJECTIONS (LP-IRF)")
    log.info("━" * 58)
    try:
        from models.local_projections import LocalProjections
        outcomes = [o for o in ["ip_yoy", "cpi_inflation", "unemp",
                                 "sp500_return", "ip_growth"]
                    if o in df.columns]
        lp = LocalProjections(
            df=df, shock_col="gpr_shock", outcome_cols=outcomes,
            horizons=cfg.LP_HORIZONS, lags=cfg.LP_LAGS,
            n_bootstrap=cfg.LP_BOOTSTRAP_REPS,
        )
        results = lp.fit_all()

        try:
            cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            lp.plot_irf_grid(save_path=str(cfg.FIGURE_DIR / "irf_grid.png"))
            log.info(f"  IRF grid → {cfg.FIGURE_DIR / 'irf_grid.png'}")
        except Exception as e:
            log.warning(f"  IRF plot: {e}")

        return {"lp": results}
    except Exception as e:
        log.error(f"  LP error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
def run_growth_at_risk(df: pd.DataFrame, l0_signal: dict | None = None) -> dict:
    # Layer 0 real-time update: inject gpr_nowcast as current-month observation
    if l0_signal and l0_signal.get("gpr_nowcast") and "gpr_z" in df.columns:
        nowcast = float(l0_signal["gpr_nowcast"])
        today = pd.Timestamp.today().normalize()
        if today not in df.index:
            last_row = df.iloc[-1].copy()
            last_row["gpr_z"] = nowcast
            last_row.name = today
            df = pd.concat([df, last_row.to_frame().T])
        else:
            df.loc[today, "gpr_z"] = nowcast
        log.info("  L0 real-time update: gpr_z=%.2f injected", nowcast)
    log.info("\n" + "━" * 58)
    log.info("  LAYER 2B — GROWTH-AT-RISK  v2  (GPR + FCI + GIPI + GPR×GIPI)")
    log.info("━" * 58)
    try:
        from models.quantile_risk import GrowthAtRisk, GaRRobustness
        horizons = cfg.GAR_HORIZONS
        gipi_ok  = ("gipi" in df.columns and df["gipi"].notna().sum() >= 36)
        results  = {}

        for spec, use_g, use_i in [
            ("baseline", False,        False),
            ("enhanced", gipi_ok,      gipi_ok and cfg.GAR_USE_INTERACTION),
        ]:
            log.info(f"\n  Spec: {spec}  gipi={use_g}  interaction={use_i}")
            gar = GrowthAtRisk(
                df=df, outcome="ip_yoy", gpr_col="gpr_z",
                gipi_col="gipi" if use_g else None,
                use_interaction=use_i,
            )
            res = gar.fit(horizons=horizons)
            for h, r in res.items():
                log.info(f"    h={h:2d}  median={r.median_forecast:+5.1f}%  "
                         f"GaR5={r.gar_5:+5.1f}%  "
                         f"P(rec)={r.prob_recession:.1%}  "
                         f"skew={r.conditional_skewness:+.2f}")
            results[spec] = {"gar": gar, "results": res}

            if spec == "enhanced":
                for h in horizons:
                    try:
                        gar.plot_fan_chart(
                            horizon=h,
                            save_path=str(cfg.FIGURE_DIR / f"gar_fan_h{h}.png"))
                        gar.plot_distribution(
                            horizon=h,
                            save_path=str(cfg.FIGURE_DIR / f"gar_dist_h{h}.png"))
                    except Exception as e:
                        log.warning(f"  GaR plot h={h}: {e}")

        # ── Robustness checks (if GIPI available) ──────────────────────────
        if gipi_ok:
            log.info("\n  ── Robustness checks ──────────────────────────────")
            try:
                rob = GaRRobustness(
                    df=df, outcome="ip_yoy", gpr_col="gpr_z", gipi_col="gipi")
                rob_result = rob.run(horizons=horizons)
                rob_summary = rob.summary()

                cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                rob.save(str(cfg.OUTPUT_DIR / "robustness_checks.csv"))
                results["robustness"] = {"result": rob_result, "summary": rob_summary}

                # Log key finding
                for h in horizons:
                    impr_05 = rob_result.improvement.get(h, {}).get(0.05, np.nan)
                    impr_10 = rob_result.improvement.get(h, {}).get(0.10, np.nan)
                    orth_sig = rob_result.orth_significant.get(h, False)
                    log.info(f"  h={h:2d}: pinball impr τ=0.05={impr_05:.1f}%  "
                             f"τ=0.10={impr_10:.1f}%  "
                             f"orth_significant={orth_sig}")
            except Exception as e:
                log.warning(f"  Robustness checks error: {e}")

        return results
    except Exception as e:
        log.error(f"  GaR error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
def run_var(df: pd.DataFrame) -> dict:
    log.info("\n" + "━" * 58)
    log.info("  LAYER 2C — STRUCTURAL VAR + FEVD + GRANGER")
    log.info("━" * 58)
    try:
        from models.var_model import VARModel
        var_vars = [v for v in cfg.VAR_VARIABLES if v in df.columns]
        log.info(f"  VAR variables ({len(var_vars)}): {var_vars}")

        vm = VARModel(df=df, variables=var_vars, lags=cfg.VAR_LAGS)
        vm.fit()
        vm.compute_irf()
        vm.compute_fevd()
        gc = vm.granger_causality()

        try:
            vm.plot_irf(save_path=str(cfg.FIGURE_DIR / "var_irf_grid.png"))
            vm.plot_fevd(save_path=str(cfg.FIGURE_DIR / "fevd.png"))
            gc.to_csv(cfg.OUTPUT_DIR / "granger_causality.csv")
            if hasattr(vm, "fevd_df_"):
                vm.fevd_df_.to_csv(cfg.OUTPUT_DIR / "fevd.csv")
            log.info("  VAR outputs saved")
        except Exception as e:
            log.warning(f"  VAR outputs: {e}")

        return {"var": vm, "granger": gc}
    except Exception as e:
        log.error(f"  VAR error: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
def export_summary(df: pd.DataFrame, l0: dict, gar: dict) -> None:
    log.info("\n" + "━" * 58)
    log.info("  LAYER 3 — EXPORT")
    log.info("━" * 58)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.OUTPUT_DIR / "master_dataset.parquet")
    df.tail(36).to_csv(cfg.OUTPUT_DIR / "master_dataset_recent.csv")

    # GaR summary table
    rows = []
    for spec, d in gar.items():
        for h, r in d.get("results", {}).items():
            rows.append({
                "spec": spec, "horizon": h,
                "median": round(r.median_forecast, 2),
                "gar_5": round(r.gar_5, 2),
                "gar_25": round(r.gar_25, 2),
                "p_neg": round(r.prob_neg_growth, 4),
                "p_rec": round(r.prob_recession, 4),
                "skew": round(r.conditional_skewness, 4),
            })
    if rows:
        tbl = pd.DataFrame(rows)
        tbl.to_csv(cfg.OUTPUT_DIR / "gar_summary.csv", index=False)
        log.info("\n  GaR summary:")
        log.info(tbl.to_string(index=False))

    # GIPI info
    if "gipi" in df.columns:
        info = {
            "n_obs": int(df["gipi"].notna().sum()),
            "mean":  round(float(df["gipi"].mean()), 3),
            "std":   round(float(df["gipi"].std()),  3),
            "last":  round(float(df["gipi"].dropna().iloc[-1]), 3)
                     if df["gipi"].notna().any() else None,
            "last_date": str(df["gipi"].dropna().index[-1].date())
                         if df["gipi"].notna().any() else None,
        }
        with open(cfg.OUTPUT_DIR / "gipi_diagnostics.json", "w") as f:
            json.dump(info, f, indent=2)
        log.info(f"\n  GIPI last={info['last']} ({info['last_date']})")

    # Robustness check summary
    if "robustness" in gar and "summary" in gar["robustness"]:
        rb = gar["robustness"]["summary"]
        rb.to_csv(cfg.OUTPUT_DIR / "robustness_checks.csv", index=False)
        log.info("  Robustness checks → robustness_checks.csv")

    log.info(f"\n  All outputs → {cfg.OUTPUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="GeoShock v2 — Geopolitical Risk → US Macro Tail Risk Pipeline")
    ap.add_argument("--layer0-only",  action="store_true")
    ap.add_argument("--skip-layer0",  action="store_true")
    ap.add_argument("--use-cache",    action="store_true")
    ap.add_argument("--no-llm",       action="store_true")
    ap.add_argument("--skip-var",     action="store_true")
    ap.add_argument("--lookback",     type=int, default=48,
                    help="GDELT lookback hours (default 48)")
    args = ap.parse_args()

    t0 = time.time()
    log.info("\n" + "═" * 58)
    log.info("  GEOSHOCK v2  —  Geo-Risk → US Macro Tail Risk")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    log.info("═" * 58)

    # Layer 0
    l0 = {}
    if not args.skip_layer0:
        l0 = run_layer0(use_llm=not args.no_llm, lookback_hours=args.lookback)
        if args.layer0_only:
            log.info("\n  --layer0-only: exiting after Layer 0.")
            return

    # Layer 1
    df = run_layer1(use_cache=args.use_cache)

    # Layer 2
    _lp  = run_local_projections(df)
    gar  = run_growth_at_risk(df)
    if not args.skip_var:
        _var = run_var(df)

    # Layer 3
    export_summary(df, l0, gar)

    # Done
    log.info(f"\n  ✓  Pipeline complete in {time.time()-t0:.0f}s")
    if l0.get("regime"):
        icon = {"CRISIS": "🔴", "ELEVATED": "🟡", "CALM": "🟢"}.get(
            l0["regime"], "⚪")
        log.info(f"  {icon} Current regime: {l0['regime']}  "
                 f"(GPR nowcast z = {l0.get('gpr_nowcast', 0):.2f})")
    log.info("═" * 58)


if __name__ == "__main__":
    main()
