# GeoShock — Geopolitical Threats, Inflation Channels and US Macro Risk

Based on Kalamara, *Fire Without Smoke: Geopolitical Threats, Inflation Channels, and the 2023–25 Decoupling* (working paper, 2026).

## Specification

- **Shocks:** standardised monthly change in GPRT (threats); GPRA (acts) and GPR for comparison.
- **GIPI:** PC1 of global energy YoY, import prices YoY, global food YoY, GSCPI and ΔT5YIE; 1998M01–2025M12,
  ΔT5YIE set to its mean before 2003M02; rescaled to unit variance.
- **Local projections:** calendar leads/lags, controls at t−1, Newey–West bandwidth h+1; state-dependent
  version fully interacted with 1[GIPI_{t−1} ≥ median] (`LocalProjections.fit_state_dependent`).
- **Inflation-at-Risk / Growth-at-Risk:** quantile regressions on GPRT_z, FCI, GIPI_{t−1}, GPRT×GIPI_{t−1}
  and the current outcome; nowcast evaluated at the last month; moving-block bootstrap (block h+12).
- **Layer 0:** ẑ = 0.6·(0.4·S) + 0.4·z^GPRT_{t−1}, S = 0.5·LLM + 0.3·P75(rule) + 0.2·10·1[AIS].
  Every signal is archived in `docs/data/signals.json` by the daily build.

## Live dashboards

**GitHub Pages — https://elenikal.github.io/geoshock/** (static, rebuilt daily).
`.github/workflows/dashboard.yml` runs `scripts/build_dashboard_data.py` every day at 07:17 UTC:
it fetches live data, runs Layer 0, re-estimates the paper's models and commits
`docs/data/dashboard.json` and the append-only Layer 0 archive `docs/data/signals.json`.
The page itself (`docs/index.html`, `docs/assets/`) only reads those files.
`GEOSHOCK_STRICT=1` makes the build fail rather than fall back to synthetic data.

One-time setup on GitHub:
1. Settings → Pages → *Deploy from a branch* → `main` / `/docs`.
2. Settings → Secrets and variables → Actions → add `FRED_API_KEY`
   (and optionally `ANTHROPIC_API_KEY` for LLM CAMEO coding).
3. Actions → *Update dashboard* → *Run workflow* for the first build.

Local test without live data: `python scripts/build_dashboard_data.py --offline`, then
`python -m http.server --directory docs`.

**Streamlit — https://geoshock.onrender.com** (interactive, on Render). Bot commits carry
`[skip render]` so the daily data update does not trigger a redeploy.

## Live pipeline

```bash
cp .env.example .env                   # add FRED_API_KEY (and ANTHROPIC_API_KEY, optional)
python run.py --no-llm                 # fetch data, Layer 0, LP, GaR + robustness, VAR, IV
python run.py --use-cache --skip-var --skip-iv --skip-layer0   # fast rerun on cached data
streamlit run dashboard/app.py
```

Note: the IV module (`models/iv_gipi.py`) still runs on synthetic instruments because no OPEC
spare-capacity series is fetched; its output is not used in the paper.

## Repository structure

```
run.py                   live pipeline (Layers 0–3)
config.py                keys, series codes, hyperparameters
data/pipeline.py         data fetch, features, GIPI
data/event_detector.py   Layer 0 (GDELT + CAMEO + tanker proxy)
models/local_projections.py
models/quantile_risk.py  GaR/IaR + robustness checks
models/var_model.py      SVAR / FEVD / Granger (not in the paper)
models/iv_gipi.py        IV check (synthetic instruments; not in the paper)
dashboard/app.py         Streamlit dashboard
```
