## Project Overview

**Title:** *Fire Without Smoke: Geopolitical Risk and the New Inflation Decoupling*

**Research question:** Why do geopolitical shocks that historically triggered inflation and recession — Middle East conflicts, energy supply disruptions, nuclear standoffs — produce benign macroeconomic outcomes in 2023–2025? We construct a new index of geopolitical inflation transmission (GIPI), document its near-zero correlation with GPR (ρ = 0.006), identify the three structural buffers responsible, and build a real-time monitoring system that nowcasts tail risk by combining a historical GaR model with live news surveillance that bridges the official index's one-month publication lag.

**Methodology:** The paper builds a four-layer empirical pipeline:

1. **Layer 0 — Real-Time Event Detection (Phase 0):** Live GDELT 2.0 news surveillance queried every run, LLM CAMEO event coding (Anthropic Claude), and an AIS tanker equity proxy (INSW, TK, TRMD, FRO) for Strait of Hormuz disruption risk. Produces a sub-monthly GPR nowcast $\hat{z}_t = f(S_t, \text{AIS}_t, \hat{z}^{\text{GPR}}_{t-1})$ that is injected into the GaR model as the current-month observation, replacing the stale official release.

2. **Layer 1 — Data Pipeline (Phase 1):** Monthly macro panel (1997–2025, N = 336) combining FRED (CPI, IP, unemployment, Fed funds, TIPS breakevens, HY spreads), Yahoo Finance (VIX, WTI crude, tanker equities), the Caldara–Iacoviello GPR index (downloaded directly from matteoiacoviello.com), NY Fed GSCPI, FAO food price index, and the Arab Light–WTI spread. Principal component of five transmission channels yields the GIPI composite.

3. **Layer 2 — Econometric Models (Phase 2):** Three specifications estimated jointly:
   - **Local Projections** (Jordà 2005): Newey-West HAC IRFs of the GPR shock on IP, CPI, and unemployment at horizons h = 0…24 months, with 500-rep bootstrap confidence bands and regime-conditional paths (calm / elevated / crisis).
   - **Growth-at-Risk** (Adrian, Boyarchenko & Giannone 2019): Quantile regressions at τ ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95} and h ∈ {3, 6, 12} months. Baseline spec (GPR + FCI) extended to enhanced spec adding GIPI and a GPR × GIPI interaction term; 21/21 interaction coefficients statistically significant; pinball loss improvement 13.7% at (h = 12, τ = 5%).
   - **Structural VAR:** Cholesky-identified SVAR with ordering GPR → Oil → FCI → IP → CPI → FFR; FEVD, historical decomposition, Granger causality tests (GPR → IP p = 0.004, GPR → CPI p = 0.003).

4. **Layer 3 — Export and Dashboard (Phase 3):** Results exported to CSV/Parquet; a Streamlit dashboard provides interactive GaR fan charts, regime monitoring, LP-IRF explorer, GIPI channel decomposition, and the live Layer 0 nowcast panel with explicit publication-lag bridging display.

**Target journals:** American Economic Review, Journal of International Economics, Review of Economics and Statistics

---

## Repository Structure

```
geoshock_v2/
│
├── run.py                         # Master pipeline orchestrator
│                                  # Flags: --layer0-only  Layer 0 only, then exit
│                                  #        --skip-var     Skip slow VAR
│                                  #        --no-llm       Rule-based CAMEO fallback
│                                  #        --use-cache    Skip API data fetch
│                                  #        --lookback N   GDELT lookback hours
│
├── config.py                      # API keys, paths, model hyperparameters
│                                  # (FRED_KEY, ANTHROPIC_KEY, GAR_HORIZONS, etc.)
│
├── requirements.txt               # Python dependencies
│
├── data/
│   ├── pipeline.py                # Layer 1: data fetch and feature engineering
│   │                              # Sources: FRED · Yahoo Finance · Iacoviello GPR
│   │                              #          NY Fed GSCPI · FAO · Arab Light–WTI
│   │                              # Output:  master_dataset.parquet + CSVs
│   │
│   ├── event_detector.py          # Layer 0: real-time event detection
│   │                              # GDELT 2.0 Doc API → LLM CAMEO coding →
│   │                              # AIS tanker proxy → composite severity score →
│   │                              # GPR nowcast → outputs/event_signal.json
│   │
│   └── _llm_cameo_helper.py       # Subprocess helper for Unicode-safe LLM calls
│                                  # Isolated child process with forced UTF-8;
│                                  # uses urllib.request to bypass httpx codec chain
│
├── models/
│   ├── local_projections.py       # Layer 2A: Jordà (2005) LP-IRF
│   │                              # Newey-West HAC SEs · bootstrap CIs ·
│   │                              # regime-conditional IRFs · event-study overlay
│   │
│   ├── quantile_risk.py           # Layer 2B: Growth-at-Risk
│   │                              # Baseline (GPR + FCI) and enhanced
│   │                              # (+ GIPI + GPR×GIPI) specs; pinball loss;
│   │                              # robustness checks; LASSO variable selection
│   │
│   └── var_model.py               # Layer 2C: Structural VAR
│                                  # Cholesky SVAR · FEVD · Granger causality ·
│                                  # historical decomposition
│
├── dashboard/
│   └── app.py                     # Streamlit interactive dashboard
│                                  # Panels: Layer 0 nowcast (with lag indicator) ·
│                                  #         GPR monitor · GaR fan charts ·
│                                  #         LP-IRF explorer · GIPI channels ·
│                                  #         Robustness · Regime history
│
└── outputs/                       # Generated artefacts (git-ignored)
    ├── master_dataset.parquet
    ├── gar_summary.csv
    ├── robustness_checks.csv
    ├── granger_causality.csv
    ├── event_signal.json           # Layer 0 latest signal (refreshed each run)
    └── figures/
        ├── fig01_channels.pdf      # GIPI transmission channels decomposition
        ├── fig02_gipi.pdf          # GIPI index time series
        ├── fig03_gpr.pdf           # GPR monitor with regime shading
        ├── fig04_decoupling.pdf    # GPR–GIPI decoupling scatter (2019–2025)
        ├── fig05_lp_irf.pdf        # Local projection impulse response functions
        ├── fig06_fan_h{3,6,12}.pdf # GaR fan charts at three horizons
        ├── fig07_densities.pdf     # Predictive density: baseline vs enhanced
        ├── fig08_robustness.pdf    # Robustness checks summary
        └── fig09_layer0.pdf        # Layer 0 real-time panel (March 2026)
```

---

## Run Commands

```bash
# Real-time nowcast only (fastest — no API data fetch)
python run.py --layer0-only --no-llm

# Full pipeline, skip slow VAR (recommended for iteration)
python run.py --skip-var --no-llm

# Full pipeline including VAR (paper replication)
python run.py --no-llm

# Full pipeline with LLM CAMEO coding (requires Anthropic API credits ~$0.002/run)
python run.py --skip-var

# Use cached data, skip fetch
python run.py --use-cache --skip-var --no-llm

# Launch interactive dashboard
streamlit run dashboard/app.py
```

---

## Data Sources

| Series | Source | Frequency | Start |
|--------|--------|-----------|-------|
| GPR Index | Caldara & Iacoviello (2022), matteoiacoviello.com | Monthly | 1985 |
| CPI, Core CPI | FRED: CPIAUCSL, CPILFESL | Monthly | 1947 |
| Industrial Production | FRED: INDPRO | Monthly | 1919 |
| Unemployment Rate | FRED: UNRATE | Monthly | 1948 |
| Federal Funds Rate | FRED: FEDFUNDS | Monthly | 1954 |
| 5Y TIPS Breakeven | FRED: T5YIE | Monthly | 2003 |
| HY OAS Spread | FRED: BAMLH0A0HYM2 | Monthly | 1997 |
| VIX | Yahoo Finance: ^VIX | Monthly | 1990 |
| WTI Crude Oil | Yahoo Finance: CL=F | Monthly | 1983 |
| NY Fed GSCPI | NY Fed website | Monthly | 1997 |
| FAO Food Price Index | fao.org | Monthly | 1990 |
| Arab Light–WTI Spread | Derived: FRED + Yahoo Finance | Monthly | 1997 |
| GDELT 2.0 News | api.gdeltproject.org (real-time) | 15-min | 2015 |
| Tanker Equities (AIS proxy) | Yahoo Finance: INSW, TK, TRMD, FRO | Daily | 2010 |
