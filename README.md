# GeoShock v2 — Geopolitical Risk → US Macro Tail Risk

**Full research pipeline** for estimating how Middle East geopolitical shocks
transmit to US macroeconomic tail risk, with real-time event detection.

---

## Architecture

```
Layer 0  EVENT DETECTION          GDELT news → LLM CAMEO coding + AIS proxy
Layer 1  DATA PIPELINE            FRED + Yahoo Finance + GPR + GSCPI
                                  + EIA Arab Light + FAO Food + 6 new channels
Layer 2  ECONOMETRIC MODELS       Local Projections · GaR v2 · Structural VAR
Layer 3  EXPORTS + DASHBOARD      Parquet · CSV · Streamlit
```

### Layer 0 — Real-Time Event Detection (NEW in v2)

Three signal sources combined into a composite `EventSignal`:

| Source | Description | Key |
|--------|-------------|-----|
| **GDELT 2.0 Doc API** | Queries last N hours of global news for ME conflict events | None (free) |
| **LLM CAMEO Coding** | Claude reads headlines → CAMEO codes + severity 0–10 + GPR z-estimate | `ANTHROPIC_API_KEY` |
| **AIS Tanker Proxy** | VLCC tanker equities (INSW/TK/TRMD/FRO) + Brent–WTI spread as Strait of Hormuz proxy | None (Yahoo Finance) |

**CAMEO** = Conflict and Mediation Event Observations. Codes 190–200 cover
military force, blockades, aerial strikes, WMD use. Severity 0–10.

**AIS proxy rationale:** When the Strait of Hormuz is under threat:
- VLCC tanker stocks drop sharply (disruption risk / re-routing cost)
- Brent–WTI spread widens (Brent carries ME supply premium)
- Both signals firing simultaneously (z > 1.5 SD) = high-conviction anomaly

*For production:* Replace Yahoo Finance proxy with real AIS APIs
(MarineTraffic, Kpler, VesselsValue, Spire Maritime).

### Layer 1 — Data Pipeline (enhanced v2)

**Core sources** (unchanged):
- FRED API: IP, CPI, unemployment, GDP, HY spread, term spread, Fed Funds, WTI
- Yahoo Finance: VIX, S&P 500, defense ETF (ITA), energy ETF (XLE), gold, TLT
- Caldara–Iacoviello GPR Index (direct download, no key)
- NY Fed GSCPI (direct download, no key)

**New inflation transmission channels:**

| FRED Code | Variable | Channel |
|-----------|----------|---------|
| `T5YIE` | 5Y TIPS breakeven | Inflation expectations |
| `T10YIE` | 10Y TIPS breakeven | Inflation expectations |
| `IR` / `IREXFUELS` | Import price index (all / ex-fuel) | Cost-push |
| `MHHNGSP` | Henry Hub natural gas | Energy / LNG |
| `PNRGINDEXM` | IMF Global Energy Price Index | Energy (broader than WTI) |
| `PFOODINDEXM` | IMF Global Food Price Index | Food / MENA disruption |
| `DTWEXBGS` | USD broad index | Pass-through amplifier |
| `WCSSTUS1` | SPR crude stocks | Supply buffer signal |
| EIA direct | Saudi Arabian Light landed cost | Geo-premium in oil |
| FAO direct | FAO Food Price Index + sub-indices | Cereals, oils, dairy |

**GIPI — Geopolitical Inflation Pressure Index:**
PC1 of [GSCPI, Import Price YoY, Global Energy YoY, Global Food YoY, ΔBreakeven 5Y].
Higher GIPI = more geopolitical inflation transmission pressure.

### Layer 2 — Models

**Local Projections (Jordà 2005)**
```
y_{t+h} - y_{t-1} = α_h + β_h · GPR_shock_t + Γ_h · X_t + lags + ε
```
Newey-West SEs, 500-rep bootstrap CIs, regime-conditional IRFs.

**Growth-at-Risk v2 (Adrian, Boyarchenko & Giannone 2019)**

*Baseline:*
```
Q_τ(y_{t+h}) = α_τ + β_τ·GPR_t + γ_τ·FCI_t + η_τ·y_t
```

*Enhanced (new):*
```
Q_τ(y_{t+h}) = α_τ + β_τ·GPR_t + γ_τ·FCI_t + δ_τ·GIPI_t
             + ζ_τ·(GPR_t × GIPI_t) + η_τ·y_t
```

GIPI as a 3rd regressor captures inflation transmission channels.
The GPR×GIPI interaction tests whether high supply-chain/inflation pressure
amplifies the tail-risk impact of geopolitical shocks.

**Structural VAR**
Cholesky ordering: GPR → GIPI → Oil → VIX → IP → CPI → HY spread → Fed Funds
FEVD + Granger causality battery.

---

## Quickstart

```bash
# 1. Clone / unzip
cd geoshock_v2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API keys
cp .env.example .env
# Edit .env: add FRED_API_KEY (required) and ANTHROPIC_API_KEY (optional)

# 4. Run full pipeline
python run.py

# 5. Run Layer 0 only (real-time event check)
python run.py --layer0-only

# 6. Launch dashboard
streamlit run dashboard/app.py
```

**CLI flags:**
```
--layer0-only    Run Layer 0 only, then exit
--skip-layer0    Skip event detection (faster for model iteration)
--use-cache      Load cached data (skip API calls)
--no-llm         Use rule-based CAMEO (no Anthropic key needed)
--skip-var       Skip VAR estimation
--lookback N     GDELT lookback hours (default 48)
```

**Layer 0 directly:**
```bash
python data/event_detector.py 72   # scan last 72 hours
```

---

## Output files

| File | Description |
|------|-------------|
| `outputs/master_dataset.parquet` | Full feature dataset |
| `outputs/master_dataset_recent.csv` | Last 36 months (CSV) |
| `outputs/event_signal.json` | Layer 0 EventSignal |
| `outputs/gar_summary.csv` | GaR results (baseline vs enhanced) |
| `outputs/gipi_diagnostics.json` | GIPI metadata |
| `outputs/granger_causality.csv` | Granger test p-values |
| `outputs/figures/irf_grid.png` | LP impulse responses |
| `outputs/figures/gar_fan_h{3,6,12}.png` | GaR fan charts |
| `outputs/figures/gar_dist_h{3,6,12}.png` | Predictive distributions |
| `outputs/figures/fevd.png` | FEVD bar chart |
| `outputs/figures/var_irf_grid.png` | VAR Cholesky IRFs |

---

## Key references

| Citation | Journal | Role in paper |
|----------|---------|---------------|
| Caldara & Iacoviello (2022) | AER 112(4) | GPR index |
| Adrian, Boyarchenko & Giannone (2019) | AER 109(4) | Growth-at-Risk baseline |
| Jordà (2005) | AER 95(1) | Local Projections |
| Kilian (2009) | AER 99(3) | Oil-macro transmission |
| Caldara, Conlisk, Iacoviello & Penn (2025) | JME forthcoming | GPR → inflation |
| Carrière-Swallow et al. (2023) | JIMF | Shipping → CPI pass-through |
| Benigno et al. (2022) | NY Fed SR 1017 | GSCPI construction |
| López-Salido & Loria (2024) | JME | Inflation-at-Risk |

---

## PENDING (paper tasks)

- [ ] Fill in author name, affiliation, email in LaTeX source
- [ ] Run LASSO-QR robustness check on GIPI variable selection
- [ ] Add event study validation numbers (Abqaiq, Soleimani)
- [ ] Add prediction market data (Polymarket/Metaculus) for live validation
- [ ] Consider GDELT-based NLP fear/uncertainty index as additional regressor

---

*GeoShock v2 — built March 2026*
