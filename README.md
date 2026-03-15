# GeoShock v2 — Geopolitical Risk → US Macro Tail Risk

**Paper:** *Fire Without Smoke: Geopolitical Risk and the New Inflation Decoupling*

## Quick start

```bash
# 1. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp config_template.py config.py
# Edit config.py: add FRED_KEY (free at fred.stlouisfed.org)
# ANTHROPIC_KEY is optional — use --no-llm to skip

# 4. Run full pipeline
python run.py --no-llm                    # full run, rule-based CAMEO
python run.py --skip-var --no-llm         # faster: skip VAR + run OOS + IV
python run.py --skip-var --skip-oos --skip-iv --no-llm  # fastest

# 5. Populate paper tables from results
python populate_oos_table.py              # fills OOS and IV tables in .tex
pdflatex geoshock_paper_v8.tex && pdflatex geoshock_paper_v8.tex

# 6. Dashboard
streamlit run dashboard/app.py
```

## Pipeline stages

| Flag         | Stage                              | Time   |
|--------------|------------------------------------|--------|
| Layer 0      | GDELT + LLM CAMEO + AIS            | 30–60s |
| Layer 1      | FRED + Yahoo + GPR + GSCPI + FAO   | 60–90s |
| Layer 2A     | Local Projections (IRFs)           | 10s    |
| Layer 2B     | Growth-at-Risk (baseline+enhanced) | 20s    |
| Layer 2C     | VAR + FEVD + Granger               | 60s    |
| Layer 2D     | Pseudo-OOS backtest (GW 2006)      | 5–10m  |
| Layer 2E     | IV endogeneity check (OPEC+SPR)    | 10s    |
| Layer 3      | Export (Parquet + CSV + figures)   | 5s     |

## Key outputs (`outputs/`)

| File                      | Contents                                      |
|---------------------------|-----------------------------------------------|
| `gar_summary.csv`         | GaR nowcasts: median, GaR5, P(rec), skew      |
| `oos_results.csv`         | OOS pinball loss, GW test stats               |
| `iv_results.csv`          | IV diagnostics: F, Sargan, DWH                |
| `robustness_checks.csv`   | 21-cell orthogonalisation + LASSO check       |
| `master_dataset.parquet`  | Full panel (1985–present)                     |
| `event_signal.json`       | Layer 0 nowcast snapshot                      |

## IV endogeneity test

The IV module (`models/iv_gipi.py`) instruments for GIPI_{t-1} using:
- **Z1**: OPEC spare capacity (IEA, lag 6 months) — `opec_spare_cap` column
- **Z2**: US SPR stocks (FRED WCSSTUS1, lag 1 month) — `spr_stocks` column

If OPEC spare capacity is not in your dataset, the module falls back to
a synthetic proxy and flags results accordingly. Add IEA OPEC spare
capacity data to `data/pipeline.py` for full results.

Decision rule (automated verdict in `outputs/iv_results.csv`):
- F < 10 → weak instruments → frame as forecasting paper
- F ≥ 10, Sargan p < 0.05 → exclusion violation
- F ≥ 10, Sargan ok, DWH p > 0.10 → GIPI exogenous; OLS preferred ✓
- F ≥ 10, Sargan ok, DWH p ≤ 0.10 → endogeneity confirmed; 2SLS needed

## Replicating paper tables

After `python run.py --skip-var`:

```bash
python populate_oos_table.py
# → fills Table 7 (OOS) and Table 8 (IV) in geoshock_paper_v8.tex
pdflatex geoshock_paper_v8.tex
pdflatex geoshock_paper_v8.tex   # second pass for cross-references
```

## Repository structure

```
geoshock_v2/
├── run.py                    Master pipeline runner
├── populate_oos_table.py     Auto-fills OOS + IV tables in .tex
├── config_template.py        API key configuration (rename to config.py)
├── requirements.txt
├── data/
│   ├── pipeline.py           Data fetch: FRED, Yahoo, GPR, GSCPI, FAO
│   ├── event_detector.py     Layer 0: GDELT + LLM CAMEO + AIS
│   └── _llm_cameo_helper.py  LLM subprocess helper
├── models/
│   ├── quantile_risk.py      GaR (baseline + enhanced + OOS/GW2006)
│   ├── local_projections.py  Jordà LP-IRFs
│   ├── var_model.py          SVAR + FEVD + Granger
│   └── iv_gipi.py            IV endogeneity check (OPEC + SPR)
└── dashboard/
    └── app.py                Streamlit real-time dashboard
```
