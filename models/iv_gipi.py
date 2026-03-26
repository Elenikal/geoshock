"""
models/iv_gipi.py
─────────────────────────────────────────────────────────────────────────────
Instrumental Variables Robustness Check for GIPI Endogeneity.

Strategy
--------
GIPI contains energy prices, food prices, supply-chain pressure, and
breakevens that respond endogenously to US IP growth (demand channel).
We instrument for GIPI_{t-1} using two supply-side, predetermined variables:

  Z1: OPEC spare capacity (lagged 6 months)
      Source: IEA Monthly Oil Market Report (or FRED STEO proxy)
      Relevance: determines whether GPR shock passes through to energy prices
      Exclusion: OPEC investment decisions made quarters earlier

  Z2: US Strategic Petroleum Reserve stocks (lagged 1 month)
      Source: FRED WCSSTUS1 (already in pipeline)
      Relevance: buffer stock dampens energy/breakeven channels of GIPI
      Exclusion: policy-determined, not driven by current US IP growth

Estimation: 2SLS via pure numpy (no statsmodels dependency).
  First stage:  GIPI_{t-1} = π·Z + γ·C + υ
  Second stage: y_{t+h}    = α + β·GPR + δ·Ĝ + ζ·(GPR×Ĝ) + controls + ε

All standard errors are HAC (Newey-West, bandwidth = floor(N^(1/3))).

Diagnostics
-----------
1. First-stage F-statistic on excluded instruments (Staiger-Stock: need F>10)
2. Sargan-Hansen J-test for overidentification (chi²(1), want p>0.10)
3. Durbin-Wu-Hausman endogeneity test (t-test on first-stage residual)
4. OLS vs 2SLS coefficient comparison

Decision rule (automated verdict)
----------------------------------
F < 10              → WEAK — frame as forecasting paper
F ≥ 10, J p<0.05   → INVALID instruments — exclusion likely violated
F ≥ 10, J p≥0.05,
  DWH p > 0.10     → EXOGENOUS — OLS preferred; endogeneity minor
F ≥ 10, J p≥0.05,
  DWH p ≤ 0.10     → IV VALID AND NEEDED — 2SLS as primary
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    from sklearn.preprocessing import StandardScaler
    _HAS_SKL = True
except ImportError:
    _HAS_SKL = False


# ─────────────────────────────────────────────────────────────────────────────
def _std(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def _ols_hac(Y: np.ndarray, X: np.ndarray, bw: int | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    OLS with Newey-West HAC standard errors.
    Returns (coefs, HAC_se, residuals).
    bw defaults to floor(N^(1/3)).
    """
    Xc  = np.column_stack([np.ones(len(Y)), X])
    b   = np.linalg.lstsq(Xc, Y, rcond=None)[0]
    e   = Y - Xc @ b
    bw_ = bw or max(1, int(np.floor(len(Y) ** (1/3))))
    xe  = Xc * e[:, None]
    S   = xe.T @ xe
    for lag in range(1, bw_ + 1):
        w  = 1 - lag / (bw_ + 1)
        Sc = xe[lag:].T @ xe[:-lag]
        S += w * (Sc + Sc.T)
    try:
        Xxi = np.linalg.inv(Xc.T @ Xc)
    except np.linalg.LinAlgError:
        Xxi = np.linalg.pinv(Xc.T @ Xc)  # fallback for near-singular
    se  = np.sqrt(np.maximum(np.diag(Xxi @ S @ Xxi), 0))
    return b, se, e


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class IVResult:
    horizon:           int
    n_obs:             int   = 0
    real_instruments:  int   = 0   # 0/1/2 real data instruments available

    # First stage
    first_stage_f:     float = np.nan
    first_stage_pval:  float = np.nan
    partial_r2_z1:     float = np.nan
    partial_r2_z2:     float = np.nan

    # Overidentification
    sargan_stat:       float = np.nan
    sargan_pval:       float = np.nan

    # Endogeneity
    hausman_stat:      float = np.nan
    hausman_pval:      float = np.nan

    # Coefficients
    gipi_coef_ols:     float = np.nan
    gipi_coef_iv:      float = np.nan
    gipi_se_ols:       float = np.nan
    gipi_se_iv:        float = np.nan

    verdict:           str   = ""


# ─────────────────────────────────────────────────────────────────────────────
class GIPIInstrumentalVariables:
    """
    2SLS robustness check for GIPI endogeneity.

    Usage
    -----
    iv = GIPIInstrumentalVariables(df)
    results = iv.run(horizons=[3, 6, 12])
    iv.summary()
    iv.save("outputs/iv_results.csv")
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        outcome:    str = "ip_yoy",
        gpr_col:    str = "gpr_z",
        gipi_col:   str = "gipi",
        spr_col:    str = "spr_stocks",
        opec_col:   str = "opec_spare_cap",
        lag_opec:   int = 6,
        lag_spr:    int = 1,
    ):
        self.df       = df.copy()
        self.outcome  = outcome
        self.gpr_col  = gpr_col
        self.gipi_col = gipi_col
        self.spr_col  = spr_col
        self.opec_col = opec_col
        self.lag_opec = lag_opec
        self.lag_spr  = lag_spr
        self._results: dict[int, IVResult] = {}

    # ──────────────────────────────────────────────────────────────────────────
    def run(self, horizons: list[int] | None = None) -> dict[int, IVResult]:
        horizons = horizons or [3, 6, 12]
        df = self._build_instruments(self.df.copy())

        for h in horizons:
            r = self._run_horizon(df, h)
            self._results[h] = r
            log.info(
                f"IV h={h}: F={r.first_stage_f:.1f}  "
                f"Sargan-p={r.sargan_pval:.3f}  "
                f"DWH-p={r.hausman_pval:.3f}  "
                f"real_instruments={r.real_instruments}/2  "
                f"→ {r.verdict}"
            )
        return self._results

    # ──────────────────────────────────────────────────────────────────────────
    def _build_instruments(self, df: pd.DataFrame) -> pd.DataFrame:
        real = 0

        # Z1: OPEC spare capacity
        if self.opec_col in df.columns and df[self.opec_col].notna().sum() >= 60:
            df["z1"] = _std(df[self.opec_col].shift(self.lag_opec))
            real += 1
            log.info(f"IV Z1: real OPEC spare cap (lag {self.lag_opec}m)")
        else:
            # Synthetic fallback for testing: inverse energy price
            log.warning("IV Z1: OPEC spare cap not found — synthetic fallback")
            if "global_energy_yoy" in df.columns:
                df["z1"] = _std(-df["global_energy_yoy"].shift(self.lag_opec))
            else:
                df["z1"] = pd.Series(
                    np.random.default_rng(1).standard_normal(len(df)), index=df.index)

        # Z2: US SPR stocks
        if self.spr_col in df.columns and df[self.spr_col].notna().sum() >= 60:
            df["z2"] = _std(df[self.spr_col].shift(self.lag_spr))
            real += 1
            log.info(f"IV Z2: real SPR stocks (lag {self.lag_spr}m)")
        else:
            log.warning("IV Z2: SPR stocks not found — synthetic fallback")
            df["z2"] = pd.Series(
                np.random.default_rng(2).standard_normal(len(df)), index=df.index)

        df.attrs["real_instruments"] = real
        return df

    # ──────────────────────────────────────────────────────────────────────────
    def _run_horizon(self, df: pd.DataFrame, h: int) -> IVResult:
        r = IVResult(horizon=h,
                     real_instruments=df.attrs.get("real_instruments", 0))

        from models.quantile_risk import build_fci
        df = df.copy()
        df["fci"] = build_fci(df)

        # Build estimation sample ─────────────────────────────────────────────
        cols = {
            "y":    df[self.outcome].shift(-h),
            "G":    df[self.gipi_col].shift(1),   # GIPI_{t-1} (endogenous)
            "z1":   df["z1"],
            "z2":   df["z2"],
            "c1":   df[self.gpr_col],              # controls
            "c2":   df["fci"],
            "c3":   df[self.outcome],              # lagged IP
        }
        panel = pd.DataFrame(cols).dropna()
        N = len(panel)
        r.n_obs = N
        if N < 60:
            r.verdict = f"too few obs ({N})"
            return r

        Y  = panel["y"].values
        G  = _std(panel["G"].values)
        Z1 = panel["z1"].values
        Z2 = panel["z2"].values
        C  = np.column_stack([_std(panel[c].values) for c in ["c1","c2","c3"]])
        bw = max(1, int(np.floor(N ** (1/3))))

        # ── First stage ───────────────────────────────────────────────────────
        Xfs = np.column_stack([Z1, Z2, C])
        b_fs, _, e_fs = _ols_hac(G, Xfs, bw)
        G_hat = np.column_stack([np.ones(N), Xfs]) @ b_fs

        # F on Z1, Z2 (restricted = controls only)
        b_r, _, e_r = _ols_hac(G, C, bw)
        rss_u = float((e_fs**2).sum())
        rss_r = float((e_r**2).sum())
        k     = 1 + Xfs.shape[1]   # constant + all regressors
        f_stat = ((rss_r - rss_u) / 2) / (rss_u / (N - k))
        r.first_stage_f    = round(float(f_stat), 2)
        r.first_stage_pval = round(float(1 - stats.f.cdf(f_stat, 2, N - k)), 4)

        # Partial R² for each instrument
        for zi, attr in [(Z1,"partial_r2_z1"), (Z2,"partial_r2_z2")]:
            other_z = Z2 if attr == "partial_r2_z1" else Z1
            Xp  = np.column_stack([other_z, C])
            g_r = _ols_hac(G,  Xp, bw)[2]
            z_r = _ols_hac(zi, Xp, bw)[2]
            c_  = np.corrcoef(g_r, z_r)[0,1]
            setattr(r, attr, round(float(c_**2), 4))

        # ── Second stage ──────────────────────────────────────────────────────
        G_hat_v = (np.column_stack([np.ones(N), Xfs]) @ b_fs)
        X2s     = np.column_stack([G_hat_v, C])
        b_2s, se_2s, e_2s = _ols_hac(Y, X2s, bw)

        # OLS naive (use actual G)
        b_ols, se_ols, e_ols = _ols_hac(Y, np.column_stack([G, C]), bw)

        r.gipi_coef_ols = round(float(b_ols[1]), 4)
        r.gipi_coef_iv  = round(float(b_2s[1]), 4)
        r.gipi_se_ols   = round(float(se_ols[1]), 4)
        r.gipi_se_iv    = round(float(se_2s[1]), 4)

        # ── Sargan-Hansen J-test ──────────────────────────────────────────────
        # Regress 2SLS residuals on all instruments + controls
        Xs = np.column_stack([Z1, Z2, C])
        b_j, _, _ = _ols_hac(e_2s, Xs, bw)
        y_j = np.column_stack([np.ones(N), Xs]) @ b_j
        ss_res = ((e_2s - y_j)**2).sum()
        ss_tot = ((e_2s - e_2s.mean())**2).sum()
        r2_j   = float(max(1 - ss_res/ss_tot, 0))
        J      = N * r2_j
        r.sargan_stat = round(J, 3)
        r.sargan_pval = round(float(1 - stats.chi2.cdf(J, 1)), 4)  # df=2-1=1

        # ── Durbin-Wu-Hausman ─────────────────────────────────────────────────
        X_dwh = np.column_stack([G, e_fs, C])
        b_dwh, se_dwh, _ = _ols_hac(Y, X_dwh, bw)
        t_h  = b_dwh[2] / se_dwh[2]   # coef on first-stage residual
        p_h  = float(2 * (1 - stats.t.cdf(abs(t_h), N - X_dwh.shape[1] - 1)))
        r.hausman_stat = round(float(t_h), 3)
        r.hausman_pval = round(p_h, 4)

        # ── Verdict ───────────────────────────────────────────────────────────
        ri = r.real_instruments
        F  = r.first_stage_f
        Jp = r.sargan_pval
        Hp = r.hausman_pval

        if ri < 2:
            r.verdict = "SYNTHETIC DATA — rerun with real OPEC/SPR data"
        elif F < 5:
            r.verdict = "WEAK INSTRUMENTS (F<5) — frame as forecasting paper"
        elif F < 10:
            r.verdict = "BORDERLINE WEAK (5≤F<10) — results indicative only"
        elif Jp < 0.05:
            r.verdict = "INSTRUMENTS INVALID — exclusion restriction likely violated"
        elif Hp > 0.10:
            r.verdict = "EXOGENOUS (DWH p>0.10) — OLS preferred; endogeneity minor"
        else:
            r.verdict = "IV VALID AND NEEDED (F≥10, Sargan ok, DWH rejects exogeneity)"

        return r

    # ──────────────────────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        rows = []
        for h, r in sorted(self._results.items()):
            rows.append({
                "h":              h,
                "N":              r.n_obs,
                "real_instr":     r.real_instruments,
                "F_stat":         r.first_stage_f,
                "F_pval":         r.first_stage_pval,
                "pR2_Z1(OPEC)":   r.partial_r2_z1,
                "pR2_Z2(SPR)":    r.partial_r2_z2,
                "Sargan_J":       r.sargan_stat,
                "Sargan_p":       r.sargan_pval,
                "DWH_t":          r.hausman_stat,
                "DWH_p":          r.hausman_pval,
                "GIPI_OLS":       r.gipi_coef_ols,
                "GIPI_2SLS":      r.gipi_coef_iv,
                "SE_OLS":         r.gipi_se_ols,
                "SE_2SLS":        r.gipi_se_iv,
                "verdict":        r.verdict,
            })
        return pd.DataFrame(rows)

    def save(self, path: str = "outputs/iv_results.csv") -> None:
        self.summary().to_csv(path, index=False)
        log.info(f"IV results → {path}")
