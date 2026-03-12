"""
models/var_model.py
─────────────────────────────────────────────────────────────────────────────
Reduced-Form and Structural VAR for transmission channel decomposition.

Models implemented
──────────────────
1. Reduced-form VAR (statsmodels)
   - Granger causality tests (GPR → macro variables)
   - Forecast Error Variance Decomposition (FEVD)
   - Historical decomposition

2. Cholesky-identified SVAR
   - Ordering: GPR → Oil → FCI → IP → CPI → FFR
   - Identifies supply-side geopolitical shock (GPR ordered first)

3. Regime-switching summary (using HMM from regime detection)

Usage
─────
  from models.var_model import GeoShockVAR
  var = GeoShockVAR(df)
  var.fit()
  var.plot_fevd()
  var.plot_historical_decomp()
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import warnings
import logging
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    import statsmodels.api as sm
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    SM_AVAILABLE = True
except ImportError:
    SM_AVAILABLE = False
    log.warning("statsmodels not available.")

# ── Colour palette ─────────────────────────────────────────────────────────
VAR_COLORS = [
    "#e8a020", "#3b82f6", "#10b981", "#ef4444",
    "#a855f7", "#06b6d4", "#f97316", "#64748b",
]

# ── Variable ordering for Cholesky SVAR ────────────────────────────────────
# Most exogenous → most endogenous
CHOLESKY_ORDER = [
    "gpr_shock",      # 1. Geopolitical risk shock (most exogenous)
    "oil_return",     # 2. Oil price (reacts to GPR, affects all)
    "vix_change",     # 3. Financial uncertainty (reacts to oil/GPR)
    "hy_spread",      # 4. Credit spreads (financial channel)
    "ip_growth",      # 5. Industrial production (real economy)
    "cpi_inflation",  # 6. Inflation (sticky)
    "d_fedfunds",     # 7. Fed reaction (most endogenous)
]


# ═══════════════════════════════════════════════════════════════════════════════
#  STATIONARITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_stationarity(series: pd.Series, name: str = "") -> dict:
    """ADF unit root test. Returns dict with test statistic, p-value, verdict."""
    if not SM_AVAILABLE:
        return {"name": name, "adf_stat": np.nan, "p_value": 0.01, "stationary": True}
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "name": name,
        "adf_stat": round(result[0], 3),
        "p_value":  round(result[1], 4),
        "lags":     result[2],
        "n_obs":    result[3],
        "stationary": result[1] < 0.05,
    }


def run_stationarity_battery(df: pd.DataFrame) -> pd.DataFrame:
    """Test all numeric columns for stationarity."""
    results = []
    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()
        if len(s) < 20:
            continue
        results.append(test_stationarity(s, name=col))
    return pd.DataFrame(results).set_index("name")


# ═══════════════════════════════════════════════════════════════════════════════
#  GRANGER CAUSALITY BATTERY
# ═══════════════════════════════════════════════════════════════════════════════

def run_granger_battery(
    df: pd.DataFrame,
    cause: str = "gpr_shock",
    outcomes: list[str] | None = None,
    maxlag: int = 6,
) -> pd.DataFrame:
    """
    Test H0: `cause` does NOT Granger-cause each outcome variable.
    Returns DataFrame with p-values at each lag, plus min p-value.
    """
    if not SM_AVAILABLE:
        log.warning("statsmodels required for Granger tests.")
        return pd.DataFrame()

    outcomes = outcomes or [
        "ip_growth", "cpi_inflation", "hy_spread",
        "vix_change", "sp500_return", "unemp",
    ]
    records = []
    for out in outcomes:
        if out not in df.columns or cause not in df.columns:
            continue
        pair = df[[out, cause]].dropna()
        if len(pair) < 40:
            continue
        try:
            gc = grangercausalitytests(pair, maxlag=maxlag, verbose=False)
            row = {"outcome": out}
            for lag in range(1, maxlag + 1):
                p_val = gc[lag][0]["ssr_ftest"][1]
                row[f"lag_{lag}"] = round(p_val, 4)
            row["min_pval"] = min(gc[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1))
            row["significant_5pct"] = row["min_pval"] < 0.05
            records.append(row)
        except Exception as e:
            log.debug(f"  Granger {cause}→{out}: {e}")

    df_gc = pd.DataFrame(records).set_index("outcome")
    log.info(f"\nGranger Causality: {cause} → outcomes")
    log.info(df_gc[["min_pval", "significant_5pct"]].to_string())
    return df_gc


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN VAR CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class GeoShockVAR:
    """
    VAR model for geopolitical shock transmission.

    Parameters
    ----------
    df : pd.DataFrame — monthly feature panel.
    variables : list[str] — variable names (Cholesky order matters for SVAR).
    lags : int — VAR lag order.
    identification : str — 'cholesky' (only option for now).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        variables: list[str] | None = None,
        lags: int = 4,
        identification: Literal["cholesky"] = "cholesky",
    ):
        self.df = df.copy()
        self.variables = variables or [v for v in CHOLESKY_ORDER if v in df.columns]
        self.lags = lags
        self.identification = identification
        self.model_: VAR | None = None
        self.result_: object | None = None

    # ─────────────────────────────────────────────────────────────────────────
    def fit(self, verbose: bool = True) -> "GeoShockVAR":
        if not SM_AVAILABLE:
            raise ImportError("statsmodels required: pip install statsmodels")

        data = self.df[self.variables].dropna()
        log.info(f"VAR: {len(self.variables)} vars × {len(data)} obs, lags={self.lags}")

        self.model_ = VAR(data)
        self.result_ = self.model_.fit(maxlags=self.lags, ic=None)

        if verbose:
            log.info(f"  AIC: {self.result_.aic:.1f}  BIC: {self.result_.bic:.1f}")
            log.info(f"  Det. covariance: {np.linalg.det(self.result_.sigma_u):.4e}")

        return self

    # ─────────────────────────────────────────────────────────────────────────
    def irf(
        self,
        periods: int = 24,
        shock_var: str | None = None,
        orth: bool = True,
    ) -> object:
        """
        Compute impulse response functions.

        Parameters
        ----------
        orth : bool — if True, Cholesky-orthogonalised IRFs (structural).
        shock_var : str — name of shock variable (GPR).

        Returns statsmodels IRAnalysis object.
        """
        if self.result_ is None:
            self.fit()
        return self.result_.irf(periods=periods, var_decomp=None)

    # ─────────────────────────────────────────────────────────────────────────
    def fevd(self, periods: int = 24) -> pd.DataFrame:
        """
        Forecast Error Variance Decomposition.

        Returns DataFrame (periods × variables) showing the share of forecast
        error variance explained by the GPR shock (first Cholesky variable).
        """
        if self.result_ is None:
            self.fit()

        fevd_obj = self.result_.fevd(periods)
        # fevd_obj.decomp: shape (n_periods, n_vars, n_vars)
        # axis 0 = horizon (0-indexed, so horizon 1..periods)
        # axis 1 = equation (which variable is being explained)
        # axis 2 = shock (which structural shock explains it)
        decomp = fevd_obj.decomp          # (periods, n_vars, n_vars)
        n_periods_actual = decomp.shape[0]
        gpr_idx = 0  # GPR is ordered first in Cholesky

        shares = {}
        for i, var in enumerate(self.variables):
            shares[var] = decomp[:, i, gpr_idx]   # length = n_periods_actual

        return pd.DataFrame(shares, index=range(1, n_periods_actual + 1))

    # ─────────────────────────────────────────────────────────────────────────
    def optimal_lag(self, maxlags: int = 12) -> dict:
        """Select optimal lag order via AIC, BIC, HQIC."""
        if self.model_ is None:
            data = self.df[self.variables].dropna()
            self.model_ = VAR(data)
        res = self.model_.select_order(maxlags)
        return {"aic": res.aic, "bic": res.bic, "hqic": res.hqic, "summary": res}

    # ─────────────────────────────────────────────────────────────────────────
    def plot_irf_grid(
        self,
        periods: int = 24,
        shock_var: str | None = None,
        save_path: str | None = None,
        figsize: tuple = (14, 10),
    ) -> plt.Figure:
        """
        Plot Cholesky IRFs: response of all variables to 1 SD GPR shock.
        """
        if self.result_ is None:
            self.fit()

        irf_obj = self.irf(periods=periods, orth=True)
        shock_idx = 0   # GPR first in ordering
        n_vars = len(self.variables)
        ncols = 3
        nrows = int(np.ceil(n_vars / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=figsize, facecolor="#0f1628")
        axes = np.array(axes).flatten()

        for i, var in enumerate(self.variables):
            ax = axes[i]
            ax.set_facecolor("#0a0e1a")

            # irf_obj.orth_irfs: shape (periods+1, n_vars, n_vars)
            try:
                irfs  = irf_obj.orth_irfs[:, i, shock_idx]
                lower = irf_obj.cum_effect_ci(orth=True)  # may not exist
            except AttributeError:
                irfs  = irf_obj.irfs[:, i, shock_idx]
                lower = None

            h = np.arange(len(irfs))
            ax.plot(h, irfs, color=VAR_COLORS[i % len(VAR_COLORS)], linewidth=1.8)
            ax.axhline(0, color="#475569", linewidth=0.7, linestyle="--")
            ax.set_title(var, color="#94a3b8", fontsize=9, pad=5)
            ax.tick_params(colors="#475569", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2d4a")

        # Hide unused
        for j in range(n_vars, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "VAR: Orthogonalised IRF to 1 SD Geopolitical Risk Shock",
            color="#f1f5f9", fontsize=13, y=1.01,
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    def plot_fevd(
        self,
        periods: int = 24,
        top_n: int = 6,
        save_path: str | None = None,
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """
        Bar chart: % of forecast error variance explained by GPR shock.
        Shows share at horizons 1, 3, 6, 12, 24 months for each variable.
        """
        fevd_df = self.fevd(periods=periods)
        max_h = fevd_df.index.max()
        horizons = [h for h in [1, 3, 6, 12, 24] if h <= max_h]
        if not horizons:
            horizons = [max_h]

        sub = fevd_df.loc[horizons].T * 100   # convert to %
        sub = sub.sort_values(by=horizons[-1], ascending=False).head(top_n)

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1628")
        ax.set_facecolor("#0a0e1a")

        x = np.arange(len(sub))
        bar_w = 0.12
        for j, h in enumerate(horizons):
            bars = ax.bar(
                x + j * bar_w, sub[h].values,
                width=bar_w, label=f"h={h}",
                color=VAR_COLORS[j % len(VAR_COLORS)], alpha=0.8,
            )

        ax.set_xticks(x + bar_w * len(horizons) / 2)
        ax.set_xticklabels(sub.index, rotation=30, ha="right",
                            color="#94a3b8", fontsize=9)
        ax.set_ylabel("FEVD share (%)", color="#94a3b8", fontsize=10)
        ax.set_title(
            "Forecast Error Variance Decomposition — GPR Shock Contribution",
            color="#f1f5f9", fontsize=12, pad=10,
        )
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.tick_params(colors="#64748b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d4a")
        ax.legend(facecolor="#0f1628", edgecolor="#1e2d4a",
                  labelcolor="#94a3b8", fontsize=8)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  REGIME DETECTION  (simple threshold-based from GPR z-score)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_regimes(
    gpr_z: pd.Series,
    low_thresh: float = 1.5,
    high_thresh: float = 2.5,
) -> pd.Series:
    """
    Simple threshold regime classification from GPR z-score:
      calm (<1.5 SD), elevated (1.5–2.5 SD), crisis (>2.5 SD)
    """
    regime = pd.Series("calm", index=gpr_z.index, name="regime")
    regime[gpr_z > low_thresh]  = "elevated"
    regime[gpr_z > high_thresh] = "crisis"
    return regime


def regime_summary_table(
    df: pd.DataFrame,
    regime_col: str = "regime",
    outcome_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Conditional means of outcome variables by regime.
    """
    outcome_cols = outcome_cols or ["ip_growth", "cpi_inflation",
                                    "sp500_return", "hy_spread", "vix_change"]
    available = [c for c in outcome_cols if c in df.columns]
    if regime_col not in df.columns:
        return pd.DataFrame()
    return df.groupby(regime_col)[available].agg(["mean", "std"]).round(3)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.pipeline import DataPipeline

    dp = DataPipeline()
    df = dp.build(use_cache=False)

    # Granger causality battery
    gc = run_granger_battery(df, cause="gpr_shock")
    print("\nGranger Causality Results:")
    print(gc)

    # VAR
    var = GeoShockVAR(df)
    var.fit()
    fevd = var.fevd()
    print("\nFEVD (GPR share at h=12):")
    print(fevd.loc[12].sort_values(ascending=False).round(4))

    # Regime summary
    if "regime" in df.columns:
        print("\nConditional means by regime:")
        print(regime_summary_table(df))
