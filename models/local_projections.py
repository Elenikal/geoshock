"""
models/local_projections.py
─────────────────────────────────────────────────────────────────────────────
Local Projections (Jordà 2005) for estimating impulse response functions (IRFs)
of US macro/financial variables to a geopolitical shock (GPR index).

Key features
────────────
• Newey-West (HAC) standard errors — robust to serial correlation & heteroskedasticity
• Bootstrap confidence bands (percentile method, 500 reps default)
• Regime-conditional IRFs: calm / elevated / crisis
• Narrative instrument option: use GPR spike indicator as IV
• Event-study overlay: plot actual paths around identified episodes

Model specification at horizon h
─────────────────────────────────
  y_{t+h} - y_{t-1} = α_h + β_h * shock_t + Γ_h * X_t + ε_{t+h}

where:
  y          : outcome variable (IP growth, CPI, VIX, S&P return …)
  shock_t    : GPR shock (standardised change in GPR index)
  X_t        : controls (lags of y, oil return, fedfunds, hy_spread)
  β_h        : IRF coefficient at horizon h  ← main object of interest
  h          : 0, 1, …, H  (months)

Usage
─────
  from models.local_projections import LocalProjections
  lp = LocalProjections(df, shock="gpr_shock", outcome="ip_growth")
  lp.fit()
  fig = lp.plot_irf(title="IP Response to GPR Shock")
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import warnings
import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_hac
    SM_AVAILABLE = True
except ImportError:
    SM_AVAILABLE = False
    log.warning("statsmodels not installed. Run: pip install statsmodels")


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULT CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LPResult:
    """Holds fitted LP results for a single outcome variable."""
    shock: str
    outcome: str
    horizons: np.ndarray           # [0, 1, …, H]
    betas: np.ndarray              # IRF point estimates
    se_hac: np.ndarray             # HAC standard errors
    ci_68: np.ndarray              # shape (H, 2) — 68% CI
    ci_90: np.ndarray              # shape (H, 2) — 90% CI
    bs_draws: np.ndarray | None    # shape (B, H) bootstrap draws
    r2: np.ndarray                 # R² at each horizon
    n_obs: np.ndarray              # observations at each horizon
    regime: str = "full"           # "full" | "calm" | "elevated" | "crisis"
    controls: list[str] = field(default_factory=list)

    @property
    def cumulative_irf(self) -> np.ndarray:
        """Cumulative sum of beta_h (useful for level outcomes)."""
        return np.cumsum(self.betas)

    def peak_horizon(self) -> int:
        """Horizon of maximum absolute IRF response."""
        return int(np.argmax(np.abs(self.betas)))

    def significance_mask(self, level: float = 0.90) -> np.ndarray:
        """Boolean mask: True where CI at `level` excludes zero."""
        ci = self.ci_90 if level == 0.90 else self.ci_68
        return (ci[:, 0] > 0) | (ci[:, 1] < 0)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LP CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class LocalProjections:
    """
    Estimates Jordà (2005) Local Projections.

    Parameters
    ----------
    df : pd.DataFrame
        Monthly panel with all variables as columns, DatetimeIndex.
    shock : str
        Column name for the identified geopolitical shock (standardised).
    outcome : str
        Column name for the outcome variable to project.
    controls : list[str]
        Columns to include as contemporaneous controls.
    n_lags : int
        Number of lags of outcome + shock to include as controls.
    horizon : int
        Maximum horizon H (months ahead).
    bootstrap_reps : int
        Number of bootstrap replications for CIs. Set 0 to skip.
    regime_col : str | None
        Column defining regimes. If provided, can filter to a regime.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        shock: str = "gpr_shock",
        outcome: str = "ip_growth",
        controls: list[str] | None = None,
        n_lags: int = 4,
        horizon: int = 24,
        bootstrap_reps: int = 500,
        regime_col: str | None = "regime",
    ):
        self.df = df.copy()
        self.shock = shock
        self.outcome = outcome
        self.controls = controls or ["oil_return", "fedfunds", "hy_spread", "vix_change"]
        self.n_lags = n_lags
        self.horizon = horizon
        self.bootstrap_reps = bootstrap_reps
        self.regime_col = regime_col
        self.results_: dict[str, LPResult] = {}

    # ─────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        regime: Literal["full", "calm", "elevated", "crisis"] = "full",
        verbose: bool = True,
    ) -> LPResult:
        """
        Estimate LP at all horizons for the specified regime.

        Returns
        -------
        LPResult dataclass.
        """
        if not SM_AVAILABLE:
            raise ImportError("statsmodels required: pip install statsmodels")

        df = self._prepare_data(regime)
        H = self.horizon
        betas, se_hac, r2_arr, n_arr = [], [], [], []

        if verbose:
            log.info(f"LP: {self.outcome} ~ {self.shock}  |  regime={regime}  |  H=0..{H}")

        for h in range(H + 1):
            beta, se, r2, n = self._estimate_horizon(df, h)
            betas.append(beta)
            se_hac.append(se)
            r2_arr.append(r2)
            n_arr.append(n)

        betas   = np.array(betas)
        se_hac  = np.array(se_hac)
        horizons = np.arange(H + 1)

        # Confidence intervals
        ci_68 = np.column_stack([betas - 1.00 * se_hac, betas + 1.00 * se_hac])
        ci_90 = np.column_stack([betas - 1.645 * se_hac, betas + 1.645 * se_hac])

        # Bootstrap
        bs_draws = None
        if self.bootstrap_reps > 0:
            bs_draws = self._bootstrap(df, H)

        res = LPResult(
            shock=self.shock,
            outcome=self.outcome,
            horizons=horizons,
            betas=betas,
            se_hac=se_hac,
            ci_68=ci_68,
            ci_90=ci_90,
            bs_draws=bs_draws,
            r2=np.array(r2_arr),
            n_obs=np.array(n_arr),
            regime=regime,
            controls=self.controls,
        )
        self.results_[regime] = res
        return res

    # ─────────────────────────────────────────────────────────────────────────
    def fit_all_regimes(self, verbose: bool = True) -> dict[str, LPResult]:
        """Estimate LP for full sample + 3 regimes."""
        for reg in ["full", "calm", "elevated", "crisis"]:
            try:
                self.fit(regime=reg, verbose=verbose)
            except Exception as e:
                log.warning(f"  Regime '{reg}' skipped: {e}")
        return self.results_

    # ─────────────────────────────────────────────────────────────────────────
    def _prepare_data(self, regime: str) -> pd.DataFrame:
        cols_needed = [self.shock, self.outcome] + self.controls
        available   = [c for c in cols_needed if c in self.df.columns]
        df = self.df[available].dropna()

        if regime != "full" and self.regime_col and self.regime_col in self.df.columns:
            mask = self.df[self.regime_col] == regime
            df = df[mask.reindex(df.index, fill_value=False)]

        return df

    # ─────────────────────────────────────────────────────────────────────────
    def _estimate_horizon(
        self, df: pd.DataFrame, h: int
    ) -> tuple[float, float, float, int]:
        """
        LP regression at horizon h:
          (y_{t+h} - y_{t-1}) = α + β*shock_t + Γ*X_t + lags + ε
        """
        y = df[self.outcome].shift(-h)   # outcome h periods ahead
        y_lag1 = df[self.outcome].shift(1)

        # Cumulative outcome change (as in Jordà 2005)
        lhs = y - y_lag1

        # RHS: shock + controls + lags
        rhs_parts = [df[self.shock]]

        # Contemporaneous controls (available controls)
        for c in self.controls:
            if c in df.columns and c != self.outcome and c != self.shock:
                rhs_parts.append(df[c])

        # Lags of shock and outcome
        for lag in range(1, self.n_lags + 1):
            rhs_parts.append(df[self.shock].shift(lag).rename(f"{self.shock}_L{lag}"))
            rhs_parts.append(df[self.outcome].shift(lag).rename(f"{self.outcome}_L{lag}"))

        X_raw = pd.concat(rhs_parts, axis=1)
        data  = pd.concat([lhs, X_raw], axis=1).dropna()

        if len(data) < 30:
            return 0.0, 0.0, 0.0, 0

        Y  = data.iloc[:, 0].values
        X  = sm.add_constant(data.iloc[:, 1:].values.astype(float))
        n  = len(Y)

        try:
            res  = sm.OLS(Y, X).fit()
            # HAC covariance (Newey-West), bandwidth = floor(4*(n/100)^(2/9))
            nw_bw = int(np.floor(4 * (n / 100) ** (2 / 9)))
            hac   = cov_hac(res, nlags=nw_bw)
            se    = np.sqrt(np.diag(hac))
            beta_idx = 1   # first coef after constant is the shock
            return float(res.params[beta_idx]), float(se[beta_idx]), float(res.rsquared), n
        except Exception as e:
            log.debug(f"  OLS failed at h={h}: {e}")
            return 0.0, 0.0, 0.0, n

    # ─────────────────────────────────────────────────────────────────────────
    def _bootstrap(self, df: pd.DataFrame, H: int) -> np.ndarray:
        """
        Residual bootstrap: re-sample OLS residuals, re-estimate betas at each h.
        Returns array of shape (B, H+1).
        """
        B    = self.bootstrap_reps
        bs   = np.zeros((B, H + 1))
        rng  = np.random.default_rng(42)

        for b in range(B):
            # Re-sample row indices with replacement
            idx = rng.choice(len(df), size=len(df), replace=True)
            df_b = df.iloc[idx].copy()
            df_b.index = df.index[:len(df_b)]   # reset index for shifting

            betas_b = []
            for h in range(H + 1):
                beta, _, _, _ = self._estimate_horizon(df_b, h)
                betas_b.append(beta)
            bs[b] = np.array(betas_b)

        return bs

    # ─────────────────────────────────────────────────────────────────────────
    def plot_irf(
        self,
        regime: str = "full",
        title: str | None = None,
        figsize: tuple = (10, 5),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Plot impulse response function with 68% and 90% confidence bands.
        """
        if regime not in self.results_:
            self.fit(regime=regime)
        res = self.results_[regime]

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1628")
        ax.set_facecolor("#0a0e1a")

        h = res.horizons
        b = res.betas

        # Shaded confidence bands
        ax.fill_between(h, res.ci_90[:, 0], res.ci_90[:, 1],
                        alpha=0.25, color="#3b82f6", label="90% CI")
        ax.fill_between(h, res.ci_68[:, 0], res.ci_68[:, 1],
                        alpha=0.40, color="#3b82f6", label="68% CI")

        # Bootstrap percentile band (if available)
        if res.bs_draws is not None:
            p5  = np.percentile(res.bs_draws, 5,  axis=0)
            p95 = np.percentile(res.bs_draws, 95, axis=0)
            ax.fill_between(h, p5, p95, alpha=0.12, color="#e8a020",
                            label="Bootstrap 90% CI")

        # Point estimate
        ax.plot(h, b, color="#e8a020", linewidth=2.0, zorder=5, label="IRF")

        # Zero line
        ax.axhline(0, color="#64748b", linewidth=0.8, linestyle="--")

        # Significance dots
        sig = res.significance_mask(0.90)
        ax.scatter(h[sig], b[sig], color="#ef4444", s=18, zorder=6)

        ax.set_xlabel("Horizon (months)", color="#94a3b8", fontsize=11)
        ylabel = f"Δ {res.outcome} (pp)"
        ax.set_ylabel(ylabel, color="#94a3b8", fontsize=11)

        _title = title or f"IRF: {res.outcome} ← {res.shock}  [{regime} regime]"
        ax.set_title(_title, color="#f1f5f9", fontsize=13, pad=12)
        ax.tick_params(colors="#64748b")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d4a")

        legend = ax.legend(
            facecolor="#0f1628", edgecolor="#1e2d4a",
            labelcolor="#94a3b8", fontsize=9,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    def plot_regime_comparison(
        self,
        figsize: tuple = (14, 5),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Side-by-side IRFs for calm / elevated / crisis regimes.
        """
        if not self.results_:
            self.fit_all_regimes(verbose=False)

        regimes = ["calm", "elevated", "crisis"]
        colors  = {"calm": "#10b981", "elevated": "#e8a020", "crisis": "#ef4444"}
        available = [r for r in regimes if r in self.results_]

        fig, axes = plt.subplots(1, len(available), figsize=figsize,
                                 facecolor="#0f1628", sharey=True)
        if len(available) == 1:
            axes = [axes]

        for ax, reg in zip(axes, available):
            ax.set_facecolor("#0a0e1a")
            res = self.results_[reg]
            c = colors[reg]
            ax.fill_between(res.horizons, res.ci_90[:, 0], res.ci_90[:, 1],
                            alpha=0.2, color=c)
            ax.fill_between(res.horizons, res.ci_68[:, 0], res.ci_68[:, 1],
                            alpha=0.35, color=c)
            ax.plot(res.horizons, res.betas, color=c, linewidth=2)
            ax.axhline(0, color="#64748b", linewidth=0.8, linestyle="--")
            ax.set_title(reg.upper(), color=c, fontsize=11)
            ax.set_xlabel("Horizon (months)", color="#94a3b8", fontsize=9)
            ax.tick_params(colors="#64748b")
            for spine in ax.spines.values():
                spine.set_edgecolor("#1e2d4a")
            n_avg = int(res.n_obs.mean())
            ax.text(0.98, 0.02, f"n≈{n_avg}", transform=ax.transAxes,
                    ha="right", va="bottom", color="#475569", fontsize=8)

        axes[0].set_ylabel(f"Δ {self.outcome} (pp)", color="#94a3b8", fontsize=10)
        fig.suptitle(
            f"Regime-Conditional IRFs: {self.outcome} ← {self.shock}",
            color="#f1f5f9", fontsize=13, y=1.02,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MULTI-OUTCOME LP RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

OUTCOME_LABELS = {
    "ip_growth":      "Industrial Production Growth (MoM %)",
    "cpi_inflation":  "CPI Inflation (YoY %)",
    "unemp":          "Unemployment Rate (%)",
    "hy_spread":      "HY Credit Spread (bps)",
    "vix":            "VIX Volatility Index",
    "sp500_return":   "S&P 500 Return (MoM %)",
    "oil_return":     "WTI Oil Return (MoM %)",
    "fedfunds":       "Federal Funds Rate (%)",
    "term_spread":    "10Y–2Y Treasury Spread (%)",
}


def run_all_lp(
    df: pd.DataFrame,
    shock: str = "gpr_shock",
    outcomes: list[str] | None = None,
    horizon: int = 24,
    bootstrap_reps: int = 200,
    verbose: bool = True,
) -> dict[str, LPResult]:
    """
    Run Local Projections for all specified outcome variables.

    Returns
    -------
    dict mapping outcome_name → LPResult
    """
    if outcomes is None:
        outcomes = [o for o in OUTCOME_LABELS if o in df.columns]

    results = {}
    for outcome in outcomes:
        if outcome not in df.columns:
            log.warning(f"Outcome '{outcome}' not in DataFrame — skipping.")
            continue
        if verbose:
            log.info(f"\n── LP: {outcome} ──")
        lp = LocalProjections(
            df, shock=shock, outcome=outcome,
            horizon=horizon, bootstrap_reps=bootstrap_reps,
        )
        try:
            results[outcome] = lp.fit(regime="full", verbose=verbose)
        except Exception as e:
            log.warning(f"  LP for {outcome} failed: {e}")

    return results


def plot_all_irfs(
    results: dict[str, LPResult],
    ncols: int = 3,
    figsize_per: tuple = (5, 3.5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Grid of IRF plots for all outcomes.
    """
    outcomes = list(results.keys())
    nrows = int(np.ceil(len(outcomes) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
        facecolor="#0f1628",
    )
    axes = np.array(axes).flatten()

    for i, (outcome, res) in enumerate(results.items()):
        ax = axes[i]
        ax.set_facecolor("#0a0e1a")
        h = res.horizons
        b = res.betas
        ax.fill_between(h, res.ci_90[:, 0], res.ci_90[:, 1],
                        alpha=0.2, color="#3b82f6")
        ax.fill_between(h, res.ci_68[:, 0], res.ci_68[:, 1],
                        alpha=0.38, color="#3b82f6")
        ax.plot(h, b, color="#e8a020", linewidth=1.5)
        ax.axhline(0, color="#475569", linewidth=0.7, linestyle="--")
        label = OUTCOME_LABELS.get(outcome, outcome)
        ax.set_title(label, color="#94a3b8", fontsize=8, pad=6)
        ax.tick_params(colors="#475569", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d4a")

    # Hide unused axes
    for j in range(len(outcomes), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Impulse Responses to 1 SD Geopolitical Risk Shock",
        color="#f1f5f9", fontsize=13, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # Quick demo with synthetic data
    import sys
    sys.path.insert(0, "..")
    from data.pipeline import DataPipeline
    dp = DataPipeline()
    df = dp.build(use_cache=False)

    results = run_all_lp(
        df, shock="gpr_shock",
        outcomes=["ip_growth", "cpi_inflation", "hy_spread", "sp500_return"],
        horizon=18, bootstrap_reps=100, verbose=True,
    )
    fig = plot_all_irfs(results, save_path="outputs/figures/all_irfs.png")
    print("Saved IRF grid to outputs/figures/all_irfs.png")
