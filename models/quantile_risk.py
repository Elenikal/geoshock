"""
models/quantile_risk.py
─────────────────────────────────────────────────────────────────────────────
Growth-at-Risk (GaR) and Tail Risk Forecasting Module.

Based on Adrian, Boyarchenko & Giannone (2019) "Vulnerable Growth" (AER).

The model estimates quantile regressions of h-quarter-ahead GDP growth
on current financial conditions and the geopolitical risk index, producing
a full predictive distribution at each forecast horizon.

Key outputs
───────────
1. Predictive quantile fan chart (the "GaR fan")
2. GaR_{5%}  — the 5th percentile of the 1-year-ahead GDP growth distribution
3. Conditional skewness & tail probability (P[GDP < 0], P[GDP < -2%])
4. Decomposition: how much of current tail risk comes from GPR vs FCI

Specification
─────────────
  Q_τ(Δy_{t+h} | x_t) = α_τ + β_τ * GPR_t + γ_τ * FCI_t + δ_τ * Δy_t

where FCI = financial conditions index (PCA of VIX, HY spread, term spread).
τ ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}

Usage
─────
  from models.quantile_risk import GrowthAtRisk
  gar = GrowthAtRisk(df)
  gar.fit(horizon=4)
  gar.plot_fan_chart(current_date="2024-10-01")
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
import matplotlib.colors as mcolors
from scipy.stats import skewnorm, norm

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

try:
    from sklearn.linear_model import QuantileRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKL_AVAILABLE = True
except ImportError:
    SKL_AVAILABLE = False
    log.warning("scikit-learn not installed. Run: pip install scikit-learn")

try:
    import statsmodels.api as sm
    from statsmodels.regression.quantile_regression import QuantReg
    SM_AVAILABLE = True
except ImportError:
    SM_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  FINANCIAL CONDITIONS INDEX (FCI)
# ═══════════════════════════════════════════════════════════════════════════════

FCI_COMPONENTS = ["vix_log", "hy_spread", "term_spread", "d_fedfunds", "sp500_return"]

# Extended fallback list — alternative column names that may exist in the dataframe
FCI_FALLBACKS = [
    "vix", "vix_change",          # alternatives to vix_log
    "hy_spread", "d_hy_spread",   # HY spread
    "term_spread", "gs10",        # term spread alternatives
    "fedfunds", "d_fedfunds",     # fed funds
    "sp500_return",               # equity
    "oil_return",                 # energy as financial condition proxy
]


def build_fci(
    df: pd.DataFrame,
    components: list[str] | None = None,
    n_components: int = 1,
    name: str = "fci",
) -> pd.Series:
    """
    Construct a Financial Conditions Index via PCA.
    Uses extended fallback list so it works even when primary cols are missing.
    With only 1 valid component, returns that component z-scored.
    """
    # Try primary components first, then extended fallbacks
    primary   = components or FCI_COMPONENTS
    available = [c for c in primary if c in df.columns and df[c].notna().sum() >= 12]

    if len(available) < 2:
        # Try fallbacks
        available = [c for c in FCI_FALLBACKS
                     if c in df.columns and df[c].notna().sum() >= 12
                     and c not in available][:5]

    if len(available) == 0:
        log.warning("FCI: no valid components found — returning zeros.")
        return pd.Series(np.zeros(len(df)), index=df.index, name=name)

    if len(available) == 1:
        log.warning(f"FCI: only 1 component ({available[0]}) — returning z-score.")
        s = df[available[0]].dropna()
        z = (s - s.mean()) / (s.std() + 1e-9)
        return z.reindex(df.index).fillna(0).rename(name)

    if not SKL_AVAILABLE:
        scaler_fn = lambda x: (x - x.mean()) / (x.std() + 1e-9)
        return df[available].apply(scaler_fn).mean(axis=1).rename(name)

    sub = df[available].dropna()
    if len(sub) < 10:
        log.warning("FCI: insufficient rows after dropna — returning zeros.")
        return pd.Series(np.zeros(len(df)), index=df.index, name=name)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(sub.values)

    pca = PCA(n_components=min(n_components, len(available)))
    pca.fit(X_scaled)
    fci_vals = pca.transform(X_scaled)[:, 0]

    # Flip sign so higher FCI = tighter conditions
    vix_col = next((c for c in ["vix_log", "vix", "vix_change"] if c in available), None)
    if vix_col:
        vix_loading = pca.components_[0, available.index(vix_col)]
        if vix_loading < 0:
            fci_vals = -fci_vals

    fci     = pd.Series(fci_vals, index=sub.index, name=name)
    explained = pca.explained_variance_ratio_[0] * 100
    log.info(f"FCI PCA: {explained:.1f}% variance explained  ({', '.join(available)})")
    return fci.reindex(df.index)


# ═══════════════════════════════════════════════════════════════════════════════
#  RESULT CONTAINER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GaRResult:
    """Stores fitted GaR model output."""
    horizon: int
    quantiles: list[float]
    # Estimated coefficients per quantile: dict[τ → dict[var → coef]]
    coefs: dict[float, dict[str, float]] = field(default_factory=dict)
    # In-sample fitted quantile paths: dict[τ → pd.Series]
    fitted: dict[float, pd.Series] = field(default_factory=dict)
    # Out-of-sample nowcast quantiles at T+h
    nowcast: dict[float, float] = field(default_factory=dict)
    # Tail risk metrics
    prob_neg_growth: float = 0.0          # P(GDP < 0)
    prob_recession: float = 0.0           # P(GDP < -2%)
    gar_5: float = 0.0                    # 5th percentile forecast
    gar_25: float = 0.0                   # 25th percentile forecast
    median_forecast: float = 0.0
    conditional_skewness: float = 0.0     # Bowley quartile skewness
    # Inference (filled by GrowthAtRisk.bootstrap_inference)
    coef_se: dict[float, dict[str, float]] = field(default_factory=dict)
    coef_p:  dict[float, dict[str, float]] = field(default_factory=dict)
    y: pd.Series | None = None            # estimation-sample outcome y_{t+h}
    origin: pd.Timestamp | None = None    # date of x_T used for the nowcast
    crossing: bool = False                # nowcast quantiles needed rearrangement


def _block_bootstrap_qr_se(
    Y: np.ndarray, X: np.ndarray, quantiles: list[float],
    B: int = 500, block: int = 12, seed: int = 0,
) -> dict[float, np.ndarray]:
    """Moving-block (Künsch 1989) bootstrap SEs of quantile-regression slopes.
    X includes the constant in column 0; returned SEs exclude it."""
    rng = np.random.default_rng(seed)
    n = len(Y)
    block = min(block, n // 4)
    n_blocks = int(np.ceil(n / block))
    draws = {q: np.empty((B, X.shape[1] - 1)) for q in quantiles}
    for b in range(B):
        starts = rng.integers(0, n - block + 1, n_blocks)
        ix = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        for q in quantiles:
            try:
                draws[q][b] = QuantReg(Y[ix], X[ix]).fit(q=q, max_iter=5000).params[1:]
            except Exception:
                draws[q][b] = np.nan
    return {q: np.nanstd(d, axis=0, ddof=1) for q, d in draws.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN GROWTH-AT-RISK CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class GrowthAtRisk:
    """
    Growth-at-Risk model: full predictive distribution of GDP/IP growth.

    Enhanced specification (v2):
      Q_τ(y_{t+h} | x_t) = α_τ + β_τ·GPR_t + γ_τ·FCI_t
                           + δ_τ·GIPI_t + ζ_τ·(GPR_t × GIPI_t) + η_τ·y_t

    where GIPI = Geopolitical Inflation Pressure Index (PC1 of supply-chain,
    energy, food, import price, and breakeven inflation variables).

    The GPR×GIPI interaction tests whether high inflation-transmission pressure
    amplifies the tail-risk impact of geopolitical shocks.

    Parameters
    ----------
    df          : monthly feature panel (output of enhanced data pipeline).
    outcome     : GDP growth variable ('gdp_growth', 'ip_yoy', 'ip_growth').
    gpr_col     : GPR shock column name.
    gipi_col    : GIPI column name (None = skip GIPI).
    use_interaction : bool — include GPR × GIPI interaction term.
    quantiles   : quantiles to estimate.
    horizon     : default forecast horizon.
    """

    QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    def __init__(
        self,
        df: pd.DataFrame,
        outcome: str = "ip_yoy",
        gpr_col: str = "gpr_z",
        gipi_col: str | None = "gipi",
        use_interaction: bool = True,
        lag_gipi: bool = True,
        quantiles: list[float] | None = None,
        horizon: int = 6,
    ):
        self.df              = df.copy()
        self.outcome         = outcome
        self.gpr_col         = gpr_col
        self.gipi_col        = gipi_col if (gipi_col and gipi_col in df.columns) else None
        self.use_interaction = use_interaction and (self.gipi_col is not None)
        self.lag_gipi        = lag_gipi   # if True, use GIPI_{t-1} in interaction (C1 fix)
        self.quantiles       = quantiles or self.QUANTILES
        self.horizon         = horizon
        self.fci_            : pd.Series | None = None
        self.results_        : dict[int, GaRResult] = {}
        self._scaler         = StandardScaler() if SKL_AVAILABLE else None

        if self.gipi_col:
            log.info(f"GaR v2: GIPI enabled ({gipi_col}), lag_gipi={lag_gipi}, "
                     f"interaction={'on' if self.use_interaction else 'off'}")
        else:
            log.info("GaR v2: GIPI not found in dataframe — running baseline GPR+FCI spec")

    # ─────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        horizons: list[int] | None = None,
        verbose: bool = True,
        nowcast_overrides: dict[str, float] | None = None,
    ) -> dict[int, GaRResult]:
        """
        Fit quantile regressions at each horizon and evaluate the predictive
        quantiles at the latest month with complete predictors (x_T).

          Baseline: Q_τ(y_{t+h}|x_t) = α + β·GPR_t + γ·FCI_t + η·y_t
          Enhanced: … + δ·GIPI_{t-1} + ζ·(GPR_t × GIPI_{t-1})

        nowcast_overrides : optional {column: value} applied to x_T only
            (e.g. {"gpr_z": Layer-0 nowcast}); never enters estimation.
        """
        if not SM_AVAILABLE:
            raise ImportError("statsmodels required.")
        horizons = horizons or [self.horizon]

        self.fci_ = build_fci(self.df)
        df = self.df.copy()
        df["fci"] = self.fci_

        if self.gipi_col:
            gipi_raw = df[self.gipi_col].shift(1) if self.lag_gipi else df[self.gipi_col]
            df["gipi_reg"] = gipi_raw

        # Apply overrides to the nowcast origin before building the interaction,
        # standardising with moments that exclude the overridden value.
        origin_cols = [self.gpr_col, "fci", self.outcome] + (["gipi_reg"] if self.gipi_col else [])
        self.origin_ = df[origin_cols].dropna().index[-1]
        gpr_mu, gpr_sd = df[self.gpr_col].mean(), df[self.gpr_col].std()
        for col, val in (nowcast_overrides or {}).items():
            df.loc[self.origin_, col] = val

        if self.use_interaction:
            g = df["gipi_reg"]
            df["gpr_x_gipi"] = ((df[self.gpr_col] - gpr_mu) / gpr_sd) * ((g - g.mean()) / g.std())

        self._design: dict[int, tuple] = {}
        for h in horizons:
            self.results_[h] = self._fit_horizon(df, h, verbose)
        return self.results_

    # ─────────────────────────────────────────────────────────────────────────
    def pred_cols_(self) -> list[str]:
        cols = [self.gpr_col, "fci"]
        if self.gipi_col:
            cols.append("gipi_reg")
        if self.use_interaction:
            cols.append("gpr_x_gipi")
        return cols + [self.outcome]

    def _fit_horizon(self, df: pd.DataFrame, h: int, verbose: bool) -> GaRResult:
        pred_cols = self.pred_cols_()
        Xdf = df[pred_cols].dropna()
        y_fwd = df[self.outcome].shift(-h).rename("y")
        data = pd.concat([y_fwd, Xdf], axis=1, join="inner").dropna()
        Y = data["y"].values

        scaler = StandardScaler().fit(data[pred_cols].values)
        X = sm.add_constant(scaler.transform(data[pred_cols].values))
        x_T = np.r_[1.0, scaler.transform(Xdf.loc[[self.origin_]].values).ravel()]

        result = GaRResult(horizon=h, quantiles=self.quantiles)
        result.y = data["y"]
        result.origin = self.origin_
        for q in self.quantiles:
            fit = QuantReg(Y, X).fit(q=q, max_iter=5000)
            result.coefs[q] = dict(zip(pred_cols, fit.params[1:]))
            result.fitted[q] = pd.Series(fit.fittedvalues, index=data.index)
            result.nowcast[q] = float(fit.params @ x_T)
        self._design[h] = (Y, X, pred_cols)

        result = self._compute_tail_metrics(result)
        if verbose:
            log.info(f"  h={h} origin={self.origin_.date()}: GaR5={result.gar_5:.2f}%  "
                     f"Median={result.median_forecast:.2f}%  P(rec)={result.prob_recession:.1%}")
        return result

    # ─────────────────────────────────────────────────────────────────────────
    def bootstrap_inference(self, B: int = 500, block: int | None = None, seed: int = 0) -> None:
        """Moving-block bootstrap SEs and two-sided normal p-values for all
        coefficients. Block length defaults to h + 12 to cover the MA
        structure of overlapping YoY outcomes."""
        for h, (Y, X, pred_cols) in self._design.items():
            se = _block_bootstrap_qr_se(Y, X, self.quantiles, B=B, block=block or h + 12, seed=seed)
            r = self.results_[h]
            for q in self.quantiles:
                b = np.array([r.coefs[q][c] for c in pred_cols])
                s = se[q]
                r.coef_se[q] = dict(zip(pred_cols, s))
                r.coef_p[q] = dict(zip(pred_cols, 2 * (1 - norm.cdf(np.abs(b / s)))))

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_tail_metrics(self, result: GaRResult) -> GaRResult:
        """Tail metrics from the nowcast quantiles (after monotone rearrangement)."""
        qs = sorted(result.nowcast)
        vals = np.sort([result.nowcast[q] for q in qs])   # Chernozhukov et al. (2010) rearrangement
        result.crossing = bool(np.any(np.diff([result.nowcast[q] for q in qs]) < 0))
        nc = dict(zip(qs, vals))
        result.nowcast = nc

        result.gar_5 = nc.get(0.05, np.nan)
        result.gar_25 = nc.get(0.25, np.nan)
        result.median_forecast = nc.get(0.50, np.nan)
        q25, q50, q75 = nc[0.25], nc[0.50], nc[0.75]
        # Bowley (1920) quartile skewness, in [-1, 1]; < 0 = longer left tail
        iqr = q75 - q25
        result.conditional_skewness = (q75 + q25 - 2 * q50) / iqr if iqr > 1e-9 else np.nan

        # Tail probabilities from a Gaussian matched to median and the 10–90 range
        sigma = max((nc[0.90] - nc[0.10]) / (norm.ppf(0.90) - norm.ppf(0.10)), 0.1)
        result.prob_neg_growth = float(norm.cdf(0, loc=q50, scale=sigma))
        result.prob_recession = float(norm.cdf(-2, loc=q50, scale=sigma))
        return result


    # ─────────────────────────────────────────────────────────────────────────
    def plot_fan_chart(
        self,
        horizon: int | None = None,
        title: str | None = None,
        n_history: int = 36,
        save_path: str | None = None,
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """
        Growth-at-Risk fan chart: historical outcome + future quantile fan.
        """
        h = horizon or self.horizon
        if h not in self.results_:
            self.fit(horizons=[h])
        result = self.results_[h]

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1628")
        ax.set_facecolor("#0a0e1a")

        # Historical outcome
        hist = self.df[self.outcome].dropna().iloc[-n_history:]
        ax.plot(hist.index, hist.values, color="#94a3b8",
                linewidth=1.5, label="Historical", zorder=5)

        # Plot in-sample quantile bands
        last_date = hist.index[-1]
        fan_colors = {
            (0.05, 0.95): ("#3b82f6", 0.12),
            (0.10, 0.90): ("#3b82f6", 0.18),
            (0.25, 0.75): ("#3b82f6", 0.28),
        }

        for (q_lo, q_hi), (color, alpha) in fan_colors.items():
            lo = result.fitted.get(q_lo)
            hi = result.fitted.get(q_hi)
            if lo is not None and hi is not None:
                lo_recent = lo.iloc[-n_history:]
                hi_recent = hi.iloc[-n_history:]
                ax.fill_between(lo_recent.index,
                                lo_recent.values, hi_recent.values,
                                alpha=alpha, color=color)

        # Median in-sample
        med = result.fitted.get(0.50)
        if med is not None:
            ax.plot(med.index[-n_history:], med.values[-n_history:],
                    color="#3b82f6", linewidth=1.0, linestyle="--",
                    alpha=0.7, label="Median fitted")

        # GaR5 line in-sample
        gar5 = result.fitted.get(0.05)
        if gar5 is not None:
            ax.plot(gar5.index[-n_history:], gar5.values[-n_history:],
                    color="#ef4444", linewidth=1.0, linestyle=":",
                    alpha=0.8, label="GaR 5th pctile")

        # Zero line
        ax.axhline(0, color="#475569", linewidth=0.8, linestyle="--")

        # Recession threshold
        ax.axhline(-2, color="#ef4444", linewidth=0.5, linestyle=":",
                   alpha=0.5, label="−2% threshold")

        # Annotate tail metrics
        nc = result.nowcast
        box_text = (
            f"Nowcast  (h={h}mo)\n"
            f"GaR 5th pctile:  {nc.get(0.05, np.nan):.1f}%\n"
            f"Median:          {nc.get(0.50, np.nan):.1f}%\n"
            f"P(GDP<0%):       {result.prob_neg_growth:.1%}\n"
            f"P(Recession):    {result.prob_recession:.1%}\n"
            f"Cond. skewness:  {result.conditional_skewness:.2f}"
        )
        ax.text(
            0.02, 0.97, box_text,
            transform=ax.transAxes, va="top", ha="left",
            fontsize=8.5, color="#94a3b8",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f1628",
                      edgecolor="#1e2d4a", alpha=0.9),
            fontfamily="monospace",
        )

        ax.set_ylabel(f"{self.outcome}", color="#94a3b8", fontsize=10)
        ax.set_xlabel("Date", color="#94a3b8", fontsize=10)
        t = title or f"Growth-at-Risk Fan Chart | Outcome: {self.outcome} | Horizon: {h} months"
        ax.set_title(t, color="#f1f5f9", fontsize=12, pad=10)
        ax.tick_params(colors="#64748b", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d4a")
        ax.legend(facecolor="#0f1628", edgecolor="#1e2d4a",
                  labelcolor="#94a3b8", fontsize=8)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    def plot_current_distribution(
        self,
        horizon: int | None = None,
        save_path: str | None = None,
        figsize: tuple = (8, 5),
    ) -> plt.Figure:
        """
        Plot the current predictive PDF of GDP/IP growth at horizon h.
        Uses skew-normal fit to the estimated quantile points.
        """
        h = horizon or self.horizon
        if h not in self.results_:
            self.fit(horizons=[h])
        result = self.results_[h]

        nc = result.nowcast
        qs = sorted([q for q in nc if not np.isnan(nc.get(q, np.nan))])
        vals = [nc[q] for q in qs]

        if len(qs) < 3:
            log.warning("Not enough quantiles for distribution plot.")
            return plt.figure()

        q50 = nc.get(0.50, np.median(vals))
        q10 = nc.get(0.10, np.percentile(vals, 10))
        q90 = nc.get(0.90, np.percentile(vals, 90))

        sigma = max((q90 - q10) / (norm.ppf(0.90) - norm.ppf(0.10)), 0.1)

        x_range = np.linspace(q50 - 4 * sigma, q50 + 4 * sigma, 400)
        pdf = norm.pdf(x_range, loc=q50, scale=sigma)

        fig, ax = plt.subplots(figsize=figsize, facecolor="#0f1628")
        ax.set_facecolor("#0a0e1a")

        # Fill tail regions
        mask_rec   = x_range <= -2
        mask_neg   = (x_range > -2) & (x_range <= 0)
        mask_pos   = x_range > 0
        ax.fill_between(x_range, 0, pdf, where=mask_rec,
                        color="#ef4444", alpha=0.5, label="Recession (<−2%)")
        ax.fill_between(x_range, 0, pdf, where=mask_neg,
                        color="#f97316", alpha=0.35, label="Contraction (<0%)")
        ax.fill_between(x_range, 0, pdf, where=mask_pos,
                        color="#10b981", alpha=0.25, label="Expansion (>0%)")

        ax.plot(x_range, pdf, color="#e8a020", linewidth=1.8)

        # Vertical lines
        ax.axvline(q50, color="#3b82f6", linewidth=1.2, linestyle="--",
                   label=f"Median: {q50:.1f}%")
        ax.axvline(nc.get(0.05, q50 - 2 * sigma),
                   color="#ef4444", linewidth=1.2, linestyle=":",
                   label=f"GaR 5%: {nc.get(0.05, np.nan):.1f}%")
        ax.axvline(0, color="#475569", linewidth=0.8, linestyle="-")

        ax.set_xlabel(f"{self.outcome} (%)", color="#94a3b8", fontsize=10)
        ax.set_ylabel("Density", color="#94a3b8", fontsize=10)
        ax.set_title(
            f"Predictive Distribution of {self.outcome}  (h={h} months ahead)",
            color="#f1f5f9", fontsize=12, pad=10,
        )
        ax.tick_params(colors="#64748b", labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2d4a")

        # Summary stats box
        stats_text = (
            f"P(recession) = {result.prob_recession:.1%}\n"
            f"P(neg. growth) = {result.prob_neg_growth:.1%}\n"
            f"Cond. skewness = {result.conditional_skewness:.2f}"
        )
        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                va="top", ha="right", fontsize=9, color="#94a3b8",
                bbox=dict(facecolor="#0f1628", edgecolor="#1e2d4a",
                          boxstyle="round,pad=0.4"),
                fontfamily="monospace")

        ax.legend(facecolor="#0f1628", edgecolor="#1e2d4a",
                  labelcolor="#94a3b8", fontsize=8)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ─────────────────────────────────────────────────────────────────────────
    def rolling_gar(
        self, horizon: int | None = None, window: int = 60
    ) -> pd.DataFrame:
        """
        Compute rolling GaR estimates — captures time variation in tail risk.

        Returns DataFrame with columns: gar5, gar25, median, gar75, gar95
        """
        h = horizon or self.horizon
        df = self.df.copy()
        if self.fci_ is not None:
            df["fci"] = self.fci_

        pred_cols = [c for c in [self.gpr_col, "fci", self.outcome]
                     if c in df.columns]
        y_fwd = df[self.outcome].shift(-h)
        data = pd.concat([y_fwd.rename("y"), df[pred_cols]], axis=1).dropna()

        records = []
        target_qs = [0.05, 0.25, 0.50, 0.75, 0.95]

        for end in range(window, len(data)):
            sub = data.iloc[max(0, end - window):end]
            Y = sub["y"].values
            X_raw = sub[pred_cols].values
            X = (X_raw - X_raw.mean(0)) / (X_raw.std(0) + 1e-9)
            row = {"date": sub.index[-1]}

            for q in target_qs:
                try:
                    if SM_AVAILABLE:
                        X_sm = sm.add_constant(X)
                        qr = QuantReg(Y, X_sm).fit(q=q, max_iter=500)
                        row[f"q{int(q*100):02d}"] = qr.fittedvalues[-1]
                    elif SKL_AVAILABLE:
                        qr = QuantileRegressor(quantile=q, alpha=0, solver="highs")
                        qr.fit(X, Y)
                        row[f"q{int(q*100):02d}"] = qr.predict(X[-1:])[0]
                    else:
                        row[f"q{int(q*100):02d}"] = np.nan
                except Exception:
                    row[f"q{int(q*100):02d}"] = np.nan
            records.append(row)

        return pd.DataFrame(records).set_index("date")


# ═══════════════════════════════════════════════════════════════════════════════
#  ROBUSTNESS CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RobustnessResult:
    """Outputs of the three pre-specified checks."""
    orth_coefs: dict[int, dict[float, float]] = field(default_factory=dict)
    orth_se:    dict[int, dict[float, float]] = field(default_factory=dict)
    orth_p:     dict[int, dict[float, float]] = field(default_factory=dict)
    lasso_alpha:    dict[int, float] = field(default_factory=dict)
    lasso_selected: dict[int, dict[float, list[str]]] = field(default_factory=dict)
    lasso_coefs:    dict[int, dict[float, dict[str, float]]] = field(default_factory=dict)
    pinball_baseline:   dict[int, dict[float, float]] = field(default_factory=dict)
    pinball_enhanced:   dict[int, dict[float, float]] = field(default_factory=dict)
    pseudo_r2_baseline: dict[int, dict[float, float]] = field(default_factory=dict)
    pseudo_r2_enhanced: dict[int, dict[float, float]] = field(default_factory=dict)
    improvement:        dict[int, dict[float, float]] = field(default_factory=dict)


class GaRRobustness:
    """
    Three pre-specified checks on the enhanced GaR specification.

    Check 1 — Orthogonalisation. GIPI_{t-1} is residualised on GPR_t (OLS);
      Q_τ(y_{t+h}) = α + β·GPR_t + γ·FCI_t + δ⊥·GIPI⊥_{t-1} + η·y_t.
      δ⊥ inference by moving-block bootstrap (block h+12).
    Check 2 — LASSO-QR over GPR, FCI, y_t and the five GIPI inputs;
      penalty chosen per horizon by 5-fold contiguous CV at τ = 0.10.
    Check 3 — In-sample pinball loss and Koenker–Machado pseudo-R² of the
      baseline vs the enhanced model (identical spec to GrowthAtRisk).
      Nested in-sample comparison: the enhanced model weakly dominates by
      construction, so this is descriptive, not a test.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        outcome: str = "ip_yoy",
        gpr_col: str = "gpr_z",
        gipi_col: str = "gipi",
        gipi_components: list[str] | None = None,
        quantiles: list[float] | None = None,
        lasso_alpha: float | None = None,
        bootstrap_reps: int = 500,
    ):
        self.df = df.copy()
        self.outcome = outcome
        self.gpr_col = gpr_col
        self.gipi_col = gipi_col
        self.gipi_components = gipi_components or [
            "gscpi", "import_price_yoy", "global_energy_yoy",
            "global_food_yoy", "d_breakeven_5y",
        ]
        self.quantiles = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.lasso_alpha = lasso_alpha
        self.bootstrap_reps = bootstrap_reps

    # ──────────────────────────────────────────────────────────────────────
    def run(self, horizons: list[int] | None = None) -> RobustnessResult:
        horizons = horizons or [3, 6, 12]
        result = RobustnessResult()
        df = self.df.copy()
        df["fci"] = build_fci(df)

        # Check 1
        g_lag = df[self.gipi_col].shift(1)
        sub = pd.concat([g_lag.rename("g"), df[self.gpr_col]], axis=1).dropna()
        ols = sm.OLS(sub["g"], sm.add_constant(sub[self.gpr_col])).fit()
        df["gipi_orth"] = ols.resid.reindex(df.index)
        log.info(f"Check 1: corr(GIPI_t-1, GPR) = {sub.corr().iloc[0, 1]:.3f}")
        for h in horizons:
            result.orth_coefs[h], result.orth_se[h], result.orth_p[h] = self._check1(df, h)

        # Check 2
        for h in horizons:
            alpha = self.lasso_alpha or self._cv_lasso_alpha(df, h)
            result.lasso_alpha[h] = alpha
            result.lasso_selected[h], result.lasso_coefs[h] = self._check2_lasso(df, h, alpha)

        # Check 3
        base = GrowthAtRisk(self.df, outcome=self.outcome, gpr_col=self.gpr_col,
                            gipi_col=None, use_interaction=False, quantiles=self.quantiles)
        enh = GrowthAtRisk(self.df, outcome=self.outcome, gpr_col=self.gpr_col,
                           gipi_col=self.gipi_col, use_interaction=True, quantiles=self.quantiles)
        rb, re = base.fit(horizons, verbose=False), enh.fit(horizons, verbose=False)
        for h in horizons:
            idx = re[h].y.index.intersection(rb[h].y.index)
            y = re[h].y.loc[idx].values
            pb_b, pb_e, r2_b, r2_e, imp = {}, {}, {}, {}, {}
            for q in self.quantiles:
                null = _pinball_loss(y, np.full(len(y), np.quantile(y, q)), q)
                pb_b[q] = _pinball_loss(y, rb[h].fitted[q].loc[idx].values, q)
                pb_e[q] = _pinball_loss(y, re[h].fitted[q].loc[idx].values, q)
                r2_b[q], r2_e[q] = 1 - pb_b[q] / null, 1 - pb_e[q] / null
                imp[q] = 100 * (pb_b[q] - pb_e[q]) / pb_b[q]
            result.pinball_baseline[h], result.pinball_enhanced[h] = pb_b, pb_e
            result.pseudo_r2_baseline[h], result.pseudo_r2_enhanced[h] = r2_b, r2_e
            result.improvement[h] = imp

        self._result = result
        return result

    # ──────────────────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        r = self._result
        rows = []
        for h in sorted(r.improvement):
            for q in self.quantiles:
                rows.append({
                    "horizon": h, "quantile": q,
                    "orth_gipi_coef": r.orth_coefs[h][q],
                    "orth_gipi_se": r.orth_se[h][q],
                    "orth_gipi_p": r.orth_p[h][q],
                    "lasso_alpha": r.lasso_alpha[h],
                    "lasso_selected": ", ".join(r.lasso_selected[h].get(q, [])),
                    "pinball_base": r.pinball_baseline[h][q],
                    "pinball_enh": r.pinball_enhanced[h][q],
                    "pinball_impr%": r.improvement[h][q],
                    "pseudo_r2_base": r.pseudo_r2_baseline[h][q],
                    "pseudo_r2_enh": r.pseudo_r2_enhanced[h][q],
                })
        return pd.DataFrame(rows)

    def save(self, path: str = "outputs/robustness_checks.csv") -> None:
        self.summary().round(4).to_csv(path, index=False)
        log.info(f"Robustness table → {path}")

    # ──────────────────────────────────────────────────────────────────────
    def _check1(self, df: pd.DataFrame, h: int):
        cols = [self.gpr_col, "fci", "gipi_orth", self.outcome]
        data = pd.concat([df[self.outcome].shift(-h).rename("y"), df[cols]], axis=1).dropna()
        Y = data["y"].values
        X = sm.add_constant(StandardScaler().fit_transform(data[cols].values))
        k = cols.index("gipi_orth")
        coefs = {q: float(QuantReg(Y, X).fit(q=q, max_iter=5000).params[k + 1])
                 for q in self.quantiles}
        se_all = _block_bootstrap_qr_se(Y, X, self.quantiles, B=self.bootstrap_reps, block=h + 12)
        se = {q: float(se_all[q][k]) for q in self.quantiles}
        p = {q: float(2 * (1 - norm.cdf(abs(coefs[q] / se[q])))) for q in self.quantiles}
        return coefs, se, p

    def _component_cols(self, df: pd.DataFrame) -> list[str]:
        return [self.gpr_col, "fci", self.outcome] + [c for c in self.gipi_components if c in df]

    def _lasso_data(self, df: pd.DataFrame, h: int):
        cols = self._component_cols(df)
        data = pd.concat([df[self.outcome].shift(-h).rename("y"), df[cols]], axis=1).dropna()
        return data["y"].values, StandardScaler().fit_transform(data[cols].values), cols

    def _cv_lasso_alpha(self, df: pd.DataFrame, h: int) -> float:
        """5-fold contiguous-block CV on pinball loss at τ = 0.10."""
        Y, X, _ = self._lasso_data(df, h)
        n, folds = len(Y), np.array_split(np.arange(len(Y)), 5)
        best, best_loss = 0.01, np.inf
        for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
            losses = []
            for val in folds:
                # drop h months either side of the validation block (overlap)
                train = np.setdiff1d(np.arange(n), np.arange(val[0] - h, val[-1] + h + 1))
                qr = QuantileRegressor(quantile=0.10, alpha=alpha, solver="highs").fit(X[train], Y[train])
                losses.append(_pinball_loss(Y[val], qr.predict(X[val]), 0.10))
            if np.mean(losses) < best_loss:
                best, best_loss = alpha, np.mean(losses)
        return best

    def _check2_lasso(self, df: pd.DataFrame, h: int, alpha: float):
        Y, X, cols = self._lasso_data(df, h)
        selected, coef_out = {}, {}
        for q in self.quantiles:
            qr = QuantileRegressor(quantile=q, alpha=alpha, solver="highs").fit(X, Y)
            keep = [i for i, c in enumerate(qr.coef_) if abs(c) > 1e-6]
            selected[q] = [cols[i] for i in keep]
            coef_out[q] = {cols[i]: round(float(qr.coef_[i]), 4) for i in keep}
        return selected, coef_out


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Mean pinball (quantile) loss: L_τ(y, ŷ) = mean[(y-ŷ)(τ - 1{y<ŷ})]."""
    err = y_true - y_pred
    return float(np.mean(np.where(err >= 0, q * err, (q - 1) * err)))


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_gar_suite(
    df: pd.DataFrame,
    outcome: str = "ip_yoy",
    horizons: list[int] | None = None,
    verbose: bool = True,
) -> tuple[GrowthAtRisk, dict[int, GaRResult]]:
    """
    Run the full GaR suite: fit multiple horizons, log summary.

    Returns
    -------
    (GrowthAtRisk instance, dict[horizon → GaRResult])
    """
    horizons = horizons or [1, 3, 6, 12]
    gar = GrowthAtRisk(df, outcome=outcome)
    results = gar.fit(horizons=horizons, verbose=verbose)
    return gar, results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.pipeline import DataPipeline

    dp = DataPipeline()
    df = dp.build(use_cache=False)

    gar, results = run_gar_suite(df, outcome="ip_yoy", horizons=[3, 6, 12])

    for h, res in results.items():
        print(f"\nHorizon {h}m: GaR5={res.gar_5:.2f}%  "
              f"Median={res.median_forecast:.2f}%  "
              f"P(rec)={res.prob_recession:.1%}")

    fig = gar.plot_fan_chart(horizon=6, save_path="outputs/figures/gar_fan.png")
    fig2 = gar.plot_current_distribution(horizon=6, save_path="outputs/figures/gar_dist.png")
    print("\nSaved fan chart and distribution plot.")
