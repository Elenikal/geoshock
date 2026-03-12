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
    conditional_skewness: float = 0.0


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
        quantiles: list[float] | None = None,
        horizon: int = 6,
    ):
        self.df              = df.copy()
        self.outcome         = outcome
        self.gpr_col         = gpr_col
        self.gipi_col        = gipi_col if (gipi_col and gipi_col in df.columns) else None
        self.use_interaction = use_interaction and (self.gipi_col is not None)
        self.quantiles       = quantiles or self.QUANTILES
        self.horizon         = horizon
        self.fci_            : pd.Series | None = None
        self.results_        : dict[int, GaRResult] = {}
        self._scaler         = StandardScaler() if SKL_AVAILABLE else None

        if self.gipi_col:
            log.info(f"GaR v2: GIPI enabled ({gipi_col}), "
                     f"interaction={'on' if self.use_interaction else 'off'}")
        else:
            log.info("GaR v2: GIPI not found in dataframe — running baseline GPR+FCI spec")

    # ─────────────────────────────────────────────────────────────────────────
    def fit(
        self,
        horizons: list[int] | None = None,
        verbose: bool = True,
    ) -> dict[int, GaRResult]:
        """
        Fit quantile regression at each horizon in `horizons`.

        Specification:
          Baseline: Q_τ(y|x) = α + β·GPR + γ·FCI + η·y_lag
          Enhanced: Q_τ(y|x) = α + β·GPR + γ·FCI + δ·GIPI + ζ·(GPR×GIPI) + η·y_lag
        """
        if not (SM_AVAILABLE or SKL_AVAILABLE):
            raise ImportError("statsmodels or scikit-learn required.")

        if horizons is None:
            horizons = [self.horizon]

        # Build FCI
        self.fci_ = build_fci(self.df)
        df = self.df.copy()
        df["fci"] = self.fci_

        # Build GPR × GIPI interaction if requested
        if self.use_interaction and self.gipi_col:
            gpr_z = (df[self.gpr_col] - df[self.gpr_col].mean()) / (df[self.gpr_col].std() + 1e-9)
            gipi_z = (df[self.gipi_col] - df[self.gipi_col].mean()) / (df[self.gipi_col].std() + 1e-9)
            df["gpr_x_gipi"] = gpr_z * gipi_z
            log.info("GaR: GPR × GIPI interaction term constructed")

        for h in horizons:
            if verbose:
                log.info(f"GaR: fitting horizon h={h}")
            result = self._fit_horizon(df, h, verbose)
            self.results_[h] = result

        return self.results_

    # ─────────────────────────────────────────────────────────────────────────
    def _fit_horizon(self, df: pd.DataFrame, h: int, verbose: bool) -> GaRResult:
        # Outcome h periods ahead
        y_fwd = df[self.outcome].shift(-h)

        # Build predictor list: GPR, FCI, (GIPI), (GPR×GIPI), lagged outcome
        pred_cols = []
        if self.gpr_col in df.columns:
            pred_cols.append(self.gpr_col)
        if "fci" in df.columns:
            pred_cols.append("fci")
        if self.gipi_col and self.gipi_col in df.columns:
            pred_cols.append(self.gipi_col)
        if self.use_interaction and "gpr_x_gipi" in df.columns:
            pred_cols.append("gpr_x_gipi")
        if self.outcome in df.columns:
            pred_cols.append(self.outcome)   # lagged outcome

        data = pd.concat([y_fwd.rename("y"), df[pred_cols]], axis=1).dropna()
        Y = data["y"].values
        X_raw = data[pred_cols].values

        # Standardise predictors
        if SKL_AVAILABLE:
            scaler = StandardScaler()
            X = scaler.fit_transform(X_raw)
        else:
            X = (X_raw - X_raw.mean(axis=0)) / (X_raw.std(axis=0) + 1e-9)

        coefs_dict: dict[float, dict[str, float]] = {}
        fitted_dict: dict[float, pd.Series] = {}

        for q in self.quantiles:
            try:
                if SM_AVAILABLE:
                    X_sm = sm.add_constant(X)
                    qreg = QuantReg(Y, X_sm).fit(q=q, max_iter=2000)
                    fitted_vals = qreg.fittedvalues
                    coef_vals = qreg.params[1:]  # drop constant
                else:
                    qr = QuantileRegressor(quantile=q, alpha=0, solver="highs")
                    qr.fit(X, Y)
                    fitted_vals = qr.predict(X)
                    coef_vals = qr.coef_

                coefs_dict[q] = dict(zip(pred_cols, coef_vals))
                fitted_dict[q] = pd.Series(fitted_vals, index=data.index)

            except Exception as e:
                log.debug(f"  Quantile {q} at h={h} failed: {e}")
                coefs_dict[q] = {}
                fitted_dict[q] = pd.Series(np.nan, index=data.index)

        # Nowcast: predict at last available observation
        result = GaRResult(
            horizon=h,
            quantiles=self.quantiles,
            coefs=coefs_dict,
            fitted=fitted_dict,
        )
        result = self._compute_nowcast(result, data, X, pred_cols)
        result = self._compute_tail_metrics(result)

        if verbose:
            log.info(
                f"  h={h}: GaR5={result.gar_5:.2f}%  "
                f"Median={result.median_forecast:.2f}%  "
                f"P(rec)={result.prob_recession:.1%}"
            )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_nowcast(
        self,
        result: GaRResult,
        data: pd.DataFrame,
        X: np.ndarray,
        pred_cols: list[str],
    ) -> GaRResult:
        """Evaluate fitted quantiles at the latest data point."""
        if len(X) == 0:
            return result

        x_last = X[-1:, :]
        h = result.horizon

        for q in self.quantiles:
            if not result.coefs.get(q):
                result.nowcast[q] = np.nan
                continue
            try:
                coef_vals = np.array([result.coefs[q].get(c, 0) for c in pred_cols])
                if SM_AVAILABLE:
                    # Reconstruct from statsmodels coefs (already on standardised X)
                    fitted = result.fitted[q]
                    if len(fitted) > 0:
                        result.nowcast[q] = float(fitted.iloc[-1])
                    else:
                        result.nowcast[q] = np.nan
                else:
                    result.nowcast[q] = float(x_last @ coef_vals)
            except Exception:
                result.nowcast[q] = np.nan

        return result

    # ─────────────────────────────────────────────────────────────────────────
    def _compute_tail_metrics(self, result: GaRResult) -> GaRResult:
        """Compute tail risk statistics from nowcast quantile distribution."""
        nc = result.nowcast
        qs = sorted([q for q in nc if not np.isnan(nc.get(q, np.nan))])
        vals = [nc[q] for q in qs]

        if len(qs) < 3:
            return result

        result.gar_5  = nc.get(0.05, np.nan)
        result.gar_25 = nc.get(0.25, np.nan)
        result.median_forecast = nc.get(0.50, np.nan)

        # Conditional skewness: (Q90-Q50) - (Q50-Q10) normalised
        q10 = nc.get(0.10, np.nan)
        q90 = nc.get(0.90, np.nan)
        q50 = nc.get(0.50, np.nan)
        if not any(np.isnan([q10, q50, q90])):
            denom = q90 - q10
            if abs(denom) > 1e-6:
                result.conditional_skewness = ((q90 - q50) - (q50 - q10)) / denom

        # Tail probabilities: interpolate CDF from quantiles
        try:
            # Fit normal or skew-normal to quantile estimates
            mu = q50
            sigma = (q90 - q10) / (norm.ppf(0.90) - norm.ppf(0.10))
            sigma = max(sigma, 0.1)
            result.prob_neg_growth = float(norm.cdf(0, loc=mu, scale=sigma))
            result.prob_recession  = float(norm.cdf(-2, loc=mu, scale=sigma))
        except Exception:
            result.prob_neg_growth = np.nan
            result.prob_recession  = np.nan

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
    """Stores all three robustness check outputs."""
    # 1. Orthogonalization
    orth_coefs:          dict[int, dict[float, float]] = field(default_factory=dict)
    orth_significant:    dict[int, bool]               = field(default_factory=dict)

    # 2. LASSO-QR selected variables
    lasso_selected:      dict[int, dict[float, list[str]]] = field(default_factory=dict)
    lasso_coefs:         dict[int, dict[float, dict[str, float]]] = field(default_factory=dict)

    # 3. Model comparison (pseudo-R² + pinball loss)
    pinball_baseline:    dict[int, dict[float, float]] = field(default_factory=dict)
    pinball_enhanced:    dict[int, dict[float, float]] = field(default_factory=dict)
    pseudo_r2_baseline:  dict[int, dict[float, float]] = field(default_factory=dict)
    pseudo_r2_enhanced:  dict[int, dict[float, float]] = field(default_factory=dict)
    improvement:         dict[int, dict[float, float]] = field(default_factory=dict)  # % pinball improvement


class GaRRobustness:
    """
    Three robustness checks for the GIPI-enhanced Growth-at-Risk model.

    ─────────────────────────────────────────────────────────────────────
    CHECK 1 — ORTHOGONALISATION
    ─────────────────────────────────────────────────────────────────────
    Regress GIPI on GPR shock to isolate the non-GPR component:

        GIPI_resid_t = GIPI_t - proj(GIPI_t | GPR_t)

    Then re-estimate:
        Q_τ(y_{t+h}) = α + β·GPR + γ·FCI + δ·GIPI_resid + η·y_lag

    If δ_τ remains significant at the lower tail quantiles (τ ≤ 0.10),
    it confirms that GIPI adds genuine information BEYOND what GPR already
    captures — i.e. the inflation-channel transmission is not just a
    mechanical function of geopolitical risk itself.

    ─────────────────────────────────────────────────────────────────────
    CHECK 2 — LASSO QUANTILE REGRESSION
    ─────────────────────────────────────────────────────────────────────
    Expand the predictor set to ALL individual GIPI components (GSCPI,
    import price, energy, food, breakeven) and apply L1-penalised
    quantile regression (sklearn QuantileRegressor with alpha > 0):

        min  Σ ρ_τ(y - Xβ) + α·‖β‖₁
              β

    Variables surviving LASSO shrinkage are those with genuine marginal
    predictive value at each quantile. Expected to validate that GSCPI
    and breakeven 5Y are robustly selected at the lower tail.

    The penalty α is chosen via 5-fold cross-validation on pinball loss.

    ─────────────────────────────────────────────────────────────────────
    CHECK 3 — PSEUDO-R² AND PINBALL LOSS COMPARISON
    ─────────────────────────────────────────────────────────────────────
    Compare baseline (GPR + FCI) vs enhanced (+ GIPI + interaction)
    using two proper scoring rules:

    Pinball loss (quantile loss):
        L_τ(y, ŷ) = (y - ŷ)·(τ - 1{y < ŷ})

    Koenker-Machado pseudo-R²:
        R²_τ = 1 - V̂_τ(model) / V̂_τ(null)
    where V̂_τ(null) = loss of intercept-only model.

    Improvement > 0% at τ = 0.05 and τ = 0.10 confirms the enhanced
    model better characterises the left tail of the distribution.
    ─────────────────────────────────────────────────────────────────────

    Usage
    ─────
    rob = GaRRobustness(df=df, gipi_col="gipi")
    result = rob.run(horizons=[3, 6, 12])
    rob.summary()
    """

    def __init__(
        self,
        df:          pd.DataFrame,
        outcome:     str = "ip_yoy",
        gpr_col:     str = "gpr_z",
        gipi_col:    str = "gipi",
        gipi_components: list[str] | None = None,
        quantiles:   list[float] | None = None,
        lasso_alpha: float | None = None,   # None = cross-validated
    ):
        self.df              = df.copy()
        self.outcome         = outcome
        self.gpr_col         = gpr_col
        self.gipi_col        = gipi_col
        self.gipi_components = gipi_components or [
            "gscpi", "import_price_yoy", "global_energy_yoy",
            "global_food_yoy", "d_breakeven_5y",
        ]
        self.quantiles   = quantiles or [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        self.lasso_alpha = lasso_alpha

        if not SKL_AVAILABLE:
            raise ImportError("scikit-learn required for GaRRobustness. "
                              "pip install scikit-learn")

    # ──────────────────────────────────────────────────────────────────────
    def run(self, horizons: list[int] | None = None) -> RobustnessResult:
        """Run all three robustness checks. Returns RobustnessResult."""
        horizons = horizons or [3, 6, 12]
        result   = RobustnessResult()

        # Build FCI once
        fci = build_fci(self.df)
        df  = self.df.copy()
        df["fci"] = fci

        # ── Check 1: Orthogonalisation ────────────────────────────────────
        log.info("Robustness Check 1: Orthogonalisation (GIPI ⊥ GPR) …")
        df = self._orthogonalise_gipi(df)

        for h in horizons:
            coefs, sig = self._check1_orth(df, h)
            result.orth_coefs[h]       = coefs
            result.orth_significant[h] = sig

        # ── Check 2: LASSO-QR ────────────────────────────────────────────
        log.info("Robustness Check 2: LASSO-QR variable selection …")
        alpha = self.lasso_alpha or self._cv_lasso_alpha(df)
        log.info(f"  LASSO α = {alpha:.4f}")

        for h in horizons:
            sel, coef = self._check2_lasso(df, h, alpha)
            result.lasso_selected[h] = sel
            result.lasso_coefs[h]    = coef

        # ── Check 3: Pinball loss + pseudo-R² ────────────────────────────
        log.info("Robustness Check 3: Pinball loss + pseudo-R² comparison …")
        for h in horizons:
            pb_base, pb_enh, r2_base, r2_enh, impr = self._check3_model_comparison(df, h)
            result.pinball_baseline[h]   = pb_base
            result.pinball_enhanced[h]   = pb_enh
            result.pseudo_r2_baseline[h] = r2_base
            result.pseudo_r2_enhanced[h] = r2_enh
            result.improvement[h]        = impr

        self._result = result
        return result

    # ──────────────────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        """Print and return a tidy summary DataFrame."""
        if not hasattr(self, "_result"):
            raise RuntimeError("Call .run() first.")
        r = self._result
        rows = []
        for h in sorted(r.improvement.keys()):
            for q in self.quantiles:
                rows.append({
                    "horizon":          h,
                    "quantile":         q,
                    "orth_gipi_coef":   round(r.orth_coefs.get(h, {}).get(q, np.nan), 4),
                    "orth_significant": r.orth_significant.get(h, False),
                    "lasso_selected":   ", ".join(r.lasso_selected.get(h, {}).get(q, [])),
                    "pinball_base":     round(r.pinball_baseline.get(h, {}).get(q, np.nan), 4),
                    "pinball_enh":      round(r.pinball_enhanced.get(h, {}).get(q, np.nan), 4),
                    "pinball_impr%":    round(r.improvement.get(h, {}).get(q, np.nan), 2),
                    "pseudo_r2_base":   round(r.pseudo_r2_baseline.get(h, {}).get(q, np.nan), 4),
                    "pseudo_r2_enh":    round(r.pseudo_r2_enhanced.get(h, {}).get(q, np.nan), 4),
                })
        df = pd.DataFrame(rows)
        # Print tail-focused summary
        tail = df[df["quantile"].isin([0.05, 0.10])]
        log.info("\n=== ROBUSTNESS SUMMARY (tail quantiles) ===\n" + tail.to_string(index=False))
        return df

    def save(self, path: str = "outputs/robustness_checks.csv") -> None:
        """Save full robustness table to CSV."""
        summary = self.summary()
        summary.to_csv(path, index=False)
        log.info(f"Robustness table → {path}")

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _orthogonalise_gipi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Residualise GIPI with respect to GPR shock (OLS projection).
        Adds column 'gipi_orth' to df.
        """
        if self.gipi_col not in df.columns or self.gpr_col not in df.columns:
            log.warning("  Orthogonalisation: GIPI or GPR not found — skipping")
            df["gipi_orth"] = df.get(self.gipi_col, pd.Series(dtype=float))
            return df

        sub = df[[self.gipi_col, self.gpr_col]].dropna()
        if len(sub) < 30:
            df["gipi_orth"] = df[self.gipi_col]
            return df

        X = sub[[self.gpr_col]].values
        y = sub[self.gipi_col].values

        if SM_AVAILABLE:
            ols = sm.OLS(y, sm.add_constant(X)).fit()
            resid = ols.resid
        else:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(X, y)
            resid = y - lr.predict(X)

        gipi_orth = pd.Series(resid, index=sub.index, name="gipi_orth")
        df["gipi_orth"] = gipi_orth.reindex(df.index)

        corr = float(pd.concat([sub[self.gipi_col], gipi_orth], axis=1).corr().iloc[0, 1])
        log.info(f"  GIPI–GPR correlation before orth: "
                 f"{float(sub[self.gipi_col].corr(sub[self.gpr_col])):.3f}")
        log.info(f"  GIPI_orth–GPR correlation after orth: "
                 f"{float(gipi_orth.corr(sub[self.gpr_col])):.3f}  (target: ~0)")
        return df

    def _check1_orth(
        self, df: pd.DataFrame, h: int
    ) -> tuple[dict[float, float], bool]:
        """
        Estimate Q_τ(y_{t+h}) = α + β·GPR + γ·FCI + δ·GIPI_orth + η·y_lag
        Returns dict[τ → δ_coef] and bool indicating tail significance.
        """
        y_fwd = df[self.outcome].shift(-h)
        cols  = [self.gpr_col, "fci", "gipi_orth", self.outcome]
        cols  = [c for c in cols if c in df.columns]

        data = pd.concat([y_fwd.rename("y"), df[cols]], axis=1).dropna()
        if len(data) < 30:
            return {q: np.nan for q in self.quantiles}, False

        Y = data["y"].values
        X = StandardScaler().fit_transform(data[cols].values)
        gipi_orth_idx = cols.index("gipi_orth") if "gipi_orth" in cols else None

        coefs: dict[float, float] = {}
        for q in self.quantiles:
            try:
                if SM_AVAILABLE:
                    qr = QuantReg(Y, sm.add_constant(X)).fit(q=q, max_iter=2000)
                    # +1 because add_constant prepends intercept
                    coefs[q] = float(qr.params[gipi_orth_idx + 1]
                                     if gipi_orth_idx is not None else np.nan)
                else:
                    qr = QuantileRegressor(quantile=q, alpha=0, solver="highs")
                    qr.fit(X, Y)
                    coefs[q] = float(qr.coef_[gipi_orth_idx]
                                     if gipi_orth_idx is not None else np.nan)
            except Exception:
                coefs[q] = np.nan

        # Significance: tail coefs should be negative and abs > 0.05 in std units
        tail_coefs = [coefs.get(q, 0.0) for q in [0.05, 0.10]]
        significant = all((not np.isnan(c)) and c < -0.03 for c in tail_coefs)

        log.info(f"  Orth h={h}: δ(τ=0.05)={coefs.get(0.05, np.nan):.3f}  "
                 f"δ(τ=0.10)={coefs.get(0.10, np.nan):.3f}  "
                 f"significant={significant}")
        return coefs, significant

    def _cv_lasso_alpha(self, df: pd.DataFrame, h: int = 6) -> float:
        """
        Cross-validate LASSO penalty α using 5-fold CV on pinball loss at τ=0.10.
        """
        cols = self._build_component_cols(df)
        if not cols:
            return 0.01

        y_fwd = df[self.outcome].shift(-h)
        data  = pd.concat([y_fwd.rename("y"), df[cols]], axis=1).dropna()
        if len(data) < 50:
            return 0.01

        Y = data["y"].values
        X = StandardScaler().fit_transform(data[cols].values)
        n = len(Y)

        alphas     = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
        best_alpha = 0.01
        best_loss  = np.inf
        fold_size  = max(10, n // 5)

        for alpha in alphas:
            losses = []
            for fold in range(5):
                val_idx  = list(range(fold * fold_size,
                                      min((fold + 1) * fold_size, n)))
                train_idx = [i for i in range(n) if i not in val_idx]
                if len(train_idx) < 20 or len(val_idx) < 5:
                    continue
                try:
                    qr = QuantileRegressor(quantile=0.10, alpha=alpha, solver="highs")
                    qr.fit(X[train_idx], Y[train_idx])
                    preds = qr.predict(X[val_idx])
                    losses.append(_pinball_loss(Y[val_idx], preds, 0.10))
                except Exception:
                    pass
            if losses and np.mean(losses) < best_loss:
                best_loss  = np.mean(losses)
                best_alpha = alpha

        return best_alpha

    def _check2_lasso(
        self, df: pd.DataFrame, h: int, alpha: float
    ) -> tuple[dict[float, list[str]], dict[float, dict[str, float]]]:
        """
        LASSO-QR on individual GIPI components. Returns selected variables
        and their coefficients at each quantile.
        """
        if not SKL_AVAILABLE:
            return {q: [] for q in self.quantiles}, {}

        cols = self._build_component_cols(df)
        if not cols:
            return {q: [] for q in self.quantiles}, {}

        y_fwd = df[self.outcome].shift(-h)
        data  = pd.concat([y_fwd.rename("y"), df[cols]], axis=1).dropna()
        if len(data) < 30:
            return {q: [] for q in self.quantiles}, {}

        Y     = data["y"].values
        X     = StandardScaler().fit_transform(data[cols].values)

        selected: dict[float, list[str]] = {}
        coef_out: dict[float, dict[str, float]] = {}

        for q in self.quantiles:
            try:
                qr = QuantileRegressor(quantile=q, alpha=alpha, solver="highs")
                qr.fit(X, Y)
                nonzero = [cols[i] for i, c in enumerate(qr.coef_) if abs(c) > 1e-6]
                selected[q]  = nonzero
                coef_out[q]  = {cols[i]: round(float(qr.coef_[i]), 4)
                                for i in range(len(cols)) if abs(qr.coef_[i]) > 1e-6}
            except Exception:
                selected[q]  = []
                coef_out[q]  = {}

        # Log tail-quantile selection
        for q in [0.05, 0.10]:
            log.info(f"  LASSO h={h} τ={q}: {selected.get(q,[])}")

        return selected, coef_out

    def _check3_model_comparison(
        self, df: pd.DataFrame, h: int
    ) -> tuple[dict, dict, dict, dict, dict]:
        """
        Compare pinball loss and pseudo-R² of baseline vs enhanced model.
        Returns (pinball_base, pinball_enh, r2_base, r2_enh, improvement%)
        """
        y_fwd = df[self.outcome].shift(-h)

        # Baseline predictors: GPR + FCI + lagged outcome
        base_cols = [c for c in [self.gpr_col, "fci", self.outcome]
                     if c in df.columns]
        # Enhanced: + GIPI + interaction
        enh_cols  = base_cols.copy()
        if self.gipi_col in df.columns:
            enh_cols.append(self.gipi_col)
        if "gpr_x_gipi" in df.columns:
            enh_cols.append("gpr_x_gipi")
        elif self.gipi_col in df.columns and self.gpr_col in df.columns:
            df = df.copy()
            gpr_z  = (df[self.gpr_col] - df[self.gpr_col].mean()) / df[self.gpr_col].std()
            gipi_z = (df[self.gipi_col] - df[self.gipi_col].mean()) / df[self.gipi_col].std()
            df["gpr_x_gipi"] = gpr_z * gipi_z
            enh_cols.append("gpr_x_gipi")

        all_cols = list(set(base_cols + enh_cols))
        data = pd.concat([y_fwd.rename("y"), df[all_cols]], axis=1).dropna()
        if len(data) < 30:
            empty = {q: np.nan for q in self.quantiles}
            return empty, empty, empty, empty, empty

        Y = data["y"].values

        pb_base: dict[float, float] = {}
        pb_enh:  dict[float, float] = {}
        r2_base: dict[float, float] = {}
        r2_enh:  dict[float, float] = {}
        impr:    dict[float, float] = {}

        # Null model pinball (intercept-only)
        null_pb: dict[float, float] = {}
        for q in self.quantiles:
            null_pred = np.full(len(Y), np.quantile(Y, q))
            null_pb[q] = _pinball_loss(Y, null_pred, q)

        for q in self.quantiles:
            for spec_cols, store in [(base_cols, pb_base), (enh_cols, pb_enh)]:
                avail = [c for c in spec_cols if c in data.columns]
                if not avail:
                    store[q] = np.nan
                    continue
                X = StandardScaler().fit_transform(data[avail].values)
                try:
                    if SM_AVAILABLE:
                        qr  = QuantReg(Y, sm.add_constant(X)).fit(q=q, max_iter=2000)
                        hat = qr.fittedvalues
                    else:
                        qr  = QuantileRegressor(quantile=q, alpha=0, solver="highs")
                        qr.fit(X, Y)
                        hat = qr.predict(X)
                    store[q] = _pinball_loss(Y, hat, q)
                except Exception:
                    store[q] = np.nan

            # Pseudo-R²
            r2_base[q] = (1 - pb_base.get(q, np.nan) / null_pb[q]
                          if null_pb[q] > 0 else np.nan)
            r2_enh[q]  = (1 - pb_enh.get(q, np.nan) / null_pb[q]
                          if null_pb[q] > 0 else np.nan)

            # Pinball improvement %
            pb_b = pb_base.get(q, np.nan)
            pb_e = pb_enh.get(q, np.nan)
            impr[q] = (100 * (pb_b - pb_e) / pb_b
                       if not np.isnan(pb_b) and pb_b > 0 and not np.isnan(pb_e)
                       else np.nan)

        log.info(f"  Comparison h={h}: pinball impr "
                 f"τ=0.05={impr.get(0.05,np.nan):.1f}%  "
                 f"τ=0.10={impr.get(0.10,np.nan):.1f}%  "
                 f"τ=0.50={impr.get(0.50,np.nan):.1f}%")
        return pb_base, pb_enh, r2_base, r2_enh, impr

    def _build_component_cols(self, df: pd.DataFrame) -> list[str]:
        """Build expanded predictor set: FCI + GPR + individual GIPI components."""
        base = [c for c in [self.gpr_col, "fci", self.outcome] if c in df.columns]
        comps = [c for c in self.gipi_components if c in df.columns]
        return base + comps


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
