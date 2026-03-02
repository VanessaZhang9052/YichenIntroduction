"""
S&P 500 Futures 40% Intraday VT Index (USD) ER — Backtest 2006–2025
====================================================================
Since intraday VWAP data is not available in this environment, this script
runs a *daily-frequency* equivalent of the methodology:

  Index(t) = Index(t-1) × (1 + w(t-1) × r(t))

  where r(t) = daily excess return of S&P 500 futures
        w(t) = min(σ_target / σ_ewma(t), LeverageCap)

This is the standard daily approximation of the intraday vol-target index
and converges to the intraday version as the rebalancing frequency increases.

Market data is simulated using a regime-switching GARCH(1,1)-inspired model
calibrated to known S&P 500 historical volatility regimes:
  • 2006–2007  : benign (σ ≈ 10 %)
  • 2008–2009  : GFC   (σ ≈ 30–70 %)
  • 2010–2012  : recovery (σ ≈ 20 %)
  • 2013–2019  : low-vol bull (σ ≈ 12–15 %)
  • 2020 Q1    : COVID crash (σ ≈ 60 %)
  • 2020 Q2–   : recovery   (σ ≈ 25 %)
  • 2021–2024  : gradual normalisation
"""

from __future__ import annotations

import math
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (saves to file)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from sp500_futures_intraday_vt_index import (
    SP500FuturesIntradayVTIndex,
    VOL_TARGET,
    LEVERAGE_CAP,
    EWMA_DECAY,
    INIT_WINDOWS,
)

# ── parameters ────────────────────────────────────────────────────────────────

START_DATE = "2006-01-03"
END_DATE   = "2024-12-31"
SEED       = 42

# For daily frequency:
#   annualise variance by × 252 (not × 19 656)
#   use a 21-day EWMA (≈ same spirit as 35-window intraday)
DAILY_EWMA_DECAY    = 1.0 - 2.0 / (21 + 1)   # λ ≈ 0.9130
DAILY_WINDOWS_YEAR  = 252


# ── regime calibration ────────────────────────────────────────────────────────

def build_regime_series(bdays: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    """
    Return a daily annualised-volatility series that mimics known S&P 500 regimes.
    The 'true' vol drives synthetic return generation; the index only *sees* the
    EWMA estimate of realised vol—not this series—so there is no look-ahead.
    """
    dates = pd.DatetimeIndex(bdays)
    vol   = pd.Series(0.12, index=dates)   # base: 12 %

    def _set(start, end, v):
        mask = (dates >= start) & (dates < end)
        vol[mask] = v

    # GFC build-up and crisis
    _set("2007-07-01", "2008-01-01", 0.18)
    _set("2008-01-01", "2008-10-01", 0.35)
    _set("2008-10-01", "2009-04-01", 0.65)   # peak GFC
    _set("2009-04-01", "2010-01-01", 0.30)

    # Eurozone / flash-crash echoes
    _set("2010-01-01", "2011-01-01", 0.22)
    _set("2011-07-01", "2012-01-01", 0.28)
    _set("2012-01-01", "2013-01-01", 0.18)

    # Long low-vol bull market
    _set("2013-01-01", "2018-01-01", 0.12)
    _set("2018-10-01", "2019-01-01", 0.22)   # Q4 2018 selloff
    _set("2019-01-01", "2020-01-01", 0.12)

    # COVID crash
    _set("2020-02-20", "2020-03-25", 0.90)
    _set("2020-03-25", "2020-09-01", 0.35)
    _set("2020-09-01", "2021-01-01", 0.22)

    # 2022 rate-hike bear market
    _set("2021-01-01", "2022-01-01", 0.14)
    _set("2022-01-01", "2022-12-01", 0.28)
    _set("2022-12-01", "2023-07-01", 0.18)
    _set("2023-07-01", "2024-01-01", 0.14)
    _set("2024-01-01", "2025-01-01", 0.13)

    # Add intra-regime noise so vol isn't perfectly flat
    noise = rng.normal(0, 0.01, size=len(vol))
    vol = (vol + noise).clip(lower=0.05)
    return vol


# ── synthetic data generation ─────────────────────────────────────────────────

def simulate_daily_returns(
    bdays: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.Series:
    """
    Simulate daily S&P 500 excess returns using regime-aware vol.
    Includes a mild Sharpe (~0.4) long-run drift minus a 3 % carry drag
    (to approximate the ER / futures basis adjustment).
    """
    regime_vol  = build_regime_series(bdays, rng)
    annual_drift = 0.07    # approximate long-run equity premium
    carry_drag   = 0.03    # rough cost-of-carry (ER vs. TR)
    net_drift    = annual_drift - carry_drag

    daily_drift = net_drift / 252
    daily_vol   = regime_vol / math.sqrt(252)

    raw    = rng.normal(daily_drift, 1.0, size=len(bdays))
    returns = raw * daily_vol.values + daily_drift
    return pd.Series(returns, index=bdays, name="daily_ret")


# ── daily index calculation ───────────────────────────────────────────────────

def run_daily_vt_index(
    daily_rets: pd.Series,
    vol_target: float   = VOL_TARGET,
    leverage_cap: float = LEVERAGE_CAP,
    ewma_decay: float   = DAILY_EWMA_DECAY,
    init_days: int      = INIT_WINDOWS,
    start_level: float  = 100.0,
) -> pd.DataFrame:
    """
    Daily-frequency vol-target index:

        variance(t) = λ·variance(t-1) + (1-λ)·r(t)²
        σ(t)        = √(variance(t) × 252)
        w(t)        = min(σ_target / σ(t), cap)
        Index(t)    = Index(t-1) × (1 + w(t-1) × r(t))
    """
    records   = []
    idx_level = start_level
    variance  = None
    weight    = 0.0
    init_sq   = []

    for date, ret in daily_rets.items():
        # variance update
        if variance is None:
            init_sq.append(ret * ret)
            if len(init_sq) >= init_days:
                variance = float(np.mean(init_sq))
        else:
            variance = ewma_decay * variance + (1.0 - ewma_decay) * ret * ret

        # annualised vol & new weight
        if variance is not None:
            ann_vol   = math.sqrt(variance * DAILY_WINDOWS_YEAR)
            new_weight = min(vol_target / ann_vol, leverage_cap) if ann_vol > 0 else leverage_cap
        else:
            ann_vol    = float("nan")
            new_weight = 0.0

        # index level: use PREVIOUS weight applied to TODAY's return
        idx_level *= 1.0 + weight * ret

        records.append(
            {
                "date":         date,
                "index_level":  idx_level,
                "target_weight": weight,
                "ann_vol":      ann_vol,
                "daily_ret":    ret,
            }
        )
        weight = new_weight

    return pd.DataFrame(records).set_index("date")


# ── benchmarks ────────────────────────────────────────────────────────────────

def unleveraged_index(daily_rets: pd.Series, start_level: float = 100.0) -> pd.Series:
    """Unlevered (1×) S&P 500 Futures ER — the raw underlying."""
    return start_level * (1.0 + daily_rets).cumprod()


def stats(label: str, daily_rets: pd.Series, ann_factor: int = 252) -> None:
    ann_ret  = daily_rets.mean() * ann_factor
    ann_vol  = daily_rets.std()  * math.sqrt(ann_factor)
    sharpe   = ann_ret / ann_vol if ann_vol else float("nan")
    cum      = (1.0 + daily_rets).prod() - 1.0
    drawdown = ((1.0 + daily_rets).cumprod() /
                (1.0 + daily_rets).cumprod().cummax() - 1.0).min()
    print(
        f"  {label:<40s}  ann.ret={ann_ret:>+7.2%}  "
        f"ann.vol={ann_vol:>6.2%}  Sharpe={sharpe:>5.2f}  "
        f"MDD={drawdown:>7.2%}  cum={cum:>+8.2%}"
    )


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_backtest(
    vt_df:      pd.DataFrame,
    raw_index:  pd.Series,
    output_path: str = "backtest_plot.png",
) -> None:
    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
    )
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    dates = vt_df.index

    # ── panel 1: index levels ────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, vt_df["index_level"],
             color="#00d4ff", linewidth=1.2, label="40% VT Index (ER)")
    ax1.plot(dates, raw_index.reindex(dates),
             color="#ff7043", linewidth=0.9, alpha=0.75, label="S&P 500 Fut. ER (1×)")
    ax1.set_yscale("log")
    ax1.set_ylabel("Index level (log scale, base = 100)", color="white")
    ax1.set_title(
        "S&P 500 Futures 40% Intraday VT Index (USD) ER  —  Backtest 2006–2024",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="white", fontsize=9)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x:.0f}")
    )
    ax1.grid(axis="y", color="#222", linewidth=0.5, linestyle="--")
    ax1.grid(axis="x", color="#222", linewidth=0.3, linestyle=":")

    # shade GFC and COVID
    def shade(ax, start, end, label=None, alpha=0.12):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color="#ff4444", alpha=alpha)
        if label:
            mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            ymin, ymax = ax.get_ylim()
            ax.text(mid, ymax * 0.97, label, ha="center", va="top",
                    color="#ff9999", fontsize=7.5)

    shade(ax1, "2008-01-01", "2009-06-30")
    shade(ax1, "2020-02-20", "2020-04-30")
    ax1.annotate("GFC", xy=(pd.Timestamp("2008-10-01"), 30),
                 color="#ff9999", fontsize=8, ha="center")
    ax1.annotate("COVID", xy=(pd.Timestamp("2020-03-15"), 30),
                 color="#ff9999", fontsize=8, ha="center")

    # ── panel 2: leverage (target weight) ────────────────────────────────────
    ax2 = axes[1]
    ax2.fill_between(dates, vt_df["target_weight"],
                     color="#9c27b0", alpha=0.5, linewidth=0)
    ax2.plot(dates, vt_df["target_weight"],
             color="#ce93d8", linewidth=0.8)
    ax2.axhline(LEVERAGE_CAP, color="#ff7043", linewidth=0.8,
                linestyle="--", alpha=0.7, label=f"Cap ({LEVERAGE_CAP:.0f}×)")
    ax2.axhline(1.0, color="#888", linewidth=0.6, linestyle=":")
    ax2.set_ylabel("Leverage (×)", color="white")
    ax2.set_ylim(0, LEVERAGE_CAP + 0.5)
    ax2.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="white", fontsize=8)
    ax2.grid(axis="y", color="#222", linewidth=0.5, linestyle="--")

    # ── panel 3: realised vol estimate ───────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(dates, vt_df["ann_vol"] * 100,
                     color="#0097a7", alpha=0.45, linewidth=0)
    ax3.plot(dates, vt_df["ann_vol"] * 100,
             color="#4dd0e1", linewidth=0.8, label="EWMA vol (ann.)")
    ax3.axhline(VOL_TARGET * 100, color="#ffeb3b", linewidth=1.0,
                linestyle="--", label=f"Target {VOL_TARGET*100:.0f}%")
    ax3.set_ylabel("Realised vol (%)", color="white")
    ax3.set_xlabel("Date", color="white")
    ax3.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="white", fontsize=8)
    ax3.grid(axis="y", color="#222", linewidth=0.5, linestyle="--")

    # x-axis formatting
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_minor_locator(mdates.YearLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0, ha="center")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nPlot saved to: {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng   = np.random.default_rng(SEED)
    bdays = pd.bdate_range(START_DATE, END_DATE)

    print("Generating synthetic S&P 500 daily data (regime-calibrated) …")
    daily_rets = simulate_daily_returns(bdays, rng)

    print("Running vol-target index calculation …")
    vt_df = run_daily_vt_index(daily_rets)

    raw_idx = unleveraged_index(daily_rets, start_level=100.0)

    # ── performance summary ──────────────────────────────────────────────────
    vt_daily_rets  = vt_df["index_level"].pct_change().dropna()
    raw_daily_rets = raw_idx.pct_change().dropna()

    print("\n── Performance Summary (2006–2024) ─────────────────────────────")
    stats("S&P 500 Fut. ER  (1× unlevered)", raw_daily_rets)
    stats("40% Intraday VT Index (USD) ER ", vt_daily_rets)

    print("\n── Final index levels ──────────────────────────────────────────")
    print(f"  S&P 500 Fut. ER  (start=100): {raw_idx.iloc[-1]:.2f}")
    print(f"  40% VT Index     (start=100): {vt_df['index_level'].iloc[-1]:.2f}")
    print(f"  Mean leverage:   {vt_df['target_weight'].mean():.2f}×")
    print(f"  Min  leverage:   {vt_df['target_weight'].min():.2f}×")
    print(f"  Max  leverage:   {vt_df['target_weight'].max():.2f}×")

    # ── plot ─────────────────────────────────────────────────────────────────
    plot_backtest(vt_df, raw_idx, output_path="/home/user/YichenIntroduction/backtest_plot.png")


if __name__ == "__main__":
    main()
