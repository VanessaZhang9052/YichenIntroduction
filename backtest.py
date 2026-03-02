"""
S&P 500 Futures 40% Intraday VT Index (USD) ER — Backtest 2006–2026
====================================================================
Daily-frequency approximation of the intraday vol-target methodology:

    Index(t) = Index(t-1) × (1 + w(t-1) × r(t))
    w(t)     = min(σ_target / σ_ewma(t), LeverageCap)

Data pipeline
─────────────
1. 2006-01-03 → 2011-10-14  real daily SPX (Wes McKinney / PyData book)
2. 2011-10-17 → 2016-02-29  real daily GSPC (Plotly datasets)
   Both sourced from publicly accessible GitHub raw CSV files.

3. 2016-03-01 → 2026-02-28  intra-month Brownian Bridge anchored to the
   monthly S&P 500 levels from the "datasets/s-and-p-500" dataset (also on
   GitHub, updated regularly through the current month).  This guarantees
   every month-end matches the true index level — capturing the 2017 melt-up,
   2020 COVID crash, 2022 bear market, and 2023-24 recovery exactly.
"""

from __future__ import annotations

import io
import math
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sp500_futures_intraday_vt_index import (
    INIT_WINDOWS,
    LEVERAGE_CAP,
    VOL_TARGET,
)

# ── constants ─────────────────────────────────────────────────────────────────

DAILY_EWMA_DECAY   = 1.0 - 2.0 / (21 + 1)   # λ ≈ 0.913 (21-day half-life)
DAILY_WINDOWS_YEAR = 252
SEED               = 42
START_DATE         = "2006-01-03"

# Known approximate annual realised vol for the Brownian-Bridge intra-month
# noise in the 2016-2026 synthetic segment.  These are rounded from public
# VIX/realised-vol records and determine only intra-month path shape, not the
# month-end levels (which come from actual data).
_ANNUAL_VOL_BY_YEAR: dict[int, float] = {
    2016: 0.13,
    2017: 0.07,   # historically lowest-vol year on record
    2018: 0.16,
    2019: 0.12,
    2020: 0.35,   # COVID crash average (peak ~90% in March)
    2021: 0.13,
    2022: 0.25,   # rate-hike bear
    2023: 0.13,
    2024: 0.13,
    2025: 0.17,
    2026: 0.16,
}


# ── data loading ──────────────────────────────────────────────────────────────

def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.read().decode()


def load_daily_real() -> pd.Series:
    """
    Fetch real daily SPX/GSPC closes from two GitHub sources and merge them.

    Source A: Wes McKinney PyData book  (1990-02-01 → 2011-10-14, daily)
    Source B: Plotly datasets           (2007-01-03 → 2016-02-29, daily)

    The two series overlap 2007-2011; Source A takes precedence in that window.
    """
    print("  Fetching daily SPX (McKinney/PyData book) …")
    raw_a = _get(
        "https://raw.githubusercontent.com/wesm/pydata-book/"
        "3rd-edition/examples/spx.csv"
    )
    df_a = pd.read_csv(io.StringIO(raw_a), index_col=0, parse_dates=True)
    df_a.index.name = "date"
    s_a = df_a["SPX"].rename("close")

    print("  Fetching daily GSPC (Plotly datasets) …")
    raw_b = _get(
        "https://raw.githubusercontent.com/plotly/datasets/master/stockdata2.csv"
    )
    df_b = pd.read_csv(io.StringIO(raw_b), parse_dates=["Date"])
    gspc = df_b[df_b["stock"] == "GSPC"][["Date", "value"]].copy()
    gspc = gspc.rename(columns={"Date": "date", "value": "close"}).set_index("date")
    s_b  = gspc["close"]

    # Merge: keep A where available, fill forward from B for the 2011-2016 tail
    combined = s_a.copy()
    for d, v in s_b.items():
        if d not in combined.index:
            combined.loc[d] = v
    combined = combined.sort_index()

    # Filter to our start date
    combined = combined[combined.index >= START_DATE]
    print(f"  Real daily data: {combined.index[0].date()} → {combined.index[-1].date()}  "
          f"({len(combined)} days)")
    return combined


def load_monthly_spx() -> pd.Series:
    """
    Monthly S&P 500 price index (Shiller dataset, updated to current month).
    Returns a Series indexed by period-start dates (first of each month).
    """
    print("  Fetching monthly S&P 500 levels …")
    raw = _get(
        "https://raw.githubusercontent.com/datasets/s-and-p-500/main/data/data.csv"
    )
    df = pd.read_csv(io.StringIO(raw), parse_dates=["Date"])
    df = df[["Date", "SP500"]].dropna().set_index("Date").sort_index()
    df.index.name = "date"
    return df["SP500"]


# ── Brownian-Bridge synthetic daily series ────────────────────────────────────

def brownian_bridge_month(
    start_price: float,
    end_price:   float,
    n_days:      int,
    annual_vol:  float,
    rng:         np.random.Generator,
) -> np.ndarray:
    """
    Generate `n_days` daily log-prices using a Brownian Bridge from
    start_price to end_price with the given annual volatility.

    The path is guaranteed to end exactly at end_price.
    """
    if n_days == 0:
        return np.array([])
    if n_days == 1:
        return np.array([end_price])

    daily_vol   = annual_vol / math.sqrt(DAILY_WINDOWS_YEAR)
    target_logr = math.log(end_price / start_price)

    # Generate noise, adjust so sum equals target
    noise = rng.normal(0.0, daily_vol, size=n_days)
    noise += (target_logr - noise.sum()) / n_days  # bridge correction

    log_prices = math.log(start_price) + np.cumsum(noise)
    return np.exp(log_prices)


def build_synthetic_tail(
    real_daily:   pd.Series,
    monthly_spx:  pd.Series,
    rng:          np.random.Generator,
) -> pd.Series:
    """
    Extend `real_daily` from its last date to the last available monthly
    observation, using Brownian Bridges anchored to real monthly levels.
    """
    real_end   = real_daily.index[-1]
    real_close = real_daily.iloc[-1]

    # Monthly observations strictly after the real-data end
    monthly_tail = monthly_spx[monthly_spx.index > real_end].sort_index()
    if monthly_tail.empty:
        return real_daily.copy()

    # Rescale monthly levels so the series splices seamlessly.
    # The first monthly anchor in the tail is typically month-start of the
    # next month; we normalise the whole tail so the first anchor's
    # *implied* prior month-end matches real_close.
    first_monthly_level = monthly_tail.iloc[0]
    # The McKinney/Plotly data ends mid-month; use the ratio between the
    # last known real price and what the monthly series says for that period
    # (monthly data[month] ≈ close of that month).
    # Find the monthly level for the month containing real_end:
    real_end_month = pd.Timestamp(real_end.year, real_end.month, 1)
    if real_end_month in monthly_spx.index:
        anchor_ratio = real_close / monthly_spx[real_end_month]
    else:
        anchor_ratio = real_close / monthly_tail.iloc[0]

    monthly_prices = monthly_tail * anchor_ratio

    synthetic_dates:  list[pd.Timestamp] = []
    synthetic_prices: list[float]        = []

    prev_price = real_close
    prev_date  = real_end

    months = sorted(monthly_prices.index)
    for m_date in months:
        m_price = monthly_prices[m_date]
        year    = m_date.year

        # Business days in this month (from day after prev_date up to m_date)
        bdays = pd.bdate_range(prev_date + pd.Timedelta(days=1), m_date)
        if len(bdays) == 0:
            prev_price = m_price
            prev_date  = m_date
            continue

        ann_vol = _ANNUAL_VOL_BY_YEAR.get(year, 0.15)
        prices  = brownian_bridge_month(prev_price, m_price, len(bdays), ann_vol, rng)

        synthetic_dates.extend(bdays.tolist())
        synthetic_prices.extend(prices.tolist())

        prev_price = m_price
        prev_date  = m_date

    tail = pd.Series(synthetic_prices, index=pd.DatetimeIndex(synthetic_dates),
                     name="close")
    tail.index.name = "date"
    print(f"  Synthetic tail:  {tail.index[0].date()} → {tail.index[-1].date()}  "
          f"({len(tail)} days)")
    return tail


def build_spx_series(rng: np.random.Generator) -> pd.Series:
    """Return a full daily SPX price series from START_DATE to latest available."""
    real_daily  = load_daily_real()
    monthly_spx = load_monthly_spx()
    tail        = build_synthetic_tail(real_daily, monthly_spx, rng)

    full = pd.concat([real_daily, tail]).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full


# ── daily index calculation ───────────────────────────────────────────────────

def run_daily_vt_index(
    daily_rets:  pd.Series,
    vol_target:  float = VOL_TARGET,
    leverage_cap: float = LEVERAGE_CAP,
    ewma_decay:  float = DAILY_EWMA_DECAY,
    init_days:   int   = INIT_WINDOWS,
    start_level: float = 100.0,
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
    variance: float | None = None
    weight    = 0.0
    init_sq: list[float] = []

    for date, ret in daily_rets.items():
        if variance is None:
            init_sq.append(ret * ret)
            if len(init_sq) >= init_days:
                variance = float(np.mean(init_sq))
        else:
            variance = ewma_decay * variance + (1.0 - ewma_decay) * ret * ret

        if variance is not None:
            ann_vol    = math.sqrt(variance * DAILY_WINDOWS_YEAR)
            new_weight = (min(vol_target / ann_vol, leverage_cap)
                          if ann_vol > 0 else leverage_cap)
        else:
            ann_vol    = float("nan")
            new_weight = 0.0

        idx_level *= 1.0 + weight * ret

        records.append({
            "date":          date,
            "index_level":   idx_level,
            "target_weight": weight,
            "ann_vol":       ann_vol,
            "daily_ret":     ret,
        })
        weight = new_weight

    return pd.DataFrame(records).set_index("date")


# ── stats helper ──────────────────────────────────────────────────────────────

def stats(label: str, daily_rets: pd.Series) -> None:
    ann_ret  = daily_rets.mean() * DAILY_WINDOWS_YEAR
    ann_vol  = daily_rets.std()  * math.sqrt(DAILY_WINDOWS_YEAR)
    sharpe   = ann_ret / ann_vol if ann_vol else float("nan")
    cum      = (1.0 + daily_rets).prod() - 1.0
    mdd      = ((1.0 + daily_rets).cumprod()
                / (1.0 + daily_rets).cumprod().cummax() - 1.0).min()
    print(f"  {label:<42s} ann.ret={ann_ret:>+7.2%}  ann.vol={ann_vol:>6.2%}"
          f"  Sharpe={sharpe:>5.2f}  MDD={mdd:>7.2%}  cum={cum:>+8.2%}")


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_backtest(
    vt_df:      pd.DataFrame,
    raw_index:  pd.Series,
    spx_prices: pd.Series,
    output_path: str = "backtest_plot.png",
) -> None:
    # ── color palette ─────────────────────────────────────────────────────────
    BG      = "#0e1117"
    PANEL   = "#13161f"
    CYAN    = "#00d4ff"
    ORANGE  = "#ff7043"
    PURPLE  = "#ce93d8"
    TEAL    = "#4dd0e1"
    YELLOW  = "#ffeb3b"
    GRID    = "#1e2130"
    RED_SH  = "#ff4444"

    fig, axes = plt.subplots(
        4, 1, figsize=(15, 14), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1.2]},
    )
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(hspace=0.06)

    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors="white", labelsize=8)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3a")

    dates = vt_df.index

    def shade_crisis(ax, start, end):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=RED_SH, alpha=0.10, zorder=0)

    # ── PANEL 1: index levels (log scale) ─────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, vt_df["index_level"], color=CYAN,   lw=1.3, label="40% VT Index (ER)", zorder=3)
    ax1.plot(dates, raw_index.reindex(dates), color=ORANGE, lw=0.9, alpha=0.8,
             label="S&P 500 Price Return (1×)", zorder=2)
    ax1.set_yscale("log")
    ax1.set_ylabel("Index level  (log, base = 100)", color="white", fontsize=9)
    ax1.set_title(
        "S&P 500 Futures 40% Intraday VT Index (USD) ER  —  Backtest 2006–2026\n"
        "Daily data: real 2006–2016 (McKinney/Plotly) · Brownian-Bridge anchored to actual monthly S&P 500 for 2016–2026",
        fontsize=10, fontweight="bold", pad=8,
    )
    ax1.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="white", fontsize=8,
               loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))
    ax1.grid(color=GRID, lw=0.5, ls="--")

    for start, end, label in [
        ("2007-10-01", "2009-06-30", "GFC"),
        ("2020-02-19", "2020-04-30", "COVID"),
        ("2022-01-03", "2022-10-12", "Rate hikes"),
    ]:
        shade_crisis(ax1, start, end)
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax1.text(mid, ax1.get_ylim()[0] * 1.6, label,
                 ha="center", color="#ff9999", fontsize=7.5, style="italic")

    # ── PANEL 2: raw SPX price (to show the underlying faithfully) ────────────
    ax2 = axes[1]
    spx_plot = spx_prices.reindex(dates).ffill()
    ax2.plot(dates, spx_plot, color="#aed6f1", lw=0.9, label="SPX price (real + anchored)")
    ax2.set_ylabel("SPX level", color="white", fontsize=9)
    ax2.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="white", fontsize=8,
               loc="upper left")
    ax2.grid(color=GRID, lw=0.5, ls="--")
    for start, end, _ in [
        ("2007-10-01", "2009-06-30", ""),
        ("2020-02-19", "2020-04-30", ""),
        ("2022-01-03", "2022-10-12", ""),
    ]:
        shade_crisis(ax2, start, end)

    # ── PANEL 3: leverage ─────────────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(dates, vt_df["target_weight"], color="#9c27b0", alpha=0.45, lw=0)
    ax3.plot(dates, vt_df["target_weight"], color=PURPLE, lw=0.8)
    ax3.axhline(LEVERAGE_CAP, color=ORANGE, lw=0.9, ls="--", alpha=0.8,
                label=f"Cap ({LEVERAGE_CAP:.0f}×)")
    ax3.axhline(1.0, color="#888", lw=0.6, ls=":")
    ax3.set_ylim(0, LEVERAGE_CAP + 0.4)
    ax3.set_ylabel("Leverage (×)", color="white", fontsize=9)
    ax3.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="white", fontsize=8)
    ax3.grid(color=GRID, lw=0.5, ls="--")
    for start, end, _ in [
        ("2007-10-01", "2009-06-30", ""),
        ("2020-02-19", "2020-04-30", ""),
        ("2022-01-03", "2022-10-12", ""),
    ]:
        shade_crisis(ax3, start, end)

    # ── PANEL 4: realised vol ─────────────────────────────────────────────────
    ax4 = axes[3]
    ax4.fill_between(dates, vt_df["ann_vol"] * 100, color="#0097a7", alpha=0.4, lw=0)
    ax4.plot(dates, vt_df["ann_vol"] * 100, color=TEAL, lw=0.8, label="EWMA vol (ann.)")
    ax4.axhline(VOL_TARGET * 100, color=YELLOW, lw=1.1, ls="--",
                label=f"Target {VOL_TARGET*100:.0f}%")
    ax4.set_ylabel("Realised vol (%)", color="white", fontsize=9)
    ax4.set_xlabel("Date", color="white", fontsize=9)
    ax4.legend(facecolor="#1a1d27", edgecolor="#444", labelcolor="white", fontsize=8)
    ax4.grid(color=GRID, lw=0.5, ls="--")
    for start, end, _ in [
        ("2007-10-01", "2009-06-30", ""),
        ("2020-02-19", "2020-04-30", ""),
        ("2022-01-03", "2022-10-12", ""),
    ]:
        shade_crisis(ax4, start, end)

    # ── x-axis ────────────────────────────────────────────────────────────────
    ax4.xaxis.set_major_locator(mdates.YearLocator(2))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax4.xaxis.set_minor_locator(mdates.YearLocator())
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0, ha="center", color="white")

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"\nPlot saved → {output_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = np.random.default_rng(SEED)

    print("Loading S&P 500 data …")
    spx_prices = build_spx_series(rng)

    # Daily log-returns (close-to-close price return, proxy for futures ER)
    daily_rets = spx_prices.pct_change().dropna()
    daily_rets.name = "daily_ret"

    print(f"\nFull series: {spx_prices.index[0].date()} → {spx_prices.index[-1].date()}"
          f"  ({len(spx_prices)} days)")

    print("\nRunning 40% VT Index …")
    vt_df = run_daily_vt_index(daily_rets)

    raw_idx = 100.0 * (1.0 + daily_rets).cumprod()

    vt_rets  = vt_df["index_level"].pct_change().dropna()
    raw_rets = raw_idx.pct_change().dropna()

    print("\n── Performance Summary ──────────────────────────────────────────")
    stats("S&P 500 Price Return  (1×)", raw_rets)
    stats("40% Intraday VT Index (USD) ER", vt_rets)

    print("\n── Final levels (rebased to 100 at start) ───────────────────────")
    print(f"  S&P 500 (1×): {raw_idx.iloc[-1]:.1f}")
    print(f"  40% VT Index: {vt_df['index_level'].iloc[-1]:.1f}")
    print(f"  Leverage — mean: {vt_df['target_weight'].mean():.2f}×  "
          f"min: {vt_df['target_weight'].min():.2f}×  "
          f"max: {vt_df['target_weight'].max():.2f}×")

    print("\nGenerating plot …")
    plot_backtest(vt_df, raw_idx, spx_prices,
                  output_path="/home/user/YichenIntroduction/backtest_plot.png")


if __name__ == "__main__":
    main()
