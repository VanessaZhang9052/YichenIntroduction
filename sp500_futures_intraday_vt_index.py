"""
S&P 500 Futures 40% Intraday VT Index (USD) ER
===============================================
Implements the S&P Dow Jones Indices methodology for the
S&P 500 Futures Intraday Volatility Target Index family.

Methodology reference:
  S&P Dow Jones Indices – "S&P 500 Futures Intraday Volatility Target Indices"
  https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-fut-id-vol-tgt-indices.pdf

Key design points (from the official methodology):
  - Underlying: E-mini S&P 500 futures (quarterly contracts)
  - Volatility target: 40% annualised
  - Leverage cap: 4× notional
  - Rebalancing: intraday at the end of every VWAP observation window
  - VWAP windows: left-closed, right-open (trades at start time included,
    trades at end time excluded)
  - Variance initialisation: first 35 intraday return windows (simple average
    of squared returns), then updated via EWMA thereafter
  - Roll: 5 business days before quarterly expiry, executed at the last
    window of that day using the next quarterly contract
  - Missing VWAP: replaced with the last available VWAP for that date/time
  - Return type: Excess Return (ER) – no cash/risk-free component
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default parameters (match the 40% Intraday VT Index specification)
# ---------------------------------------------------------------------------

VOL_TARGET: float = 0.40          # 40% annualised volatility target
LEVERAGE_CAP: float = 4.0         # Maximum allowed leverage
WINDOWS_PER_DAY: int = 78         # 5-minute windows in 9:30–16:00 ET session
TRADING_DAYS_PER_YEAR: int = 252
WINDOWS_PER_YEAR: int = WINDOWS_PER_DAY * TRADING_DAYS_PER_YEAR   # ≈ 19 656
INIT_WINDOWS: int = 35            # Variance initialisation period (windows)
# EWMA decay: λ = 1 – 2/(N+1)  where N = INIT_WINDOWS
EWMA_DECAY: float = 1.0 - 2.0 / (INIT_WINDOWS + 1)               # ≈ 0.9444
ROLL_DAYS_BEFORE_EXPIRY: int = 5  # Business days before expiry to roll


# ---------------------------------------------------------------------------
# Date / roll helpers
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> date:
    """Return the third Friday of a given month."""
    first = date(year, month, 1)
    # weekday(): Monday=0 … Friday=4
    offset = (4 - first.weekday()) % 7      # days to first Friday
    return first + timedelta(days=offset + 14)


def quarterly_expiry_dates(start_year: int, end_year: int) -> list[date]:
    """
    E-mini S&P 500 futures expire on the third Friday of March, June,
    September, and December.
    """
    expiries: list[date] = []
    for year in range(start_year, end_year + 1):
        for month in (3, 6, 9, 12):
            expiries.append(_third_friday(year, month))
    return sorted(expiries)


def build_roll_schedule(
    business_days: pd.DatetimeIndex,
    expiry_dates: list[date],
    roll_days_before: int = ROLL_DAYS_BEFORE_EXPIRY,
) -> pd.DataFrame:
    """
    For each business day determine:
      - active_expiry  : expiry date of the futures contract held
      - next_expiry    : expiry date of the contract to roll into
      - is_roll_date   : True on the day the roll is executed

    Roll date = the business day that is exactly `roll_days_before`
    business days before the expiry.

    Returns a DataFrame indexed by Timestamp with those columns.
    """
    bdays = pd.DatetimeIndex(business_days).normalize()
    bday_list = sorted(bdays.tolist())
    bday_set = set(bday_list)

    rows: list[dict] = []
    prev_roll_ts: pd.Timestamp | None = None

    for idx, expiry in enumerate(expiry_dates):
        expiry_ts = pd.Timestamp(expiry)
        # Business days up to and including expiry
        eligible = [d for d in bday_list if d <= expiry_ts]
        if len(eligible) < roll_days_before + 1:
            continue
        roll_ts = eligible[-(roll_days_before + 1)]
        next_expiry = expiry_dates[idx + 1] if idx + 1 < len(expiry_dates) else None

        # Active window: from the day after previous roll through this roll day
        window_start = (
            bday_list[0] if prev_roll_ts is None
            else bday_list[bday_list.index(prev_roll_ts) + 1]
        )

        for d in bday_list:
            if window_start <= d <= roll_ts:
                rows.append(
                    {
                        "date": d,
                        "active_expiry": expiry,
                        "next_expiry": next_expiry,
                        "is_roll_date": d == roll_ts,
                    }
                )

        prev_roll_ts = roll_ts

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Variance / vol helpers
# ---------------------------------------------------------------------------

def ewma_update(prev_var: float, ret: float, decay: float = EWMA_DECAY) -> float:
    """
    EWMA variance update:
        Var(t,h) = λ · Var(t,h-1) + (1-λ) · r(t,h)²
    """
    return decay * prev_var + (1.0 - decay) * ret * ret


def annualise_vol(per_window_var: float, windows_per_year: int = WINDOWS_PER_YEAR) -> float:
    """σ_annual = √(Var_per_window × windows_per_year)"""
    return math.sqrt(per_window_var * windows_per_year)


def target_weight(
    ann_vol: float,
    vol_target: float = VOL_TARGET,
    leverage_cap: float = LEVERAGE_CAP,
) -> float:
    """
    w = min(σ_target / σ_realised, LeverageCap)

    If realised vol is zero (degenerate), the cap is returned.
    """
    if ann_vol <= 0.0:
        return leverage_cap
    return min(vol_target / ann_vol, leverage_cap)


# ---------------------------------------------------------------------------
# Core index calculator
# ---------------------------------------------------------------------------

class SP500FuturesIntradayVTIndex:
    """
    S&P 500 Futures 40% Intraday VT Index (USD) Excess Return.

    Parameters
    ----------
    vol_target : float
        Annualised volatility target (default 0.40 → 40%).
    leverage_cap : float
        Maximum leverage (default 4.0).
    windows_per_day : int
        Number of intraday VWAP observation windows per trading day
        (default 78, i.e. 5-minute windows in the 9:30–16:00 ET session).
    trading_days_per_year : int
        Approximate number of trading days per year (default 252).
    init_windows : int
        Number of windows used to initialise the variance estimate via a
        simple average of squared returns before switching to EWMA
        (default 35, per the S&P DJI methodology).
    ewma_decay : float | None
        EWMA decay factor λ.  If None, computed as 1 – 2/(init_windows+1).
    roll_days_before_expiry : int
        Business days before quarterly expiry on which the roll is executed
        (default 5, per the S&P DJI methodology).
    """

    def __init__(
        self,
        vol_target: float = VOL_TARGET,
        leverage_cap: float = LEVERAGE_CAP,
        windows_per_day: int = WINDOWS_PER_DAY,
        trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
        init_windows: int = INIT_WINDOWS,
        ewma_decay: float | None = None,
        roll_days_before_expiry: int = ROLL_DAYS_BEFORE_EXPIRY,
    ) -> None:
        self.vol_target = vol_target
        self.leverage_cap = leverage_cap
        self.windows_per_day = windows_per_day
        self.trading_days_per_year = trading_days_per_year
        self.windows_per_year = windows_per_day * trading_days_per_year
        self.init_windows = init_windows
        self.ewma_decay = (
            ewma_decay if ewma_decay is not None
            else 1.0 - 2.0 / (init_windows + 1)
        )
        self.roll_days_before_expiry = roll_days_before_expiry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        vwap_data: pd.DataFrame,
        start_level: float = 100.0,
    ) -> pd.DataFrame:
        """
        Compute intraday index levels from intraday VWAP data.

        Parameters
        ----------
        vwap_data : pd.DataFrame
            Must have a MultiIndex of (date, window_number) and a column
            named ``'vwap'`` containing the VWAP of the E-mini S&P 500
            futures for that observation window.

            - *date*   – ``datetime.date`` (or anything ``pd.Timestamp``
              can parse)
            - *window* – integer 1 … N (1 = first window of the day)
            - *vwap*   – float (NaN → replaced by previous non-NaN VWAP)

        start_level : float
            Index starting value (default 100.0).

        Returns
        -------
        pd.DataFrame
            One row per (date, window) with columns:

            ===============  ===============================================
            index_level      Index value at the *end* of the window
            target_weight    Leverage applied to *this* window's return
            ann_vol          Estimated annualised volatility at window end
            intraday_return  VWAP return for this window
            vwap             VWAP used
            ===============  ===============================================
        """
        vwap_data = vwap_data.sort_index()

        index_level: float = start_level
        variance: float | None = None       # per-window EWMA variance
        current_weight: float = 0.0         # weight applied to next window
        prev_vwap: float | None = None
        init_squared_rets: list[float] = [] # used during initialisation
        window_count: int = 0

        records: list[dict] = []

        for (day, win), row in vwap_data.iterrows():
            vwap_val: float = float(row["vwap"])

            # ── 0. Handle missing VWAP (forward-fill with last known) ──────
            if np.isnan(vwap_val):
                vwap_val = prev_vwap if prev_vwap is not None else vwap_val

            if prev_vwap is None:
                # Very first data point – nothing to compute yet
                prev_vwap = vwap_val
                continue

            # ── 1. Intraday return  r(t,h) = VWAP(t,h)/VWAP(t,h-1) – 1 ──
            ret: float = vwap_val / prev_vwap - 1.0

            # ── 2. Variance update ────────────────────────────────────────
            if variance is None:
                # Initialisation phase: accumulate squared returns
                init_squared_rets.append(ret * ret)
                if len(init_squared_rets) >= self.init_windows:
                    # Initialise variance as simple mean of squared returns
                    variance = float(np.mean(init_squared_rets))
                    window_count = self.init_windows
            else:
                variance = ewma_update(variance, ret, self.ewma_decay)
                window_count += 1

            # ── 3. Annualised vol & new target weight ─────────────────────
            if variance is not None:
                ann_vol = annualise_vol(variance, self.windows_per_year)
                new_weight = target_weight(ann_vol, self.vol_target, self.leverage_cap)
            else:
                ann_vol = float("nan")
                new_weight = 0.0

            # ── 4. Index level update ─────────────────────────────────────
            # Index(t,h) = Index(t,h-1) × (1 + w(t,h-1) × r(t,h))
            # where w(t,h-1) = current_weight (set at end of previous window)
            index_level = index_level * (1.0 + current_weight * ret)

            records.append(
                {
                    "date": day,
                    "window": win,
                    "index_level": index_level,
                    "target_weight": current_weight,
                    "ann_vol": ann_vol,
                    "intraday_return": ret,
                    "vwap": vwap_val,
                }
            )

            # Update state for next window
            current_weight = new_weight
            prev_vwap = vwap_val

        return pd.DataFrame(records)

    def daily_levels(self, intraday: pd.DataFrame) -> pd.DataFrame:
        """
        Extract end-of-day official levels (last window of each day).

        Parameters
        ----------
        intraday : pd.DataFrame
            Output of :meth:`compute`.

        Returns
        -------
        pd.DataFrame
            Columns: date, index_level, target_weight, ann_vol.
        """
        return (
            intraday
            .groupby("date", sort=True)
            .last()
            .reset_index()[["date", "index_level", "target_weight", "ann_vol"]]
        )

    def daily_returns(self, daily: pd.DataFrame) -> pd.Series:
        """
        Compute daily percentage returns from daily index levels.

        Parameters
        ----------
        daily : pd.DataFrame
            Output of :meth:`daily_levels`.

        Returns
        -------
        pd.Series
            Daily returns indexed by date.
        """
        lvl = daily.set_index("date")["index_level"]
        return lvl.pct_change().rename("daily_return")


# ---------------------------------------------------------------------------
# VWAP computation from raw tick / bar data
# ---------------------------------------------------------------------------

def compute_vwap_windows(
    raw_intraday: pd.DataFrame,
    price_col: str = "close",
    volume_col: str = "volume",
    window_minutes: int = 5,
    session_start: str = "09:30",
    session_end: str = "16:00",
) -> pd.DataFrame:
    """
    Aggregate raw intraday trade data into VWAP observation windows.

    Uses the S&P DJI convention of *left-closed, right-open* windows:
    trades at the window start time are included; trades at the window
    end time belong to the next window.

    Parameters
    ----------
    raw_intraday : pd.DataFrame
        DatetimeIndex (timezone-aware or naïve ET timestamps), with columns
        ``price_col`` (trade price) and ``volume_col`` (trade volume/shares).
    price_col : str
        Column containing trade prices.
    volume_col : str
        Column containing trade volumes.
    window_minutes : int
        Window length in minutes (default 5).
    session_start : str
        Start of the regular trading session (default "09:30").
    session_end : str
        End of the regular trading session (default "16:00").

    Returns
    -------
    pd.DataFrame
        MultiIndex (date, window_number), column ``'vwap'``.
        Missing VWAPs (zero-volume windows) are NaN; callers may forward-fill
        them or rely on the index engine's built-in forward-fill.
    """
    df = raw_intraday[[price_col, volume_col]].copy()
    df.index = pd.DatetimeIndex(df.index)

    # Filter to regular session
    df = df.between_time(session_start, session_end)

    # VWAP = Σ(price × volume) / Σ(volume) per window
    def _vwap(grp: pd.DataFrame) -> float:
        pv = (grp[price_col] * grp[volume_col]).sum()
        v = grp[volume_col].sum()
        return pv / v if v > 0 else float("nan")

    resampled = (
        df.resample(f"{window_minutes}min", closed="left", label="left")
        .apply(_vwap)
        .to_frame("vwap")
        .between_time(session_start, session_end)
    )

    # Build (date, window_number) MultiIndex
    resampled["date"] = resampled.index.date
    resampled["window"] = resampled.groupby(resampled.index.date).cumcount() + 1
    return resampled.set_index(["date", "window"])[["vwap"]]


# ---------------------------------------------------------------------------
# Synthetic data example (no external data required)
# ---------------------------------------------------------------------------

def run_example(n_days: int = 252, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Demonstrate the index calculation with synthetic intraday VWAP data.

    Simulates an E-mini futures series with ~20% annual realised volatility
    (so the 40% vol-target index should average ≈2× leverage).

    Parameters
    ----------
    n_days : int
        Number of simulated trading days.
    seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    intraday : pd.DataFrame
        Full intraday index results.
    daily : pd.DataFrame
        End-of-day index levels.
    """
    rng = np.random.default_rng(seed)

    # Per-window vol calibrated to ~20% annualised
    per_window_vol = 0.20 / math.sqrt(WINDOWS_PER_YEAR)

    trading_days = pd.bdate_range("2020-01-02", periods=n_days)
    records: list[dict] = []
    price = 3_000.0

    for day in trading_days:
        for win in range(1, WINDOWS_PER_DAY + 1):
            ret = rng.normal(0.0, per_window_vol)
            price *= 1.0 + ret
            records.append({"date": day.date(), "window": win, "vwap": price})

    vwap_df = pd.DataFrame(records).set_index(["date", "window"])

    engine = SP500FuturesIntradayVTIndex()
    intraday = engine.compute(vwap_df, start_level=100.0)
    daily = engine.daily_levels(intraday)

    return intraday, daily


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    intraday, daily = run_example(n_days=252)

    print("=" * 60)
    print("S&P 500 Futures 40% Intraday VT Index (USD) ER")
    print("Synthetic demonstration (252 trading days)")
    print("=" * 60)

    print("\nFirst 5 intraday rows:")
    print(intraday.head(5).to_string(index=False))

    print("\nEnd-of-day levels (first 10 days):")
    print(daily.head(10).to_string(index=False))

    ann_ret = (daily["index_level"].iloc[-1] / daily["index_level"].iloc[0]) ** (
        252 / len(daily)
    ) - 1
    real_vol = daily.set_index("date")["index_level"].pct_change().std() * math.sqrt(252)

    print(f"\nFinal index level : {daily['index_level'].iloc[-1]:.4f}")
    print(f"Annualised return : {ann_ret:.2%}")
    print(f"Realised ann. vol : {real_vol:.2%}  (target = 40%)")
    print(f"Mean leverage     : {daily['target_weight'].mean():.4f}×")
    print(f"Max leverage      : {daily['target_weight'].max():.4f}×")
