import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
 
# ── DIRECTORIES ───────────────────────────────────────────────────────────────
INPUT_DIR  = Path("data") / "intermediate"
OUTPUT_DIR = Path("data") / "clean"
 
PROLONGED_GAP_DAYS = 3   # gaps longer than this get flagged as synthetic
ROLLING_WINDOW     = 7   # days for rolling mean imputation
WINSOR_LOW         = 0.01
WINSOR_HIGH        = 0.99
 
# Variables where missing = "not used that day" → fill with 0
ZERO_FILL_VARS = [
    "appCat.builtin", "appCat.communication", "appCat.entertainment",
    "appCat.finance", "appCat.game", "appCat.office", "appCat.other",
    "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities",
    "appCat.weather",
]
 
# Variables that are genuinely missing sensor/self-report data → time-series impute
TIMESERIES_VARS = [
    "mood", "circumplex.arousal", "circumplex.valence",
    "screen", "activity",
]
 
# Variables to re-encode as binary presence flags (always 1 when present)
BINARY_PRESENCE_VARS = ["call", "sms"]
 
# Variables to Winsorize (heavy-tailed appCat durations + screen)
WINSORISE_VARS = [
    "appCat.builtin", "appCat.communication", "appCat.entertainment",
    "appCat.finance", "appCat.game", "appCat.office", "appCat.other",
    "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities",
    "appCat.weather", "screen", "activity",
]
 
 
def _setup_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
 
def ipath(name):
    return INPUT_DIR / name
 
 
def opath(name):
    return OUTPUT_DIR / name
 
 
# ── STEP 1: REMOVE HARD ERRORS ────────────────────────────────────────────────
def remove_hard_errors(df):
    """Remove physically impossible values (e.g. negative durations)."""
    print("\n── Step 1: Remove hard errors")
    before = len(df)
 
    # Duration columns cannot be negative
    duration_cols = ZERO_FILL_VARS + ["screen", "activity"]
    for col in duration_cols:
        if col in df.columns:
            n = (df[col] < 0).sum()
            if n > 0:
                print(f"   {col}: setting {n} negative values to NaN")
                df.loc[df[col] < 0, col] = np.nan
 
    print(f"   Rows before: {before:,}  →  after: {len(df):,}")
    return df
 
 
# ── STEP 2: WINSORIZE ─────────────────────────────────────────────────────────
def winsorise(df):
    """Cap extreme values at 1st/99th percentile per variable (not per subject)."""
    print("\n── Step 2: Winsorize heavy-tailed variables")
    for col in WINSORISE_VARS:
        if col not in df.columns:
            continue
        lo = df[col].quantile(WINSOR_LOW)
        hi = df[col].quantile(WINSOR_HIGH)
        n_lo = (df[col] < lo).sum()
        n_hi = (df[col] > hi).sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        print(f"   {col:<35} clipped [{lo:.3f}, {hi:.3f}]  "
              f"(low={n_lo}, high={n_hi})")
    return df
 
 
# ── STEP 3: BINARY RE-ENCODING ────────────────────────────────────────────────
def encode_binary_presence(df):
    """Re-encode call/sms as 0/1 presence flags instead of constant-1 / NaN."""
    print("\n── Step 3: Re-encode binary presence variables (call, sms)")
    for col in BINARY_PRESENCE_VARS:
        if col not in df.columns:
            continue
        df[col] = df[col].notna().astype(int)
        print(f"   {col}: 1 if observed that day, 0 otherwise  "
              f"(1s: {df[col].sum()}, 0s: {(df[col]==0).sum()})")
    return df
 
 
# ── STEP 4: ZERO-FILL appCat MISSINGNESS ─────────────────────────────────────
def zero_fill_app_cats(df):
    """Fill missing appCat values with 0 (missing = app not used that day)."""
    print("\n── Step 4: Zero-fill app category variables")
    for col in ZERO_FILL_VARS:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        print(f"   {col:<35} filled {n:>4} NaNs with 0")
    return df
 
 
# ── STEP 5: TIME-SERIES IMPUTATION WITH SYNTHETIC FLAG ────────────────────────
def _fill_series_with_flag(series, window=ROLLING_WINDOW, gap_thresh=PROLONGED_GAP_DAYS):
    """
    For a single subject's time series (indexed by date, sorted):
      - Compute a rolling mean (min_periods=1) over past `window` days
      - Forward-fill gaps using that rolling mean
      - Return (filled_series, synthetic_flag_series)
        where synthetic_flag = 1 for any day that was NaN AND belonged to a
        gap longer than `gap_thresh` days
    """
    filled   = series.copy().astype(float)
    flag     = pd.Series(0, index=series.index, name=series.name + "_synthetic")
 
    # Identify contiguous NaN runs and their lengths
    is_nan   = series.isna()
    # Label each NaN run with a group id
    run_id   = (~is_nan).cumsum()
 
    for gid, group in filled[is_nan].groupby(run_id[is_nan]):
        gap_len = len(group)
        if gap_len > gap_thresh:
            flag.loc[group.index] = 1
 
    # Rolling mean on non-NaN values, then forward-fill NaNs from it
    roll = filled.fillna(method="ffill").rolling(window=window, min_periods=1).mean()
    filled = filled.fillna(roll)
    # Any remaining NaN at the very start (no history): backward fill as last resort
    filled = filled.fillna(method="bfill")
 
    return filled, flag
 
 
def impute_timeseries(df):
    """
    Apply rolling-mean + forward-fill imputation per subject per variable.
    Adds {var}_synthetic columns for gaps > PROLONGED_GAP_DAYS.
    """
    print(f"\n── Step 5: Time-series imputation "
          f"(rolling window={ROLLING_WINDOW}d, gap threshold={PROLONGED_GAP_DAYS}d)")
 
    df = df.sort_values(["id", "date"]).copy()
    synthetic_cols = {}
 
    for col in TIMESERIES_VARS:
        if col not in df.columns:
            continue
 
        filled_all = []
        flag_all   = []
 
        for subj, grp in df.groupby("id"):
            series = grp.set_index("date")[col]
            filled, flag = _fill_series_with_flag(series)
            filled_all.append(filled.rename(col))
            flag_all.append(flag)
 
        filled_series = pd.concat(filled_all).reset_index(drop=True)
        flag_series   = pd.concat(flag_all).reset_index(drop=True)
 
        n_filled    = df[col].isna().sum()
        n_synthetic = flag_series.sum()
        df[col]     = filled_series.values
        synthetic_cols[col + "_synthetic"] = flag_series.values
 
        print(f"   {col:<30} filled {n_filled:>4} NaNs  |  "
              f"{n_synthetic:>3} rows flagged synthetic")
 
    # Add all synthetic flag columns
    for colname, values in synthetic_cols.items():
        df[colname] = values
 
    return df
 
 
# ── STEP 6: REPORT ────────────────────────────────────────────────────────────
def print_report(df_before, df_after):
    print("\n" + "=" * 70)
    print("CLEANING REPORT")
    print("=" * 70)
    print(f"Shape before: {df_before.shape}  →  after: {df_after.shape}")
 
    print("\nRemaining NaNs per column:")
    miss = df_after.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss) == 0:
        print("   None — fully imputed.")
    else:
        for col, n in miss.items():
            print(f"   {col:<40} {n:>4} ({n/len(df_after)*100:.1f}%)")
 
    synth_cols = [c for c in df_after.columns if c.endswith("_synthetic")]
    if synth_cols:
        print("\nSynthetic flag summary (rows marked as highly imputed):")
        for col in synth_cols:
            n = df_after[col].sum()
            print(f"   {col:<45} {int(n):>4} rows ({n/len(df_after)*100:.1f}%)")
 
 
# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────
def run(input_path=None):
    _setup_dirs()
 
    if input_path is None:
        input_path = ipath("df_wide.csv")
 
    print("=" * 70)
    print("DATA CLEANING")
    print("=" * 70)
    print(f"Reading: {input_path}")
 
    df = pd.read_csv(input_path, parse_dates=["date"])
    df_before = df.copy()
 
    df = remove_hard_errors(df)
    df = winsorise(df)
    df = encode_binary_presence(df)
    df = zero_fill_app_cats(df)
    df = impute_timeseries(df)
 
    print_report(df_before, df)
 
    out = opath("df_clean.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
 
    return df
 
 
# ── STANDALONE ENTRY POINT ────────────────────────────────────────────────────
if __name__ == "__main__":
    run()