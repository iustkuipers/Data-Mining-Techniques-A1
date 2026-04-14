"""
Task 1C: Feature Engineering
==============================
Sliding-window aggregation (5-day history) per subject plus:
  - Lag features mood/arousal/valence at t-1..t-5
    Motivation: mood inertia — recent mood dominates next-day prediction
    (Kuppens et al., 2010; Cao et al., 2020).
  - Mood momentum (OLS slope over window)
    Captures rising/falling trend orthogonal to mean and variance.
  - Short-term volatility (3-day std)
    Distinct from window std; captures recent instability.
  - Person-level features (subject mean & std mood over full study)
    Gives the model a personality prior — whether this is generally a
    high- or low-mood subject.
  - Subject-relative class labels (Low/Medium/High relative to each
    subject's own median ± 0.5σ).
    Motivation: intra-individual variation is the clinically meaningful
    signal (Kuppens et al., 2010; Cao et al., 2020).

Design choices vs previous version:
  - Window reduced 7 → 5 days: fewer days required per instance means
    significantly more instances from subjects with sparse mood data.
    Empirically recovered ~600 instances on the real dataset.
  - Lags reduced 1-7 → 1-5 to match the shorter window.
  - Subjects with >50% mood missingness dropped before feature creation:
    AS14.25, AS14.32, AS14.33 contribute almost no usable instances but
    distort per-subject median/std estimates used for class labelling.
  - MIN_VALID_HIST reduced 3 → 2: an instance with 2 observed mood days
    in the 5-day window is noisy but preferable to discarding it entirely
    given the data scarcity.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

INPUT_DIR  = Path("data") / "clean"
OUTPUT_DIR = Path("data") / "model"
PLOTS_DIR  = Path("plots") / "features"

HISTORY_WINDOW   = 5     # days; reduced from 7 to recover more instances
MIN_VALID_HIST   = 1     # min non-NaN mood days in window; reduced from 3
MAX_MOOD_MISSING = 50.0  # drop subjects with more than this % mood missingness
TARGET_VAR       = "mood"

MOOD_VARS     = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]
DURATION_VARS = ["screen", "appCat.builtin", "appCat.communication",
                 "appCat.entertainment", "appCat.finance", "appCat.game",
                 "appCat.office", "appCat.other", "appCat.social",
                 "appCat.travel", "appCat.unknown", "appCat.utilities",
                 "appCat.weather"]
EVENT_VARS    = ["call", "sms"]
LAG_VARS      = ["mood", "circumplex.arousal", "circumplex.valence"]
LAG_DAYS      = list(range(1, HISTORY_WINDOW + 1))  # 1..5


def _setup():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _agg(window, col, stats):
    if col not in window.columns:
        return {}
    s = window[col].dropna()
    out = {}
    for stat in stats:
        key = f"{col}_{HISTORY_WINDOW}d_{stat}"
        if   len(s) == 0:    out[key] = np.nan
        elif stat == "mean": out[key] = s.mean()
        elif stat == "std":  out[key] = s.std() if len(s) > 1 else 0
        elif stat == "min":  out[key] = s.min()
        elif stat == "max":  out[key] = s.max()
        elif stat == "sum":  out[key] = s.sum()
    return out


def _momentum(window, col):
    s = window[col].dropna() if col in window.columns else pd.Series(dtype=float)
    if len(s) < 2:
        return np.nan
    return np.polyfit(np.arange(len(s), dtype=float), s.values, 1)[0]


def _rel_class(val, median, std):
    lo, hi = median - 0.5 * std, median + 0.5 * std
    return "Low" if val < lo else ("High" if val > hi else "Medium")


def _drop_high_missing(df, threshold=MAX_MOOD_MISSING):
    """Remove subjects whose mood missingness exceeds threshold %."""
    miss_pct = (df.groupby("id")["mood"]
                  .apply(lambda x: x.isna().mean() * 100)
                  .reset_index(name="miss_pct"))
    drop = miss_pct[miss_pct["miss_pct"] > threshold]["id"].tolist()
    if drop:
        print(f"  Dropping {len(drop)} subjects with >{threshold}% mood missingness: {drop}")
        df = df[~df["id"].isin(drop)].copy()
    else:
        print(f"  No subjects exceed {threshold}% mood missingness threshold.")
    return df


def create_feature_dataset(df_clean):
    df = df_clean.sort_values(["id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    print(f"  Subjects before missingness filter: {df['id'].nunique()}")
    df = _drop_high_missing(df)
    print(f"  Subjects after filter: {df['id'].nunique()}")

    # Person-level stats over the full (filtered) study period
    pstats = (df.groupby("id")["mood"]
                .agg(subj_mood_mean="mean", subj_mood_std="std")
                .reset_index())
    df = df.merge(pstats, on="id", how="left")

    rows = []
    for subj, sdf in df.groupby("id", sort=False):
        sdf   = sdf.sort_values("date").reset_index(drop=True)
        s_med = sdf["mood"].median()
        s_std = sdf["mood"].std()
        if np.isnan(s_std) or s_std == 0:
            s_std = 1.0

        for t in range(HISTORY_WINDOW, len(sdf) - 1):
            hist   = sdf.iloc[t - HISTORY_WINDOW:t]
            target = sdf.iloc[t + 1]
            cur    = sdf.iloc[t]

            if pd.isna(target[TARGET_VAR]):
                continue
            if hist[TARGET_VAR].notna().sum() < MIN_VALID_HIST:
                continue

            inst = {
                "instance_id":    len(rows),
                "subject":        subj,
                "date_target":    target["date"],
                "day_of_week":    cur["date"].dayofweek,
                "week_number":    cur["date"].isocalendar().week,
                "subj_mood_mean": cur["subj_mood_mean"],
                "subj_mood_std":  cur["subj_mood_std"],
            }

            for col in MOOD_VARS:
                inst.update(_agg(hist, col, ["mean", "std", "min", "max"]))
            for col in DURATION_VARS:
                inst.update(_agg(hist, col, ["sum", "mean"]))
            for col in EVENT_VARS:
                inst.update(_agg(hist, col, ["sum"]))

            for lag in LAG_DAYS:
                idx = t - lag
                for col in LAG_VARS:
                    inst[f"{col}_lag{lag}"] = sdf.iloc[idx][col] if idx >= 0 else np.nan

            inst["mood_momentum_5d"]    = _momentum(hist, "mood")
            inst["arousal_momentum_5d"] = _momentum(hist, "circumplex.arousal")
            inst["mood_std_3d"]         = sdf.iloc[t-3:t]["mood"].std()

            # 1. Weekend Social Interaction
            # Checks if the current day is a weekend (5=Sat, 6=Sun) and multiplies by social usage
            is_weekend = 1 if inst["day_of_week"] >= 5 else 0
            social_sum = inst.get("appCat.social_5d_sum", 0.0)
            inst["interaction_weekend_social"] = is_weekend * (social_sum if pd.notna(social_sum) else 0)
            
            # 2. Communication to Screen Ratio
            # Active communication vs. passive doomscrolling
            comm_sum   = inst.get("appCat.communication_5d_sum", 0.0)
            screen_sum = inst.get("screen_5d_sum", 0.0)
            if pd.notna(screen_sum) and screen_sum > 0:
                inst["interaction_comm_ratio"] = (comm_sum if pd.notna(comm_sum) else 0) / screen_sum
            else:
                inst["interaction_comm_ratio"] = 0.0
                
            # 3. Work/Life Balance Proxy
            # Difference between entertainment and office/utility usage
            ent_sum = inst.get("appCat.entertainment_5d_sum", 0.0)
            off_sum = inst.get("appCat.office_5d_sum", 0.0)
            inst["interaction_ent_vs_work"] = (ent_sum if pd.notna(ent_sum) else 0) - (off_sum if pd.notna(off_sum) else 0)
            
            inst["mood_target"] = target[TARGET_VAR]
            inst["mood_class"]  = _rel_class(target[TARGET_VAR], s_med, s_std)

            inst["missing_sensor_count_5d_mean"] = hist[DURATION_VARS].isna().sum(axis=1).mean()

            rows.append(inst)

    df_out = pd.DataFrame(rows)
    print(f"\n  Instances : {len(df_out)}")
    print(f"  Features  : {df_out.shape[1]}")
    print(f"  Class distribution (subject-relative):")
    print("  " + df_out["mood_class"]
          .value_counts().reindex(["Low", "Medium", "High"]).to_string())
    print()
    return df_out


def plot_features(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].hist(df["mood_target"].dropna(), bins=30, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel("Mood (next day)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Target Distribution")

    vc = df["mood_class"].value_counts().reindex(["Low", "Medium", "High"])
    axes[1].bar(vc.index, vc.values,
                color=["#e74c3c", "#f39c12", "#2ecc71"], alpha=0.85)
    axes[1].set_title("Subject-Relative Class Distribution")
    axes[1].set_ylabel("Count")
    for i, v in enumerate(vc.values):
        axes[1].text(i, v + 3, str(v), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plots/features/feature_overview.png")


def run(input_path=None):
    _setup()
    input_path = input_path or (INPUT_DIR / "df_clean.csv")
    print(f"\nFEATURE ENGINEERING")
    print(f"  Window: {HISTORY_WINDOW}d | Min valid hist: {MIN_VALID_HIST} "
          f"| Max mood missing: {MAX_MOOD_MISSING}%\n")

    df_clean = pd.read_csv(input_path, parse_dates=["date"])
    df_feat  = create_feature_dataset(df_clean)
    plot_features(df_feat)

    out = OUTPUT_DIR / "features_train.csv"
    df_feat.to_csv(out, index=False)
    print(f"  Saved: {out}")
    return df_feat


if __name__ == "__main__":
    run()