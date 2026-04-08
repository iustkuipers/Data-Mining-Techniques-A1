import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION VARIABLES (EASY TO CHANGE)
# ============================================================================

INPUT_DIR  = Path("data") / "clean"
OUTPUT_DIR = Path("data") / "model"
PLOTS_DIR  = Path("plots") / "features"

# Feature engineering parameters
HISTORY_WINDOW_DAYS = 7          # Days of history to aggregate
MIN_HISTORY_NON_NAN = 3          # Minimum non-NaN days in window to create instance
ENFORCE_SAME_SUBJECT = True      # Never cross subjects in window
TARGET_VAR = "mood"              # Variable to predict (for next day)

# Feature groups to create
MOOD_VARS = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]
DURATION_VARS = ["screen", "appCat.builtin", "appCat.communication", 
                 "appCat.entertainment", "appCat.finance", "appCat.game", 
                 "appCat.office", "appCat.other", "appCat.social", 
                 "appCat.travel", "appCat.unknown", "appCat.utilities", 
                 "appCat.weather"]
EVENT_VARS = ["call", "sms"]

# Aggregation functions for each group
MOOD_STATS = ["mean", "std", "min", "max"]        # Mood-like: variability matters
DURATION_STATS = ["sum", "mean"]                  # Duration: total + daily avg
EVENT_STATS = ["sum"]                             # Events: just count

# Quality tracking
TRACK_SYNTHETIC = True           # Include flag for synthetic/interpolated days
TRACK_MISSINGNESS = True         # Include missingness count in window

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _setup_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def ipath(name): return INPUT_DIR / name
def opath(name): return OUTPUT_DIR / name
def ppath(name): return PLOTS_DIR / name

def aggregate_window(window_df, col, stats):
    """
    Aggregate a column over a window using specified statistics.
    Returns dict of {stat_name: value} pairs.
    """
    if col not in window_df.columns:
        return {}
    
    series = window_df[col].dropna()
    result = {}
    
    for stat in stats:
        feature_name = f"{col}_{HISTORY_WINDOW_DAYS}d_{stat}"
        
        if len(series) == 0:
            result[feature_name] = np.nan
        elif stat == "mean":
            result[feature_name] = series.mean()
        elif stat == "std":
            result[feature_name] = series.std() if len(series) > 1 else 0
        elif stat == "min":
            result[feature_name] = series.min()
        elif stat == "max":
            result[feature_name] = series.max()
        elif stat == "sum":
            result[feature_name] = series.sum()
        else:
            result[feature_name] = np.nan
    
    return result

# ============================================================================
# MAIN FEATURE ENGINEERING
# ============================================================================

def create_feature_dataset(df_clean):
    """
    Create instance-based dataset for ML:
    - Each instance = prediction at day t for target at day t+1
    - Features = aggregation of previous HISTORY_WINDOW_DAYS
    - One instance per subject per day (with sufficient history)
    """
    
    print("=" * 70)
    print("FEATURE ENGINEERING: SLIDING WINDOW AGGREGATION")
    print("=" * 70)
    print(f"History window: {HISTORY_WINDOW_DAYS} days")
    print(f"Target variable: {TARGET_VAR}")
    print()
    
    df_clean = df_clean.sort_values(["id", "date"]).copy()
    df_clean["date"] = pd.to_datetime(df_clean["date"])
    
    instances = []
    n_created = 0
    n_skipped_no_history = 0
    n_skipped_no_target = 0
    
    for subj, subj_df in df_clean.groupby("id", sort=False):
        subj_df = subj_df.sort_values("date").reset_index(drop=True)
        
        # For each day in this subject's timeline
        for t in range(HISTORY_WINDOW_DAYS, len(subj_df) - 1):
            # Window: days [t - HISTORY_WINDOW_DAYS, t - 1] (history)
            # Target: day t + 1 (next day)
            
            history_window = subj_df.iloc[t - HISTORY_WINDOW_DAYS:t].copy()
            target_row = subj_df.iloc[t + 1]
            current_row = subj_df.iloc[t]
            
            # Check if target is available
            if pd.isna(target_row[TARGET_VAR]):
                n_skipped_no_target += 1
                continue
            
            # Check if we have sufficient non-NaN history
            n_target_obs = history_window[TARGET_VAR].notna().sum()
            if n_target_obs < MIN_HISTORY_NON_NAN:
                n_skipped_no_history += 1
                continue
            
            # Build feature vector
            instance = {
                "instance_id": len(instances),
                "subject": subj,
                "date_from": history_window["date"].min(),
                "date_to": history_window["date"].max(),
                "date_prediction": current_row["date"],
                "date_target": target_row["date"],
            }
            
            # Context features
            instance["day_of_week"] = current_row["date"].dayofweek  # 0=Monday
            instance["week_number"] = current_row["date"].isocalendar().week
            
            # Aggregate mood-like variables
            for col in MOOD_VARS:
                instance.update(aggregate_window(history_window, col, MOOD_STATS))
            
            # Aggregate duration variables
            for col in DURATION_VARS:
                instance.update(aggregate_window(history_window, col, DURATION_STATS))
            
            # Aggregate event variables
            for col in EVENT_VARS:
                instance.update(aggregate_window(history_window, col, EVENT_STATS))
            
            # Quality tracking
            if TRACK_SYNTHETIC:
                synthetic_cols = [c for c in history_window.columns if c.endswith("_synthetic")]
                for syn_col in synthetic_cols:
                    feature_name = f"{syn_col}_{HISTORY_WINDOW_DAYS}d_sum"
                    instance[feature_name] = history_window[syn_col].sum()
            
            if TRACK_MISSINGNESS:
                all_data_cols = [c for c in history_window.columns 
                                if not c.startswith(("id", "date")) and not c.endswith("_synthetic")]
                n_cells_total = len(history_window) * len(all_data_cols)
                n_cells_nan = history_window[all_data_cols].isna().sum().sum()
                instance[f"missingness_{HISTORY_WINDOW_DAYS}d_pct"] = (n_cells_nan / n_cells_total * 100)
            
            # TARGET
            instance[f"{TARGET_VAR}_target"] = target_row[TARGET_VAR]
            
            instances.append(instance)
            n_created += 1
    
    df_features = pd.DataFrame(instances)
    
    print()
    print("INSTANCE CREATION SUMMARY")
    print("-" * 70)
    print(f"Instances created:           {n_created}")
    print(f"Skipped (no target):         {n_skipped_no_target}")
    print(f"Skipped (insufficient hist): {n_skipped_no_history}")
    print(f"Total potential instances:   {n_created + n_skipped_no_target + n_skipped_no_history}")
    print()
    print(f"Dataset shape: {df_features.shape}")
    print(f"Columns: {list(df_features.columns)}")
    print()
    
    return df_features


# ============================================================================
# REPORTING AND VISUALIZATION
# ============================================================================

def report_features(df_features):
    """Summarize the feature dataset."""
    print("=" * 70)
    print("FEATURE DATASET REPORT")
    print("=" * 70)
    
    print(f"\nSubjects: {df_features['subject'].nunique()}")
    print(f"Instances per subject:")
    print(df_features.groupby("subject").size().describe().to_string())
    
    print(f"\nDate range: {df_features['date_from'].min()} to {df_features['date_to'].max()}")
    
    # Target statistics
    target_col = f"{TARGET_VAR}_target"
    print(f"\nTarget variable ({target_col}) statistics:")
    print(df_features[target_col].describe().to_string())
    
    # Missing values in features
    feature_cols = [c for c in df_features.columns 
                   if not c.startswith(("instance", "subject", "date", "week", "day_of"))]
    missing = df_features[feature_cols].isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) > 0:
        print(f"\nFeatures with missing values:")
        for col, n in missing.head(10).items():
            print(f"  {col:<50} {n} ({n/len(df_features)*100:.1f}%)")
    else:
        print(f"\nNo missing values in features.")
    
    print()


def plot_features(df_features):
    """Visualize key feature statistics."""
    sns.set_theme(style="whitegrid")
    
    # Plot A: Target distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    target_col = f"{TARGET_VAR}_target"
    ax.hist(df_features[target_col].dropna(), bins=30, color="#4C72B0", edgecolor="white")
    ax.set_xlabel(f"{TARGET_VAR} (next day)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {target_col}")
    plt.tight_layout()
    plt.savefig(ppath("target_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {ppath('target_distribution.png')}")
    
    # Plot B: Instances per subject
    fig, ax = plt.subplots(figsize=(12, 5))
    subj_counts = df_features.groupby("subject").size().sort_values(ascending=False)
    ax.bar(range(len(subj_counts)), subj_counts.values, color="#4C72B0", alpha=0.7, edgecolor="black")
    ax.set_xticks(range(len(subj_counts)))
    ax.set_xticklabels(subj_counts.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of instances")
    ax.set_title("Training instances per subject")
    ax.axhline(subj_counts.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {subj_counts.mean():.0f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(ppath("instances_per_subject.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {ppath('instances_per_subject.png')}")
    
    # Plot C: Day of week distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    dow_counts = df_features["day_of_week"].value_counts().sort_index()
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.bar(dow_counts.index, dow_counts.values, color="#4C72B0", alpha=0.7, edgecolor="black")
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.set_ylabel("Number of instances")
    ax.set_title("Distribution of instances by day of week")
    plt.tight_layout()
    plt.savefig(ppath("instances_by_dow.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {ppath('instances_by_dow.png')}")
    
    print()


# ============================================================================
# MAIN
# ============================================================================

def run(input_path=None):
    _setup_dirs()
    
    if input_path is None:
        input_path = ipath("df_clean.csv")
    
    print()
    print("FEATURE ENGINEERING PIPELINE")
    print(f"Input: {input_path}")
    print()
    
    df_clean = pd.read_csv(input_path, parse_dates=["date"])
    
    df_features = create_feature_dataset(df_clean)
    report_features(df_features)
    
    print("Generating visualizations...")
    plot_features(df_features)
    
    out = opath("features_train.csv")
    df_features.to_csv(out, index=False)
    print(f"Saved: {out}")
    print()
    
    return df_features


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run()
