import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
 
# ── DIRECTORIES ───────────────────────────────────────────────────────────────
PLOTS_DIR = Path("plots") / "data_exploration"
DATA_DIR  = Path("data") 
OUTPUT_DIR = Path("data") / "intermediate"
 
def _setup_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
 
def ppath(name):
    return PLOTS_DIR / name
 
 
def dpath(name):
    return DATA_DIR / name

 
def opath(name):
    return OUTPUT_DIR / name
 
 
# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────
def run(csv_path=dpath("raw/dataset_mood_smartphone.csv")):
    _setup_dirs()
 
    # ── 0. LOAD ───────────────────────────────────────────────────────────────
    df_raw = pd.read_csv(csv_path, index_col=0, parse_dates=["time"])
 
    print("=" * 70)
    print("SECTION 1 — RAW SHAPE & DTYPES")
    print("=" * 70)
    print(f"Rows:    {len(df_raw):,}")
    print(f"Columns: {list(df_raw.columns)}")
    print(f"\nDtypes:\n{df_raw.dtypes}")
    print(f"\nFirst 5 rows:\n{df_raw.head()}")
 
    # ── 1. BASIC COUNTS ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 2 — SUBJECTS & TIME RANGE")
    print("=" * 70)
    subjects = df_raw["id"].unique()
    print(f"Number of unique subjects: {len(subjects)}")
    print(f"Subject IDs: {sorted(subjects)}")
    print(f"\nTime range:  {df_raw['time'].min()}  ->  {df_raw['time'].max()}")
    print(f"Total days spanned: {(df_raw['time'].max() - df_raw['time'].min()).days}")
 
    records_per_subject = df_raw.groupby("id").size().sort_values(ascending=False)
    print(f"\nRecords per subject:\n{records_per_subject.to_string()}")
    print(f"\nMean records/subject: {records_per_subject.mean():.1f}")
    print(f"Std  records/subject: {records_per_subject.std():.1f}")
 
    # ── 2. VARIABLE INVENTORY ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 3 — VARIABLE INVENTORY")
    print("=" * 70)
    variables = df_raw["variable"].unique()
    print(f"Number of unique variables: {len(variables)}")
    print(f"Variables: {sorted(variables)}")
    var_counts = df_raw["variable"].value_counts()
    print(f"\nObservation count per variable:\n{var_counts.to_string()}")
 
    # ── 3. WIDE PIVOT — one row per (id, date) ────────────────────────────────
    df_raw["date"] = df_raw["time"].dt.date
    df_wide = (df_raw
               .groupby(["id", "date", "variable"])["value"]
               .mean()
               .unstack("variable")
               .reset_index())
    df_wide["date"] = pd.to_datetime(df_wide["date"])
 
    print("\n" + "=" * 70)
    print("SECTION 4 — WIDE DATASET (per-subject per-day averages)")
    print("=" * 70)
    print(f"Shape: {df_wide.shape}")
    print(f"Columns: {list(df_wide.columns)}")
 
    # ── 4. DESCRIPTIVE STATISTICS ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 5 — DESCRIPTIVE STATISTICS (wide format, numeric cols)")
    print("=" * 70)
    num_cols = df_wide.select_dtypes(include=np.number).columns.tolist()
    desc = df_wide[num_cols].describe().T
    desc["skewness"] = df_wide[num_cols].skew()
    desc["kurtosis"] = df_wide[num_cols].kurt()
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(desc.to_string())
 
    # ── 5. MISSING VALUE ANALYSIS ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 6 — MISSING VALUES (wide format)")
    print("=" * 70)
    miss     = df_wide[num_cols].isnull().sum().sort_values(ascending=False)
    miss_pct = (miss / len(df_wide) * 100).round(2)
    miss_df  = pd.DataFrame({"missing_count": miss, "missing_%": miss_pct})
    print(miss_df.to_string())
    print(f"\nOverall missingness: {df_wide[num_cols].isnull().mean().mean()*100:.2f}%")
    print("\nPer-subject % missing for 'mood':")
    mood_miss = (df_wide.groupby("id")["mood"]
                 .apply(lambda x: x.isnull().mean() * 100).round(2))
    print(mood_miss.to_string())
 
    # ── 6. OUTLIER SCAN ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 7 — OUTLIER SCAN (IQR method, per variable)")
    print("=" * 70)
    for col in num_cols:
        series = df_wide[col].dropna()
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((series < lo) | (series > hi)).sum()
        print(f"  {col:<35} range=[{series.min():.3f}, {series.max():.3f}]  "
              f"IQR_bounds=[{lo:.3f}, {hi:.3f}]  outliers={n_out} ({n_out/len(series)*100:.1f}%)")
 
    # ── 7. MOOD DEEP DIVE ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 8 — MOOD DEEP DIVE")
    print("=" * 70)
    mood = df_wide["mood"].dropna()
    print(f"Count:    {len(mood)}")
    print(f"Mean:     {mood.mean():.4f}")
    print(f"Median:   {mood.median():.4f}")
    print(f"Std:      {mood.std():.4f}")
    print(f"Min/Max:  {mood.min()} / {mood.max()}")
    print(f"Skewness: {mood.skew():.4f}")
    print(f"Kurtosis: {mood.kurt():.4f}")
    print(f"\nMood value_counts:\n{mood.round().value_counts().sort_index().to_string()}")
    print("\nPer-subject mood stats:")
    print(df_wide.groupby("id")["mood"]
          .agg(["count", "mean", "std", "min", "max"])
          .round(3).to_string())
 
    # ── 8. TEMPORAL PATTERNS ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 9 — TEMPORAL PATTERNS")
    print("=" * 70)
    df_wide["dayofweek"] = df_wide["date"].dt.day_name()
    dow_mood = df_wide.groupby("dayofweek")["mood"].mean().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    print("Average mood by day of week:")
    print(dow_mood.round(3).to_string())
    df_wide["week"] = df_wide["date"].dt.isocalendar().week.astype(int)
    print("\nObservations per calendar week (all subjects):")
    print(df_wide.groupby("week").size().to_string())
 
    # ── 9. CORRELATIONS ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SECTION 10 — CORRELATIONS WITH MOOD")
    print("=" * 70)
    corr_with_mood = (df_wide[num_cols]
                      .corr()["mood"]
                      .drop("mood")
                      .sort_values(key=abs, ascending=False))
    print(corr_with_mood.round(4).to_string())
 
    # ── 10. PLOTS ─────────────────────────────────────────────────────────────
    sns.set_theme(style="whitegrid", palette="muted")
 
    # Plot A — mood distribution + per-subject boxplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_wide["mood"].dropna(), bins=20, edgecolor="white", color="#4C72B0")
    axes[0].set_title("Mood Distribution (1-10)")
    axes[0].set_xlabel("Mood")
    axes[0].set_ylabel("Count")
    mood_data = [df_wide[df_wide["id"]==s]["mood"].dropna().values for s in sorted(subjects)]
    axes[1].boxplot(mood_data, labels=sorted(subjects), vert=True)
    axes[1].set_title("Mood Distribution per Subject")
    axes[1].set_xticklabels(sorted(subjects), rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Mood")
    plt.tight_layout()
    plt.savefig(ppath("mood_distribution.png"), dpi=150)
    plt.close()
    print(f"\nSaved: {ppath('mood_distribution.png')}")
 
    # Plot B — missing value heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    miss_matrix = df_wide[num_cols].isnull().astype(int)
    sns.heatmap(miss_matrix.T, cmap="Blues", cbar=False,
                xticklabels=False, yticklabels=True, ax=ax)
    ax.set_title("Missing Value Map (each column = one day-record, rows = variables)")
    ax.set_xlabel("Day-records (sorted by date)")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    plt.savefig(ppath("missing_heatmap.png"), dpi=150)
    plt.close()
    print(f"Saved: {ppath('missing_heatmap.png')}")
 
    # Plot C — correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    corr_mat = df_wide[num_cols].corr()
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.3, annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation Matrix (daily averages)")
    plt.tight_layout()
    plt.savefig(ppath("correlation_heatmap.png"), dpi=150)
    plt.close()
    print(f"Saved: {ppath('correlation_heatmap.png')}")
 
    # Plot D — mood over time per subject
    subjects_sorted = sorted(subjects)
    n_subj = len(subjects_sorted)
    fig, axes = plt.subplots(n_subj, 1, figsize=(14, n_subj * 2), sharex=False)
    if n_subj == 1:
        axes = [axes]
    for ax, subj in zip(axes, subjects_sorted):
        sub = df_wide[df_wide["id"] == subj].sort_values("date")
        ax.plot(sub["date"], sub["mood"], marker="o", ms=3, lw=1, color="#4C72B0")
        ax.set_ylim(1, 10)
        ax.set_ylabel(subj, fontsize=7, rotation=0, labelpad=60, va="center")
        ax.tick_params(axis="x", labelsize=7)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Mood Over Time per Subject", y=1.01)
    plt.tight_layout()
    plt.savefig(ppath("mood_timeseries.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {ppath('mood_timeseries.png')}")
 
    # Plot E — variable distributions (small multiples, split by group)
    app_cols   = [c for c in num_cols if c.startswith("appCat")]
    other_cols = [c for c in num_cols if not c.startswith("appCat")]
    for group_name, cols in [("sensor", other_cols), ("appCat", app_cols)]:
        n = len(cols)
        ncols_grid = 4
        nrows_grid = int(np.ceil(n / ncols_grid))
        fig, axes = plt.subplots(nrows_grid, ncols_grid,
                                 figsize=(ncols_grid * 4, nrows_grid * 3))
        axes = axes.flatten()
        for i, col in enumerate(cols):
            axes[i].hist(df_wide[col].dropna(), bins=30,
                         edgecolor="white", color="#DD8452")
            axes[i].set_title(col, fontsize=9)
            axes[i].set_xlabel("value", fontsize=7)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"Distributions - {group_name} variables", y=1.01)
        plt.tight_layout()
        plt.savefig(ppath(f"distributions_{group_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {ppath(f'distributions_{group_name}.png')}")
 
    # Plot F — mood by day of week
    fig, ax = plt.subplots(figsize=(8, 4))
    dow_mood.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title("Average Mood by Day of Week")
    ax.set_ylabel("Mean Mood")
    ax.set_xlabel("")
    ax.set_ylim(1, 10)
    plt.tight_layout()
    plt.savefig(ppath("mood_dow.png"), dpi=150)
    plt.close()
    print(f"Saved: {ppath('mood_dow.png')}")
 
    # Plot G — observation counts per variable per subject
    obs_counts = (df_raw.groupby(["id", "variable"])
                  .size()
                  .unstack("variable")
                  .fillna(0))
    obs_counts.plot(kind="bar", stacked=True, figsize=(14, 5), colormap="tab20")
    plt.title("Observation Counts per Variable per Subject")
    plt.ylabel("Count")
    plt.xlabel("Subject")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(ppath("obs_per_variable_subject.png"), dpi=150)
    plt.close()
    print(f"Saved: {ppath('obs_per_variable_subject.png')}")
 
    # ── SAVE PROCESSED DATA ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)
    df_wide.to_csv(opath("df_wide.csv"), index=False)
    print(f"Saved: {opath('df_wide.csv')}")
 
    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)
 
    return df_wide
 
 
# ── STANDALONE ENTRY POINT ────────────────────────────────────────────────────
if __name__ == "__main__":
    run()