import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
 
# MAPPEN
INPUT_DIR  = Path("data") / "intermediate"
OUTPUT_DIR = Path("data") / "clean"
PLOTS_DIR  = Path("plots") / "data_clean"
 
def _setup_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
 
def ipath(name): return INPUT_DIR / name
def opath(name): return OUTPUT_DIR / name
def ppath(name): return PLOTS_DIR / name
 
ZERO_FILL_VARS = [
    "appCat.builtin", "appCat.communication", "appCat.entertainment",
    "appCat.finance", "appCat.game", "appCat.office", "appCat.other",
    "appCat.social", "appCat.travel", "appCat.unknown", "appCat.utilities",
    "appCat.weather", "call", "sms",
]
 
TIMESERIES_VARS = [
    "mood", "circumplex.arousal", "circumplex.valence", "screen", "activity",
]
 
# Alle kolommen die als bewijs van activiteit tellen voor de drempelwaarde check
ACTIVITY_EVIDENCE_COLS = (
    ZERO_FILL_VARS
    + ["screen", "mood", "circumplex.arousal", "circumplex.valence", "activity"]
)
 
PROLONGED_GAP_DAYS = 3
MIN_ACTIVE_COLS    = 2   # minimaal aantal andere kolommen met echte data
FFILL_MAX_DAYS     = 5   # maximum days to forward-fill mood after long gaps
 
 
# STAP 1: HARDE FOUTEN VERWIJDEREN
def step1_remove_hard_errors(df):
    print("Step 1: hard errors")
    n_fixed = 0
    for col in ZERO_FILL_VARS + ["screen", "activity"]:
        if col not in df.columns:
            continue
        mask = df[col] < 0
        if mask.any():
            df.loc[mask, col] = np.nan
            print(f"  {col}: {int(mask.sum())} negative value(s) -> NaN")
            n_fixed += int(mask.sum())
    print(f"  Total fixed: {n_fixed}")
    print()
    return df
 
 
# STAP 2: CONDITIONELE NULVULLING PER RIJ
# Per rij (= één proefpersoon op één dag) tellen we hoeveel ANDERE kolommen
# een echte niet-nul waarde hebben. Als dat er minstens MIN_ACTIVE_COLS zijn
# beschouwen we de proefpersoon als actief op die dag en vullen we NaN met 0.
# Bij minder dan MIN_ACTIVE_COLS actieve kolommen laten we NaN staan.
# Dit werkt per rij dus automatisch per proefpersoon per dag.
def step2_conditional_zero_fill(df):
    print("Step 2: Aggressive zero-fill for duration/event variables")
    
    # If it's an app category, call, or sms, and it's NaN, it means 0 seconds/events.
    for col in ZERO_FILL_VARS:
        if col not in df.columns:
            continue
        
        is_nan = df[col].isna()
        n_filled = int(is_nan.sum())
        
        df.loc[is_nan, col] = 0
        print(f"  {col:<35} filled {n_filled:>4} NaNs with 0")

    print()
    return df
 
 
# STAP 3 EN 4: INTERPOLATIE MET SYNTHETISCHE VLAG
# Per proefpersoon per variabele:
#   - Gebruik datumindex (niet rij-index) voor correcte kalenderdagmeting
#   - Voor elk paar opeenvolgende observaties: als gat <= 3 dagen -> interpoleer
#   - Als gat > 3 dagen -> laat NaN, zet synthetische vlag
#   - Raak nooit rijen aan voor de eerste of na de laatste observatie
#
# Technische aanpak om de write-back bug te vermijden:
#   - Bouw per proefpersoon een volledige datum-geindexeerde Series
#   - Vul die in segmenten
#   - Sla resultaten op in een nieuw DataFrame en merge terug op (id, date)
def step3_4_interpolate_and_flag(df):
    print(f"Step 3-4: interpolation (gap <= {PROLONGED_GAP_DAYS}d) + synthetic flag (gap > {PROLONGED_GAP_DAYS}d)")
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
 
    for col in TIMESERIES_VARS:
        if col not in df.columns:
            continue
 
        flag_col  = col + "_synthetic"
        df[flag_col] = 0
 
        # Werk met een kopie van de kolom als Series geindexeerd op (id, date)
        # zodat de write-back via .loc op (id, date) betrouwbaar werkt
        new_values = df[col].copy()   # behoudt originele integer index
        new_flags  = df[flag_col].copy()
 
        n_interpolated = 0
        n_flagged      = 0
 
        for subj, grp in df.groupby("id", sort=False):
            # Maak een Series geindexeerd op datum (pd.Timestamp)
            s = grp.set_index("date")[col].sort_index()
            # s.index zijn nu Timestamps, s.values zijn de waarden
 
            observed_dates = s[s.notna()].index.tolist()
            if len(observed_dates) < 2:
                continue
 
            for i in range(len(observed_dates) - 1):
                d1       = observed_dates[i]
                d2       = observed_dates[i + 1]
                gap_days = (d2 - d1).days
 
                # Welke rij-indices van df horen bij dit subject en dit datuminterval?
                row_mask = (df["id"] == subj) & (df["date"] >= d1) & (df["date"] <= d2)
                row_idx  = df.index[row_mask]
 
                if gap_days <= PROLONGED_GAP_DAYS:
                    # Kort gat: interpoleer het segment d1 tot d2
                    segment      = s.loc[d1:d2].copy()
                    segment_fill = segment.interpolate(method="index")
 
                    # Schrijf de geinterpoleerde waarden terug via de rij-indices
                    for ridx in row_idx:
                        date_val = df.loc[ridx, "date"]
                        if date_val in segment_fill.index:
                            filled_val = segment_fill.loc[date_val]
                            if pd.isna(new_values.loc[ridx]) and pd.notna(filled_val):
                                new_values.loc[ridx] = filled_val
                                n_interpolated += 1
 
                else:
                    # Lang gat: markeer rijen tussen d1 en d2 (exclusief eindpunten)
                    between_mask = (
                        (df["id"] == subj)
                        & (df["date"] > d1)
                        & (df["date"] < d2)
                    )
                    between_idx = df.index[between_mask]
                    new_flags.loc[between_idx] = 1
                    n_flagged += len(between_idx)
 
        df[col]      = new_values
        df[flag_col] = new_flags
 
        n_still_nan = int(df[col].isna().sum())
        print(f"  {col:<30} {n_interpolated} interpolated, "
              f"{n_flagged} flagged, {n_still_nan} still NaN")
 
    print()
    return df
 
 
# STAP 4B: GECAPPED FORWARD-FILL VOOR RESTERENDE NaN IN TIMESERIES
# Na de interpolatie blijven lange gaten (>3 dagen) als NaN staan.
# Deze gaten zijn grotendeels studieverlaters en sensoruitval, niet willekeurig.
# We vullen ze met forward-fill gecapped op FFILL_MAX_DAYS dagen:
#   - Mood inertia (Kuppens et al., 2010) rechtvaardigt het doortrekken van
#     de laatste bekende waarde als korte-termijn schatting.
#   - Capping op 5 dagen voorkomt dat verouderde waarden te lang worden
#     doorgevoerd; na 5 dagen zonder observatie blijft NaN staan.
#   - Alleen toegepast op TIMESERIES_VARS (mood, arousal, valence, screen,
#     activity) — niet op event/duration variabelen die al 0-gevuld zijn.
#   - Synthetic flag blijft 1 voor forward-filled rijen zodat het model
#     onderscheid kan maken tussen geobserveerde en ingevulde waarden.
def step4b_capped_ffill(df):
    print(f"Step 4b: capped forward-fill (max {FFILL_MAX_DAYS}d) for remaining NaN")
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    for col in TIMESERIES_VARS:
        if col not in df.columns:
            continue
        flag_col  = col + "_synthetic"
        n_filled  = 0

        for subj, grp in df.groupby("id", sort=False):
            idx      = grp.index
            series   = df.loc[idx, col].copy()
            dates    = df.loc[idx, "date"].values

            # Forward-fill with day-count cap per subject
            last_val  = np.nan
            last_date = None

            for i, (ridx, date) in enumerate(zip(idx, dates)):
                val = series[ridx]
                if pd.notna(val):
                    last_val  = val
                    last_date = date
                elif pd.notna(last_val):
                    gap = (pd.Timestamp(date) - pd.Timestamp(last_date)).days
                    if gap <= FFILL_MAX_DAYS:
                        df.loc[ridx, col] = last_val
                        if flag_col in df.columns:
                            df.loc[ridx, flag_col] = 1  # mark as synthetic
                        n_filled += 1
                    # else: leave as NaN — gap too large

        n_still_nan = int(df[col].isna().sum())
        print(f"  {col:<30} {n_filled} forward-filled, {n_still_nan} still NaN")

    print()
    return df


def step4c_subject_median_fill(df):
    print("Step 4c: Filling remaining timeseries NaNs with subject-specific medians")
    
    for col in TIMESERIES_VARS:
        if col not in df.columns:
            continue
            
        # Fill remaining NaNs with the median of THAT specific subject
        df[col] = df.groupby("id")[col].transform(lambda x: x.fillna(x.median()))
        
        n_still_nan = int(df[col].isna().sum())
        print(f"  {col:<30} {n_still_nan} still NaN after subject-median fill")
        
    print()
    return df

# STAP 5: VERIFICATIE
def step5_report(df_before, df_after):
    print("Step 5: verification")
    orig_cols   = [c for c in df_before.columns if c in df_after.columns]
    miss_before = int(df_before[orig_cols].isnull().sum().sum())
    miss_after  = int(df_after[orig_cols].isnull().sum().sum())
    print(f"  NaN cells: {miss_before:,} -> {miss_after:,}")
    print()
 
    data_cols   = [c for c in orig_cols if not c.endswith("_synthetic")]
    remaining   = df_after[data_cols].isnull().sum()
    remaining   = remaining[remaining > 0].sort_values(ascending=False)
    if len(remaining) == 0:
        print("  No NaNs remaining in data columns.")
    else:
        print("  Remaining NaNs:")
        for col, n in remaining.items():
            print(f"    {col:<35} {n:>4} ({n/len(df_after)*100:.1f}%)")
 
    print()
    for col in [c for c in df_after.columns if c.endswith("_synthetic")]:
        n = int(df_after[col].sum())
        print(f"  Flag {col:<40} {n} rows")
    print()
 
 
# VERGELIJKINGSPLOTS
def make_plots(df_before, df_after):
    sns.set_theme(style="whitegrid", palette="muted")
    data_cols_before = [c for c in df_before.select_dtypes(include=np.number).columns
                        if not c.endswith("_synthetic")]
 
    # Plot A: missing heatmap voor en na
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    for ax, df, title in [(axes[0], df_before, "Before cleaning"),
                          (axes[1], df_after,  "After cleaning")]:
        cols = [c for c in data_cols_before if c in df.columns]
        sns.heatmap(df[cols].isnull().astype(int).T,
                    cmap="Blues", cbar=False,
                    xticklabels=False, yticklabels=True, ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Day-records")
    fig.suptitle("Missing Value Map: Before vs After", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(ppath("missing_heatmap_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {ppath('missing_heatmap_comparison.png')}")
 
    # Plot B: missingness percentage per variabele
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, df, title in [(axes[0], df_before, "Before"),
                          (axes[1], df_after,  "After")]:
        cols = [c for c in data_cols_before if c in df.columns]
        pct  = (df[cols].isnull().mean() * 100).sort_values(ascending=True)
        colors = ["#c0392b" if v > 0 else "#4C72B0" for v in pct]
        ax.barh(pct.index, pct.values, color=colors, alpha=0.85)
        ax.set_xlabel("Missing (%)")
        ax.set_xlim(0, 100)
        ax.set_title(title)
    fig.suptitle("Missingness per Variable: Before vs After")
    plt.tight_layout()
    plt.savefig(ppath("missingness_bars.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {ppath('missingness_bars.png')}")
 
    # Plot C: verdelingen voor en na
    fig, axes = plt.subplots(2, len(TIMESERIES_VARS),
                             figsize=(len(TIMESERIES_VARS) * 4, 7))
    for ci, col in enumerate(TIMESERIES_VARS):
        if col not in df_before.columns:
            continue
        for ri, (df, color, label) in enumerate([
            (df_before, "#DD8452", "Before"),
            (df_after,  "#4C72B0", "After"),
        ]):
            ax = axes[ri][ci]
            ax.hist(df[col].dropna(), bins=30, color=color,
                    edgecolor="white", alpha=0.85)
            ax.set_title(f"{col}\n{label}", fontsize=8)
            ax.set_xlabel("value", fontsize=7)
    fig.suptitle("Distributions Before vs After")
    plt.tight_layout()
    plt.savefig(ppath("distributions_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {ppath('distributions_comparison.png')}")
 
    # Plot D: mood tijdreeks na cleaning voor de 6 meest ontbrekende proefpersonen
    most_missing = (df_before.groupby("id")["mood"]
                    .apply(lambda x: x.isna().sum())
                    .sort_values(ascending=False)
                    .head(6).index.tolist())
 
    fig, axes = plt.subplots(len(most_missing), 1,
                             figsize=(14, len(most_missing) * 2.5))
    if len(most_missing) == 1:
        axes = [axes]
 
    for ax, subj in zip(axes, most_missing):
        after = df_after[df_after["id"] == subj].sort_values("date")
        flag  = "mood_synthetic"
 
        ax.plot(after["date"], after["mood"], color="#95a5a6", lw=1, zorder=1)
 
        real = after[after[flag] == 0] if flag in after.columns else after
        ax.scatter(real["date"], real["mood"],
                   color="#4C72B0", s=15, zorder=3, label="observed")
 
        if flag in after.columns:
            synth = after[after[flag] == 1]
            if len(synth):
                ax.scatter(synth["date"], synth["mood"],
                           marker="x", color="#c0392b", s=30,
                           zorder=4, label="synthetic (gap >3d)")
 
        ax.set_ylim(1, 10)
        ax.set_ylabel(subj, fontsize=7, rotation=0, labelpad=60, va="center")
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=6, loc="upper right")
 
    axes[-1].set_xlabel("Date")
    fig.suptitle("Mood After Cleaning  —  blue=observed  grey=interpolated  red=synthetic",
                 y=1.02, fontsize=10)
    plt.tight_layout()
    plt.savefig(ppath("mood_timeseries_after.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {ppath('mood_timeseries_after.png')}")
    print()
 
    # Plot E: 3-color heatmap
    plot_three_color_heatmap(df_before, df_after)
 
    # Plot F: Per-subject activity threshold
    plot_per_subject_activity_threshold(df_after)
 
 
# PLOT E: 3-COLOUR HEATMAP (NaN vs 0 vs Real Values)
def plot_three_color_heatmap(df_before, df_after):
    """
    Show NaN, 0, and real values in different colors.
    White = real value > 0
    Light gray = 0 (filled or original)
    Dark blue = NaN (missing)
    """
    data_cols = [c for c in df_after.select_dtypes(include=np.number).columns
                 if not c.endswith("_synthetic")]
    
    # Create numeric encoding: NaN=2 (blue), 0=1 (gray), >0=0 (white)
    def encode_values(df):
        encoded = np.zeros_like(df[data_cols].values, dtype=float)
        for i, col in enumerate(data_cols):
            encoded[:, i] = df[col].apply(
                lambda x: 2 if pd.isna(x) else (1 if x == 0 else 0)
            ).values
        return encoded
    
    enc_before = encode_values(df_before)
    enc_after = encode_values(df_after)
    
    cmap = plt.cm.colors.ListedColormap(["white", "#d3d3d3", "#4C72B0"])
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    for ax, enc, title in [(axes[0], enc_before, "Before Cleaning"),
                           (axes[1], enc_after, "After Cleaning")]:
        im = ax.imshow(enc.T, cmap=cmap, aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(data_cols)))
        ax.set_yticklabels(data_cols, fontsize=8)
        ax.set_xticks([])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Day-records")
    
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", pad=0.01, shrink=0.8)
    cbar.set_ticks([0.33, 1, 1.67])
    cbar.set_ticklabels(["Real value", "Zero (not used)", "Missing"], fontsize=8)
    
    fig.suptitle("Three-Color Heatmap: NaN vs 0 vs Real Values", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(ppath("heatmap_three_color.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {ppath('heatmap_three_color.png')}")


# PLOT F: PER-SUBJECT ACTIVITY LEVEL WITH THRESHOLD
def plot_per_subject_activity_threshold(df_after):
    """
    For each subject, show how many columns had real data.
    Green = above threshold (≥4, so NaNs get filled)
    Red = below threshold (<4, so NaNs stay NaN)
    """
    data_cols = [c for c in df_after.select_dtypes(include=np.number).columns
                 if not c.endswith("_synthetic")]
    
    # Count real (non-NaN, non-zero) values per row
    df_after["real_count"] = (df_after[data_cols].notna() & (df_after[data_cols] > 0)).sum(axis=1)
    
    subjects = sorted(df_after["id"].unique())
    n_subj = len(subjects)
    
    fig, axes = plt.subplots(n_subj, 1, figsize=(14, n_subj * 1.5), sharex=False)
    if n_subj == 1:
        axes = [axes]
    
    for ax, subj in zip(axes, subjects):
        subj_data = df_after[df_after["id"] == subj].sort_values("date").reset_index(drop=True)
        x = range(len(subj_data))
        
        # Plot as bars, color-coded by threshold
        colors = ["#27ae60" if c >= MIN_ACTIVE_COLS else "#e74c3c"
                  for c in subj_data["real_count"]]
        ax.bar(x, subj_data["real_count"], color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axhline(MIN_ACTIVE_COLS, color="black", linestyle="--", linewidth=1.5,
                   label=f"Threshold ({MIN_ACTIVE_COLS})")
        
        ax.set_ylabel(subj, fontsize=8, rotation=0, labelpad=50)
        ax.set_ylim(0, len(data_cols))
        ax.tick_params(axis="x", labelsize=6)
        ax.legend(fontsize=6, loc="upper right")
    
    axes[-1].set_xlabel("Day-records per subject (sorted by date)")
    fig.suptitle("Activity Level per Subject: Green ≥4 (fill NaN→0)  |  Red <4 (keep NaN)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(ppath("activity_threshold_per_subject.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {ppath('activity_threshold_per_subject.png')}")
    print()


def run(input_path=None):
    _setup_dirs()
 
    if input_path is None:
        input_path = ipath("df_wide.csv")
 
    print()
    print("DATA CLEANING")
    print(f"Input: {input_path}")
    print()
 
    df = pd.read_csv(input_path, parse_dates=["date"])
    df_before = df.copy()
 
    df = step1_remove_hard_errors(df)
    df = step2_conditional_zero_fill(df)
    df = step3_4_interpolate_and_flag(df)
    df = step4b_capped_ffill(df)
    df = step4c_subject_median_fill(df)
    step5_report(df_before, df)
 
    print("Generating plots...")
    make_plots(df_before, df)
 
    out = opath("df_clean.csv")
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
    print()
 
    return df
 
 
# LOS STARTPUNT
if __name__ == "__main__":
    run()