import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
 
# MAPPEN
PLOTS_DIR  = Path("plots") / "data_exploration"
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("data") / "intermediate"
 
def _setup_dirs():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
def ppath(name): return PLOTS_DIR / name
def dpath(name): return DATA_DIR / name
def opath(name): return OUTPUT_DIR / name
 
 
# DEFINITIES VAN VARIABELE TYPEN
# Ruwe data staat in long format: elke rij is een meting of sessie met tijdstempel.
#
# appCat.* en screen: elke ruwe waarde is de duur van EEN SESSIE in seconden.
#   Aggregeren met SUM om het totaal aantal seconden per dag te krijgen.
#   Harde bovengrens is 86400s (een volledige dag). Elke categoriewaarde hierboven
#   is fysiek onmogelijk. De SUM over ALLE duurkolommen samen kan ook niet boven
#   86400s uitkomen en wordt apart gecontroleerd als plausibiliteitstest.
#
# mood, circumplex.* en activity: elke ruwe waarde is een zelfgerapporteerde score.
#   Aggregeren met MEAN omdat meerdere scores per dag hetzelfde construct meten.
#   Grenzen zijn bepaald door het studieprotocol.
#
# call en sms: elke ruwe waarde is 1 wanneer een gebeurtenis plaatsvond.
#   Aggregeren met SUM om het aantal gebeurtenissen per dag te krijgen.
#   Geen harde bovengrens maar de waarde moet wel >= 0 zijn.
 
DURATION_VARS = [
    "screen",
    "appCat.builtin", "appCat.communication", "appCat.entertainment",
    "appCat.finance",  "appCat.game",          "appCat.office",
    "appCat.other",    "appCat.social",         "appCat.travel",
    "appCat.unknown",  "appCat.utilities",      "appCat.weather",
]
MEAN_VARS  = ["mood", "circumplex.arousal", "circumplex.valence", "activity"]
COUNT_VARS = ["call", "sms"]
 
# PLAUSIBILITEITSGRENZEN
# Elk item: variabele -> (hard minimum, hard maximum, eenheid, onderbouwing)
# hard max = None betekent geen vaste bovengrens voor die variabele.
SANITY_BOUNDS = {
    "mood":                 (1,     10,    "mean (1 tot 10)",    "Studieprotocol schaal"),
    "circumplex.arousal":   (-2,    2,     "mean (-2 tot 2)",    "Studieprotocol schaal"),
    "circumplex.valence":   (-2,    2,     "mean (-2 tot 2)",    "Studieprotocol schaal"),
    "activity":             (0,     1,     "mean (0 tot 1)",     "Studieprotocol schaal"),
    "screen":               (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "call":                 (0,     None,  "aantal/dag",         "Moet niet negatief zijn"),
    "sms":                  (0,     None,  "aantal/dag",         "Moet niet negatief zijn"),
    "appCat.builtin":       (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "appCat.communication": (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "appCat.entertainment": (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "appCat.finance":       (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "appCat.game":          (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "appCat.office":        (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "appCat.other":         (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "appCat.social":        (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "appCat.travel":        (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "appCat.unknown":       (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "appCat.utilities":     (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
    "appCat.weather":       (0,     86400, "totaal sec/dag",     "Kan niet meer dan 24u zijn"),
}
 
SECS_PER_DAY = 86400
 
 
# HOOFDFUNCTIE
def run(csv_path=dpath("raw/dataset_mood_smartphone.csv")):
    _setup_dirs()
    sns.set_theme(style="whitegrid", palette="muted")
 
    # 0. INLADEN
    df_raw = pd.read_csv(csv_path, index_col=0, parse_dates=["time"])
    df_raw["date"] = df_raw["time"].dt.date
 
    print("=" * 70)
    print("SECTION 1 RUWE VORM EN DTYPES")
    print("=" * 70)
    print(f"Rijen:    {len(df_raw):,}")
    print(f"Kolommen: {list(df_raw.columns)}")
    print(f"\nDtypes:\n{df_raw.dtypes}")
    print(f"\nEerste 5 rijen:\n{df_raw.head()}")
 
    # 1. BASISAANTALLEN
    print("\n" + "=" * 70)
    print("SECTION 2 PROEFPERSONEN EN TIJDSBEREIK")
    print("=" * 70)
    subjects = df_raw["id"].unique()
    print(f"Aantal unieke proefpersonen: {len(subjects)}")
    print(f"Proefpersoon IDs: {sorted(subjects)}")
    print(f"\nTijdsbereik: {df_raw['time'].min()} tot {df_raw['time'].max()}")
    print(f"Totaal aantal dagen: {(df_raw['time'].max() - df_raw['time'].min()).days}")
 
    rps = df_raw.groupby("id").size().sort_values(ascending=False)
    print(f"\nRecords per proefpersoon:\n{rps.to_string()}")
    print(f"\nGemiddeld records per proefpersoon: {rps.mean():.1f}")
    print(f"Standaarddeviatie records per proefpersoon: {rps.std():.1f}")
 
    # 2. OVERZICHT VAN VARIABELEN
    print("\n" + "=" * 70)
    print("SECTION 3 OVERZICHT VARIABELEN")
    print("=" * 70)
    variables = df_raw["variable"].unique()
    print(f"Aantal unieke variabelen: {len(variables)}")
    print(f"Variabelen: {sorted(variables)}")
    print(f"\nAantal observaties per variabele:\n{df_raw['variable'].value_counts().to_string()}")
 
    # 3. WIDE PIVOT EEN RIJ PER ID EN DATUM
    # Aggregatie per variabele: SUM voor duren en tellingen, MEAN voor scores.
    # Hier wordt niets verwijderd of aangepast. Dit zijn ruwe geaggregeerde data.
    parts = []
    for var, grp in df_raw.groupby("variable"):
        if var in DURATION_VARS or var in COUNT_VARS:
            agg = grp.groupby(["id", "date"])["value"].sum()
        else:
            agg = grp.groupby(["id", "date"])["value"].mean()
        agg.name = var
        parts.append(agg)
 
    df_wide = pd.concat(parts, axis=1).reset_index()
    df_wide["date"] = pd.to_datetime(df_wide["date"])
 
    print("\n" + "=" * 70)
    print("SECTION 4 WIDE DATASET (geaggregeerd per proefpersoon per dag)")
    print("=" * 70)
    print(f"Vorm: {df_wide.shape}")
    print(f"Kolommen: {list(df_wide.columns)}")
    print(f"\nToegepaste aggregatie:")
    print(f"  SUM  (totaal per dag) -> {DURATION_VARS + COUNT_VARS}")
    print(f"  MEAN (gemiddelde per dag) -> {MEAN_VARS}")
 
    # 4. BESCHRIJVENDE STATISTIEK
    print("\n" + "=" * 70)
    print("SECTION 5 BESCHRIJVENDE STATISTIEK")
    print("=" * 70)
    num_cols = df_wide.select_dtypes(include=np.number).columns.tolist()
    desc = df_wide[num_cols].describe().T
    desc["skewness"] = df_wide[num_cols].skew()
    desc["kurtosis"] = df_wide[num_cols].kurt()
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(desc.to_string())
 
    # 5. ANALYSE VAN ONTBREKENDE WAARDEN
    print("\n" + "=" * 70)
    print("SECTION 6 ONTBREKENDE WAARDEN")
    print("=" * 70)
    miss     = df_wide[num_cols].isnull().sum().sort_values(ascending=False)
    miss_pct = (miss / len(df_wide) * 100).round(2)
    print(pd.DataFrame({"ontbrekend_aantal": miss, "ontbrekend_%": miss_pct}).to_string())
    print(f"\nTotale missingness: {df_wide[num_cols].isnull().mean().mean()*100:.2f}%")
    print("\nMissingness voor mood per proefpersoon (%):")
    print(df_wide.groupby("id")["mood"]
          .apply(lambda x: x.isnull().mean() * 100).round(2).to_string())
 
    # 6. IQR SCAN VAN UITBIJTERS
    # Belangrijk onderscheid voor het rapport:
    # IQR grens is puur statistisch en gebaseerd op de verdeling van de data.
    # Dit heeft geen kennis van wat geldig of betekenisvol is.
    # Harde grenzen zijn domeingestuurd op basis van studieprotocol en fysica.
    # Alleen harde grenzen rechtvaardigen het weggooien van waarden als echte fouten.
    # Voor mood en circumplex ligt de IQR grens BINNEN de geldige protocolschaal.
    # IQR gemarkeerde waarden zijn dus geen fouten maar echte lage stemming of hoge arousal.
    # Verwijderen zou bias introduceren door de dataset kunstmatig smal te maken rond het gemiddelde.
    # Voor appCat duurvariabelen zijn extreme waarden aannemelijk menselijk gedrag.
    # Daarom wordt Winsorisatie toegepast in plaats van verwijdering (Wilcox 2012).
    print("\n" + "=" * 70)
    print("SECTION 7 IQR UITBIJTER SCAN")
    print("=" * 70)
    print("NOOT: IQR markeert statistische extremen op basis van de dataverdeling.")
    print("      Vergelijk met Section 8 harde grenzen om onderscheid te maken tussen")
    print("      'ongebruikelijk maar geldig' en 'fysiek onmogelijk'.\n")
    for col in num_cols:
        s = df_wide[col].dropna()
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = int(((s < lo) | (s > hi)).sum())
        print(f"  {col:<35} bereik=[{s.min():.2f}, {s.max():.2f}]  "
              f"IQR grens=[{lo:.2f}, {hi:.2f}]  "
              f"gemarkeerd={n_out} ({n_out/len(s)*100:.1f}%)")
 
    # 7. PLAUSIBILITEITSCONTROLE MET HARDE FYSIEKE EN PROTOCOLGRENZEN
    # Schendingen hier zijn ondubbelzinnige meetfouten die in de reinigingsstap worden verwijderd.
    print("\n" + "=" * 70)
    print("SECTION 8 PLAUSIBILITEITSCONTROLE HARDE GRENSSCHENDINGEN")
    print("=" * 70)
    print("NOOT: Hier worden geen waarden verwijderd. Schendingen worden gemarkeerd voor de reinigingsstap.\n")
    hdr = (f"  {'Variabele':<26} {'Eenheid':<16} {'HardMin':>8} {'HardMax':>9} "
           f"{'ObsMin':>10} {'ObsMax':>12} {'<Min':>6} {'>Max':>6}  Onderbouwing")
    print(hdr)
    print("  " + "=" * 112)
 
    sanity_rows = []
    for col, (hmin, hmax, unit, rationale) in SANITY_BOUNDS.items():
        if col not in df_wide.columns:
            continue
        s = df_wide[col].dropna()
        obs_min, obs_max = s.min(), s.max()
        n_below = int((s < hmin).sum())
        n_above = int((s > hmax).sum()) if hmax is not None else 0
        flag = "  SCHENDING" if (n_below > 0 or n_above > 0) else ""
        hmax_str = str(hmax) if hmax is not None else "geen"
        print(f"  {col:<26} {unit:<16} {hmin:>8} {hmax_str:>9} "
              f"  {obs_min:>10.2f}   {obs_max:>10.2f}   {n_below:>4}   {n_above:>4}  "
              f"{rationale}{flag}")
        sanity_rows.append({
            "variabele": col, "eenheid": unit,
            "hard_min": hmin,
            "hard_max": hmax if hmax is not None else "geen",
            "obs_min": round(obs_min, 3),
            "obs_max": round(obs_max, 3),
            "n_onder_min": n_below,
            "n_boven_max": n_above,
            "onderbouwing": rationale,
        })
 
    sanity_df = pd.DataFrame(sanity_rows)
 
    # Controle over variabelen heen: som van alle duurkolommen per rij mag niet boven 86400s komen
    print()
    print("  Kruisvariabele controle: som van alle duurkolommen per dagrij")
    dur_cols_present = [c for c in DURATION_VARS if c in df_wide.columns]
    daily_total = df_wide[dur_cols_present].sum(axis=1)
    n_cross = int((daily_total > SECS_PER_DAY).sum())
    print(f"  Maximale gecombineerde duur per dag: {daily_total.max():.0f}s "
          f"({daily_total.max()/3600:.1f}u)")
    print(f"  Rijen waar gecombineerd totaal boven 86400s ligt: {n_cross} "
          f"({'SCHENDING' if n_cross > 0 else 'OK'})")
 
    n_viol_vars = len(sanity_df[(sanity_df["n_onder_min"]>0)|(sanity_df["n_boven_max"]>0)])
    print(f"\n  Samenvatting: {n_viol_vars} variabelen hebben harde grensschendingen per kolom.")
    print(f"               {n_cross} dagrijen overschrijden 86400s als alle duurkolommen worden opgeteld.")
 
    # 8. UITVALANALYSE PER PROEFPERSOON
    # De grote blauwe blok rechts in de missing heatmap is geen willekeurige missingness.
    # Proefpersonen vielen uit de studie en genereerden daarna geen data meer.
    # We verwijderen deze periode NIET globaal want proefpersonen die langer actief bleven
    # zouden dan geldig data verliezen. De per proefpersoon structuur handelt dit
    # automatisch af: vroeg uitgevallen proefpersonen leveren gewoon minder trainingsinstanties.
    print("\n" + "=" * 70)
    print("SECTION 9 UITVALANALYSE PER PROEFPERSOON")
    print("=" * 70)
    print("NOOT: De grote blok missingness rechts in de heatmap is uitval uit de studie,")
    print("      geen willekeurige sensorfout. We kappen de data niet af want proefpersonen")
    print("      die langer actief bleven mogen niet gekort worden.\n")
 
    subject_dates = df_wide.groupby("id")["date"].agg(["min", "max", "count"]).copy()
    subject_dates.columns = ["eerste_dag", "laatste_dag", "actieve_dagrecords"]
    subject_dates["actieve_dagen"] = (subject_dates["laatste_dag"] - subject_dates["eerste_dag"]).dt.days
    subject_dates["mood_missingness_%"] = (
        df_wide.groupby("id")["mood"].apply(lambda x: x.isnull().mean() * 100).round(1)
    )
    subject_dates = subject_dates.sort_values("laatste_dag")
    print(subject_dates.to_string())
 
    studie_einde = df_wide["date"].max()
    subject_dates["dagen_voor_einde_gestopt"] = (studie_einde - subject_dates["laatste_dag"]).dt.days
    vroeg_gestopt = subject_dates[subject_dates["dagen_voor_einde_gestopt"] > 14]
    print(f"\nProefpersonen die meer dan 14 dagen voor studieëinde stopten: {len(vroeg_gestopt)}")
    print(vroeg_gestopt[["laatste_dag", "dagen_voor_einde_gestopt", "mood_missingness_%"]].to_string())
 
    # 9. VERDIEPING OP MOOD
    print("\n" + "=" * 70)
    print("SECTION 10 VERDIEPING OP MOOD")
    print("=" * 70)
    mood = df_wide["mood"].dropna()
    print(f"Aantal: {len(mood)}  Gemiddelde: {mood.mean():.4f}  Mediaan: {mood.median():.4f}  "
          f"Std: {mood.std():.4f}  Min: {mood.min()}  Max: {mood.max()}")
    print(f"Scheefheid: {mood.skew():.4f}  Kurtosis: {mood.kurt():.4f}")
    print(f"\nMood waardentelling:\n{mood.round().value_counts().sort_index().to_string()}")
    print("\nMood statistieken per proefpersoon:")
    print(df_wide.groupby("id")["mood"]
          .agg(["count", "mean", "std", "min", "max"]).round(3).to_string())
 
    # 10. TIJDELIJKE PATRONEN
    print("\n" + "=" * 70)
    print("SECTION 11 TIJDELIJKE PATRONEN")
    print("=" * 70)
    df_wide["dagvandweek"] = df_wide["date"].dt.day_name()
    dow_mood = df_wide.groupby("dagvandweek")["mood"].mean().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    print("Gemiddelde mood per dag van de week:")
    print(dow_mood.round(3).to_string())
 
    # 11. CORRELATIES MET MOOD
    print("\n" + "=" * 70)
    print("SECTION 12 CORRELATIES MET MOOD")
    print("=" * 70)
    corr_mood = (df_wide[num_cols].corr()["mood"]
                 .drop("mood").sort_values(key=abs, ascending=False))
    print(corr_mood.round(4).to_string())
 
    # =========================================================================
    # FIGUREN
    # =========================================================================
 
    subjects_sorted = sorted(subjects)
 
    # Figuur A: moodspreiding plus boxplot per proefpersoon
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_wide["mood"].dropna(), bins=20, edgecolor="white", color="#4C72B0")
    axes[0].set_title("Mood Distribution (1 to 10)")
    axes[0].set_xlabel("Mood"); axes[0].set_ylabel("Count")
    mood_data = [df_wide[df_wide["id"]==s]["mood"].dropna().values for s in subjects_sorted]
    axes[1].boxplot(mood_data, labels=subjects_sorted, vert=True)
    axes[1].set_title("Mood Distribution per Subject")
    axes[1].set_xticklabels(subjects_sorted, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Mood")
    plt.tight_layout()
    plt.savefig(ppath("mood_distribution.png"), dpi=150); plt.close()
    print(f"\nOpgeslagen: {ppath('mood_distribution.png')}")
 
    # Figuur B: heatmap van ontbrekende waarden
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(df_wide[num_cols].isnull().astype(int).T,
                cmap="Blues", cbar=False, xticklabels=False, yticklabels=True, ax=ax)
    ax.set_title("Missing Value Map (columns = day records, rows = variables)")
    ax.set_xlabel("Day Records"); ax.set_ylabel("Variable")
    plt.tight_layout()
    plt.savefig(ppath("missing_heatmap.png"), dpi=150); plt.close()
    print(f"Opgeslagen: {ppath('missing_heatmap.png')}")
 
    # Figuur C: correlatieheatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    corr_mat = df_wide[num_cols].corr()
    sns.heatmap(corr_mat, mask=np.triu(np.ones_like(corr_mat, dtype=bool)),
                annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.3, annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation Matrix (Daily Aggregates)")
    plt.tight_layout()
    plt.savefig(ppath("correlation_heatmap.png"), dpi=150); plt.close()
    print(f"Opgeslagen: {ppath('correlation_heatmap.png')}")
 
    # Figuur D: mood door de tijd per proefpersoon
    n_subj = len(subjects_sorted)
    fig, axes = plt.subplots(n_subj, 1, figsize=(14, n_subj * 2), sharex=False)
    if n_subj == 1: axes = [axes]
    for ax, subj in zip(axes, subjects_sorted):
        sub = df_wide[df_wide["id"] == subj].sort_values("date")
        ax.plot(sub["date"], sub["mood"], marker="o", ms=3, lw=1, color="#4C72B0")
        ax.set_ylim(1, 10)
        ax.set_ylabel(subj, fontsize=7, rotation=0, labelpad=60, va="center")
        ax.tick_params(axis="x", labelsize=7)
    axes[-1].set_xlabel("Date")
    fig.suptitle("Mood Over Time per Subject", y=1.01)
    plt.tight_layout()
    plt.savefig(ppath("mood_timeseries.png"), dpi=150, bbox_inches="tight"); plt.close()
    print(f"Opgeslagen: {ppath('mood_timeseries.png')}")
 
    # Figuur E: verdelingen van sensor en appCat apart
    app_cols_all = [c for c in num_cols if c.startswith("appCat")]
    other_cols   = [c for c in num_cols if not c.startswith("appCat")]
    for group_name, cols in [("sensor", other_cols), ("appCat", app_cols_all)]:
        n = len(cols); ncg = 4; nrg = int(np.ceil(n / ncg))
        fig, axes = plt.subplots(nrg, ncg, figsize=(ncg * 4, nrg * 3))
        axes = axes.flatten()
        for i, col in enumerate(cols):
            axes[i].hist(df_wide[col].dropna(), bins=30, edgecolor="white", color="#DD8452")
            axes[i].set_title(col, fontsize=9); axes[i].set_xlabel("Value", fontsize=7)
        for j in range(i + 1, len(axes)): axes[j].set_visible(False)
        panel_label = "Sensor Variables" if group_name == "sensor" else "App Category Variables"
        fig.suptitle(f"Distributions of {panel_label}", y=1.01)
        plt.tight_layout()
        plt.savefig(ppath(f"distributions_{group_name}.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Opgeslagen: {ppath(f'distributions_{group_name}.png')}")
 
    # Figuur F: mood per weekdag
    fig, ax = plt.subplots(figsize=(8, 4))
    dow_mood.plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_title("Average Mood by Day of Week"); ax.set_ylabel("Average Mood")
    ax.set_ylim(1, 10); plt.tight_layout()
    plt.savefig(ppath("mood_dow.png"), dpi=150); plt.close()
    print(f"Opgeslagen: {ppath('mood_dow.png')}")
 
    # Figuur G: aantallen observaties per variabele per proefpersoon
    obs_counts = df_raw.groupby(["id","variable"]).size().unstack("variable").fillna(0)
    obs_counts.plot(kind="bar", stacked=True, figsize=(14, 5), colormap="tab20")
    plt.title("Observation Counts per Variable per Subject")
    plt.ylabel("Count"); plt.xlabel("Subject")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(ppath("obs_per_variable_subject.png"), dpi=150); plt.close()
    print(f"Opgeslagen: {ppath('obs_per_variable_subject.png')}")
 
    # Figuur H: sensor en score boxplots met harde grenzen eroverheen
    # Doel: laten zien dat IQR gemarkeerde punten voor mood, circumplex en activity
    # BINNEN de protocolschaal vallen en dus NIET verwijderd moeten worden.
    # De IQR grens voor mood ligt op [5.25, 8.85] terwijl de geldige schaal 1 tot 10 is.
    # Een mood van 3 is geen fout maar een depressieve deelnemer op een slechte dag.
    # Gestreepte rode lijnen maken de harde protocolgrenzen zichtbaar op dezelfde as.
    sensor_plot_vars = [c for c in
                        ["mood","circumplex.arousal","circumplex.valence","activity","screen"]
                        if c in df_wide.columns]
    fig, axes = plt.subplots(1, len(sensor_plot_vars),
                             figsize=(4 * len(sensor_plot_vars), 5))
    if len(sensor_plot_vars) == 1: axes = [axes]
    for ax, col in zip(axes, sensor_plot_vars):
        data = df_wide[col].dropna()
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#AEC6E8", color="#2c5f8a"),
                   medianprops=dict(color="#c0392b", linewidth=2),
                   flierprops=dict(marker="o", markerfacecolor="#e67e22",
                                   markersize=5, alpha=0.7, linestyle="none"))
        n_viol = 0
        if col in SANITY_BOUNDS:
            hmin, hmax = SANITY_BOUNDS[col][0], SANITY_BOUNDS[col][1]
            ax.axhline(hmin, color="#c0392b", linestyle="--",
                       linewidth=1.3, label=f"min={hmin}")
            if hmax is not None:
                ax.axhline(hmax, color="#c0392b", linestyle="--",
                           linewidth=1.3, label=f"max={hmax}")
            n_viol = int((data < hmin).sum())
            if hmax is not None:
                n_viol += int((data > hmax).sum())
        suffix = f"\n({n_viol} violations)" if n_viol > 0 else ""
        ax.set_title(col + suffix, fontsize=9,
                     color="#c0392b" if n_viol > 0 else "black")
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.legend(fontsize=7)
    fig.suptitle("Sensor and Scored Variables with Hard Bounds\n"
                 "Orange points = IQR outliers  |  Dashed red = protocol bounds\n"
                 "Key insight: IQR outliers within bounds are VALID data points",
                 y=1.03)
    plt.tight_layout()
    plt.savefig(ppath("boxplots_sensor_bounds.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Opgeslagen: {ppath('boxplots_sensor_bounds.png')}")
 
    # Figuur I: appCat boxplots op logschaal met harde grens van 86400s
    # Logschaal is essentieel want de verdelingen zijn sterk naar rechts scheef
    # met scheefheid tot 19 en kurtosis tot 480. Op lineaire as zijn uitbijters onzichtbaar.
    # Waarden zijn met 1 verhoogd zodat nullen zichtbaar blijven op de logaritmische as.
    # Rode streepjeslijn is de harde grens van 86400s. Rode subtitel betekent een schending.
    # Extreme waarden zoals 10 uur entertainment zijn ongebruikelijk maar aannemelijk gedrag.
    # Daarom worden ze gewinsorieerd in de reinigingsstap en niet verwijderd (Wilcox 2012).
    app_cols_present = [c for c in num_cols if c.startswith("appCat")]
    n = len(app_cols_present); ncg = 4; nrg = int(np.ceil(n / ncg))
    fig, axes = plt.subplots(nrg, ncg, figsize=(ncg * 4, nrg * 4))
    axes = axes.flatten()
    for i, col in enumerate(app_cols_present):
        data = df_wide[col].dropna()
        n_neg   = int((data < 0).sum())
        n_above = int((data > SECS_PER_DAY).sum())
        axes[i].boxplot(data + 1, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#F5CBA7", color="#a04000"),
                        medianprops=dict(color="#c0392b", linewidth=2),
                        flierprops=dict(marker="o", markerfacecolor="#e67e22",
                                        markersize=3, alpha=0.6, linestyle="none"))
        axes[i].set_yscale("log")
        axes[i].axhline(SECS_PER_DAY + 1, color="#c0392b", linestyle="--",
                        linewidth=1.3, label="86400s (24h)")
        title_lines = [col.replace("appCat.", "")]
        if n_neg > 0:   title_lines.append(f"{n_neg} negative!")
        if n_above > 0: title_lines.append(f"{n_above} above 24h!")
        has_viol = n_neg > 0 or n_above > 0
        axes[i].set_title("\n".join(title_lines), fontsize=8,
                          color="#c0392b" if has_viol else "black")
        axes[i].tick_params(axis="x", bottom=False, labelbottom=False)
        axes[i].legend(fontsize=6)
    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    fig.suptitle("App Category Variables on Log Scale (values + 1 to handle zeros)\n"
                 "Dashed red = 86400s hard cap (24h)  |  Orange points = IQR outliers\n"
                 "Red subtitle = variable has hard bound violations",
                 y=1.03)
    plt.tight_layout()
    plt.savefig(ppath("boxplots_appcat_logscale.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Opgeslagen: {ppath('boxplots_appcat_logscale.png')}")
 
    # Figuur J: samenvatting van harde grensschendingen per variabele
    viol_df = sanity_df[
        (sanity_df["n_onder_min"] > 0) | (sanity_df["n_boven_max"] > 0)
    ].copy()
    if len(viol_df) > 0:
        fig, ax = plt.subplots(figsize=(max(6, len(viol_df) * 1.4), 4))
        x = np.arange(len(viol_df)); w = 0.35
        ax.bar(x - w/2, viol_df["n_onder_min"], w,
             label="Below hard minimum (e.g. negative duration)",
               color="#c0392b", alpha=0.85)
        ax.bar(x + w/2, viol_df["n_boven_max"], w,
             label="Above hard maximum (e.g. more than 86400s per day)",
               color="#e67e22", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(viol_df["variabele"], rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Number of Day Records")
        ax.set_title("Hard Bound Violations per Variable\n"
                "(physically impossible values flagged for removal during cleaning)")
        ax.legend(); plt.tight_layout()
        plt.savefig(ppath("hard_bound_violations.png"), dpi=150); plt.close()
        print(f"Opgeslagen: {ppath('hard_bound_violations.png')}")
    else:
        print("Geen harde grensschendingen gevonden, figuur J wordt overgeslagen.")
 
    # Figuur K: verdeling van gecombineerde dagelijkse duur over alle variabelen
    # Waarden boven 86400s zijn fysiek onmogelijk ook als geen losse kolom de grens overschrijdt.
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(daily_total / 3600, bins=50, edgecolor="white", color="#5B9BD5")
    ax.axvline(24, color="#c0392b", linestyle="--",
               linewidth=1.5, label=f"24h hard cap ({n_cross} rows exceed)")
    ax.set_xlabel("Total Screen and App Category Use per Day (hours)")
    ax.set_ylabel("Number of Day Records")
    ax.set_title("Distribution of Combined Daily Duration\n"
                 "(sum of screen and all app category columns per day record)")
    ax.legend(); plt.tight_layout()
    plt.savefig(ppath("daily_total_duration.png"), dpi=150); plt.close()
    print(f"Opgeslagen: {ppath('daily_total_duration.png')}")
 
    # Figuur L: laatste actieve dag per proefpersoon
    # Visualiseert wanneer elke proefpersoon uit de studie viel.
    # De blauwe blok rechts in de missing heatmap komt exact overeen met deze uitvaldata.
    fig, ax = plt.subplots(figsize=(12, 5))
    subject_dates_sorted = subject_dates.sort_values("laatste_dag")
    colors = ["#c0392b" if d > 14 else "#4C72B0"
              for d in subject_dates_sorted["dagen_voor_einde_gestopt"]]
    ax.barh(subject_dates_sorted.index,
            subject_dates_sorted["actieve_dagrecords"],
            color=colors, alpha=0.85)
    ax.set_xlabel("Number of Active Day Records")
    ax.set_title("Active Data Records per Subject\n"
                 "Red = stopped more than 14 days before study end  |  Blue = active until near the end")
    ax.axvline(subject_dates["actieve_dagrecords"].mean(), color="gray",
               linestyle="--", linewidth=1, label="Mean")
    ax.legend(); plt.tight_layout()
    plt.savefig(ppath("subject_dropout.png"), dpi=150); plt.close()
    print(f"Opgeslagen: {ppath('subject_dropout.png')}")
 
    # UITVOER OPSLAAN
    print("UITVOER OPSLAAN")
    df_wide.to_csv(opath("df_wide.csv"), index=False)
    sanity_df.to_csv(opath("sanity_check_table.csv"), index=False)
    subject_dates.to_csv(opath("subject_dates.csv"))
    print(f"Wide data saved:         {opath('df_wide.csv')}")
    print(f"Plausibility table saved: {opath('sanity_check_table.csv')}")
    print(f"Subject date overview saved:  {opath('subject_dates.csv')}")
 
    print("\n" + "=" * 70)
    print("EDA COMPLETED")
    print("=" * 70)
 
    return df_wide
 
 
# seperate startingpoint
if __name__ == "__main__":
    run()