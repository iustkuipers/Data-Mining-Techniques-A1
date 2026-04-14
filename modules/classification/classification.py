"""
Task 2A: Classification (Advanced)
====================================
Two classifiers predicting next-day mood (Low / Medium / High,
subject-relative bins: below/within/above subject's median ± 0.5σ).

  1. XGBoost + Optuna
       Tabular model on the 1C feature dataset.
       Optuna TPE sampler: Bayesian optimisation samples promising
       hyperparameter regions rather than randomly, converging faster
       (Bergstra et al., 2011).
       Class imbalance handled via sample_weight (balanced).

  2. LSTM + Additive Attention (PyTorch)
       Inherently temporal; trained on raw 5-day daily sequences.
       Attention (Bahdanau et al., 2015) computes a weighted sum over
       all hidden states so the model learns which days are most
       predictive, rather than relying only on the final hidden state.
       Class imbalance handled via CrossEntropyLoss(weight=...).
       ReduceLROnPlateau halves lr after 5 stagnant epochs.
       Gradient clipping (norm=1.0) stabilises LSTM training.

Evaluation — Leave-One-Subject-Out (LOSO) cross-validation:
  Train on all subjects except one, test on the held-out subject.
  Rotate through all subjects and aggregate predictions.

  Why LOSO over temporal 80/20 split?
    With only 25 subjects and ~1100–1800 instances, a per-subject 80/20
    temporal split produces ~230 test instances — too few for stable
    macro-F1 estimates (high variance across random seeds). LOSO uses
    ALL instances for testing exactly once, producing a single stable
    macro-F1 over the full dataset. It is the standard evaluation
    protocol for small-N longitudinal studies in health informatics
    (Saeb et al., 2015; Cao et al., 2020).
    Temporal integrity is preserved: within each fold, training subjects
    contribute all their days and the test subject's days are unseen.

Design choices:
  - Subjects with >50% mood missingness dropped (same as data_feature.py).
  - SEQ_LEN=5 matches the 5-day window in Task 1C for consistency.
  - Optuna is run once on the full training set of each fold with a
    20% internal hold-out (temporal within that fold's training data).
    For efficiency, hyperparameters are tuned on fold 0 only and reused
    across folds — acceptable at this dataset scale.

References:
  Chen & Guestrin (2016). XGBoost. KDD.
  Hochreiter & Schmidhuber (1997). LSTM. Neural Computation 9(8).
  Bahdanau et al. (2015). Neural Machine Translation. ICLR.
  Bergstra et al. (2011). Algorithms for Hyper-Parameter Optimization. NeurIPS.
  Cao et al. (2020). Predicting individual mood. J. Biomed. Inf.
  Kuppens et al. (2010). Emotional inertia. J. Pers. Soc. Psychol.
  Saeb et al. (2015). Mobile Phone Sensor Correlates of Depressive Symptom
    Severity. J. Med. Internet Res.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

# ── paths ─────────────────────────────────────────────────────────────────────
FEATURE_PATH     = Path("data") / "model" / "features_train.csv"
CLEAN_PATH       = Path("data") / "clean"  / "df_clean.csv"
PLOTS_DIR        = Path("plots") / "classification"

CLASS_LABELS     = ["Low", "Medium", "High"]
MAX_MOOD_MISSING = 50.0

# XGBoost / Optuna
XGB_TRIALS       = 50

# LSTM
SEQ_LEN          = 5     # matches 5-day window in Task 1C
LSTM_HIDDEN      = 64    # 32 underfits; 128 overfits at this dataset size
LSTM_DROPOUT     = 0.3
LSTM_EPOCHS      = 60
LSTM_PATIENCE    = 10
LSTM_BATCH       = 32
LSTM_LR          = 1e-3
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── shared helpers ────────────────────────────────────────────────────────────

def _setup():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def _save_cm(y_true, y_pred, title, fname, cmap):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred),
                           display_labels=CLASS_LABELS).plot(
                               ax=ax, colorbar=False, cmap=cmap)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()

def _drop_high_missing(df, id_col="id", mood_col="mood"):
    miss = df.groupby(id_col)[mood_col].apply(lambda x: x.isna().mean() * 100)
    drop = miss[miss > MAX_MOOD_MISSING].index.tolist()
    if drop:
        print(f"  Dropping subjects >{MAX_MOOD_MISSING}% mood missing: {drop}")
        df = df[~df[id_col].isin(drop)].copy()
    return df

def _loso_folds(subjects):
    """Yield (test_subject, train_subjects) for each LOSO fold."""
    subjects = list(subjects)
    for s in subjects:
        yield s, [o for o in subjects if o != s]


# ── XGBoost ───────────────────────────────────────────────────────────────────

def _tune_xgb(Xtr, ytr, sw):
    """Run Optuna on a 20% hold-out of the training data. Returns best_params."""
    n_val = max(1, int(len(Xtr) * 0.20))
    Xtr2, Xval = Xtr[:-n_val], Xtr[-n_val:]
    ytr2, yval = ytr[:-n_val], ytr[-n_val:]
    sw2         = sw[:-n_val]

    def objective(trial):
        m = xgb.XGBClassifier(
            n_estimators     = trial.suggest_int("n_estimators", 100, 500),
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            objective="multi:softmax", num_class=3,
            eval_metric="mlogloss", random_state=42, n_jobs=2, verbosity=0,
        )
        m.fit(Xtr2, ytr2, sample_weight=sw2, verbose=False)
        return f1_score(yval, m.predict(Xval), average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=XGB_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value


def run_xgboost(df_features):
    print("\n── XGBoost + Optuna (LOSO) ──────────────────────────────────")

    df = df_features.dropna(subset=["mood_class"]).copy()
    subjects = sorted(df["subject"].unique())
    print(f"  Subjects: {len(subjects)} | Instances: {len(df)}")

    drop_cols = {"instance_id", "subject", "date_target",
                 "mood_target", "mood_class"}
    feat_cols = [c for c in df.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

    le  = LabelEncoder().fit(CLASS_LABELS)
    # Instead of global SimpleImputer
    df[feat_cols] = df.groupby("subject")[feat_cols].transform(lambda x: x.fillna(x.mean()))
    # Then fill any remaining with 0
    df[feat_cols] = df[feat_cols].fillna(0)

    # Tune hyperparameters once on all-but-first-subject fold
    print(f"  Tuning hyperparameters ({XGB_TRIALS} Optuna trials)...")
    first_test = subjects[0]
    train_init = df[df["subject"] != first_test]
    Xtr_init   = train_init[feat_cols].values
    ytr_init   = le.transform(train_init["mood_class"])
    cw_init    = compute_class_weight("balanced", classes=np.arange(3), y=ytr_init)
    sw_init    = np.array([cw_init[y] for y in ytr_init])
    best_params, best_val_f1 = _tune_xgb(Xtr_init, ytr_init, sw_init)
    print(f"  Best val macro-F1: {best_val_f1:.4f} | params: {best_params}")

    # LOSO evaluation with fixed hyperparameters
    all_true, all_pred = [], []
    feat_importances   = np.zeros(len(feat_cols))

    for test_subj, train_subjs in _loso_folds(subjects):
        train_df = df[df["subject"].isin(train_subjs)]
        test_df  = df[df["subject"] == test_subj]
        if len(test_df) == 0:
            continue

        Xtr = train_df[feat_cols].values
        Xte = test_df[feat_cols].values
        ytr = le.transform(train_df["mood_class"])
        yte = le.transform(test_df["mood_class"])
        cw  = compute_class_weight("balanced", classes=np.arange(3), y=ytr)
        sw  = np.array([cw[y] for y in ytr])

        model = xgb.XGBClassifier(
            **best_params,
            objective="multi:softmax", num_class=3,
            eval_metric="mlogloss", random_state=42, n_jobs=2, verbosity=0,
        )
        model.fit(Xtr, ytr, sample_weight=sw, verbose=False)
        yp = model.predict(Xte)

        all_true.extend(yte)
        all_pred.extend(yp)
        feat_importances += model.feature_importances_

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_LABELS, zero_division=0)

    print(f"  LOSO Accuracy: {acc:.4f}  |  LOSO Macro-F1: {f1:.4f}")
    print(report)

    # Feature importance (averaged across folds)
    avg_imp = pd.Series(feat_importances / len(subjects), index=feat_cols)
    fig, ax = plt.subplots(figsize=(10, 7))
    avg_imp.sort_values().tail(20).plot(kind="barh", ax=ax,
                                        color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Mean gain (averaged over LOSO folds)")
    ax.set_title("XGBoost – Top 20 Feature Importances (LOSO)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgb_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    _save_cm(y_true, y_pred, "XGBoost – Confusion Matrix (LOSO)",
             "xgb_confusion_matrix.png", "Blues")
    print("  Saved: xgb_feature_importance, xgb_confusion_matrix")
    return {"accuracy": acc, "f1_macro": f1}


# ── LSTM + Attention ──────────────────────────────────────────────────────────

class _MoodDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class MoodLSTM(nn.Module):
    """
    LSTM with additive attention (Bahdanau et al., 2015).
    Context = weighted sum of all T hidden states.
    Architecture: LSTM(64) → Attention → Dropout(0.3) → Dense(32) → Dense(3)
    """
    def __init__(self, n_feat):
        super().__init__()
        h = LSTM_HIDDEN
        self.lstm   = nn.LSTM(n_feat, h, batch_first=True)
        self.attn_w = nn.Linear(h, h)
        self.attn_v = nn.Linear(h, 1, bias=False)
        self.drop1  = nn.Dropout(LSTM_DROPOUT)
        self.fc1    = nn.Linear(h, 32)
        self.drop2  = nn.Dropout(0.2)
        self.fc2    = nn.Linear(32, 3)

    def forward(self, x):
        H, _    = self.lstm(x)
        scores  = self.attn_v(torch.tanh(self.attn_w(H)))
        weights = torch.softmax(scores, dim=1)
        context = (weights * H).sum(dim=1)
        return self.fc2(self.drop2(torch.relu(self.fc1(self.drop1(context)))))


def _build_subject_sequences(df_clean):
    """
    Returns dict: subject -> {X: (n, SEQ_LEN, n_feat), y: (n,), dates: (n,)}
    Scaling is done per-fold inside run_lstm to avoid leakage.
    """
    df = df_clean.sort_values(["id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = _drop_high_missing(df, id_col="id", mood_col="mood")

    skip  = {"id", "date", "dayofweek", "week"}
    synth = {c for c in df.columns if c.endswith("_synthetic")}
    fc    = [c for c in df.columns
             if c not in skip | synth
             and pd.api.types.is_numeric_dtype(df[c])]

    sstats = df.groupby("id")["mood"].agg(sm="median", ss="std").reset_index()
    subj_data = {}

    for subj, g in df.groupby("id", sort=False):
        g    = g.sort_values("date").reset_index(drop=True)
        data = g[fc].values.astype(np.float32)
        data[np.isnan(data)] = 0.0
        mood  = g["mood"].values
        dates = g["date"].values
        n     = len(g)

        row = sstats[sstats["id"] == subj].iloc[0]
        s_m = row["sm"]
        s_s = row["ss"] if not np.isnan(row["ss"]) and row["ss"] > 0 else 1.0

        Xs, ys, ds = [], [], []
        for t in range(SEQ_LEN, n - 1):
            tgt = mood[t + 1]
            if np.isnan(tgt):
                continue
            lbl = (0 if tgt < s_m - 0.5*s_s else
                   2 if tgt > s_m + 0.5*s_s else 1)
            Xs.append(data[t - SEQ_LEN:t])
            ys.append(lbl)
            ds.append(dates[t + 1])

        if Xs:
            subj_data[subj] = {
                "X": np.array(Xs, np.float32),
                "y": np.array(ys, np.int64),
            }

    return subj_data, fc


def _train_lstm(Xtr, ytr, n_feat):
    """Train one LSTM fold and return the best model."""
    cw   = compute_class_weight("balanced", classes=np.arange(3), y=ytr)
    cw_t = torch.tensor(cw, dtype=torch.float32).to(DEVICE)

    n_val = max(1, int(len(Xtr) * 0.10))
    full  = _MoodDS(Xtr, ytr)
    tr_ds, val_ds = torch.utils.data.random_split(
        full, [len(Xtr) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    tr_ld  = DataLoader(tr_ds,  LSTM_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    val_ld = DataLoader(val_ds, LSTM_BATCH, shuffle=False)

    model  = MoodLSTM(n_feat).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)
    crit   = nn.CrossEntropyLoss(weight=cw_t)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5)

    best_loss, best_state, wait = float("inf"), None, 0

    for ep in range(1, LSTM_EPOCHS + 1):
        model.train()
        for X, y in tr_ld:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X, y in val_ld:
                X, y = X.to(DEVICE), y.to(DEVICE)
                vl  += crit(model(X), y).item() * len(y)
        vl /= max(len(val_ds), 1)
        sched.step(vl)

        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait       = 0
        else:
            wait += 1
        if wait >= LSTM_PATIENCE:
            break

    model.load_state_dict(best_state)
    return model


def run_lstm(df_clean):
    print("\n── LSTM + Attention (LOSO, PyTorch) ─────────────────────────")

    subj_data, feat_cols = _build_subject_sequences(df_clean)
    subjects = sorted(subj_data.keys())
    n_feat   = subj_data[subjects[0]]["X"].shape[2]

    print(f"  Subjects: {len(subjects)} | Features: {n_feat} | Device: {DEVICE}")
    total_seqs = sum(len(d["y"]) for d in subj_data.values())
    print(f"  Total sequences: {total_seqs}")

    all_true, all_pred = [], []
    tr_losses_all, val_losses_all = [], []

    for fold_i, (test_subj, train_subjs) in enumerate(_loso_folds(subjects)):
        # Collect train sequences
        Xtr = np.concatenate([subj_data[s]["X"] for s in train_subjs])
        ytr = np.concatenate([subj_data[s]["y"] for s in train_subjs])
        Xte = subj_data[test_subj]["X"]
        yte = subj_data[test_subj]["y"]

        if len(Xte) == 0:
            continue

        # Scale on train, apply to test — no leakage
        nf  = Xtr.shape[2]
        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr.reshape(-1, nf)).reshape(-1, SEQ_LEN, nf)
        Xte = sc.transform(Xte.reshape(-1, nf)).reshape(-1, SEQ_LEN, nf)

        model = _train_lstm(Xtr, ytr, n_feat)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(Xte, dtype=torch.float32)
                          .to(DEVICE)).argmax(1).cpu().numpy()

        all_true.extend(yte)
        all_pred.extend(preds)

        fold_f1 = f1_score(yte, preds, average="macro", zero_division=0)
        print(f"  Fold {fold_i+1:>2}/{len(subjects)} "
              f"(test={test_subj}) | n={len(yte):>3} | macro-F1={fold_f1:.3f}")

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_LABELS, zero_division=0)

    print(f"\n  LOSO Accuracy: {acc:.4f}  |  LOSO Macro-F1: {f1:.4f}")
    print(report)

    _save_cm(y_true, y_pred, "LSTM+Attention – Confusion Matrix (LOSO)",
             "lstm_confusion_matrix.png", "Oranges")
    print("  Saved: lstm_confusion_matrix")
    return {"accuracy": acc, "f1_macro": f1}


# ── comparison ────────────────────────────────────────────────────────────────

def _comparison_plot(results):
    names = list(results.keys())
    accs  = [r["accuracy"] for r in results.values()]
    f1s   = [r["f1_macro"] for r in results.values()]
    x, w  = np.arange(len(names)), 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w/2, accs, w, label="Accuracy", color="#4C72B0", alpha=0.85)
    ax.bar(x + w/2, f1s,  w, label="Macro-F1", color="#DD8452", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Classifier Comparison (LOSO CV)")
    ax.axhline(1/3, color="gray", linestyle="--", linewidth=1,
               label="Random baseline (3-class)")
    ax.legend(); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── entry point ───────────────────────────────────────────────────────────────

def run(feature_path=None, clean_path=None):
    _setup()
    df_feat  = pd.read_csv(feature_path or FEATURE_PATH,
                           parse_dates=["date_target"])
    df_clean = pd.read_csv(clean_path or CLEAN_PATH,
                           parse_dates=["date"])

    xgb_r  = run_xgboost(df_feat)
    lstm_r = run_lstm(df_clean)

    results = {"XGBoost": xgb_r, "LSTM+Attention": lstm_r}
    print("\n── Summary ──────────────────────────────────────────────────")
    print(f"{'Model':<20} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 42)
    for name, r in results.items():
        print(f"{name:<20} {r['accuracy']:>10.4f} {r['f1_macro']:>10.4f}")

    _comparison_plot(results)
    print("\n  Saved: model_comparison.png")
    return results


if __name__ == "__main__":
    run()