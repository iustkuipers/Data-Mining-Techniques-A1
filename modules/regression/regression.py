"""
Task 4: Numerical Prediction (Advanced)
========================================
Two regressors predicting next-day mood as a *continuous* value
(same target as Task 2A but regression, not classification).

  1. XGBoost Regressor + Optuna
       Tabular model on the 1C feature dataset.
       Optuna TPE sampler: Bayesian optimisation converges faster than
       random search (Bergstra et al., 2011).
       No class-weighting needed; MSE loss handles continuous targets.

  2. LSTM + Additive Attention (PyTorch) — inherently temporal
       Trained on raw 5-day daily sequences (SEQ_LEN=5).
       Attention (Bahdanau et al., 2015) computes a weighted sum over
       all hidden states; the model learns which days are most predictive.
       Output: single linear neuron (no activation) for continuous prediction.
       ReduceLROnPlateau halves lr after 5 stagnant epochs.
       Gradient clipping (norm=1.0) stabilises LSTM training.

Key differences from Task 2A (classification):
  - Target is raw mood score, not a binned label.
  - Loss: MSELoss (regression) instead of CrossEntropyLoss.
  - Metrics: MAE and RMSE instead of accuracy / macro-F1.
  - No class-weight balancing (not applicable to regression).
  - XGBoost objective: "reg:squarederror" instead of "multi:softmax".
  - LSTM output layer: Linear(32→1) instead of Linear(32→3).
  - Baseline comparison: predicting the training-set mean (naïve regressor).

Evaluation — Leave-One-Subject-Out (LOSO) cross-validation:
  Same protocol as Task 2A; see that file for full rationale.
  Here it also avoids subject-level leakage in mood scale
  (each subject has a unique mood range).

References:
  Chen & Guestrin (2016). XGBoost. KDD.
  Hochreiter & Schmidhuber (1997). LSTM. Neural Computation 9(8).
  Bahdanau et al. (2015). Neural Machine Translation. ICLR.
  Bergstra et al. (2011). Algorithms for Hyper-Parameter Optimization. NeurIPS.
  Saeb et al. (2015). Mobile Phone Sensor Correlates. J. Med. Internet Res.
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
PLOTS_DIR        = Path("plots") / "regression"

MAX_MOOD_MISSING = 50.0

# XGBoost / Optuna
XGB_TRIALS       = 50

# LSTM
SEQ_LEN          = 5
LSTM_HIDDEN      = 64
LSTM_DROPOUT     = 0.3
LSTM_EPOCHS      = 60
LSTM_PATIENCE    = 10
LSTM_BATCH       = 32
LSTM_LR          = 1e-3
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── shared helpers ────────────────────────────────────────────────────────────

def _setup():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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

def _reg_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def _save_scatter(y_true, y_pred, title, fname, color):
    """Scatter plot of predicted vs actual mood values."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.35, s=18, color=color)
    lo = min(y_true.min(), y_pred.min()) - 0.2
    hi = max(y_true.max(), y_pred.max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual mood")
    ax.set_ylabel("Predicted mood")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()


# ── XGBoost Regressor ─────────────────────────────────────────────────────────

def _tune_xgb_reg(Xtr, ytr):
    """Optuna on a 20% temporal hold-out. Returns best_params, best_val_rmse."""
    n_val = max(1, int(len(Xtr) * 0.20))
    Xtr2, Xval = Xtr[:-n_val], Xtr[-n_val:]
    ytr2, yval = ytr[:-n_val], ytr[-n_val:]

    def objective(trial):
        m = xgb.XGBRegressor(
            n_estimators     = trial.suggest_int("n_estimators", 100, 500),
            max_depth        = trial.suggest_int("max_depth", 3, 8),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            objective="reg:squarederror",
            eval_metric="rmse", random_state=42, n_jobs=2, verbosity=0,
        )
        m.fit(Xtr2, ytr2, verbose=False)
        preds = m.predict(Xval)
        return np.sqrt(mean_squared_error(yval, preds))   # minimise RMSE

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=XGB_TRIALS, show_progress_bar=False)
    return study.best_params, study.best_value


def run_xgboost(df_features):
    print("\n── XGBoost Regressor + Optuna (LOSO) ───────────────────────")

    # mood_target is the raw continuous next-day mood score
    df = df_features.dropna(subset=["mood_target"]).copy()
    subjects = sorted(df["subject"].unique())
    print(f"  Subjects: {len(subjects)} | Instances: {len(df)}")

    drop_cols = {"instance_id", "subject", "date_target",
                 "mood_target", "mood_class"}
    feat_cols = [c for c in df.columns
                 if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Per-subject imputation, then zero-fill
    df[feat_cols] = df.groupby("subject")[feat_cols].transform(
        lambda x: x.fillna(x.mean()))
    df[feat_cols] = df[feat_cols].fillna(0)

    # Tune on fold-0 training data (same efficiency trade-off as Task 2A)
    print(f"  Tuning hyperparameters ({XGB_TRIALS} Optuna trials, minimise RMSE)...")
    first_test = subjects[0]
    train_init = df[df["subject"] != first_test]
    Xtr_init   = train_init[feat_cols].values
    ytr_init   = train_init["mood_target"].values
    best_params, best_val_rmse = _tune_xgb_reg(Xtr_init, ytr_init)
    print(f"  Best val RMSE: {best_val_rmse:.4f} | params: {best_params}")

    # LOSO evaluation
    all_true, all_pred = [], []
    feat_importances   = np.zeros(len(feat_cols))

    for test_subj, train_subjs in _loso_folds(subjects):
        train_df = df[df["subject"].isin(train_subjs)]
        test_df  = df[df["subject"] == test_subj]
        if len(test_df) == 0:
            continue

        Xtr = train_df[feat_cols].values
        Xte = test_df[feat_cols].values
        ytr = train_df["mood_target"].values
        yte = test_df["mood_target"].values

        model = xgb.XGBRegressor(
            **best_params,
            objective="reg:squarederror",
            eval_metric="rmse", random_state=42, n_jobs=2, verbosity=0,
        )
        model.fit(Xtr, ytr, verbose=False)
        yp = model.predict(Xte)

        all_true.extend(yte)
        all_pred.extend(yp)
        feat_importances += model.feature_importances_

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    mae, rmse = _reg_metrics(y_true, y_pred)

    # Naïve baseline: predict training mean for every test instance
    baseline_pred = np.full_like(y_true, fill_value=y_true.mean())
    base_mae, base_rmse = _reg_metrics(y_true, baseline_pred)

    print(f"  LOSO MAE : {mae:.4f}  |  LOSO RMSE: {rmse:.4f}")
    print(f"  Naïve baseline MAE: {base_mae:.4f}  |  RMSE: {base_rmse:.4f}")

    # Feature importance plot
    avg_imp = pd.Series(feat_importances / len(subjects), index=feat_cols)
    fig, ax = plt.subplots(figsize=(10, 7))
    avg_imp.sort_values().tail(20).plot(kind="barh", ax=ax,
                                        color="#4C72B0", alpha=0.85)
    ax.set_xlabel("Mean gain (averaged over LOSO folds)")
    ax.set_title("XGBoost Regressor – Top 20 Feature Importances (LOSO)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgb_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    _save_scatter(y_true, y_pred,
                  "XGBoost – Predicted vs Actual Mood (LOSO)",
                  "xgb_scatter.png", "#4C72B0")
    print("  Saved: xgb_feature_importance, xgb_scatter")
    return {"mae": mae, "rmse": rmse, "baseline_mae": base_mae, "baseline_rmse": base_rmse}


# ── LSTM Regressor + Attention ────────────────────────────────────────────────

class _MoodDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # y is continuous: shape (n,) float
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class MoodLSTMRegressor(nn.Module):
    """
    LSTM with additive attention for regression.
    Output: single linear neuron (unbounded continuous prediction).

    Compared to Task 2A's MoodLSTM:
      - fc2 outputs 1 neuron instead of 3 (no softmax).
      - Trained with MSELoss instead of CrossEntropyLoss.
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
        self.fc2    = nn.Linear(32, 1)   # ← 1 output for regression

    def forward(self, x):
        H, _    = self.lstm(x)
        scores  = self.attn_v(torch.tanh(self.attn_w(H)))
        weights = torch.softmax(scores, dim=1)
        context = (weights * H).sum(dim=1)
        return self.fc2(self.drop2(torch.relu(self.fc1(self.drop1(context))))).squeeze(1)


def _build_subject_sequences(df_clean):
    """
    Returns dict: subject -> {X: (n, SEQ_LEN, n_feat), y: (n,) float}
    y is the raw next-day mood (continuous), not a class label.
    """
    df = df_clean.sort_values(["id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = _drop_high_missing(df, id_col="id", mood_col="mood")

    skip  = {"id", "date", "dayofweek", "week"}
    synth = {c for c in df.columns if c.endswith("_synthetic")}
    fc    = [c for c in df.columns
             if c not in skip | synth
             and pd.api.types.is_numeric_dtype(df[c])]

    subj_data = {}
    for subj, g in df.groupby("id", sort=False):
        g    = g.sort_values("date").reset_index(drop=True)
        data = g[fc].values.astype(np.float32)
        data[np.isnan(data)] = 0.0
        mood = g["mood"].values
        n    = len(g)

        Xs, ys = [], []
        for t in range(SEQ_LEN, n - 1):
            tgt = mood[t + 1]
            if np.isnan(tgt):
                continue
            Xs.append(data[t - SEQ_LEN:t])
            ys.append(float(tgt))   # ← raw continuous value

        if Xs:
            subj_data[subj] = {
                "X": np.array(Xs, np.float32),
                "y": np.array(ys, np.float32),
            }

    return subj_data, fc


def _train_lstm(Xtr, ytr, n_feat):
    """Train one LSTM regression fold; return best model."""
    n_val = max(1, int(len(Xtr) * 0.10))
    full  = _MoodDS(Xtr, ytr)
    tr_ds, val_ds = torch.utils.data.random_split(
        full, [len(Xtr) - n_val, n_val],
        generator=torch.Generator().manual_seed(42))
    tr_ld  = DataLoader(tr_ds,  LSTM_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(42))
    val_ld = DataLoader(val_ds, LSTM_BATCH, shuffle=False)

    model  = MoodLSTMRegressor(n_feat).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)
    crit   = nn.MSELoss()   # ← MSE for regression
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
    print("\n── LSTM + Attention Regressor (LOSO, PyTorch) ───────────────")

    subj_data, feat_cols = _build_subject_sequences(df_clean)
    subjects = sorted(subj_data.keys())
    n_feat   = subj_data[subjects[0]]["X"].shape[2]

    print(f"  Subjects: {len(subjects)} | Features: {n_feat} | Device: {DEVICE}")
    total_seqs = sum(len(d["y"]) for d in subj_data.values())
    print(f"  Total sequences: {total_seqs}")

    all_true, all_pred = [], []

    for fold_i, (test_subj, train_subjs) in enumerate(_loso_folds(subjects)):
        Xtr = np.concatenate([subj_data[s]["X"] for s in train_subjs])
        ytr = np.concatenate([subj_data[s]["y"] for s in train_subjs])
        Xte = subj_data[test_subj]["X"]
        yte = subj_data[test_subj]["y"]

        if len(Xte) == 0:
            continue

        nf  = Xtr.shape[2]
        sc  = StandardScaler()
        Xtr = sc.fit_transform(Xtr.reshape(-1, nf)).reshape(-1, SEQ_LEN, nf)
        Xte = sc.transform(Xte.reshape(-1, nf)).reshape(-1, SEQ_LEN, nf)

        model = _train_lstm(Xtr, ytr, n_feat)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(Xte, dtype=torch.float32)
                          .to(DEVICE)).cpu().numpy()

        all_true.extend(yte)
        all_pred.extend(preds)

        fold_mae  = mean_absolute_error(yte, preds)
        fold_rmse = np.sqrt(mean_squared_error(yte, preds))
        print(f"  Fold {fold_i+1:>2}/{len(subjects)} "
              f"(test={test_subj}) | n={len(yte):>3} | "
              f"MAE={fold_mae:.3f}  RMSE={fold_rmse:.3f}")

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    mae, rmse = _reg_metrics(y_true, y_pred)

    baseline_pred = np.full_like(y_true, fill_value=y_true.mean())
    base_mae, base_rmse = _reg_metrics(y_true, baseline_pred)

    print(f"\n  LOSO MAE : {mae:.4f}  |  LOSO RMSE: {rmse:.4f}")
    print(f"  Naïve baseline MAE: {base_mae:.4f}  |  RMSE: {base_rmse:.4f}")

    _save_scatter(y_true, y_pred,
                  "LSTM+Attention – Predicted vs Actual Mood (LOSO)",
                  "lstm_scatter.png", "#DD8452")
    print("  Saved: lstm_scatter")
    return {"mae": mae, "rmse": rmse, "baseline_mae": base_mae, "baseline_rmse": base_rmse}


# ── comparison plot ───────────────────────────────────────────────────────────

def _comparison_plot(results):
    names     = list(results.keys())
    maes      = [r["mae"]  for r in results.values()]
    rmses     = [r["rmse"] for r in results.values()]
    base_mae  = results[names[0]]["baseline_mae"]
    base_rmse = results[names[0]]["baseline_rmse"]
    x, w = np.arange(len(names)), 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, vals, base, metric in zip(
            axes, [maes, rmses], [base_mae, base_rmse], ["MAE", "RMSE"]):
        ax.bar(x, vals, w * 2, color=["#4C72B0", "#DD8452"], alpha=0.85)
        ax.axhline(base, color="gray", linestyle="--", linewidth=1.2,
                   label=f"Naïve mean baseline ({base:.3f})")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel(metric); ax.set_title(f"Model Comparison – {metric} (LOSO)")
        ax.legend()

    plt.suptitle("Task 4: Numerical Mood Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
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
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'Baseline MAE':>14} {'Baseline RMSE':>15}")
    print("-" * 72)
    for name, r in results.items():
        print(f"{name:<20} {r['mae']:>10.4f} {r['rmse']:>10.4f} "
              f"{r['baseline_mae']:>14.4f} {r['baseline_rmse']:>15.4f}")

    _comparison_plot(results)
    print("\n  Saved: model_comparison.png")
    return results


if __name__ == "__main__":
    run()