"""
STEP 3 — Train XGBoost Lead Scoring Model
==========================================
Reads:  data/features.csv
        data/feature_columns.json
Saves:  models/xgb_lead_scorer.model
        models/scaler.pkl
        models/training_report.json
        models/feature_importance.png
        models/shap_importance.png   (if shap installed)
        models/score_distribution.png
        models/cv_fold_results.png
        models/actual_vs_predicted.png
        models/narr_signal_analysis.png
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

BASE      = Path(__file__).parent
DATA_DIR  = BASE / "data"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ── Feature groups (for colour coding & analysis) ─────────────────────────────
 
FEATURE_GROUPS = {
    "narr_payment":    ["narr_has_token_payment", "narr_has_payment",
                        "narr_has_booking", "narr_score_total",
                        "narr_score_max", "narr_score_last"],
    "narr_intent":     ["narr_has_lead_mature", "narr_has_confirmed",
                        "narr_has_property_select", "narr_has_site_visit_done",
                        "narr_has_interested", "narr_positive_fup_count",
                        "narr_fup_with_text_count"],
    "narr_negative":   ["narr_has_negative", "narr_has_not_interested"],
    "meeting_quality": ["fup_meeting_done_count", "fup_visit_done_count",
                        "fup_walk_in_count", "fup_positive_meeting_count",
                        "fup_last_meeting_type_enc", "fup_max_meeting_type_enc",
                        "fup_engagement_score"],
    "followup":        ["fup_count", "fup_call_count", "fup_scheduled_count",
                        "fup_postponed_count", "fup_attempt_call_count",
                        "fup_avg_days_gap", "fup_days_since_last",
                        "fup_total_span_days", "fup_velocity",
                        "fup_has_narration", "fup_pct_qualified"],
    "stage_interest":  ["lead_stage_enc", "interest_level_enc",
                        "fup_max_interest", "fup_last_interest",
                        "fup_interest_trend", "fup_stage_upgraded",
                        "fup_last_stage", "fup_max_stage",
                        "next_meeting_type_enc"],
    "budget":          ["budget_from", "budget_to", "budget_range",
                        "has_budget", "budget_log", "fup_has_budget",
                        "fup_max_budget_log", "fup_avg_budget_log",
                        "fup_budget_trend"],
    "lead_attrs":      ["source_enc", "purpose_enc", "has_plot_category",
                        "has_mode_of_payment", "has_plan_to_buy",
                        "has_meeting_date", "has_next_meeting",
                        "days_since_assign", "days_since_posting",
                        "days_to_next_meeting", "interest_residential",
                        "interest_commercial", "lead_status_active",
                        "lead_narr_score", "fup_max_plot_size"],
}

GROUP_COLOURS = {
    "narr_payment":    "#B91C1C",   # dark red   – payment signals
    "narr_intent":     "#DC2626",   # red        – intent signals
    "narr_negative":   "#6B7280",   # grey       – negative signals
    "meeting_quality": "#1D4ED8",   # dark blue  – meeting quality
    "followup":        "#2563EB",   # blue       – followup activity3
    "stage_interest":  "#16A34A",   # green      – stage / interest
    "budget":          "#D97706",   # amber      – budget
    "lead_attrs":      "#7C3AED",   # purple     – lead attributes
}


def get_feature_group(col: str) -> str:
    for group, cols in FEATURE_GROUPS.items():
        if col in cols:
            return group
    # Auto-detect by prefix if not explicitly listed
    if col.startswith("narr_has_token") or col.startswith("narr_has_pay") or col.startswith("narr_has_book"):
        return "narr_payment"
    if col.startswith("narr_"):
        return "narr_intent"
    if col.startswith("fup_meeting") or col.startswith("fup_visit") or col.startswith("fup_walk"):
        return "meeting_quality"
    if col.startswith("fup_"):
        return "followup"
    if "budget" in col:
        return "budget"
    if "stage" in col or "interest" in col:
        return "stage_interest"
    return "lead_attrs"


# ── Score helpers ─────────────────────────────────────────────────────────────

def score_to_label(score: float) -> str:
    if score >= 80: return "Very Hot"
    if score >= 60: return "Hot"
    if score >= 40: return "Warm"
    if score >= 20: return "Cool"
    return "Cold"


def score_bucket(score: float) -> int:
    """Stratification bucket 0–4."""
    if score >= 80: return 4
    if score >= 60: return 3
    if score >= 40: return 2
    if score >= 20: return 1
    return 0


BUCKET_LABELS = {4: "Very Hot", 3: "Hot", 2: "Warm", 1: "Cool", 0: "Cold"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    csv_path  = DATA_DIR / "features.csv"
    json_path = DATA_DIR / "feature_columns.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}  —  run step2 first.")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}  —  run step2 first.")

    df        = pd.read_csv(csv_path, encoding="utf-8")
    feat_cols = json.loads(json_path.read_text(encoding="utf-8"))

    # Keep only columns that actually exist in the CSV
    missing   = [c for c in feat_cols if c not in df.columns]
    feat_cols = [c for c in feat_cols if c in df.columns]

    if missing:
        print(f"  [WARN] {len(missing)} feature columns not in CSV (will be skipped):")
        for m in missing:
            print(f"          – {m}")

    if "score" not in df.columns:
        raise ValueError("'score' column missing from features.csv — re-run step2.")

    X = df[feat_cols].fillna(-1).astype(float)
    y = df["score"].astype(float)

    return X, y, feat_cols, df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_score_distribution(y: pd.Series, preds: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, title, colour in [
        (axes[0], y,     "Proxy Score Distribution (Actual)",    "#2563EB"),
        (axes[1], preds, "Predicted Score Distribution",          "#16A34A"),
    ]:
        ax.hist(data, bins=30, color=colour, edgecolor="white", alpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.axvline(np.mean(data),   color="red",    linestyle="--",
                   label=f"Mean {np.mean(data):.1f}")
        ax.axvline(np.median(data), color="orange", linestyle="--",
                   label=f"Median {np.median(data):.1f}")

        # Bucket zone shading
        for lo, hi, col, lbl in [
            (80, 100, "#FEE2E2", "Very Hot"),
            (60,  80, "#FEF3C7", "Hot"),
            (40,  60, "#D1FAE5", "Warm"),
        ]:
            ax.axvspan(lo, hi, alpha=0.15, color=col, label=lbl)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "score_distribution.png", dpi=150)
    plt.close()
    print(f"  ✓ Score distribution      → {MODEL_DIR / 'score_distribution.png'}")


def plot_actual_vs_predicted(y: pd.Series, preds: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 6))

    # Colour points by bucket
    colours_map = {4: "#B91C1C", 3: "#DC2626", 2: "#D97706", 1: "#2563EB", 0: "#6B7280"}
    bucket_arr  = np.array([score_bucket(s) for s in y])
    for b, lbl in BUCKET_LABELS.items():
        mask = bucket_arr == b
        if mask.sum():
            ax.scatter(y[mask], preds[mask], alpha=0.5, s=20,
                       color=colours_map[b], label=lbl)

    lo = min(float(y.min()), float(preds.min())) - 2
    hi = max(float(y.max()), float(preds.max())) + 2
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect fit")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual Score",    fontsize=11)
    ax.set_ylabel("Predicted Score", fontsize=11)
    ax.set_title("Actual vs Predicted Lead Score", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")

    mae = mean_absolute_error(y, preds)
    r2  = r2_score(y, preds)
    ax.text(0.05, 0.92, f"MAE={mae:.2f}  R²={r2:.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF3C7"))
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()
    print(f"  ✓ Actual vs predicted     → {MODEL_DIR / 'actual_vs_predicted.png'}")


def plot_feature_importance(model: xgb.XGBRegressor, feat_cols: list):
    importance = pd.Series(model.feature_importances_, index=feat_cols)
    top_n      = min(30, len(importance))
    top        = importance.nlargest(top_n).sort_values()

    colours = [GROUP_COLOURS[get_feature_group(col)] for col in top.index]

    fig, ax = plt.subplots(figsize=(11, max(7, top_n * 0.32)))
    ax.barh(top.index, top.values, color=colours)
    ax.set_title(f"Top {top_n} Feature Importances (XGBoost gain)", fontsize=13)
    ax.set_xlabel("Importance Score")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=GROUP_COLOURS[g], label=g.replace("_", " ").title())
        for g in GROUP_COLOURS
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"  ✓ Feature importance      → {MODEL_DIR / 'feature_importance.png'}")


def plot_cv_fold_results(fold_maes: list, fold_r2s: list, fold_rmses: list):
    folds = list(range(1, len(fold_maes) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, vals, title, ylabel, colour in [
        (axes[0], fold_maes,  "CV MAE per Fold",  "MAE",  "#2563EB"),
        (axes[1], fold_rmses, "CV RMSE per Fold", "RMSE", "#7C3AED"),
        (axes[2], fold_r2s,   "CV R² per Fold",   "R²",   "#16A34A"),
    ]:
        ax.bar(folds, vals, color=colour, edgecolor="white", alpha=0.85)
        ax.axhline(np.mean(vals), color="red", linestyle="--",
                   label=f"Mean {np.mean(vals):.3f}")
        ax.set_title(title)
        ax.set_xlabel("Fold")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "cv_fold_results.png", dpi=150)
    plt.close()
    print(f"  ✓ CV fold results         → {MODEL_DIR / 'cv_fold_results.png'}")


def plot_narr_signal_analysis(df_full: pd.DataFrame, feat_cols: list):
    """
    Show how narration signals correlate with proxy score.
    Plots average score for leads WITH vs WITHOUT each narration flag.
    """
    narr_flag_cols = [
        c for c in feat_cols
        if c.startswith("narr_has_") and c in df_full.columns
    ]
    if not narr_flag_cols or "score" not in df_full.columns:
        return

    with_scores    = []
    without_scores = []
    labels         = []

    for col in narr_flag_cols:
        with_mask    = df_full[col] == 1
        without_mask = df_full[col] == 0
        if with_mask.sum() == 0:
            continue
        with_scores.append(df_full.loc[with_mask,    "score"].mean())
        without_scores.append(df_full.loc[without_mask, "score"].mean())
        labels.append(col.replace("narr_has_", "").replace("_", " ").title())

    if not labels:
        return

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 5))
    ax.bar(x - w / 2, with_scores,    w, label="Signal Present", color="#DC2626", alpha=0.85)
    ax.bar(x + w / 2, without_scores, w, label="Signal Absent",  color="#6B7280", alpha=0.60)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Average Proxy Score")
    ax.set_title("Average Lead Score: Narration Signal Present vs Absent", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(df_full["score"].mean(), color="blue", linestyle="--",
               alpha=0.5, label="Overall mean")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "narr_signal_analysis.png", dpi=150)
    plt.close()
    print(f"  ✓ Narration signal chart  → {MODEL_DIR / 'narr_signal_analysis.png'}")


def plot_bucket_breakdown(y: pd.Series, preds: np.ndarray):
    """Stacked bar: actual bucket vs predicted bucket."""
    bucket_labels_list = ["Cold", "Cool", "Warm", "Hot", "Very Hot"]
    actual_counts  = pd.Series([score_to_label(s) for s in y]).value_counts()
    predict_counts = pd.Series([score_to_label(s) for s in preds]).value_counts()

    colours = ["#6B7280", "#2563EB", "#16A34A", "#D97706", "#DC2626"]
    x       = np.arange(len(bucket_labels_list))
    w       = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    act_vals  = [actual_counts.get(l, 0)  for l in bucket_labels_list]
    pred_vals = [predict_counts.get(l, 0) for l in bucket_labels_list]

    ax.bar(x - w / 2, act_vals,  w, color=colours, alpha=0.90, label="Actual")
    ax.bar(x + w / 2, pred_vals, w, color=colours, alpha=0.55,
           edgecolor="black", linewidth=0.6, label="Predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels_list)
    ax.set_ylabel("Number of Leads")
    ax.set_title("Score Bucket Distribution: Actual vs Predicted", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "bucket_breakdown.png", dpi=150)
    plt.close()
    print(f"  ✓ Bucket breakdown        → {MODEL_DIR / 'bucket_breakdown.png'}")


# ── Training ──────────────────────────────────────────────────────────────────

def get_xgb_params(n_samples: int) -> dict:
    """Scale regularisation with dataset size."""
    if n_samples < 200:
        depth, n_est, lr = 3, 150, 0.03
    elif n_samples < 500:
        depth, n_est, lr = 4, 250, 0.04
    elif n_samples < 1000:
        depth, n_est, lr = 5, 400, 0.05
    else:
        depth, n_est, lr = 6, 600, 0.05

    return {
        "objective":        "reg:squarederror",
        "eval_metric":      "mae",
        "n_estimators":     n_est,
        "max_depth":        depth,
        "learning_rate":    lr,
        "subsample":        0.80,
        "colsample_bytree": 0.75,
        "colsample_bylevel":0.80,
        "min_child_weight": max(2, n_samples // 200),
        "reg_alpha":        0.20,
        "reg_lambda":       1.5,
        "gamma":            0.10,
        "random_state":     42,
        "n_jobs":           -1,
        "tree_method":      "hist",
    }


def run_cross_validation(model_params: dict,
                          X_scaled: pd.DataFrame,
                          y: pd.Series,
                          n_splits: int = 5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_maes, fold_r2s, fold_rmses = [], [], []

    print(f"\n  Running {n_splits}-Fold Cross-Validation …")
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_scaled), 1):
        Xtr, Xva = X_scaled.iloc[tr_idx], X_scaled.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx],        y.iloc[va_idx]

        m = xgb.XGBRegressor(**model_params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        preds = np.clip(m.predict(Xva), 1, 100)
        mae   = mean_absolute_error(yva, preds)
        r2    = r2_score(yva, preds)
        rmse  = np.sqrt(mean_squared_error(yva, preds))

        fold_maes.append(mae)
        fold_r2s.append(r2)
        fold_rmses.append(rmse)

        print(f"    Fold {fold}: MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.3f}")

    return fold_maes, fold_r2s, fold_rmses


def bucket_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_b = np.array([score_bucket(s) for s in y_true])
    pred_b = np.array([score_bucket(s) for s in y_pred])
    overall = float(np.mean(true_b == pred_b))

    per_bucket = {}
    for b, label in BUCKET_LABELS.items():
        mask = true_b == b
        if mask.sum() > 0:
            per_bucket[label] = round(float(np.mean(pred_b[mask] == b)), 3)

    return {"overall": round(overall, 3), **per_bucket}


def narration_feature_stats(df: pd.DataFrame, feat_cols: list) -> dict:
    """
    Summarise narration signal coverage in the dataset.
    Returns dict suitable for the training report.
    """
    narr_cols = [c for c in feat_cols if c.startswith("narr_") and c in df.columns]
    stats     = {}
    for col in narr_cols:
        n_present = int((df[col] != 0).sum())
        pct       = round(100 * n_present / len(df), 1)
        avg_score_with    = round(float(df.loc[df[col] != 0, "score"].mean()), 1) \
                            if n_present > 0 else None
        avg_score_without = round(float(df.loc[df[col] == 0, "score"].mean()), 1) \
                            if (df[col] == 0).sum() > 0 else None
        stats[col] = {
            "n_leads_with_signal": n_present,
            "pct_leads":           pct,
            "avg_score_with":      avg_score_with,
            "avg_score_without":   avg_score_without,
        }
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    print("=" * 60)
    print("STEP 3 — Training XGBoost Lead Scorer")
    print("=" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    X, y, feat_cols, df = load_data()
    n_samples, n_feats  = X.shape

    # Separate narration features for reporting
    narr_feat_cols    = [c for c in feat_cols if c.startswith("narr_")]
    non_narr_feat_cols = [c for c in feat_cols if not c.startswith("narr_")]

    print(f"\n  Samples        : {n_samples}")
    print(f"  Features total : {n_feats}")
    print(f"    – Narration  : {len(narr_feat_cols)}")
    print(f"    – Other      : {len(non_narr_feat_cols)}")
    print(f"  Score  min={y.min():.1f}  mean={y.mean():.1f}  "
          f"median={y.median():.1f}  max={y.max():.1f}")

    # Bucket breakdown
    bucket_series = pd.Series([score_to_label(s) for s in y])
    bucket_counts = bucket_series.value_counts()
    print("\n  Score bucket breakdown:")
    for lbl in ["Very Hot", "Hot", "Warm", "Cool", "Cold"]:
        cnt = bucket_counts.get(lbl, 0)
        pct = 100 * cnt / n_samples
        print(f"    {lbl:10s}: {cnt:5d}  ({pct:.1f}%)")

    # Narration signal coverage
    print("\n  Narration signal coverage:")
    for col in narr_feat_cols:
        if col in df.columns:
            n_present = int((df[col] != 0).sum())
            pct       = 100 * n_present / n_samples
            avg_s     = df.loc[df[col] != 0, "score"].mean() if n_present else 0
            print(f"    {col:<35}: {n_present:4d} leads ({pct:5.1f}%)  "
                  f"avg_score={avg_s:.1f}")

    if n_samples < 30:
        print("\n[WARN] Very few samples — model reliability will be low.")

    # ── Scale ─────────────────────────────────────────────────────────────
    scaler   = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), columns=feat_cols, index=X.index
    )

    # ── Params ────────────────────────────────────────────────────────────
    params = get_xgb_params(n_samples)
    print(f"\n  XGBoost params:")
    for k, v in params.items():
        print(f"    {k:22s}: {v}")

    # ── Cross-validation ──────────────────────────────────────────────────
    n_splits = min(5, max(2, n_samples // 20))
    fold_maes, fold_r2s, fold_rmses = run_cross_validation(
        params, X_scaled, y, n_splits=n_splits
    )

    print(f"\n  CV Summary ({n_splits} folds):")
    print(f"    MAE  : {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f}")
    print(f"    RMSE : {np.mean(fold_rmses):.2f} ± {np.std(fold_rmses):.2f}")
    print(f"    R²   : {np.mean(fold_r2s):.3f} ± {np.std(fold_r2s):.3f}")

    # ── Final fit on all data ─────────────────────────────────────────────
    print("\n  Fitting final model on full dataset …")
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_scaled, y,
        eval_set=[(X_scaled, y)],
        verbose=100,
    )

    preds     = np.clip(model.predict(X_scaled), 1, 100)
    mae_full  = mean_absolute_error(y, preds)
    rmse_full = np.sqrt(mean_squared_error(y, preds))
    r2_full   = r2_score(y, preds)
    b_acc     = bucket_accuracy(y.values, preds)

    print(f"\n  Full-data metrics:")
    print(f"    MAE        : {mae_full:.2f}")
    print(f"    RMSE       : {rmse_full:.2f}")
    print(f"    R²         : {r2_full:.3f}")
    print(f"    Bucket acc : overall={b_acc['overall']:.1%}  "
          + "  ".join(f"{k}={v:.1%}" for k, v in b_acc.items() if k != "overall"))

    # ── Feature importance by group ────────────────────────────────────────
    importance    = pd.Series(model.feature_importances_, index=feat_cols)
    group_imp     = {}
    for group in GROUP_COLOURS:
        group_cols = [c for c in feat_cols if get_feature_group(c) == group]
        group_imp[group] = round(float(importance[group_cols].sum()), 4) \
                           if group_cols else 0.0
    narr_total_imp = sum(
        v for g, v in group_imp.items() if g.startswith("narr_")
    )

    print(f"\n  Feature importance by group:")
    for grp, imp in sorted(group_imp.items(), key=lambda x: -x[1]):
        print(f"    {grp:<20}: {imp:.4f}")
    print(f"    {'[narr total]':<20}: {narr_total_imp:.4f}")

    # ── Save model & scaler ────────────────────────────────────────────────
    model_path  = MODEL_DIR / "xgb_lead_scorer.model"
    scaler_path = MODEL_DIR / "scaler.pkl"

    model.save_model(str(model_path))
    joblib.dump(scaler, str(scaler_path))

    # ── Save report ────────────────────────────────────────────────────────
    narr_stats = narration_feature_stats(df, feat_cols)

    report = {
        "n_samples":   n_samples,
        "n_features":  n_feats,
        "n_narr_features":     len(narr_feat_cols),
        "n_non_narr_features": len(non_narr_feat_cols),
        "feature_cols":        feat_cols,
        "narr_feature_cols":   narr_feat_cols,
        "params": params,
        "score_stats": {
            "min":    round(float(y.min()),    2),
            "max":    round(float(y.max()),    2),
            "mean":   round(float(y.mean()),   2),
            "median": round(float(y.median()), 2),
            "std":    round(float(y.std()),    2),
        },
        "bucket_breakdown": {
            lbl: int(bucket_counts.get(lbl, 0))
            for lbl in ["Very Hot", "Hot", "Warm", "Cool", "Cold"]
        },
        "cv_folds":       n_splits,
        "cv_mae_mean":    round(float(np.mean(fold_maes)),  3),
        "cv_mae_std":     round(float(np.std(fold_maes)),   3),
        "cv_rmse_mean":   round(float(np.mean(fold_rmses)), 3),
        "cv_rmse_std":    round(float(np.std(fold_rmses)),  3),
        "cv_r2_mean":     round(float(np.mean(fold_r2s)),   3),
        "cv_r2_std":      round(float(np.std(fold_r2s)),    3),
        "full_mae":       round(mae_full,  3),
        "full_rmse":      round(rmse_full, 3),
        "full_r2":        round(r2_full,   3),
        "bucket_accuracy": b_acc,
        "feature_importance_by_group": group_imp,
        "narr_total_importance":       round(narr_total_imp, 4),
        "narration_signal_stats":      narr_stats,
    }

    report_path = MODEL_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n  Saved artefacts:")
    print(f"  ✓ Model   → {model_path}")
    print(f"  ✓ Scaler  → {scaler_path}")
    print(f"  ✓ Report  → {report_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n  Generating plots …")
    # Attach predictions and score to df for narration plot
    df_plot        = df.copy()
    df_plot["score"] = y.values

    plot_score_distribution(y, preds)
    plot_actual_vs_predicted(y, preds)
    plot_feature_importance(model, feat_cols)
    plot_cv_fold_results(fold_maes, fold_r2s, fold_rmses)
    plot_narr_signal_analysis(df_plot, feat_cols)
    plot_bucket_breakdown(y, preds)

    # ── SHAP ───────────────────────────────────────────────────────────────
    if HAS_SHAP:
        print("\n  Computing SHAP values …")
        try:
            sample_size = min(500, n_samples)
            X_shap = (X_scaled.sample(sample_size, random_state=42)
                      if n_samples > sample_size else X_scaled)

            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)

            # Beeswarm
            shap.summary_plot(shap_values, X_shap, show=False, max_display=25)
            plt.savefig(MODEL_DIR / "shap_importance.png",
                        bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  ✓ SHAP beeswarm       → {MODEL_DIR / 'shap_importance.png'}")

            # Bar
            shap.summary_plot(shap_values, X_shap, plot_type="bar",
                              show=False, max_display=25)
            plt.savefig(MODEL_DIR / "shap_bar.png",
                        bbox_inches="tight", dpi=150)
            plt.close()
            print(f"  ✓ SHAP bar            → {MODEL_DIR / 'shap_bar.png'}")

            # Narration-only SHAP bar
            narr_shap_cols = [c for c in feat_cols if c.startswith("narr_")]
            if narr_shap_cols:
                narr_idx = [feat_cols.index(c) for c in narr_shap_cols
                            if c in feat_cols]
                if narr_idx:
                    narr_shap = shap_values[:, narr_idx]
                    narr_Xshap = X_shap[narr_shap_cols]
                    shap.summary_plot(narr_shap, narr_Xshap,
                                      plot_type="bar", show=False,
                                      max_display=len(narr_shap_cols))
                    plt.title("SHAP Importance — Narration Features Only")
                    plt.savefig(MODEL_DIR / "shap_narr_bar.png",
                                bbox_inches="tight", dpi=150)
                    plt.close()
                    print(f"  ✓ SHAP narr bar       → {MODEL_DIR / 'shap_narr_bar.png'}")

        except Exception as e:
            print(f"  [WARN] SHAP failed: {e}")
    else:
        print("\n  (pip install shap  for SHAP explainability plots)")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Samples           : {n_samples}")
    print(f"  Features          : {n_feats}  ({len(narr_feat_cols)} narration)")
    print(f"  CV  MAE           : {np.mean(fold_maes):.2f} ± {np.std(fold_maes):.2f}")
    print(f"  CV  R²            : {np.mean(fold_r2s):.3f}")
    print(f"  Bucket accuracy   : {b_acc['overall']:.1%} overall")
    print(f"  Narr importance   : {narr_total_imp:.4f} "
          f"({100*narr_total_imp/max(sum(group_imp.values()),1e-9):.1f}% of total)")
    print("=" * 60)

    # Narration signal summary
    print("\n  Narration signals impact on score:")
    print(f"  {'Signal':<35} {'Leads':>6}  {'Avg score WITH':>14}  {'Avg score WITHOUT':>17}")
    print("  " + "-" * 76)
    for col, st in narr_stats.items():
        if st["n_leads_with_signal"] > 0:
            with_s    = f"{st['avg_score_with']:.1f}"    if st["avg_score_with"]    else "N/A"
            without_s = f"{st['avg_score_without']:.1f}" if st["avg_score_without"] else "N/A"
            print(f"  {col:<35} {st['n_leads_with_signal']:>6}  "
                  f"{with_s:>14}  {without_s:>17}")

    print("\nRun step4_predict.py to score new leads.\n")


if __name__ == "__main__":
    train()