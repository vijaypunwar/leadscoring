"""
STEP 4 — Predict Lead Score
============================
Usage (CLI):
    # Single lead + followups
    python step4_predict.py --lead lead.json --followups followups.json

    # Batch predict from CSV
    python step4_predict.py --batch leads_batch.csv --output predictions.csv

    # Score all leads from data/leads_raw.json + data/followups_by_lead.json
    python step4_predict.py --score-all

    # Interactive mode
    python step4_predict.py --interactive

Or import and call:
    from step4_predict import predict_score, batch_predict
    result = predict_score(lead_dict, followups_list)
"""

import json
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Re-use feature builders from step2
sys.path.insert(0, str(Path(__file__).parent))
from step2_feature_engineering import (
    lead_features,
    followup_features as build_followup_features,
    get_followup_narration_features,
    extract_all_narration_text,
    score_single_narration,
    classify_narration,
    build_feature_row,
    parse_date,
    safe_float,
    enc,
    INTEREST_MAP,
    STAGE_MAP,
    MEETING_TYPE_MAP,
)

BASE      = Path(__file__).parent
MODEL_DIR = BASE / "models"
DATA_DIR  = BASE / "data"

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ── Score tiers ───────────────────────────────────────────────────────────────

SCORE_TIERS = [
    (80, "🔥 Very Hot",  "Immediate priority — highest conversion likelihood. Contact today."),
    (60, "🌟 Hot",       "High intent — follow up urgently within 24 hours."),
    (40, "⭐ Warm",      "Moderate interest — nurture with regular follow-ups."),
    (20, "❄️  Cool",     "Low engagement — re-engage with fresh offer or information."),
    ( 0, "💤 Cold",      "Minimal signals — low priority, revisit later."),
]


def score_label(score: float) -> str:
    for threshold, label, _ in SCORE_TIERS:
        if score >= threshold:
            return label
    return "💤 Cold"


def score_recommendation(score: float) -> str:
    for threshold, _, rec in SCORE_TIERS:
        if score >= threshold:
            return rec
    return SCORE_TIERS[-1][2]


def score_priority(score: float) -> int:
    """1 = highest priority, 5 = lowest."""
    for i, (threshold, _, _) in enumerate(SCORE_TIERS, 1):
        if score >= threshold:
            return i
    return 5


# ── Feature groups for human-readable explanation ─────────────────────────────

FEATURE_FRIENDLY_NAMES = {
    # Narration / NLP features (New Step 2 names)
    "narr_text_count":          "Followups with any notes",
    "narr_positive_count":      "Followups with positive narration",
    "narr_negative_count":      "Followups with negative narration",
    "narr_token":               "Token / advance payment mentioned",
    "narr_payment":             "Payment signal in notes",
    "narr_booking":             "Booking confirmed in notes",
    "narr_conversion":          "Lead marked as mature / converted",
    "narr_confirmation":        "Deal confirmed / agreement signed",
    "narr_property_selected":   "Property / plot selected",
    "narr_site_visit":          "Site visit done (from notes)",
    "narr_strong_interest":     "Strong interest expressed in notes",
    "narr_moderate_interest":   "Moderate interest expressed in notes",
    "narr_positive_quality":    "Positive budget/price quality signal",
    "narr_not_interested":      "Explicitly not interested",
    "narr_no_response":         "No response / no reply signal",
    "narr_contact_issue":       "Phone switched off / unreachable",
    "narr_cancelled":           "Plan or visit cancelled",
    "narr_postponed":           "Visit or meeting postponed",
    "narr_budget_negative":     "Budget or affordability issue",
    "narr_delay":               "Requested contact later / delay",
    "narr_last_attempt":        "Final attempt / last call reached",
    "narr_latest_signal_score": "Most recent followup narration score",
    "narr_best_signal_score":   "Best single followup narration score",
    "narr_total_signal_score":  "Overall narration sentiment score",

    # Meeting / Activity features (New Step 2 names)
    "fup_count":                "Total followup count",
    "fup_positive_meeting_count":"Positive meeting interactions",
    "fup_scheduled_count":      "Scheduled meetings/visits",
    "fup_call_count":           "Calls connected",
    "fup_postponed_count":      "Postponed meetings (negative)",
    "fup_attempt_call_count":   "Unanswered call attempts (negative)",
    "fup_walk_in_count":        "Walk-in visits",
    "fup_visit_done_count":     "Site visits completed",
    "fup_meeting_done_count":   "Meetings completed",
    "fup_avg_days_gap":         "Average days gap between followups",
    "fup_total_span_days":      "Total followup span (days)",
    "fup_days_since_last":      "Days since last followup",
    "fup_max_interest":         "Highest interest level in followups",
    "fup_last_interest":        "Most recent interest level",
    "fup_interest_trend":       "Interest trending up/down",
    "fup_has_budget":           "Budget mentioned in followups",
    "fup_max_budget_log":       "Highest budget mentioned (log)",
    "fup_avg_budget_log":       "Average budget log",
    "fup_budget_trend":         "Budget trending up/down",
    "fup_has_narration":        "Has any followup notes",
    "fup_last_stage":           "Most recent followup stage",
    "fup_max_stage":            "Highest stage reached",
    "fup_stage_upgraded":       "Stage upgraded during followups",
    "fup_max_plot_size":        "Max plot size mentioned",
    "fup_pct_qualified":        "% of followups at qualified+ stage",
    "fup_last_meeting_type_enc":"Last meeting type quality",
    "fup_max_meeting_type_enc": "Best meeting type quality",
    "fup_connected_call_rate":  "Call connection rate",
    "fup_positive_meeting_rate":"Positive meeting rate",
    "fup_postponed_rate":       "Postponement rate",
    "fup_attempt_rate":         "Call attempt rate",
    "fup_quality_ratio":        "Followup interaction quality ratio",

    # Lead attributes
    "interest_level_enc":       "Interest level (High/Med/Low)",
    "lead_stage_enc":           "Current lead stage",
    "next_meeting_type_enc":    "Next scheduled meeting type",
    "source_enc":               "Lead source",
    "purpose_enc":              "Purchase purpose (Own/Investment)",
    "budget_from":              "Budget lower limit",
    "budget_to":                "Budget upper limit",
    "budget_range":             "Budget range",
    "has_budget":               "Budget information available",
    "budget_log":               "Budget (log scale)",
    "has_plot_category":        "Plot category specified",
    "has_mode_of_payment":      "Payment mode specified",
    "has_plan_to_buy":          "Has a plan to buy",
    "has_meeting_date":         "Meeting date set",
    "has_next_meeting":         "Next meeting scheduled",
    "days_since_assign":        "Days since lead assigned",
    "days_since_posting":       "Days since lead posted",
    "days_to_next_meeting":     "Days to next meeting",
    "interest_residential":     "Residential interest flag",
    "interest_commercial":      "Commercial interest flag",
    "lead_status_active_raw":   "Lead status active (raw)",
    "lead_narr_score":          "Lead-level narration score",
    "lead_narr_token":          "Lead-level token mentioned",
    "lead_narr_payment":        "Lead-level payment mentioned",
    "lead_narr_booking":        "Lead-level booking mentioned",
    "lead_narr_conversion":     "Lead-level conversion mentioned",
    "lead_narr_property_selected":"Lead-level property selected",
    "lead_narr_site_visit":     "Lead-level site visit done",
    "lead_narr_strong_interest":"Lead-level strong interest",
    "lead_narr_not_interested": "Lead-level not interested",
    "lead_narr_no_response":    "Lead-level no response",
    "lead_narr_cancelled":      "Lead-level cancelled",
    "lead_narr_budget_negative":"Lead-level budget problem",
    "lead_narr_delay":          "Lead-level delayed contact",
    "lead_is_closed_raw":       "Lead is closed",
    "lead_is_cancelled_raw":    "Lead is cancelled",
    "lead_is_dead_raw":         "Lead is dead",

    # Status features (from classify_lead_status)
    "status_is_closed":         "Lead lifecycle: closed",
    "status_is_cancelled":      "Lead lifecycle: cancelled",
    "status_is_dead":           "Lead lifecycle: dead / lost",
    "status_terminal_positive": "Lead lifecycle: converted / positive terminal",
    "status_terminal_negative": "Lead lifecycle: dead / cancelled terminal",
    "status_closed_lost":       "Lead lifecycle: closed without conversion",
    "status_is_active":         "Lead lifecycle: active",
}


def friendly_name(col: str) -> str:
    return FEATURE_FRIENDLY_NAMES.get(col, col.replace("_", " ").title())


# ── Narration signal extraction for explanation ────────────────────────────────

def get_narration_signals(lead: Dict, followups: List[Dict]) -> List[Dict]:
    """
    Extract matched narration keywords from lead + followup text
    for human-readable explanation.
    """
    signals = []

    # Lead-level narration
    lead_text = extract_all_narration_text(lead)
    if lead_text.strip():
        score = score_single_narration(lead_text)
        classified = classify_narration(lead_text)
        active_labels = [key for key, val in classified.items() if val == 1]
        if active_labels:
            signals.append({
                "source":  "Lead record",
                "text":    lead_text[:150],
                "score":   round(score, 1),
                "signals": active_labels,
            })

    # Followup narrations
    for i, f in enumerate(followups, 1):
        text = extract_all_narration_text(f)
        if text.strip():
            score = score_single_narration(text)
            classified = classify_narration(text)
            active_labels = [key for key, val in classified.items() if val == 1]
            if active_labels:
                signals.append({
                    "source":  f"Followup #{i}",
                    "text":    text[:150],
                    "score":   round(score, 1),
                    "signals": active_labels,
                })

    return signals


# ── Model loader (singleton) ───────────────────────────────────────────────────

class ModelLoader:
    _model     = None
    _scaler    = None
    _feat_cols = None
    _importances = None

    @classmethod
    def get(cls) -> Tuple[xgb.XGBRegressor, object, List[str], Dict[str, float]]:
        if cls._model is None:
            cls._load()
        return cls._model, cls._scaler, cls._feat_cols, cls._importances

    @classmethod
    def _load(cls):
        model_path  = MODEL_DIR / "xgb_lead_scorer.model"
        scaler_path = MODEL_DIR / "scaler.pkl"
        feat_path   = DATA_DIR  / "feature_columns.json"

        for p in [model_path, scaler_path, feat_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing: {p}\n"
                    "Run step3_train_model.py first."
                )

        cls._model = xgb.XGBRegressor()
        cls._model.load_model(str(model_path))
        cls._scaler    = joblib.load(str(scaler_path))
        cls._feat_cols = json.loads(feat_path.read_text(encoding="utf-8"))
        cls._importances = dict(
            zip(cls._feat_cols, cls._model.feature_importances_)
        )
        print(f"  [Model] Loaded  ({len(cls._feat_cols)} features)")


# ── SHAP explanation ──────────────────────────────────────────────────────────

def get_shap_explanation(
    X_scaled: np.ndarray,
    feat_cols: List[str],
    model: xgb.XGBRegressor,
    n_features: int = 8,
) -> List[Dict]:
    if not HAS_SHAP:
        return []
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_scaled)

        contributions = []
        for i, feat in enumerate(feat_cols):
            sv = float(shap_vals[0, i]) if shap_vals.ndim == 2 else float(shap_vals[0])
            contributions.append({
                "feature":          feat,
                "friendly_name":    friendly_name(feat),
                "shap_value":       round(sv, 4),
                "abs_contribution": round(abs(sv), 4),
                "direction":        "▲ positive" if sv > 0 else "▼ negative",
            })

        contributions.sort(key=lambda x: -x["abs_contribution"])
        return contributions[:n_features]

    except Exception as e:
        print(f"[WARN] SHAP failed: {e}")
        return []


def get_importance_explanation(
    row: Dict,
    feat_cols: List[str],
    importances: Dict[str, float],
    n_features: int = 8,
) -> List[Dict]:
    """Fallback feature explanation using XGBoost gain × feature value."""
    contributions = []
    for feat in feat_cols:
        imp   = importances.get(feat, 0.0)
        value = row.get(feat, -1)
        mag   = min(abs(float(value)) / 10.0, 1.0) if float(value) > 0 else 0.0
        weighted = imp * mag
        contributions.append({
            "feature":       feat,
            "friendly_name": friendly_name(feat),
            "importance":    round(imp,      4),
            "value":         round(float(value), 3),
            "contribution":  round(weighted, 4),
            "direction":     "▲ positive" if weighted > 0 else "▼ negative",
        })

    contributions.sort(key=lambda x: -abs(x["contribution"]))
    return contributions[:n_features]


# ── Main predict function ─────────────────────────────────────────────────────

def predict_score(
    lead: Dict,
    followups: Optional[List[Dict]] = None,
) -> Dict:
    """
    Predict lead conversion score.

    Args:
        lead      : Single lead dict from CRM API
        followups : List of followup dicts for this lead

    Returns dict:
        lead_id, score, label, priority, recommendation,
        top_factors, narration_signals, feature_snapshot,
        shap_available, n_followups
    """
    if followups is None:
        followups = []

    model, scaler, feat_cols, importances = ModelLoader.get()

    # ── Build features using consolidated Step 2 structure ──────────────────
    row = build_feature_row(lead, followups)

    # Project to the columns expected by XGBoost Booster
    X        = np.array([[row.get(c, 0.0) for c in feat_cols]], dtype=np.float32)
    X_scaled = scaler.transform(X)

    # ── Predict ───────────────────────────────────────────────────────────
    raw   = float(model.predict(X_scaled)[0])
    score = float(np.clip(round(raw, 1), 1, 100))

    # ── Explanation ───────────────────────────────────────────────────────
    top_factors = get_shap_explanation(X_scaled, feat_cols, model, n_features=8)
    if not top_factors:
        top_factors = get_importance_explanation(row, feat_cols, importances, n_features=8)

    # ── Narration signals ─────────────────────────────────────────────────
    narration_signals = get_narration_signals(lead, followups)

    # ── Feature snapshot (key fields aligned with Step 2) ─────────────────
    feature_snapshot = {
        "interest_level":        str(lead.get("Potential") or lead.get("InterestLevel") or "N/A"),
        "lead_stage":            str(lead.get("stage_name") or lead.get("lead_stages") or "N/A"),
        "followup_count":        int(row.get("fup_count", 0)),
        "days_since_last_fup":   int(row.get("fup_days_since_last", -1)),
        "meetings_done":         int(row.get("fup_meeting_done_count", 0)),
        "site_visits":           int(row.get("fup_visit_done_count", 0)),
        "has_budget":            bool(row.get("has_budget", 0)),
        "budget_to":             float(row.get("budget_to", 0)),
        "narr_score_total":      float(row.get("narr_total_signal_score", 0)),
        "has_token_payment":     bool(row.get("narr_token", 0)),
        "has_lead_mature":       bool(row.get("narr_conversion", 0)),
        "has_booking":           bool(row.get("narr_booking", 0)),
        "engagement_score":      float(row.get("fup_quality_ratio", 0)),
        "pct_qualified":         float(row.get("fup_pct_qualified", 0)),
    }

    return {
        "lead_id":          str(lead.get("LeadNo") or lead.get("Id") or "UNKNOWN"),
        "score":            score,
        "label":            score_label(score),
        "priority":         score_priority(score),
        "recommendation":   score_recommendation(score),
        "top_factors":      top_factors,
        "narration_signals":narration_signals,
        "feature_snapshot": feature_snapshot,
        "shap_available":   HAS_SHAP,
        "n_followups":      len(followups),
    }


# ── Batch predict ─────────────────────────────────────────────────────────────

def batch_predict(
    leads: List[Dict],
    followups_dict: Dict[str, List[Dict]],
    verbose: bool = True,
) -> List[Dict]:
    """
    Predict scores for a list of leads.

    Args:
        leads          : List of lead dicts
        followups_dict : {str(lead_id): [followup_list]}
        verbose        : Print progress

    Returns:
        List of prediction result dicts
    """
    results = []
    errors  = 0

    for i, lead in enumerate(leads, 1):
        lead_id   = str(lead.get("LeadNo") or lead.get("Id") or f"LEAD_{i}")
        followups = followups_dict.get(lead_id, [])

        try:
            result = predict_score(lead, followups)
            results.append(result)
            if verbose and i % 50 == 0:
                print(f"    Processed {i}/{len(leads)} …")
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  [ERROR] Lead {lead_id}: {e}")
            results.append({
                "lead_id":       lead_id,
                "score":         0,
                "label":         "ERROR",
                "priority":      99,
                "recommendation":"Could not score this lead.",
                "error":         str(e),
            })

    if verbose and errors:
        print(f"  [WARN] {errors} leads failed scoring.")

    # Sort by score descending
    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return results


def batch_predict_to_df(results: List[Dict]) -> pd.DataFrame:
    """Flatten batch results into a clean DataFrame."""
    rows = []
    for r in results:
        snap = r.get("feature_snapshot", {})
        rows.append({
            "lead_id":            r.get("lead_id"),
            "score":              r.get("score"),
            "label":              r.get("label"),
            "priority":           r.get("priority"),
            "recommendation":     r.get("recommendation"),
            "n_followups":        r.get("n_followups", 0),
            "interest_level":     snap.get("interest_level"),
            "lead_stage":         snap.get("lead_stage"),
            "followup_count":     snap.get("followup_count"),
            "days_since_last_fup":snap.get("days_since_last_fup"),
            "meetings_done":      snap.get("meetings_done"),
            "site_visits":        snap.get("site_visits"),
            "has_budget":         snap.get("has_budget"),
            "budget_to":          snap.get("budget_to"),
            "narr_score_total":   snap.get("narr_score_total"),
            "has_token_payment":  snap.get("has_token_payment"),
            "has_lead_mature":    snap.get("has_lead_mature"),
            "has_booking":        snap.get("has_booking"),
            "engagement_score":   snap.get("engagement_score"),
        })
    return pd.DataFrame(rows)


# ── Display helpers ───────────────────────────────────────────────────────────

def print_result(result: Dict, verbose: bool = True):
    """Pretty-print a single prediction result."""
    W = 70
    print("\n" + "=" * W)
    print(f"  Lead ID        : {result['lead_id']}")
    print(f"  Score          : {result['score']} / 100")
    print(f"  Label          : {result['label']}")
    print(f"  Priority       : #{result.get('priority', '?')}")
    print(f"  Recommendation : {result['recommendation']}")
    print(f"  Followups      : {result.get('n_followups', 0)}")

    # ── Feature snapshot ──────────────────────────────────────────────────
    snap = result.get("feature_snapshot", {})
    if snap:
        print("\n  Feature Snapshot:")
        print(f"    Interest level     : {snap.get('interest_level')}")
        print(f"    Lead stage         : {snap.get('lead_stage')}")
        print(f"    Followup count     : {snap.get('followup_count')}")
        print(f"    Days since last fup: {snap.get('days_since_last_fup')}")
        print(f"    Meetings done      : {snap.get('meetings_done')}")
        print(f"    Site visits        : {snap.get('site_visits')}")
        print(f"    Has budget         : {snap.get('has_budget')}  "
              f"(to: {snap.get('budget_to', 0):,.0f})")
        print(f"    Narration score    : {snap.get('narr_score_total')}")
        print(f"    Token payment      : {'✓ YES' if snap.get('has_token_payment') else 'No'}")
        print(f"    Lead mature        : {'✓ YES' if snap.get('has_lead_mature') else 'No'}")
        print(f"    Booking confirmed  : {'✓ YES' if snap.get('has_booking') else 'No'}")
        print(f"    Engagement score   : {snap.get('engagement_score')}")

    # ── Narration signals ─────────────────────────────────────────────────
    narr = result.get("narration_signals", [])
    if narr:
        print(f"\n  Narration Signals Detected ({len(narr)} records):")
        for ns in narr:
            print(f"\n    [{ns['source']}]  score={ns['score']}")
            print(f"      Text    : \"{ns['text'][:100]}\"")
            print(f"      Signals : {', '.join(ns['signals'])}")

    # ── Top factors ───────────────────────────────────────────────────────
    if verbose and result.get("top_factors"):
        method = "SHAP" if result.get("shap_available") else "Importance"
        print(f"\n  Top Contributing Factors ({method}):")
        for i, f in enumerate(result["top_factors"], 1):
            direction = f.get("direction", "")
            name      = f.get("friendly_name", f.get("feature", ""))
            if "shap_value" in f:
                print(f"    {i}. {name:<45} {direction}  "
                      f"SHAP={f['shap_value']:>7.4f}")
            else:
                print(f"    {i}. {name:<45} {direction}  "
                      f"imp={f['importance']:.4f}  val={f['value']}")

    print("=" * W + "\n")


def print_batch_summary(results: List[Dict]):
    """Print summary stats for a batch of predictions."""
    scores = [r.get("score", 0) for r in results if r.get("label") != "ERROR"]
    if not scores:
        print("[WARN] No valid predictions.")
        return

    W = 70
    print("\n" + "=" * W)
    print("  BATCH PREDICTION SUMMARY")
    print("=" * W)
    print(f"  Total leads     : {len(results)}")
    print(f"  Successfully scored: {len(scores)}")
    print(f"  Score range     : {min(scores):.1f} – {max(scores):.1f}")
    print(f"  Mean score      : {np.mean(scores):.1f}")
    print(f"  Median score    : {np.median(scores):.1f}")

    # Tier breakdown
    print("\n  Tier breakdown:")
    label_counts = pd.Series([r.get("label") for r in results]).value_counts()
    tier_order   = ["🔥 Very Hot", "🌟 Hot", "⭐ Warm", "❄️  Cool", "💤 Cold", "ERROR"]
    for tier in tier_order:
        cnt = label_counts.get(tier, 0)
        if cnt:
            pct = 100 * cnt / len(results)
            print(f"    {tier:<15}: {cnt:4d}  ({pct:5.1f}%)")

    # Top 10 leads
    top10 = sorted(results, key=lambda r: r.get("score", 0), reverse=True)[:10]
    print("\n  Top 10 Leads by Score:")
    print(f"    {'Lead ID':<15} {'Score':>6}  {'Label':<15}  Recommendation")
    print("    " + "-" * 62)
    for r in top10:
        print(f"    {r['lead_id']:<15} {r.get('score', 0):>6.1f}  "
              f"{r.get('label', ''):<15}  "
              f"{r.get('recommendation', '')[:40]}")

    print("=" * W + "\n")


# ── CLI commands ──────────────────────────────────────────────────────────────

def cmd_single(lead_path: str, followups_path: Optional[str] = None):
    print("=" * 70)
    print("STEP 4 — Single Lead Score Prediction")
    print("=" * 70)

    lead      = json.loads(Path(lead_path).read_text(encoding="utf-8"))
    followups = []

    if followups_path and Path(followups_path).exists():
        raw = json.loads(Path(followups_path).read_text(encoding="utf-8"))
        followups = raw if isinstance(raw, list) else [raw]

    result = predict_score(lead, followups)
    print_result(result, verbose=True)
    return result


def cmd_batch(
    batch_csv: str,
    followups_json: Optional[str] = None,
    output_csv: str = "predictions.csv",
):
    print("=" * 70)
    print("STEP 4 — Batch Lead Score Prediction")
    print("=" * 70)

    leads_df       = pd.read_csv(batch_csv, encoding="utf-8")
    leads          = leads_df.to_dict("records")
    followups_dict = {}

    if followups_json and Path(followups_json).exists():
        followups_dict = json.loads(
            Path(followups_json).read_text(encoding="utf-8")
        )

    print(f"\n  Input leads     : {batch_csv}  ({len(leads)} rows)")
    print(f"  Input followups : {followups_json or 'None'}")

    results = batch_predict(leads, followups_dict, verbose=True)
    df      = batch_predict_to_df(results)

    out = Path(output_csv)
    df.to_csv(out, index=False, encoding="utf-8")

    print_batch_summary(results)
    print(f"  Output saved → {out}")
    return results, df


def cmd_score_all(output_csv: str = "data/all_lead_scores.csv"):
    """
    Score every lead in data/leads_raw.json
    using data/followups_by_lead.json.
    """
    print("=" * 70)
    print("STEP 4 — Score All Leads")
    print("=" * 70)

    leads_path = DATA_DIR / "leads_raw.json"
    fups_path  = DATA_DIR / "followups_by_lead.json"

    if not leads_path.exists():
        print(f"[ERROR] {leads_path} not found. Run step1 first.")
        return

    leads = json.loads(leads_path.read_text(encoding="utf-8"))
    fups  = json.loads(fups_path.read_text(encoding="utf-8")) \
            if fups_path.exists() else {}

    print(f"\n  Leads        : {len(leads)}")
    print(f"  Followup IDs : {len(fups)}")

    # Build followups_dict keyed by str(lead_id)
    followups_dict = {str(k): v for k, v in fups.items()}

    results = batch_predict(leads, followups_dict, verbose=True)
    df      = batch_predict_to_df(results)

    out = Path(output_csv)
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")

    print_batch_summary(results)

    # Also save full results as JSON (includes narration signals)
    json_out = out.with_suffix(".json")
    json_out.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n  CSV saved  → {out}")
    print(f"  JSON saved → {json_out}")
    return results, df


def cmd_interactive():
    print("=" * 70)
    print("STEP 4 — Interactive Lead Scoring")
    print("=" * 70)
    print("Enter 'quit' to exit.\n")

    while True:
        lead_path = input("Lead JSON file path (or 'quit'): ").strip()
        if lead_path.lower() == "quit":
            break
        if not Path(lead_path).exists():
            print(f"[ERROR] Not found: {lead_path}\n")
            continue

        fup_path = input("Followups JSON (press Enter to skip): ").strip()
        try:
            cmd_single(lead_path, fup_path or None)
        except Exception as e:
            print(f"[ERROR] {e}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="XGBoost Lead Score Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python step4_predict.py --lead lead.json --followups followups.json
  python step4_predict.py --batch leads.csv --followups fups.json --output out.csv
  python step4_predict.py --score-all
  python step4_predict.py --interactive
        """
    )

    parser.add_argument("--lead",        help="Lead JSON file (single mode)")
    parser.add_argument("--followups",   help="Followups JSON file")
    parser.add_argument("--batch",       help="Leads CSV file (batch mode)")
    parser.add_argument("--score-all",   action="store_true",
                        help="Score all leads from data/leads_raw.json")
    parser.add_argument("--output",      default="predictions.csv",
                        help="Output CSV path (default: predictions.csv)")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")

    args = parser.parse_args()

    if args.interactive:
        cmd_interactive()
    elif args.score_all:
        cmd_score_all(args.output)
    elif args.batch:
        cmd_batch(args.batch, args.followups, args.output)
    elif args.lead:
        cmd_single(args.lead, args.followups)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()