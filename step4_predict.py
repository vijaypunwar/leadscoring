"""
STEP 4 — Predict Lead Score
============================
Usage (CLI):
    python step4_predict.py --lead lead.json --followups followups.json
    python step4_predict.py --batch leads_batch.csv --output predictions.csv
    python step4_predict.py --score-all
    python step4_predict.py --interactive

Or import:
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

sys.path.insert(0, str(Path(__file__).parent))
from step2_feature_engineering import (
    build_feature_row,
    calculate_lead_score,
    extract_narration_text,
    count_keyword_hits,
    POSITIVE_KEYWORDS,
    NEGATIVE_KEYWORDS,
    get_lead_id,
)

BASE      = Path(__file__).parent
MODEL_DIR = BASE / "models"
DATA_DIR  = BASE / "data"

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


SCORE_TIERS = [
    (80, "Very Hot",  "Immediate priority — highest conversion likelihood. Contact today."),
    (60, "Hot",       "High intent — follow up urgently within 24 hours."),
    (40, "Warm",      "Moderate interest — nurture with regular follow-ups."),
    (20, "Cool",      "Low engagement — re-engage with fresh offer or information."),
    ( 0, "Cold",      "Minimal signals — low priority, revisit later."),
]


def score_label(score: float) -> str:
    for threshold, label, _ in SCORE_TIERS:
        if score >= threshold:
            return label
    return "Cold"


def score_recommendation(score: float) -> str:
    for threshold, _, rec in SCORE_TIERS:
        if score >= threshold:
            return rec
    return SCORE_TIERS[-1][2]


def score_priority(score: float) -> int:
    for i, (threshold, _, _) in enumerate(SCORE_TIERS, 1):
        if score >= threshold:
            return i
    return 5


FEATURE_FRIENDLY_NAMES = {
    "lead_interest_enc":            "Interest level (High/Med/Low)",
    "lead_status_enc":              "Lead status (A/D/C)",
    "lead_stage_enc":               "Current lead stage",
    "next_meeting_type_enc":        "Next meeting type",
    "lead_budget_from":             "Budget lower limit",
    "lead_budget_to":               "Budget upper limit",
    "lead_has_budget":              "Budget information available",
    "lead_is_closed":               "Lead is closed",
    "days_since_posting":           "Days since lead posted",
    "days_since_followup_created":  "Days since followup created",
    "fup_count":                    "Total followup count",
    "fup_positive_hits":            "Positive keyword hits in notes",
    "fup_negative_hits":            "Negative keyword hits in notes",
    "fup_has_audio_long":           "Audio longer than 2 minutes",
    "fup_total_audio_seconds":      "Total audio duration (seconds)",
    "fup_max_budget":               "Highest budget in followups",
    "fup_avg_days_gap":             "Average days between followups",
    "fup_days_since_last":          "Days since last followup",
}


def friendly_name(col: str) -> str:
    return FEATURE_FRIENDLY_NAMES.get(col, col.replace("_", " ").title())


def get_narration_signals(lead: Dict, followups: List[Dict]) -> List[Dict]:
    signals = []

    def pack(source: str, text: str) -> Optional[Dict]:
        text = (text or "").strip()
        if not text:
            return None
        pos = count_keyword_hits(text, POSITIVE_KEYWORDS)
        neg = count_keyword_hits(text, NEGATIVE_KEYWORDS)
        if pos == 0 and neg == 0:
            return None
        tags = []
        if pos:
            tags.append(f"positive_hits={pos}")
        if neg:
            tags.append(f"negative_hits={neg}")
        return {
            "source": source,
            "text": text[:150],
            "score": round(pos - neg, 1),
            "signals": tags,
        }

    s = pack("Lead record", extract_narration_text(lead))
    if s:
        signals.append(s)

    for i, f in enumerate(followups, 1):
        s = pack(f"Followup #{i}", extract_narration_text(f))
        if s:
            signals.append(s)
        audio = str(f.get("audioDescription") or "").strip()
        if audio:
            s2 = pack(f"Followup #{i} audio", audio.lower())
            if s2:
                signals.append(s2)

    return signals


class ModelLoader:
    _model = None
    _scaler = None
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
                    f"Missing: {p}\nRun step3_train_model.py first."
                )

        cls._model = xgb.XGBRegressor()
        cls._model.load_model(str(model_path))
        cls._scaler = joblib.load(str(scaler_path))
        cls._feat_cols = json.loads(feat_path.read_text(encoding="utf-8"))
        cls._importances = dict(zip(cls._feat_cols, cls._model.feature_importances_))
        print(f"  [Model] Loaded  ({len(cls._feat_cols)} features)")


def get_shap_explanation(X_scaled, feat_cols, model, n_features: int = 8) -> List[Dict]:
    if not HAS_SHAP:
        return []
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_scaled)
        contributions = []
        for i, feat in enumerate(feat_cols):
            sv = float(shap_vals[0, i]) if shap_vals.ndim == 2 else float(shap_vals[0])
            contributions.append({
                "feature": feat,
                "friendly_name": friendly_name(feat),
                "shap_value": round(sv, 4),
                "abs_contribution": round(abs(sv), 4),
                "direction": "positive" if sv > 0 else "negative",
            })
        contributions.sort(key=lambda x: -x["abs_contribution"])
        return contributions[:n_features]
    except Exception as e:
        print(f"[WARN] SHAP failed: {e}")
        return []


def get_importance_explanation(row, feat_cols, importances, n_features: int = 8) -> List[Dict]:
    contributions = []
    for feat in feat_cols:
        imp = importances.get(feat, 0.0)
        value = row.get(feat, -1)
        mag = min(abs(float(value)) / 10.0, 1.0) if float(value) > 0 else 0.0
        weighted = imp * mag
        contributions.append({
            "feature": feat,
            "friendly_name": friendly_name(feat),
            "importance": round(imp, 4),
            "value": round(float(value), 3),
            "contribution": round(weighted, 4),
            "direction": "positive" if weighted > 0 else "negative",
        })
    contributions.sort(key=lambda x: -abs(x["contribution"]))
    return contributions[:n_features]


def predict_score(lead: Dict, followups: Optional[List[Dict]] = None) -> Dict:
    if followups is None:
        followups = []

    model, scaler, feat_cols, importances = ModelLoader.get()
    row = build_feature_row(lead, followups)

    X = np.array([[row.get(c, 0.0) for c in feat_cols]], dtype=np.float32)
    X_scaled = scaler.transform(X)

    raw = float(model.predict(X_scaled)[0])
    score = float(np.clip(round(raw, 1), 1, 95))
    rule_score = float(row.get("score", calculate_lead_score(lead, followups)))

    top_factors = get_shap_explanation(X_scaled, feat_cols, model, n_features=8)
    if not top_factors:
        top_factors = get_importance_explanation(row, feat_cols, importances, n_features=8)

    narration_signals = get_narration_signals(lead, followups)

    feature_snapshot = {
        "interest_level": str(lead.get("InterestLevel") or lead.get("Potential") or "N/A"),
        "lead_status": str(lead.get("LeadStatus") or "N/A"),
        "lead_stage": str(lead.get("lead_stages") or lead.get("stage_name") or "N/A"),
        "followup_count": int(row.get("fup_count", 0)),
        "days_since_last_fup": int(row.get("fup_days_since_last", -1)),
        "positive_hits": int(row.get("fup_positive_hits", 0)),
        "negative_hits": int(row.get("fup_negative_hits", 0)),
        "has_long_audio": bool(row.get("fup_has_audio_long", 0)),
        "audio_seconds": float(row.get("fup_total_audio_seconds", 0)),
        "has_budget": bool(row.get("lead_has_budget", 0) or row.get("fup_max_budget", 0)),
        "budget_to": float(row.get("lead_budget_to", 0) or row.get("fup_max_budget", 0)),
        "rule_score": rule_score,
    }

    return {
        "lead_id": get_lead_id(lead),
        "score": score,
        "rule_score": rule_score,
        "label": score_label(score),
        "priority": score_priority(score),
        "recommendation": score_recommendation(score),
        "top_factors": top_factors,
        "narration_signals": narration_signals,
        "feature_snapshot": feature_snapshot,
        "shap_available": HAS_SHAP,
        "n_followups": len(followups),
    }


def batch_predict(leads: List[Dict], followups_dict: Dict[str, List[Dict]], verbose: bool = True) -> List[Dict]:
    results, errors = [], 0
    for i, lead in enumerate(leads, 1):
        lead_id = get_lead_id(lead)
        followups = followups_dict.get(lead_id, []) or followups_dict.get(str(lead_id), [])
        try:
            results.append(predict_score(lead, followups))
            if verbose and i % 50 == 0:
                print(f"    Processed {i}/{len(leads)} …")
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  [ERROR] Lead {lead_id}: {e}")
            results.append({
                "lead_id": lead_id,
                "score": 0,
                "label": "ERROR",
                "priority": 99,
                "recommendation": "Could not score this lead.",
                "error": str(e),
            })
    if verbose and errors:
        print(f"  [WARN] {errors} leads failed scoring.")
    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return results


def batch_predict_to_df(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        snap = r.get("feature_snapshot", {})
        rows.append({
            "lead_id": r.get("lead_id"),
            "score": r.get("score"),
            "rule_score": r.get("rule_score"),
            "label": r.get("label"),
            "priority": r.get("priority"),
            "recommendation": r.get("recommendation"),
            "n_followups": r.get("n_followups", 0),
            "interest_level": snap.get("interest_level"),
            "lead_status": snap.get("lead_status"),
            "lead_stage": snap.get("lead_stage"),
            "followup_count": snap.get("followup_count"),
            "days_since_last_fup": snap.get("days_since_last_fup"),
            "positive_hits": snap.get("positive_hits"),
            "negative_hits": snap.get("negative_hits"),
            "has_long_audio": snap.get("has_long_audio"),
            "has_budget": snap.get("has_budget"),
            "budget_to": snap.get("budget_to"),
        })
    return pd.DataFrame(rows)


def print_result(result: Dict, verbose: bool = True):
    W = 70
    print("\n" + "=" * W)
    print(f"  Lead ID        : {result['lead_id']}")
    print(f"  Model score    : {result['score']} / 95")
    print(f"  Rule score     : {result.get('rule_score', 'N/A')}")
    print(f"  Label          : {result['label']}")
    print(f"  Priority       : #{result.get('priority', '?')}")
    print(f"  Recommendation : {result['recommendation']}")
    print(f"  Followups      : {result.get('n_followups', 0)}")

    snap = result.get("feature_snapshot", {})
    if snap:
        print("\n  Feature Snapshot:")
        print(f"    Interest level     : {snap.get('interest_level')}")
        print(f"    Lead status        : {snap.get('lead_status')}")
        print(f"    Lead stage         : {snap.get('lead_stage')}")
        print(f"    Followup count     : {snap.get('followup_count')}")
        print(f"    Days since last fup: {snap.get('days_since_last_fup')}")
        print(f"    Positive hits      : {snap.get('positive_hits')}")
        print(f"    Negative hits      : {snap.get('negative_hits')}")
        print(f"    Long audio (>2m)   : {'YES' if snap.get('has_long_audio') else 'No'}")
        print(f"    Has budget         : {snap.get('has_budget')}  (to: {snap.get('budget_to', 0):,.0f})")

    narr = result.get("narration_signals", [])
    if narr:
        print(f"\n  Narration Signals ({len(narr)} records):")
        for ns in narr:
            print(f"\n    [{ns['source']}]  net={ns['score']}")
            print(f"      Text    : \"{ns['text'][:100]}\"")
            print(f"      Signals : {', '.join(ns['signals'])}")

    if verbose and result.get("top_factors"):
        method = "SHAP" if result.get("shap_available") else "Importance"
        print(f"\n  Top Contributing Factors ({method}):")
        for i, f in enumerate(result["top_factors"], 1):
            name = f.get("friendly_name", f.get("feature", ""))
            direction = f.get("direction", "")
            if "shap_value" in f:
                print(f"    {i}. {name:<45} {direction}  SHAP={f['shap_value']:>7.4f}")
            else:
                print(f"    {i}. {name:<45} {direction}  imp={f['importance']:.4f}  val={f['value']}")
    print("=" * W + "\n")


def print_batch_summary(results: List[Dict]):
    scores = [r.get("score", 0) for r in results if r.get("label") != "ERROR"]
    if not scores:
        print("[WARN] No valid predictions.")
        return
    W = 70
    print("\n" + "=" * W)
    print("  BATCH PREDICTION SUMMARY")
    print("=" * W)
    print(f"  Total leads          : {len(results)}")
    print(f"  Successfully scored  : {len(scores)}")
    print(f"  Score range          : {min(scores):.1f} – {max(scores):.1f}")
    print(f"  Mean score           : {np.mean(scores):.1f}")
    print(f"  Median score         : {np.median(scores):.1f}")

    print("\n  Tier breakdown:")
    label_counts = pd.Series([r.get("label") for r in results]).value_counts()
    for tier in ["Very Hot", "Hot", "Warm", "Cool", "Cold", "ERROR"]:
        cnt = int(label_counts.get(tier, 0))
        if cnt:
            print(f"    {tier:<12}: {cnt:4d}  ({100 * cnt / len(results):5.1f}%)")

    top10 = sorted(results, key=lambda r: r.get("score", 0), reverse=True)[:10]
    print("\n  Top 10 Leads by Score:")
    print(f"    {'Lead ID':<15} {'Score':>6}  {'Label':<12}  Recommendation")
    print("    " + "-" * 62)
    for r in top10:
        print(f"    {str(r['lead_id']):<15} {r.get('score', 0):>6.1f}  "
              f"{str(r.get('label', '')):<12}  {str(r.get('recommendation', ''))[:40]}")
    print("=" * W + "\n")


def cmd_single(lead_path: str, followups_path: Optional[str] = None):
    print("=" * 70)
    print("STEP 4 — Single Lead Score Prediction")
    print("=" * 70)
    lead = json.loads(Path(lead_path).read_text(encoding="utf-8"))
    followups = []
    if followups_path and Path(followups_path).exists():
        raw = json.loads(Path(followups_path).read_text(encoding="utf-8"))
        followups = raw if isinstance(raw, list) else [raw]
    result = predict_score(lead, followups)
    print_result(result, verbose=True)
    return result


def cmd_batch(batch_csv: str, followups_json: Optional[str] = None, output_csv: str = "predictions.csv"):
    print("=" * 70)
    print("STEP 4 — Batch Lead Score Prediction")
    print("=" * 70)
    leads = pd.read_csv(batch_csv, encoding="utf-8").to_dict("records")
    followups_dict = {}
    if followups_json and Path(followups_json).exists():
        followups_dict = json.loads(Path(followups_json).read_text(encoding="utf-8"))
    print(f"\n  Input leads     : {batch_csv}  ({len(leads)} rows)")
    print(f"  Input followups : {followups_json or 'None'}")
    results = batch_predict(leads, followups_dict, verbose=True)
    df = batch_predict_to_df(results)
    out = Path(output_csv)
    df.to_csv(out, index=False, encoding="utf-8")
    print_batch_summary(results)
    print(f"  Output saved → {out}")
    return results, df


def cmd_score_all(output_csv: str = "data/all_lead_scores.csv"):
    print("=" * 70)
    print("STEP 4 — Score All Leads")
    print("=" * 70)
    leads_path = DATA_DIR / "leads_raw.json"
    fups_path  = DATA_DIR / "followups_by_lead.json"
    if not leads_path.exists():
        print(f"[ERROR] {leads_path} not found. Run step1 first.")
        return
    leads = json.loads(leads_path.read_text(encoding="utf-8"))
    fups = json.loads(fups_path.read_text(encoding="utf-8")) if fups_path.exists() else {}
    print(f"\n  Leads        : {len(leads)}")
    print(f"  Followup IDs : {len(fups)}")
    followups_dict = {str(k): v for k, v in fups.items()}
    results = batch_predict(leads, followups_dict, verbose=True)
    df = batch_predict_to_df(results)
    out = Path(output_csv)
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print_batch_summary(results)
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
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


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost Lead Score Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lead", help="Lead JSON file (single mode)")
    parser.add_argument("--followups", help="Followups JSON file")
    parser.add_argument("--batch", help="Leads CSV file (batch mode)")
    parser.add_argument("--score-all", action="store_true",
                        help="Score all leads from data/leads_raw.json")
    parser.add_argument("--output", default="predictions.csv",
                        help="Output CSV path (default: predictions.csv)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
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