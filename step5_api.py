"""
STEP 5 — REST API (Flask) with Swagger UI
==========================================
  GET  /                    → API info & available routes
  GET  /health              → liveness probe
  POST /score               → score a lead (first time)
  PUT  /score/<lead_id>     → re-score / update existing lead
  GET  /score/<lead_id>     → fetch cached score
  GET  /scores              → list all cached scores
  GET  /swagger.json        → OpenAPI specification
  GET  /docs                → Swagger UI

Run:
  python step5_api.py
  http://localhost:8000/docs
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify
from step4_predict import predict_score
from step2_feature_engineering import (
    get_lead_id,
    score_lead_status,
    score_interest_level,
    score_meeting_details,
    score_next_meeting_details,
    score_next_meeting_status,
    score_budget,
    score_lead_stage,
    score_call_status,
    score_next_task_type,
    score_audio_duration,
    score_audio_description,
    score_response_speed,
    extract_narration_text,
    count_keyword_hits,
    POSITIVE_KEYWORDS,
    NEGATIVE_KEYWORDS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)
score_store: dict = {}

PROJECT_ROOT = Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODEL_FILE   = MODELS_DIR / "xgb_lead_scorer.model"
SCALER_FILE  = MODELS_DIR / "scaler.pkl"


def _check_artifacts():
    missing = []
    if not MODEL_FILE.exists():
        missing.append(str(MODEL_FILE))
    if not SCALER_FILE.exists():
        missing.append(str(SCALER_FILE))
    if missing:
        log.error("MISSING MODEL ARTIFACTS — run: python step3_train_model.py")
        raise FileNotFoundError(f"Missing model files: {missing}")


_check_artifacts()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_lead_id(lead: dict):
    for key in ("LeadNo", "lead_id", "leadId", "id", "ID"):
        val = lead.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    lid = get_lead_id(lead)
    try:
        return int(lid)
    except (ValueError, TypeError):
        return lid


def _validate_body(body) -> tuple:
    if not body:
        return None, None, {"error": "JSON body required"}
    lead = body.get("lead")
    if not lead:
        return None, None, {"error": "'lead' key missing from request body"}
    if not isinstance(lead, dict):
        return None, None, {"error": "'lead' must be a JSON object"}
    followups = body.get("followups", [])
    if not isinstance(followups, list):
        return None, None, {"error": "'followups' must be a JSON array"}
    return lead, followups, None


def _to_python(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _to_python(obj.tolist())
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


# Original bands (unchanged)
def _score_category(score: float) -> str:
    if score <= 20:
        return "Cold"
    if score <= 40:
        return "Warm"
    if score <= 60:
        return "Interested"
    if score <= 80:
        return "Hot"
    return "Very Hot"


def _recommendation_from_score(score: float) -> str:
    if score > 80:
        return "Contact immediately — very hot lead"
    if score > 60:
        return "Follow up within 24 hours — hot lead"
    if score > 40:
        return "Continue regular nurturing — interested lead"
    if score > 20:
        return "Re-engage with a fresh offer or information — warm lead"
    return "Low priority — cold lead, revisit later"


def _parameter_scores(lead: dict, followups: list) -> list:
    """Informational per-field rule scores. Does not replace lead.score."""
    applied = {}
    ls_score, _ = score_lead_status(lead)
    interest, applied = score_interest_level(lead, followups, applied)
    meeting, applied = score_meeting_details(followups, applied)
    next_det, applied = score_next_meeting_details(followups, applied)
    next_st, applied = score_next_meeting_status(lead, followups, applied)
    budget, applied = score_budget(lead, followups, applied)
    stage, applied = score_lead_stage(lead, followups, applied)
    call_st, applied = score_call_status(followups, applied)
    task, applied = score_next_task_type(followups, applied)
    audio_dur, applied = score_audio_duration(followups, applied)
    audio_desc, applied = score_audio_description(followups, applied)
    speed, applied = score_response_speed(lead, applied)

    all_text = extract_narration_text(lead) + " " + " ".join(
        extract_narration_text(f) for f in followups
    )
    neg_hits = count_keyword_hits(all_text, NEGATIVE_KEYWORDS)
    neg_penalty = min(neg_hits * 2.0, 15.0)

    return [
        {"parameter": "LeadStatus", "value": lead.get("LeadStatus"), "score": round(float(ls_score), 2)},
        {"parameter": "InterestLevel", "value": lead.get("InterestLevel") or lead.get("Potential"), "score": round(float(interest), 2)},
        {"parameter": "MeetingDetail", "value": next((extract_narration_text(f) for f in followups if extract_narration_text(f)), extract_narration_text(lead) or None), "score": round(float(meeting), 2)},
        {"parameter": "NextMeetingDetails", "value": next((f.get("NextMeetingDetails") for f in followups if f.get("NextMeetingDetails")), lead.get("NextMeetingDetails")), "score": round(float(next_det), 2)},
        {"parameter": "NextMeetingType", "value": lead.get("NextMeetingType"), "score": round(float(next_st), 2)},
        {"parameter": "budget_from_to", "value": {"budget_from": lead.get("budget_from"), "budget_to": lead.get("budget_to")}, "score": round(float(budget), 2)},
        {"parameter": "lead_stages", "value": lead.get("lead_stages") or lead.get("stage_name"), "score": round(float(stage), 2)},
        {"parameter": "f_call_status", "value": next((f.get("f_call_status") for f in followups if f.get("f_call_status")), lead.get("f_call_status")), "score": round(float(call_st), 2)},
        {"parameter": "NextTaskType", "value": next((f.get("NextTaskType") for f in followups if f.get("NextTaskType")), lead.get("NextTaskType")), "score": round(float(task), 2)},
        {"parameter": "AudioDuration", "value": next((f.get("AudioDuration") for f in followups if f.get("AudioDuration") not in (None, "")), None), "score": round(float(audio_dur), 2)},
        {"parameter": "audioDescription", "value": next((f.get("audioDescription") for f in followups if f.get("audioDescription")), None), "score": round(float(audio_desc), 2)},
        {"parameter": "FollowupCreatedAt_response_speed", "value": lead.get("FollowupCreatedAt") or lead.get("PostingDate"), "score": round(float(speed), 2)},
        {"parameter": "negative_keyword_penalty", "value": {"negative_hits": neg_hits}, "score": round(float(-neg_penalty), 2)},
    ]


def _extract_status_flags(result: dict) -> dict:
    keys = [
        "status_is_closed", "status_is_cancelled", "status_is_dead",
        "status_terminal_positive", "status_terminal_negative",
        "status_closed_lost", "status_is_active",
    ]
    return {k: int(result.get(k, 0)) for k in keys}


def _extract_score_breakdown(result: dict) -> dict:
    keys = ["lead_quality", "engagement", "buying_intent", "negative_risk", "recency"]
    breakdown = {k: result.get(k) for k in keys if k in result}
    return breakdown if breakdown else None


def _build_response(lead_id, score: float, followups: list, scored_at: str,
                    result: dict = None, lead: dict = None) -> dict:
    score = float(score)
    response = {
        "lead_id": lead_id,
        "score": round(score, 1),
        "score_category": _score_category(score),
        "total_followups": len(followups) if followups else 0,
        "recommendations": _recommendation_from_score(score),
        "datetime": scored_at,
    }
    if result:
        status_flags = _extract_status_flags(result)
        if any(status_flags.values()):
            response["status"] = status_flags
        breakdown = _extract_score_breakdown(result)
        if breakdown:
            response["score_breakdown"] = breakdown
    if lead is not None:
        response["parameter_scores"] = _parameter_scores(lead, followups or [])
    return response


SWAGGER_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Lead Scoring API",
        "description": (
            "XGBoost lead score (main field: score) plus optional informational "
            "parameter_scores for each scoring rule."
        ),
        "version": "1.1.0",
    },
    "servers": [{"url": "http://localhost:8000", "description": "Local development server"}],
    "tags": [
        {"name": "Info", "description": "Service metadata & health"},
        {"name": "Score", "description": "Score and retrieve lead scores"},
        {"name": "Docs", "description": "API documentation"},
    ],
    "paths": {
        "/": {"get": {"tags": ["Info"], "summary": "API information", "responses": {"200": {"description": "OK"}}}},
        "/health": {"get": {"tags": ["Info"], "summary": "Health check", "responses": {"200": {"description": "OK"}}}},
        "/score": {
            "post": {
                "tags": ["Score"],
                "summary": "Score a new lead",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ScoreRequest"}}},
                },
                "responses": {
                    "201": {"description": "Lead scored successfully",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ScoreResponse"}}}},
                    "400": {"description": "Validation error"},
                    "500": {"description": "Prediction failure"},
                },
            }
        },
        "/score/{lead_id}": {
            "put": {
                "tags": ["Score"],
                "summary": "Re-score / update an existing lead",
                "parameters": [{"name": "lead_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ScoreRequest"}}},
                },
                "responses": {"200": {"description": "Lead re-scored successfully"}},
            },
            "get": {
                "tags": ["Score"],
                "summary": "Get cached score",
                "parameters": [{"name": "lead_id", "in": "path", "required": True, "schema": {"type": "integer"}}],
                "responses": {"200": {"description": "Cached score found"}, "404": {"description": "Lead not yet scored"}},
            },
        },
        "/scores": {
            "get": {
                "tags": ["Score"],
                "summary": "List all cached scores",
                "parameters": [{
                    "name": "category", "in": "query", "required": False,
                    "schema": {"type": "string", "enum": ["Cold", "Warm", "Interested", "Hot", "Very Hot"]},
                }],
                "responses": {"200": {"description": "Summary of all scored leads"}},
            }
        },
        "/swagger.json": {"get": {"tags": ["Docs"], "summary": "OpenAPI specification (JSON)", "responses": {"200": {"description": "OK"}}}},
        "/docs": {"get": {"tags": ["Docs"], "summary": "Swagger UI", "responses": {"200": {"description": "HTML"}}}},
    },
    "components": {
        "schemas": {
            "ScoreRequest": {
                "type": "object",
                "required": ["lead"],
                "properties": {
                    "lead": {"type": "object"},
                    "followups": {"type": "array", "items": {"type": "object"}},
                },
            },
            "ParameterScore": {
                "type": "object",
                "properties": {
                    "parameter": {"type": "string"},
                    "value": {},
                    "score": {"type": "number", "description": "Informational rule contribution only"},
                },
            },
            "ScoreResponse": {
                "type": "object",
                "properties": {
                    "lead_id": {"type": "integer", "example": 3678},
                    "score": {"type": "number", "example": 72.4, "description": "Main XGBoost lead score"},
                    "score_category": {"type": "string", "enum": ["Cold", "Warm", "Interested", "Hot", "Very Hot"]},
                    "total_followups": {"type": "integer", "example": 3},
                    "recommendations": {"type": "string"},
                    "datetime": {"type": "string", "format": "date-time"},
                    "parameter_scores": {
                        "type": "array",
                        "description": "Informational per-parameter scores; not the lead total",
                        "items": {"$ref": "#/components/schemas/ParameterScore"},
                    },
                },
            },
            "Error": {"type": "object", "properties": {"error": {"type": "string"}, "hint": {"type": "string"}}},
        }
    },
}

SWAGGER_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Lead Scoring API - Swagger UI</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css" />
    <style>body { margin: 0; background: #fafafa; }</style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: "/swagger.json",
                dom_id: "#swagger-ui",
                presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
            });
        };
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "api": "Lead Scoring API",
        "version": "1.1.0",
        "status": "running",
        "timestamp": _utc_now(),
        "leads_scored": len(score_store),
        "docs_url": "http://localhost:8000/docs",
        "score_categories": ["Cold", "Warm", "Interested", "Hot", "Very Hot"],
        "endpoints": [
            {"method": "GET",  "path": "/",             "description": "API info and available routes"},
            {"method": "GET",  "path": "/health",       "description": "Liveness probe"},
            {"method": "POST", "path": "/score",        "description": "Score a lead for the first time"},
            {"method": "PUT",  "path": "/score/<id>",   "description": "Re-score an existing lead"},
            {"method": "GET",  "path": "/score/<id>",   "description": "Fetch cached score"},
            {"method": "GET",  "path": "/scores",       "description": "List all cached scores (optional ?category=)"},
            {"method": "GET",  "path": "/docs",         "description": "Swagger UI"},
            {"method": "GET",  "path": "/swagger.json", "description": "OpenAPI specification"},
        ],
        "example_curl": {
            "score_new_lead":    'curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d @sample_input.json',
            "update_lead_score": 'curl -X PUT  http://localhost:8000/score/57249 -H "Content-Type: application/json" -d @sample_input.json',
            "get_cached_score":  "curl http://localhost:8000/score/57249",
            "list_all_scores":   "curl http://localhost:8000/scores",
            "list_hot_leads":    "curl 'http://localhost:8000/scores?category=Hot'",
        },
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_ready": True,
        "leads_scored": len(score_store),
        "timestamp": _utc_now(),
    }), 200


@app.route("/swagger.json", methods=["GET"])
def swagger_spec():
    return jsonify(SWAGGER_SPEC), 200


@app.route("/docs", methods=["GET"])
def swagger_ui():
    return SWAGGER_UI_HTML, 200


@app.route("/score", methods=["POST"])
def score_lead():
    body = request.get_json(force=True, silent=True)
    lead, followups, err = _validate_body(body)
    if err:
        log.warning("POST /score validation failed: %s", err)
        return jsonify(err), 400
    try:
        result = predict_score(lead, followups)
    except Exception as exc:
        log.exception("predict_score failed")
        return jsonify({"error": str(exc)}), 500

    result = _to_python(result)
    lead_id = result.get("lead_id")
    if lead_id is None:
        lead_id = _extract_lead_id(lead)
    if lead_id is None:
        return jsonify({"error": "Unable to extract lead_id from lead object"}), 400
    try:
        lead_id = int(lead_id)
    except (ValueError, TypeError):
        pass

    scored_at = _utc_now()
    score_val = float(result.get("score", 0))
    response = _build_response(lead_id, score_val, followups, scored_at, result, lead)
    score_store[lead_id] = response.copy()
    log.info("Scored lead %s → %.1f [%s]", lead_id, score_val, response["score_category"])
    return jsonify(response), 201


@app.route("/score/<int:lead_id>", methods=["PUT"])
def update_score(lead_id: int):
    body = request.get_json(force=True, silent=True)
    lead, followups, err = _validate_body(body)
    if err:
        return jsonify(err), 400

    previous = score_store.get(lead_id)
    previous_score = previous["score"] if previous else None
    lead["id"] = lead_id
    lead["LeadNo"] = lead_id

    try:
        result = predict_score(lead, followups)
    except Exception as exc:
        log.exception("predict_score failed for lead %s", lead_id)
        return jsonify({"error": str(exc)}), 500

    result = _to_python(result)
    score_val = float(result.get("score", 0))
    response = _build_response(lead_id, score_val, followups, _utc_now(), result, lead)
    score_store[lead_id] = response.copy()
    if previous_score is not None:
        log.info("Updated lead %s → %.1f Δ%+.1f", lead_id, score_val, score_val - previous_score)
    else:
        log.info("Updated lead %s → %.1f [%s]", lead_id, score_val, response["score_category"])
    return jsonify(response), 200


@app.route("/score/<int:lead_id>", methods=["GET"])
def get_score(lead_id: int):
    cached = score_store.get(lead_id)
    if not cached:
        return jsonify({
            "error": f"Lead {lead_id} has not been scored yet",
            "hint": f"POST /score first, then GET /score/{lead_id}",
        }), 404
    return jsonify(cached), 200


@app.route("/scores", methods=["GET"])
def list_scores():
    category_filter = request.args.get("category")
    valid_categories = {"Cold", "Warm", "Interested", "Hot", "Very Hot"}
    summary = list(score_store.values())
    if category_filter:
        if category_filter not in valid_categories:
            return jsonify({
                "error": f"Invalid category '{category_filter}'",
                "hint": f"Use one of: {sorted(valid_categories)}",
            }), 400
        summary = [s for s in summary if s.get("score_category") == category_filter]
    summary.sort(key=lambda x: x["score"], reverse=True)
    by_category = {cat: 0 for cat in ["Cold", "Warm", "Interested", "Hot", "Very Hot"]}
    for s in score_store.values():
        cat = s.get("score_category")
        if cat in by_category:
            by_category[cat] += 1
    return jsonify({"total": len(summary), "by_category": by_category, "leads": summary}), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Route not found",
        "hint": "Visit GET /docs for Swagger UI or GET / for routes",
        "docs_url": "http://localhost:8000/docs",
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed on this route", "hint": "Visit GET /docs for Swagger UI"}), 405


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  Lead Scoring API")
    log.info("  API docs     : http://localhost:8000/docs")
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=8000, debug=False)