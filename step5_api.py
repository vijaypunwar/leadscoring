"""
STEP 5 — REST API (Flask) with Swagger UI
==========================================
Exposes endpoints:

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

Then open in your browser:
  http://localhost:8000/docs
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is on path so step4_predict can be imported
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify
from step4_predict import predict_score

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── In-memory score store (replace with DB in production) ─────────────────────
score_store: dict = {}

# ── Paths to trained artifacts ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR   = PROJECT_ROOT / "models"
MODEL_FILE   = MODELS_DIR / "xgb_lead_scorer.model"
SCALER_FILE  = MODELS_DIR / "scaler.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Startup validation
# ─────────────────────────────────────────────────────────────────────────────
def _check_artifacts():
    """Fail fast with a clear message if the model has not been trained yet."""
    missing = []
    if not MODEL_FILE.exists():
        missing.append(str(MODEL_FILE))
    if not SCALER_FILE.exists():
        missing.append(str(SCALER_FILE))
    if missing:
        log.error("=" * 60)
        log.error(" MISSING MODEL ARTIFACTS")
        for m in missing:
            log.error("   - %s", m)
        log.error(" Run step 3 first:  python step3_train_model.py")
        log.error("=" * 60)
        raise FileNotFoundError(f"Missing model files: {missing}")


_check_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _utc_now() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _extract_lead_id(lead: dict):
    """Try to extract a numeric lead ID from common key names."""
    for key in ("LeadNo", "lead_id", "leadId", "id", "ID"):
        val = lead.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    return None


def _validate_body(body) -> tuple:
    """
    Validate the request body.

    Returns
    -------
    (lead, followups, error_response)
        If validation passes  → (lead_dict, followups_list, None)
        If validation fails   → (None, None, error_dict)
    """
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
    """
    Recursively convert NumPy types, datetime, and other non-JSON-serializable
    objects into plain Python types so Flask jsonify can handle them.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.generic):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _to_python(obj.tolist())
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Score category & recommendation logic
#
# Aligned with step 2 (feature engineering) score bands:
#
#     0-20    Cold
#     21-40   Warm
#     41-60   Interested
#     61-80   Hot
#     81-100  Very Hot
# ─────────────────────────────────────────────────────────────────────────────
def _score_category(score: float) -> str:
    """Map a 0-100 score to a category name (matches step 2)."""
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
    """Map a 0-100 score to a recommended action."""
    if score > 80:
        return "Contact immediately — very hot lead"
    if score > 60:
        return "Follow up within 24 hours — hot lead"
    if score > 40:
        return "Continue regular nurturing — interested lead"
    if score > 20:
        return "Re-engage with a fresh offer or information — warm lead"
    return "Low priority — cold lead, revisit later"


def _extract_status_flags(result: dict) -> dict:
    """
    Pull status classification flags out of the predict_score result,
    if predict_score is exposing them.

    Returns a dict of the status_* flags (defaults to 0 if missing).
    """
    keys = [
        "status_is_closed",
        "status_is_cancelled",
        "status_is_dead",
        "status_terminal_positive",
        "status_terminal_negative",
        "status_closed_lost",
        "status_is_active",
    ]
    return {k: int(result.get(k, 0)) for k in keys}


def _extract_score_breakdown(result: dict) -> dict:
    """
    Pull heuristic score component breakdown from predict_score result
    (if available). Returns None if no breakdown fields are present.
    """
    keys = [
        "lead_quality",
        "engagement",
        "buying_intent",
        "negative_risk",
        "recency",
    ]
    breakdown = {k: result.get(k) for k in keys if k in result}
    return breakdown if breakdown else None


def _build_response(
    lead_id: int,
    score: float,
    followups: list,
    scored_at: str,
    result: dict = None,
) -> dict:
    """Build the standardized response, aligned with step 2 outputs."""
    score = float(score)
    response = {
        "lead_id": lead_id,
        "score": round(score, 1),
        "score_category": _score_category(score),
        "total_followups": len(followups) if followups else 0,
        "recommendations": _recommendation_from_score(score),
        "datetime": scored_at,
    }

    # Optional enriched fields if predict_score returned them.
    if result:
        status_flags = _extract_status_flags(result)
        if any(status_flags.values()):
            response["status"] = status_flags

        breakdown = _extract_score_breakdown(result)
        if breakdown:
            response["score_breakdown"] = breakdown

    return response


# ─────────────────────────────────────────────────────────────────────────────
# OpenAPI / Swagger specification
# ─────────────────────────────────────────────────────────────────────────────
SWAGGER_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Lead Scoring API",
        "description": (
            "End-to-end lead scoring powered by an XGBoost regression model. "
            "Accepts a lead payload and optional follow-ups, then returns a "
            "0–100 score with a category (Cold / Warm / Interested / Hot / "
            "Very Hot), follow-up count, recommendation, optional status "
            "flags (closed / cancelled / dead / terminal-positive), and an "
            "optional score breakdown (lead quality, engagement, buying "
            "intent, negative risk, recency)."
        ),
        "version": "1.1.0",
        "contact": {"name": "API Support"},
    },
    "servers": [
        {"url": "http://localhost:8000", "description": "Local development server"}
    ],
    "tags": [
        {"name": "Info", "description": "Service metadata & health"},
        {"name": "Score", "description": "Score and retrieve lead scores"},
        {"name": "Docs", "description": "API documentation"},
    ],
    "paths": {
        "/": {
            "get": {
                "tags": ["Info"],
                "summary": "API information",
                "operationId": "index",
                "responses": {
                    "200": {
                        "description": "Basic API info and available routes",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ApiInfo"}
                            }
                        },
                    }
                },
            }
        },
        "/health": {
            "get": {
                "tags": ["Info"],
                "summary": "Health check",
                "operationId": "health",
                "responses": {
                    "200": {
                        "description": "Health status",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Health"}
                            }
                        },
                    }
                },
            }
        },
        "/score": {
            "post": {
                "tags": ["Score"],
                "summary": "Score a new lead",
                "operationId": "score_lead",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ScoreRequest"}
                        }
                    },
                },
                "responses": {
                    "201": {
                        "description": "Lead scored successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ScoreResponse"}
                            }
                        },
                    },
                    "400": {
                        "description": "Validation error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        },
                    },
                    "500": {
                        "description": "Prediction failure",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        },
                    },
                },
            }
        },
        "/score/{lead_id}": {
            "put": {
                "tags": ["Score"],
                "summary": "Re-score / update an existing lead",
                "operationId": "update_score",
                "parameters": [
                    {
                        "name": "lead_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                        "description": "Numeric ID of the lead to update",
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ScoreRequest"}
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Lead re-scored successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ScoreResponse"}
                            }
                        },
                    },
                    "400": {
                        "description": "Validation error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        },
                    },
                    "500": {
                        "description": "Prediction failure",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        },
                    },
                },
            },
            "get": {
                "tags": ["Score"],
                "summary": "Get cached score",
                "operationId": "get_score",
                "parameters": [
                    {
                        "name": "lead_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                        "description": "Numeric ID of the lead",
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Cached score found",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ScoreResponse"}
                            }
                        },
                    },
                    "404": {
                        "description": "Lead not yet scored",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Error"}
                            }
                        },
                    },
                },
            },
        },
        "/scores": {
            "get": {
                "tags": ["Score"],
                "summary": "List all cached scores",
                "operationId": "list_scores",
                "parameters": [
                    {
                        "name": "category",
                        "in": "query",
                        "required": False,
                        "schema": {
                            "type": "string",
                            "enum": ["Cold", "Warm", "Interested", "Hot", "Very Hot"],
                        },
                        "description": "Optional filter by score category",
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Summary of all scored leads",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ScoreList"}
                            }
                        },
                    }
                },
            }
        },
        "/swagger.json": {
            "get": {
                "tags": ["Docs"],
                "summary": "OpenAPI specification (JSON)",
                "operationId": "swagger_spec",
                "responses": {
                    "200": {
                        "description": "OpenAPI 3.0 JSON spec",
                        "content": {
                            "application/json": {
                                "schema": {"type": "object"}
                            }
                        },
                    }
                },
            }
        },
        "/docs": {
            "get": {
                "tags": ["Docs"],
                "summary": "Swagger UI",
                "operationId": "swagger_ui",
                "responses": {
                    "200": {
                        "description": "HTML page rendering Swagger UI",
                        "content": {"text/html": {"schema": {"type": "string"}}},
                    }
                },
            }
        },
    },
    "components": {
        "schemas": {
            "ApiInfo": {
                "type": "object",
                "properties": {
                    "api": {"type": "string", "example": "Lead Scoring API"},
                    "version": {"type": "string", "example": "1.1.0"},
                    "status": {"type": "string", "example": "running"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "leads_scored": {"type": "integer", "example": 0},
                    "docs_url": {"type": "string", "example": "http://localhost:8000/docs"},
                },
            },
            "Health": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "example": "ok"},
                    "models_ready": {"type": "boolean", "example": True},
                    "leads_scored": {"type": "integer", "example": 0},
                    "timestamp": {"type": "string", "format": "date-time"},
                },
            },
            "ScoreRequest": {
                "type": "object",
                "required": ["lead"],
                "properties": {
                    "lead": {
                        "type": "object",
                        "description": (
                            "Lead fields such as LeadNo, Potential/InterestLevel, "
                            "sourceCode, stage_name, LeadStatus, NextMeetingType, "
                            "budget_from, budget_to, MeetingDetail, etc."
                        ),
                        "example": {
                            "LeadNo": 3678,
                            "Potential": "Medium",
                            "sourceCode": "FB",
                            "stage_name": "Qualified",
                            "LeadStatus": "A",
                            "budget_from": 5000000,
                            "budget_to": 8000000,
                        },
                    },
                    "followups": {
                        "type": "array",
                        "description": "Optional list of follow-up records for this lead",
                        "items": {"type": "object"},
                        "example": [
                            {
                                "LeadNo": 3678,
                                "meeting_type_name": "Meeting Done",
                                "stage_name": "Qualified",
                                "Potential": "High",
                                "MeetingDate": "2024-05-20",
                                "MeetingDetail": "Client visited site, very interested, discussing payment plan.",
                            }
                        ],
                    },
                },
            },
            "StatusFlags": {
                "type": "object",
                "description": "Lead lifecycle status flags derived from CRM data and narration.",
                "properties": {
                    "status_is_closed":         {"type": "integer", "example": 0},
                    "status_is_cancelled":      {"type": "integer", "example": 0},
                    "status_is_dead":           {"type": "integer", "example": 0},
                    "status_terminal_positive": {"type": "integer", "example": 0},
                    "status_terminal_negative": {"type": "integer", "example": 0},
                    "status_closed_lost":       {"type": "integer", "example": 0},
                    "status_is_active":         {"type": "integer", "example": 1},
                },
            },
            "ScoreBreakdown": {
                "type": "object",
                "description": "Heuristic score components (each on its own scale).",
                "properties": {
                    "lead_quality":  {"type": "number", "example": 12.0, "description": "0–20"},
                    "engagement":    {"type": "number", "example": 18.5, "description": "0–25"},
                    "buying_intent": {"type": "number", "example": 22.0, "description": "0–35"},
                    "negative_risk": {"type": "number", "example":  0.0, "description": "0–15 (penalty)"},
                    "recency":       {"type": "number", "example":  4.0, "description": "0–5"},
                },
            },
            "ScoreResponse": {
                "type": "object",
                "properties": {
                    "lead_id": {"type": "integer", "example": 3678},
                    "score": {
                        "type": "number",
                        "format": "float",
                        "example": 72.4,
                        "description": "Score from 0 to 100",
                    },
                    "score_category": {
                        "type": "string",
                        "enum": ["Cold", "Warm", "Interested", "Hot", "Very Hot"],
                        "example": "Hot",
                        "description": "Category derived from the 0–100 score band",
                    },
                    "total_followups": {
                        "type": "integer",
                        "example": 3,
                        "description": "Number of follow-up records provided",
                    },
                    "recommendations": {
                        "type": "string",
                        "example": "Follow up within 24 hours — hot lead",
                        "description": "Recommended action based on score tier",
                    },
                    "datetime": {
                        "type": "string",
                        "format": "date-time",
                        "example": "2024-06-01T10:00:00+00:00",
                        "description": "UTC timestamp when the score was generated",
                    },
                    "status": {
                        "$ref": "#/components/schemas/StatusFlags",
                        "description": "Included when the underlying model exposes status flags",
                    },
                    "score_breakdown": {
                        "$ref": "#/components/schemas/ScoreBreakdown",
                        "description": "Included when the underlying model exposes component scores",
                    },
                },
            },
            "ScoreList": {
                "type": "object",
                "properties": {
                    "total": {"type": "integer", "example": 2},
                    "by_category": {
                        "type": "object",
                        "additionalProperties": {"type": "integer"},
                        "example": {
                            "Cold": 1,
                            "Warm": 0,
                            "Interested": 0,
                            "Hot": 1,
                            "Very Hot": 0,
                        },
                    },
                    "leads": {
                        "type": "array",
                        "items": {"$ref": "#/components/schemas/ScoreResponse"},
                    },
                },
            },
            "Error": {
                "type": "object",
                "properties": {
                    "error": {"type": "string", "example": "JSON body required"},
                    "hint": {"type": "string", "example": "Visit GET /docs for Swagger UI"},
                },
            },
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
    <style>
        html { box-sizing: border-box; overflow: -moz-scrollbars-vertical; overflow-y: scroll; }
        *, *:before, *:after { box-sizing: inherit; }
        body { margin: 0; background: #fafafa; }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script>
        window.onload = function() {
            window.ui = SwaggerUIBundle({
                url: "/swagger.json",
                dom_id: "#swagger-ui",
                layout: "BaseLayout",
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIBundle.presets.standalone
                ],
            });
        };
    </script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    """
    Root route — returns API info and all available endpoints.
    """
    return jsonify({
        "api":          "Lead Scoring API",
        "version":      "1.1.0",
        "status":       "running",
        "timestamp":    _utc_now(),
        "leads_scored": len(score_store),
        "docs_url":     "http://localhost:8000/docs",
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
    """Simple liveness probe."""
    return jsonify({
        "status":        "ok",
        "models_ready":  True,
        "leads_scored":  len(score_store),
        "timestamp":     _utc_now(),
    }), 200


@app.route("/swagger.json", methods=["GET"])
def swagger_spec():
    """Return the OpenAPI 3.0 JSON specification."""
    return jsonify(SWAGGER_SPEC), 200


@app.route("/docs", methods=["GET"])
def swagger_ui():
    """Serve Swagger UI HTML."""
    return SWAGGER_UI_HTML, 200


@app.route("/score", methods=["POST"])
def score_lead():
    """
    Score a lead for the first time.

    Request body: { "lead": {...}, "followups": [...] }
    """
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
        log.warning("Could not determine lead_id from payload")
        return jsonify({"error": "Unable to extract lead_id from lead object"}), 400

    scored_at = _utc_now()
    score_val = float(result.get("score", 0))
    response = _build_response(lead_id, score_val, followups, scored_at, result)

    score_store[lead_id] = response.copy()
    log.info(
        "Scored   lead %-8s → %5.1f  [%s]  (%s)",
        lead_id, score_val, response["score_category"], response["recommendations"],
    )

    return jsonify(response), 201


@app.route("/score/<int:lead_id>", methods=["PUT"])
def update_score(lead_id: int):
    """
    Re-score an existing lead (e.g. after new follow-ups arrive).

    Request body: { "lead": {...}, "followups": [...] }
    """
    body = request.get_json(force=True, silent=True)
    lead, followups, err = _validate_body(body)
    if err:
        log.warning("PUT /score/%s validation failed: %s", lead_id, err)
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
    scored_at = _utc_now()
    score_val = float(result.get("score", 0))
    response = _build_response(lead_id, score_val, followups, scored_at, result)

    score_store[lead_id] = response.copy()

    if previous_score is not None:
        delta = round(score_val - previous_score, 2)
        log.info(
            "Updated  lead %-8s → %5.1f  [%s]  (%s)  Δ%+.1f",
            lead_id, score_val, response["score_category"],
            response["recommendations"], delta,
        )
    else:
        log.info(
            "Updated  lead %-8s → %5.1f  [%s]  (%s)",
            lead_id, score_val, response["score_category"], response["recommendations"],
        )

    return jsonify(response), 200


@app.route("/score/<int:lead_id>", methods=["GET"])
def get_score(lead_id: int):
    """
    Retrieve the latest cached score for a lead without re-scoring.
    """
    cached = score_store.get(lead_id)
    if not cached:
        log.warning("GET /score/%s → not found in store", lead_id)
        return jsonify({
            "error": f"Lead {lead_id} has not been scored yet",
            "hint":  f"POST /score first, then GET /score/{lead_id}",
        }), 404

    log.info(
        "Fetched  lead %-8s → %5.1f  [%s]",
        lead_id, cached["score"], cached.get("score_category", "?"),
    )
    return jsonify(cached), 200


@app.route("/scores", methods=["GET"])
def list_scores():
    """
    Return a summary of all scored leads.

    Optional query params:
        ?category=Cold|Warm|Interested|Hot|Very Hot
    """
    category_filter = request.args.get("category")
    valid_categories = {"Cold", "Warm", "Interested", "Hot", "Very Hot"}

    summary = list(score_store.values())

    if category_filter:
        if category_filter not in valid_categories:
            return jsonify({
                "error": f"Invalid category '{category_filter}'",
                "hint":  f"Use one of: {sorted(valid_categories)}",
            }), 400
        summary = [s for s in summary if s.get("score_category") == category_filter]

    # Sort by score descending (hottest leads first)
    summary.sort(key=lambda x: x["score"], reverse=True)

    # Category counts across the full store (not just the filtered view)
    by_category = {cat: 0 for cat in ["Cold", "Warm", "Interested", "Hot", "Very Hot"]}
    for s in score_store.values():
        cat = s.get("score_category")
        if cat in by_category:
            by_category[cat] += 1

    log.info(
        "Listed scores — %d leads%s",
        len(summary),
        f" (filter={category_filter})" if category_filter else "",
    )
    return jsonify({
        "total": len(summary),
        "by_category": by_category,
        "leads": summary,
    }), 200


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Route not found",
        "hint":  "Visit GET /docs for Swagger UI or GET / for routes",
        "docs_url": "http://localhost:8000/docs",
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "error": "Method not allowed on this route",
        "hint":  "Visit GET /docs for Swagger UI",
    }), 405


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  Lead Scoring API")
    log.info("  Project root : %s", PROJECT_ROOT)
    log.info("  Models dir   : %s", MODELS_DIR)
    log.info("  API docs     : http://localhost:8000/docs")
    log.info("  OpenAPI spec : http://localhost:8000/swagger.json")
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=8000, debug=False)