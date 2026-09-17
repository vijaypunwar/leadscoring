# step2_feature_engineering.py
import json
import math
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
LEADS_FILE = DATA_DIR / "leads_raw.json"
FOLLOWUPS_FILE = DATA_DIR / "followups_by_lead.json"
OUTPUT_CSV = DATA_DIR / "features.csv"
OUTPUT_FEATURE_COLUMNS = DATA_DIR / "feature_columns.json"

# ─────────────────────────── Constants ───────────────────────────
INTEREST_MAP = {"High": 3, "Medium": 2, "Low": 1, "": 0}
LEAD_STATUS_MAP = {"A": 8, "a": 8, "active": 8, "D": 1, "d": 1, "done": 1, "C": 0, "c": 0, "close": 0, "closed": 0, "": 0}
LEAD_STATUS_NEG = {"C": True, "c": True, "close": True, "closed": True}
LEAD_STATUS_DONE = {"D": True, "d": True, "done": True}
NEXT_MEETING_TYPE_MAP = {
    "Pending": 1, "Done": 8, "Close": 0, "Closed": 0, "Cancelled": 0,
    "Visit Done": 9, "Meeting Done": 10, "In-House Meeting": 8,
    "Out Door Meeting": 8, "Walk In Client": 7, "Open House/Event Meeting": 7,
    "Visit Schedule": 6, "Meeting Schedule": 5, "": 0
}
NEXT_MEETING_NEG = {"Pending": False, "Done": False, "Close": True, "Closed": True, "Cancelled": True}

POSITIVE_KEYWORDS = [
    r"\bbook(ed|ing)?\b", r"\bbuy(ing)?\b", r"\bpurchase(d)?\b", r"\binvest(ing|ment)?\b",
    r"\bown(er|ership)?\b", r"\bplot\b", r"\bproperty\b", r"\bsite\s*visit\b", r"\bvisit(ed)?\b",
    r"\binterested\b", r"\blike(d)?\b", r"\bagree(d)?\b", r"\bconfirm(ed|ation)?\b",
    r"\bready\b", r"\bseriously\b", r"\bserious\b", r"\bgood\s*(client|customer|buyer)\b",
    r"\baffordable\b", r"\bbudget\s*(ok|fine|match)\b", r"\bprice\s*(ok|fine|agreed|match)\b",
    r"\btoken\b", r"\bpayment\b", r"\badvance\b", r"\bdeal\b", r"\bsign(ed)?\b",
    r"\bagreement\b", r"\bcontract\b", r"\bconvert(ed|sion)?\b", r"\bmature\b",
    r"\bqualified\b", r"\bhot\b", r"\bvery\s*interested\b", r"\bhighly\s*interested\b"
]
NEGATIVE_KEYWORDS = [
    r"\bnot\s*interested\b", r"\bno\s*interest\b", r"\bdead\b", r"\brefus(e|ed|al)\b",
    r"\bdecline(d)?\b", r"\bcan'?t\s*afford\b", r"\bcannot\s*afford\b", r"\bbeyond\s*budget\b",
    r"\bno\s*response\b", r"\bno\s*reply\b", r"\bnot\s*respond(ing|ed)?\b",
    r"\bnot\s*pick(ing)?\b", r"\bunreach(able|ed)?\b", r"\bswitch(ed)?\s*off\b",
    r"\bnot\s*working\b", r"\bno\s*answer\b", r"\bcancel(led|ed|s)?\b",
    r"\bpostpone(d|s)?\b", r"\bdelay(ed|s)?\b", r"\blast\s*attempt\b", r"\bfinal\s*attempt\b"
]

# ─────────────────────────── Helpers ───────────────────────────
def load_json(path: Path):
    if not path.exists(): return None
    content = path.read_text(encoding="utf-8").strip()
    return json.loads(content) if content else None

def safe_float(value, default: float = 0.0) -> float:
    if value in (None, "", "null", "none", "nan"): return default
    try: return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError): return default

def normalize_text(value: Any) -> str:
    if value is None: return ""
    if isinstance(value, (dict, list)):
        vals = value.values() if isinstance(value, dict) else value
        text = " ".join(str(v) for v in vals if v is not None)
    else:
        text = str(value)
    text = text.strip().lower()
    return "" if text in {"null", "none", "nan"} else text

def normalize_value(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return "" if text.lower() in {"null", "none", "nan"} else text

def enc(mapping: dict, value: Any) -> int:
    return mapping.get(normalize_value(value), mapping.get(normalize_value(value).lower(), 0))

def parse_date(value: Any) -> Optional[datetime]:
    if value in (None, "", "null", "none"): return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d %H:%M:%S"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except (ValueError, TypeError):
            continue
    try:
        parsed = pd.to_datetime(str(value).strip(), errors="coerce")
        if not pd.isna(parsed): return parsed.to_pydatetime()
    except Exception:
        pass
    return None

def days_since(value: Any) -> float:
    dt = parse_date(value)
    return float(max((datetime.now() - dt).days, 0)) if dt else -1.0

def clip(value: float, low: float, high: float) -> float:
    return float(np.clip(value, low, high))

def get_lead_id(lead: dict) -> str:
    val = lead.get("LeadNo") or lead.get("Id")
    if val in (None, "", "null", "None"): return "0"
    try: return str(int(float(val)))
    except (TypeError, ValueError): return str(val).strip()

def extract_narration_text(record: dict) -> str:
    if not isinstance(record, dict): return ""
    parts = []
    for field in ("MeetingDetail", "NextMeetingDetails", "AgendaDetail", "Narration", "notes", "Note", "audioDescription"):
        v = record.get(field)
        if v is None: continue
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, dict):
            parts.extend(nv.strip() for nv in v.values() if isinstance(nv, str) and nv.strip())
        elif isinstance(v, list):
            parts.extend(str(i) for i in v if i)
    return " ".join(parts).strip().lower()

def count_keyword_hits(text: str, patterns: List[str]) -> int:
    if not text: return 0
    return sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))

def get_audio_seconds(audio_duration: Any) -> float:
    if audio_duration in (None, "", "null", "none"): return 0.0
    try:
        if isinstance(audio_duration, (int, float)): return float(audio_duration)
        parts = str(audio_duration).strip().split(":")
        if len(parts) == 2: return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(parts[0])
    except (ValueError, TypeError):
        return 0.0

def get_budget_pair(record: dict) -> Tuple[float, float]:
    b_from = safe_float(record.get("budget_from"))
    b_to = safe_float(record.get("budget_to"))
    return max(b_from, 0.0), max(b_to, 0.0)

# ─────────────────────────── Lead Scoring Rules ───────────────────────────
# Each rule contributes a (key, score, applied_flag) tuple. A key can only apply ONCE
# across the entire history (lead + all followups) — once applied, it's locked in.

def score_lead_status(lead: dict) -> Tuple[float, Dict[str, bool]]:
    """1. LeadStatus: A=active, D=done, C=close."""
    applied, score = {}, 0.0
    raw = normalize_value(lead.get("LeadStatus"))
    if not raw: return score, applied
    if LEAD_STATUS_NEG.get(raw, False):       # close
        score += 0.0;  applied["ls_close"] = True
    elif LEAD_STATUS_DONE.get(raw, False):    # done
        score += 4.0;  applied["ls_done"] = True
    else:                                     # active (default treated as A)
        score += 6.0;  applied["ls_active"] = True
    return score, applied

def score_interest_level(lead: dict, followups: List[dict], applied: Dict[str, bool]) -> Tuple[float, Dict[str, bool]]:
    """2. interestLevel: low/medium/high — take best across lead + followups (deduped)."""
    if "interest" in applied: return 0.0, applied
    values = [normalize_value(lead.get("InterestLevel"))]
    for f in followups:
        v = normalize_value(f.get("InterestLevel"))
        if v: values.append(v)
    best = max((INTEREST_MAP.get(v, 0) for v in values), default=0)
    applied["interest"] = True
    return float({0: 0, 1: 3, 2: 6, 3: 10}.get(best, 0)), applied

def score_meeting_details(followups: List[dict], applied: Dict[str, bool]) -> Tuple[float, Dict[str, bool]]:
    """3. meetingDetails: count positive keyword hits across all followups (one-time score)."""
    if "meeting_details" in applied: return 0.0, applied
    total_hits = sum(count_keyword_hits(extract_narration_text(f), POSITIVE_KEYWORDS) for f in followups)
    applied["meeting_details"] = True
    return float(min(total_hits * 1.5, 12.0)), applied

def score_next_meeting_details(followups: List[dict], applied: Dict[str, bool]) -> Tuple[float, Dict[str, bool]]:
    """4. nextMeetingDetails: same scoring as meetingDetails but for NextMeetingDetails field."""
    if "next_meeting_details" in applied: return 0.0, applied
    total_hits = 0
    for f in followups:
        v = f.get("NextMeetingDetails")
        if v and str(v).strip():
            total_hits += count_keyword_hits(str(v).lower(), POSITIVE_KEYWORDS)
    applied["next_meeting_details"] = True
    return float(min(total_hits * 1.2, 8.0)), applied

def score_next_meeting_status(lead: dict, followups: List[dict], applied: Dict[str, bool]) -> Tuple[float, Dict[str, bool]]:
    """5. nextMeetingStatus: pending/done/close."""
    if "next_meeting_status" in applied: return 0.0, applied
    raw = normalize_value(lead.get("NextMeetingType"))
    if not raw: raw = "Pending"
    if raw in ("Close", "Closed", "Cancelled"):
        applied["next_meeting_status"] = True
        return 0.0, applied
    if raw == "Done":
        applied["next_meeting_status"] = True
        return 5.0, applied
    # Pending: small positive signal if any meeting details exist
    has_meeting = any(extract_narration_text(f) for f in followups)
    applied["next_meeting_status"] = True
    return (3.0 if has_meeting else 1.0), applied

def score_budget(lead: dict, followups: List[dict], applied: Dict[str, bool]) -> Tuple[float, Dict[str, bool]]:
    """6. budget_from or budget_to: any positive budget amount across lead + followups."""
    if "budget" in applied: return 0.0, applied
    all_records = [lead] + list(followups)
    max_budget = 0.0
    for r in all_records:
        b_from, b_to = get_budget_pair(r)
        max_budget = max(max_budget, b_from, b_to)
    applied["budget"] = True
    if max_budget <= 0: return 0.0, applied
    # Tier-based score
    if max_budget >= 50_000_000: return 8.0, applied
    if max_budget >= 20_000_000: return 6.0, applied
    if max_budget >= 10_000_000: return 4.0, applied
    if max_budget >= 1_000_000:  return 3.0, applied
    return 1.0, applied

def score_lead_stage(lead: dict, followups: List[dict], applied: Dict[str, bool]) -> Tuple[float, Dict[str, bool]]:
    """7. leadStage: done/close/active."""
    if "lead_stage" in applied: return 0.0, applied
    raw = normalize_value(lead.get("lead_stages")).lower()
    if not raw: raw = "active"
    applied["lead_stage"] = True
    if raw in ("close", "closed", "cancelled"): return 0.0, applied
    if raw in ("done", "completed"): return 4.0, applied
    # Active stage — bonus for advanced stages
    stage_bonus = {
        "new lead": 1.0, "contacted": 2.0, "qualified": 4.0,
        "presentation done": 5.0, "site visit / zoom": 6.0,
        "followup": 6.0, "token received": 9.0, "booking": 10.0
    }.get(raw, 3.0)
    return stage_bonus, applied

def score_call_status(followups: List[dict], applied: Dict[str, bool]) -> Tuple[float, Dict[str, bool]]:
    """8. f_call_status: walk-in client etc. (best signal across followups)."""
    if "call_status" in applied: return 0.0, applied
    values = []
    for f in followups:
        v = normalize_value(f.get("f_call_status"))
        if v: values.append(v.lower())
    applied["call_status"] = True
    if not values: return 0.0, applied
    mapping = {
        "walk-in client / meeting": 6.0, "walk-in client": 6.0,
        "meeting done": 6.0, "visit done": 6.0, "in-house meeting": 5.0,
        "call connected": 3.0, "followup call": 2.0,
        "attempt call": 0.5, "call attempt": 0.5, "social calls": 1.0
    }
    best = max((mapping.get(v, 1.0) for v in values), default=0.0)
    return best, applied

def score_next_task_type(followups: List[dict], applied: Dict[str, bool]) -> Tuple[float, Dict[str, bool]]:
    """9. NextTaskType: call etc. (best signal across followups)."""
    if "next_task_type" in applied: return 0.0, applied
    values = []
    for f in followups:
        v = normalize_value(f.get("NextTaskType"))
        if v: values.append(v.lower())
    applied["next_task_type"] = True
    if not values: return 0.0, applied
    mapping = {"call": 2.0, "meeting": 3.0, "visit": 3.0, "site visit": 3.0, "followup": 2.0}
    return max((mapping.get(v, 0.5) for v in values), default=0.0), applied

def score_audio_duration(followups: List[dict], applied: Dict[str, bool]) -> Tuple[float, Dict[str, bool]]:
    """10. audioDuration: if > 2 minutes then +5."""
    if "audio_duration" in applied: return 0.0, applied
    applied["audio_duration"] = True
    for f in followups:
        seconds = get_audio_seconds(f.get("AudioDuration"))
        if seconds > 120: return 5.0, applied
    return 0.0, applied

def score_audio_description(followups: List[dict], applied: Dict[str, bool]) -> Tuple[float, Dict[str, bool]]:
    """11. audioDescription: positive talk about buying/property/interest."""
    if "audio_description" in applied: return 0.0, applied
    total_pos = 0
    total_neg = 0
    for f in followups:
        text = normalize_text(f.get("audioDescription"))
        if not text: continue
        total_pos += count_keyword_hits(text, POSITIVE_KEYWORDS)
        total_neg += count_keyword_hits(text, NEGATIVE_KEYWORDS)
    applied["audio_description"] = True
    net = total_pos - total_neg
    if net <= 0: return 0.0, applied
    return float(min(net * 1.0, 6.0)), applied

def score_response_speed(lead: dict, applied: Dict[str, bool]) -> Tuple[float, Dict[str, bool]]:
    """(Lead parameter) FollowupCreatedAt → now: faster = higher score."""
    if "response_speed" in applied: return 0.0, applied
    created_str = normalize_value(lead.get("FollowupCreatedAt") or lead.get("PostingDate"))
    dt = parse_date(created_str)
    applied["response_speed"] = True
    if not dt: return 0.0, applied
    # Cap to future dates just in case
    delta = abs((datetime.now() - dt).total_seconds())
    minutes = delta / 60.0
    if minutes <= 60:    return 5.0, applied   # within 1 hour
    if minutes <= 1440:  return 4.0, applied   # within 1 day
    if minutes <= 4320:  return 3.0, applied   # within 3 days
    if minutes <= 10080: return 2.0, applied   # within 1 week
    if minutes <= 43200: return 1.0, applied   # within 30 days
    return 0.0, applied

def calculate_lead_score(lead: dict, followups: List[dict]) -> float:
    """Aggregate the 11 scored parameters + response speed, capped at 1..95."""
    applied: Dict[str, bool] = {}
    parts = [
        score_lead_status(lead),
        score_interest_level(lead, followups, applied),
        score_meeting_details(followups, applied),
        score_next_meeting_details(followups, applied),
        score_next_meeting_status(lead, followups, applied),
        score_budget(lead, followups, applied),
        score_lead_stage(lead, followups, applied),
        score_call_status(followups, applied),
        score_next_task_type(followups, applied),
        score_audio_duration(followups, applied),
        score_audio_description(followups, applied),
        score_response_speed(lead, applied),
    ]
    total = sum(score for score, _ in parts)
    # Penalty: count negative keywords in narration (this is a one-time cross-field penalty)
    all_text = extract_narration_text(lead) + " " + " ".join(extract_narration_text(f) for f in followups)
    neg_hits = count_keyword_hits(all_text, NEGATIVE_KEYWORDS)
    total -= min(neg_hits * 2.0, 15.0)
    return clip(round(total, 2), 1.0, 95.0)

# ─────────────────────────── Feature Engineering (kept for ML) ───────────────────────────
def lead_features(lead: dict) -> dict:
    b_from, b_to = get_budget_pair(lead)
    intr = normalize_value(lead.get("InterestLevel"))
    stg = normalize_value(lead.get("lead_stages"))
    ls = normalize_value(lead.get("LeadStatus"))
    nxt = normalize_value(lead.get("NextMeetingType"))
    return {
        "lead_interest_enc": INTEREST_MAP.get(intr, 0),
        "lead_status_enc": LEAD_STATUS_MAP.get(ls, 0),
        "lead_stage_enc": enc({"New Lead": 1, "Contacted": 2, "Qualified": 3, "Presentation Done": 4,
                               "Site Visit / Zoom": 5, "FollowUp": 6, "Token Received": 8, "Booking": 8}, stg),
        "next_meeting_type_enc": NEXT_MEETING_TYPE_MAP.get(nxt, 0),
        "lead_budget_from": b_from, "lead_budget_to": b_to,
        "lead_has_budget": int(b_to > 0),
        "lead_is_closed": int(ls.lower() in ("c", "close", "closed") or nxt.lower() in ("close", "closed")),
        "days_since_posting": days_since(lead.get("PostingDate")),
        "days_since_followup_created": days_since(lead.get("FollowupCreatedAt")),
    }

def followup_features(followups: List[dict]) -> dict:
    empty = {
        "fup_count": 0, "fup_positive_hits": 0, "fup_negative_hits": 0,
        "fup_has_audio_long": 0, "fup_total_audio_seconds": 0.0,
        "fup_max_budget": 0.0, "fup_avg_days_gap": -1.0, "fup_days_since_last": -1.0,
    }
    if not followups: return empty
    pos = neg = 0
    audio_secs = 0.0
    has_long_audio = 0
    budgets = []
    for f in followups:
        text = extract_narration_text(f)
        pos += count_keyword_hits(text, POSITIVE_KEYWORDS)
        neg += count_keyword_hits(text, NEGATIVE_KEYWORDS)
        secs = get_audio_seconds(f.get("AudioDuration"))
        audio_secs += secs
        if secs > 120: has_long_audio = 1
        _, b_to = get_budget_pair(f)
        budgets.append(b_to)
    dates = sorted(d for f in followups if (d := parse_date(f.get("PostingDate") or f.get("Date"))))
    avg_gap = -1.0
    if len(dates) > 1:
        avg_gap = float(np.mean([(dates[i] - dates[i-1]).days for i in range(1, len(dates))]))
    days_last = float(max((datetime.now() - dates[-1]).days, 0)) if dates else -1.0
    return {
        "fup_count": len(followups),
        "fup_positive_hits": pos, "fup_negative_hits": neg,
        "fup_has_audio_long": has_long_audio, "fup_total_audio_seconds": round(audio_secs, 2),
        "fup_max_budget": max(budgets) if budgets else 0.0,
        "fup_avg_days_gap": round(avg_gap, 2), "fup_days_since_last": round(days_last, 2),
    }

def score_label(score: float) -> str:
    for limit, label in [(20, "Cold"), (40, "Warm"), (60, "Interested"), (80, "Hot")]:
        if score <= limit: return label
    return "Very Hot"

def build_feature_row(lead: dict, followups: List[dict]) -> dict:
    row = {"lead_id": get_lead_id(lead)}
    row.update(lead_features(lead))
    row.update(followup_features(followups))
    row["score"] = calculate_lead_score(lead, followups)
    row["score_category"] = score_label(row["score"])
    return row

def main():
    leads = load_json(LEADS_FILE)
    followups_by_lead = load_json(FOLLOWUPS_FILE) or {}
    if leads is None or not isinstance(leads, list) or not isinstance(followups_by_lead, dict): return
    rows = []
    for lead in leads:
        lead_id = get_lead_id(lead)
        followups = followups_by_lead.get(lead_id, [])
        if isinstance(followups, list) and followups:
            rows.append(build_feature_row(lead, followups))
    if not rows: return
    df = pd.DataFrame(rows)
    excluded = {"lead_id", "score", "score_category"}
    feature_columns = [c for c in df.columns if c not in excluded]
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    OUTPUT_FEATURE_COLUMNS.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
