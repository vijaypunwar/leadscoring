"""
STEP 2 - Feature Engineering + Lead Scoring
"""

import json
import math
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
LEADS_FILE = DATA_DIR / "leads_raw.json"
FOLLOWUPS_FILE = DATA_DIR / "followups_by_lead.json"

OUTPUT_CSV = DATA_DIR / "features.csv"
OUTPUT_FEATURE_COLUMNS = DATA_DIR / "feature_columns.json"

INTEREST_MAP = {
    "High": 3,
    "Medium": 2,
    "Low": 1,
    "": 0,
}

STAGE_MAP = {
    "Booking": 8,
    "Token Received": 8,
    "FollowUp": 6,
    "Site Visit / Zoom": 5,
    "Presentation Done": 4,
    "Qualified": 3,
    "Contacted": 2,
    "New Lead": 1,
    "": 0,
}


MEETING_TYPE_MAP = {
    "Meeting Done": 10,
    "Visit Done": 9,
    "In-House Meeting": 8,
    "Out Door Meeting": 8,
    "Open House/Event Meeting": 7,
    "Walk In Client": 7,
    "Visit Schedule": 6,
    "Meeting Schedule": 5,
    "Done": 5,
    "Call Connected": 4,
    "Social Calls": 3,
    "Followup Call": 3,
    "Attempt Call": 2,
    "Call Attempt": 2,
    "Pending": 1,
    "Visit Postpone": 1,
    "Meeting Postpone": 1,
    "": 0,
}


SOURCE_MAP = {
    "FB": 1,
    "WEB": 1,
    "REF": 3,
    "WALK-IN": 4,
    "CALL": 5,
    "": 0,
}


PURPOSE_MAP = {
    "Own": 2,
    "Investment": 1,
    "": 0,
}

POSITIVE_MEETINGS = {
    "Meeting Done",
    "Visit Done",
    "In-House Meeting",
    "Out Door Meeting",
    "Open House/Event Meeting",
    "Walk In Client",
}

SCHEDULED_MEETINGS = {
    "Visit Schedule",
    "Meeting Schedule",
}

CALL_TYPES = {
    "Call Connected",
    "Followup Call",
    "Social Calls",
    "Attempt Call",
    "Call Attempt",
}

ATTEMPT_TYPES = {
    "Attempt Call",
    "Call Attempt",
}

POSTPONED_TYPES = {
    "Visit Postpone",
    "Meeting Postpone",
}

VISIT_DONE_TYPES = {
    "Visit Done",
}

MEETING_DONE_TYPES = {
    "Meeting Done",
}

CLOSED_VALUES = {
    "closed",
    "close",
    "closing",
}

CANCELLED_VALUES = {
    "cancelled",
    "canceled",
    "cancel",
    "cancelled lead",
    "canceled lead",
}

DEAD_VALUES = {
    "dead",
    "dead lead",
    "invalid",
    "lost",
    "not interested",
}

MEETING_TYPE_FIELDS = [
    "meeting_type_name",
    "FMeetingStatus",
    "FCallStatus",
    "f_call_status",
    "NextMeetingType",
]

DATE_FIELDS = [
    "meetingdatestring",
    "MeetingDate",
    "Date",
    "PostingDate",
    "AssignDate",
    "FollowupDate",
]

STAGE_FIELDS = [
    "stage_name",
    "lead_stages",
    "LeadStage",
    "CurrentStage",
]

INTEREST_FIELDS = [
    "Potential",
    "InterestLevel",
    "interest_level",
]

NARRATION_FIELDS = [
    "MeetingDetail",
    "AgendaDetail",
    "Narration",
    "NextMeetingDetails",
    "notes",
    "Note",
]

TOKEN_PATTERNS = [
    r"\btoken\b",
    r"\btokan\b",
    r"\btoken\s*money\b",
    r"\btoken\s*paid\b",
    r"\bpaid\s*token\b",
    r"\btoken\s*received\b",
]

PAYMENT_PATTERNS = [
    r"\bpayment\s*done\b",
    r"\bpayment\s*received\b",
    r"\badvance\s*paid\b",
    r"\badvance\s*payment\b",
    r"\bdown\s*payment\b",
    r"\btransfer\s*(amount|money|rs|pk)?\b",
]

BOOKING_PATTERNS = [
    r"\bbooking\s*done\b",
    r"\bbooked\b",
    r"\bbooking\s*confirmed\b",
    r"\bdeal\s*done\b",
    r"\bdeal\s*finalized\b",
    r"\bdeal\s*finalised\b",
]

CONVERSION_PATTERNS = [
    r"\bconverted\b",
    r"\bconvert(ed)?\b",
    r"\blead\s*mature\b",
    r"\bmature\s*lead\b",
    r"\bmature\b",
]

CONFIRMATION_PATTERNS = [
    r"\bconfirmed\b",
    r"\bconfirmation\b",
    r"\bfinalizing\b",
    r"\bfinalising\b",
    r"\bsigned\s*(agreement|contract|deed|noc)\b",
    r"\bagreement\s*signed\b",
    r"\bcontract\s*signed\b",
]

PROPERTY_SELECTED_PATTERNS = [
    r"\bplot\s*selected\b",
    r"\bunit\s*selected\b",
    r"\bproperty\s*selected\b",
    r"\bselected\s*(a\s*)?(plot|unit|flat|house|property)\b",
    r"\bfinal\s*(plot|unit|property)\s*selected\b",
]

SITE_VISIT_PATTERNS = [
    r"\bsite\s*visit\s*done\b",
    r"\bvisit\s*done\b",
    r"\bvisited\s*(the\s*)?(site|plot|property|project)\b",
    r"\bvisit(ed)?\s*(the\s*)?(site|plot|property|project)\b",
]

STRONG_INTEREST_PATTERNS = [
    r"\bready\s*to\s*buy\b",
    r"\bready\s*to\s*purchase\b",
    r"\bwants?\s*to\s*(buy|purchase|invest)\b",
    r"\bvery\s*interested\b",
    r"\bhighly\s*interested\b",
    r"\bserious\s*(buyer|client|customer)?\b",
    r"\bserious\s*about\b",
]

MODERATE_INTEREST_PATTERNS = [
    r"\binterested\b",
    r"\bwill\s*(visit|come|meet)\b",
    r"\bvisit\s*(on|at)\b",
    r"\bnext\s*(meeting|visit|call)\b",
    r"\bmeeting\s*scheduled\b",
    r"\bvisit\s*scheduled\b",
    r"\bscheduled\b",
    r"\bqualified\b",
    r"\bpresentation\s*done\b",
]

POSITIVE_QUALITY_PATTERNS = [
    r"\bbudget\s*(ok|fine|approved|matched|suitable)\b",
    r"\baffordable\b",
    r"\bprice\s*(ok|fine|agreed|negotiated|matched)\b",
    r"\bprice\s*agreed\b",
]


NOT_INTERESTED_PATTERNS = [
    r"\bnot\s*interested\b",
    r"\bno\s*interest\b",
    r"\bnot\s*interest\b",
    r"\bdead\s*lead\b",
    r"\bdead\b",
    r"\brefused\b",
    r"\brefuse\b",
    r"\bdeclined\b",
    r"\bdecline\b",
    r"\bdo\s*not\s*want\b",
    r"\bdoes\s*not\s*want\b",
    r"\bnot\s*looking\b",
]

NO_RESPONSE_PATTERNS = [
    r"\bno\s*response\b",
    r"\bno\s*reply\b",
    r"\bnot\s*responded\b",
    r"\bnot\s*responding\b",
    r"\bdid\s*not\s*respond\b",
    r"\bdoes\s*not\s*respond\b",
    r"\bnot\s*replied\b",
    r"\bdid\s*not\s*reply\b",
    r"\bcalled\s*(but\s*)?not\s*responded\b",
    r"\bcalled\s*(but\s*)?no\s*response\b",
    r"\bcall(ed)?\s*(but\s*)?no\s*reply\b",
    r"\bnot\s*answering\b",
    r"\bnot\s*answer\b",
    r"\bno\s*answer\b",
]

CONTACT_ISSUE_PATTERNS = [
    r"\bnot\s*picking\b",
    r"\bnot\s*picking\s*(the\s*)?(call|phone)\b",
    r"\bnumber\s*(not\s*)?(working|available|reachable)\b",
    r"\bnumber\s*issue\b",
    r"\bswitched\s*off\b",
    r"\bphone\s*off\b",
    r"\bcontact\s*not\s*available\b",
    r"\bunreachable\b",
]

CANCELLED_PATTERNS = [
    r"\bcancelled\b",
    r"\bcanceled\b",
    r"\bcancel\b",
    r"\bplan\s*cancelled\b",
    r"\bplan\s*canceled\b",
    r"\bvisit\s*cancelled\b",
    r"\bvisit\s*canceled\b",
    r"\bmeeting\s*cancelled\b",
    r"\bmeeting\s*canceled\b",
]

POSTPONED_PATTERNS = [
    r"\bpostponed\b",
    r"\bpostpone\b",
    r"\bpostponing\b",
    r"\bvisit\s*postponed\b",
    r"\bmeeting\s*postponed\b",
    r"\bplan\s*postponed\b",
]

BUDGET_NEGATIVE_PATTERNS = [
    r"\bnot\s*afford\b",
    r"\bnot\s*affordable\b",
    r"\bcan'?t\s*afford\b",
    r"\bcannot\s*afford\b",
    r"\bbeyond\s*budget\b",
    r"\boutside\s*budget\b",
    r"\bno\s*budget\b",
    r"\bbudget\s*issue\b",
    r"\bbudget\s*problem\b",
]

DELAY_PATTERNS = [
    r"\bwill\s*contact\s*(us\s*)?(later|himself|herself)\b",
    r"\bcontact\s*(us\s*)?later\b",
    r"\bwill\s*revert\s*later\b",
    r"\bwill\s*revert\b",
    r"\bcall\s*later\b",
    r"\bget\s*back\s*(to\s*us)?\s*later\b",
    r"\bwhen\s*required\b",
    r"\bwhen\s*needed\b",
]

LAST_ATTEMPT_PATTERNS = [
    r"\blast\s*attempt\b",
    r"\bfinal\s*attempt\b",
    r"\blast\s*call\b",
    r"\bfinal\s*call\b",
]

SCORE_MAX_LEAD_QUALITY = 20.0
SCORE_MAX_ENGAGEMENT = 25.0
SCORE_MAX_BUYING_INTENT = 35.0
SCORE_MAX_NEGATIVE_RISK = 15.0
SCORE_MAX_RECENCY = 5.0

def load_json(path: Path):
    """Load JSON safely."""
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        return None

    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        print(f"[ERROR] Could not read {path}: {exc}")
        return None

    if not text:
        print(f"[ERROR] File is empty: {path}")
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {path}: {exc}")
        return None


def safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    if value is None:
        return default

    if isinstance(value, str):
        value = value.strip()

        if value.lower() in {"", "null", "none", "nan"}:
            return default

        # Remove common comma formatting.
        value = value.replace(",", "")

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_text(value: Any) -> str:
    """Normalize arbitrary values into lowercase searchable text."""
    if value is None:
        return ""

    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        text = " ".join(
            str(v)
            for v in value.values()
            if v is not None
        )
    elif isinstance(value, list):
        text = " ".join(str(v) for v in value)
    else:
        text = str(value)

    text = text.strip()

    if text.lower() in {"null", "none", "nan"}:
        return ""

    return text.lower()


def normalize_value(value: Any) -> str:
    """Normalize categorical API values."""
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"null", "none", "nan"}:
        return ""

    return text


def enc(mapping: dict, value: Any) -> int:
    """Safe categorical encoding."""
    return mapping.get(normalize_value(value), 0)


def get_first_nonempty(
    obj: dict,
    fields: List[str],
    default: Any = None,
):
    """Return first non-empty field."""
    if not isinstance(obj, dict):
        return default

    for field in fields:
        value = obj.get(field)

        if value is None:
            continue

        if str(value).strip().lower() in {"", "null", "none"}:
            continue

        return value

    return default


def parse_date(value: Any) -> Optional[datetime]:
    """Parse supported CRM date formats."""
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {"null", "none"}:
        return None

    try:
        parsed = pd.to_datetime(text, errors="coerce")

        if not pd.isna(parsed):
            return parsed.to_pydatetime()
    except Exception:
        pass

    formats = (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    )

    for fmt in formats:
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue

    return None


def days_since(value: Any) -> float:
    """Days from date to now."""
    dt = parse_date(value)

    if dt is None:
        return -1.0

    return float(max((datetime.now() - dt).days, 0))


def clip(value: float, low: float, high: float) -> float:
    """Safe numeric clipping."""
    return float(np.clip(value, low, high))

def get_meeting_type(record: dict) -> str:
    value = get_first_nonempty(
        record,
        MEETING_TYPE_FIELDS,
        "",
    )

    return normalize_value(value)


def get_followup_date(record: dict) -> Optional[datetime]:
    for field in DATE_FIELDS:
        value = record.get(field)

        if value is None:
            continue

        dt = parse_date(value)

        if dt:
            return dt

    return None


def get_stage(record: dict) -> str:
    value = get_first_nonempty(
        record,
        STAGE_FIELDS,
        "",
    )

    return normalize_value(value)


def get_interest(record: dict) -> str:
    value = get_first_nonempty(
        record,
        INTEREST_FIELDS,
        "",
    )

    return normalize_value(value)


def get_budget_pair(record: dict) -> Tuple[float, float]:
    b_from = safe_float(record.get("budget_from"))
    b_to = safe_float(record.get("budget_to"))

    detail = record.get("MeetingDetail")

    if isinstance(detail, dict):
        if b_from <= 0:
            b_from = safe_float(detail.get("budget_from"))

        if b_to <= 0:
            b_to = safe_float(detail.get("budget_to"))

    return max(b_from, 0.0), max(b_to, 0.0)


def get_lead_id(lead: dict) -> str:
    """Resolve LeadNo / Id safely."""
    value = lead.get("LeadNo")

    if value in (None, "", "null", "None"):
        value = lead.get("Id")

    if value in (None, "", "null", "None"):
        return "0"

    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()

def extract_all_narration_text(record: dict) -> str:
    """
    Concatenate all narration-related fields.

    Nested dictionaries are flattened into their string values.
    """
    if not isinstance(record, dict):
        return ""

    parts = []

    for field in NARRATION_FIELDS:
        value = record.get(field)

        if value is None:
            continue

        if isinstance(value, str):
            text = value.strip()

            if text and text.lower() not in {"null", "none"}:
                parts.append(text)

        elif isinstance(value, dict):
            for nested_value in value.values():
                if isinstance(nested_value, str):
                    text = nested_value.strip()

                    if text:
                        parts.append(text)

        elif isinstance(value, list):
            for item in value:
                if item:
                    parts.append(str(item))

    return " ".join(parts).strip().lower()

def contains_any(text: str, patterns: List[str]) -> bool:
    """Return True when at least one regex pattern matches."""
    if not text:
        return False

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def count_matches(text: str, patterns: List[str]) -> int:
    """
    Count distinct matched patterns.

    This is diagnostic only; it should not automatically be interpreted
    as multiple independent buying events.
    """
    if not text:
        return 0

    return sum(
        1
        for pattern in patterns
        if re.search(pattern, text, re.IGNORECASE)
    )


def classify_narration(text: str) -> Dict[str, int]:
    text = normalize_text(text)

    if not text:
        return {
            "token": 0,
            "payment": 0,
            "booking": 0,
            "conversion": 0,
            "confirmation": 0,
            "property_selected": 0,
            "site_visit": 0,
            "strong_interest": 0,
            "moderate_interest": 0,
            "positive_quality": 0,
            "not_interested": 0,
            "no_response": 0,
            "contact_issue": 0,
            "cancelled": 0,
            "postponed": 0,
            "budget_negative": 0,
            "delay": 0,
            "last_attempt": 0,
        }

    return {
        "token": int(contains_any(text, TOKEN_PATTERNS)),
        "payment": int(contains_any(text, PAYMENT_PATTERNS)),
        "booking": int(contains_any(text, BOOKING_PATTERNS)),
        "conversion": int(contains_any(text, CONVERSION_PATTERNS)),
        "confirmation": int(contains_any(text, CONFIRMATION_PATTERNS)),
        "property_selected": int(
            contains_any(text, PROPERTY_SELECTED_PATTERNS)
        ),
        "site_visit": int(
            contains_any(text, SITE_VISIT_PATTERNS)
        ),
        "strong_interest": int(
            contains_any(text, STRONG_INTEREST_PATTERNS)
        ),
        "moderate_interest": int(
            contains_any(text, MODERATE_INTEREST_PATTERNS)
        ),
        "positive_quality": int(
            contains_any(text, POSITIVE_QUALITY_PATTERNS)
        ),
        "not_interested": int(
            contains_any(text, NOT_INTERESTED_PATTERNS)
        ),
        "no_response": int(
            contains_any(text, NO_RESPONSE_PATTERNS)
        ),
        "contact_issue": int(
            contains_any(text, CONTACT_ISSUE_PATTERNS)
        ),
        "cancelled": int(
            contains_any(text, CANCELLED_PATTERNS)
        ),
        "postponed": int(
            contains_any(text, POSTPONED_PATTERNS)
        ),
        "budget_negative": int(
            contains_any(text, BUDGET_NEGATIVE_PATTERNS)
        ),
        "delay": int(
            contains_any(text, DELAY_PATTERNS)
        ),
        "last_attempt": int(
            contains_any(text, LAST_ATTEMPT_PATTERNS)
        ),
    }


def score_single_narration(text: str) -> float:
    """
    Diagnostic NLP score for ONE narration.

    This is intentionally capped and is NOT directly added multiple times
    to the final lead score.
    """
    signals = classify_narration(text)

    positive = 0.0
    negative = 0.0

    # Positive hierarchy.
    if signals["token"]:
        positive += 40
    elif signals["payment"]:
        positive += 34
    elif signals["booking"]:
        positive += 38
    elif signals["conversion"]:
        positive += 35
    elif signals["confirmation"]:
        positive += 25
    elif signals["property_selected"]:
        positive += 20
    elif signals["site_visit"]:
        positive += 15
    elif signals["strong_interest"]:
        positive += 12
    elif signals["moderate_interest"]:
        positive += 6
    elif signals["positive_quality"]:
        positive += 4

    # Negative hierarchy.
    if signals["not_interested"]:
        negative += 30

    if signals["cancelled"]:
        negative += 22

    if signals["no_response"]:
        negative += 12

    if signals["contact_issue"]:
        negative += 10

    if signals["budget_negative"]:
        negative += 12

    if signals["postponed"]:
        negative += 7

    if signals["delay"]:
        negative += 6

    if signals["last_attempt"]:
        negative += 5

    return round(
        clip(positive - negative, -40, 40),
        2,
    )

def get_followup_narration_features(
    followups: List[dict],
) -> dict:
    empty = {
        "narr_text_count": 0,
        "narr_positive_count": 0,
        "narr_negative_count": 0,

        "narr_token": 0,
        "narr_payment": 0,
        "narr_booking": 0,
        "narr_conversion": 0,
        "narr_confirmation": 0,
        "narr_property_selected": 0,
        "narr_site_visit": 0,
        "narr_strong_interest": 0,
        "narr_moderate_interest": 0,
        "narr_positive_quality": 0,

        "narr_not_interested": 0,
        "narr_no_response": 0,
        "narr_contact_issue": 0,
        "narr_cancelled": 0,
        "narr_postponed": 0,
        "narr_budget_negative": 0,
        "narr_delay": 0,
        "narr_last_attempt": 0,

        "narr_latest_signal_score": 0.0,
        "narr_best_signal_score": 0.0,
        "narr_total_signal_score": 0.0,
    }

    if not followups:
        return empty

    texts = []

    for fup in followups:
        text = extract_all_narration_text(fup)

        if text:
            texts.append(text)

    if not texts:
        return empty

    signal_rows = [
        classify_narration(text)
        for text in texts
    ]

    scores = [
        score_single_narration(text)
        for text in texts
    ]

    result = {
        "narr_text_count": len(texts),
        "narr_positive_count": sum(1 for x in scores if x > 0),
        "narr_negative_count": sum(1 for x in scores if x < 0),

        "narr_token": int(any(x["token"] for x in signal_rows)),
        "narr_payment": int(any(x["payment"] for x in signal_rows)),
        "narr_booking": int(any(x["booking"] for x in signal_rows)),
        "narr_conversion": int(any(x["conversion"] for x in signal_rows)),
        "narr_confirmation": int(any(x["confirmation"] for x in signal_rows)),
        "narr_property_selected": int(
            any(x["property_selected"] for x in signal_rows)
        ),
        "narr_site_visit": int(
            any(x["site_visit"] for x in signal_rows)
        ),
        "narr_strong_interest": int(
            any(x["strong_interest"] for x in signal_rows)
        ),
        "narr_moderate_interest": int(
            any(x["moderate_interest"] for x in signal_rows)
        ),
        "narr_positive_quality": int(
            any(x["positive_quality"] for x in signal_rows)
        ),

        "narr_not_interested": int(
            any(x["not_interested"] for x in signal_rows)
        ),
        "narr_no_response": int(
            any(x["no_response"] for x in signal_rows)
        ),
        "narr_contact_issue": int(
            any(x["contact_issue"] for x in signal_rows)
        ),
        "narr_cancelled": int(
            any(x["cancelled"] for x in signal_rows)
        ),
        "narr_postponed": int(
            any(x["postponed"] for x in signal_rows)
        ),
        "narr_budget_negative": int(
            any(x["budget_negative"] for x in signal_rows)
        ),
        "narr_delay": int(
            any(x["delay"] for x in signal_rows)
        ),
        "narr_last_attempt": int(
            any(x["last_attempt"] for x in signal_rows)
        ),

        # These are diagnostic features.
        # The final score DOES NOT add all three.
        "narr_latest_signal_score": round(scores[-1], 2),
        "narr_best_signal_score": round(max(scores), 2),
        "narr_total_signal_score": round(
            sum(scores),
            2,
        ),
    }

    return result

def normalized_status(value: Any) -> str:
    return normalize_text(value)


def is_closed_value(value: Any) -> bool:
    return normalized_status(value) in CLOSED_VALUES


def is_cancelled_value(value: Any) -> bool:
    return normalized_status(value) in CANCELLED_VALUES


def is_dead_value(value: Any) -> bool:
    return normalized_status(value) in DEAD_VALUES


def classify_lead_status(
    lead: dict,
    followups: List[dict],
    narration_features: Optional[dict] = None,
) -> dict:

    if narration_features is None:
        narration_features = get_followup_narration_features(followups)

    next_meeting_type = normalized_status(
        lead.get("NextMeetingType")
    )

    lead_status = normalized_status(
        lead.get("LeadStatus")
    )

    lead_stage = normalized_status(
        lead.get("stage_name")
        or lead.get("lead_stages")
    )

    meeting_types = [
        normalized_status(get_meeting_type(f))
        for f in followups
    ]

    closed = (
        next_meeting_type in CLOSED_VALUES
        or lead_status in CLOSED_VALUES
        or lead_stage in CLOSED_VALUES
        or any(mt in CLOSED_VALUES for mt in meeting_types)
    )

    cancelled = (
        is_cancelled_value(lead.get("NextMeetingType"))
        or is_cancelled_value(lead.get("LeadStatus"))
        or is_cancelled_value(lead.get("stage_name"))
        or bool(narration_features["narr_cancelled"])
    )

    dead = (
        is_dead_value(lead.get("LeadStatus"))
        or bool(narration_features["narr_not_interested"])
    )

    terminal_positive = bool(
        narration_features["narr_token"]
        or narration_features["narr_payment"]
        or narration_features["narr_booking"]
        or narration_features["narr_conversion"]
    )

    terminal_negative = bool(
        cancelled
        or dead
    )
    closed_lost = bool(
        closed
        and not terminal_positive
    )

    return {
        "status_is_closed": int(closed),
        "status_is_cancelled": int(cancelled),
        "status_is_dead": int(dead),
        "status_terminal_positive": int(terminal_positive),
        "status_terminal_negative": int(terminal_negative),
        "status_closed_lost": int(closed_lost),
        "status_is_active": int(
            not closed
            and not cancelled
            and not dead
        ),
    }
def lead_features(lead: dict) -> dict:
    """Build XGBoost-ready numeric features from lead record."""

    next_dt = parse_date(
        lead.get("NextMeetingDate")
    )

    if next_dt:
        days_to_next = max(
            (next_dt - datetime.now()).days,
            0,
        )
    else:
        days_to_next = -1

    b_from, b_to = get_budget_pair(lead)

    budget_range = max(
        b_to - b_from,
        0.0,
    )

    interest_raw = normalize_value(
        lead.get("Potential")
        or lead.get("InterestLevel")
        or ""
    )

    stage_raw = normalize_value(
        lead.get("stage_name")
        or lead.get("lead_stages")
        or ""
    )

    lead_narration = extract_all_narration_text(
        lead
    )

    lead_narr_score = score_single_narration(
        lead_narration
    )

    lead_nlp = classify_narration(
        lead_narration
    )

    interest_nature = normalize_text(
        lead.get("InterestInNature")
    )

    lead_status = normalize_text(
        lead.get("LeadStatus")
    )

    next_meeting_type = normalize_text(
        lead.get("NextMeetingType")
    )

    return {
        "interest_level_enc": enc(
            INTEREST_MAP,
            interest_raw,
        ),

        "lead_stage_enc": enc(
            STAGE_MAP,
            stage_raw,
        ),

        "next_meeting_type_enc": enc(
            MEETING_TYPE_MAP,
            lead.get("NextMeetingType"),
        ),

        "source_enc": enc(
            SOURCE_MAP,
            lead.get("sourceCode"),
        ),

        "purpose_enc": enc(
            PURPOSE_MAP,
            lead.get("purpose"),
        ),
        "budget_from": b_from,

        "budget_to": b_to,

        "budget_range": budget_range,

        "has_budget": int(
            b_to > 0
        ),

        "budget_log": round(
            math.log1p(b_to),
            4,
        ),
        "has_plot_category": int(
            bool(normalize_value(
                lead.get("plot_category")
            ))
        ),

        "has_mode_of_payment": int(
            bool(normalize_value(
                lead.get("modeOfPayment")
            ))
        ),

        "has_plan_to_buy": int(
            bool(normalize_value(
                lead.get("planToBuy")
            ))
        ),

        "has_meeting_date": int(
            bool(parse_date(
                lead.get("MeetingDate")
            ))
        ),

        "has_next_meeting": int(
            bool(parse_date(
                lead.get("NextMeetingDate")
            ))
        ),
        "days_since_assign": days_since(
            lead.get("AssignDate")
        ),

        "days_since_posting": days_since(
            lead.get("PostingDate")
        ),

        "days_to_next_meeting": float(
            days_to_next
        ),
        "interest_residential": int(
            interest_nature == "residential"
        ),

        "interest_commercial": int(
            interest_nature == "commercial"
        ),
        "lead_status_active_raw": int(
            lead_status == "a"
        ),
        "lead_narr_score": round(
            lead_narr_score,
            2,
        ),

        "lead_narr_token": lead_nlp["token"],

        "lead_narr_payment": lead_nlp["payment"],

        "lead_narr_booking": lead_nlp["booking"],

        "lead_narr_conversion": lead_nlp["conversion"],

        "lead_narr_property_selected": lead_nlp[
            "property_selected"
        ],

        "lead_narr_site_visit": lead_nlp[
            "site_visit"
        ],

        "lead_narr_strong_interest": lead_nlp[
            "strong_interest"
        ],

        "lead_narr_not_interested": lead_nlp[
            "not_interested"
        ],

        "lead_narr_no_response": lead_nlp[
            "no_response"
        ],

        "lead_narr_cancelled": lead_nlp[
            "cancelled"
        ],

        "lead_narr_budget_negative": lead_nlp[
            "budget_negative"
        ],

        "lead_narr_delay": lead_nlp[
            "delay"
        ],
        "lead_is_closed_raw": int(
            next_meeting_type in CLOSED_VALUES
        ),

        "lead_is_cancelled_raw": int(
            next_meeting_type in CANCELLED_VALUES
        ),

        "lead_is_dead_raw": int(
            lead_status in DEAD_VALUES
            or stage_raw.lower() in DEAD_VALUES
        ),
    }
def followup_features(
    followups: List[dict],
) -> dict:
    empty = {
        "fup_count": 0,

        "fup_positive_meeting_count": 0,
        "fup_scheduled_count": 0,
        "fup_call_count": 0,
        "fup_postponed_count": 0,
        "fup_attempt_call_count": 0,
        "fup_walk_in_count": 0,
        "fup_visit_done_count": 0,
        "fup_meeting_done_count": 0,

        "fup_avg_days_gap": -1.0,
        "fup_total_span_days": 0.0,
        "fup_days_since_last": -1.0,

        "fup_max_interest": 0,
        "fup_last_interest": 0,
        "fup_interest_trend": 0,

        "fup_has_budget": 0,
        "fup_max_budget_log": 0.0,
        "fup_avg_budget_log": 0.0,
        "fup_budget_trend": 0,

        "fup_has_narration": 0,

        "fup_last_stage": 0,
        "fup_max_stage": 0,
        "fup_stage_upgraded": 0,

        "fup_max_plot_size": 0.0,

        "fup_pct_qualified": 0.0,

        "fup_last_meeting_type_enc": 0,
        "fup_max_meeting_type_enc": 0,

        "fup_connected_call_rate": 0.0,
        "fup_positive_meeting_rate": 0.0,
        "fup_postponed_rate": 0.0,
        "fup_attempt_rate": 0.0,

        "fup_quality_ratio": 0.0,
    }

    if not followups:
        return empty

    n = len(followups)
    meeting_types = [
        get_meeting_type(f)
        for f in followups
    ]

    positive_count = sum(
        1
        for mt in meeting_types
        if mt in POSITIVE_MEETINGS
    )

    scheduled_count = sum(
        1
        for mt in meeting_types
        if mt in SCHEDULED_MEETINGS
    )

    call_count = sum(
        1
        for mt in meeting_types
        if mt in CALL_TYPES
    )

    postponed_count = sum(
        1
        for mt in meeting_types
        if mt in POSTPONED_TYPES
    )

    attempt_count = sum(
        1
        for mt in meeting_types
        if mt in ATTEMPT_TYPES
    )

    walk_in_count = sum(
        1
        for mt in meeting_types
        if mt == "Walk In Client"
    )

    visit_done = sum(
        1
        for mt in meeting_types
        if mt in VISIT_DONE_TYPES
    )

    meeting_done = sum(
        1
        for mt in meeting_types
        if mt in MEETING_DONE_TYPES
    )

    type_encs = [
        enc(
            MEETING_TYPE_MAP,
            mt,
        )
        for mt in meeting_types
    ]
    dates = sorted(
        d
        for d in (
            get_followup_date(f)
            for f in followups
        )
        if d is not None
    )

    if len(dates) > 1:
        gaps = [
            (dates[i] - dates[i - 1]).days
            for i in range(1, len(dates))
        ]

        avg_gap = float(
            np.mean(gaps)
        )

        total_span = float(
            (dates[-1] - dates[0]).days
        )
    else:
        avg_gap = -1.0
        total_span = 0.0

    if dates:
        days_last = float(
            max(
                (datetime.now() - dates[-1]).days,
                0,
            )
        )
    else:
        days_last = -1.0
    interest_vals = [
        enc(
            INTEREST_MAP,
            get_interest(f),
        )
        for f in followups
    ]

    if interest_vals:
        max_interest = max(
            interest_vals
        )

        last_interest = interest_vals[-1]
    else:
        max_interest = 0
        last_interest = 0

    if len(interest_vals) >= 2:
        interest_trend = (
            int(
                interest_vals[-1]
                > interest_vals[0]
            )
            -
            int(
                interest_vals[-1]
                < interest_vals[0]
            )
        )
    else:
        interest_trend = 0
    stage_vals = [
        enc(
            STAGE_MAP,
            get_stage(f),
        )
        for f in followups
    ]

    if stage_vals:
        last_stage = stage_vals[-1]
        max_stage = max(stage_vals)
    else:
        last_stage = 0
        max_stage = 0

    stage_upgraded = int(
        len(stage_vals) >= 2
        and max(stage_vals[1:]) > stage_vals[0]
    )

    qualified_count = sum(
        1
        for f in followups
        if get_stage(f)
        in {
            "Qualified",
            "FollowUp",
            "Booking",
            "Token Received",
        }
    )
    budgets = []

    for f in followups:
        _, b_to = get_budget_pair(f)

        if b_to > 0:
            budgets.append(b_to)

    if budgets:
        has_budget = 1
        max_budget_log = math.log1p(
            max(budgets)
        )
        avg_budget_log = math.log1p(
            float(np.mean(budgets))
        )
    else:
        has_budget = 0
        max_budget_log = 0.0
        avg_budget_log = 0.0

    if len(budgets) >= 2:
        budget_trend = (
            int(
                budgets[-1]
                > budgets[0]
            )
            -
            int(
                budgets[-1]
                < budgets[0]
            )
        )
    else:
        budget_trend = 0
    plot_sizes = [
        safe_float(
            f.get("plot_size")
        )
        for f in followups
        if safe_float(
            f.get("plot_size")
        ) > 0
    ]

    max_plot_size = (
        max(plot_sizes)
        if plot_sizes
        else 0.0
    )
    has_narration = int(
        any(
            extract_all_narration_text(f)
            for f in followups
        )
    )
    positive_meeting_rate = (
        positive_count / n
    )

    postponed_rate = (
        postponed_count / n
    )

    attempt_rate = (
        attempt_count / n
    )

    connected_calls = sum(
        1
        for mt in meeting_types
        if mt == "Call Connected"
    )

    connected_call_rate = (
        connected_calls / call_count
        if call_count > 0
        else 0.0
    )

    quality_ratio = (
        positive_count
        + scheduled_count * 0.5
        + connected_calls * 0.25
    ) / max(n, 1)

    return {
        "fup_count": n,

        "fup_positive_meeting_count": positive_count,
        "fup_scheduled_count": scheduled_count,
        "fup_call_count": call_count,
        "fup_postponed_count": postponed_count,
        "fup_attempt_call_count": attempt_count,
        "fup_walk_in_count": walk_in_count,
        "fup_visit_done_count": visit_done,
        "fup_meeting_done_count": meeting_done,

        "fup_avg_days_gap": round(
            avg_gap,
            2,
        ),

        "fup_total_span_days": round(
            total_span,
            2,
        ),

        "fup_days_since_last": round(
            days_last,
            2,
        ),

        "fup_max_interest": max_interest,
        "fup_last_interest": last_interest,
        "fup_interest_trend": interest_trend,

        "fup_has_budget": has_budget,

        "fup_max_budget_log": round(
            max_budget_log,
            4,
        ),

        "fup_avg_budget_log": round(
            avg_budget_log,
            4,
        ),

        "fup_budget_trend": budget_trend,

        "fup_has_narration": has_narration,

        "fup_last_stage": last_stage,
        "fup_max_stage": max_stage,
        "fup_stage_upgraded": stage_upgraded,

        "fup_max_plot_size": max_plot_size,

        "fup_pct_qualified": round(
            qualified_count / n,
            4,
        ),

        "fup_last_meeting_type_enc": (
            type_encs[-1]
            if type_encs
            else 0
        ),

        "fup_max_meeting_type_enc": (
            max(type_encs)
            if type_encs
            else 0
        ),

        "fup_connected_call_rate": round(
            connected_call_rate,
            4,
        ),

        "fup_positive_meeting_rate": round(
            positive_meeting_rate,
            4,
        ),

        "fup_postponed_rate": round(
            postponed_rate,
            4,
        ),

        "fup_attempt_rate": round(
            attempt_rate,
            4,
        ),

        "fup_quality_ratio": round(
            quality_ratio,
            4,
        ),
    }
def calculate_lead_quality_score(
    lead: dict,
    lf: dict,
) -> float:

    score = 0.0

    interest = lf["interest_level_enc"]

    # 0 / 1 / 2 / 3 -> 0 / 4 / 8 / 12
    score += {
        0: 0,
        1: 4,
        2: 8,
        3: 12,
    }.get(
        interest,
        0,
    )

    # Budget availability.
    if lf["has_budget"]:
        score += 5

    # Small bonus for an actual plan-to-buy field.
    if lf["has_plan_to_buy"]:
        score += 2

    # Meeting information is useful but small.
    if lf["has_meeting_date"]:
        score += 1

    return clip(
        score,
        0,
        SCORE_MAX_LEAD_QUALITY,
    )


def calculate_engagement_score(
    ff: dict,
) -> float:
    score = 0.0

    # Actual completed interaction.
    score += min(
        ff["fup_visit_done_count"] * 5.0,
        10.0,
    )

    score += min(
        ff["fup_meeting_done_count"] * 5.0,
        8.0,
    )

    score += min(
        ff["fup_walk_in_count"] * 4.0,
        6.0,
    )

    # Scheduled meeting is positive but weaker than completed meeting.
    score += min(
        ff["fup_scheduled_count"] * 1.5,
        4.0,
    )

    # Connected calls show actual contact.
    score += min(
        ff["fup_connected_call_rate"] * 3.0,
        3.0,
    )

    # Penalties.
    score -= min(
        ff["fup_postponed_count"] * 1.5,
        6.0,
    )

    score -= min(
        ff["fup_attempt_call_count"] * 0.75,
        5.0,
    )

    return clip(
        score,
        0,
        SCORE_MAX_ENGAGEMENT,
    )


def calculate_buying_intent_score(
    lead: dict,
    ff: dict,
    nf: dict,
) -> float:
    score = 0.0
    if nf["narr_token"]:
        return 35.0

    if nf["narr_booking"]:
        return 34.0

    if nf["narr_payment"]:
        return 32.0

    if nf["narr_conversion"]:
        return 30.0
    if nf["narr_property_selected"]:
        score += 20.0

    elif nf["narr_site_visit"]:
        score += 14.0

    elif nf["narr_strong_interest"]:
        score += 12.0

    elif nf["narr_moderate_interest"]:
        score += 7.0

    elif nf["narr_positive_quality"]:
        score += 4.0

    stage = ff["fup_max_stage"]

    score += {
        0: 0.0,
        1: 1.0,
        2: 2.0,
        3: 4.0,
        4: 5.0,
        5: 7.0,
        6: 9.0,
        8: 12.0,
    }.get(
        stage,
        0.0,
    )

    if ff["fup_last_interest"] >= 3:
        score += 3.0

    elif ff["fup_last_interest"] == 2:
        score += 2.0

    if ff["fup_interest_trend"] > 0:
        score += 2.0

    if ff["fup_has_budget"]:
        score += 2.0

    return clip(
        score,
        0,
        SCORE_MAX_BUYING_INTENT,
    )


def calculate_negative_risk_score(
    lead: dict,
    nf: dict,
    status: dict,
) -> float:
    penalty = 0.0
    if nf["narr_not_interested"]:
        penalty += 15.0
    if nf["narr_cancelled"]:
        penalty += 12.0

    # Explicit CRM cancelled status.
    if status["status_is_cancelled"]:
        penalty += 12.0

    if nf["narr_no_response"]:
        penalty += 7.0

    if nf["narr_contact_issue"]:
        penalty += 5.0
    if nf["narr_budget_negative"]:
        penalty += 6.0

    if nf["narr_postponed"]:
        penalty += 4.0

    if nf["narr_delay"]:
        penalty += 4.0
    if status["status_closed_lost"]:
        penalty += 15.0

    # Explicit dead lead.
    if status["status_is_dead"]:
        penalty += 15.0

    return clip(
        penalty,
        0,
        SCORE_MAX_NEGATIVE_RISK,
    )


def calculate_recency_score(
    ff: dict,
) -> float:
    days_last = ff["fup_days_since_last"]

    if days_last < 0:
        return 0.0

    if days_last <= 3:
        return 5.0

    if days_last <= 7:
        return 4.0

    if days_last <= 14:
        return 3.0

    if days_last <= 30:
        return 2.0

    if days_last <= 60:
        return 1.0

    return 0.0
def calculate_lead_score(
    lead: dict,
    followups: List[dict],
) -> float:
    lf = lead_features(
        lead
    )

    ff = followup_features(
        followups
    )

    nf = get_followup_narration_features(
        followups
    )

    status = classify_lead_status(
        lead,
        followups,
        nf,
    )
    if status["status_terminal_positive"]:
        if nf["narr_token"]:
            return 98.0

        if nf["narr_booking"]:
            return 96.0

        if nf["narr_payment"]:
            return 94.0

        if nf["narr_conversion"]:
            return 92.0
    if (
        status["status_is_cancelled"]
        or status["status_is_dead"]
    ):
        return 5.0
    if status["status_closed_lost"]:
        return 5.0

    lead_quality = calculate_lead_quality_score(
        lead,
        lf,
    )

    engagement = calculate_engagement_score(
        ff,
    )

    buying_intent = calculate_buying_intent_score(
        lead,
        ff,
        nf,
    )

    negative_penalty = calculate_negative_risk_score(
        lead,
        nf,
        status,
    )

    recency = calculate_recency_score(
        ff,
    )

    raw_score = (
        lead_quality
        + engagement
        + buying_intent
        + recency
        - negative_penalty
    )
    score = (
        raw_score / 85.0
    ) * 100.0

    if nf["narr_not_interested"]:
        score = min(
            score,
            15.0,
        )

    elif nf["narr_cancelled"]:
        score = min(
            score,
            18.0,
        )

    elif status["status_closed_lost"]:
        score = min(
            score,
            15.0,
        )

    elif nf["narr_no_response"]:
        score = min(
            score,
            45.0,
        )

    return round(
        clip(score, 0, 100),
        1,
    )

def score_breakdown(
    lead: dict,
    followups: List[dict],
) -> dict:
    lf = lead_features(
        lead
    )

    ff = followup_features(
        followups
    )

    nf = get_followup_narration_features(
        followups
    )

    status = classify_lead_status(
        lead,
        followups,
        nf,
    )

    lead_quality = calculate_lead_quality_score(
        lead,
        lf,
    )

    engagement = calculate_engagement_score(
        ff,
    )

    buying_intent = calculate_buying_intent_score(
        lead,
        ff,
        nf,
    )

    negative_risk = calculate_negative_risk_score(
        lead,
        nf,
        status,
    )

    recency = calculate_recency_score(
        ff,
    )

    return {
        "lead_quality": round(
            lead_quality,
            2,
        ),

        "engagement": round(
            engagement,
            2,
        ),

        "buying_intent": round(
            buying_intent,
            2,
        ),

        "negative_risk": round(
            negative_risk,
            2,
        ),

        "recency": round(
            recency,
            2,
        ),

        "status_is_closed": status[
            "status_is_closed"
        ],

        "status_is_cancelled": status[
            "status_is_cancelled"
        ],

        "status_is_dead": status[
            "status_is_dead"
        ],

        "status_terminal_positive": status[
            "status_terminal_positive"
        ],

        "status_closed_lost": status[
            "status_closed_lost"
        ],

        "final_score": calculate_lead_score(
            lead,
            followups,
        ),
    }

def diagnose_followup_fields(
    followups: List[dict],
    sample_size: int = 3,
):
    """Print resolved CRM fields for debugging."""

    print("\n" + "-" * 70)
    print("FOLLOWUP FIELD DIAGNOSIS")
    print("-" * 70)

    for i, fup in enumerate(
        followups[:sample_size]
    ):
        budget_from, budget_to = get_budget_pair(
            fup
        )

        text = extract_all_narration_text(
            fup
        )

        signals = classify_narration(
            text
        )

        signal_score = score_single_narration(
            text
        )

        

        active_signals = [
            key
            for key, value in signals.items()
            if value
        ]

    
def score_label(score: float) -> str:
    if score <= 20:
        return "Cold"

    if score <= 40:
        return "Warm"

    if score <= 60:
        return "Interested"

    if score <= 80:
        return "Hot"

    return "Very Hot"


def score_reason(
    lead: dict,
    followups: List[dict],
) -> str:

    nf = get_followup_narration_features(
        followups
    )

    status = classify_lead_status(
        lead,
        followups,
        nf,
    )

    reasons = []

    if nf["narr_token"]:
        reasons.append(
            "token/payment signal"
        )

    elif nf["narr_booking"]:
        reasons.append(
            "booking signal"
        )

    elif nf["narr_payment"]:
        reasons.append(
            "payment signal"
        )

    elif nf["narr_conversion"]:
        reasons.append(
            "conversion signal"
        )

    elif nf["narr_property_selected"]:
        reasons.append(
            "property selected"
        )

    elif nf["narr_site_visit"]:
        reasons.append(
            "site visit"
        )

    elif nf["narr_strong_interest"]:
        reasons.append(
            "strong interest"
        )

    elif nf["narr_moderate_interest"]:
        reasons.append(
            "moderate interest"
        )

    if nf["narr_not_interested"]:
        reasons.append(
            "not interested"
        )

    if nf["narr_no_response"]:
        reasons.append(
            "no response"
        )

    if nf["narr_cancelled"]:
        reasons.append(
            "cancelled"
        )

    if nf["narr_budget_negative"]:
        reasons.append(
            "budget issue"
        )

    if nf["narr_delay"]:
        reasons.append(
            "delayed contact"
        )

    if status["status_closed_lost"]:
        reasons.append(
            "closed without conversion"
        )

    if not reasons:
        reasons.append(
            "limited buying signals"
        )

    return ", ".join(reasons[:4])

def build_feature_row(
    lead: dict,
    followups: List[dict],
) -> dict:

    lead_id = get_lead_id(
        lead
    )

    row = {
        "lead_id": lead_id
    }
    row.update(
        lead_features(
            lead
        )
    )
    row.update(
        followup_features(
            followups
        )
    )
    row.update(
        get_followup_narration_features(
            followups
        )
    )
    nf = get_followup_narration_features(
        followups
    )

    status = classify_lead_status(
        lead,
        followups,
        nf,
    )

    row.update(
        status
    )
    score = calculate_lead_score(
        lead,
        followups,
    )
    row["score"] = score
    row["score_category"] = score_label(
        score
    )

    return row
def main():
    leads = load_json(
        LEADS_FILE
    )

    followups_by_lead = (
        load_json(
            FOLLOWUPS_FILE
        )
        or {}
    )

    if leads is None:
        return

    if not isinstance(leads, list):
        print(
            "[ERROR] leads_raw.json must contain a list."
        )
        return

    if not isinstance(
        followups_by_lead,
        dict,
    ):
        
        return
    sample_followups = next(
        iter(
            followups_by_lead.values()
        ),
        [],
    )

    if sample_followups:
        diagnose_followup_fields(
            sample_followups,
            sample_size=2,
        )

    lead_ids = {
        get_lead_id(
            lead
        )
        for lead in leads
    }

    fup_keys = {
        str(key)
        for key in followups_by_lead.keys()
    }

    overlap = (
        lead_ids
        & fup_keys
    )
    if not overlap:

       
        return
    rows = []

    skipped = 0

    for lead in leads:

        lead_id = get_lead_id(
            lead
        )

        followups = followups_by_lead.get(
            lead_id,
            [],
        )
        if not isinstance(
            followups,
            list,
        ):
            followups = []
        if not followups:
            skipped += 1
            continue

        row = build_feature_row(
            lead,
            followups,
        )

        rows.append(
            row
        )
    if not rows:
        print(
            "[ERROR] No feature rows generated."
        )
        return
    df = pd.DataFrame(
        rows
    )
    excluded_columns = {
        "lead_id",
        "score",
        "score_category",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]
    for column in feature_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df[feature_columns] = (
        df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )
    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8",
    )
    OUTPUT_FEATURE_COLUMNS.write_text(
        json.dumps(
            feature_columns,
            indent=2,
        ),
        encoding="utf-8",
    )

    bins = [
        0,
        20,
        40,
        60,
        80,
        100,
    ]

    labels = [
        "0-20   Cold",
        "21-40  Warm",
        "41-60  Interested",
        "61-80  Hot",
        "81-100 Very Hot",
    ]

    score_buckets = pd.cut(
        df["score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    print()

    for label, count in (
        score_buckets
        .value_counts()
        .sort_index()
        .items()
    ):

        percentage = (
            count
            / len(df)
            * 100
        )
    status_columns = [
        "status_is_closed",
        "status_is_cancelled",
        "status_is_dead",
        "status_terminal_positive",
        "status_closed_lost",
        "status_is_active",
    ]

    for column in status_columns:

        if column not in df.columns:
            continue

        count = int(
            df[column].sum()
        )

        percentage = (
            count
            / len(df)
            * 100
        )

    nlp_columns = [
        column
        for column in df.columns
        if column.startswith("narr_")
    ]

    for column in nlp_columns:

        if column not in df.columns:
            continue

        unique_values = set(
            df[column].dropna().unique()
        )

        if unique_values.issubset(
            {0, 1}
        ):

            count = int(
                df[column].sum()
            )

            if count > 0:
                print(
                    f"{column:<35} : "
                    f"{count:>5} leads"
                )

    lowest = (
        df[
            [
                "lead_id",
                "score",
                "score_category",
                "status_is_closed",
                "status_is_cancelled",
                "status_closed_lost",
            ]
        ]
        .sort_values(
            "score"
        )
        .head(10)
    )
    highest = (
        df[
            [
                "lead_id",
                "score",
                "score_category",
                "status_terminal_positive",
                "narr_token",
                "narr_payment",
                "narr_booking",
                "narr_conversion",
            ]
        ]
        .sort_values(
            "score",
            ascending=False,
        )
        .head(10)
    )

    for index, feature in enumerate(
        feature_columns,
        start=1,
    ):
        print(
            f"{index:>3}. {feature}"
        )
if __name__ == "__main__":
    main()