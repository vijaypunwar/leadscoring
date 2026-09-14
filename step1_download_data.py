"""
STEP 1 - Download Leads & Followups from SheCRM API
Auth  : HTTP Basic Auth (username + password)
Method: GET
"""

import requests
import json
import time
from pathlib import Path
from requests.auth import HTTPBasicAuth

# ── Config ────────────────────────────────────────────────────────────────────
USERNAME  = ""
PASSWORD  = ""
PAGE_SIZE = 50
MAX_LEADS = 5000

# LEADS_URL     = "https://winwin.shetecrm.com/api/v1/MobileApp/GetMyLeads"
LEADS_URL     = "https://shetec.com.pk/ace_properties/api/v1/MobileApp/GetAllLeads"

# FOLLOWUPS_URL = "https://winwin.shetecrm.com/api/v1/MobileApp/GetFollowUpbyLeadId"
FOLLOWUPS_URL = "https://shetec.com.pk/ace_properties/api/v1/MobileApp/GetAllLeadsFollowups?LeadId"


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

AUTH    = HTTPBasicAuth(USERNAME, PASSWORD)
HEADERS = {"Content-Type": "application/json"}


# ── Helpers ─────────────────────────────────────────────────────────────────

def save_json(path: Path, data):
    """Save JSON with UTF-8 encoding (handles Arabic/special characters)."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def call_api(url: str, params: dict, retries: int = 3):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, auth=AUTH, headers=HEADERS, params=params, timeout=30)
            print(f"HTTP {r.status_code}", end="")

            if r.status_code != 200:
                print(f"  body={r.text[:300]}")
                r.raise_for_status()

            print()
            return r.json()

        except requests.exceptions.HTTPError as e:
            print(f"\n  [attempt {attempt}] HTTP error: {e}")
        except requests.exceptions.ConnectionError as e:
            print(f"\n  [attempt {attempt}] Connection error: {e}")
        except requests.exceptions.Timeout:
            print(f"\n  [attempt {attempt}] Timeout")
        except Exception as e:
            print(f"\n  [attempt {attempt}] Error: {e}")

        if attempt < retries:
            wait = 2 ** attempt
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    return {}


def extract_list(resp) -> list:
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        for key in ("data", "Data", "leads", "Leads", "result", "Result",
                    "records", "Records", "items", "Items"):
            val = resp.get(key)
            if isinstance(val, list):
                return val
    return []


# ── Download leads ────────────────────────────────────────────────────────────

def download_leads() -> list:
    all_leads, page = [], 1

    while len(all_leads) < MAX_LEADS:
        batch = min(PAGE_SIZE, MAX_LEADS - len(all_leads))
        print(f"  Page {page} ({batch} records)... ", end="")

        resp = call_api(LEADS_URL, {"PageNo": page, "PageSize": batch})

        if not resp:
            print("  Empty response — stopping.")
            break

        records = extract_list(resp)

        if not records:
            print(f"\n  No records in response.")
            print(f"  Keys: {list(resp.keys()) if isinstance(resp, dict) else type(resp)}")
            print(f"  Sample: {json.dumps(resp)[:400]}")
            break

        all_leads.extend(records)
        print(f"  Total: {len(all_leads)}")

        if len(records) < batch:
            print("  Last page reached.")
            break

        page += 1
        time.sleep(0.3)

    return all_leads[:MAX_LEADS]


# ── Download followups ────────────────────────────────────────────────────────

def download_followups(client_id, lead_id) -> list:
    """
    GET /GetAllLeadsFollowups?LeadId=YYY
    Matches Dart URL: GetAllLeadsFollowups?LeadId=$leadId
    """
    resp = call_api(FOLLOWUPS_URL, {"clientId": client_id, "LeadId": lead_id})
    return extract_list(resp)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("STEP 1 - Downloading Leads")
    print(f"  URL  : {LEADS_URL}")
    print(f"  User : {USERNAME}")
    print("=" * 60)

    leads = download_leads()

    if not leads:
        print("\n[ERROR] No leads downloaded.")
        return

    # ── Save leads with UTF-8 encoding ────────────────────────────────────────
    save_json(DATA_DIR / "leads_raw.json", leads)
    print(f"\n  {len(leads)} leads saved to data/leads_raw.json")

    # ── Print first lead to identify clientId field name ──────────────────────
    if leads:
        print("\n  Sample lead fields (to verify clientId field name):")
        sample = leads[0]
        for k, v in sample.items():
            print(f"    {k}: {v}")

    # ── Followups ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Downloading Followups...")
    print("=" * 60)

    all_followups = []
    followups_by_lead = {}

    for i, lead in enumerate(leads, 1):
        lead_id = (lead.get("LeadNo") or lead.get("Id")
                   or lead.get("lead_id") or lead.get("LeadId"))

        # clientId from lead — try common field names
        client_id = (lead.get("MemberId") or lead.get("ClientId")
                     or lead.get("clientId") or lead.get("ClientId2")
                     or lead.get("client_id") or "")

        if not lead_id:
            continue

        print(f"  [{i}/{len(leads)}] Lead {lead_id} (client {client_id})... ", end="")
        fups = download_followups(client_id, lead_id)
        print(f"{len(fups)} followups")

        for f in fups:
            f["_lead_id"] = lead_id

        followups_by_lead[str(lead_id)] = fups
        all_followups.extend(fups)
        time.sleep(0.2)

    save_json(DATA_DIR / "followups_raw.json", all_followups)
    save_json(DATA_DIR / "followups_by_lead.json", followups_by_lead)

    print(f"\n  {len(all_followups)} followups saved to data/followups_raw.json")
    print("  Grouped by lead saved    to data/followups_by_lead.json")
    print("\nDone! Run step2_feature_engineering.py next.")


if __name__ == "__main__":
    main()