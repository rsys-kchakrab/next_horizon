
# FILE: processing/data_enhancer.py
# Purpose: Robust post-parse fixes from cleaned text:
# - Extract name/email/phone/location from header
# - Normalize date ranges (map 'till date' etc. -> 'Present'; unify dashes)
# - Derive most recent work experience (title/company/start/end)
# - Mark education entries with future graduation dates as in-progress
from __future__ import annotations
import re, datetime as dt
from typing import Dict, Any, List, Tuple

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\d[\s.-]?){9,13}")
CITY_HINTS = ["Bengaluru","Bangalore","Kolkata","Mumbai","Delhi","Hyderabad","Chennai","Pune","India"]

MONTHS = {
    "jan":1,"january":1,"feb":2,"february":2,"mar":3,"march":3,"apr":4,"april":4,"may":5,"jun":6,"june":6,
    "jul":7,"july":7,"aug":8,"august":8,"sep":9,"sept":9,"september":9,"oct":10,"october":10,"nov":11,"november":11,
    "dec":12,"december":12
}

PRESENT_WORDS = r"(?:present|till\s*date|till\s*now|to\s*date|current|ongoing)"
DASH = r"[–—-]"  # en dash/em dash/hyphen

def _clean_text(text: str) -> str:
    t = text or ""
    # unify dash variants and map 'till date' etc. to 'Present'
    t = re.sub(r"[\u2012\u2013\u2014\u2212]", "-", t)  # various dashes -> hyphen
    t = re.sub(r"(?i)till\s*-?\s*date|till\s*-?\s*now|to\s*-?\s*date|current|ongoing", "Present", t)
    return t

def _infer_name_from_email(email: str) -> str:
    # e.g., "john.doe_87@company.com" -> "John Doe"
    if not email: return ""
    local = email.split("@")[0]
    parts = re.split(r"[._\-\d]+", local)
    parts = [p for p in parts if p]
    if 1 <= len(parts) <= 3:
        return " ".join(w.capitalize() for w in parts)
    return ""

def extract_personal_info(text: str, existing: Dict[str, Any] | None = None) -> Dict[str, str]:
    t = _clean_text(text or "")
    lines = [ln.strip() for ln in t.splitlines()]
    email_m = EMAIL_RE.search(t)
    phone_m = PHONE_RE.search(t)
    email = email_m.group(0) if email_m else ""
    phone = phone_m.group(0) if phone_m else ""
    # name: first non-empty, non-label header; else infer from email
    name = ""
    header_candidates = lines[:8]
    for ln in header_candidates:
        if EMAIL_RE.search(ln) or PHONE_RE.search(ln):
            continue
        low = ln.lower()
        if any(k in low for k in ["resume","curriculum vitae","cv","profile","summary","contact","email","phone"]):
            continue
        if 2 <= len(ln.split()) <= 6 and len(ln) <= 60:
            name = ln.strip(":-|, ")
            break
    if not name:
        name = _infer_name_from_email(email)
    # location
    location = ""
    for ln in header_candidates + lines[8:20]:
        if any(city.lower() in ln.lower() for city in CITY_HINTS):
            location = ln
            break
    return {"name": name, "email": email, "phone": phone, "location": location}

def _parse_month_year(s: str) -> Tuple[int,int] | None:
    s = s.strip()
    m = re.match(r"(?i)([A-Za-z]{3,9})\s+(\d{4})", s)
    if m:
        mo = MONTHS.get(m.group(1).lower())
        yr = int(m.group(2))
        if mo: return (yr, mo)
    m = re.match(r"(\d{2})/(\d{4})", s)
    if m:
        return (int(m.group(2)), int(m.group(1)))
    return None

def _extract_recent_experience_lines(text: str) -> Dict[str, Any]:
    t = _clean_text(text or "")
    lines = [ln.strip() for ln in t.splitlines()]
    # scan lines for a date range like "Jan 2022 - Present" or "05/2021 - 10/2023"
    rng_re = re.compile(r"(?i)((?:[A-Za-z]{3,9}\s+\d{4}|\d{2}/\d{4}))\s*%s\s*((?:Present|[A-Za-z]{3,9}\s+\d{4}|\d{2}/\d{4}))" % DASH)
    for i, ln in enumerate(lines):
        m = rng_re.search(ln)
        if m:
            # role/company likely in preceding two non-empty lines
            prev = [lines[j] for j in range(max(0, i-3), i) if lines[j].strip()]
            role = prev[0] if prev else ""
            company = prev[1] if len(prev) > 1 else ""
            start = m.group(1).strip()
            end = m.group(2).strip()
            if re.match(r"(?i)present", end): end = "Present"
            return {"title": role.strip(",:|- "), "company": company.strip(",:|- "), "start_date": start, "end_date": end or "Present", "responsibilities": ""}
    return {}

def normalize_education_in_progress(struct: Dict[str, Any]) -> Dict[str, Any]:
    sj = struct.copy()
    today = dt.date.today()
    ed = sj.get("education", []) or []
    out = []
    for e in ed:
        gd = (e or {}).get("graduation_date", "")
        in_prog = False
        if gd:
            # try to parse YYYY or MMM YYYY or MM/YYYY
            yr = None; mo = 12
            m1 = re.match(r"^(\d{4})$", gd.strip())
            m2 = re.match(r"(?i)^([A-Za-z]{3,9})\s+(\d{4})$", gd.strip())
            m3 = re.match(r"^(\d{2})/(\d{4})$", gd.strip())
            try:
                if m1:
                    yr = int(m1.group(1))
                elif m2:
                    yr = int(m2.group(2)); mo = MONTHS.get(m2.group(1).lower(), 12)
                elif m3:
                    mo = int(m3.group(1)); yr = int(m3.group(2))
                if yr:
                    dt_val = dt.date(yr, mo, 1)
                    if dt_val > today:
                        in_prog = True
            except Exception:
                pass
        e2 = dict(e or {})
        if in_prog:
            e2["in_progress"] = True
        out.append(e2)
    sj["education"] = out
    return sj

def backfill_from_text(cleaned_text: str, structured_json: Dict[str, Any]) -> Dict[str, Any]:
    sj = (structured_json or {}).copy()
    # 1) personal info
    pi = sj.get("personal_info", {}) or {}
    inferred = extract_personal_info(cleaned_text, pi)
    for k in ["name","email","phone","location"]:
        if not (pi.get(k) or "").strip() and inferred.get(k):
            pi[k] = inferred[k]
    sj["personal_info"] = pi

    # 2) experience (first/most recent)
    wx = sj.get("work_experience", []) or []
    if not wx:
        guess = _extract_recent_experience_lines(cleaned_text)
        if guess:
            wx = [guess]
    else:
        # fill missing start/end if empty
        if (not wx[0].get("start_date")) or (not wx[0].get("end_date")):
            guess = _extract_recent_experience_lines(cleaned_text)
            if guess:
                wx[0]["start_date"] = wx[0].get("start_date") or guess.get("start_date","")
                wx[0]["end_date"] = wx[0].get("end_date") or guess.get("end_date","")
                if re.match(r"(?i)present", wx[0]["end_date"] or ""):
                    wx[0]["end_date"] = "Present"
    sj["work_experience"] = wx

    # 3) current role fallback
    cr = sj.get("current_role", {}) or {}
    if not cr.get("role") and wx:
        cr["role"] = wx[0].get("title","")
    if not cr.get("company") and wx:
        cr["company"] = wx[0].get("company","")
    if not cr.get("end_date") and wx:
        cr["end_date"] = wx[0].get("end_date","")
    sj["current_role"] = cr

    # 4) education in-progress flag
    sj = normalize_education_in_progress(sj)

    return sj
