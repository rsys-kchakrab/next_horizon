
# FILE: utils/data_enhancer.py
# Purpose: Robust post-parse fixes from cleaned text:
# - Extract location from header
# - Normalize date ranges (map 'till date' etc. -> 'Present'; unify dashes)
# - Derive most recent work experience (title/company/start/end)
# - Mark education entries with future graduation dates as in-progress
from __future__ import annotations
import re, datetime as dt
from typing import Dict, Any, List, Tuple

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

def extract_location_info(text: str, existing: str | None = None) -> str:
    """Extract location information from resume text"""
    t = _clean_text(text or "")
    lines = [ln.strip() for ln in t.splitlines()]
    
    # If location already exists, return it
    if existing and existing.strip():
        return existing.strip()
    
    # Look for location patterns in the first few lines
    location_keywords = [
        r'\b(?:based|located|residing|living)\s+(?:in|at)\s+([^,\n]+(?:,\s*[^,\n]+)*)',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z]{2}|[A-Z][a-z]+)(?:,\s*([A-Z][a-z]+))?\b',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*),\s*([A-Z][a-z]+)\b'
    ]
    
    for line in lines[:10]:  # Check first 10 lines
        for pattern in location_keywords:
            match = re.search(pattern, line)
            if match:
                return match.group(0).strip()
    
    return ""

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
    
    # 1) location info
    current_location = sj.get("location", "")
    inferred_location = extract_location_info(cleaned_text, current_location)
    if inferred_location and not current_location.strip():
        sj["location"] = inferred_location

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
