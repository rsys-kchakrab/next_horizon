# FILE: ai/openai_client.py - OpenAI Integration
from __future__ import annotations
from typing import Any, Dict, List
import os
import re
import json
import numpy as np

def _cosine(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a.dot(b) / (na * nb))

def _norm(s: str) -> str:
    """Normalize text string"""
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)

def openai_rank_roles(resume_text: str, role_snippets: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rank job roles using OpenAI embeddings"""
    resume_text = _norm(resume_text)
    if not role_snippets:
        return []
        
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
    texts = [str(s.get("snippet", "")) for s in role_snippets]
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = "text-embedding-3-small"
        emb_resume = client.embeddings.create(model=model, input=resume_text).data[0].embedding
        emb_snips = client.embeddings.create(model=model, input=texts).data
        scores = [_cosine(emb_resume, emb_s.embedding) for emb_s in emb_snips]
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}. Please check your API key and connection.")
    
    ranked = list(zip(role_snippets, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    out = []
    for r, sc in ranked[:max(1, int(top_k))]:
        out.append({
            "role_title": r.get("title", ""),
            "score": float(sc),
            "link": r.get("link", "")
        })
    return out

def openai_rank_jds(resume_text: str, jd_rows: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rank job descriptions using OpenAI embeddings"""
    resume_text = _norm(resume_text)
    if not jd_rows:
        return []
        
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
    texts = [str(x.get("jd_text", "")) for x in jd_rows]
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = "text-embedding-3-small"
        emb_resume = client.embeddings.create(model=model, input=resume_text).data[0].embedding
        emb_snips = client.embeddings.create(model=model, input=texts).data
        scores = [_cosine(emb_resume, emb_s.embedding) for emb_s in emb_snips]
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}. Please check your API key and connection.")
    
    ranked = list(zip(jd_rows, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    out = []
    for row, sc in ranked[:max(1, int(top_k))]:
        out.append({
            "role_title": row.get("role_title", ""),
            "company": row.get("company", ""),
            "title": row.get("source_title", "") or row.get("title", ""),
            "link": row.get("source_url", "") or row.get("link", ""),
            "match_percent": round(float(sc * 100.0), 1),
        })
    return out

def openai_rank_courses(gaps, resume_text: str, snippets: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """Rank course snippets against the user's gaps + resume using OpenAI embeddings"""
    gaps = [str(g) for g in (gaps or []) if str(g).strip()]
    bundle = " ".join(gaps + [resume_text or ""]).strip()
    
    # Handle both training course format and general snippet format
    docs = []
    for s in (snippets or []):
        if isinstance(s, dict):
            # For training courses, use description as the text to embed
            snippet_text = s.get("description", "") or s.get("snippet", "") or s.get("title", "")
            docs.append(str(snippet_text))
        else:
            docs.append(str(s))

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = "text-embedding-3-small"
        emb_q = client.embeddings.create(model=model, input=bundle).data[0].embedding
        emb_docs = client.embeddings.create(model=model, input=docs).data
        scores = [
            float(np.dot(emb_q, d.embedding) / (np.linalg.norm(emb_q) * np.linalg.norm(d.embedding) + 1e-9))
            for d in emb_docs
        ]
    except Exception as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}. Please check your API key and connection.")

    pairs = list(zip(snippets or [], scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    
    out = []
    for r, sc in pairs[:max(1, int(top_k))]:
        # Handle both training course format and general snippet format
        title = r.get("title", "Course")
        link = r.get("link", "")
        provider = r.get("provider", "") or r.get("source", "")
        
        out.append({
            "title": title,
            "provider": provider,
            "link": link,
            "match_percent": round(float(sc * 100.0), 1)
        })
    return out

def openai_parse_resume(resume_text: str) -> Dict[str, Any]:
    """Parse resume text into structured data using OpenAI"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    # Define the expected schema
    schema = {
        "professional_summary": "",
        "current_role": {"role": "", "company": ""},
        "technical_skills": [""],
        "career_level": "",
        "industry_focus": "",
        "work_experience": [
            {"title": "", "company": "", "start_date": "", "end_date": "", "responsibilities": ""}
        ],
        "key_achievements": [""],
        "soft_skills": [""],
        "location": "",
        "projects": [""],
        "education": [
            {"degree": "", "institution": "", "graduation_date": ""}
        ],
        "certifications": [""],
    }
    
    prompt = f"""
    Analyze the following resume text and extract structured professional information.
    Return ONLY valid JSON matching EXACTLY this schema (keys & nesting):
    
    {json.dumps(schema, indent=2)}
    
    IMPORTANT: Do NOT extract personal identifying information like name, email, or phone number.
    Only extract location (country, state, city if available).
    Use empty strings/lists if information is unknown. No commentary outside JSON.
    
    Resume text:
    {resume_text}
    """
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        structured_data = json.loads(response.choices[0].message.content)
        return structured_data
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract structured data: {str(e)}")
