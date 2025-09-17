# FILE: ai/openai_client.py - OpenAI Integration
from __future__ import annotations
from typing import Any, Dict, List
import os
import re
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
    docs = [str(s.get("snippet", "")) for s in (snippets or [])]

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
        title = r.get("title", "Course")
        link = r.get("link", "")
        prov = r.get("source", "")
        out.append({
            "title": title,
            "provider": prov,
            "link": link,
            "match_percent": round(float(sc * 100.0), 1)
        })
    return out
