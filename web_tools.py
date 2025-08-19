
# FILE: web_tools.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os, re

def _cosine(a, b):
    import numpy as np
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(a.dot(b) / (na*nb))

def _norm(s: str) -> str:
    s = (s or "").strip()
    return re.sub(r"\s+", " ", s)

def _tfidf_scores(query: str, docs: List[str]) -> List[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    corpus = [query] + docs
    vec = TfidfVectorizer(stop_words="english", max_features=3000)
    X = vec.fit_transform(corpus)
    qv, D = X[0:1], X[1:]
    return cosine_similarity(qv, D)[0].tolist()

def serpapi_search(query: str, num: int = 10) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query: return []
    results: List[Dict[str, Any]] = []
    key = os.getenv("SERPAPI_API_KEY")
    if key:
        try:
            import requests
            params = {"engine":"google","q": query, "num": int(num), "api_key": key}
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=25)
            r.raise_for_status()
            data = r.json()
            for it in (data.get("organic_results") or [])[:num]:
                results.append({"title": it.get("title",""), "link": it.get("link",""),
                                "snippet": it.get("snippet",""), "source":"serpapi"})
            if results: return results
        except Exception:
            pass
    try:
        try:
            from ddgs import DDGS
        except Exception:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for x in ddgs.text(query, max_results=num):
                results.append({"title": x.get("title",""),
                                "link": x.get("href") or x.get("url",""),
                                "snippet": x.get("body",""),
                                "source": "ddg"})
    except Exception:
        return []
    return results

def openai_rank_roles(resume_text: str, role_snippets: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    resume_text = _norm(resume_text)
    if not role_snippets: return []
    api_key = os.getenv("OPENAI_API_KEY")
    texts = [str(s.get("snippet","")) for s in role_snippets]
    scores: List[float] = []
    if api_key:
        try:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                model = "text-embedding-3-small"
                emb_resume = client.embeddings.create(model=model, input=resume_text).data[0].embedding
                emb_snips = client.embeddings.create(model=model, input=texts).data
                for s, emb_s in zip(role_snippets, emb_snips):
                    scores.append(_cosine(emb_resume, emb_s.embedding))
            except Exception:
                import openai
                openai.api_key = api_key
                model = "text-embedding-3-small"
                emb_resume = openai.Embedding.create(model=model, input=resume_text)["data"][0]["embedding"]
                emb_snips = openai.Embedding.create(model=model, input=texts)["data"]
                for s, emb_s in zip(role_snippets, emb_snips):
                    scores.append(_cosine(emb_resume, emb_s["embedding"]))
        except Exception:
            scores = _tfidf_scores(resume_text, texts)
    else:
        scores = _tfidf_scores(resume_text, texts)
    ranked = list(zip(role_snippets, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)
    out = []
    for r, sc in ranked[:max(1, int(top_k))]:
        out.append({"role_title": r.get("title",""), "score": float(sc), "link": r.get("link","")})
    return out

def openai_rank_jds(resume_text: str, jd_rows: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    resume_text = _norm(resume_text)
    if not jd_rows: return []
    api_key = os.getenv("OPENAI_API_KEY")
    texts = [str(x.get("jd_text","")) for x in jd_rows]
    scores: List[float] = []
    if api_key:
        try:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                model = "text-embedding-3-small"
                emb_resume = client.embeddings.create(model=model, input=resume_text).data[0].embedding
                emb_snips = client.embeddings.create(model=model, input=texts).data
                for r, emb_s in zip(jd_rows, emb_snips):
                    scores.append(_cosine(emb_resume, emb_s.embedding))
            except Exception:
                import openai
                openai.api_key = api_key
                model = "text-embedding-3-small"
                emb_resume = openai.Embedding.create(model=model, input=resume_text)["data"][0]["embedding"]
                emb_snips = openai.Embedding.create(model=model, input=texts)["data"]
                for r, emb_s in zip(jd_rows, emb_snips):
                    scores.append(_cosine(emb_resume, emb_s["embedding"]))
        except Exception:
            scores = _tfidf_scores(resume_text, texts)
    else:
        scores = _tfidf_scores(resume_text, texts)
    ranked = list(zip(jd_rows, scores)); ranked.sort(key=lambda x: x[1], reverse=True)
    out = []
    for row, sc in ranked[:max(1, int(top_k))]:
        out.append({
            "role_title": row.get("role_title",""),
            "company": row.get("company",""),
            "title": row.get("source_title","") or row.get("title",""),
            "link": row.get("source_url","") or row.get("link",""),
            "match_percent": round(float(sc*100.0), 1),
        })
    return out


def openai_rank_courses(gaps, resume_text: str, snippets: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Rank course snippets against the user's gaps + resume.
    snippets: [{title, link, snippet}, ...]
    Returns: list of dicts {title, link, provider?, match_percent}
    """
    import os, re
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    gaps = [str(g) for g in (gaps or []) if str(g).strip()]
    bundle = " ".join(gaps + [resume_text or ""]).strip()
    docs = [str(s.get("snippet","")) for s in (snippets or [])]

    # Try OpenAI embeddings if available
    api_key = os.getenv("OPENAI_API_KEY"); scores = None
    if api_key:
        try:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                model = "text-embedding-3-small"
                emb_q = client.embeddings.create(model=model, input=bundle).data[0].embedding
                emb_docs = client.embeddings.create(model=model, input=docs).data
                scores = [float(np.dot(emb_q, d.embedding) / (np.linalg.norm(emb_q)*np.linalg.norm(d.embedding)+1e-9)) for d in emb_docs]
            except Exception:
                import openai, numpy as np
                openai.api_key = api_key
                model = "text-embedding-3-small"
                emb_q = openai.Embedding.create(model=model, input=bundle)["data"][0]["embedding"]
                emb_docs = openai.Embedding.create(model=model, input=docs)["data"]
                scores = [float(np.dot(emb_q, d["embedding"]) / (np.linalg.norm(emb_q)*np.linalg.norm(d["embedding"])+1e-9)) for d in emb_docs]
        except Exception:
            scores = None
    if scores is None:
        vec = TfidfVectorizer(stop_words="english", max_features=4000)
        X = vec.fit_transform([bundle] + docs)
        qv, D = X[0:1], X[1:]
        scores = cosine_similarity(qv, D)[0].tolist()

    pairs = list(zip(snippets or [], scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    out = []
    for r, sc in pairs[:max(1, int(top_k))]:
        title = r.get("title","Course"); link = r.get("link",""); prov = r.get("source","")
        out.append({"title": title, "provider": prov, "link": link, "match_percent": round(float(sc*100.0), 1)})
    return out
