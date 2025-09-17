# FILE: utils/ml_inference.py - ML Model Inference Utilities
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _get_clf_and_classes(model):
    """Extract classifier and classes from a trained pipeline"""
    clf = None; classes = None
    try: clf = model.named_steps.get("clf")
    except Exception: pass
    if clf is None:
        try:
            for _, step in getattr(model, "named_steps", {}).items():
                if hasattr(step, "fit") and hasattr(step, "predict"): clf = step
        except Exception: pass
    try: classes = getattr(clf, "classes_", None) or getattr(model, "classes_", None)
    except Exception: classes = getattr(model, "classes_", None)
    return clf, classes

def _softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax to convert scores to probabilities"""
    x = x.astype(float); x = x - np.max(x); e = np.exp(x); s = e.sum()
    return e/s if s>0 else np.full_like(x, 1.0/len(x))

def rank_roles_for_resume(model, resume_text: str, top_k: int = 5):
    """Rank job roles for a given resume using a trained model"""
    if not resume_text: return []
    clf, classes = _get_clf_and_classes(model)
    if classes is None: return []
    try: X = model.transform([resume_text])
    except Exception:
        vec = None
        try: vec = model.named_steps.get("tfidf")
        except Exception:
            for _, step in getattr(model, "named_steps", {}).items():
                if hasattr(step, "transform"): vec = step; break
        if vec is None: return []
        X = vec.transform([resume_text])
    try: probs = clf.predict_proba(X)[0]
    except Exception:
        try: probs = _softmax(np.ravel(clf.decision_function(X)))
        except Exception: return []
    idx = np.argsort(probs)[::-1][:max(1, int(top_k))]
    return [{"role_title": str(classes[i]), "score": float(probs[i])} for i in idx]

def _get_vectorizer_from_pipeline(model):
    """Extract TF-IDF vectorizer from a trained pipeline"""
    try:
        vec = model.named_steps.get("tfidf")
        if isinstance(vec, TfidfVectorizer): return vec
    except Exception: pass
    try:
        for _, step in getattr(model, "named_steps", {}).items():
            if isinstance(step, TfidfVectorizer): return step
    except Exception: pass
    return None

def rank_jds_within_role(model, resume_text: str, jd_df, role: str, top_n: int = 5):
    """Rank job descriptions within a specific role using similarity matching"""
    import pandas as pd
    if jd_df is None or not isinstance(jd_df, pd.DataFrame) or jd_df.empty: return []
    rows = jd_df[jd_df["role_title"]==role]
    if rows.empty: return []
    rows = rows.copy(); rows["jd_text"] = rows["jd_text"].astype(str).fillna("")
    texts = rows["jd_text"].tolist()
    vec = _get_vectorizer_from_pipeline(model)
    if vec is None:
        vec = TfidfVectorizer(stop_words="english", max_features=3000)
        D = vec.fit_transform(texts); R = vec.transform([resume_text or ""])
    else:
        try: D = vec.transform(texts); R = vec.transform([resume_text or ""])
        except Exception:
            vec2 = TfidfVectorizer(stop_words="english", max_features=3000)
            D = vec2.fit_transform(texts); R = vec2.transform([resume_text or ""])
    sims = cosine_similarity(R, D)[0]
    ix = np.argsort(sims)[::-1][:max(1, int(top_n))]
    picked = rows.iloc[ix]
    out = []
    for (_, r), sc in zip(picked.iterrows(), sims[ix]):
        out.append({
            "role_title": role,
            "company": r.get("company",""),
            "title": r.get("source_title",""),
            "link": r.get("source_url",""),
            "match_percent": round(float(sc*100.0), 1),
        })
    return out

def recommend_courses_by_model(
    gaps: List[str], train_df, model, text_col: str = "description",
    title_col: str = "title", provider_col: str = "provider", link_col: str = "link",
    top_n_per_gap: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """Recommend courses based on skill gaps using a trained model"""
    import pandas as pd
    if not isinstance(train_df, pd.DataFrame) or train_df.empty:
        return {}
    # Get vectorizer from pipeline or fit a temporary one
    vec = None
    try:
        for _, step in getattr(model, "named_steps", {}).items():
            if isinstance(step, TfidfVectorizer): vec = step; break
    except Exception:
        pass
    texts = train_df[text_col].astype(str).fillna("")
    if vec is None:
        vec = TfidfVectorizer(stop_words="english", max_features=5000).fit(texts)
        D = vec.transform(texts)
    else:
        try:
            D = vec.transform(texts)
        except Exception:
            vec = TfidfVectorizer(stop_words="english", max_features=5000).fit(texts)
            D = vec.transform(texts)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for gap in (gaps or []):
        q = str(gap or "").strip()
        if not q: continue
        qv = vec.transform([q])
        sims = (qv @ D.T).toarray()[0]
        idx = np.argsort(sims)[::-1][:max(1, int(top_n_per_gap))]
        recs = []
        for i in idx:
            recs.append({
                "title": str(train_df.iloc[i].get(title_col, "")),
                "provider": str(train_df.iloc[i].get(provider_col, "")),
                "link": str(train_df.iloc[i].get(link_col, "")),
                "match_percent": round(float(sims[i]*100.0), 1),
            })
        out[q] = recs
    return out

def recommend_courses_from_training_dataset(
    gaps: List[str], training_df, top_n_per_gap: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Recommend courses from web-scraped training dataset based on skill gaps.
    Uses both skill matching and content similarity.
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    if not isinstance(training_df, pd.DataFrame) or training_df.empty:
        return {}
    
    # Ensure required columns exist
    required_cols = ['skill', 'title', 'description', 'provider', 'link']
    missing_cols = [col for col in required_cols if col not in training_df.columns]
    if missing_cols:
        return {}
    
    recommendations = {}
    
    for gap in gaps:
        gap_lower = str(gap).lower().strip()
        if not gap_lower:
            continue
        
        # Method 1: Direct skill matching
        skill_matches = training_df[
            training_df['skill'].str.lower().str.contains(gap_lower, na=False)
        ].copy()
        
        # Method 2: Content similarity matching
        if skill_matches.empty or len(skill_matches) < top_n_per_gap:
            # Create combined text for similarity search
            training_df['combined_text'] = (
                training_df['title'].astype(str) + " " + 
                training_df['description'].astype(str) + " " +
                training_df['skill'].astype(str)
            ).str.lower()
            
            # Use TF-IDF for content similarity
            vectorizer = TfidfVectorizer(
                stop_words='english', 
                max_features=5000,
                ngram_range=(1, 2)
            )
            
            try:
                tfidf_matrix = vectorizer.fit_transform(training_df['combined_text'])
                gap_vector = vectorizer.transform([gap_lower])
                
                # Calculate cosine similarity
                similarities = cosine_similarity(gap_vector, tfidf_matrix)[0]
                
                # Get top matches
                top_indices = similarities.argsort()[::-1][:top_n_per_gap * 2]
                content_matches = training_df.iloc[top_indices].copy()
                content_matches['similarity_score'] = similarities[top_indices]
                
                # Filter out very low similarity scores
                content_matches = content_matches[content_matches['similarity_score'] > 0.1]
                
            except Exception:
                content_matches = pd.DataFrame()
        else:
            content_matches = pd.DataFrame()
        
        # Combine results, prioritizing skill matches
        combined_matches = []
        
        # Add skill matches first (higher priority)
        for _, row in skill_matches.head(top_n_per_gap).iterrows():
            combined_matches.append({
                'title': str(row.get('title', 'Unknown Course')),
                'provider': str(row.get('provider', 'Unknown')),
                'link': str(row.get('link', '')),
                'hours': row.get('hours'),
                'price': str(row.get('price', 'unknown')),
                'rating': row.get('rating'),
                'match_type': 'skill_match',
                'match_percent': 95.0  # High score for direct skill matches
            })
        
        # Fill remaining slots with content matches
        remaining_slots = top_n_per_gap - len(combined_matches)
        if remaining_slots > 0 and not content_matches.empty:
            for _, row in content_matches.head(remaining_slots).iterrows():
                # Skip if already added
                if any(match['title'] == str(row.get('title', '')) for match in combined_matches):
                    continue
                    
                similarity_score = row.get('similarity_score', 0.0)
                combined_matches.append({
                    'title': str(row.get('title', 'Unknown Course')),
                    'provider': str(row.get('provider', 'Unknown')),
                    'link': str(row.get('link', '')),
                    'hours': row.get('hours'),
                    'price': str(row.get('price', 'unknown')),
                    'rating': row.get('rating'),
                    'match_type': 'content_match',
                    'match_percent': round(float(similarity_score * 100), 1)
                })
        
        recommendations[gap] = combined_matches
    
    return recommendations
