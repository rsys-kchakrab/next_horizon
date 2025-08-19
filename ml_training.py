
# FILE: ml_training.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import os

# ---------------- Resume text preparation ----------------
def prepare_resume_text(resume_doc: Dict[str, Any]) -> str:
    parts: List[str] = []
    if not isinstance(resume_doc, dict):
        return str(resume_doc) if resume_doc else ""
    cr = resume_doc.get("current_role") or {}
    if isinstance(cr, dict):
        parts.append(str(cr.get("role",""))); parts.append(str(cr.get("summary","")))
    elif isinstance(cr, str):
        parts.append(cr)
    skills = resume_doc.get("technical_skills") or resume_doc.get("skills") or []
    if isinstance(skills, (list, tuple)):
        parts.extend([str(s) for s in skills if s])
    elif isinstance(skills, str):
        parts.append(skills)
    for w in (resume_doc.get("work_experience") or []):
        if not isinstance(w, dict): parts.append(str(w)); continue
        parts.append(str(w.get("title",""))); parts.append(str(w.get("company",""))); parts.append(str(w.get("summary","")))
        b = w.get("bullets") or w.get("responsibilities")
        if isinstance(b, str): parts.append(b)
        elif isinstance(b, (list, tuple)): parts.extend([str(x) for x in b if x])
    for p in (resume_doc.get("projects") or []):
        if isinstance(p, dict):
            parts.append(str(p.get("name",""))); parts.append(str(p.get("description","")))
        else:
            parts.append(str(p))
    return " ".join([x for x in parts if x]).strip()

# ---------------- Role ranking for a trained model ----------------
def _get_clf_and_classes(model):
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
    x = x.astype(float); x = x - np.max(x); e = np.exp(x); s = e.sum()
    return e/s if s>0 else np.full_like(x, 1.0/len(x))

def rank_roles_for_resume(model, resume_text: str, top_k: int = 5):
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

# ---------------- JD ranking within a role ----------------
from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity

def _get_vectorizer_from_pipeline(model):
    try:
        vec = model.named_steps.get("tfidf")
        if isinstance(vec, _TfidfVectorizer): return vec
    except Exception: pass
    try:
        for _, step in getattr(model, "named_steps", {}).items():
            if isinstance(step, _TfidfVectorizer): return step
    except Exception: pass
    return None

def rank_jds_within_role(model, resume_text: str, jd_df, role: str, top_n: int = 5):
    import pandas as pd
    if jd_df is None or not isinstance(jd_df, pd.DataFrame) or jd_df.empty: return []
    rows = jd_df[jd_df["role_title"]==role]
    if rows.empty: return []
    rows = rows.copy(); rows["jd_text"] = rows["jd_text"].astype(str).fillna("")
    texts = rows["jd_text"].tolist()
    vec = _get_vectorizer_from_pipeline(model)
    if vec is None:
        vec = _TfidfVectorizer(stop_words="english", max_features=3000)
        D = vec.fit_transform(texts); R = vec.transform([resume_text or ""])
    else:
        try: D = vec.transform(texts); R = vec.transform([resume_text or ""])
        except Exception:
            vec2 = _TfidfVectorizer(stop_words="english", max_features=3000)
            D = vec2.fit_transform(texts); R = vec2.transform([resume_text or ""])
    sims = _cosine_similarity(R, D)[0]
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

# ---------------- Training APIs (role & course models) ----------------
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, Any]
    model_path: Optional[str] = None  # where it was saved (if saved)

def _filter_rare_labels(df, label_col: str, min_per_class: int = 2):
    vc = df[label_col].value_counts()
    keep = set(vc[vc >= int(min_per_class)].index.tolist())
    return df[df[label_col].isin(keep)].copy(), {k:int(v) for k,v in vc.to_dict().items()}

def _make_classifier(model_type: str):
    model_type = (model_type or "logreg").lower()
    if model_type == "linearsvc":
        return LinearSVC()
    return LogisticRegression(max_iter=2000, class_weight="balanced")

def _predict_proba_or_softmax(clf, X):
    try:
        return clf.predict_proba(X)
    except Exception:
        try:
            dec = clf.decision_function(X)
            import numpy as _np
            if dec.ndim == 1:  # binary
                dec = _np.vstack([-dec, dec]).T
            e = _np.exp(dec - dec.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        except Exception:
            return None

def _topk_accuracy(probs, y_true, classes, ks=(1,3,5)):
    import numpy as _np
    out = {}
    if probs is None or classes is None: return {f"top{k}_acc": None for k in ks}
    order = _np.argsort(probs, axis=1)[:, ::-1]
    for k in ks:
        topk = classes[order[:, :min(k, probs.shape[1])]]
        match = [(yt in row) for yt, row in zip(y_true, topk)]
        out[f"top{k}_acc"] = float(_np.mean(match))
    return out

def _get_detailed_metrics(y_true, y_pred, y_proba, classes, model_type: str):
    """Generate comprehensive validation metrics for classification tasks"""
    import numpy as np
    
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    prec_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro")
    rec_micro = recall_score(y_true, y_pred, average="micro")
    
    # Classification report (per-class metrics)
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Top-K accuracy
    topk_metrics = _topk_accuracy(y_proba, y_true, classes, ks=(1, 3, 5)) if y_proba is not None else {}
    
    # Class distribution analysis
    unique_true, counts_true = np.unique(y_true, return_counts=True)
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    
    class_distribution = {
        "true_distribution": dict(zip(unique_true, counts_true.tolist())),
        "predicted_distribution": dict(zip(unique_pred, counts_pred.tolist()))
    }
    
    # Model complexity metrics
    n_classes = len(classes)
    n_samples = len(y_true)
    
    return {
        "summary": {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "precision_macro": prec_macro,
            "precision_micro": prec_micro,
            "recall_macro": rec_macro,
            "recall_micro": rec_micro,
        },
        "topk": topk_metrics,
        "per_class": class_report,
        "confusion_matrix": conf_matrix.tolist(),
        "class_labels": classes.tolist() if hasattr(classes, 'tolist') else list(classes),
        "class_distribution": class_distribution,
        "model_complexity": {
            "n_classes": n_classes,
            "n_test_samples": n_samples,
            "classes_with_predictions": len(unique_pred),
            "model_type": model_type
        }
    }

def _calculate_class_balance_score(y):
    """Calculate a balance score for class distribution (1.0 = perfectly balanced, 0.0 = extremely imbalanced)"""
    import numpy as np
    from collections import Counter
    
    counts = Counter(y)
    total = len(y)
    n_classes = len(counts)
    
    if n_classes <= 1:
        return 1.0
    
    # Expected count if perfectly balanced
    expected_count = total / n_classes
    
    # Calculate how far each class is from expected
    deviations = [abs(count - expected_count) / expected_count for count in counts.values()]
    
    # Average deviation (0 = perfect balance, higher = more imbalanced)
    avg_deviation = sum(deviations) / len(deviations)
    
    # Convert to balance score (1 = perfect, 0 = worst)
    balance_score = max(0.0, 1.0 - (avg_deviation / 2.0))  # Normalize to 0-1 range
    
    return balance_score

def train_role_model(
    jd_df, *, text_col: str = "jd_text", label_col: str = "role_title",
    model_type: str = "logreg", test_size: float = 0.2, random_state: int = 42,
    min_per_class: int = 2, ngram_range: Tuple[int,int] = (1,2), max_features: int = 50000,
    min_df: int = 1, max_df: float = 0.95, use_sublinear_tf: bool = True,
    model_name: Optional[str] = None, out_dir: str = "./models", **kwargs
) -> TrainResult:
    """Train a role classifier with configurable hyperparameters."""
    import pandas as pd
    if not isinstance(jd_df, pd.DataFrame):
        raise ValueError("jd_df must be a pandas DataFrame")
    df = jd_df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df, label_dist = _filter_rare_labels(df, label_col, min_per_class=min_per_class)
    if df.empty:
        raise ValueError("No data left after filtering rare labels.")
    X = df[text_col].values; y = df[label_col].values
    stratify = y if len(set(y)) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=float(test_size), random_state=random_state, stratify=stratify)
    
    # Enhanced TfidfVectorizer with configurable parameters
    vectorizer = TfidfVectorizer(
        stop_words="english", 
        ngram_range=ngram_range, 
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=use_sublinear_tf,
        norm='l2'
    )
    
    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf", _make_classifier(model_type)),
    ])
    pipe.fit(Xtr, ytr)
    
    # Predictions and probabilities
    ypred = pipe.predict(Xte)
    probs = _predict_proba_or_softmax(pipe.named_steps["clf"], pipe.named_steps["tfidf"].transform(Xte))
    classes = getattr(pipe.named_steps["clf"], "classes_", None)
    
    # Calculate comprehensive metrics
    detailed_metrics = _get_detailed_metrics(yte, ypred, probs, classes, model_type)
    
    # Class balance analysis
    balance_score = _calculate_class_balance_score(ytr)
    
    # Compile all metrics
    metrics = {
        "validation": detailed_metrics,
        "data_quality": {
            "class_balance_score": balance_score,
            "n_train": int(len(Xtr)), 
            "n_test": int(len(Xte)),
            "label_distribution_before_filter": label_dist,
        },
        "hyperparameters": {
            "ngram_range": ngram_range,
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "use_sublinear_tf": use_sublinear_tf,
            "test_size": test_size,
            "min_per_class": min_per_class,
            "model_type": model_type
        }
    }
    model_path = None
    if model_name:
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, f"{model_name}.joblib")
        try:
            import joblib as _joblib
            _joblib.dump(pipe, model_path)
        except Exception:
            # fallback: no joblib available in environment
            pass
    return TrainResult(model=pipe, metrics=metrics, model_path=model_path)

def train_training_model(
    train_df, *, text_col: str = "description", label_col: Optional[str] = "skill",
    model_type: str = "logreg", test_size: float = 0.2, random_state: int = 42,
    ngram_range: Tuple[int,int] = (1,2), max_features: int = 50000, min_per_class: int = 2,
    min_df: int = 1, max_df: float = 0.95, use_sublinear_tf: bool = True,
    model_name: Optional[str] = None, out_dir: str = "./models", **kwargs
) -> TrainResult:
    """Course model trainer with configurable hyperparameters. If label_col is None, returns TF-IDF retrieval model."""
    import pandas as pd
    if not isinstance(train_df, pd.DataFrame):
        raise ValueError("train_df must be a pandas DataFrame")
    model_path = None
    if label_col and label_col in train_df.columns:
        df = train_df[[text_col, label_col]].dropna().copy()
        df[text_col] = df[text_col].astype(str).str.strip()
        df[label_col] = df[label_col].astype(str).str.strip()
        df, label_dist = _filter_rare_labels(df, label_col, min_per_class=min_per_class)
        if df.empty:
            raise ValueError("No data left after filtering rare labels.")
        X = df[text_col].values; y = df[label_col].values
        stratify = y if len(set(y)) > 1 else None
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=float(test_size), random_state=random_state, stratify=stratify)
        
        # Enhanced TfidfVectorizer with configurable parameters
        vectorizer = TfidfVectorizer(
            stop_words="english", 
            ngram_range=ngram_range, 
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=use_sublinear_tf,
            norm='l2'
        )
        
        pipe = Pipeline([("tfidf", vectorizer), ("clf", _make_classifier(model_type))])
        pipe.fit(Xtr, ytr)
        
        # Predictions and metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        ypred = pipe.predict(Xte)
        probs = _predict_proba_or_softmax(pipe.named_steps["clf"], pipe.named_steps["tfidf"].transform(Xte))
        classes = getattr(pipe.named_steps["clf"], "classes_", None)
        
        # Calculate comprehensive metrics
        detailed_metrics = _get_detailed_metrics(yte, ypred, probs, classes, model_type)
        balance_score = _calculate_class_balance_score(ytr)
        
        metrics = {
            "validation": detailed_metrics,
            "data_quality": {
                "class_balance_score": balance_score,
                "n_train": int(len(Xtr)), 
                "n_test": int(len(Xte)),
                "label_distribution_before_filter": label_dist,
            },
            "hyperparameters": {
                "ngram_range": ngram_range,
                "max_features": max_features,
                "min_df": min_df,
                "max_df": max_df,
                "use_sublinear_tf": use_sublinear_tf,
                "test_size": test_size,
                "min_per_class": min_per_class,
                "model_type": model_type
            }
        }
        if model_name:
            os.makedirs(out_dir, exist_ok=True)
            model_path = os.path.join(out_dir, f"{model_name}.joblib")
            try:
                import joblib as _joblib
                _joblib.dump(pipe, model_path)
            except Exception:
                pass
        return TrainResult(model=pipe, metrics=metrics, model_path=model_path)
    
    # Unsupervised TF-IDF model for retrieval
    texts = train_df[text_col].astype(str).fillna("").tolist()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    
    # Enhanced TfidfVectorizer for unsupervised mode
    vectorizer = TfidfVectorizer(
        stop_words="english", 
        ngram_range=ngram_range, 
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=use_sublinear_tf,
        norm='l2'
    )
    
    pipe = Pipeline([("tfidf", vectorizer)])
    pipe.fit(texts)
    if model_name:
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, f"{model_name}.joblib")
        try:
            import joblib as _joblib
            _joblib.dump(pipe, model_path)
        except Exception:
            pass
    metrics = {
        "mode": "unsupervised", 
        "n_items": int(len(texts)),
        "hyperparameters": {
            "ngram_range": ngram_range,
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "use_sublinear_tf": use_sublinear_tf
        }
    }
    return TrainResult(model=pipe, metrics=metrics, model_path=model_path)

# ---------------- Course recommendation using a trained model ----------------
from sklearn.feature_extraction.text import TfidfVectorizer

def recommend_courses_by_model(
    gaps: List[str], train_df, model, text_col: str = "description",
    title_col: str = "title", provider_col: str = "provider", link_col: str = "link",
    top_n_per_gap: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
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
