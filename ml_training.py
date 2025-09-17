
# FILE: ml_training.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import os

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
