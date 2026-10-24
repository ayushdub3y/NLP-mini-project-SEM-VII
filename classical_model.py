"""Statistical Feature Extraction & Baseline Classifier (Sublinear TF-IDF + Multinomial Logistic Regression).

Implements deterministic linear feature attribution:
phi(t, k) = x_t * w_{k, t}
"""
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import emotion_model

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "tfidf_baseline.joblib"

# Preserve functional negation tokens in stopword removal
NEGATION_WORDS = {
    "not", "never", "no", "nor", "neither", "without",
    "hardly", "scarcely", "barely", "cannot", "cant",
    "dont", "wont", "isnt", "wasnt", "didnt", "couldnt", "shouldnt"
}
CUSTOM_STOP_WORDS = sorted(list(ENGLISH_STOP_WORDS - NEGATION_WORDS))


class ClassicalEmotionModel:
    """Sublinear TF-IDF vectorizer + Multinomial Logistic Regression baseline."""

    def __init__(self, c=1.0, max_iter=1000, random_state=42):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words=CUSTOM_STOP_WORDS,
            min_df=1,
        )
        self.classifier = LogisticRegression(
            C=c,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
        )
        self.classes_ = list(emotion_model.EMOTIONS)
        self.is_fitted = False

    def fit(self, texts, labels):
        """Fit vectorizer and multinomial logistic regression model."""
        if not texts or not labels:
            raise ValueError("Texts and labels cannot be empty.")
        x_vec = self.vectorizer.fit_transform(texts)
        self.classifier.fit(x_vec, labels)
        self.classes_ = list(self.classifier.classes_)
        self.is_fitted = True
        return self

    def predict(self, texts):
        """Predict top emotion label for each input text."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        x_vec = self.vectorizer.transform(texts)
        return self.classifier.predict(x_vec).tolist()

    def predict_proba(self, texts):
        """Return probability distribution across all 7 emotions for each text."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        x_vec = self.vectorizer.transform(texts)
        prob_matrix = self.classifier.predict_proba(x_vec)
        results = []
        for row in prob_matrix:
            probs = {label: float(val) for label, val in zip(self.classifier.classes_, row)}
            # Ensure all standard EMOTIONS are present
            scores = {em: probs.get(em, 0.0) for em in emotion_model.EMOTIONS}
            total = sum(scores.values()) or 1.0
            results.append({k: v / total for k, v in scores.items()})
        return results

    def explain(self, text, target_class=None, top_k=5):
        """Compute deterministic feature attribution phi(t, k) = x_t * w_{k, t} for target class."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        x_vec = self.vectorizer.transform([text])
        if x_vec.nnz == 0:
            return []

        if target_class is None:
            pred_probs = self.predict_proba([text])[0]
            target_class = max(pred_probs, key=pred_probs.get)

        if target_class not in self.classifier.classes_:
            return []

        class_idx = list(self.classifier.classes_).index(target_class)
        # Weights for target class: shape (n_classes, n_features) or (1, n_features) if binary
        if len(self.classifier.classes_) == 2 and self.classifier.coef_.shape[0] == 1:
            weights = self.classifier.coef_[0] if class_idx == 1 else -self.classifier.coef_[0]
        else:
            weights = self.classifier.coef_[class_idx]
        feature_names = self.vectorizer.get_feature_names_out()

        # Non-zero indices in x_vec
        _, col_indices = x_vec.nonzero()
        attributions = []
        for col in col_indices:
            token_salience = float(x_vec[0, col] * weights[col])
            attributions.append({
                "feature": str(feature_names[col]),
                "salience": token_salience,
                "weight": float(weights[col]),
                "tfidf": float(x_vec[0, col]),
            })

        # Sort descending by salience
        attributions.sort(key=lambda item: item["salience"], reverse=True)
        return attributions[:top_k]

    def save(self, output_path):
        """Serialize pipeline artifact to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, model_path):
        """Load serialized pipeline artifact from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Baseline model artifact not found at {path}")
        return joblib.load(path)


@lru_cache(maxsize=1)
def get_baseline_model(model_path=None):
    """Retrieve cached baseline model, or train on data/evaluation.csv if not persisted."""
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if path.exists():
        try:
            return ClassicalEmotionModel.load(path)
        except Exception:
            pass
    
    # Train fresh baseline on evaluation.csv if artifact does not exist
    csv_path = Path(__file__).parent / "data" / "evaluation.csv"
    if csv_path.exists():
        import csv
        with csv_path.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        texts = [r["text"] for r in rows if r.get("text") and r.get("label")]
        labels = [r["label"] for r in rows if r.get("text") and r.get("label")]
        if texts and labels:
            model = ClassicalEmotionModel()
            model.fit(texts, labels)
            try:
                model.save(path)
            except Exception:
                pass
            return model
    return None
