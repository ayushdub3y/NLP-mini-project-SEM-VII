"""Tests for statistical baseline feature extraction, attribution, and persistence."""
import pytest
from pathlib import Path

from classical_model import (
    ClassicalEmotionModel,
    CUSTOM_STOP_WORDS,
    DEFAULT_MODEL_PATH,
    NEGATION_WORDS,
    get_baseline_model,
)
import emotion_model


def test_negation_words_preserved_in_stopwords():
    # Negation words must NOT be removed as stopwords
    for word in ["not", "never", "no", "without"]:
        assert word in NEGATION_WORDS
        assert word not in CUSTOM_STOP_WORDS


def test_model_training_and_prediction():
    texts = [
        "I am so joyful and happy today!",
        "I feel terrible sadness and grief.",
        "This makes me furious and angry.",
        "I am terrified and afraid of the dark.",
        "This is disgusting and gross.",
        "I was so shocked and surprised by the sudden news.",
        "The weather is mild today.",
    ]
    labels = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]

    model = ClassicalEmotionModel()
    model.fit(texts, labels)
    assert model.is_fitted

    preds = model.predict(texts)
    assert len(preds) == len(texts)

    probs = model.predict_proba(texts)
    assert len(probs) == len(texts)
    for row in probs:
        assert set(row.keys()) == set(emotion_model.EMOTIONS)
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-5)


def test_deterministic_feature_attribution():
    texts = [
        "I feel happy and joyful",
        "I feel sad and alone",
    ]
    labels = ["joy", "sadness"]
    model = ClassicalEmotionModel()
    model.fit(texts, labels)

    explanations = model.explain("I feel happy and joyful", target_class="joy", top_k=3)
    assert len(explanations) > 0
    top_feature = explanations[0]["feature"]
    assert top_feature in ["feel happy", "happy", "joyful", "feel"]
    assert explanations[0]["salience"] > 0
    assert "weight" in explanations[0]
    assert "tfidf" in explanations[0]

    # Test second class attribution in binary mode (verifying coef_ shape handling)
    sad_explanations = model.explain("I feel sad and alone", target_class="sadness", top_k=3)
    assert len(sad_explanations) > 0
    assert sad_explanations[0]["salience"] > 0


def test_serialization_and_deserialization(tmp_path):
    texts = ["happy entry", "sad entry"]
    labels = ["joy", "sadness"]
    model = ClassicalEmotionModel()
    model.fit(texts, labels)

    save_path = tmp_path / "baseline_test.joblib"
    model.save(save_path)
    assert save_path.exists()

    loaded = ClassicalEmotionModel.load(save_path)
    assert loaded.is_fitted
    original_pred = model.predict(texts)
    loaded_pred = loaded.predict(texts)
    assert original_pred == loaded_pred


def test_get_baseline_model():
    model = get_baseline_model()
    assert model is not None
    assert model.is_fitted
    prob = model.predict_proba(["I am having a wonderful day"])[0]
    assert "joy" in prob
