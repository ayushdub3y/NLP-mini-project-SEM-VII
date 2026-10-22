"""Opt-in checks against actual cached/downloaded weights."""
import os

import pytest

import emotion_model as em

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MODEL_TESTS") != "1", reason="Set RUN_MODEL_TESTS=1 to run actual model inference."
)


def test_short_scores_match_original_pipeline():
    texts = ["I'm so happy!", "I am terrified.", "I miss my friend."]
    original = em._classifier()(texts, truncation=True, max_length=512)
    scores = em._scores(texts)
    for actual, expected in zip(scores, original):
        assert actual == pytest.approx({row["label"]: row["score"] for row in expected}, abs=1e-6)


def test_long_entry_uses_every_token_and_the_final_window():
    classifier = em._classifier()
    tokenizer = classifier.tokenizer
    text = "I walked to college today. " * 100 + "I am terrified and scared about the exam tomorrow."
    encoded = tokenizer([text], truncation=True, max_length=512, return_overflowing_tokens=True)
    assert len(encoded["input_ids"]) > 1
    token_count = sum(len(window) - 2 for window in encoded["input_ids"])
    assert token_count == len(tokenizer(text, add_special_tokens=False)["input_ids"])
    decoded_windows = [tokenizer.decode(window, skip_special_tokens=True) for window in encoded["input_ids"]]
    original = classifier(decoded_windows, truncation=True, max_length=512)
    expected = {label: 0.0 for label in em.EMOTIONS}
    for ids, probabilities in zip(encoded["input_ids"], original):
        for row in probabilities:
            expected[row["label"]] += row["score"] * (len(ids) - 2) / token_count
    scores, counts = em._analyse([text])
    assert counts == [len(encoded["input_ids"])]
    assert scores[0] == pytest.approx(expected, abs=1e-5)


def test_real_detect_sentence_breakdown():
    result = em.detect("I'm so happy! I am terrified about tomorrow.")
    assert len(result["sentences"]) == 2
    assert result["sentences"][0]["emotion"] == "joy"
    assert result["sentences"][1]["emotion"] == "fear"
    assert sum(result["scores"].values()) == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("text", ["i stole chips", "i laughed on a disabled person", "i beat cancer", "i lost my virginity"])
def test_actual_model_mistakes_are_not_presented_as_personal_emotions(text):
    result = em.detect(text)
    assert result["reading_status"] == "needs_context"
    assert result["uncertain"]
    assert sum(result["scores"].values()) == pytest.approx(1, abs=1e-6)
