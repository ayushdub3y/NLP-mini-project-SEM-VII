"""Adversarial event/feeling distinctions independent of classifier confidence."""
import pytest

import counselling
import emotion_model
import reading_policy

SCREENSHOT_CASES = [
    "i stole chips", "i laughed on a disabled person", "i beat cancer", "i lost my virginity",
]


@pytest.mark.parametrize("text", SCREENSHOT_CASES + [
    "I passed my test", "I broke a window", "I went to the shop",
    "I had sex for the first time", "I recovered from an illness", "I made fun of a classmate",
    "I took something without asking", "I lost my keys", "My friend is happy", "I am not sad",
    "I don't feel happy", "My friend is sad but I am happy",
])
def test_event_reports_and_attribution_need_context(text):
    assert reading_policy.assess(text)["status"] == "needs_context"


@pytest.mark.parametrize("text", [
    "I'm happy", "I am terrified", "I feel sad", "I feel relieved after treatment",
    "I beat cancer and I feel grateful", "I lost my virginity and I feel happy",
    "I stole chips and feel guilty", "I miss my friend", "I regret what I did",
    "I feel disconnected from everyone", "I am proud of my friend",
])
def test_personal_feelings_are_not_rejected_for_the_topic(text):
    assert reading_policy.assess(text)["status"] == "supported"


@pytest.mark.parametrize("text", SCREENSHOT_CASES)
def test_high_model_score_does_not_bypass_applicability(text, monkeypatch):
    scores = {label: 0.001 for label in emotion_model.EMOTIONS}
    scores["sadness"] = 0.994
    monkeypatch.setattr(emotion_model, "_analyse", lambda inputs: ([scores] * len(inputs), [1] * len(inputs)))
    result = emotion_model.detect(text)
    assert result["confidence"] == 0.994
    assert result["scores"] == scores
    assert result["reading_status"] == "needs_context"
    assert result["uncertain"]
    reply = counselling.respond(result)
    assert "How did that feel for you?" in reply["prompts"]
    assert reply != counselling.SUPPORT["sadness"]


def test_joy_feedback_does_not_celebrate_the_action():
    reply = counselling.respond({"top": "joy", "uncertain": False, "reading_status": "supported"})
    assert "Does that fit" in reply["validation"]
    assert "Share the good news with someone close to you." not in reply["strategies"]
    assert any("anyone else" in step for step in reply["strategies"])


def test_long_distance_syntactic_negation():
    # Long distance negation across prepositional phrase
    text = "I did not, despite all their reassurance, feel particularly happy."
    result = reading_policy.assess(text)
    assert result["status"] == "needs_context"
    assert result["reason"] == "negated_feeling"


def test_third_person_affective_attribution():
    text = "My supervisor was furious."
    result = reading_policy.assess(text)
    assert result["status"] == "needs_context"
    assert result["reason"] == "other_person_context"


@pytest.mark.parametrize("text", [
    "I no longer feel happy",
    "I never feel joy",
    "I hardly feel happy anymore",
    "I cannot feel happy about this",
])
def test_adverbial_and_compound_negations(text):
    result = reading_policy.assess(text)
    assert result["status"] == "needs_context"
    assert result["reason"] == "negated_feeling"


