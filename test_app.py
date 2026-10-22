"""Exercise session state and failure paths through Streamlit's runtime."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import emotion_model


@pytest.fixture
def app(monkeypatch):
    def unavailable():
        raise RuntimeError("Model unavailable")
    monkeypatch.setattr(emotion_model, "_classifier", unavailable)
    at = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()
    assert not at.exception
    return at


def click(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def reading(text):
    scores = {label: (0.94 if label == "joy" else 0.01) for label in emotion_model.EMOTIONS}
    return {"scores": scores, "top": "joy", "confidence": 0.94,
            "sentences": [], "uncertain": False, "chunks": 1}


def test_model_failure_preserves_input_without_recording_entry(app):
    app.text_area[0].input("Today was pleasant.")
    click(app, "Reflect on this")
    assert app.info
    assert app.session_state["entry"] == "Today was pleasant."
    assert app.session_state["last"] is None
    assert app.session_state["history"] == []


def test_crisis_support_survives_model_failure_and_skips_inference(app, monkeypatch):
    def forbidden(text):
        pytest.fail("Crisis path must not call model")
    monkeypatch.setattr(emotion_model, "detect", forbidden)
    app.text_area[0].input("I don’t want to live")
    click(app, "Reflect on this")
    assert app.session_state["last"]["crisis"]
    assert app.session_state["last"]["result"] is None
    assert len(app.session_state["history"]) == 1
    assert any("You are not alone" in element.value for element in app.markdown)
    assert not any("What may help" in element.value for element in app.markdown)


def test_clear_removes_current_entry_and_result(app, monkeypatch):
    monkeypatch.setattr(emotion_model, "detect", reading)
    app.text_area[0].input("Happy today!")
    click(app, "Reflect on this")
    assert len(app.session_state["history"]) == 1
    click(app, "Clear journal")
    assert app.session_state["history"] == []
    assert app.session_state["entry"] == ""
    assert app.session_state["last"] is None


def test_empty_resubmission_clears_stale_result(app, monkeypatch):
    monkeypatch.setattr(emotion_model, "detect", reading)
    app.text_area[0].input("Happy today!")
    click(app, "Reflect on this")
    app.text_area[0].input(" ")
    click(app, "Reflect on this")
    assert app.session_state["last"] is None
    assert len(app.session_state["history"]) == 1
    assert "Write a sentence" in app.info[0].value


@pytest.mark.parametrize("text", ["i stole chips", "i laughed on a disabled person", "i beat cancer", "i lost my virginity"])
def test_context_request_hides_strong_label_and_keeps_journal_unscored(app, monkeypatch, text):
    result = {**reading(text), "reading_status": "needs_context", "uncertain": True}
    monkeypatch.setattr(emotion_model, "detect", lambda value: result)
    app.text_area[0].input(text)
    click(app, "Reflect on this")
    markup = " ".join(item.value for item in app.markdown)
    assert "How did that feel for you?" in markup
    assert 'class="result-title"' not in markup
    assert any("unconfirmed guess" in expander.label for expander in app.expander)
    entry = app.session_state["history"][0]
    assert entry["top"] == "needs context"
    assert entry["confidence"] is None
    assert not any(label in entry for label in emotion_model.EMOTIONS)
