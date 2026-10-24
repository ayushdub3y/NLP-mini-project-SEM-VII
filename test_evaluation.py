import pytest

from evaluate import classification_metrics, load_cases


def test_metrics_include_missing_classes_and_confusion():
    result = classification_metrics(["joy", "sadness", "sadness"], ["joy", "joy", "sadness"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["per_class"]["joy"]["precision"] == 0.5
    assert result["per_class"]["sadness"]["recall"] == 0.5
    assert result["macro_f1"] == pytest.approx((2 / 3 + 2 / 3) / 7)
    assert sum(map(sum, result["confusion_matrix"])) == 3


def test_invalid_dataset_fails_before_loading_model(tmp_path):
    source = tmp_path / "invalid.csv"
    source.write_text("text,label\nhello,unknown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 2"):
        load_cases(source)


def test_evaluate_dual_models_mocked(monkeypatch):
    import emotion_model
    from evaluate import evaluate_dual_models
    from types import SimpleNamespace

    dummy_scores = {e: 0.1 for e in emotion_model.EMOTIONS}
    dummy_scores["joy"] = 0.4
    monkeypatch.setattr(emotion_model, "_classifier", lambda: SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash="abc"))))
    monkeypatch.setattr(emotion_model, "detect", lambda text: {
        "top": "joy", "confidence": 0.4, "scores": dummy_scores, "uncertain": False, "reading_status": "supported"
    })
    cases = [{"text": "I feel joyful and happy today.", "label": "joy"}]
    results = evaluate_dual_models(cases)
    assert "transformer" in results
    assert results["transformer"]["metrics"]["accuracy"] == 1.0
    if results.get("divergence"):
        assert "mean_js_divergence" in results["divergence"]

