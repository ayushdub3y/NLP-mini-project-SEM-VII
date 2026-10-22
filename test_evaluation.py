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
