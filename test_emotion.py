"""Offline regression tests; importing tests never downloads a model."""
from types import SimpleNamespace

import pytest
import torch

import counselling
import emotion_model as em
import journal


class ToyTokenizer:
    model_max_length = 6

    def __call__(self, texts, **kwargs):
        inputs, owners = [], []
        for owner, text in enumerate(texts):
            tokens = [{"joy": 1, "sadness": 2}.get(word, 3) for word in text.split()]
            size = kwargs["max_length"] - 2
            for start in range(0, len(tokens), size):
                inputs.append([0, *tokens[start:start + size], 0])
                owners.append(owner)
        return {"input_ids": inputs, "attention_mask": [[1] * len(row) for row in inputs],
                "overflow_to_sample_mapping": owners}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def pad(self, windows, **kwargs):
        length = max(len(row["input_ids"]) for row in windows)
        return {key: torch.tensor([row[key] + [0] * (length - len(row[key])) for row in windows])
                for key in ("input_ids", "attention_mask")}


class ToyModel:
    def __init__(self):
        self.labels = list(reversed(em.EMOTIONS))
        self.config = SimpleNamespace(id2label=dict(enumerate(self.labels)))
        self.seen = []
        self.batch_sizes = []

    def __call__(self, input_ids, attention_mask):
        self.batch_sizes.append(len(input_ids))
        self.seen.extend([row[mask.bool()].tolist()[1:-1] for row, mask in zip(input_ids, attention_mask)])
        logits = torch.zeros((len(input_ids), 7))
        logits[:, self.labels.index("joy")] = (input_ids == 1).sum(dim=1) * 3
        logits[:, self.labels.index("sadness")] = (input_ids == 2).sum(dim=1) * 3
        return SimpleNamespace(logits=logits)


@pytest.fixture
def toy(monkeypatch):
    model = ToyModel()
    classifier = SimpleNamespace(tokenizer=ToyTokenizer(), model=model, device=torch.device("cpu"))
    monkeypatch.setattr(em, "_classifier", lambda: classifier)
    return model


def test_windows_include_tail_and_keep_samples_separate(toy):
    scores, counts = em._analyse(["word " * 12 + "sadness sadness", "joy joy"])
    assert counts == [4, 1]
    assert [token for window in toy.seen[:4] for token in window] == [3] * 12 + [2, 2]
    assert scores[0]["sadness"] > scores[0]["joy"]
    assert max(scores[1], key=scores[1].get) == "joy"
    assert all(sum(row.values()) == pytest.approx(1) for row in scores)


def test_partial_window_is_weighted_by_content_tokens(toy):
    scores, _ = em._analyse(["joy joy joy joy sadness"])
    first, tail = em._scores(["joy joy joy joy", "sadness"])
    assert scores[0]["joy"] == pytest.approx((first["joy"] * 4 + tail["joy"]) / 5)


def test_batches_are_bounded(toy):
    em._scores(["word " * 100])
    assert max(toy.batch_sizes) <= em.BATCH_SIZE
    assert sum(map(len, toy.seen)) == 100


def test_short_input_and_sentence_results(toy):
    result = em.detect("joy")
    assert result["top"] == "joy"
    assert result["chunks"] == 1
    assert result["sentences"] == []
    result = em.detect("joy. sadness.")
    assert [s["sentence"] for s in result["sentences"]] == ["joy.", "sadness."]


def test_empty_and_oversized_input_never_load_model(monkeypatch):
    def unavailable():
        pytest.fail("Model should not load")
    monkeypatch.setattr(em, "_classifier", unavailable)
    assert em.detect(" \n ") is None
    assert em._scores([]) == []
    with pytest.raises(ValueError):
        em.detect("x" * (em.MAX_ENTRY_CHARS + 1))
    with pytest.raises(TypeError):
        em.detect(None)


@pytest.mark.parametrize("text, expected", [
    ('No! OK. I cried.', ['No!', 'OK.', 'I cried.']),
    ('She said "I am sad." Then left.', ['She said "I am sad."', 'Then left.']),
    ('Dr. A. Smith helped. I felt better.', ['Dr. A. Smith helped.', 'I felt better.']),
    ('I scored 3.5 today.\nGreat!', ['I scored 3.5 today.', 'Great!']),
])
def test_sentence_boundaries(text, expected):
    assert em.split_sentences(text) == expected


@pytest.mark.parametrize("text", [
    "I want to kill myself", "I can’t go on", "I don’t want to live",
    "I do not want to exist", "I want to take my own life", "I wish I were dead",
    "I wish I wasn't alive", "self‑harm", "I want to\nkill myself", "I'm planning to overdose",
])
def test_crisis_variants(text):
    assert counselling.crisis_check(text)


@pytest.mark.parametrize("text", ["I'm killing it at work today", "The exam hurt my confidence", "I'm alive and happy"])
def test_benign_phrases(text):
    assert not counselling.crisis_check(text)


def test_reply_does_not_mutate_shared_templates():
    reply = counselling.respond({"top": "joy"})
    reply["strategies"].clear()
    assert counselling.respond({"top": "joy"})["strategies"]


def test_uncertainty_and_crisis_do_not_count_as_confident_distress():
    assert counselling.respond({"top": "joy", "uncertain": True}) != counselling.SUPPORT["joy"]
    assert counselling.trend_flag([{"top": "sadness", "uncertain": True}] * 5) is None
    assert counselling.trend_flag([{"top": "support requested"}] * 5) is None
    assert counselling.trend_flag([{"top": "sadness"}] * 3)
    assert counselling.trend_flag([{"top": "sadness"}] * 2) is None
    with pytest.raises(ValueError):
        counselling.trend_flag([], window=0)


@pytest.mark.parametrize("text", ['=HYPERLINK("evil")', "  +1+1", "@SUM(A1)", "\tformula", "-5"])
def test_csv_formula_escape(text):
    import csv
    import io
    history = [{"text": text, "top": "neutral"}]
    row = next(csv.DictReader(io.StringIO(journal.export_csv(history))))
    assert row["text"] == "'" + text
    assert history[0]["text"] == text
