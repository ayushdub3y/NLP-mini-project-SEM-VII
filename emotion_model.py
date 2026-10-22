"""English emotion inference with complete, bounded token-window coverage."""
import math
import re
from functools import lru_cache
from threading import RLock

import reading_policy

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
MODEL_REVISION = "0e1cd914e3d46199ed785853e12b57304e04178b"
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MAX_ENTRY_CHARS = 20_000
MAX_TOKENS = 512
BATCH_SIZE = 8
_INFERENCE_LOCK = RLock()


@lru_cache(maxsize=1)
def _classifier():
    from transformers import pipeline

    return pipeline("text-classification", model=MODEL_NAME, revision=MODEL_REVISION, top_k=None, device=-1)


def _analyse(texts):
    """Average window probabilities by content-token count, without dropping tails.

    Windows do not overlap, so no tokens are double-counted. Short inputs retain
    the model's ordinary softmax scores. Averaged scores are an aggregation
    heuristic, not calibrated probabilities of a person's emotional state.
    """
    if not texts:
        return [], []
    import torch

    with _INFERENCE_LOCK:
        classifier = _classifier()
        tokenizer = classifier.tokenizer
        labels = [classifier.model.config.id2label[i].lower() for i in range(len(EMOTIONS))]
        if set(labels) != set(EMOTIONS):
            raise ValueError("The model does not expose the expected seven emotion labels.")
        max_length = min(MAX_TOKENS, tokenizer.model_max_length)
        encoded = tokenizer(
            texts, truncation=True, max_length=max_length,
            return_overflowing_tokens=True, padding=False,
        )
        owners = encoded.pop("overflow_to_sample_mapping")
        windows = [
            {key: values[i] for key, values in encoded.items()}
            for i in range(len(owners))
        ]
        totals = [{emotion: 0.0 for emotion in EMOTIONS} for _ in texts]
        weights = [0 for _ in texts]
        counts = [0 for _ in texts]
        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        for start in range(0, len(windows), BATCH_SIZE):
            batch_windows = windows[start:start + BATCH_SIZE]
            batch = tokenizer.pad(batch_windows, padding=True, return_tensors="pt")
            batch = {key: value.to(classifier.device) for key, value in batch.items()}
            with torch.inference_mode():
                probabilities = classifier.model(**batch).logits.softmax(dim=-1).cpu().tolist()
            if len(probabilities) != len(batch_windows):
                raise ValueError("The model returned an incomplete batch.")
            for offset, scores in enumerate(probabilities):
                if len(scores) != len(labels) or any(not math.isfinite(v) or v < 0 for v in scores):
                    raise ValueError("The model returned invalid emotion scores.")
                index = start + offset
                owner = owners[index]
                weight = max(1, len(windows[index]["input_ids"]) - special_tokens)
                for label, score in zip(labels, scores):
                    totals[owner][label] += score * weight
                weights[owner] += weight
                counts[owner] += 1
        if any(weight == 0 for weight in weights):
            raise ValueError("The tokenizer did not return a window for each input.")
        return [
            {label: value / weights[i] for label, value in scores.items()}
            for i, scores in enumerate(totals)
        ], counts


def _scores(texts):
    return _analyse(texts)[0]


def split_sentences(text):
    """Lightweight English splitter; retains short sentences and quoted endings."""
    parts = []
    start = 0
    for boundary in re.finditer(r"[.!?]+[\"'\u2019\u201d)]*(?=\s|$)|\n+", text):
        prefix = text[start:boundary.end()].strip()
        # Avoid common titles and single initials (e.g. Dr. A. Smith).
        if boundary.group().startswith(".") and re.search(
            r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|[A-Z])\.$", prefix
        ):
            continue
        if prefix:
            parts.append(prefix)
        start = boundary.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def detect(text):
    """Return scores, top label, sentence readings, and aggregation metadata."""
    if not isinstance(text, str):
        raise TypeError("Entry must be a string.")
    text = text.strip()
    if not text:
        return None
    if len(text) > MAX_ENTRY_CHARS:
        raise ValueError(f"Entry exceeds the {MAX_ENTRY_CHARS:,} character limit.")
    sentences = split_sentences(text)
    inputs = [text] + (sentences if len(sentences) > 1 else [])
    scores, counts = _analyse(inputs)
    overall = scores[0]
    ranked = sorted(overall, key=overall.get, reverse=True)
    top = ranked[0]
    confidence = overall[top]
    applicability = reading_policy.assess(text)
    uncertain = (confidence < 0.5 or confidence - overall[ranked[1]] < 0.15
                 or applicability["status"] == "needs_context")
    per_sentence = []
    for sentence, values in zip(inputs[1:], scores[1:]):
        label = max(values, key=values.get)
        sentence_policy = reading_policy.assess(sentence)
        per_sentence.append({"sentence": sentence, "emotion": label, "confidence": values[label],
                             "reading_status": sentence_policy["status"]})
    return {
        "scores": overall, "top": top, "confidence": confidence,
        "sentences": per_sentence, "uncertain": uncertain, "chunks": counts[0],
        "reading_status": applicability["status"], "reading_reason": applicability["reason"],
    }
