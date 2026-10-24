"""English emotion inference with strided window tokenization and Normalized Shannon Entropy pooling."""
from functools import lru_cache
import math
from pathlib import Path
import re
from threading import RLock

import reading_policy

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
MODEL_REVISION = "0e1cd914e3d46199ed785853e12b57304e04178b"
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MAX_ENTRY_CHARS = 20_000
MAX_TOKENS = 512
DEFAULT_STRIDE = 64
SALIENT_GAMMA = 1.5
BATCH_SIZE = 8
_INFERENCE_LOCK = RLock()


@lru_cache(maxsize=1)
def _classifier():
    from transformers import pipeline

    return pipeline("text-classification", model=MODEL_NAME, revision=MODEL_REVISION, top_k=None, device=-1)


def compute_normalized_entropy(scores, epsilon=1e-12):
    """Calculate Normalized Shannon Entropy H(p) in [0, 1].

    H(p) = - 1/ln(K) * sum(p_k * ln(p_k + epsilon))
    A uniform distribution yields H -> 1.0 (maximum uncertainty).
    A peaked/one-hot distribution yields H -> 0.0 (maximum confidence).
    """
    values = list(scores.values()) if isinstance(scores, dict) else list(scores)
    k = len(values)
    if k <= 1:
        return 0.0
    total = sum(values)
    probs = [v / total for v in values] if total > 0 else [1.0 / k] * k
    h = -sum(p * math.log(max(p, epsilon)) for p in probs) / math.log(k)
    return max(0.0, min(1.0, float(h)))


def jensen_shannon_divergence(p_dist, q_dist, epsilon=1e-12):
    """Calculate Jensen-Shannon Divergence D_JS(P || Q) normalized to [0, 1]."""
    labels = EMOTIONS
    p = [p_dist.get(label, 0.0) for label in labels]
    q = [q_dist.get(label, 0.0) for label in labels]
    total_p = sum(p) or 1.0
    total_q = sum(q) or 1.0
    p = [v / total_p for v in p]
    q = [v / total_q for v in q]
    m = [0.5 * (pv + qv) for pv, qv in zip(p, q)]

    kl_pm = sum(pv * math.log(max(pv, epsilon) / max(mv, epsilon)) for pv, mv in zip(p, m) if pv > 0)
    kl_qm = sum(qv * math.log(max(qv, epsilon) / max(mv, epsilon)) for qv, mv in zip(q, m) if qv > 0)
    js = 0.5 * kl_pm + 0.5 * kl_qm
    return max(0.0, min(1.0, float(js / math.log(2))))


def _analyse(texts, stride=DEFAULT_STRIDE, pooling="entropy"):
    """Tokenize into bounded windows (with optional stride) and aggregate probabilities.

    When pooling='entropy', window weights are scaled by Normalized Shannon Entropy:
        Omega_m = N_m * (1.0 - H(p_m)) ** gamma
    This ensures acute emotional spikes dominate over neutral paragraphs.
    When pooling='content_tokens', weights fall back to linear content token counts.
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
        
        tokenizer_kwargs = {
            "truncation": True,
            "max_length": max_length,
            "return_overflowing_tokens": True,
            "padding": False,
        }
        if stride > 0:
            tokenizer_kwargs["stride"] = stride
            
        encoded = tokenizer(texts, **tokenizer_kwargs)
        owners = encoded.pop("overflow_to_sample_mapping")
        windows = [
            {key: values[i] for key, values in encoded.items()}
            for i in range(len(owners))
        ]
        totals = [{emotion: 0.0 for emotion in EMOTIONS} for _ in texts]
        weights = [0.0 for _ in texts]
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
                n_tokens = max(1, len(windows[index]["input_ids"]) - special_tokens)

                if pooling == "entropy":
                    window_scores = {label: score for label, score in zip(labels, scores)}
                    h = compute_normalized_entropy(window_scores)
                    salience = max(0.0, 1.0 - h) ** SALIENT_GAMMA
                    omega = max(n_tokens * salience, 1e-6)
                else:
                    omega = float(n_tokens)

                for label, score in zip(labels, scores):
                    totals[owner][label] += score * omega
                weights[owner] += omega
                counts[owner] += 1

        if any(w == 0 for w in weights):
            raise ValueError("The tokenizer did not return a window for each input.")

        aggregated = []
        for i, scores in enumerate(totals):
            total_w = weights[i] or 1.0
            norm_scores = {label: value / total_w for label, value in scores.items()}
            # Normalize to guarantee exact sum = 1.0
            sum_norm = sum(norm_scores.values()) or 1.0
            aggregated.append({k: v / sum_norm for k, v in norm_scores.items()})

        return aggregated, counts


def _scores(texts, **kwargs):
    return _analyse(texts, **kwargs)[0]


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
    """Return scores, top label, sentence readings, and telemetry metadata."""
    if not isinstance(text, str):
        raise TypeError("Entry must be a string.")
    text = text.strip()
    if not text:
        return None
    if len(text) > MAX_ENTRY_CHARS:
        raise ValueError(f"Entry exceeds the {MAX_ENTRY_CHARS:,} character limit.")
    sentences = split_sentences(text)
    inputs = [text] + (sentences if len(sentences) > 1 else [])
    try:
        scores, counts = _analyse(inputs, stride=DEFAULT_STRIDE, pooling="entropy")
    except TypeError:
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

    # Telemetry: Baseline comparison and Jensen-Shannon Divergence
    baseline_result = None
    divergence_js = None
    try:
        from classical_model import get_baseline_model
        baseline_model = get_baseline_model()
        if baseline_model and baseline_model.is_fitted:
            b_probs = baseline_model.predict_proba([text])[0]
            b_top = max(b_probs, key=b_probs.get)
            b_attributions = baseline_model.explain(text, target_class=b_top, top_k=5)
            divergence_js = jensen_shannon_divergence(overall, b_probs)
            baseline_result = {
                "top": b_top,
                "confidence": b_probs[b_top],
                "scores": b_probs,
                "attributions": b_attributions,
                "divergence_js": divergence_js,
            }
    except Exception:
        pass

    result = {
        "scores": overall, "top": top, "confidence": confidence,
        "sentences": per_sentence, "uncertain": uncertain, "chunks": counts[0],
        "reading_status": applicability["status"], "reading_reason": applicability["reason"],
        "entropy": compute_normalized_entropy(overall),
    }
    if baseline_result:
        result["baseline"] = baseline_result
        result["divergence_js"] = divergence_js

    return result
