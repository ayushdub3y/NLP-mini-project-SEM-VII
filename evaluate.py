"""Evaluate labelled English CSV data; no inference is performed on import."""
import argparse
import csv
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import time

import emotion_model


def classification_metrics(expected, predicted):
    labels = emotion_model.EMOTIONS
    if not expected or len(expected) != len(predicted):
        raise ValueError("Expected and predicted labels must be non-empty and equally sized.")
    if any(label not in labels for label in [*expected, *predicted]):
        raise ValueError("Unknown emotion label.")
    matrix = [[0] * len(labels) for _ in labels]
    for truth, guess in zip(expected, predicted):
        matrix[labels.index(truth)][labels.index(guess)] += 1
    per_class = {}
    for i, label in enumerate(labels):
        correct = matrix[i][i]
        support = sum(matrix[i])
        guesses = sum(row[i] for row in matrix)
        precision = correct / guesses if guesses else 0.0
        recall = correct / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {
        "accuracy": sum(a == b for a, b in zip(expected, predicted)) / len(expected),
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(labels),
        "per_class": per_class, "labels": labels, "confusion_matrix": matrix,
    }


def load_cases(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"text", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Dataset must contain text and label columns.")
        cases = list(reader)
    if not cases:
        raise ValueError("Dataset is empty.")
    for number, case in enumerate(cases, start=2):
        if (not case["text"] or not case["text"].strip()
                or len(case["text"].strip()) > emotion_model.MAX_ENTRY_CHARS
                or case["label"] not in emotion_model.EMOTIONS):
            raise ValueError(f"Invalid text or emotion label on CSV row {number}.")
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "data" / "evaluation.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "reports" / "evaluation.json")
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    start = time.perf_counter()
    classifier = emotion_model._classifier()
    load_seconds = time.perf_counter() - start
    readings, durations = [], []
    for case in cases:
        start = time.perf_counter()
        readings.append(emotion_model.detect(case["text"]))
        durations.append(time.perf_counter() - start)
    expected = [case["label"] for case in cases]
    predicted = [result["top"] for result in readings]
    metrics = classification_metrics(expected, predicted)
    majority = max(emotion_model.EMOTIONS, key=expected.count)
    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": emotion_model.MODEL_NAME,
        "model_revision": getattr(classifier.model.config, "_commit_hash", None),
        "dataset": str(args.dataset.resolve()), "samples": len(cases),
        "caveat": "The bundled data are hand-written smoke examples, not a held-out benchmark or clinical validation.",
        "metrics": metrics,
        "majority_baseline": {"label": majority, **classification_metrics(expected, [majority] * len(cases))},
        "model_load_seconds": load_seconds,
        "mean_inference_seconds": sum(durations) / len(durations),
        "uncertain_entries": sum(result["uncertain"] for result in readings),
        "context_requests": sum(result["reading_status"] == "needs_context" for result in readings),
        "metric_scope": "Metrics evaluate raw transformer labels on all rows, including entries the application abstains on.",
        "errors": [{"text": case["text"], "expected": case["label"], "predicted": result["top"],
                    "model_score": result["confidence"]}
                   for case, result in zip(cases, readings) if case["label"] != result["top"]],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "streamlit", "plotly")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Samples: {len(cases)} | accuracy: {metrics['accuracy']:.1%} | macro-F1: {metrics['macro_f1']:.3f}")
    print(f"Mean inference: {report['mean_inference_seconds']:.3f}s | errors: {len(report['errors'])}")
    print(f"Report: {args.output}")
    print(report["caveat"])


if __name__ == "__main__":
    main()
