"""Evaluate labelled English CSV data with dual-model comparative benchmarking and telemetry."""
import argparse
import csv
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import time

import emotion_model
from classical_model import ClassicalEmotionModel, get_baseline_model


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


def evaluate_dual_models(cases, baseline_path=None):
    texts = [c["text"] for c in cases]
    expected = [c["label"] for c in cases]

    # Contextual Transformer Evaluation
    start_trans_load = time.perf_counter()
    classifier = emotion_model._classifier()
    trans_load_seconds = time.perf_counter() - start_trans_load

    readings, trans_durations = [], []
    for text in texts:
        t0 = time.perf_counter()
        readings.append(emotion_model.detect(text))
        trans_durations.append(time.perf_counter() - t0)

    trans_predicted = [r["top"] for r in readings]
    trans_metrics = classification_metrics(expected, trans_predicted)
    trans_mean_lat = sum(trans_durations) / len(trans_durations)

    # Statistical Baseline Evaluation
    baseline_metrics = None
    baseline_mean_lat = None
    js_divergences = []
    baseline_predicted = []
    high_divergence_cases = []

    model_baseline = None
    if baseline_path and Path(baseline_path).exists():
        try:
            model_baseline = ClassicalEmotionModel.load(baseline_path)
        except Exception:
            pass
    if model_baseline is None:
        model_baseline = get_baseline_model()

    if model_baseline and model_baseline.is_fitted:
        base_durations = []
        for text in texts:
            t0 = time.perf_counter()
            probs = model_baseline.predict_proba([text])[0]
            base_durations.append(time.perf_counter() - t0)
            top_base = max(probs, key=probs.get)
            baseline_predicted.append(top_base)

        baseline_metrics = classification_metrics(expected, baseline_predicted)
        baseline_mean_lat = sum(base_durations) / len(base_durations)

        for i, (r, text, exp) in enumerate(zip(readings, texts, expected)):
            b_probs = model_baseline.predict_proba([text])[0]
            js = emotion_model.jensen_shannon_divergence(r["scores"], b_probs)
            js_divergences.append(js)
            if js > 0.35:
                high_divergence_cases.append({
                    "text": text,
                    "expected": exp,
                    "transformer_pred": r["top"],
                    "baseline_pred": baseline_predicted[i],
                    "js_divergence": round(js, 4),
                })

    agreement_rate = (
        sum(t == b for t, b in zip(trans_predicted, baseline_predicted)) / len(texts)
        if baseline_predicted else None
    )

    return {
        "transformer": {
            "metrics": trans_metrics,
            "mean_inference_seconds": trans_mean_lat,
            "load_seconds": trans_load_seconds,
            "readings": readings,
            "revision": getattr(classifier.model.config, "_commit_hash", None),
        },
        "baseline": {
            "metrics": baseline_metrics,
            "mean_inference_seconds": baseline_mean_lat,
            "predicted": baseline_predicted,
        } if baseline_metrics else None,
        "divergence": {
            "mean_js_divergence": sum(js_divergences) / len(js_divergences) if js_divergences else 0.0,
            "max_js_divergence": max(js_divergences) if js_divergences else 0.0,
            "agreement_rate": agreement_rate,
            "high_divergence_samples": high_divergence_cases,
        } if js_divergences else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "data" / "evaluation.csv")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "reports" / "evaluation.json")
    parser.add_argument("--baseline", type=Path, default=Path(__file__).parent / "models" / "tfidf_baseline.joblib")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    results = evaluate_dual_models(cases, baseline_path=args.baseline)

    trans_info = results["transformer"]
    metrics = trans_info["metrics"]
    readings = trans_info["readings"]
    expected = [c["label"] for c in cases]
    majority = max(emotion_model.EMOTIONS, key=expected.count)

    report = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": emotion_model.MODEL_NAME,
        "model_revision": trans_info["revision"],
        "dataset": str(args.dataset.resolve()),
        "samples": len(cases),
        "caveat": "The bundled data are hand-written smoke examples, not a held-out benchmark or clinical validation.",
        "metrics": metrics,
        "majority_baseline": {"label": majority, **classification_metrics(expected, [majority] * len(cases))},
        "model_load_seconds": trans_info["load_seconds"],
        "mean_inference_seconds": trans_info["mean_inference_seconds"],
        "uncertain_entries": sum(result["uncertain"] for result in readings),
        "context_requests": sum(result["reading_status"] == "needs_context" for result in readings),
        "metric_scope": "Metrics evaluate raw transformer labels on all rows, including entries the application abstains on.",
        "errors": [{"text": case["text"], "expected": case["label"], "predicted": result["top"],
                    "model_score": result["confidence"]}
                   for case, result in zip(cases, readings) if case["label"] != result["top"]],
        "dual_model_benchmark": {
            "transformer": {
                "name": emotion_model.MODEL_NAME,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "mean_latency_sec": trans_info["mean_inference_seconds"],
            },
            "statistical_baseline": {
                "name": "Sublinear TF-IDF (1-2 grams) + Logistic Regression",
                "accuracy": results["baseline"]["metrics"]["accuracy"] if results.get("baseline") else None,
                "macro_f1": results["baseline"]["metrics"]["macro_f1"] if results.get("baseline") else None,
                "mean_latency_sec": results["baseline"]["mean_inference_seconds"] if results.get("baseline") else None,
            } if results.get("baseline") else None,
            "telemetry": results.get("divergence"),
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "streamlit", "plotly", "scikit-learn", "spacy")
            if importlib.util.find_spec(name) is not None
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print("           DUAL-MODEL BENCHMARK & EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Dataset: {args.dataset.name} ({len(cases)} samples)")
    print(f"Contextual Model ({emotion_model.MODEL_NAME}):")
    print(f"  Accuracy: {metrics['accuracy']:.1%} | Macro-F1: {metrics['macro_f1']:.3f} | Latency: {trans_info['mean_inference_seconds'] * 1000:.1f}ms")

    if results.get("baseline"):
        b_metrics = results["baseline"]["metrics"]
        b_lat = results["baseline"]["mean_inference_seconds"]
        print(f"Statistical Baseline (TF-IDF + LR):")
        print(f"  Accuracy: {b_metrics['accuracy']:.1%} | Macro-F1: {b_metrics['macro_f1']:.3f} | Latency: {b_lat * 1000:.1f}ms")

    if results.get("divergence"):
        div = results["divergence"]
        print(f"Disagreement & Calibration Telemetry:")
        print(f"  Model Agreement: {div['agreement_rate']:.1%}")
        print(f"  Mean JS Divergence: {div['mean_js_divergence']:.4f}")
        print(f"  High Divergence Cases (D_JS > 0.35): {len(div['high_divergence_samples'])}")

    print(f"Report saved to: {args.output}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
