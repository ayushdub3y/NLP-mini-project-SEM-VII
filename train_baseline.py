"""Train and serialize the statistical baseline classifier (Sublinear TF-IDF + Logistic Regression).

Usage:
    python train_baseline.py --dataset data/evaluation.csv --output models/tfidf_baseline.joblib
"""
import argparse
import csv
from pathlib import Path

from classical_model import ClassicalEmotionModel
from evaluate import classification_metrics


def train_baseline(dataset_path, output_path):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    with dataset_path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    texts = [r["text"].strip() for r in rows if r.get("text") and r.get("label")]
    labels = [r["label"].strip() for r in rows if r.get("text") and r.get("label")]

    if not texts or not labels:
        raise ValueError("No valid training samples found in dataset.")

    print(f"Training statistical baseline on {len(texts)} samples from {dataset_path}...")
    model = ClassicalEmotionModel()
    model.fit(texts, labels)

    # Evaluate on training corpus
    predictions = model.predict(texts)
    metrics = classification_metrics(labels, predictions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)

    print(f"Model saved to: {output_path}")
    print(f"Training Accuracy: {metrics['accuracy']:.1%}")
    print(f"Training Macro-F1: {metrics['macro_f1']:.3f}")
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).parent / "data" / "evaluation.csv",
        help="Path to labeled training CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "models" / "tfidf_baseline.joblib",
        help="Destination path for serialized model artifact.",
    )
    args = parser.parse_args()
    train_baseline(args.dataset, args.output)


if __name__ == "__main__":
    main()
