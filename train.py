"""
Обучение модели классификации новостей AG News.
Запуск: python train.py
Артефакты: artifacts/pipeline.joblib, artifacts/meta.json, artifacts/etalon.json
"""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

MODEL_VERSION = "1.0.0"
LABELS = ["World", "Sports", "Business", "Sci-Tech"]


def main():
    print("Загрузка AG News...")
    ds = load_dataset("fancyzhx/ag_news")
    train = ds["train"]
    test = ds["test"]

    X_train = train["text"]
    y_train = train["label"]
    X_test = test["text"]
    y_test = test["label"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50_000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    print("Обучение...")
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"Accuracy: {acc:.4f}, F1-macro: {f1:.4f}")

    pipeline_path = ARTIFACTS / "pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)

    data_hash = hashlib.md5("fancyzhx/ag_news".encode()).hexdigest()[:8]
    meta = {
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "fancyzhx/ag_news",
        "data_hash": data_hash,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "metrics": {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)},
        "labels": LABELS,
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    samples = [
        "Stocks rally as oil prices fall",
        "Team wins championship in overtime",
        "Apple releases new iPad model",
        "World leaders meet for climate summit",
        "Central bank raises interest rates",
    ]
    etalon = []
    for text in samples:
        proba = pipeline.predict_proba([text])[0]
        idx = int(proba.argmax())
        etalon.append({
            "text": text,
            "expected_label": idx,
            "expected_label_name": LABELS[idx],
            "expected_proba": round(float(proba[idx]), 4),
        })
    (ARTIFACTS / "etalon.json").write_text(json.dumps(etalon, ensure_ascii=False, indent=2))

    print("Готово. Артефакты в artifacts/")


if __name__ == "__main__":
    main()
