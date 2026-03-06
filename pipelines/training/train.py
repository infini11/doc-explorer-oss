import json
import logging
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from langchain_community.embeddings import HuggingFaceEmbeddings
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Parameters
EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR            = Path("storage/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_mtsamples(n: int = 5000) -> tuple[list[str], list[str]]:
    """Loads the MTSamples dataset from HuggingFace.
       Contains medical transcriptions labeled by specialty.

    Args:
        n (int, optional): Number of examples to load. Defaults to 5000.

    Returns:
        tuple[list[str], list[str]]: docs, labels
        Label examples: "Cardiology", "Neurology", "Orthopedic", "Radiology"...
    """

    logger.info(f"Loading MTSamples ({n} examples)...")
    ds = load_dataset("MedNLI/mednli", split=f"train[:{n}]")

    # MTSamples: "transcription" field = text, "medical_specialty" = label
    texts  = [row["sentence1"] + " " + row["sentence2"] for row in ds]
    labels = [row["gold_label"] for row in ds]

    logger.info(f"Dataset loaded: {len(texts)} examples, {len(set(labels))} classes")
    logger.info(f"Classes: {sorted(set(labels))}")
    return texts, labels


def load_chatdoctor(n: int = 5000) -> tuple[list[str], list[str]]:
    """Alternative: ChatDoctor dataset.
    Simulated severity labels are generated for classification.

    Args:
        n (int, optional): Number of examples to load. Defaults to 5000.

    Returns:
        tuple[list[str], list[str]]: docs, labels
    """

    logger.info(f"Loading ChatDoctor ({n} examples)...")
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split=f"train[:{n}]")

    texts = [f"{row['input']} {row['output']}" for row in ds]

    # Simulated labels based on keywords (in production: manually annotated)
    labels = []
    for row in ds:
        text = (row["input"] + row["output"]).lower()
        if any(w in text for w in ["emergency", "severe", "urgent", "critical", "hospital"]):
            labels.append("high")
        elif any(w in text for w in ["chronic", "persistent", "moderate", "consult"]):
            labels.append("medium")
        else:
            labels.append("low")

    logger.info(f"Dataset loaded: {len(texts)} examples")
    logger.info(f"Label distribution: { {l: labels.count(l) for l in set(labels)} }")
    return texts, labels


def embed_texts(texts: list[str]) -> np.ndarray:
    """_Transforms a list of texts into an embedding matrix (N, 384).
    Uses the same model as chunker.py for consistency.ary_

    Args:
        texts (list[str]): docs to embed

    Returns:
        np.ndarray: embedding matrix (N, 384) where N = len(texts)
    """
    logger.info(f"Generating embeddings ({len(texts)} texts)...")

    embedder = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectors = embedder.embed_documents(texts)
    X = np.array(vectors)

    logger.info(f"Embeddings generated: shape={X.shape}")
    return X


def train(
    texts: list[str],
    labels: list[str]
) -> XGBClassifier:
    """
    Trains the XGBoost classifier.

    Args:
        texts    : medical texts (one per document)
        labels   : label for each text

    Returns:
        The trained model
    """
    X = embed_texts(texts)

    le = LabelEncoder()
    y  = le.fit_transform(labels)
    logger.info(f"Encoded classes: { dict(zip(le.classes_, le.transform(le.classes_))) }")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Split: {len(X_train)} train / {len(X_test)} test")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
    )

    # Training
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Metrics
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    # Log classification report as artifact
    report_path = MODEL_DIR / "classification_report.txt"
    report_path.write_text(report)

    # Log class mapping as artifact
    classes_path = MODEL_DIR / "label_encoder_classes.json"
    classes_path.write_text(json.dumps(list(le.classes_)))

    # Also save locally
    local_model_path = MODEL_DIR / "doc_classifier.json"
    model.save_model(str(local_model_path))

    # Output summary
    print(f"\n{'='*50}")
    print(f"{'='*50}")
    print(f" Accuracy   : {acc:.4f}")
    print(f" F1 score   : {f1:.4f}")
    print(f"\n{report}")
    print(f" Model saved → {local_model_path}")

    logger.info(f"Training done — accuracy={acc:.4f} f1={f1:.4f}")
    return model


def predict(text: str) -> dict:
    """Predicts the class of a new medical text.
    Loads the model from disk.

    Args:
        text (str): input medical text to classify

    Raises:
        FileNotFoundError: failed to find the model file (run train.py first)

    Returns:
        dict: prediction results
    """
    model_path  = MODEL_DIR / "doc_classifier.json"
    classes_path = MODEL_DIR / "label_encoder_classes.json"

    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run train.py first.")

    model   = XGBClassifier()
    model.load_model(str(model_path))
    classes = json.loads(classes_path.read_text())

    X          = embed_texts([text])
    pred_idx   = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]

    return {
        "label":      classes[pred_idx],
        "confidence": round(float(pred_proba[pred_idx]), 4),
        "all_scores": {c: round(float(p), 4) for c, p in zip(classes, pred_proba)},
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Loading ChatDoctor dataset...")
    texts, labels = load_chatdoctor(n=300)

    print("\nStarting training...")
    model = train(texts, labels)

    # Prediction test
    print("\n--- Prediction test ---")
    test_text = "Patient with chest pain and difficulty breathing, needs immediate attention"
    result    = predict(test_text)
    print(f"Text      : {test_text}")
    print(f"Label     : {result['label']} (confidence: {result['confidence']})")
    print(f"Scores    : {result['all_scores']}")
    