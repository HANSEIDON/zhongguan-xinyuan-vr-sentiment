import json
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def load_eval_pair(eval_path: str, labeled_path: str):
    # establishing id + label pair
    with open(labeled_path, "r", encoding="utf-8") as f:
        labeled_data = json.load(f)
    id2label = {}
    for item in labeled_data:
        _id = item.get("id")
        label = item.get("label")
        if _id is not None and label is not None:
            id2label[_id] = label

    # establishing [id, content, label]
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    ids = []
    texts = []
    true_labels = []

    for item in eval_data:
        _id = item.get("id")
        content = item.get("content", "")
        if _id is None or not content:
            continue
        if _id not in id2label:
            # skip uncensored id
            continue
        ids.append(_id)
        texts.append(content.strip())
        true_labels.append(id2label[_id])

    return ids, texts, true_labels


def evaluate_eval_files(model, vectorizer, eval_path: str, labeled_path: str):
    ids, texts, true_labels = load_eval_pair(eval_path, labeled_path)
    print("Target Samples: %s" % eval_path)
    print(f"Total eval samples: {len(texts)}")

    X_eval = vectorizer.transform(texts)

    y_pred = model.predict(X_eval)

    print("=== Classification Report (per-class F1) ===")
    print(classification_report(true_labels, y_pred, digits=4))

    print("=== Accuracy ===")
    print(accuracy_score(true_labels, y_pred))

    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    print(confusion_matrix(true_labels, y_pred))

    return ids, true_labels, y_pred
