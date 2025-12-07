import json
from typing import List


def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts: List[str] = []
    labels: List[str] = []

    for item in data:
        content = item.get("content", "")
        label = item.get("label", "")
        if not content or not label:
            continue
        texts.append(content)
        labels.append(label)

    return texts, labels
