import settings

import json
from collections import Counter


# 1. 파일 경로 설정
file_path = str(settings.DATA_HOME / "train/usual_train.txt")

# 2. 데이터 불러오기
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. 데이터 양 확인
print(f"총 데이터 개수: {len(data)}개")
print("-" * 30)

# 4. 처음 5개 데이터 눈으로 확인하기
print("--- [처음 5개 데이터 미리보기] ---")
for item in data[:5]:
    print(f"ID: {item['id']}")
    print(f"내용: {item['content']}")
    print(f"감정(Label): {item['label']}")
    print("-" * 20)

# 5. 데이터셋에 어떤 감정들이 있는지 확인 (라벨 분포)
labels = [item["label"] for item in data]

label_counts = Counter(labels)

print("\n--- [감정 라벨별 데이터 개수] ---")
for label, count in label_counts.items():
    print(f"{label}: {count}개")
