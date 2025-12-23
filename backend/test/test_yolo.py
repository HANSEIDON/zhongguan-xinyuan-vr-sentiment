from ultralytics import YOLO
import json
import os
from PIL import Image

# 1. 모델 및 이미지 설정
model = YOLO("yolov8x.pt")  # 모델 로드
image_path = os.path.join("static", "room.jpg")  # 이미지 경로

# 2. 감지할 가구 목록 (여기에 없는 건 무시함)
# bottle, cup 등은 제외하고 가구 위주로만 설정
TARGET_CLASSES = [
    "bed",
    "couch",
    "sofa",
    "chair",
    "dining table",
    "refrigerator",
    "microwave",
]

# 3. YOLO 추론 실행
results = model(image_path)


# 4. 파노라마 좌표 변환 함수
def convert_xy_to_pitch_yaw(x, y, img_width, img_height):
    # YOLO 좌표(2D)를 파노라마 좌표(3D 구체)로 변환
    yaw = (x / img_width) * 360 - 180
    pitch = ((y / img_height) * 180 - 90) * -1
    return pitch, yaw


# test_yolo.py 의 중간 for문 부분을 이걸로 덮어쓰세요

detected_objects = []
detected_classes = set()  # ★ 이미 찾은 가구 이름을 저장할 통

img = Image.open(image_path)
img_width, img_height = img.size

# 5. 결과 처리
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]

        # (1) 필터링: 우리가 원하는 가구가 아니면 건너뜀
        if name not in TARGET_CLASSES:
            continue

        # ★ (2) 중복 제거: 이미 찾은 가구면 건너뜀 (여기 추가됨!)
        # 예: chair를 이미 찾았으면, 두 번째 chair는 무시함
        if name in detected_classes:
            continue

        # 목록에 이름 등록 (이제 이 가구는 다시 안 찾음)
        detected_classes.add(name)

        # 좌표 추출
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # 위치 보정: 침대나 가구의 '정중앙'을 잡도록 설정
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 파노라마 (Pitch, Yaw)로 변환
        pitch, yaw = convert_xy_to_pitch_yaw(center_x, center_y, img_width, img_height)

        # 데이터 저장
        detected_objects.append(
            {"pitch": pitch, "yaw": yaw, "type": "info", "text": name}
        )

# (3) 옷장(Wardrobe) 수동 추가 (기존 코드 유지)
detected_objects.append(
    {"pitch": -5.0, "yaw": 120.0, "type": "info", "text": "wardrobe"}
)


# 6. JSON 파일로 저장
json_path = "static/data.json"
output_data = {"panorama_image": "/static/room.jpg", "hotSpots": detected_objects}

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(
    f"✅ 처리가 완료되었습니다! 감지된 가구: {[obj['text'] for obj in detected_objects]}"
)
print("이제 'python app.py'를 실행하고 브라우저를 강력 새로고침 하세요.")
