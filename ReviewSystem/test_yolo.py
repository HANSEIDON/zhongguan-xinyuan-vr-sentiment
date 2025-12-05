import json
import cv2
import os
from ultralytics import YOLO

# ==========================================
# [설정] 위치 미세 조정
# ==========================================
YAW_OFFSET = 0
PITCH_OFFSET = -5
# ==========================================

# 1. 경로 설정 (여기가 핵심입니다!)
# 현재 이 파일(test_yolo.py)이 있는 폴더 위치를 찾습니다.
current_folder = os.path.dirname(os.path.abspath(__file__))
# static 폴더 위치를 지정합니다.
static_folder = os.path.join(current_folder, "static")

# 이미지 파일의 정확한 경로: ReviewSystem/static/room.jpg
image_file_path = os.path.join(static_folder, "room.jpg")

# 2. 파일이 진짜 있는지 확인 (에러 방지)
if not os.path.exists(image_file_path):
    print(f"\n❌ 에러 발생! 파일을 찾을 수 없습니다.")
    print(f"찾으려는 위치: {image_file_path}")
    print("👉 'room.jpg' 파일이 'static' 폴더 안에 들어있는지 꼭 확인하세요!")
    exit()

# 3. 모델 로드 및 이미지 읽기
print(f"📸 사진 로딩 중... ({image_file_path})")
model = YOLO("yolov8x.pt")
img = cv2.imread(image_file_path)

# 이미지 읽기 실패 시 방어 코드
if img is None:
    print("❌ 이미지를 읽을 수 없습니다. 파일이 손상되었거나 경로가 잘못되었습니다.")
    exit()

img_height, img_width = img.shape[:2]
print(f"✅ 사진 로드 성공: {img_width}x{img_height}")

# 4. YOLO 실행
results = model.predict(source=img, save=False, conf=0.35, iou=0.5, agnostic_nms=True)
result = results[0]

hotspots = []

print(f"\n--- 감지된 물체 ---")

for box in result.boxes:
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]
    x, y, w, h = box.xywh[0].tolist()

    if class_name in ["chair", "couch", "bed", "dining table", "bench"]:
        target_y = y + (h * 0.3)
    else:
        target_y = y

    yaw = ((x / img_width) - 0.5) * 360
    vertical_fov = 180
    pitch = ((img_height / 2) - target_y) / img_height * vertical_fov

    yaw += YAW_OFFSET
    pitch += PITCH_OFFSET

    if -85 < pitch < 85:
        hotspots.append(
            {"pitch": pitch, "yaw": yaw, "type": "info", "text": class_name}
        )

# 5. JSON 저장 (static 폴더 안에 저장)
json_save_path = os.path.join(static_folder, "data.json")

final_data = {
    "panorama_image": "/static/room.jpg",  # 웹브라우저가 읽을 경로
    "hotSpots": hotspots,
}

with open(json_save_path, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=4, ensure_ascii=False)

print(f"\n✅ data.json 생성 완료!")
print(f"저장된 위치: {json_save_path}")
print("이제 'python app.py'를 실행하세요.")
