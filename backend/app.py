import settings

import json
from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime

# AI 관련 라이브러리
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)
app.template_folder = settings.BACKEND_HOME / "templates"
app.static_folder = settings.BACKEND_HOME / "static"
app.static_url_path = "/static"

# ----------------------------------------------------
# [설정] 사용자 이름 매핑 (IP -> 별명)
# ----------------------------------------------------
user_ip_map = {}
nickname_list = [
    "Alice",
    "Bob",
    "Charlie",
    "David",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
    "Ivan",
    "Judy",
    "Kevin",
    "Lily",
    "Mallory",
    "Niaj",
    "Oscar",
    "Peggy",
]


def get_nickname(ip_address):
    if ip_address not in user_ip_map:
        idx = len(user_ip_map) % len(nickname_list)
        user_ip_map[ip_address] = nickname_list[idx]
    return user_ip_map[ip_address]


# ----------------------------------------------------
# [AI 준비] 모델 로딩
# ----------------------------------------------------
print("Loading AI Model...")

device = torch.device("cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(settings.USE_PRETRAINED)
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.USE_PRETRAINED, num_labels=6
    )

    checkpoint = torch.load(
        settings.MODEL_HOME / settings.USE_LOCAL_MODEL / settings.USE_PT_NAME,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("✅ AI Model Loaded Successfully!")
except Exception as e:
    print(f"⚠️ 모델 로딩 실패 (기본 모드로 동작합니다): {e}")
    model = None


# ----------------------------------------------------
# [AI 기능 1] 감정 분석 함수
# ----------------------------------------------------
def analyze_sentiment(text):
    if model is None:
        return "😐", "neutral"

    try:
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=140,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        pred_id = int(torch.argmax(output.logits, dim=-1).cpu().item())

        label_map = {
            0: "happy",
            1: "angry",
            2: "sad",
            3: "fear",
            4: "surprise",
            5: "neutral",
        }
        emoji_map = {
            "happy": "😊",
            "angry": "🤬",
            "sad": "😢",
            "fear": "😱",
            "surprise": "😧",
            "neutral": "😐",
        }

        label_text = label_map[pred_id]
        emoji = emoji_map[label_text]
        return emoji, label_text
    except Exception as exc:
        print(f"error while analyzing sentiment: {exc}")
        return "😐", "neutral"


# ----------------------------------------------------
# [AI 기능 2] 사물 인식 매핑 함수
# ----------------------------------------------------
def detect_target_object(text):
    text = text.lower()
    mapping = {
        "bed": [
            "床",
            "bed",
            "shuijiao",
            "睡觉",
            "床垫",
            "床单",
            "枕头",
            "被子",
            "铺",
            "塌",
        ],
        "couch": ["沙发", "sofa", "couch", "沙發", "坐垫", "软椅", "躺椅"],
        "chair": ["椅子", "chair", "seat", "座", "凳子", "板凳", "靠背椅", "转椅"],
        "dining table": [
            "餐桌",
            "table",
            "桌子",
            "食桌",
            "饭桌",
            "茶几",
            "台子",
            "案子",
            "写字台",
            "书桌",
        ],
        "toilet": [
            "马桶",
            "toilet",
            "卫生间",
            "厕所",
            "洗手间",
            "茅房",
            "便池",
            "坐便器",
            "卫浴",
        ],
        "tv": ["电视", "tv", "monitor", "screen", "屏幕", "显示器", "彩电", "投影"],
        "laptop": ["笔记本", "电脑", "computer", "PC", "笔电", "macbook", "手提电脑"],
        "mouse": ["鼠标", "mouse", "滑鼠"],
        "keyboard": ["键盘", "keyboard", "键盤"],
        "cell phone": ["手机", "phone", "电话", "手提电话", "智能机", "iphone", "华为"],
        "microwave": ["微波炉", "microwave", "热饭", "烤箱"],
        "refrigerator": ["冰箱", "fridge", "冰柜", "冷藏"],
        "clock": ["时钟", "clock", "钟表", "闹钟", "挂钟"],
        "potted plant": ["花", "plant", "草", "盆栽", "植物", "绿植", "花草", "树"],
        "book": ["书", "book", "本子", "教材", "课本", "读物", "杂志", "小说"],
        "trash can": ["垃圾桶", "trash", "废物箱", "纸篓"],
        "lamp": ["灯", "lamp", "台灯", "路灯", "照明", "光"],
        "door": ["门", "door", "门口", "大门", "房门"],
        "wardrobe": [
            "衣柜",
            "wardrobe",
            "closet",
            "柜子",
            "大衣柜",
            "储物柜",
            "衣服",
            "挂衣",
        ],
    }

    found_targets = []
    for yolo_class, keywords in mapping.items():
        for keyword in keywords:
            if keyword in text:
                if yolo_class not in found_targets:
                    found_targets.append(yolo_class)
    return found_targets


# 임시 반영구 DB (비효율)
class ReviewDB:
    def __init__(self, load=str(settings.BACKEND_HOME / "review.json")):
        self._load = load
        with open(self._load, "r") as j:
            self._reviews = json.load(j)

    def save(self):
        with open(self._load, "w") as j:
            j.write(json.dumps(self._reviews))

    def new(self, review):
        self._reviews.insert(0, review)
        self.save()

    def get(self) -> list[dict]:
        return self._reviews

json_db = ReviewDB()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/panorama")
def panorama():
    return render_template("panorama.html")


# ----------------------------------------------------
# [핵심 변경] 리뷰 제출 및 분석 라우트
# ----------------------------------------------------
@app.route("/submit_review", methods=["POST"])
def submit_review():
    full_review = request.form.get("review_content", "")
    rating = int(request.form.get("rating", 3))

    # 1. IP 기반 닉네임 생성
    user_ip = request.remote_addr
    user_name = get_nickname(user_ip)

    # 2. 감정 분석: 문장을 쪼개지 않고 '통째로' 분석합니다.
    # 이렇게 해야 "床垫简直是yyds!" 처럼 문맥이 필요한 문장을 정확히 인식합니다.
    main_emoji, _ = analyze_sentiment(full_review)

    # 3. 사물 인식: 전체 텍스트에서 언급된 모든 가구를 찾습니다.
    # (re.split을 쓰지 않으므로 문장이 잘려서 가구를 못 찾는 일도 방지됩니다)
    found_targets = detect_target_object(full_review)

    sub_reviews = []

    # 4. 찾은 모든 가구에 대해 '전체 문장의 감정'을 적용합니다.
    for target in found_targets:
        sub_reviews.append(
            {
                "target": target,
                "segment_text": full_review,  # 부분 문장이 아닌 전체 문장을 연결
                "emoji": main_emoji,  # 전체 문장의 감정 이모티콘
            }
        )

    # 5. DB 저장
    new_review = {
        "name": user_name,
        "rating": rating,
        "text": full_review,  # 원본 전체 글
        "sub_reviews": sub_reviews,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # 최신 리뷰가 위로 오도록 저장
    json_db.new(new_review)
    return redirect(url_for("index"))


@app.route("/api/reviews", methods=["GET"])
def api_reviews():
    return jsonify(json_db.get())


if __name__ == "__main__":
    app.run(debug=False, port=5000)
