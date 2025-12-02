from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import json
import re
from datetime import datetime

# 임시 데이터베이스 (서버 끄면 초기화됨)
# target: 이 리뷰가 어떤 물건(영어)에 대한 것인지 저장
db_reviews = []

app = Flask(__name__)
app.template_folder = os.path.dirname(os.path.abspath(__file__))
app.static_url_path = "/static"
app.static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


# 1. 중국어 -> YOLO 영어 클래스 매핑 함수
def detect_target_object(text):
    text = text.lower()
    mapping = {
        # --- 가구 (Furniture) ---
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
        # --- 전자제품 (Electronics) ---
        "tv": ["电视", "tv", "monitor", "screen", "屏幕", "显示器", "彩电", "投影"],
        "laptop": ["笔记本", "电脑", "computer", "PC", "笔电", "macbook", "手提电脑"],
        "mouse": ["鼠标", "mouse", "滑鼠"],
        "keyboard": ["键盘", "keyboard", "键盤"],
        "cell phone": ["手机", "phone", "电话", "手提电话", "智能机", "iphone", "华为"],
        "microwave": ["微波炉", "microwave", "热饭", "烤箱"],
        "oven": ["烤箱", "oven", "炉子"],
        "toaster": ["面包机", "toaster", "烤面包"],
        "refrigerator": ["冰箱", "fridge", "冰柜", "冷藏"],
        "clock": ["时钟", "clock", "钟表", "闹钟", "挂钟"],
        # --- 소품/기타 (Accessories & Others) ---
        "potted plant": ["花", "plant", "草", "盆栽", "植物", "绿植", "花草", "树"],
        "vase": ["花瓶", "vase", "瓶子"],
        "book": ["书", "book", "本子", "教材", "课本", "读物", "杂志", "小说"],
        "backpack": ["书包", "bag", "背包", "双肩包", "行李", "包包"],
        "handbag": ["手提包", "purse", "挎包", "皮包"],
        "suitcase": ["行李箱", "suitcase", "拉杆箱", "箱子"],
        "umbrella": ["雨伞", "umbrella", "伞", "阳伞"],
        "bottle": ["瓶子", "bottle", "水瓶", "饮料", "矿泉水"],
        "cup": ["杯子", "cup", "水杯", "茶杯", "咖啡杯"],
        "bowl": ["碗", "bowl", "饭碗"],
        "trash can": ["垃圾桶", "trash", "废物箱", "纸篓"],
        "lamp": ["灯", "lamp", "台灯", "路灯", "照明", "光"],
        "mirror": ["镜子", "mirror", "穿衣镜", "梳妆镜"],
        "window": ["窗户", "window", "窗帘", "玻璃", "窗台"],
        "door": ["门", "door", "门口", "大门", "房门"],
    }
    found_targets = []
    for yolo_class, keywords in mapping.items():
        for keyword in keywords:
            if keyword in text:
                # 중복 방지를 위해 리스트에 없으면 추가
                if yolo_class not in found_targets:
                    found_targets.append(yolo_class)

    return found_targets  # 이제 리스트를 반환합니다! (예: ['couch', 'chair'])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/panorama")
def panorama():
    return render_template("panorama.html")


@app.route('/submit_review', methods=['POST'])
def submit_review():
    full_review = request.form.get('review_content')
    rating = int(request.form.get('rating'))
    user_name = '用户' + datetime.now().strftime("%H%M")

    # 1. 문장 쪼개기
    segments = re.split(r'[，,。!！?？\n]+', full_review)
    
    # 2. 쪼개진 조각들 분석해서 'sub_reviews' 리스트 만들기
    sub_reviews = [] # 세부 내용 담을 바구니
    
    for segment in segments:
        segment = segment.strip()
        if not segment: continue

        # 이 조각이 어떤 물건인지 확인
        targets = detect_target_object(segment) # 리스트 반환 (예: ['couch'])
        
        if targets:
            for target in targets:
                # 조각 정보를 담음 (예: 소파에 대한 칭찬)
                sub_reviews.append({
                    'target': target,
                    'segment_text': segment
                })

    # 3. DB에는 "하나의 리뷰"로 저장하되, 세부 정보를 안에 포함시킴
    new_review = {
        'name': user_name,
        'rating': rating,
        'text': full_review,   # 메인 화면용 전체 문장
        'sub_reviews': sub_reviews # 파노라마용 세부 조각 리스트
    }
    
    db_reviews.insert(0, new_review)

    return redirect(url_for('index'))
    
@app.route("/api/reviews", methods=["GET"])
def api_reviews():
    return jsonify(db_reviews)


if __name__ == "__main__":
    app.run(debug=False, port=5000)

