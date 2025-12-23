import settings

import os
import json
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from harvesttext import HarvestText
import zhconv
from itertools import cycle


# ==========================================
# 1. 설정 및 하이퍼파라미터 (Configuration)
# ==========================================
class Config:
    # !!!파일 경로 수정해야함!!!!!
    TRAIN_FILE = str(settings.DATA_HOME / "train/usual_train.txt")
    EVAL_FILE  = str(settings.DATA_HOME / "eval/usual_eval_labeled.txt")

    # 모델 및 학습 설정
    MODEL_NAME = "bert-base-chinese"
    NUM_LABELS = 6  # (데이터 로딩 시 자동으로 업데이트됨)
    MAX_LEN = 128  # 문장 최대 길이
    BATCH_SIZE = 32  # 배치 크기 (메모리 에러나면 16으로 줄이기)
    EPOCHS = 4  # 학습 에폭 수
    LEARNING_RATE = 2e-5  # 학습률 (BERT 미세조정 표준값)

    # 장치 설정 (GPU 우선 사용)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAVE_PREFIX = settings.MODEL_HOME / "bert"  # 최고 성능 모델 저장 파일명


config = Config()


# ==========================================
# 2. 데이터 전처리 함수 (Preprocessing)
# ==========================================
def remove_url(src):
    """URL 링크 제거"""
    vTEXT = re.sub(
        r"(https|http)?:\\/\\/(\\w|\\.|\\/|\\?|\\=|\\&|\\%)*\\b",
        "",
        src,
        flags=re.MULTILINE,
    )
    return vTEXT


def preprocess_data(file_path):
    """파일을 읽어서 전처리(번체->간체, 노이즈 제거) 후 리스트로 반환"""
    print(f"📂 Loading data from: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 도구 초기화 (텍스트 정제용)
    ht = HarvestText()

    cleaned_data = []

    for item in tqdm(raw_data, desc="Preprocessing"):
        content = item.get("content", "")
        label = item.get("label", None)

        if not content:
            continue

        # [수정됨] (1) 번체 -> 간체 (zhconv 사용)
        # 'zh-cn' 옵션은 중국 본토 간체로 변환합니다.
        content = zhconv.convert(content, "zh-cn")

        # (2) 노이즈 및 URL 제거 (HarvestText 사용)
        content = ht.clean_text(content, emoji=False)
        content = remove_url(content)

        if not content.strip():
            continue

        # (3) 결과 저장
        clean_item = {"content": content}
        if label:
            clean_item["label"] = label

        cleaned_data.append(clean_item)

    print(f"✅ Processed {len(cleaned_data)} samples.")
    return cleaned_data


# ==========================================
# 3. 데이터셋 클래스 (Dataset)
# ==========================================
class EmotionDataset(Dataset):
    def __init__(self, data_list, tokenizer, label_map, max_len=128):
        self.data = data_list
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["content"]
        label_str = item.get("label", None)

        # 토큰화 (Text -> Input IDs)
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        result = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

        # 라벨을 숫자로 변환
        if label_str:
            result["labels"] = torch.tensor(self.label_map[label_str], dtype=torch.long)

        return result


# ==========================================
# 4. 학습 및 검증 함수 (Train & Eval)
# ==========================================
def train_epoch(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch + 1}")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({"loss": loss.item()})

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    model.eval()
    preds = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, prediction = torch.max(outputs.logits, dim=1)

            preds.extend(prediction.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="weighted")

    return acc, f1


# ==========================================
# 5. 시각화 함수 모음
# ==========================================
def get_all_predictions(model, data_loader, device):
    """시각화를 위해 모델의 모든 예측 결과(확률 포함) 추출"""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Visualizing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)  # 확률 계산
            _, preds = torch.max(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_learning_curve(history):
    """1. 학습 곡선 (Learning Curve)"""
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-o", label="Training Loss")
    plt.title("Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_f1"], "r-o", label="Validation F1")
    plt.title("Validation Weighted F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(config.SAVE_PREFIX / "learning_curve.png")
    print("📈 학습 곡선 저장 완료: learning_curve.png")


def plot_confusion_matrix_custom(y_true, y_pred, id2label):
    """2. 혼동 행렬 (Confusion Matrix)"""
    cm = confusion_matrix(y_true, y_pred)
    label_names = [id2label[i] for i in range(len(id2label))]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (BERT)")
    plt.savefig(config.SAVE_PREFIX / "confusion_matrix.png")
    print("🔥 혼동 행렬 저장 완료: confusion_matrix.png")


def plot_multiclass_roc(y_true, y_probs, num_classes, id2label):
    """3. 다중 클래스 ROC 곡선 (One-vs-Rest)"""
    # 정답 라벨을 원-핫 인코딩
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(
        ["blue", "red", "green", "orange", "purple", "cyan", "brown", "pink"]
    )

    for i, color in zip(range(num_classes), colors):
        label_name = id2label[i]
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{label_name} (AUC = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(config.SAVE_PREFIX / "roc_curve.png")
    print(" ROC 곡선 저장 완료: roc_curve.png")


# ==========================================
# 6. 메인 실행 블록 (Main Execution)
# ==========================================
if __name__ == "__main__":
    print(f"🚀 Using Device: {config.DEVICE}")

    # 1. 데이터 로드 및 전처리
    try:
        train_data = preprocess_data(config.TRAIN_FILE)
        eval_data = preprocess_data(config.EVAL_FILE)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        exit()

    # 2. 라벨 자동 감지 및 맵핑
    all_labels = sorted(list(set([d["label"] for d in train_data])))
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"🏷️ Detected Labels ({len(all_labels)}): {label2id}")
    config.NUM_LABELS = len(all_labels)

    # 3. 토크나이저 & 데이터셋 준비
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    train_dataset = EmotionDataset(train_data, tokenizer, label2id, config.MAX_LEN)
    eval_dataset = EmotionDataset(eval_data, tokenizer, label2id, config.MAX_LEN)

    # 4. 데이터로더
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 5. 모델 초기화
    print(f"🤖 Loading Model: {config.MODEL_NAME}...")
    model = BertForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=config.NUM_LABELS
    )
    model.to(config.DEVICE)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # 6. 학습 루프
    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = 0.0

    print("\n🏁 Start Training...")
    for epoch in range(config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, config.DEVICE, epoch)
        acc, f1 = evaluate(model, eval_loader, config.DEVICE)

        history["train_loss"].append(train_loss)
        history["val_acc"].append(acc)
        history["val_f1"].append(f1)

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}"
        )

        # 최고 기록 갱신 시 저장
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), config.SAVE_PREFIX / "best.pt")
            print(f"💾 Best Model Saved! (F1: {best_f1:.4f})")

        print("-" * 30)

    print(f"\n🎉 Training Complete. Best F1 Score: {best_f1:.4f}")

    # ==========================================
    # 7. 최종 시각화 (Best Model 사용)
    # ==========================================
    print("\n📊 Generating Visualizations...")

    # (1) 학습 곡선
    plot_learning_curve(history)

    # (2) 베스트 모델 로드 (학습된 가중치 불러오기)
    print("Loading best model for analysis...")
    model.load_state_dict(torch.load(config.SAVE_PREFIX / "best.pt"))
    model.to(config.DEVICE)

    # (3) 예측 데이터 추출
    y_true, y_pred, y_probs = get_all_predictions(model, eval_loader, config.DEVICE)

    # (4) 혼동 행렬 & ROC 곡선 그리기
    plot_confusion_matrix_custom(y_true, y_pred, id2label)
    plot_multiclass_roc(y_true, y_probs, config.NUM_LABELS, id2label)

    print("\n✨ All Done! 결과 이미지와 best.pt 파일을 확인하세요.")
