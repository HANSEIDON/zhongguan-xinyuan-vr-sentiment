import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

# [중요] 기존 코드에서 필요한 설정과 함수들을 빌려옵니다.
from bert_all_in_one import Config, preprocess_data, EmotionDataset, evaluate


def main():
    # 1. 설정 가져오기 (경로가 맞는지 확인하세요!)
    config = Config()

    # 2. 장치 설정 (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Testing on Device: {device}")

    # 3. 라벨 맵핑 복원하기 (훈련 때와 똑같은 순서여야 함)
    # 훈련 데이터를 로드해서 라벨 순서를 알아냅니다.
    print("🔄 Recovering Label Map from training data...")
    train_data = preprocess_data(config.TRAIN_FILE)
    all_labels = sorted(list(set([d["label"] for d in train_data])))
    label2id = {label: i for i, label in enumerate(all_labels)}
    # id2label = {i: label for label, i in label2id.items()} # 필요하면 사용

    print(f"🏷️ Labels: {label2id}")
    config.NUM_LABELS = len(all_labels)

    # 4. 테스트 데이터 준비
    # 기존 코드의 전처리 함수를 그대로 재사용합니다.
    print("📂 Loading Test Data...")
    test_data = preprocess_data(config.EVAL_FILE)

    # 토크나이저는 모델과 짝꿍이므로 다시 불러옵니다.
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)

    test_dataset = EmotionDataset(test_data, tokenizer, label2id, config.MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 5. 모델 뼈대 만들기 (빈 껍데기)
    print("🤖 Initializing Model Structure...")
    model = BertForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=config.NUM_LABELS
    )

    # 6. [핵심] 저장된 'best.pt' 가중치 불러오기
    model_path = "./best.pt"  # 파일 경로 확인!
    print(f"💾 Loading Weights from {model_path}...")

    # 만약 GPU에서 학습하고 CPU에서 돌린다면 map_location이 필요할 수 있음
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 7. 평가 실행 (시험 보기)
    print("\n🏁 Start Evaluation...")
    acc, f1 = evaluate(model, test_loader, device)

    # 8. 성적표 출력
    print("=" * 40)
    print("🏆 Final Test Result")
    print(f"✅ Accuracy : {acc:.4f}")
    print(f"✅ F1 Score : {f1:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
