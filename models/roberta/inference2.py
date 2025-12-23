import settings

import torch
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--input", type=str, help="input text")

    parser.add_argument("--device", default="cuda", type=str, help="cpu or cuda")

    parser.add_argument(
        "--model_name",
        default="hfl/chinese-roberta-wwm-ext",
        type=str,
        help="huggingface transformer model name",
    )
    parser.add_argument(
        "--model_path",
        default=str(settings.MODEL_HOME / "roberta/best.py"),
        type=str,
        help="model checkpoint path",
    )
    parser.add_argument(
        "--num_labels", default=6, type=int, help="number of labels for classification"
    )

    parser.add_argument(
        "--max_length", default=140, type=int, help="max sequence length for tokenizer"
    )

    args, unknown = parser.parse_known_args()

    if args.input is None:
        args.input = input("감정을 분류할 문장을 입력하세요: ")

    return args


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    )

    print(f"Loading checkpoint: {args.model_path} ...")
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint.get("state_dict", checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
    print(
        f"missing_keys: {missing_keys}\n"
        f"===================================================================\n"
    )
    print(
        f"unexpected_keys: {unexpected_keys}\n"
        f"===================================================================\n"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    encoded = tokenizer(
        args.input,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = encoded.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    model.to(device)
    model.eval()

    with torch.no_grad():
        if token_type_ids is not None:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            output = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = output.logits
    pred_id = int(torch.argmax(logits, dim=-1).cpu().item())

    label_map = {
        0: "happy",
        1: "angry",
        2: "sad",
        3: "fear",
        4: "surprise",
        5: "neutral",
    }
    pred_label = label_map.get(pred_id, f"unknown({pred_id})")

    print(f"\n입력 문장: {args.input}")
    print(f"예측 라벨: {pred_label} (id={pred_id})")


if __name__ == "__main__":
    main()
