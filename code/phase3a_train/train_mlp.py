"""2a의 선택된 얼굴로 CLIP 특징·3-class MLP를 학습하고 원래 예측을 평가한다."""

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("no_glasses", "regular_glasses", "smart_glasses")
BACKBONE = "openai/clip-vit-base-patch32"
FACE_THRESHOLD = 0.5
GLASSES_THRESHOLD = 0.5
FACE_EXPAND = 1.8
BATCH_SIZE = 64
HIDDEN_DIM = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
TEST_RATIO = 0.2
SEED = 42


def build_samples(data):
    annotations = {ann["id"]: ann for ann in data["annotations"]}
    glasses_faces = {ann["face_id"] for ann in data["annotations"] if ann["category_id"] == 2}
    samples, excluded = [], []
    for image in data["images"]:
        if "selected_face_id" not in image:
            raise ValueError("갱신한 phase2a_extract/extract_bbox.py로 JSON을 다시 생성하세요.")
        face_id = image["selected_face_id"]
        if face_id is None:
            excluded.append({"image_id": image["id"], "file_name": image["file_name"],
                             "reason": image.get("selection_status", "face_review")})
            continue
        face = annotations.get(face_id)
        if face is None or face["category_id"] != 1 or face["image_id"] != image["id"]:
            raise ValueError(f"선택한 얼굴 주석을 확인하세요: {image['file_name']}")
        if face.get("score", 0) <= FACE_THRESHOLD:
            excluded.append({"image_id": image["id"], "file_name": image["file_name"], "reason": "face_score"})
            continue
        if "glasses_probability" not in face or "glasses_status" not in face:
            raise ValueError("2a를 다시 실행해 얼굴별 안경 확률·상태를 저장하세요.")
        label = CLASSES.index(image["file_name"].split("/")[0])
        samples.append({"image_id": image["id"], "file_name": image["file_name"], "face_id": face_id,
                        "bbox": face["bbox"], "label": label, "face_score": face["score"],
                        "glasses_probability": face["glasses_probability"], "glasses_status": face["glasses_status"],
                        "glasses_bbox_present": face_id in glasses_faces})
    return samples, excluded


def stratified_split(samples):
    generator = random.Random(SEED)
    train, test = [], []
    for label in range(len(CLASSES)):
        ids = [i for i, sample in enumerate(samples) if sample["label"] == label]
        if len(ids) < 2:
            raise ValueError(f"{CLASSES[label]}: 선택된 얼굴 이미지가 최소 2장 필요합니다.")
        generator.shuffle(ids)
        split = max(1, min(len(ids) - 1, int(len(ids) * (1 - TEST_RATIO))))
        train.extend(ids[:split])
        test.extend(ids[split:])
    return train, test


def crop_face(image, box):
    x, y, w, h = box
    margin = (FACE_EXPAND - 1) / 2
    return image.crop((max(0, math.floor(x - w * margin)), max(0, math.floor(y - h * margin)),
                       min(image.width, math.ceil(x + w * (1 + margin))), min(image.height, math.ceil(y + h * (1 + margin)))))


def extract_features(root, samples, device):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import CLIPModel, CLIPProcessor

    cache = str(root / "ckpt/pretrained/clip")
    processor = CLIPProcessor.from_pretrained(BACKBONE, cache_dir=cache, local_files_only=True)
    clip = CLIPModel.from_pretrained(BACKBONE, cache_dir=cache, local_files_only=True).to(device).eval()
    clip.requires_grad_(False)

    class FaceDataset(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            sample = samples[index]
            with Image.open(root / "data/images" / sample["file_name"]) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            return processor(images=crop_face(image, sample["bbox"]), return_tensors="pt")["pixel_values"][0]

    loader = DataLoader(FaceDataset(), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device == "cuda")
    features = []
    with torch.inference_mode():
        for pixels in loader:
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                feature = clip.visual_projection(clip.vision_model(pixel_values=pixels.to(device, non_blocking=True)).pooler_output)
            features.append(torch.nn.functional.normalize(feature.float(), dim=-1))
    return torch.cat(features)


def evaluate_predictions(samples, test_ids, probabilities):
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(test_ids), len(CLASSES)) or not np.isfinite(probabilities).all():
        raise ValueError("MLP 예측 확률의 크기와 값을 확인하세요.")
    records, cm = [], np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for index, probability in zip(test_ids, probabilities):
        sample = samples[index]
        label, prediction = sample["label"], int(probability.argmax())
        cm[label, prediction] += 1
        wearing = sample["glasses_probability"] >= GLASSES_THRESHOLD
        conflict = (prediction > 0) != wearing
        missing_box = wearing and not sample["glasses_bbox_present"]
        reasons = []
        if conflict:
            reasons.append("presence_conflict")
        if missing_box:
            reasons.append("glasses_bbox_missing")
        records.append({
            "image_id": sample["image_id"], "file_name": sample["file_name"], "face_id": sample["face_id"],
            "actual": CLASSES[label], "prediction": CLASSES[prediction],
            "prediction_probability": float(probability[prediction]), "label_probability": float(probability[label]),
            "p_no_glasses": float(probability[0]), "p_regular_glasses": float(probability[1]),
            "p_smart_glasses": float(probability[2]), "face_score": sample["face_score"],
            "glasses_probability": sample["glasses_probability"], "glasses_status": sample["glasses_status"],
            "glasses_bbox_present": sample["glasses_bbox_present"],
            "final_class": "review" if conflict else CLASSES[prediction], "reason": "|".join(reasons),
        })
    metrics = {"test_size": len(records), "test_accuracy": float(cm.trace() / max(1, cm.sum())),
               "confusion_matrix": cm.tolist(),
               "presence_conflicts": sum("presence_conflict" in row["reason"] for row in records),
               "missing_glasses_bbox": sum("glasses_bbox_missing" in row["reason"] for row in records),
               "review_count": sum(row["final_class"] == "review" for row in records)}
    return metrics, records


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion(cm, path):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted", ylabel="Actual", title="3-class MLP test (raw)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_error(ax, root, row, annotations, glasses):
    from matplotlib.patches import Rectangle

    with Image.open(root / "data/images" / row["file_name"]) as original:
        ax.imshow(ImageOps.exif_transpose(original).convert("RGB"))
    face = annotations[row["face_id"]]
    for annotation in [face] + glasses.get(face["id"], []):
        x, y, w, h = annotation["bbox"]
        color = "lime" if annotation["category_id"] == 1 else "red"
        text = f"face={annotation['score']:.3f}" if annotation["category_id"] == 1 else f"P(glasses)={annotation['score']:.3f}"
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2))
        ax.text(x + 2, y + 2, text, fontsize=8, color=color, va="top",
                bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=1))
    ax.set_title(f"MLP: {row['actual']} → {row['prediction']}\n"
                 f"P(pred)={row['prediction_probability']:.3f} | P(label)={row['label_probability']:.3f}\n"
                 f"{Path(row['file_name']).name}\nfinal={row['final_class']} {row['reason']}", fontsize=9)
    ax.axis("off")


def plot_errors(root, data, records, output):
    import matplotlib.pyplot as plt

    annotations = {ann["id"]: ann for ann in data["annotations"]}
    glasses = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation["category_id"] == 2:
            glasses[annotation["face_id"]].append(annotation)
    wrong = [row for row in records if row["actual"] != row["prediction"]]
    for path in output.glob("misclassified_*.png"):
        path.unlink()
    paths = []
    for start in range(0, len(wrong), 15):
        batch = wrong[start:start + 15]
        fig, axes = plt.subplots(math.ceil(len(batch) / 5), 5, figsize=(20, math.ceil(len(batch) / 5) * 4), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, row in zip(axes.flat, batch):
            draw_error(ax, root, row, annotations, glasses)
        fig.tight_layout()
        path = output / f"misclassified_{start // 15 + 1:03d}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path.name)
    return paths


def main():
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = json.loads((ROOT / "data/annotations/face_glasses.json").read_text(encoding="utf-8"))
    samples, excluded = build_samples(data)
    output = ROOT / "results/phase3a_train"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "excluded_images.csv", excluded, ["image_id", "file_name", "reason"])
    train_ids, test_ids = stratified_split(samples)
    print("클래스별 이미지 수:", {label: sum(s["label"] == i for s in samples) for i, label in enumerate(CLASSES)})
    print(f"device={device} | train={len(train_ids)} | test={len(test_ids)} | face_review={len(excluded)}")
    features = extract_features(ROOT, samples, device)
    labels = torch.tensor([sample["label"] for sample in samples], dtype=torch.long, device=device)
    train_index, test_index = torch.tensor(train_ids, device=device), torch.tensor(test_ids, device=device)
    x_train, y_train = features[train_index], labels[train_index]
    mlp = nn.Sequential(nn.Linear(features.shape[1], HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, len(CLASSES))).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    mlp.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(mlp(x_train), y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0 or epoch + 1 == EPOCHS:
            print(f"{epoch + 1:03d}/{EPOCHS} | train_loss={loss.item():.4f}")
    mlp.eval()
    with torch.inference_mode():
        probabilities = mlp(features[test_index]).float().softmax(1).cpu().numpy()
    metrics, records = evaluate_predictions(samples, test_ids, probabilities)
    metrics.update({"classes": list(CLASSES), "train_size": len(train_ids), "source_images": len(data["images"]), "eligible_images": len(samples),
                    "excluded_images": len(excluded), "seed": SEED})
    checkpoint = ROOT / "ckpt/phase3a_train/clip_mlp_3class.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": mlp.state_dict(), "classes": list(CLASSES), "backbone": BACKBONE,
                "feature_dim": features.shape[1], "hidden_dim": HIDDEN_DIM, "normalize_features": True,
                "face_expand": FACE_EXPAND, "face_threshold": FACE_THRESHOLD, "seed": SEED}, checkpoint)
    write_csv(output / "predictions.csv", records, list(records[0]))
    plot_confusion(metrics["confusion_matrix"], output / "confusion_matrix.png")
    metrics["misclassified_plots"] = plot_errors(ROOT, data, records, output)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"test_acc={metrics['test_accuracy']:.3f} | conflicts={metrics['presence_conflicts']} | "
          f"missing_bbox={metrics['missing_glasses_bbox']} | review={metrics['review_count']}")
    print("결과:", output)
    print("체크포인트:", checkpoint)


if __name__ == "__main__":
    main()
