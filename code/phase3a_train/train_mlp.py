"""메인 학습: 2a bbox 생성 → 고정 CLIP·3-class MLP → confusion matrix·가중치 저장."""

import json
import math
import random
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
            raise ValueError("갱신한 phase2a_extract/bbox/extract_bbox.py로 JSON을 다시 생성하세요.")
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


def main():
    import sys
    import cv2
    import torch
    from collections import Counter
    from torch.utils.data import Dataset, DataLoader
    from transformers import CLIPModel, CLIPProcessor

    sys.path.insert(0, str(ROOT / "code/phase2a_extract/bbox"))
    from extract_bbox import GlassesExtractor, extract, ensure_weights, YUNET_URL

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FACE_WEIGHTS = ensure_weights(ROOT / "ckpt/pretrained/face_detection_yunet_2023mar.onnx", YUNET_URL)
    MLP_WEIGHTS = ROOT / "ckpt/phase3a_train/clip_mlp_3class.pt"
    MLP_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    cv2.setNumThreads(1)
    BATCH_SIZE = 32
    face_detector = cv2.FaceDetectorYN.create(str(FACE_WEIGHTS), '', (640, 640), FACE_THRESHOLD, 0.3, 5000)
    glasses = GlassesExtractor(ROOT)
    data = extract(ROOT, face_detector, glasses)
    samples, excluded = build_samples(data)
    train_ids, test_ids = stratified_split(samples)
    print('학습 대상:', dict(Counter(CLASSES[s['label']] for s in samples)), '제외:', dict(Counter(s['reason'] for s in excluded)))
    processor = CLIPProcessor.from_pretrained(BACKBONE, cache_dir=str(ROOT / 'ckpt/pretrained/clip'), local_files_only=True)
    clip = CLIPModel.from_pretrained(BACKBONE, cache_dir=str(ROOT / 'ckpt/pretrained/clip'), local_files_only=True).to(DEVICE).eval()
    clip.requires_grad_(False)
    class Faces(Dataset):
        def __len__(self): return len(samples)
        def __getitem__(self, i):
            with Image.open(ROOT / 'data/images' / samples[i]['file_name']) as image:
                return processor(images=crop_face(ImageOps.exif_transpose(image).convert('RGB'), samples[i]['bbox']), return_tensors='pt')['pixel_values'][0]
    features = []
    with torch.inference_mode():
        for pixels in DataLoader(Faces(), batch_size=64, num_workers=0, pin_memory=DEVICE.type == 'cuda'):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
                feature = clip.visual_projection(clip.vision_model(pixel_values=pixels.to(DEVICE, non_blocking=True)).pooler_output)
            features.append(torch.nn.functional.normalize(feature.float(), dim=-1))
    features = torch.cat(features)
    labels = torch.tensor([s['label'] for s in samples], device=DEVICE)
    x_train, y_train = features[train_ids].clone(), labels[train_ids]
    mlp = torch.nn.Sequential(torch.nn.Linear(features.shape[1], HIDDEN_DIM), torch.nn.ReLU(), torch.nn.Linear(HIDDEN_DIM, 3)).to(DEVICE)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=LEARNING_RATE)
    mlp.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(mlp(x_train), y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0: print(f'epoch={epoch+1}/{EPOCHS} loss={loss.item():.4f}')
    mlp.eval()
    with torch.inference_mode():
        pred = mlp(features[test_ids]).argmax(1)
        cm = torch.bincount(labels[test_ids] * 3 + pred, minlength=9).reshape(3, 3).cpu()
    print('train/test:', len(train_ids), len(test_ids), 'accuracy:', cm.diag().sum().item()/len(test_ids), 'confusion:', cm.tolist())
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm.numpy(), cmap='Blues')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j].item()), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES, xlabel='Predicted', ylabel='Actual', title=f'Test accuracy: {cm.diag().sum().item()/len(test_ids):.2%}')
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right')
    fig.tight_layout()
    plt.show()

    torch.save({'state_dict': {k:v.detach().cpu() for k,v in mlp.state_dict().items()}, 'classes': list(CLASSES),
                'backbone': BACKBONE, 'feature_dim': features.shape[1], 'hidden_dim': HIDDEN_DIM, 'normalize_features': True,
                'face_expand': FACE_EXPAND, 'face_threshold': FACE_THRESHOLD, 'seed': SEED,
                'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'train_files': [samples[i]['file_name'] for i in train_ids],
                'test_files': [samples[i]['file_name'] for i in test_ids], 'confusion_matrix': cm.tolist()}, MLP_WEIGHTS)
    print("체크포인트:", MLP_WEIGHTS)


if __name__ == "__main__":
    main()
