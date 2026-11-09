"""현재 이미지의 얼굴 crop → 기존 CLIP·3-class MLP 은닉층 → t-SNE."""

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.manifold import TSNE
from transformers import CLIPModel, CLIPProcessor

import train_mlp as train

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/phase2a_extract/bbox"))
import extract_bbox as bbox

OUTPUT = ROOT / "results/phase3a_train/face_only"
BATCH_SIZE = 32


def main():
    cv2.setNumThreads(1)
    paths = [(path, label) for label in train.CLASSES for path in sorted((ROOT / "data/images" / label).glob("*.jpg"))]
    print("Image snapshot:", {label: sum(c == label for _, c in paths) for label in train.CLASSES}, flush=True)
    checkpoint = ROOT / "ckpt/phase3a_train/clip_mlp_3class.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["classes"] == list(train.CLASSES)
    train.FACE_EXPAND = state["face_expand"]
    bbox.FACE_THRESHOLD = state["face_threshold"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(state["backbone"], cache_dir=str(ROOT / "ckpt/pretrained/clip"), local_files_only=True)
    clip = CLIPModel.from_pretrained(state["backbone"], cache_dir=str(ROOT / "ckpt/pretrained/clip"), local_files_only=True).to(device).eval()
    mlp = torch.nn.Sequential(torch.nn.Linear(state["feature_dim"], state["hidden_dim"]), torch.nn.ReLU(), torch.nn.Linear(state["hidden_dim"], 3)).to(device).eval()
    mlp.load_state_dict(state["state_dict"])
    detector = cv2.FaceDetectorYN.create(str(ROOT / "ckpt/pretrained/face_detection_yunet_2023mar.onnx"), "", (640, 640), bbox.FACE_THRESHOLD, 0.3, 5000)
    crops, pending, records, excluded, features = [], [], [], [], []

    def flush():
        if not crops:
            return
        pixels = processor(images=crops, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            encoded = clip.visual_projection(clip.vision_model(pixel_values=pixels).pooler_output)
        encoded = encoded.float()
        if state["normalize_features"]:
            encoded = torch.nn.functional.normalize(encoded, dim=-1)
        with torch.inference_mode():
            hidden = mlp[:2](encoded)
            probabilities = mlp[2](hidden).softmax(1)
            output = torch.cat((hidden, probabilities), dim=1).cpu().numpy()
        features.append(output[:, :state["hidden_dim"]])
        for row, probability in zip(pending, output[:, state["hidden_dim"]:]):
            row.update(prediction=train.CLASSES[int(probability.argmax())], **{f"p_{label}": float(p) for label, p in zip(train.CLASSES, probability)})
            records.append(row)
        crops.clear()
        pending.clear()
        print(f"MLP features: {len(records)}", flush=True)

    for path, label in paths:
        try:
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
            faces = [{"id": i, "bbox": box, "score": score, "area": box[2] * box[3]} for i, (box, score) in enumerate(bbox.detect_faces(detector, image))]
            face_id, status = bbox.select_face(faces, image.width, image.height)
            if face_id is None:
                excluded.append({"path": path.relative_to(ROOT).as_posix(), "label": label, "reason": status})
                continue
            face = faces[face_id]
            crops.append(train.crop_face(image, face["bbox"]))
            pending.append({"path": path.relative_to(ROOT).as_posix(), "label": label, "face_score": face["score"], "face_bbox": json.dumps(face["bbox"])})
            if len(crops) == BATCH_SIZE:
                flush()
        except (OSError, ValueError, SyntaxError) as error:
            excluded.append({"path": path.relative_to(ROOT).as_posix(), "label": label, "reason": str(error)})
    flush()
    hidden = np.concatenate(features)
    assert np.isfinite(hidden).all() and len(hidden) == len(records)
    embedding = TSNE(n_components=2, perplexity=min(30, len(records) - 1), learning_rate="auto", init="pca", random_state=42, max_iter=1000).fit_transform(hidden)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for row, point in zip(records, embedding):
        row.update(tsne_x=float(point[0]), tsne_y=float(point[1]))
    for name, rows, fields in (("tsne_points.csv", records, list(records[0])), ("tsne_excluded.csv", excluded, ["path", "label", "reason"])):
        with (OUTPUT / name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    labels = np.array([row["label"] for row in records])
    for label, color in zip(train.CLASSES, ("#dc2626", "#eab308", "#2563eb")):
        mask = labels == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=27, c=color, alpha=0.8, edgecolors="white", linewidths=0.3, label=f"{label} (n={mask.sum()})")
    ax.set(title=f"3-class MLP hidden features ({state['hidden_dim']}-D) → t-SNE\nN={len(records)} | folder labels | face crop {train.FACE_EXPAND}x | seed=42", xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend(frameon=True)
    ax.grid(alpha=0.15)
    fig.savefig(OUTPUT / "tsne.png", dpi=180)
    plt.close(fig)
    summary = {"checkpoint": str(checkpoint), "snapshot": {label: sum(c == label for _, c in paths) for label in train.CLASSES}, "included": {label: int((labels == label).sum()) for label in train.CLASSES}, "excluded": len(excluded), "face_expand": train.FACE_EXPAND, "feature": "MLP ReLU hidden output", "feature_dim": state["hidden_dim"], "seed": 42, "perplexity": min(30, len(records) - 1)}
    (OUTPUT / "tsne_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
