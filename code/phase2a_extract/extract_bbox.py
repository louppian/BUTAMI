"""사람 이미지별 얼굴·안경 bbox를 원본 좌표의 COCO JSON으로 저장한다."""

import json
import math
import shutil
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("no_glasses", "regular_glasses", "smart_glasses")
BATCH_SIZE = 32
FACE_THRESHOLD = 0.5
GLASSES_THRESHOLD = 0.5
FACE_EXPAND = 1.8
FACE_MAX_SIZE = 640
YUNET_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
CLASSIFIER_URL = "https://github.com/mantasu/glasses-detector/releases/download/v1.0.0/classification_anyglasses_shufflenet_v2_x1_0.pth"
GLASSES_URL = "https://github.com/mantasu/glasses-detector/releases/download/v1.0.0/segmentation_full_lraspp_mobilenet_v3_large.pth"


def ensure_weights(path, url):
    if path.is_file():
        with path.open("rb") as file:
            if path.stat().st_size > 100_000 and not file.read(128).startswith(b"version https://git-lfs"):
                return path
        raise ValueError(f"가중치 파일을 확인하세요: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as file:
            shutil.copyfileobj(response, file)
        with temporary.open("rb") as file:
            if temporary.stat().st_size <= 100_000 or file.read(128).startswith(b"version https://git-lfs"):
                raise ValueError(f"가중치 다운로드 결과를 확인하세요: {url}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def clip_box(box, width, height):
    x, y, w, h = box
    if not all(math.isfinite(float(v)) for v in box) or w <= 0 or h <= 0:
        return None
    x0, y0 = max(0, min(width, math.floor(x))), max(0, min(height, math.floor(y)))
    x1, y1 = max(0, min(width, math.ceil(x + w))), max(0, min(height, math.ceil(y + h)))
    return [x0, y0, x1 - x0, y1 - y0] if x1 > x0 and y1 > y0 else None


def expand_box(box, width, height):
    x, y, w, h = box
    return clip_box([x - w * (FACE_EXPAND - 1) / 2, y - h * (FACE_EXPAND - 1) / 2, w * FACE_EXPAND, h * FACE_EXPAND], width, height)


def detect_faces(detector, image):
    import cv2

    width, height = image.size
    scale = min(1, FACE_MAX_SIZE / max(width, height))
    size = max(1, round(width * scale)), max(1, round(height * scale))
    bgr = np.asarray(image)[:, :, ::-1].copy()
    if size != image.size:
        bgr = cv2.resize(bgr, size)
    detector.setInputSize(size)
    _, faces = detector.detect(bgr)
    if faces is None:
        return []
    boxes = []
    for face in faces:
        score = float(face[-1])
        if not math.isfinite(score) or not FACE_THRESHOLD <= score <= 1:
            continue
        box = clip_box([face[0] * width / size[0], face[1] * height / size[1], face[2] * width / size[0], face[3] * height / size[1]], width, height)
        if box is not None:
            boxes.append((box, score))
    return boxes


class GlassesExtractor:
    def __init__(self, root):
        import torch

        from torchvision.models import shufflenet_v2_x1_0
        from torchvision.models.segmentation import lraspp_mobilenet_v3_large

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        weights = ensure_weights(root / "ckpt/pretrained/segmentation_full_lraspp_mobilenet_v3_large.pth", GLASSES_URL)
        self.model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=1)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)
        weights = ensure_weights(root / "ckpt/pretrained/torch/checkpoints/classification_anyglasses_shufflenet_v2_x1_0.pth", CLASSIFIER_URL)
        self.classifier = shufflenet_v2_x1_0(weights=None)
        self.classifier.fc = torch.nn.Linear(1024, 1)
        self.classifier.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.classifier = self.classifier.to(self.device).eval()
        self.classifier.requires_grad_(False)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        self.positions = torch.arange(256, device=self.device)

    def __call__(self, images):
        import torch

        pixels = np.stack([np.asarray(image.resize((256, 256))) for image in images]).transpose(0, 3, 1, 2).copy()
        with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            pixels = torch.from_numpy(pixels).to(self.device, dtype=torch.float32).div_(255)
            pixels = (pixels - self.mean) / self.std
            probabilities = self.classifier(pixels).float().sigmoid().flatten()
            masks = self.model(pixels)["out"][:, 0] > 0
            rows, columns = masks.any(dim=2), masks.any(dim=1)
            # 마스크 대신 bbox 네 좌표와 착용 확률만 배치당 한 번 CPU로 전달한다.
            return torch.stack((
                torch.where(columns, self.positions, 256).amin(1),
                torch.where(rows, self.positions, 256).amin(1),
                torch.where(columns, self.positions + 1, 0).amax(1),
                torch.where(rows, self.positions + 1, 0).amax(1),
                probabilities,
            ), dim=1).cpu().tolist()


def project_box(bounds, crop):
    x0, y0, x1, y1 = bounds
    if x1 <= x0 or y1 <= y0:
        return None
    x, y, w, h = crop
    return clip_box([x + x0 * w / 256, y + y0 * h / 256, (x1 - x0) * w / 256, (y1 - y0) * h / 256], x + w, y + h)


def extract(root, detector, segmenter):
    data = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "face"}, {"id": 2, "name": "glasses"}]}
    paths = [path for label in CLASSES for path in sorted((root / "data/images" / label).glob("*.jpg"))]
    if not paths:
        raise ValueError(f"입력 이미지를 확인하세요: {root / 'data/images'}")
    pending = []

    def annotate(image_id, category, box, score, face_id=None):
        annotation = {"id": len(data["annotations"]) + 1, "image_id": image_id, "category_id": category,
                      "bbox": box, "area": box[2] * box[3], "iscrowd": 0, "score": score}
        if face_id is not None:
            annotation["face_id"] = face_id
        data["annotations"].append(annotation)
        return annotation

    def flush():
        bounds = segmenter([item[3] for item in pending])
        if len(bounds) != len(pending):
            raise ValueError("안경 추출 결과와 입력 이미지 수가 다릅니다.")
        for (image_id, face, crop, _), result in zip(pending, bounds):
            probability = float(result[4])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("안경 착용 확률을 확인하세요.")
            face["glasses_probability"] = probability
            if probability < GLASSES_THRESHOLD:
                continue
            box = project_box(result[:4], crop)
            if box is not None:
                annotate(image_id, 2, box, probability, face["id"])
        pending.clear()

    for image_id, path in enumerate(paths, 1):
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        width, height = image.size
        data["images"].append({"id": image_id, "file_name": path.relative_to(root / "data/images").as_posix(), "width": width, "height": height})
        for box, score in detect_faces(detector, image):
            face = annotate(image_id, 1, box, score)
            crop = expand_box(box, width, height)
            x, y, w, h = crop
            pending.append((image_id, face, crop, image.crop((x, y, x + w, y + h))))
            if len(pending) == BATCH_SIZE:
                flush()
        if image_id % 100 == 0:
            print(f"bbox 추출: {image_id}/{len(paths)}")
    if pending:
        flush()
    output = root / "data/annotations/face_glasses.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"저장: {output} | images={len(paths)} | faces={sum(a['category_id'] == 1 for a in data['annotations'])} | glasses={sum(a['category_id'] == 2 for a in data['annotations'])}")
    return data


def main():
    import cv2

    weights = ensure_weights(ROOT / "ckpt/pretrained/face_detection_yunet_2023mar.onnx", YUNET_URL)
    detector = cv2.FaceDetectorYN.create(str(weights), "", (FACE_MAX_SIZE, FACE_MAX_SIZE), FACE_THRESHOLD, 0.3, 5000)
    segmenter = GlassesExtractor(ROOT)
    print(f"YuNet: OpenCV CPU | GlassesSegmenter: {segmenter.device} | batch_size={BATCH_SIZE}")
    extract(ROOT, detector, segmenter)


if __name__ == "__main__":
    main()
