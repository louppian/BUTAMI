"""일반 안경 착용·미착용·스마트글래스 착용 사람 이미지를 각 500장 수집한다."""

import csv
import math
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import imagehash
import requests
from ddgs import DDGS
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
TARGET = 500
BATCH_SIZE = 32
WORKERS = 8
CLIP_THRESHOLD = 0.7
CLASSES = ("no_glasses", "regular_glasses", "smart_glasses")
QUERIES = {
    "no_glasses": [
        "portrait person without glasses", "man face without eyeglasses",
        "woman face without eyeglasses", "person outdoors without glasses portrait",
        "person indoors without glasses portrait", "side profile person without glasses",
    ],
    "regular_glasses": [
        "person wearing prescription eyeglasses portrait", "man wearing eyeglasses portrait",
        "woman wearing eyeglasses portrait", "person wearing metal frame eyeglasses",
        "person wearing acetate eyeglasses", "person wearing rimless eyeglasses",
        "person wearing eyeglasses side view", "person wearing round eyeglasses",
    ],
    "smart_glasses": [
        "person wearing Ray-Ban Meta Wayfarer", "person wearing Ray-Ban Meta Headliner",
        "person wearing Ray-Ban Stories", "person wearing XREAL Air",
        "person wearing XREAL One", "person wearing VITURE smart glasses",
        "person wearing Rokid Max", "person wearing Bose Frames",
        "person wearing Amazon Echo Frames", "person wearing Solos AirGo",
        "person wearing Even Realities G1", "person wearing Huawei Eyewear",
    ],
}
# 앞의 세 문장 순서는 CLASSES와 같다. 나머지는 수집 제외 후보다.
DESCRIPTIONS = [
    "A photo of one person without any glasses on their face.",
    "A photo of one person wearing ordinary prescription eyeglasses.",
    "A photo of one person wearing electronic smart glasses.",
    "A product photo of glasses alone, without a person wearing them.",
    "A photo of multiple people together.",
    "A collage with multiple pictures in separate panels.",
    "An advertisement with text, diagrams, and multiple product views.",
    "A close-up of glasses or accessories without a visible face.",
    "A photo of an unrelated object without a person.",
]
FIELDS = ["path", "label", "query", "image_url", "source_url", "title", "phash", "filter_score", "collection"]
local = threading.local()


def download(result):
    try:
        if not hasattr(local, "session"):
            local.session = requests.Session()
            local.session.headers["User-Agent"] = "Mozilla/5.0"
        with local.session.get(result["image"], timeout=(5, 15), stream=True) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > 20 * 1024 * 1024:
                    return None
        with Image.open(BytesIO(content)) as original:
            if min(original.size) < 256 or original.width * original.height > 20_000_000:
                return None
            image = ImageOps.exif_transpose(original).convert("RGB")
        return result, image, imagehash.phash(image)
    except (requests.RequestException, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return None


def load_clip(root):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = root / "ckpt/pretrained/clip"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(cache))
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(cache)).to(device).eval()
    model.requires_grad_(False)
    text = processor(text=DESCRIPTIONS, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        features = model.text_projection(model.text_model(**text).pooler_output)
        features = features / features.norm(dim=-1, keepdim=True)
    return processor, model, features, device


def score_images(images, processor, model, text_features, device):
    import torch

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        features = model.visual_projection(model.vision_model(**inputs).pooler_output)
        features = features / features.norm(dim=-1, keepdim=True)
        return (model.logit_scale.exp() * features @ text_features.T).float().softmax(dim=-1).cpu()


def main():
    root = ROOT
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    metadata = root / "results/phase1a_images/metadata.csv"
    seen, hashes, counts = set(), [], dict.fromkeys(CLASSES, 0)
    records = {}
    if metadata.exists():
        with metadata.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != FIELDS:
                raise ValueError("기존 제품 수집 이미지와 metadata.csv를 별도로 보관한 뒤 사람 이미지 수집을 실행하세요.")
            for row in reader:
                if row["collection"] != "people" or row["label"] not in CLASSES:
                    raise ValueError("metadata.csv의 수집 종류와 클래스 정보를 확인하세요.")
                records[row["path"]] = row
                if (root / row["path"]).is_file():
                    seen.add(row["image_url"])

    # 제품 수집 파일이 사람 이미지의 목표 수량에 섞이는 것을 막는다.
    for label in CLASSES:
        folder = root / "data/images" / label
        folder.mkdir(parents=True, exist_ok=True)
        for path in sorted(folder.glob("*.jpg")):
            if path.relative_to(root).as_posix() not in records:
                raise ValueError(f"사람 수집 기록이 없는 이미지입니다. 기존 제품 이미지를 별도로 보관하세요: {path}")
            with Image.open(path) as image:
                hashes.append(imagehash.phash(image))
            counts[label] += 1

    if all(count >= TARGET for count in counts.values()):
        print("수집 완료:", counts)
        return
    processor, model, text_features, device = load_clip(root)
    search = DDGS(timeout=20)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} | batch_size={BATCH_SIZE} | download_workers={WORKERS}")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, metadata.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if file.tell() == 0:
            writer.writeheader()
            file.flush()
        for label_id, label in enumerate(CLASSES):
            folder, terms = root / "data/images" / label, QUERIES[label]
            attempted = set()
            number = max((int(path.stem.rsplit("_", 1)[1]) for path in folder.glob(f"{label}_*.jpg")), default=0)
            per_query = dict.fromkeys(terms, 0)
            quota = math.ceil(TARGET / len(terms)) + 10
            for page in range(1, 6):
                if counts[label] >= TARGET:
                    break
                for query in terms:
                    if counts[label] >= TARGET:
                        break
                    if per_query[query] >= quota:
                        continue
                    try:
                        results = search.images(query + " -collage -collection", max_results=100, page=page)
                    except Exception as error:
                        print(f"검색 실패: {query} / page={page}: {error}")
                        continue
                    candidates = []
                    for result in results:
                        url = result.get("image")
                        if url and url not in seen and url not in attempted:
                            attempted.add(url)
                            candidates.append(result)
                    for start in range(0, len(candidates), BATCH_SIZE):
                        if counts[label] >= TARGET or per_query[query] >= quota:
                            break
                        batch = []
                        for item in pool.map(download, candidates[start:start + BATCH_SIZE]):
                            if item is None:
                                continue
                            phash = item[2]
                            if any(phash - previous <= 4 for previous in hashes) or any(phash - previous[2] <= 4 for previous in batch):
                                continue
                            batch.append(item)
                        if not batch:
                            continue
                        scores = score_images([item[1] for item in batch], processor, model, text_features, device)
                        for (result, image, phash), score in zip(batch, scores):
                            if counts[label] >= TARGET or per_query[query] >= quota:
                                break
                            if score.argmax().item() != label_id or score[label_id].item() < CLIP_THRESHOLD:
                                continue
                            number += 1
                            path = folder / f"{label}_{number:06d}.jpg"
                            image.save(path, quality=95)
                            writer.writerow({
                                "path": path.relative_to(root).as_posix(), "label": label, "query": query,
                                "image_url": result["image"], "source_url": result.get("url", ""),
                                "title": result.get("title", ""), "phash": str(phash),
                                "filter_score": round(score[label_id].item(), 4), "collection": "people",
                            })
                            file.flush()
                            hashes.append(phash)
                            seen.add(result["image"])
                            counts[label] += 1
                            per_query[query] += 1
                        print(f"{label}: {counts[label]}/{TARGET} | batch={len(batch)} | query={query}")
    for label, count in counts.items():
        print(f"{label}: {count}/{TARGET}" + (" — 검색 범위 내 후보 부족 또는 검색 실패로 미달" if count < TARGET else " — 완료"))


if __name__ == "__main__":
    main()

