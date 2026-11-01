"""일반 안경·스마트글래스 제품 이미지 수집. CLI 없이 파일을 실행한다."""

import csv
import math
import time
import warnings
from io import BytesIO
from pathlib import Path

import imagehash
import requests
import torch
from ddgs import DDGS
from PIL import Image, ImageOps
from transformers import CLIPModel, CLIPProcessor


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    target = 500
    queries = {
        "regular_glasses": [
            "single pair prescription eyeglasses product",
            "metal frame prescription eyeglasses product",
            "acetate frame prescription eyeglasses product",
            "rimless prescription eyeglasses product",
            "round prescription eyeglasses product",
            "rectangular prescription eyeglasses product",
            "titanium prescription eyeglasses product",
            "browline prescription eyeglasses product",
            "clear frame prescription eyeglasses product",
            "half rim prescription eyeglasses product",
        ],
        "smart_glasses": [
            "Ray-Ban Meta Wayfarer smart glasses product",
            "Ray-Ban Meta Headliner smart glasses product",
            "Ray-Ban Stories smart glasses product",
            "XREAL Air smart glasses product",
            "XREAL One smart glasses product",
            "VITURE smart glasses product",
            "Rokid Max smart glasses product",
            "Bose Frames smart glasses product",
            "Amazon Echo Frames smart glasses product",
            "Solos AirGo smart glasses product",
            "Even Realities G1 smart glasses product",
            "Huawei Eyewear smart glasses product",
        ],
    }
    descriptions = [
        "A product photo of exactly one complete pair of eyeglasses, shown once.",
        "A photo of multiple pairs of eyeglasses together.",
        "A collage showing glasses from multiple angles in separate panels.",
        "An advertisement with diagrams, text, and multiple product views.",
        "A close-up showing only part of eyeglasses.",
        "A person wearing glasses.",
        "A product kit showing glasses with cases and accessories.",
        "A photo of an unrelated object without any eyeglasses.",
    ]
    (root / "results/phase1a_images").mkdir(parents=True, exist_ok=True)
    metadata = root / "results/phase1a_images/metadata.csv"
    fields = ["path", "label", "query", "image_url", "source_url", "title", "phash", "filter_score"]
    seen = set()
    if metadata.exists():
        with metadata.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                if (root / row["path"]).is_file():
                    seen.add(row["image_url"])

    # 파일이 실제로 존재하는지 확인하고 기존 이미지도 중복 비교에 포함한다.
    hashes = []
    counts = {}
    for label in queries:
        folder = root / "data/images" / label
        folder.mkdir(parents=True, exist_ok=True)
        counts[label] = 0
        for path in sorted(folder.glob("*.jpg")):
            with Image.open(path) as image:
                hashes.append(imagehash.phash(image))
            counts[label] += 1

    if any(count < target for count in counts.values()):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cache = root / "ckpt/pretrained/clip"
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(cache))
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=str(cache)).to(device).eval()
        search = DDGS(timeout=20)
        with requests.Session() as session, metadata.open("a", encoding="utf-8", newline="") as file:
            session.headers["User-Agent"] = "Mozilla/5.0"
            writer = csv.DictWriter(file, fieldnames=fields)
            if file.tell() == 0:
                writer.writeheader()
                file.flush()

            for label, terms in queries.items():
                folder = root / "data/images" / label
                number = max((int(path.stem.rsplit("_", 1)[1]) for path in folder.glob(f"{label}_*.jpg")), default=0)
                # 검색어별 저장량을 제한해 한 제품·형태에 편중되는 것을 줄인다.
                per_query = dict.fromkeys(terms, 0)
                quota = math.ceil(target / len(terms)) + 10
                for page in range(1, 6):
                    if counts[label] >= target:
                        break
                    for query in terms:
                        if counts[label] >= target:
                            break
                        if per_query[query] >= quota:
                            continue
                        try:
                            results = search.images(query + " -collage -collection", max_results=100, page=page)
                        except Exception as error:
                            print(f"검색 실패: {query} / page={page}: {error}")
                            continue
                        for result in results:
                            if counts[label] >= target or per_query[query] >= quota:
                                break
                            url = result.get("image")
                            if not url or url in seen:
                                continue
                            seen.add(url)
                            time.sleep(0.25)
                            try:
                                with session.get(url, timeout=(5, 15), stream=True) as response:
                                    response.raise_for_status()
                                    content = bytearray()
                                    for chunk in response.iter_content(65536):
                                        content.extend(chunk)
                                        if len(content) > 20 * 1024 * 1024:
                                            raise ValueError("이미지 20MB 초과")
                                with warnings.catch_warnings():
                                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                                    with Image.open(BytesIO(content)) as original:
                                        image = ImageOps.exif_transpose(original).convert("RGB")
                                if min(image.size) < 256:
                                    continue
                                phash = imagehash.phash(image)
                            except (requests.RequestException, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                                continue
                            if any(phash - previous <= 4 for previous in hashes):
                                continue

                            inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True).to(device)
                            with torch.inference_mode():
                                scores = model(**inputs).logits_per_image[0].softmax(dim=0)
                            if scores.argmax().item() != 0 or scores[0].item() < 0.7:
                                continue

                            number += 1
                            path = folder / f"{label}_{number:06d}.jpg"
                            image.save(path, quality=95)
                            writer.writerow({"path": path.relative_to(root).as_posix(), "label": label, "query": query, "image_url": url, "source_url": result.get("url", ""), "title": result.get("title", ""), "phash": str(phash), "filter_score": round(scores[0].item(), 4)})
                            file.flush()
                            hashes.append(phash)
                            counts[label] += 1
                            per_query[query] += 1
                            print(f"{label}: {counts[label]}/{target} | {path.name}")

    for label, count in counts.items():
        print(f"{label}: {count}/{target}" + (" — 검색 범위 내 후보 부족 또는 검색 실패로 미달" if count < target else " — 완료"))
