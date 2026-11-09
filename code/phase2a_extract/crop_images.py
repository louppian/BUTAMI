"""각 클래스 500장의 가장 큰 얼굴을 원본 크기로 PNG 저장."""
import csv
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
LOCAL = threading.local()


def detect(image, size, threshold):
    if not hasattr(LOCAL, "detector"):
        LOCAL.detector = cv2.FaceDetectorYN.create(str(ROOT / "ckpt/pretrained/face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.5, 0.3, 5000)
    scale = min(2, size / max(image.size))
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    pixels = cv2.resize(np.asarray(image)[:, :, ::-1], (width, height))
    LOCAL.detector.setScoreThreshold(threshold)
    LOCAL.detector.setInputSize((width, height))
    _, faces = LOCAL.detector.detect(pixels)
    if faces is None:
        return []
    return [(float(f[0] * image.width / width), float(f[1] * image.height / height),
             float(f[2] * image.width / width), float(f[3] * image.height / height), float(f[-1])) for f in faces]


def crop(job):
    path, prefix, number = job
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    faces = detect(image, 960, 0.5)
    method = "yunet_0.5"
    if not faces:
        method = "yunet_multiscale_0.3"
        for size in (640, 1600, 320):
            faces = detect(image, size, 0.3)
            if faces:
                break
    if not faces:
        return {"source": path.relative_to(ROOT).as_posix(), "output": "", "bbox": "", "score": "", "method": "review"}
    x, y, w, h, score = max(faces, key=lambda f: f[2] * f[3])
    box = max(0, math.floor(x)), max(0, math.floor(y)), min(image.width, math.ceil(x + w)), min(image.height, math.ceil(y + h))
    output = ROOT / "data/crop_images" / f"{prefix}_{number:03d}.png"
    image.crop(box).save(output)
    return {"source": path.relative_to(ROOT).as_posix(), "output": output.relative_to(ROOT).as_posix(), "bbox": str(box), "score": round(score, 5), "method": method}


def main():
    cv2.setNumThreads(1)
    (ROOT / "data/crop_images").mkdir(parents=True, exist_ok=True)
    jobs = [(path, prefix, i) for label, prefix in (("no_glasses", "noglass"), ("regular_glasses", "glass"), ("smart_glasses", "smart"))
            for i, path in enumerate(sorted((ROOT / "data/images" / label).glob("*.jpg")), 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(crop, jobs))
    compact(rows)
    manifest = ROOT / "results/phase2a_extract/crop_images.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    missing = [r["source"] for r in rows if not r["output"]]
    print(f"saved={len(rows)-len(missing)}/{len(rows)}; review={missing}", flush=True)


def compact(rows):
    counts, staged = {}, []
    for row in rows:
        if not row["output"]:
            continue
        path = ROOT / row["output"]
        prefix = path.stem.split("_")[0]
        counts[prefix] = counts.get(prefix, 0) + 1
        temporary = path.with_suffix(".png.tmp")
        path.rename(temporary)
        staged.append((temporary, path.with_name(f"{prefix}_{counts[prefix]:03d}.png"), row))
    for temporary, target, row in staged:
        temporary.rename(target)
        row["output"] = target.relative_to(ROOT).as_posix()


if __name__ == "__main__":
    main()
