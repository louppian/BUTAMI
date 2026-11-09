"""저장 이미지 목록과 캐시된 검색 후보의 단계별 탈락 사유를 제출한다."""
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
import recrawl_smart as r

ROOT = r.ROOT
WORK = ROOT / "results/phase1a_images/.smart_recrawl_clear"
OUT = ROOT / "results/phase1a_images/audit"


def prepare(candidate):
    item = r.download(candidate)
    if item is None:
        return candidate, None, "download_decode_or_image_size"
    if not hasattr(r.LOCAL, "detector"):
        r.LOCAL.detector = r.load_face_detector(ROOT)
    try:
        reason = r.face_reason(r.LOCAL.detector, item[1])
    except r.cv2.error:
        reason = "face_error"
    return candidate, item, reason


def main():
    OUT.mkdir(exist_ok=True)
    rows = r.metadata_rows(WORK / r.RESULT_DIR / "metadata.csv", smart=True)
    r.write_metadata(OUT / "collected.csv", rows)
    sheet = Image.new("RGB", (1200, ((len(rows) + 4) // 5) * 180), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        image = Image.open(WORK / row["path"]).convert("RGB")
        sheet.paste(ImageOps.contain(image, (232, 150)), ((i % 5) * 240, (i // 5) * 180))
        draw.text(((i % 5) * 240 + 4, (i // 5) * 180 + 153), f'{Path(row["path"]).stem} / {row["filter_score"]}', fill="black")
    sheet.save(OUT / "collected.jpg")
    with sqlite3.connect(WORK / r.RESULT_DIR / "smart_recrawl.sqlite3") as db:
        jobs = db.execute("SELECT query, results FROM jobs WHERE query LIKE 'public/%'").fetchall()
    candidates = []
    for query, payload in jobs:
        # 검색어마다 앞의 30개를 같은 기준으로 재검사한다.
        candidates.extend({**item, "query": query} for item in json.loads(payload)[:30])
    r.cv2.setNumThreads(1)
    processor, model, text, device = r.crawl.load_clip(ROOT)
    lens = r.load_lens_features(processor, model, device)
    summary = defaultdict(Counter)
    fields = ["query", "image", "reason", "smart_score", "winner", "clear", "tinted", "no_glasses"]
    with (OUT / "rejections.csv").open("w", encoding="utf-8-sig", newline="") as file, ThreadPoolExecutor(16) as pool:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for start in range(0, len(candidates), 32):
            prepared = list(pool.map(prepare, candidates[start:start + 32]))
            passed = [item[1][1] for item in prepared if item[2] == "face_pass"]
            scores = iter(r.score_batch(passed, processor, model, text, lens, device)) if passed else iter(())
            for candidate, item, reason in prepared:
                row = {"query": candidate["query"], "image": candidate["image"], "reason": reason}
                if reason == "face_pass":
                    score = next(scores)
                    winner = int(score[:-3].argmax())
                    row.update(smart_score=float(score[2]), winner=r.crawl.DESCRIPTIONS[winner],
                               clear=float(score[-3]), tinted=float(score[-2]), no_glasses=float(score[-1]))
                    row["reason"] = ("clip_class" if winner != 2 else "clip_score" if score[2] < r.CLIP_THRESHOLD
                                     else "lens" if score[-3] <= max(score[-2:]) else "filter_pass")
                summary[row["query"]][row["reason"]] += 1
                writer.writerow(row)
            file.flush()
            print(f"audit {min(start+32,len(candidates))}/{len(candidates)}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
