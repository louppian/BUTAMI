"""한국어·영어 검색 → YuNet 단일 얼굴 검증 → CLIP 선별 → 클래스별 500장 저장."""

import csv
import json
import re
import warnings
import sqlite3
import threading
import time
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import Counter
from pathlib import Path
from itertools import zip_longest
from urllib.parse import urlencode
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps

import image_crawl as crawl

ROOT = Path(__file__).resolve().parents[2]
TARGET = 500
BATCH_SIZE = 32
WORKERS = 32
CLIP_THRESHOLD = 0.7
FACE_THRESHOLD = 0.9
MIN_FACE_SIZE = 96
MIN_FACE_AREA = 0.01
FACE_MAX_SIZE = 960
CLASSES = crawl.CLASSES
QUERIES = {label: list(terms) for label, terms in crawl.QUERIES.items()}
# 얼굴 필터를 유지하면서 부족한 수량을 보충할 인물·기종별 검색어.
QUERIES["no_glasses"] += [
    "남자 얼굴 인물 사진", "여자 얼굴 인물 사진", "남성 증명사진", "여성 증명사진",
    "남자 정면 프로필 사진", "여자 정면 프로필 사진", "중년 남자 인물 사진", "중년 여자 인물 사진",
    "노년 남성 초상 사진", "노년 여성 초상 사진", "한국 남성 얼굴 클로즈업", "한국 여성 얼굴 클로즈업",
    "남자 배우 프로필 얼굴", "여자 배우 프로필 얼굴", "남성 자연광 얼굴 사진", "여성 자연광 얼굴 사진",
    "male portrait headshot", "female portrait headshot", "man face close up portrait", "woman face close up portrait",
    "middle aged man portrait", "middle aged woman portrait", "elderly man face portrait", "elderly woman face portrait",
    "Korean man portrait headshot", "Korean woman portrait headshot", "Japanese man portrait face", "Japanese woman portrait face",
    "Indian man portrait headshot", "Indian woman portrait headshot", "African man portrait face", "African woman portrait face",
    "male actor headshot", "female actor headshot", "man natural light portrait", "woman natural light portrait",
]
QUERIES["regular_glasses"] += [
    "안경 남자 증명사진", "안경 여자 증명사진", "안경 남성 프로필 사진", "안경 여성 프로필 사진",
    "안경 중년 남자 얼굴", "안경 중년 여자 얼굴", "안경 노년 남성 인물", "안경 노년 여성 인물",
    "안경 쓴 한국 남성 얼굴", "안경 쓴 한국 여성 얼굴", "안경 쓴 남자 클로즈업", "안경 쓴 여자 클로즈업",
    "male eyeglasses headshot", "female eyeglasses headshot", "middle aged man eyeglasses portrait",
    "middle aged woman eyeglasses portrait", "elderly man eyeglasses portrait", "elderly woman eyeglasses portrait",
    "Asian man eyeglasses portrait", "Asian woman eyeglasses portrait", "black man eyeglasses portrait",
    "black woman eyeglasses portrait", "man reading glasses face", "woman reading glasses face",
]
QUERIES["smart_glasses"] += [
    "구글 글래스 착용 얼굴", "구글 글라스 착용 사람", "구글 글래스 익스플로러 착용", "구글 글래스 엔터프라이즈 착용",
    "엡손 모베리오 착용", "레노버 씽크리얼리티 착용", "오포 에어글래스 착용", "샤오미 스마트글래스 착용",
    "레이밴 메타 리뷰 얼굴", "레이밴 메타 안경 착용 정면", "레이밴 메타 안경 착용 옆모습", "메타 스마트안경 시연 착용",
    "Google Glass wearing portrait", "Google Glass Explorer wearing face", "Google Glass Enterprise Edition wearing",
    "Epson Moverio wearing face", "Lenovo ThinkReality A3 wearing", "Oppo Air Glass wearing face",
    "Xiaomi smart glasses wearing face", "Ray Ban Meta review wearing face", "Ray Ban Meta demo wearing portrait",
    "Ray Ban Stories review wearing face", "Bose Frames review wearing face", "Amazon Echo Frames review wearing face",
    "Solos AirGo review wearing face", "Even Realities G1 review wearing face", "Huawei smart glasses review wearing face",
    "XREAL One Pro wearing face", "Rokid Max review wearing face", "VITURE Pro wearing face",
]
QUERIES["no_glasses"] += [f"{origin} {person} portrait {setting}"
                          for origin in ("Korean", "Japanese", "Chinese", "Indian", "African", "European", "American", "Latin American")
                          for person in ("man", "woman") for setting in ("studio", "outdoors", "close up", "headshot")]
QUERIES["regular_glasses"] += [f"{origin} {person} wearing eyeglasses {setting}"
                               for origin in ("Korean", "Japanese", "Chinese", "Indian", "African", "European", "American", "Latin American")
                               for person in ("man", "woman") for setting in ("portrait", "headshot", "close up", "outdoors")]
QUERIES["smart_glasses"] += [f"{product} {setting}"
                             for product in ("Google Glass", "Google Glass Enterprise", "Ray Ban Meta", "Ray Ban Stories", "Bose Frames", "Amazon Echo Frames", "Solos AirGo", "Even Realities G1", "XREAL Air", "Rokid Max")
                             for setting in ("man wearing close up", "woman wearing portrait", "person wearing front view", "person wearing side view")]
QUERIES = {label: list(dict.fromkeys(terms)) for label, terms in QUERIES.items()}
QUERIES = {label: list(dict.fromkeys(terms + [f"{query} {year}" for year in (2026, 2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018) for query in terms]))
           for label, terms in QUERIES.items()}
YOUTUBE_QUERIES = {
    "no_glasses": ["얼굴 브이로그", "일상 브이로그 얼굴", "남자 토크 얼굴", "여자 토크 얼굴", "남자 인터뷰", "여자 인터뷰", "안경 안쓴 브이로그", "안경 안쓴 메이크업", "man talking vlog", "woman talking vlog", "male interview face", "female interview face", "face closeup vlog", "skincare routine face"],
    "regular_glasses": ["안경 쓴 브이로그", "안경 리뷰 착용", "안경테 추천 착용", "안경 쓴 남자 인터뷰", "안경 쓴 여자 브이로그", "뿔테 안경 착용 후기", "금속테 안경 착용", "prescription glasses review face", "eyeglasses try on", "eyeglasses frames review", "man with glasses interview", "woman with glasses vlog", "round eyeglasses try on", "rimless glasses try on"],
    "smart_glasses": ["레이밴 메타 리뷰", "레이밴 메타 착용 후기", "메타 레이밴 2세대 리뷰", "엑스리얼 에어 착용", "엑스리얼 원 리뷰", "비튜어 프로 리뷰", "로키드 맥스 리뷰", "구글 글래스 착용", "이븐 리얼리티 G1 리뷰", "Ray-Ban Meta review wearing", "Ray-Ban Stories review wearing", "XREAL One review", "VITURE Pro review", "Rokid Max review", "Google Glass review wearing", "Bose Frames review", "Amazon Echo Frames review", "Even Realities G1 review"],
}
GOOGLE_BROWSER = None
GOOGLE_RUNTIME = None


def bilingual(terms):
    korean = [q for q in terms if re.search(r"[가-힣]", q)]
    english = [q for q in terms if not re.search(r"[가-힣]", q)]
    return [q for pair in zip_longest(english, korean) for q in pair if q]


QUERIES = {label: bilingual(terms) for label, terms in QUERIES.items()}
YOUTUBE_QUERIES = {label: bilingual(terms) for label, terms in YOUTUBE_QUERIES.items()}


def google_results(query, session, page=1):
    """독립 Edge 브라우저에서 렌더링한 구글 이미지의 원본 주소를 읽는다."""
    global GOOGLE_BROWSER, GOOGLE_RUNTIME
    if GOOGLE_BROWSER is None:
        from playwright.sync_api import sync_playwright
        GOOGLE_RUNTIME = sync_playwright().start()
        browser = GOOGLE_RUNTIME.chromium.launch(channel="msedge", headless=False)
        GOOGLE_BROWSER = browser.new_page(locale="ko-KR", viewport={"width": 1440, "height": 1000})
    tab = GOOGLE_BROWSER
    tab.goto("https://www.google.com/search?" + urlencode({"udm": 2, "q": query}), wait_until="domcontentloaded", timeout=45000)
    if "/sorry/" in tab.url or tab.locator('iframe[src*="recaptcha"]').count():
        print("구글 CAPTCHA: 열린 Edge 창에서 직접 확인하면 수집을 이어갑니다.", flush=True)
        tab.wait_for_url(lambda url: "/search" in url and "/sorry/" not in url, timeout=0)
    tab.locator('[data-lpage]').first.wait_for(timeout=20000)
    results = parse_google_html(tab.content())
    if not results:
        raise ValueError("구글 HTML의 원본 이미지 필드를 확인하세요.")
    print(f"구글 HTML 일괄 추출: {query} | 원본 {len(results)}개", flush=True)
    return results


def parse_google_html(html):
    results, seen = [], set()
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\[\s*\d+,\s*"(?:\\.|[^"\\])*",\s*\[\s*"https?://', html):
        try:
            record = decoder.raw_decode(html, match.start())[0]
            image = record[3][0]
            info = next(item["2003"] for item in record if isinstance(item, dict) and "2003" in item)
            if isinstance(image, str) and image.startswith(("http://", "https://")) and image not in seen:
                results.append({"image": image, "url": info[2], "title": info[3] or ""})
                seen.add(image)
        except (ValueError, IndexError, TypeError, StopIteration):
            continue
    return results


def youtube_results(query, session):
    response = session.get("https://www.youtube.com/results", params={"search_query": query}, timeout=(5, 20))
    response.raise_for_status()
    match = re.search(r"(?:var\s+)?ytInitialData\s*=\s*", response.text)
    if match is None:
        raise ValueError("유튜브 검색 결과의 ytInitialData를 확인하세요.")
    stack = [json.JSONDecoder().raw_decode(response.text[match.end():])[0]]
    results, seen = [], set()
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            video = value.get("videoRenderer")
            if video:
                video_id = video.get("videoId", "")
                if re.fullmatch(r"[\w-]{11}", video_id, flags=re.ASCII) and video_id not in seen:
                    seen.add(video_id)
                    title = video.get("title", {})
                    results.append({"image": f"https://i.ytimg.com/vi/{video_id}/maxresdefault.jpg",
                                    "fallback_image": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                                    "url": f"https://www.youtube.com/watch?v={video_id}",
                                    "title": title.get("simpleText") or "".join(run.get("text", "") for run in title.get("runs", []))})
            stack.extend(reversed(list(value.values())))
        elif isinstance(value, list):
            stack.extend(reversed(value))
    return results


def search_jobs():
    # 구글 이미지 브라우저 검색과 유튜브 검색을 세 클래스에 번갈아 적용한다.
    for page in range(1, 2):
        for i in range(max(map(len, QUERIES.values()))):
            for label in CLASSES:
                if i < len(QUERIES[label]):
                    yield label, "google: " + QUERIES[label][i], "google", page
                if page == 1 and i < len(YOUTUBE_QUERIES[label]):
                    yield label, "youtube: " + YOUTUBE_QUERIES[label][i], "youtube", 1


def download(result):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = crawl.requests.Session()
        LOCAL.session.headers["User-Agent"] = "Mozilla/5.0"
    for position, url in enumerate((result["image"], result.get("fallback_image"))):
        if url:
            LOCAL.stats["original_requests" if position == 0 else "fallback_requests"] += 1
            try:
                started, content = time.perf_counter(), bytearray()
                with LOCAL.session.get(url, timeout=(10, 30), stream=True) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(65536):
                        content.extend(chunk)
                        if len(content) > 20 * 1024 * 1024 or time.perf_counter() - started > 60:
                            LOCAL.stats["body_limit" if len(content) > 20 * 1024 * 1024 else "download_deadline"] += 1
                            break
                    else:
                        with Image.open(BytesIO(content)) as original:
                            if min(original.size) < 256 or original.width * original.height > 20_000_000:
                                LOCAL.stats["image_size_reject"] += 1
                                continue
                            image = ImageOps.exif_transpose(original).convert("RGB")
                        return {**result, "image": url}, image, crawl.imagehash.phash(image)
            except crawl.requests.Timeout:
                LOCAL.stats["request_timeout"] += 1
            except crawl.requests.HTTPError:
                LOCAL.stats["http_error"] += 1
            except crawl.requests.RequestException:
                LOCAL.stats["connection_error"] += 1
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                LOCAL.stats["image_decode_error"] += 1
    return None


def load_face_detector(root):
    weights = root / "ckpt/pretrained/face_detection_yunet_2023mar.onnx"
    if not weights.is_file():
        raise FileNotFoundError(f"YuNet 가중치를 준비하세요: {weights}")
    # 0.5 이상 얼굴 후보를 모두 찾고, 단일 얼굴의 점수는 0.9 이상으로 제한한다.
    return cv2.FaceDetectorYN.create(str(weights), "", (320, 320), 0.5, 0.3, 5000)


def face_status(detector, image):
    width, height = image.size
    scale = min(1, FACE_MAX_SIZE / max(width, height))
    size = max(1, round(width * scale)), max(1, round(height * scale))
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if size != image.size:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    detector.setInputSize(size)
    _, faces = detector.detect(bgr)
    if faces is None:
        return "face_missing"
    if len(faces) != 1:
        return "face_multiple"
    face = faces[0]
    if not np.isfinite(face).all() or face[-1] < FACE_THRESHOLD:
        return "face_score_reject"
    x, y, w, h = face[:4] * np.array([width / size[0], height / size[1]] * 2)
    if min(w, h) < MIN_FACE_SIZE or w * h < width * height * MIN_FACE_AREA:
        return "face_size_reject"
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return "face_clipped"
    points = face[4:14].reshape(5, 2) * (width / size[0], height / size[1])
    return "face_pass" if ((points[:, 0] >= x) & (points[:, 0] < x + w)
                           & (points[:, 1] >= y) & (points[:, 1] < y + h)).all() else "face_landmark_reject"


def clear_face(detector, image):
    return face_status(detector, image) == "face_pass"


LOCAL = threading.local()
STATE = ROOT / "results/phase1a_images/crawl_state.sqlite3"


def connect_state():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(STATE, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS jobs(label TEXT, query TEXT, results TEXT, done INTEGER DEFAULT 0, PRIMARY KEY(label,query))")
    db.execute("CREATE TABLE IF NOT EXISTS attempts(label TEXT, url TEXT, PRIMARY KEY(label,url))")
    db.commit()
    return db


def load_similarity_text(processor, model, device):
    import torch
    inputs = processor(text=["A photograph of a person without glasses.", "A photograph of a person wearing eyeglasses.",
                             "A photograph of a person wearing smart glasses.", "A photograph of a person wearing sunglasses."],
                       return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        features = model.get_text_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)


def score_batch(images, processor, model, text_features, similarity_text, device):
    import torch
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        probability = (model.logit_scale.exp() * features @ text_features.T).float().softmax(dim=-1)
        cosine = (features @ similarity_text.T).float()
        return torch.cat((probability, cosine), dim=-1).cpu().numpy()


def prepare_image(result):
    LOCAL.stats = Counter()
    item = download(result)
    if item is None:
        LOCAL.stats["download_failed"] += 1
        return None, LOCAL.stats
    LOCAL.stats["download_ok"] += 1
    if not hasattr(LOCAL, "detector"):
        LOCAL.detector = load_face_detector(ROOT)
    try:
        status = face_status(LOCAL.detector, item[1])
        LOCAL.stats[status] += 1
        return (item if status == "face_pass" else None), LOCAL.stats
    except cv2.error:
        LOCAL.stats["face_error"] += 1
        return None, LOCAL.stats


def search_producer(queue, counts, stop):
    db = connect_state()
    session = crawl.requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
    try:
        for label, query, source, page in search_jobs():
            if stop.is_set() or all(n >= TARGET for n in counts.values()):
                break
            if counts[label] >= TARGET:
                continue
            row = db.execute("SELECT results,done FROM jobs WHERE label=? AND query=?", (label, query)).fetchone()
            if row and row[1]:
                continue
            try:
                results = json.loads(row[0]) if row else SEARCH_HANDLERS[source](query.split(": ", 1)[1], session, page)
            except Exception as error:
                print(f"검색 실패: {query}: {error}", flush=True)
                if GOOGLE_BROWSER is not None and GOOGLE_BROWSER.is_closed():
                    break
                continue
            if not row:
                db.execute("INSERT INTO jobs(label,query,results) VALUES(?,?,?)", (label, query, json.dumps(results, ensure_ascii=False)))
                db.commit()
            queue.put((label, query, source, results))
    finally:
        queue.put(None)
        session.close()
        db.close()
        if GOOGLE_BROWSER is not None:
            GOOGLE_BROWSER.context.browser.close()
            GOOGLE_RUNTIME.stop()


def main():
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    cv2.setNumThreads(1)
    metadata = ROOT / "results/phase1a_images/metadata.csv"
    db = connect_state()
    records = list(csv.DictReader(metadata.open(encoding="utf-8", newline=""))) if metadata.exists() else []
    counts, seen, hashes = dict.fromkeys(CLASSES, 0), set(), []
    for row in records:
        path = ROOT / row["path"]
        if row["label"] not in counts or not path.is_file():
            raise ValueError(f"CSV와 이미지 확인 필요: {path}")
        counts[row["label"]] += 1
        seen.add(row["image_url"])
        hashes.append(int(row["phash"], 16))
    if any(len(list((ROOT / "data/images" / label).glob("*.jpg"))) != counts[label] for label in CLASSES):
        raise ValueError("CSV와 JPEG 수량 확인 필요")
    if all(n >= TARGET for n in counts.values()):
        print("수집 완료:", counts, flush=True)
        db.close()
        return
    attempted = set(db.execute("SELECT label,url FROM attempts"))
    processor, model, text_features, device = crawl.load_clip(ROOT)
    similarity_text = load_similarity_text(processor, model, device)
    number = {label: max((int(p.stem.rsplit("_", 1)[1]) for p in (ROOT / "data/images" / label).glob(f"{label}_*.jpg")), default=0) for label in CLASSES}
    queue, stop = Queue(4), threading.Event()
    producer = threading.Thread(target=search_producer, args=(queue, counts, stop), daemon=True)
    producer.start()
    started, last_report, baseline = time.perf_counter(), time.perf_counter(), sum(counts.values())
    futures, remaining, jobs = {}, {}, {}
    stages, job_stages = Counter(), {}
    exhausted = False
    metadata.parent.mkdir(parents=True, exist_ok=True)
    print(f"시작 {counts} | 이미지 검색 + 다운로드/얼굴 검사 32병렬 + CLIP={device} | CLIP={CLIP_THRESHOLD}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, metadata.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=crawl.FIELDS)
        if file.tell() == 0:
            writer.writeheader()
        while not exhausted or futures:
            if all(n >= TARGET for n in counts.values()):
                stop.set()
                break
            while len(futures) < 256 and not exhausted:
                try:
                    job = queue.get(timeout=0.1 if not futures else 0)
                except Empty:
                    break
                if job is None:
                    exhausted = True
                    break
                label, query, source, results = job
                key = label, query
                jobs[key] = [0, 0, 0, time.perf_counter()]
                job_stages[key] = Counter(candidates=len(results))
                stages["candidates"] += len(results)
                remaining[key] = 0
                for result in results:
                    url = result.get("image")
                    if counts[label] >= TARGET or not url or url in seen or (label, url) in attempted:
                        stages["url_skipped"] += 1
                        job_stages[key]["url_skipped"] += 1
                        continue
                    attempted.add((label, url))
                    remaining[key] += 1
                    futures[pool.submit(prepare_image, result)] = (key, url)
                if remaining[key] == 0:
                    db.execute("UPDATE jobs SET done=1 WHERE label=? AND query=?", key)
                    db.commit()
            if futures:
                ready, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                batch, completed = [], []
                for future in ready:
                    key, url = futures.pop(future)
                    item, report = future.result()
                    stages.update(report)
                    job_stages[key].update(report)
                    completed.append((key, url))
                    jobs[key][0] += 1
                    if item is not None:
                        jobs[key][1] += 1
                        if counts[key[0]] < TARGET and all((int(str(item[2]), 16) ^ h).bit_count() > 4 for h in hashes):
                            batch.append((key, item))
                        else:
                            reason = "target_skipped" if counts[key[0]] >= TARGET else "phash_duplicate"
                            stages[reason] += 1
                            job_stages[key][reason] += 1
                if batch:
                    for offset in range(0, len(batch), BATCH_SIZE):
                        part = batch[offset:offset + BATCH_SIZE]
                        scores = score_batch([item[1] for _, item in part], processor, model, text_features, similarity_text, device)
                        for (key, (result, image, phash)), score in zip(part, scores):
                            stages["clip_tested"] += 1
                            job_stages[key]["clip_tested"] += 1
                            label, query = key
                            index, h = CLASSES.index(label), int(str(phash), 16)
                            face_cosine, glasses_cosine = float(score[-4]), float(score[-3:].max())
                            accepted = (face_cosine > glasses_cosine) if index == 0 else (score[:len(crawl.DESCRIPTIONS)].argmax() == index and score[index] >= CLIP_THRESHOLD)
                            stages["clip_pass" if accepted else "clip_reject"] += 1
                            job_stages[key]["clip_pass" if accepted else "clip_reject"] += 1
                            if not accepted:
                                continue
                            if counts[label] >= TARGET or any((h ^ previous).bit_count() <= 4 for previous in hashes):
                                reason = "target_skipped" if counts[label] >= TARGET else "phash_duplicate"
                                stages[reason] += 1
                                job_stages[key][reason] += 1
                                continue
                            number[label] += 1
                            path = ROOT / "data/images" / label / f"{label}_{number[label]:06d}.jpg"
                            temporary = path.with_suffix(".jpg.part")
                            image.save(temporary, format="JPEG", quality=95)
                            temporary.replace(path)
                            writer.writerow({"path": path.relative_to(ROOT).as_posix(), "label": label, "query": query,
                                             "image_url": result["image"], "source_url": result.get("url", ""), "title": result.get("title", ""),
                                             "phash": str(phash), "filter_score": round(face_cosine - glasses_cosine if index == 0 else float(score[index]), 4), "collection": "people"})
                            file.flush()
                            hashes.append(h)
                            seen.add(result["image"])
                            counts[label] += 1
                            jobs[key][2] += 1
                            stages["saved"] += 1
                            job_stages[key]["saved"] += 1
                for key, url in completed:
                    db.execute("INSERT OR IGNORE INTO attempts VALUES(?,?)", (key[0], url))
                    remaining[key] -= 1
                    if remaining[key] == 0:
                        db.execute("UPDATE jobs SET done=1 WHERE label=? AND query=?", key)
                        tested, faces, saved, begin = jobs[key]
                        print(f"{key[0]} {counts[key[0]]}/500 | 검사={tested} face={faces} 저장={saved} {time.perf_counter()-begin:.1f}s | {key[1]}", flush=True)
                        report = {"label": key[0], "query": key[1], "seconds": round(time.perf_counter()-begin, 1), "stages": dict(job_stages[key])}
                        with (metadata.parent / "crawl_stages.jsonl").open("a", encoding="utf-8") as log:
                            log.write(json.dumps(report, ensure_ascii=False) + "\n")
                        print("단계별:", json.dumps(report, ensure_ascii=False), flush=True)
                db.commit()
            now = time.perf_counter()
            if now - last_report >= 60:
                added = sum(counts.values()) - baseline
                status = {"counts": counts, "added": added, "elapsed_seconds": round(now-started, 1), "saved_per_minute": round(added*60/(now-started), 2), "inflight": len(futures), "stages": dict(stages)}
                (metadata.parent / "crawl_status.json").write_text(json.dumps(status, ensure_ascii=False), encoding="utf-8")
                print("실측:", status, flush=True)
                last_report = now
    stop.set()
    db.close()
    print("최종 수집 수량:", counts, flush=True)
    print("최종 단계별:", json.dumps(dict(stages), ensure_ascii=False), flush=True)
    if any(n < TARGET for n in counts.values()):
        raise RuntimeError(f"검색 후보 소진: {counts}")


SEARCH_HANDLERS = {"google": google_results, "youtube": lambda query, session, page: youtube_results(query, session)}


if __name__ == "__main__":
    main()
