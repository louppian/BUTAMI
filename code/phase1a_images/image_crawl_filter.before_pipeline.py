"""한국어·영어 검색 → YuNet 단일 얼굴 검증 → CLIP 선별 → 클래스별 500장 저장."""

import csv
import json
import math
import re
import warnings
import atexit
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from itertools import zip_longest
from urllib.parse import urlencode

import cv2
import numpy as np
from PIL import Image

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


def bilingual(terms):
    korean = [q for q in terms if re.search(r"[가-힣]", q)]
    english = [q for q in terms if not re.search(r"[가-힣]", q)]
    return [q for pair in zip_longest(english, korean) for q in pair if q]


QUERIES = {label: bilingual(terms) for label, terms in QUERIES.items()}
YOUTUBE_QUERIES = {label: bilingual(terms) for label, terms in YOUTUBE_QUERIES.items()}


def google_results(query, session, page=1):
    """독립 Edge 브라우저에서 렌더링한 구글 이미지의 원본 주소를 읽는다."""
    global GOOGLE_BROWSER
    if GOOGLE_BROWSER is None:
        from playwright.sync_api import sync_playwright
        runtime = sync_playwright().start()
        browser = runtime.chromium.launch(channel="msedge", headless=False)
        GOOGLE_BROWSER = browser.new_page(locale="ko-KR", viewport={"width": 1440, "height": 1000})
        atexit.register(runtime.stop)
        atexit.register(browser.close)
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
    for url in (result["image"], result.get("fallback_image")):
        if url:
            try:
                item = crawl.download({**result, "image": url})
                if item is not None:
                    return item
            except SyntaxError:  # 손상된 EXIF/TIFF 메타데이터
                pass
    return None


def load_face_detector(root):
    weights = root / "ckpt/pretrained/face_detection_yunet_2023mar.onnx"
    if not weights.is_file():
        raise FileNotFoundError(f"YuNet 가중치를 준비하세요: {weights}")
    # 0.5 이상 얼굴 후보를 모두 찾고, 단일 얼굴의 점수는 0.9 이상으로 제한한다.
    return cv2.FaceDetectorYN.create(str(weights), "", (320, 320), 0.5, 0.3, 5000)


def clear_face(detector, image):
    width, height = image.size
    scale = min(1, FACE_MAX_SIZE / max(width, height))
    size = max(1, round(width * scale)), max(1, round(height * scale))
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if size != image.size:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    detector.setInputSize(size)
    _, faces = detector.detect(bgr)
    if faces is None or len(faces) != 1:
        return False
    face = faces[0]
    if not np.isfinite(face).all() or face[-1] < FACE_THRESHOLD:
        return False
    x, y, w, h = face[:4] * np.array([width / size[0], height / size[1]] * 2)
    if min(w, h) < MIN_FACE_SIZE or w * h < width * height * MIN_FACE_AREA:
        return False
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return False
    points = face[4:14].reshape(5, 2) * (width / size[0], height / size[1])
    return bool(((points[:, 0] >= x) & (points[:, 0] < x + w)
                 & (points[:, 1] >= y) & (points[:, 1] < y + h)).all())


def main():
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    metadata = ROOT / "results/phase1a_images/metadata.csv"
    detector = load_face_detector(ROOT)
    records, seen, hashes = {}, set(), []
    counts = dict.fromkeys(CLASSES, 0)
    per_query = {label: dict.fromkeys(["google: " + q for q in QUERIES[label]] + ["youtube: " + q for q in YOUTUBE_QUERIES[label]], 0) for label in CLASSES}
    if metadata.exists():
        with metadata.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != crawl.FIELDS:
                raise ValueError(f"사람 이미지 수집 기록을 확인하세요: {metadata}")
            for row in reader:
                if row["collection"] != "people" or row["label"] not in CLASSES:
                    raise ValueError(f"수집 종류·클래스를 확인하세요: {metadata}")
                records[row["path"]] = row
    for label in CLASSES:
        folder = ROOT / "data/images" / label
        folder.mkdir(parents=True, exist_ok=True)
        for path in sorted(folder.glob("*.jpg")):
            row = records.get(path.relative_to(ROOT).as_posix())
            if row is None or row["label"] != label:
                raise ValueError(f"이미지에 대응하는 수집 기록을 확인하세요: {path}")
            with Image.open(path) as original:
                image = original.convert("RGB")
                if row["query"] != "local_import" and not clear_face(detector, image):
                    raise ValueError(f"기존 이미지가 얼굴 필터 조건에 미달합니다. 해당 이미지와 CSV 행을 별도 보관한 뒤 실행하세요: {path}")
                hashes.append(crawl.imagehash.phash(image))
            seen.add(row["image_url"])
            counts[label] += 1
            if row["query"] in per_query[label]:
                per_query[label][row["query"]] += 1
    if all(count >= TARGET for count in counts.values()):
        print("얼굴 필터 검증·수집 완료:", counts)
        return
    processor, model, text_features, device = crawl.load_clip(ROOT)
    youtube = crawl.requests.Session()
    youtube.headers["User-Agent"] = "Mozilla/5.0"
    seen_videos = {row["source_url"] for row in records.values() if "youtube.com/watch" in row["source_url"]}
    metadata.parent.mkdir(parents=True, exist_ok=True)
    print(f"CLIP={device} | YuNet=CPU | 얼굴 점수≥{FACE_THRESHOLD} | 목표={TARGET}×{len(CLASSES)}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool, metadata.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=crawl.FIELDS)
        if file.tell() == 0:
            writer.writeheader()
            file.flush()
        attempted = {label: set() for label in CLASSES}
        number = {label: max((int(path.stem.rsplit("_", 1)[1]) for path in (ROOT / "data/images" / label).glob(f"{label}_*.jpg")), default=0) for label in CLASSES}
        quota = {label: math.ceil(TARGET / len(per_query[label])) + 10 for label in CLASSES}
        for label, query, source, page in search_jobs():
            if all(count >= TARGET for count in counts.values()):
                break
            if counts[label] >= TARGET or per_query[label][query] >= quota[label]:
                continue
            try:
                results = youtube_results(query.removeprefix("youtube: "), youtube) if source == "youtube" else google_results(query.removeprefix("google: "), youtube, page)
            except Exception as error:
                if "CAPTCHA" in str(error):
                    raise
                print(f"검색 실패: {query} / page={page}: {error}", flush=True)
                continue
            candidates, urls = [], set()
            for result in results:
                url = result.get("image")
                key = result["url"] if source == "youtube" else url
                if url and url not in seen and key not in attempted[label] and key not in urls and (source != "youtube" or key not in seen_videos):
                    candidates.append(result)
                    urls.add(key)
            downloaded = face_pass = accepted = 0
            for start in range(0, len(candidates), BATCH_SIZE):
                if counts[label] >= TARGET or per_query[label][query] >= quota[label]:
                    break
                batch = []
                pending = candidates[start:start + BATCH_SIZE]
                attempted[label].update(result["url"] if source == "youtube" else result["image"] for result in pending)
                for item in pool.map(download, pending):
                    if item is None:
                        continue
                    downloaded += 1
                    if any(item[2] - previous <= 4 for previous in hashes):
                        continue
                    if clear_face(detector, item[1]):
                        batch.append(item)
                face_pass += len(batch)
                if not batch:
                    continue
                scores = crawl.score_images([item[1] for item in batch], processor, model, text_features, device).numpy()
                for (result, image, phash), score in zip(batch, scores):
                    if counts[label] >= TARGET or per_query[label][query] >= quota[label]:
                        break
                    if score.argmax() != CLASSES.index(label) or score[CLASSES.index(label)] < CLIP_THRESHOLD:
                        continue
                    if any(phash - previous <= 4 for previous in hashes):
                        continue
                    number[label] += 1
                    path = ROOT / "data/images" / label / f"{label}_{number[label]:06d}.jpg"
                    temporary = path.with_suffix(".jpg.part")
                    try:
                        image.save(temporary, format="JPEG", quality=95)
                        writer.writerow({"path": path.relative_to(ROOT).as_posix(), "label": label, "query": query,
                                         "image_url": result["image"], "source_url": result.get("url", ""),
                                         "title": result.get("title", ""), "phash": str(phash),
                                         "filter_score": round(float(score[CLASSES.index(label)]), 4), "collection": "people"})
                        file.flush()
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
                    hashes.append(phash)
                    seen.add(result["image"])
                    if source == "youtube":
                        seen_videos.add(result["url"])
                    counts[label] += 1
                    per_query[label][query] += 1
                    accepted += 1
            print(f"{label}: {counts[label]}/{TARGET} | source={source} | candidates={len(candidates)} | download={downloaded} | face={face_pass} | saved={accepted} | page={page} | query={query}", flush=True)
    print("최종 수집 수량:", counts)
    if any(count < TARGET for count in counts.values()):
        raise RuntimeError(f"검색 범위 내 얼굴·CLIP 필터 통과 후보가 부족합니다. 확보 수량: {counts}")


if __name__ == "__main__":
    main()
