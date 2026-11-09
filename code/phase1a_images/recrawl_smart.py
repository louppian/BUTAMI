"""네이버·유튜브에서 투명 렌즈 스마트글래스 착용 사진을 재수집한다.

기존 프로젝트에서 image_crawl.py와 같은 폴더에 두고 실행한다.
"""

import csv
try:
    import fcntl
except ImportError:
    import msvcrt
import json
import re
import shutil
import subprocess
import os
import sqlite3
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np
from PIL import Image, ImageOps

import image_crawl as crawl

ROOT = Path(__file__).resolve().parents[2]
SEARCH_PYTHON = Path.home() / "AppData/Local/Programs/Python/Python313/python.exe"
TARGET = 500
BATCH_SIZE = 32
WORKERS = 16
DOWNLOAD_TIMEOUT = (8, 8)
CLIP_THRESHOLD = 0.7
FACE_THRESHOLD = 0.9
MIN_FACE_SIZE = 96
MIN_FACE_AREA = 0.01
FACE_MAX_SIZE = 960
LABEL = "smart_glasses"
PRODUCTS = (
    ("레이밴 메타", "Ray-Ban Meta"),
    ("레이밴 스토리즈", "Ray-Ban Stories"),
    ("엑스리얼 에어", "XREAL Air"),
    ("엑스리얼 원", "XREAL One"),
    ("비튜어", "VITURE"),
    ("로키드 맥스", "Rokid Max"),
    ("보스 프레임", "Bose Frames"),
    ("아마존 에코 프레임", "Amazon Echo Frames"),
    ("솔로스 에어고", "Solos AirGo"),
    ("이븐 리얼리티 G1", "Even Realities G1"),
    ("화웨이 아이웨어", "Huawei Eyewear"),
)
NAVER_SEARCH = "https://search.naver.com/search.naver"
NAVER_NEXT = "https://s.search.naver.com/"
NAVER_REFERER = {"Referer": "https://search.naver.com/"}
SEARCH_TIMEOUT = (8, 20)
IMAGE_DIR = Path("data/images") / LABEL
RESULT_DIR = Path("results/phase1a_images")
YUNET_WEIGHTS = Path("ckpt/pretrained/face_detection_yunet_2023mar.onnx")
LENS_PROMPTS = (
    "A photo of a person wearing eyeglasses with clear transparent lenses, with the eyes visible through the lenses.",
    "A photo of a person wearing sunglasses with dark or tinted lenses.",
    "A photo of a person without any glasses on their face.",
)
JSON_DECODER = json.JSONDecoder()
JSON_OBJECT = re.compile(r'\{(?=\s*")')
NEXT_URL = re.compile(r'(?<![\w$])(?:url|"url")\s*:\s*("(?:\\.|[^"\\])*")')


def image_objects(value):
    if isinstance(value, dict):
        if "originalUrl" in value:
            yield value
            return
        value = value.values()
    elif not isinstance(value, list):
        return
    for child in value:
        yield from image_objects(child)


def first_page_items(text):
    end = 0
    for match in JSON_OBJECT.finditer(text):
        if match.start() < end:
            continue
        try:
            value, end = JSON_DECODER.raw_decode(text, match.start())
        except ValueError:
            continue
        yield from image_objects(value)


def first_page_next(text):
    for match in NEXT_URL.finditer(text):
        try:
            url = json.loads(match[1])
        except ValueError:
            continue
        if url.startswith(NAVER_NEXT) and "pc_tab_more" in url:
            return url
    return None


class NaverSearch:
    def __init__(self):
        self.pages = defaultdict(list)

    def fetch(self, query, session, cursor):
        if cursor is None:
            response = session.get(
                NAVER_SEARCH, params={"where": "image", "query": query}, timeout=SEARCH_TIMEOUT
            )
            response.raise_for_status()
            return first_page_items(response.text), first_page_next(response.text)

        response = session.get(cursor, headers=NAVER_REFERER, timeout=SEARCH_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("items"), list):
            raise ValueError(f"네이버 이미지 응답에 items 목록이 없습니다 ({query}).")
        return data["items"], data.get("url")

    def __call__(self, query, session, page=1):
        if page < 1:
            raise ValueError("네이버 이미지 페이지는 1부터 시작합니다.")
        pages = self.pages[query]
        while len(pages) < page:
            cursor = pages[-1][1] if pages else None
            if pages and cursor is None:
                break
            items, following = self.fetch(query, session, cursor)

            candidates = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                image = item.get("originalUrl")
                if isinstance(image, str) and image.startswith(("https://", "http://")):
                    candidates.append({
                        "image": image,
                        "fallback_image": item.get("viewerThumb"),
                        "url": item.get("link", ""),
                        "title": item.get("title", ""),
                    })
            if (
                not isinstance(following, str)
                or not following.startswith(NAVER_NEXT)
                or any(following == previous for _, previous in pages)
            ):
                following = None
            pages.append((candidates, following))
        results = [item.copy() for item in pages[page - 1][0]] if page <= len(pages) else []
        print(f"NAVER {page}: {query} | {len(results)} candidates", flush=True)
        return results


def youtube_results(query, session):
    response = session.get("https://www.youtube.com/results", params={"search_query": query}, timeout=SEARCH_TIMEOUT)
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


class RetryDownload(Exception):
    pass


def download(result, *, retry=False):
    if not hasattr(LOCAL, "session"):
        LOCAL.session = crawl.requests.Session()
        LOCAL.session.headers["User-Agent"] = "Mozilla/5.0"
    transient = False
    errors = crawl.requests.exceptions
    for url in dict.fromkeys((result["image"], result.get("fallback_image"))):
        if not url:
            continue
        try:
            with LOCAL.session.get(url, timeout=DOWNLOAD_TIMEOUT, stream=True) as response:
                response.raise_for_status()
                started, content = time.perf_counter(), bytearray()
                for chunk in response.iter_content(65536):
                    content.extend(chunk)
                    if len(content) > 20 * 1024 * 1024:
                        break
                    if time.perf_counter() - started > 8:
                        transient = True
                        break
                else:
                    with Image.open(BytesIO(content)) as original:
                        if min(original.size) < 256 or original.width * original.height > 20_000_000:
                            continue
                        image = ImageOps.exif_transpose(original).convert("RGB")
                    return {**result, "image": url}, image, crawl.imagehash.phash(image)
        except errors.HTTPError as error:
            status = error.response.status_code
            transient |= status in (408, 425, 429) or status >= 500
        except (errors.Timeout, errors.ConnectionError, errors.ChunkedEncodingError):
            transient = True
        except (errors.RequestException, OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            continue
    if retry and transient:
        raise RetryDownload
    return None


def load_face_detector(root):
    weights = root / YUNET_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"YuNet 가중치를 준비하세요: {weights}")
    # 0.5 이상 얼굴 후보를 모두 찾고, 단일 얼굴의 점수는 0.9 이상으로 제한한다.
    return cv2.FaceDetectorYN.create(str(weights), "", (320, 320), 0.5, 0.3, 5000)


def face_reason(detector, image):
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
        return "face_score"
    x, y, w, h = face[:4] * np.array([width / size[0], height / size[1]] * 2)
    if min(w, h) < MIN_FACE_SIZE or w * h < width * height * MIN_FACE_AREA:
        return "face_size"
    if x < 0 or y < 0 or x + w > width or y + h > height:
        return "face_clipped"
    points = face[4:14].reshape(5, 2) * (width / size[0], height / size[1])
    return "face_pass" if ((points[:, 0] >= x) & (points[:, 0] < x + w)
                           & (points[:, 1] >= y) & (points[:, 1] < y + h)).all() else "face_landmarks"


def clear_face(detector, image):
    return face_reason(detector, image) == "face_pass"


LOCAL = threading.local()


def metadata_rows(path, *, smart):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "label" not in (reader.fieldnames or ()):
            raise ValueError(f"메타데이터에 label 열이 없습니다: {path}")
        return [row for row in reader if (row["label"] == LABEL) == smart]


def write_metadata(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=crawl.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_marker(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def publish(work, source, metadata, results, journal, state):
    archive = Path(state["archive"]).resolve()
    if archive.parent != results or not archive.name.startswith("smart_before_recrawl_"):
        raise ValueError(f"잘못된 백업 경로입니다: {archive}")

    new_images, old_images = work / IMAGE_DIR, archive / LABEL
    combined = work / "metadata_ready.csv"
    try:
        if new_images.exists():
            if not old_images.exists():
                source.rename(old_images)
            new_images.rename(source)
        if combined.exists():
            combined.replace(metadata)
    except OSError:
        if combined.exists() and old_images.exists():
            if not new_images.exists():
                source.rename(new_images)
            old_images.rename(source)
        raise

    write_marker(results / "smart_recrawl_backup.json", state)
    journal.replace(work / "published.json")
    print(f"기존 스마트글래스 보관: {archive}", flush=True)


def product_queries():
    priority = ("Even Realities G1", "Huawei Eyewear", "Amazon Echo Frames", "Solos AirGo", "Ray-Ban Meta", "Ray-Ban Stories")
    products = sorted(PRODUCTS, key=lambda pair: priority.index(pair[1]) if pair[1] in priority else len(priority))
    return tuple(query for korean, english in products for query in (
        f'"{english}" clear prescription lenses wearing -unboxing -teardown',
        f'"{korean}" 투명 도수렌즈 착용샷 -언박싱 -분해',
        f'"{english}" clear glasses on face -unboxing -teardown',
        f'"{korean}" 안경 착용 셀카 -언박싱 -분해'))


def search_jobs(queries):
    for query in queries:
        for site in ("bing", "youtube", "google", "naver"):
            yield f"public/{site}: {query}", query, site


def public_results(site, query):
    run = subprocess.run([str(SEARCH_PYTHON), str(ROOT / "code/phase1a_images/public_search.py")],
                         input=json.dumps({"site": site, "query": query}), capture_output=True,
                         text=True, encoding="utf-8", errors="replace", cwd=ROOT, timeout=600,
                         env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    if run.returncode:
        raise ValueError(run.stderr[-1500:])
    results = json.loads(run.stdout.strip().splitlines()[-1])
    print(f"{site}: {query} | {len(results)} candidates", flush=True)
    return results


def connect_state(database):
    db = sqlite3.connect(database, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS jobs(query TEXT PRIMARY KEY, results TEXT NOT NULL, done INTEGER DEFAULT 0)")
    db.execute("CREATE TABLE IF NOT EXISTS attempts(url TEXT PRIMARY KEY)")
    db.commit()
    return db


def enqueue(queue, item, stop):
    while not stop.is_set():
        try:
            queue.put(item, timeout=0.2)
            return
        except Full:
            continue


def search_producer(database, queries, queue, stop):
    naver = NaverSearch()
    try:
        with closing(connect_state(database)) as db, crawl.requests.Session() as session:
            session.headers["User-Agent"] = "Mozilla/5.0"
            for name, query, page in search_jobs(queries):
                if stop.is_set():
                    break
                cached = db.execute("SELECT results,done FROM jobs WHERE query=?", (name,)).fetchone()
                if cached and cached[1]:
                    continue
                if cached:
                    results = json.loads(cached[0])
                else:
                    try:
                        results = public_results(page, query)
                    except (crawl.requests.exceptions.RequestException, ValueError, subprocess.TimeoutExpired) as error:
                        print(f"검색 실패: {name}: {error}", flush=True)
                        continue
                    db.execute("INSERT INTO jobs(query,results) VALUES(?,?)", (name, json.dumps(results, ensure_ascii=False)))
                    db.commit()
                enqueue(queue, (name, results), stop)
    except Exception as error:
        enqueue(queue, error, stop)
    finally:
        enqueue(queue, None, stop)


def prepare_image(result, root):
    item = download(result, retry=True)
    if item is None:
        return None
    if not hasattr(LOCAL, "detector"):
        LOCAL.detector = load_face_detector(root)
    try:
        return item if clear_face(LOCAL.detector, item[1]) else None
    except cv2.error:
        return None


def load_lens_features(processor, model, device):
    import torch

    inputs = processor(text=list(LENS_PROMPTS), return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        features = model.get_text_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)


def score_batch(images, processor, model, text_features, lens_features, device):
    import torch

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        probabilities = (model.logit_scale.exp() * features @ text_features.T).float().softmax(dim=-1)
        lenses = (features @ lens_features.T).float()
        return torch.cat((probabilities, lenses), dim=-1).cpu().numpy()


def accepts_clear_glasses(score, class_index):
    return (
        score[:-3].argmax() == class_index
        and score[class_index] >= CLIP_THRESHOLD
        and score[-3] > max(score[-2], score[-1])
    )


@dataclass
class Job:
    name: str
    remaining: int = 0
    tested: int = 0
    faces: int = 0
    saved: int = 0
    retries: int = 0
    started: float = field(default_factory=time.perf_counter)


def collect(root, work, queries, reference_rows):
    images = work / IMAGE_DIR
    results = work / RESULT_DIR
    metadata = results / "metadata.csv"
    database = results / "smart_recrawl.sqlite3"
    rows = metadata_rows(metadata, smart=True)
    paths = {work / row["path"] for row in rows}
    if paths != set(images.glob("*.jpg")) or len(paths) != len(rows):
        raise ValueError(f"CSV와 JPEG를 확인하세요: {metadata}")
    count = len(rows)
    if count >= TARGET:
        return
    baseline = count
    seen = {row["image_url"] for row in (*reference_rows, *rows)}
    hashes = [int(row["phash"], 16) for row in (*reference_rows, *rows)]
    number = max((int(path.stem.rsplit("_", 1)[1]) for path in paths), default=0)
    weights = root / YUNET_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"YuNet 가중치를 준비하세요: {weights}")
    warnings.simplefilter("error", Image.DecompressionBombWarning)
    cv2.setNumThreads(1)
    processor, model, text_features, device = crawl.load_clip(root)
    lens_features = load_lens_features(processor, model, device)
    class_index = crawl.CLASSES.index(LABEL)
    queue, stop = Queue(4), threading.Event()
    producer = threading.Thread(target=search_producer, args=(database, queries, queue, stop), daemon=True)
    futures, exhausted = {}, False
    started = last_report = time.perf_counter()
    print(f"{LABEL} {count}/{TARGET} | 다운로드 {WORKERS}병렬 | CLIP={device}", flush=True)

    with closing(connect_state(database)) as db, metadata.open("a", encoding="utf-8", newline="") as file, ThreadPoolExecutor(max_workers=WORKERS) as pool:
        attempted = {url for (url,) in db.execute("SELECT url FROM attempts")}
        writer = csv.DictWriter(file, fieldnames=crawl.FIELDS)
        producer.start()
        try:
            while count < TARGET and (not exhausted or futures):
                while len(futures) < WORKERS * 4 and not exhausted:
                    try:
                        message = queue.get(timeout=0 if futures else 0.1)
                    except Empty:
                        break
                    if message is None:
                        exhausted = True
                        break
                    if isinstance(message, Exception):
                        raise message
                    name, candidates = message
                    job = Job(name)
                    for result in candidates:
                        url = result.get("image")
                        if not url or url in seen or url in attempted:
                            continue
                        attempted.add(url)
                        job.remaining += 1
                        futures[pool.submit(prepare_image, result, root)] = (job, url)
                    if not job.remaining:
                        db.execute("UPDATE jobs SET done=1 WHERE query=?", (name,))
                        db.commit()

                ready = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)[0] if futures else ()
                batch, completed = [], []
                for future in ready:
                    job, url = futures.pop(future)
                    job.tested += 1
                    try:
                        item = future.result()
                    except RetryDownload:
                        job.retries += 1
                        completed.append((job, None))
                        continue
                    completed.append((job, url))
                    if item is not None:
                        job.faces += 1
                        fingerprint = int(str(item[2]), 16)
                        if all((fingerprint ^ previous).bit_count() > 4 for previous in hashes):
                            batch.append((job, item, fingerprint))

                for offset in range(0, len(batch), BATCH_SIZE):
                    part = batch[offset:offset + BATCH_SIZE]
                    scores = score_batch([item[1] for _, item, _ in part], processor, model, text_features, lens_features, device)
                    for (job, (result, image, phash), fingerprint), score in zip(part, scores):
                        if count >= TARGET or not accepts_clear_glasses(score, class_index):
                            continue
                        if any((fingerprint ^ previous).bit_count() <= 4 for previous in hashes):
                            continue
                        number += 1
                        path = images / f"{LABEL}_{number:06d}.jpg"
                        temporary = path.with_suffix(".jpg.part")
                        image.save(temporary, format="JPEG", quality=95)
                        temporary.replace(path)
                        writer.writerow({
                            "path": path.relative_to(work).as_posix(), "label": LABEL, "query": job.name,
                            "image_url": result["image"], "source_url": result.get("url", ""),
                            "title": result.get("title", ""), "phash": str(phash),
                            "filter_score": round(float(score[class_index]), 4), "collection": "people",
                        })
                        file.flush()
                        hashes.append(fingerprint)
                        seen.add(result["image"])
                        count += 1
                        job.saved += 1

                for job, url in completed:
                    if url is not None:
                        db.execute("INSERT OR IGNORE INTO attempts VALUES(?)", (url,))
                    job.remaining -= 1
                    if not job.remaining:
                        if not job.retries:
                            db.execute("UPDATE jobs SET done=1 WHERE query=?", (job.name,))
                        print(
                            f"{LABEL} {count}/{TARGET} | 검사={job.tested} YuNet={job.faces} "
                            f"저장={job.saved} 재시도 대기={job.retries} "
                            f"{time.perf_counter()-job.started:.1f}s | {job.name}",
                            flush=True,
                        )
                if completed:
                    db.commit()
                now = time.perf_counter()
                if now - last_report >= 60:
                    elapsed, added = now - started, count - baseline
                    status = {
                        "counts": {LABEL: count}, "added": added, "elapsed_seconds": round(elapsed, 1),
                        "saved_per_minute": round(added * 60 / elapsed, 2), "inflight": len(futures),
                    }
                    write_marker(results / "crawl_status.json", status)
                    print(f"실측: {status}", flush=True)
                    last_report = now
        finally:
            stop.set()
            for future in futures:
                future.cancel()
            producer.join()

    if count < TARGET:
        raise RuntimeError(f"스마트글래스 {count}/{TARGET}장 수집. 다음 실행에서 이어갑니다.")


def recrawl(root, results, source, metadata):
    work = results / ".smart_recrawl_clear"
    journal = work / "publish.json"
    if journal.exists():
        publish(work, source, metadata, results, journal, json.loads(journal.read_text(encoding="utf-8")))
        return
    if (work / "published.json").exists():
        print("투명 렌즈 스마트글래스 수집 결과가 이미 반영됐습니다.", flush=True)
        return

    queries = product_queries()
    if not queries:
        raise ValueError("원래 스마트글래스 쿼리에서 제품명을 찾지 못했습니다.")
    if not source.is_dir():
        raise FileNotFoundError(source)

    work_images, work_results = work / IMAGE_DIR, work / RESULT_DIR
    work_images.mkdir(parents=True, exist_ok=True)
    work_results.mkdir(parents=True, exist_ok=True)
    work_metadata = work_results / "metadata.csv"
    reference_rows = metadata_rows(metadata, smart=False)
    if not work_metadata.exists():
        initial = work_metadata.with_suffix(".tmp")
        write_metadata(initial, ())
        initial.replace(work_metadata)

    collect(root, work, queries, reference_rows)

    rows = metadata_rows(work_metadata, smart=True)
    if len(rows) < TARGET:
        raise ValueError(f"스마트글래스 목표 수량을 확인하세요: {work_metadata}")
    for row in rows:
        path = work / row["path"]
        if path.parent != work_images or not path.is_file() or not path.stat().st_size:
            raise ValueError(f"CSV에 기록된 사진을 확인하세요: {path}")

    combined = work / "metadata_ready.csv"
    write_metadata(combined, metadata_rows(metadata, smart=False) + rows)
    shutil.copymode(metadata, combined)
    if results.stat().st_dev != source.parent.stat().st_dev or work.stat().st_dev != metadata.parent.stat().st_dev:
        raise OSError("기존 경로와 수집 경로가 같은 파일시스템에 있어야 사진과 CSV를 교체할 수 있습니다.")

    archive = results / f"smart_before_recrawl_{datetime.now():%Y%m%d_%H%M%S_%f}"
    archive.mkdir()
    shutil.copy2(metadata, archive / "metadata_all.csv")
    state = {"archive": str(archive), "queries": queries}
    write_marker(journal, state)
    publish(work, source, metadata, results, journal, state)


def main():
    root = ROOT.resolve()
    results = (root / RESULT_DIR).resolve()
    source = (root / IMAGE_DIR).resolve()
    metadata = (results / "metadata.csv").resolve()
    if not all(path.is_relative_to(root) for path in (results, source, metadata)):
        raise ValueError("사진과 메타데이터 경로가 프로젝트 폴더 안에 있어야 합니다.")
    with (results / ".smart_recrawl.lock").open("a") as lock:
        try:
            if "fcntl" in globals():
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("스마트글래스 재수집이 이미 실행 중입니다.") from None
        recrawl(root, results, source, metadata)


if __name__ == "__main__":
    main()
