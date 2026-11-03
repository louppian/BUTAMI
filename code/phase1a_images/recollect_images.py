"""기존 이미지·수집 기록·bbox JSON을 백업한 뒤 1a 수집을 새로 시작한다."""

import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("no_glasses", "regular_glasses", "smart_glasses")


def archive_collection(root):
    paths = [path for label in CLASSES for path in (root / "data/images" / label).glob("*") if path.name != ".gitkeep"]
    paths += [root / "results/phase1a_images/metadata.csv", root / "data/annotations/face_glasses.json"]
    paths = [path for path in paths if path.exists()]
    if not paths:
        return None
    backup = root / "data/archive" / datetime.now().strftime("people_%Y%m%d_%H%M%S_%f")
    backup.mkdir(parents=True)
    print("기존 수집 데이터 백업:", backup)
    for path in paths:
        destination = backup / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
    return backup


def main():
    import image_crawl

    archive_collection(ROOT)
    print("새 사람 이미지 수집 시작. 중단 후 이어받기는 image_crawl.py를 실행하세요.")
    image_crawl.main()


if __name__ == "__main__":
    main()
