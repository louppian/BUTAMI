"""모델·네트워크를 대체해 수집 저장·이어받기와 bbox/COCO 좌표를 검사한다."""

import csv
import importlib.util
import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Hash:
    def __init__(self, value):
        self.value = value

    def __sub__(self, other):
        return 0 if self.value == other.value else 10

    def __str__(self):
        return str(self.value)


with patch.dict("sys.modules", {
    "requests": types.SimpleNamespace(), "ddgs": types.SimpleNamespace(DDGS=Mock()),
    "imagehash": types.SimpleNamespace(phash=lambda image: Hash(image.getpixel((0, 0))[0])),
}):
    crawl = load("crawl", "code/phase1a_images/image_crawl.py")
bbox = load("bbox", "code/phase2a_extract/extract_bbox.py")


class CollectionTests(unittest.TestCase):
    def test_three_classes_filter_duplicates_target_and_resume(self):
        queries = {label: [str(i)] for i, label in enumerate(crawl.CLASSES)}

        class Search:
            def images(self, query, **kwargs):
                label = int(query[0])
                return [{"image": f"{label}-{i}", "label_id": label, "item": i} for i in range(5)]

        def download(result):
            image = Image.new("RGB", (300, 300), (result["label_id"] * 40 + result["item"], 0, 0))
            score = np.zeros(len(crawl.DESCRIPTIONS))
            score[(result["label_id"] + 1) % 3 if result["item"] == 0 else result["label_id"]] = 0.6 if result["item"] == 1 else 0.95
            image.info["scores"] = score
            # 후보 2/3은 같은 이미지다. 배치가 달라도 중복을 제거한다.
            return result, image, Hash(result["label_id"] * 40 + (2 if result["item"] == 3 else result["item"]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.multiple(crawl, ROOT=root, TARGET=2, BATCH_SIZE=3, WORKERS=1, QUERIES=queries), \
                 patch.object(crawl, "DDGS", return_value=Search()), patch.object(crawl, "download", side_effect=download), \
                 patch.object(crawl, "load_clip", return_value=(None, None, None, "cpu")) as model, \
                 patch.object(crawl, "score_images", side_effect=lambda images, *args: np.stack([image.info["scores"] for image in images])), redirect_stdout(io.StringIO()):
                crawl.main()
                self.assertEqual(model.call_count, 1)
                with (root / "results/phase1a_images/metadata.csv").open(newline="") as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 6)
                for label in crawl.CLASSES:
                    self.assertEqual(len(list((root / "data/images" / label).glob("*.jpg"))), 2)
                    self.assertEqual(sum(row["label"] == label for row in rows), 2)
                self.assertTrue(all(row["collection"] == "people" for row in rows))
                model.reset_mock()
                crawl.main()
                model.assert_not_called()
                with (root / "results/phase1a_images/metadata.csv").open(newline="") as file:
                    self.assertEqual(len(list(csv.DictReader(file))), 6)

    def test_rejected_url_can_be_collected_under_correct_class(self):
        class Search:
            def images(self, *args, **kwargs):
                return [{"image": "same-url"}]

        scores = np.zeros((1, len(crawl.DESCRIPTIONS)))
        scores[0, 2] = 0.95
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.multiple(crawl, ROOT=root, TARGET=1, WORKERS=1), \
                 patch.object(crawl, "DDGS", return_value=Search()), \
                 patch.object(crawl, "download", return_value=({"image": "same-url"}, Image.new("RGB", (300, 300)), Hash(100))) as download, \
                 patch.object(crawl, "load_clip", return_value=(None, None, None, "cpu")), \
                 patch.object(crawl, "score_images", return_value=scores), redirect_stdout(io.StringIO()):
                crawl.main()
            self.assertEqual(download.call_count, 3)
            self.assertEqual(len(list((root / "data/images/smart_glasses").glob("*.jpg"))), 1)
            with (root / "results/phase1a_images/metadata.csv").open(newline="") as file:
                self.assertEqual([row["label"] for row in csv.DictReader(file)], ["smart_glasses"])

    def test_existing_product_records_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "results/phase1a_images/metadata.csv"
            metadata.parent.mkdir(parents=True)
            old = "path,label,query,image_url,source_url,title,phash,filter_score\n"
            metadata.write_text(old)
            with patch.object(crawl, "ROOT", root), self.assertRaises(ValueError):
                crawl.main()
            self.assertEqual(metadata.read_text(), old)

    def test_unrecorded_images_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "data/images/no_glasses/old.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (300, 300)).save(image)
            original = image.read_bytes()
            with patch.object(crawl, "ROOT", root), self.assertRaises(ValueError):
                crawl.main()
            self.assertEqual(image.read_bytes(), original)


class BboxTests(unittest.TestCase):
    def test_face_resize_restores_original_coordinates(self):
        detector = Mock()
        detector.detect.return_value = (0, np.array([[100, 50, 200, 100], [600, 300, 100, 100]], dtype=np.float32))
        cv2 = types.SimpleNamespace(resize=lambda image, size: np.zeros((size[1], size[0], 3), dtype=np.uint8))
        with patch.dict("sys.modules", {"cv2": cv2}):
            result = bbox.detect_faces(detector, Image.new("RGB", (1600, 800)))
        detector.setInputSize.assert_called_once_with((640, 320))
        self.assertEqual(result, [[250, 125, 500, 250], [1500, 750, 100, 50]])

    def test_crop_projection_clipping_and_empty_mask(self):
        self.assertEqual(bbox.project_box([64, 64, 192, 192], [10, 20, 100, 80]), [35, 40, 50, 40])
        self.assertIsNone(bbox.project_box([256, 256, 0, 0], [10, 20, 100, 80]))
        self.assertEqual(bbox.clip_box([-5.5, -1.5, 20, 12], 10, 10), [0, 0, 10, 10])
        self.assertIsNone(bbox.clip_box([float("nan"), 0, 10, 10], 100, 100))
        self.assertIsNone(bbox.clip_box([20, 20, 0, 2], 100, 100))

    def make_images(self, root):
        for label, size in zip(bbox.CLASSES, [(100, 80), (120, 100), (140, 110)]):
            path = root / "data/images" / label / f"{label}_000001.jpg"
            path.parent.mkdir(parents=True)
            Image.new("RGB", size).save(path)

    def test_coco_multiple_faces_empty_glasses_and_no_face_image(self):
        segmenter = Mock(side_effect=[[[256, 256, 0, 0], [64, 64, 192, 192]], [[0, 0, 256, 256]]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_images(root)
            with patch.object(bbox, "BATCH_SIZE", 2), \
                 patch.object(bbox, "detect_faces", side_effect=[[[10, 10, 40, 40]], [[0, 0, 20, 20], [90, 60, 30, 40]], []]), \
                 redirect_stdout(io.StringIO()):
                data = bbox.extract(root, Mock(), segmenter)
            saved = json.loads((root / "data/annotations/face_glasses.json").read_text())
            self.assertEqual(saved, data)
            self.assertEqual([len(call.args[0]) for call in segmenter.call_args_list], [2, 1])
            self.assertEqual(len(data["images"]), 3)
            self.assertEqual(sum(a["category_id"] == 1 for a in data["annotations"]), 3)
            self.assertEqual(sum(a["category_id"] == 2 for a in data["annotations"]), 2)
            self.assertFalse(any(a["image_id"] == 1 and a["category_id"] == 2 for a in data["annotations"]))
            self.assertFalse(any(a["image_id"] == 3 for a in data["annotations"]))
            self.assertEqual(len({a["id"] for a in data["annotations"]}), len(data["annotations"]))
            for annotation in data["annotations"]:
                image = data["images"][annotation["image_id"] - 1]
                x, y, w, h = annotation["bbox"]
                self.assertTrue(0 <= x < x + w <= image["width"])
                self.assertTrue(0 <= y < y + h <= image["height"])
                self.assertEqual(annotation["area"], w * h)
                self.assertEqual(annotation["iscrowd"], 0)
            self.assertTrue(all(not Path(image["file_name"]).is_absolute() for image in data["images"]))

    def test_failure_preserves_existing_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_images(root)
            output = root / "data/annotations/face_glasses.json"
            output.parent.mkdir(parents=True)
            output.write_text('{"previous": true}\n')
            with patch.object(bbox, "detect_faces", return_value=[[1, 1, 20, 20]]), self.assertRaises(RuntimeError):
                bbox.extract(root, Mock(), Mock(side_effect=RuntimeError("inference failed")))
            self.assertEqual(output.read_text(), '{"previous": true}\n')
            self.assertFalse(output.with_suffix(".json.tmp").exists())

    def test_empty_input(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            bbox.extract(Path(directory), Mock(), Mock())

    def test_weights_download_reuse_and_invalid_download_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pretrained/model.onnx"
            with patch.object(bbox, "urlopen", return_value=io.BytesIO(b"x" * 120_000)) as network:
                self.assertEqual(bbox.ensure_weights(path, "https://example.test/model"), path)
                bbox.ensure_weights(path, "https://example.test/model")
                self.assertEqual(network.call_count, 1)
            self.assertEqual(path.stat().st_size, 120_000)
            path.unlink()
            with patch.object(bbox, "urlopen", return_value=io.BytesIO(b"version https://git-lfs.github.com/spec/v1\n")), self.assertRaises(ValueError):
                bbox.ensure_weights(path, "https://example.test/model")
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".onnx.part").exists())


if __name__ == "__main__":
    unittest.main()
