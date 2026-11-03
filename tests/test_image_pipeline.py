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
train = load("train", "code/phase3a_train/train_mlp.py")
recollect = load("recollect", "code/phase1a_images/recollect_images.py")


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
        detector.detect.return_value = (0, np.array([[100, 50, 200, 100] + [0] * 10 + [0.92], [600, 300, 100, 100] + [0] * 10 + [0.87]], dtype=np.float32))
        cv2 = types.SimpleNamespace(resize=lambda image, size: np.zeros((size[1], size[0], 3), dtype=np.uint8))
        with patch.dict("sys.modules", {"cv2": cv2}):
            result = bbox.detect_faces(detector, Image.new("RGB", (1600, 800)))
        detector.setInputSize.assert_called_once_with((640, 320))
        self.assertEqual([box for box, score in result], [[250, 125, 500, 250], [1500, 750, 100, 50]])
        self.assertAlmostEqual(result[0][1], 0.92, places=6)
        self.assertAlmostEqual(result[1][1], 0.87, places=6)

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
        segmenter = Mock(side_effect=[[[256, 256, 0, 0, 0.05], [64, 64, 192, 192, 0.9]], [[0, 0, 256, 256, 0.8]]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_images(root)
            with patch.object(bbox, "BATCH_SIZE", 2), \
                 patch.object(bbox, "detect_faces", side_effect=[[([10, 10, 40, 40], 0.97)], [([0, 0, 20, 20], 0.96), ([90, 60, 30, 40], 0.95)], []]), \
                 redirect_stdout(io.StringIO()):
                data = bbox.extract(root, Mock(), segmenter)
            saved = json.loads((root / "data/annotations/face_glasses.json").read_text())
            self.assertEqual(saved, data)
            self.assertEqual([len(call.args[0]) for call in segmenter.call_args_list], [2, 1])
            self.assertEqual(len(data["images"]), 3)
            self.assertIsNotNone(data["images"][0]["selected_face_id"])
            self.assertEqual(data["images"][0]["selection_status"], "selected")
            self.assertIsNone(data["images"][1]["selected_face_id"])
            self.assertEqual(data["images"][1]["selection_status"], "multiple_faces")
            self.assertEqual(data["images"][2]["selection_status"], "no_face")
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
                self.assertTrue(0 <= annotation["score"] <= 1)
                if annotation["category_id"] == 1:
                    self.assertIn("glasses_probability", annotation)
                else:
                    face = next(a for a in data["annotations"] if a["id"] == annotation["face_id"])
                    self.assertEqual(face["category_id"], 1)
                    self.assertEqual(face["image_id"], annotation["image_id"])
                    self.assertEqual(face["glasses_probability"], annotation["score"])
            self.assertTrue(all(not Path(image["file_name"]).is_absolute() for image in data["images"]))

    def test_face_score_threshold(self):
        detector = Mock()
        detector.detect.return_value = (0, np.array([[10, 10, 20, 20] + [0] * 10 + [score] for score in [0.49, 0.5, 0.51, float("nan")]]))
        with patch.dict("sys.modules", {"cv2": types.SimpleNamespace()}):
            faces = bbox.detect_faces(detector, Image.new("RGB", (100, 100)))
        self.assertEqual([score for box, score in faces], [0.5, 0.51])

    def test_nonempty_glasses_masks_are_filtered_by_presence_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_images(root)
            segmenter = Mock(return_value=[[0, 0, 256, 256, p] for p in [0.039, 0.499, 0.5]])
            with patch.object(bbox, "detect_faces", return_value=[([10, 10, 40, 40], 0.95)]), redirect_stdout(io.StringIO()):
                data = bbox.extract(root, Mock(), segmenter)
            faces = [a for a in data["annotations"] if a["category_id"] == 1]
            glasses = [a for a in data["annotations"] if a["category_id"] == 2]
            self.assertEqual([a["glasses_probability"] for a in faces], [0.039, 0.499, 0.5])
            self.assertEqual(len(glasses), 1)
            self.assertEqual(glasses[0]["image_id"], 3)
            self.assertEqual(glasses[0]["score"], 0.5)
            self.assertEqual(glasses[0]["face_id"], faces[2]["id"])
            self.assertTrue(all(a["score"] == 0.95 for a in faces))

    def test_multiple_people_use_their_own_presence_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_images(root)
            segmenter = Mock(return_value=[[0, 0, 256, 256, 0.1], [0, 0, 256, 256, 0.95]])
            with patch.object(bbox, "detect_faces", side_effect=[[([1, 1, 20, 20], 0.8), ([60, 30, 20, 20], 0.9)], [], []]), redirect_stdout(io.StringIO()):
                data = bbox.extract(root, Mock(), segmenter)
            faces = [a for a in data["annotations"] if a["category_id"] == 1]
            glasses = [a for a in data["annotations"] if a["category_id"] == 2]
            self.assertEqual(len(glasses), 1)
            self.assertEqual(glasses[0]["face_id"], faces[1]["id"])
            self.assertEqual(glasses[0]["score"], 0.95)
            self.assertGreater(glasses[0]["bbox"][0], 40)

    def test_extraction_and_training_keep_empty_mask_separate_from_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_images(root)
            segmenter = Mock(return_value=[[256, 256, 0, 0, 0.99], [0, 0, 256, 256, 0.01], [0, 0, 256, 256, 0.99]])
            with patch.object(bbox, "detect_faces", return_value=[([1, 1, 20, 20], 0.9)]), redirect_stdout(io.StringIO()):
                data = bbox.extract(root, Mock(), segmenter)
            samples, excluded = train.build_samples(data)
            self.assertEqual(excluded, [])
            self.assertEqual([s["glasses_status"] for s in samples], ["empty_mask", "below_threshold", "detected"])
            metrics, records = train.evaluate_predictions(samples, [0, 1, 2], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
            self.assertEqual(metrics["test_accuracy"], 1.0)
            self.assertEqual(records[0]["prediction"], "no_glasses")
            self.assertEqual(records[0]["final_class"], "review")
            self.assertEqual(records[0]["reason"], "presence_conflict|glasses_bbox_missing")
            self.assertFalse(samples[0]["glasses_bbox_present"])

    def test_failure_preserves_existing_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_images(root)
            output = root / "data/annotations/face_glasses.json"
            output.parent.mkdir(parents=True)
            output.write_text('{"previous": true}\n')
            with patch.object(bbox, "detect_faces", return_value=[([1, 1, 20, 20], 0.9)]), self.assertRaises(RuntimeError):
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



class FaceSelectionTests(unittest.TestCase):
    def test_single_large_face_is_selected_over_tiny_high_score_face(self):
        faces = [{"id": 1, "score": 0.99, "area": 50}, {"id": 2, "score": 0.8, "area": 1000}]
        self.assertEqual(bbox.select_face(faces, 100, 100), (2, "selected"))

    def test_multiple_faces_are_kept_for_review(self):
        faces = [{"id": 1, "score": 0.99, "area": 1000}, {"id": 2, "score": 0.8, "area": 2000}]
        self.assertEqual(bbox.select_face(faces, 100, 100), (None, "multiple_faces"))

    def test_no_face_and_tiny_face_are_review_cases(self):
        self.assertEqual(bbox.select_face([], 100, 100), (None, "no_face"))
        self.assertEqual(bbox.select_face([{"id": 1, "score": 0.99, "area": 50}], 100, 100), (None, "no_eligible_face"))


class TrainingTests(unittest.TestCase):
    def data(self):
        images, annotations = [], []
        for i, label in enumerate(train.CLASSES, 1):
            images.append({"id": i, "file_name": f"{label}/sample.jpg", "width": 100, "height": 80,
                           "selected_face_id": i + 10, "selection_status": "selected"})
            annotations.append({"id": i + 10, "image_id": i, "category_id": 1, "bbox": [10, 10, 30, 30],
                                "score": 0.9, "glasses_probability": [0.1, 0.99, 0.9][i - 1],
                                "glasses_status": ["below_threshold", "detected", "empty_mask"][i - 1]})
        annotations.append({"id": 21, "image_id": 2, "category_id": 2, "face_id": 12,
                            "bbox": [10, 12, 30, 8], "score": 0.99})
        return {"images": images, "annotations": annotations}

    def test_training_labels_are_preserved_despite_presence_model_disagreement(self):
        data = self.data()
        data["annotations"][1]["glasses_probability"] = 0.01
        data["annotations"][1]["glasses_status"] = "below_threshold"
        data["annotations"].pop()
        samples, excluded = train.build_samples(data)
        self.assertEqual([s["label"] for s in samples], [0, 1, 2])
        self.assertEqual(excluded, [])
        self.assertFalse(samples[1]["glasses_bbox_present"])

    def test_missing_face_selection_is_reported(self):
        data = self.data()
        data["images"][2]["selected_face_id"] = None
        data["images"][2]["selection_status"] = "multiple_faces"
        samples, excluded = train.build_samples(data)
        self.assertEqual(len(samples), 2)
        self.assertEqual(excluded, [{"image_id": 3, "file_name": "smart_glasses/sample.jpg", "reason": "multiple_faces"}])

    def test_old_schema_and_wrong_image_face_are_rejected(self):
        data = self.data()
        del data["images"][0]["selected_face_id"]
        with self.assertRaises(ValueError):
            train.build_samples(data)
        data = self.data()
        data["images"][0]["selected_face_id"] = 12
        with self.assertRaises(ValueError):
            train.build_samples(data)

    def test_face_score_filter_is_strictly_greater_than_half(self):
        data = self.data()
        data["annotations"][0]["score"] = 0.5
        samples, excluded = train.build_samples(data)
        self.assertEqual([s["label"] for s in samples], [1, 2])
        self.assertEqual(excluded[0]["reason"], "face_score")

    def test_raw_predictions_and_probabilities_survive_conflicts(self):
        samples, _ = train.build_samples(self.data())
        probabilities = [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]
        metrics, records = train.evaluate_predictions(samples, [0, 1, 2], probabilities)
        self.assertEqual([r["prediction"] for r in records], ["regular_glasses", "no_glasses", "smart_glasses"])
        self.assertEqual([r["final_class"] for r in records], ["review", "review", "smart_glasses"])
        self.assertEqual(metrics["confusion_matrix"], [[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        self.assertAlmostEqual(metrics["test_accuracy"], 1 / 3)
        self.assertEqual(metrics["presence_conflicts"], 2)
        self.assertEqual(metrics["missing_glasses_bbox"], 1)
        self.assertEqual(metrics["review_count"], 2)
        self.assertEqual(records[1]["glasses_probability"], 0.99)
        self.assertEqual(records[1]["p_no_glasses"], 0.7)
        self.assertEqual(records[2]["reason"], "glasses_bbox_missing")
        for original, row in zip(probabilities, records):
            self.assertEqual(original, [row["p_no_glasses"], row["p_regular_glasses"], row["p_smart_glasses"]])

    def test_split_is_reproducible_disjoint_and_contains_every_class(self):
        samples = [{"label": label} for label in range(3) for _ in range(5)]
        train_ids, test_ids = train.stratified_split(samples)
        self.assertEqual((train_ids, test_ids), train.stratified_split(samples))
        self.assertFalse(set(train_ids) & set(test_ids))
        self.assertEqual(set(train_ids) | set(test_ids), set(range(15)))
        self.assertEqual({samples[i]["label"] for i in train_ids}, {0, 1, 2})
        self.assertEqual({samples[i]["label"] for i in test_ids}, {0, 1, 2})

    def test_error_plot_contains_face_and_its_glasses_boxes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = self.data()
        samples, _ = train.build_samples(data)
        _, records = train.evaluate_predictions(samples, [1], [[0.7, 0.2, 0.1]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "data/images/regular_glasses/sample.jpg"
            image.parent.mkdir(parents=True)
            Image.new("RGB", (100, 80)).save(image)
            fig, ax = plt.subplots()
            train.draw_error(ax, root, records[0], {a["id"]: a for a in data["annotations"]}, {12: [data["annotations"][-1]]})
            self.assertEqual(len(ax.patches), 2)
            self.assertEqual(ax.patches[0].get_bbox().bounds, (10, 10, 30, 30))
            self.assertEqual(ax.patches[1].get_bbox().bounds, (10, 12, 30, 8))
            self.assertIn("regular_glasses → no_glasses", ax.get_title())
            self.assertTrue(any("P(glasses)=0.990" in text.get_text() for text in ax.texts))
            plt.close(fig)



class RecollectionTests(unittest.TestCase):
    def test_collection_is_archived_with_annotations_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {f"data/images/{label}/sample.jpg": f"old-{label}".encode() for label in recollect.CLASSES}
            files.update({"results/phase1a_images/metadata.csv": b"old metadata", "data/annotations/face_glasses.json": b'{"old": true}'})
            for name, content in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            placeholder = root / "data/images/no_glasses/.gitkeep"
            placeholder.write_text("\n")
            docs = root / "results/phase1a_images/results.md"
            docs.write_text("keep documentation")
            weights = root / "ckpt/pretrained/model.pth"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"keep weights")
            with redirect_stdout(io.StringIO()):
                backup = recollect.archive_collection(root)
            for name, content in files.items():
                self.assertFalse((root / name).exists())
                self.assertEqual((backup / name).read_bytes(), content)
            self.assertEqual(docs.read_text(), "keep documentation")
            self.assertEqual(weights.read_bytes(), b"keep weights")
            self.assertEqual(placeholder.read_text(), "\n")
            self.assertFalse((backup / "data/images/no_glasses/.gitkeep").exists())

    def test_empty_collection_has_no_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(recollect.archive_collection(root))
            self.assertFalse((root / "data/archive").exists())

    def test_new_collection_starts_after_old_annotation_is_archived(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "data/annotations/face_glasses.json"
            annotation.parent.mkdir(parents=True)
            annotation.write_text('{"old": true}')
            called = []

            def collect():
                self.assertFalse(annotation.exists())
                self.assertEqual(len(list((root / "data/archive").iterdir())), 1)
                called.append(True)

            with patch.object(recollect, "ROOT", root), patch.dict("sys.modules", {"image_crawl": types.SimpleNamespace(main=collect)}), redirect_stdout(io.StringIO()):
                recollect.main()
            self.assertEqual(called, [True])


if __name__ == "__main__":
    unittest.main()
