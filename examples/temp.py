#!/usr/bin/env python3
"""Detect faces with YuNet, then classify/localize glasses on each crop.

Labels: C = glasses presence probability; D = glasses detector score.
Green boxes require both thresholds. Models can still make mistakes.
Requires Python 3.12+, glasses-detector, opencv-python-headless and FFmpeg.
YuNet weights are downloaded from the official OpenCV model zoo if missing.
"""
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import uuid
import urllib.request
import sys
from types import SimpleNamespace


# 설정은 여기서 직접 수정합니다. 실행 인자는 사용하지 않습니다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_VIDEO = PROJECT_ROOT / "data" / "videos" / "video1.mp4"
OUTPUT_VIDEO = PROJECT_ROOT / "results" / "phase2b_video_check" / "video1_glasses_30fps.mp4"
FACE_MODEL = PROJECT_ROOT / "ckpt" / "pretrained" / "face_detection_yunet_2023mar.onnx"
DEVICE = "cuda"  # GPU가 없으면 "cpu"
FPS = 30
CLASSIFICATION_THRESHOLD = 0.5
DETECTION_THRESHOLD = 0.3
FACE_THRESHOLD = 0.7
FACE_EXPAND = 1.8
BATCH_SIZE = 8
GLASSES_KIND = "anyglasses"
MAX_SECONDS = None  # 먼저 10으로 설정해 미리보기; 전체 영상은 None

# 이전 위치에 설치된 패키지 대신 이 저장소의 외부 소스를 사용합니다.
THIRD_PARTY_SOURCE = PROJECT_ROOT / "third_party" / "glasses-detector" / "src"
sys.path.insert(0, str(THIRD_PARTY_SOURCE))

def ensure_face_model(path):
    if path.is_file() and path.stat().st_size > 10000:
        return
    if path.exists():
        raise RuntimeError(f"Invalid face model file; set FACE_MODEL to a different path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.download")
    url = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    print("Downloading OpenCV YuNet face model...", flush=True)
    try:
        urllib.request.urlretrieve(url, temporary)
        if temporary.stat().st_size < 10000:
            raise RuntimeError("Downloaded face model is not a valid ONNX binary")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class FaceGlassesPipeline:
    def __init__(self, args, classifier, detector, font, width, height):
        import cv2
        self.cv2, self.args = cv2, args
        self.classifier, self.detector, self.font = classifier, detector, font
        self.width, self.height = width, height
        self.face_detector = cv2.FaceDetectorYN.create(
            str(args.face_model), "", (320, 320), args.face_threshold, 0.3, 5000
        )

    def face_crops(self, image):
        import numpy as np
        w, h = image.size
        scale = min(1.0, 1280 / max(w, h))
        sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
        rgb = np.asarray(image.resize((sw, sh)))
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        self.face_detector.setInputSize((sw, sh))
        _, faces = self.face_detector.detect(bgr)
        result = []
        if faces is None:
            return result
        for face in faces:
            x, y, fw, fh = [float(v) for v in face[:4]]
            x, fw = x * w / sw, fw * w / sw
            y, fh = y * h / sh, fh * h / sh
            if min(fw, fh) < 12:
                continue
            cx, cy = x + fw / 2, y + fh / 2
            side = max(fw, fh) * self.args.face_expand
            left, top = max(0, math.floor(cx-side/2)), max(0, math.floor(cy-side/2))
            right, bottom = min(w, math.ceil(cx+side/2)), min(h, math.ceil(cy+side/2))
            if right > left and bottom > top:
                result.append(((left, top, right, bottom), image.crop((left, top, right, bottom))))
        return result

    def annotate(self, images):
        import torch
        from PIL import ImageDraw
        jobs = []
        face_counts = [0] * len(images)
        for frame_id, image in enumerate(images):
            for region, crop in self.face_crops(image):
                jobs.append((frame_id, region, crop))
                face_counts[frame_id] += 1
        annotations = [[] for _ in images]
        for start in range(0, len(jobs), self.args.batch_size):
            chunk = jobs[start:start+self.args.batch_size]
            with torch.inference_mode():
                probabilities = self.classifier.predict([job[2] for job in chunk], format="proba")
                selected = []
                for i, value in enumerate(probabilities):
                    if not math.isfinite(float(value)):
                        raise RuntimeError("Non-finite classification probability")
                    if float(value) >= self.args.threshold:
                        selected.append(i)
                scores = []
                def collect_scores(module, inputs, outputs):
                    scores.extend([out["scores"].detach().cpu().tolist() for out in outputs])
                if not selected:
                    continue
                hook = self.detector.model.register_forward_hook(collect_scores)
                try:
                    boxes = self.detector.predict([chunk[i][2] for i in selected], format="int")
                finally:
                    hook.remove()
            if len(scores) != len(boxes) or len(boxes) != len(selected):
                raise RuntimeError("Mismatch between detector boxes and scores")
            for local_id, crop_boxes, crop_scores in zip(selected, boxes, scores):
                frame_id, (left, top, right, bottom), _ = chunk[local_id]
                if len(crop_boxes) != len(crop_scores):
                    raise RuntimeError("Mismatch between number of boxes and scores")
                for box, score in zip(crop_boxes, crop_scores):
                    if not math.isfinite(float(score)) or float(score) < self.args.det_threshold:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    x1, x2 = max(left, min(right-1, left+x1)), max(left, min(right-1, left+x2))
                    y1, y2 = max(top, min(bottom-1, top+y1)), max(top, min(bottom-1, top+y2))
                    if x2 > x1 and y2 > y1:
                        annotations[frame_id].append(((x1,y1,x2,y2), float(probabilities[local_id]), float(score)))
        for image, items in zip(images, annotations):
            draw = ImageDraw.Draw(image)
            for box, probability, score in items:
                draw.rectangle(box, outline=(0,255,0), width=max(3, self.width//400))
                text = f"glasses C:{probability:.1%} D:{score:.1%}"
                bounds = draw.textbbox((0,0), text, font=self.font)
                tw, th = bounds[2]-bounds[0], bounds[3]-bounds[1]
                tx = max(0, min(box[0], self.width-tw-8))
                ty = max(0, box[1]-th-10)
                draw.rectangle((tx,ty,tx+tw+8,ty+th+8), fill=(0,255,0))
                draw.text((tx+4-bounds[0],ty+4-bounds[1]),text,font=self.font,fill="black")
        return images, sum(len(items) for items in annotations), sum(face_counts)


def main():
    args = SimpleNamespace(
        input=Path(INPUT_VIDEO), output=Path(OUTPUT_VIDEO), fps=FPS,
        threshold=CLASSIFICATION_THRESHOLD, batch_size=BATCH_SIZE,
        device=DEVICE, kind=GLASSES_KIND, det_threshold=DETECTION_THRESHOLD,
        face_threshold=FACE_THRESHOLD, face_expand=FACE_EXPAND,
        face_model=Path(FACE_MODEL), max_seconds=MAX_SECONDS,
    )
    if not math.isfinite(args.fps) or args.fps <= 0 or args.batch_size < 1:
        raise ValueError("FPS and batch size must be positive")
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not 0 <= args.det_threshold <= 1 or not 0 <= args.face_threshold <= 1:
        raise ValueError("Detector thresholds must be between 0 and 1")
    if not math.isfinite(args.face_expand) or args.face_expand <= 0:
        raise ValueError("Face expansion must be positive")
    if args.max_seconds is not None and (not math.isfinite(args.max_seconds) or args.max_seconds <= 0):
        raise ValueError("Preview duration must be positive")
    if not args.input.is_file():
        raise ValueError(f"Input does not exist: {args.input}")
    if args.output.exists() or args.output.resolve() == args.input.resolve():
        raise ValueError("Output must be a new file, different from input")
    for executable in ["ffmpeg", "ffprobe"]:
        if not shutil.which(executable):
            raise ValueError(f"{executable} is missing; install FFmpeg first")
    encoders_text = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.STDOUT, text=True)
    encoders = {parts[1] for line in encoders_text.splitlines() if len(parts := line.split()) >= 2}
    if "libx264" in encoders:
        encoding_options = ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
    elif "mpeg4" in encoders:
        encoding_options = ["-c:v", "mpeg4", "-q:v", "3"]
    else:
        raise ValueError("FFmpeg has neither libx264 nor mpeg4 encoder")
    print(f"Video encoder: {encoding_options[1]}", flush=True)
    ensure_face_model(args.face_model)

    import torch
    from PIL import Image, ImageDraw, ImageFont
    from glasses_detector import GlassesClassifier, GlassesDetector

    torch.hub.set_dir(str(PROJECT_ROOT / "ckpt" / "pretrained" / "torch"))

    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(args.input),
    ]))
    if not probe.get("streams"):
        raise RuntimeError("No video stream found")
    width, height = (int(probe["streams"][0][key]) for key in ["width", "height"])
    classifier = GlassesClassifier(kind=args.kind, size="medium", device=args.device)
    detector = GlassesDetector(kind="worn", size="medium", device=args.device)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", max(16, height // 40))
    except OSError:
        font = ImageFont.load_default()
    pipeline = FaceGlassesPipeline(args, classifier, detector, font, width, height)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + f".{uuid.uuid4().hex}.partial.mp4")
    decoded_bytes = width * height * 3
    count = positives = 0
    faces_seen = 0
    duration_options = ["-t", str(args.max_seconds)] if args.max_seconds is not None else []
    started = time.monotonic()
    decoder = encoder = None
    with tempfile.TemporaryFile() as decode_log, tempfile.TemporaryFile() as encode_log:
        try:
            # FFmpeg applies orientation metadata before reporting raw frames.
            # Disable autorotation to keep the probed dimensions and coordinates consistent.
            decoder = subprocess.Popen([
                "ffmpeg", "-nostdin", "-v", "error", "-noautorotate", "-i", str(args.input),
                "-map", "0:v:0", "-an", "-vf", f"fps={args.fps}",
                *duration_options,
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ], stdout=subprocess.PIPE, stderr=decode_log)
            encoder = subprocess.Popen([
                "ffmpeg", "-nostdin", "-v", "error", "-n", "-f", "rawvideo",
                "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(args.fps),
                "-i", "pipe:0", "-i", str(args.input), "-map", "0:v:0", "-map", "1:a?",
                *encoding_options,
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-movflags", "+faststart", *duration_options, str(temporary),
            ], stdin=subprocess.PIPE, stderr=encode_log)
            eof = False
            while not eof:
                images = []
                for _ in range(args.batch_size):
                    raw = decoder.stdout.read(decoded_bytes)
                    if not raw:
                        eof = True
                        break
                    if len(raw) != decoded_bytes:
                        raise RuntimeError("Incomplete decoded frame")
                    images.append(Image.frombytes("RGB", (width, height), raw))
                if not images:
                    break
                images, drawn_count, face_count = pipeline.annotate(images)
                positives += drawn_count
                faces_seen += face_count
                for image in images:
                    encoder.stdin.write(image.tobytes())
                    count += 1
                if count <= args.batch_size or count % max(1, round(args.fps * 10)) < len(images):
                    print(f"frames={count} | video={count/args.fps:.1f}s | faces={faces_seen} | drawn_boxes={positives} | elapsed={time.monotonic()-started:.1f}s", flush=True)
            decoder.stdout.close()
            if decoder.wait() != 0:
                raise RuntimeError("Video decoding failed")
            encoder.stdin.close()
            if encoder.wait() != 0 or count == 0:
                raise RuntimeError("Video encoding failed or input was empty")
            temporary.replace(args.output)
        except BaseException:
            for process in [decoder, encoder]:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
            for name, log in [("decoder", decode_log), ("encoder", encode_log)]:
                log.seek(0)
                tail = log.read().decode(errors="replace")[-3000:]
                if tail:
                    print(f"{name}: {tail}", flush=True)
            temporary.unlink(missing_ok=True)
            raise
    print(f"Saved: {args.output.resolve()} | {count} frames | {args.fps:g} FPS", flush=True)


if __name__ == "__main__":
    main()
