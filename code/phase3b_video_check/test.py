"""영상 경로 하나를 받아 tests/<입력 파일명>에 추론 영상을 저장한다."""

import argparse


class YuNet:
    """동일한 YuNet ONNX 가중치를 PyTorch 장치에서 실행한다."""
    def __init__(self, path, device, threshold=0.5):
        import onnx
        from onnx2torch import convert
        self.device, self.threshold = device, threshold
        graph = onnx.load(str(path))
        self.names = [output.name for output in graph.graph.output]
        self.model = convert(graph).to(device).eval().requires_grad_(False)
        # Reshape/Resize의 고정 크기 상수는 CPU에 유지해 매 호출 장치 동기화를 줄인다.
        self.model.initializers.cpu()
        print(f'YuNet: {next(self.model.parameters()).device} (PyTorch)', flush=True)

    def detect(self, image):
        import torch
        from torchvision.ops import nms
        with torch.inference_mode():
            pixels = torch.as_tensor(image, device=self.device).permute(2, 0, 1).unsqueeze(0).float()
            pixels = torch.nn.functional.pad(pixels, (0, -image.shape[1] % 32, 0, -image.shape[0] % 32))
            outputs = dict(zip(self.names, self.model(pixels)))
            faces = []
            # OpenCV YuNet의 bbox·landmark·score 복원 규칙:
            # https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_detect.cpp
            for stride in (8, 16, 32):
                y, x = torch.meshgrid(torch.arange(pixels.shape[2] // stride, device=self.device),
                                      torch.arange(pixels.shape[3] // stride, device=self.device), indexing='ij')
                grid = torch.stack((x, y), -1).reshape(-1, 2)
                box = outputs[f'bbox_{stride}'][0]
                size = box[:, 2:].exp() * stride
                corner = (box[:, :2] + grid) * stride - size / 2
                landmarks = (outputs[f'kps_{stride}'][0].reshape(-1, 5, 2) + grid[:, None]) * stride
                score = (outputs[f'cls_{stride}'][0].clamp(0, 1) * outputs[f'obj_{stride}'][0].clamp(0, 1)).sqrt()
                faces.append(torch.cat((corner, size, landmarks.flatten(1), score), 1))
            faces = torch.cat(faces)
            faces = faces[torch.isfinite(faces).all(1) & (faces[:, -1] >= self.threshold) & (faces[:, 2:4] > 0).all(1)]
            faces = faces[faces[:, -1].argsort(descending=True)[:5000]]
            boxes = faces[:, :4].trunc()
            boxes = torch.cat((boxes[:, :2], boxes[:, :2] + boxes[:, 2:]), 1)
            faces = faces[nms(boxes, faces[:, -1], 0.3)]
            return faces.cpu().numpy()


def main(video):
    from pathlib import Path
    from time import perf_counter
    import sys
    import cv2
    import numpy as np
    import torch
    import torchvision.transforms as T
    from PIL import Image

    ROOT = Path(__file__).resolve().parents[2]
    SOURCE = Path(video).expanduser()
    SOURCE = (SOURCE if SOURCE.is_absolute() else ROOT / SOURCE).resolve()
    if not SOURCE.is_file():
        raise FileNotFoundError(f'영상 경로를 확인하세요: {SOURCE}')
    WEIGHTS = ROOT / 'ckpt/pretrained/RTv4-S-hgnet.pth'
    OUTPUT = ROOT / 'tests' / SOURCE.name
    if SOURCE == OUTPUT.resolve():
        raise ValueError('입력 영상과 출력 영상의 경로를 구분해 주세요.')
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FACE_WEIGHTS = ROOT / 'ckpt/pretrained/face_detection_yunet_2023mar.onnx'
    THRESHOLD = 0.5
    FACE_THRESHOLD = 0.5
    BATCH_SIZE = 8
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert SOURCE.is_file(), SOURCE
    MLP_WEIGHTS = ROOT / 'ckpt/phase3a_train/clip_mlp_3class.pt'
    if not MLP_WEIGHTS.is_file():
        raise FileNotFoundError(f'메인 가중치를 준비해 주세요: {MLP_WEIGHTS}')
    sys.path.insert(0, str(ROOT / 'code/phase2a_extract/bbox'))
    from extract_bbox import GlassesExtractor, project_box, ensure_weights, YUNET_URL
    ensure_weights(FACE_WEIGHTS, YUNET_URL)
    if not WEIGHTS.is_file():
        import gdown
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        temporary = WEIGHTS.with_suffix('.pth.tmp')
        try:
            # RT-DETRv4 공식 README의 RT-DETRv4-S 체크포인트.
            if gdown.download(id='1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z', output=str(temporary)) is None:
                raise RuntimeError('RT-DETRv4-S 가중치 다운로드를 확인해 주세요.')
            if temporary.stat().st_size <= 100_000:
                raise ValueError('RT-DETRv4-S 다운로드 파일의 크기를 확인해 주세요.')
            temporary.replace(WEIGHTS)
        finally:
            temporary.unlink(missing_ok=True)
    sys.path.insert(0, str(ROOT / 'third_party/RT-DETRv4'))
    from engine.core import YAMLConfig
    from engine.data.dataset.coco_dataset import mscoco_category2name

    print(f'{DEVICE=} | {WEIGHTS.name}: {WEIGHTS.stat().st_size:,} bytes')

    from transformers import CLIPModel, CLIPProcessor
    import math


    cfg = YAMLConfig(str(ROOT / 'third_party/RT-DETRv4/configs/rtv4/rtv4_hgnetv2_s_coco.yml'))
    cfg.yaml_cfg['HGNetv2']['pretrained'] = False
    checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    model = cfg.model
    print(model.load_state_dict(checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']))
    # 고정 위치 임베딩도 모델과 함께 장치로 이동하도록 등록합니다.
    for idx in model.encoder.use_encoder_idx:
        name = f'pos_embed{idx}'
        if hasattr(model.encoder, name) and name not in model.encoder._buffers:
            embedding = getattr(model.encoder, name)
            delattr(model.encoder, name)
            model.encoder.register_buffer(name, embedding, persistent=False)
    model = model.deploy().to(DEVICE).eval()
    postprocessor = cfg.postprocessor.deploy().to(DEVICE).eval()
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    classes = list(mscoco_category2name.values())
    del checkpoint

    PERSON = classes.index('person')
    face_detector = YuNet(FACE_WEIGHTS, DEVICE, FACE_THRESHOLD)

    checkpoint = torch.load(MLP_WEIGHTS, map_location='cpu', weights_only=True)
    assert checkpoint['classes'] == ['no_glasses', 'regular_glasses', 'smart_glasses']
    mlp = torch.nn.Sequential(torch.nn.Linear(checkpoint['feature_dim'], checkpoint['hidden_dim']), torch.nn.ReLU(), torch.nn.Linear(checkpoint['hidden_dim'], 3)).to(DEVICE).eval()
    mlp.load_state_dict(checkpoint['state_dict'])
    def load_clip(cls):
        try:
            return cls.from_pretrained(checkpoint['backbone'], cache_dir=str(ROOT / 'ckpt/pretrained/clip'), local_files_only=True)
        except OSError:
            return cls.from_pretrained(checkpoint['backbone'], cache_dir=str(ROOT / 'ckpt/pretrained/clip'))

    processor = load_clip(CLIPProcessor)
    clip = load_clip(CLIPModel).to(DEVICE).eval()
    FACE_EXPAND = checkpoint['face_expand']
    NORMALIZE = checkpoint['normalize_features']
    glasses = GlassesExtractor(ROOT)
    CLASS_NAMES = ('normal', 'glass', 'smart')
    CLASS_COLORS = ((0, 0, 255), (0, 255, 255), (255, 0, 0))  # OpenCV BGR
    print(f'MLP: {MLP_WEIGHTS.name} | person/face/안경/분류 임계값: {THRESHOLD}')
    del checkpoint

    def draw_box(frame, box, caption, color):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, caption, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


    cap = cv2.VideoCapture(str(SOURCE))
    assert cap.isOpened(), f'영상 열기 실패: {SOURCE}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert fps > 0 and width > 0 and height > 0, '영상 메타데이터 확인 필요'
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    assert writer.isOpened(), f'영상 저장기 열기 실패: {OUTPUT}'
    sizes = torch.tensor([[width, height]], device=DEVICE).expand(BATCH_SIZE, -1)
    processed = 0
    started = perf_counter()
    try:
        with torch.inference_mode():
            while True:
                frames = []
                for _ in range(BATCH_SIZE):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frames.append(frame)
                if not frames:
                    break
                images = torch.stack([transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))) for frame in frames]).to(DEVICE)
                labels, boxes, scores = postprocessor(model(images), sizes[:len(frames)])
                detections = torch.cat((boxes, labels.unsqueeze(-1), scores.unsqueeze(-1)), -1).cpu().numpy()
                people = [rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] == PERSON) & (rows[:, 5] >= THRESHOLD)] for rows in detections]
                faces, crops = [], []
                # 원본에서 얼굴과 1.8배 crop을 수집한 뒤 배치 추론합니다.
                for i, (frame, rows) in enumerate(zip(frames, people)):
                    for x1, y1, x2, y2, _, score in rows:
                        x1, x2 = np.clip([np.floor(x1), np.ceil(x2)], 0, width).astype(int)
                        y1, y2 = np.clip([np.floor(y1), np.ceil(y2)], 0, height).astype(int)
                        if x2 - x1 < 16 or y2 - y1 < 16:
                            continue
                        found = face_detector.detect(np.ascontiguousarray(frame[y1:y2, x1:x2]))
                        if found is None:
                            continue
                        for face in found:
                            if not np.isfinite(face).all() or face[-1] < FACE_THRESHOLD:
                                continue
                            fx, fy, fw, fh = map(float, face[:4])
                            if fw <= 0 or fh <= 0:
                                continue
                            fx1, fy1 = max(x1, math.floor(x1 + fx)), max(y1, math.floor(y1 + fy))
                            fx2, fy2 = min(x2, math.ceil(x1 + fx + fw)), min(y2, math.ceil(y1 + fy + fh))
                            if fx2 <= fx1 or fy2 <= fy1:
                                continue
                            margin = (FACE_EXPAND - 1) / 2
                            cx1, cy1 = max(0, math.floor(fx1 - (fx2 - fx1) * margin)), max(0, math.floor(fy1 - (fy2 - fy1) * margin))
                            cx2, cy2 = min(width, math.ceil(fx2 + (fx2 - fx1) * margin)), min(height, math.ceil(fy2 + (fy2 - fy1) * margin))
                            faces.append((i, (fx1, fy1, fx2, fy2), float(face[-1]), (cx1, cy1, cx2 - cx1, cy2 - cy1)))
                            crops.append(Image.fromarray(cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)))
                predictions, bounds = [], []
                for offset in range(0, len(crops), 32):
                    batch = crops[offset:offset + 32]
                    pixels = processor(images=batch, return_tensors='pt')['pixel_values'].to(DEVICE)
                    with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
                        features = clip.visual_projection(clip.vision_model(pixel_values=pixels).pooler_output)
                    features = features.float()
                    if NORMALIZE:
                        features = torch.nn.functional.normalize(features, dim=-1)
                    predictions.extend(mlp(features).softmax(1).cpu().tolist())
                    bounds.extend(glasses(batch))
                for frame, rows in zip(frames, people):
                    for x1, y1, x2, y2, _, score in rows:
                        x1, x2 = np.clip([x1, x2], 0, width - 1).astype(int)
                        y1, y2 = np.clip([y1, y2], 0, height - 1).astype(int)
                        draw_box(frame, (x1, y1, x2, y2), f'person {score:.2f}', (0, 255, 0))
                for (i, (x1, y1, x2, y2), face_score, crop), probability, bound in zip(faces, predictions, bounds):
                    frame = frames[i]
                    face_box = (x1, y1, min(width - 1, x2), min(height - 1, y2))
                    label = int(np.argmax(probability))
                    confidence = probability[label]
                    if confidence < THRESHOLD:
                        draw_box(frame, face_box, f'face {face_score:.2f}', (0, 0, 255))
                        continue
                    color = CLASS_COLORS[label]
                    caption = f'{CLASS_NAMES[label]} {confidence:.2f}'
                    box = project_box(bound[:4], crop) if label > 0 and bound[4] >= THRESHOLD else None
                    if box is not None:
                        draw_box(frame, face_box, f'face {face_score:.2f}', (0, 0, 255))
                        gx, gy, gw, gh = box
                        draw_box(frame, (gx, gy, min(width - 1, gx + gw), min(height - 1, gy + gh)), caption, color)
                    else:
                        draw_box(frame, face_box, f'face {face_score:.2f}', (0, 0, 255))
                        if label == 0:
                            cv2.putText(frame, caption, (x1, min(height - 5, y1 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                for frame in frames:
                    writer.write(frame)
                processed += len(frames)
                if processed % (BATCH_SIZE * 25) == 0:
                    elapsed = perf_counter() - started
                    print(f'{processed}/{total} frames | 실제 영상 {processed / fps:.2f}초 | 처리 {elapsed:.2f}초 | {abs(1 - elapsed / (processed / fps)) * 100:.2f}% {"빠름" if elapsed <= processed / fps else "느림"}', flush=True)
    finally:
        cap.release()
        writer.release()
    elapsed = perf_counter() - started
    print(f'{processed}/{total} frames 저장 완료: {OUTPUT} | 실제 영상 {processed / fps:.2f}초 | 처리 {elapsed:.2f}초 | {abs(1 - elapsed / (processed / fps)) * 100:.2f}% {"빠름" if elapsed <= processed / fps else "느림"}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="영상 추론 결과를 tests/<입력 파일명>에 저장")
    parser.add_argument("video", help="입력 영상 경로 (상대 경로는 프로젝트 루트 기준)")
    main(parser.parse_args().video)
