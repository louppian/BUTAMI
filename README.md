# BUTAMI

일반 안경과 스마트글래스를 구분하기 위한 데이터 수집·안경 영역 추출·분류 학습 프로젝트.

검출·학습 기준 모델은 [AIMIA-EAI/RT-DETRv4](https://github.com/AIMIA-EAI/RT-DETRv4)입니다.
현재 실행 가능한 코드는 **YuNet + 기존 glasses-detector를 이용한 얼굴별 안경 검출 기준선**입니다.
RT-DETRv4 안경 검출기 학습, feature 추출, 일반/스마트글래스 분류기는 아직 구현되지 않았습니다.

## 폴더 구조

```text
BUTAMI/
├── code/
│   ├── phase1a_images/
│   ├── phase1b_videos/
│   ├── phase2a_extract/
│   ├── phase2b_video_check/
│   │   └── main.py
│   ├── phase3a_train/
│   └── phase3b_video_check/
├── data/
│   ├── images/
│   ├── videos/
│   └── annotations/
├── results/
│   ├── phase1a_images/
│   ├── phase1b_videos/
│   ├── phase2a_extract/
│   ├── phase2b_video_check/
│   ├── phase3a_train/
│   └── phase3b_video_check/
├── ckpt/
│   ├── pretrained/
│   ├── phase2a_extract/
│   └── phase3a_train/
└── third_party/
    └── glasses-detector/
```

- `code`와 `results`의 하위 폴더는 이름이 같으며 단계별로 1:1 대응합니다.
- `data`: 수집한 원본 이미지·영상과 정답 주석. 파생 결과와 구분합니다.
- `results`: 추출 프레임·crop·feature·결과 영상·평가·로그 등 단계별 생성물.
- `ckpt/pretrained`: 다운로드한 사전학습 가중치. PyTorch 다운로드 캐시는 `ckpt/pretrained/torch/`에 저장됩니다.
- `ckpt/phase2a_extract`: 학습한 안경 검출기 체크포인트.
- `ckpt/phase3a_train`: 학습한 일반/스마트글래스 분류기 체크포인트.
- `third_party`: 외부 소스와 원본 라이선스를 유지하는 위치.
- 현재 데이터·결과·체크포인트 폴더는 `.gitkeep`만 추적하며, 실제 내용은 Git에서 제외합니다.

## 단계별 작업

| 단계 | 작업 | 결과 위치 | 상태 |
| --- | --- | --- | --- |
| 1a | 일반 안경·스마트글래스 이미지 수집, 출처 기록, 중복 제거, 제품 라벨·bbox 주석 | `results/phase1a_images/` | 예정 |
| 1b | 스마트글래스·일반 안경·미착용 영상 수집, 프레임 샘플링 | `results/phase1b_videos/` | 예정 |
| 2a | RT-DETRv4 안경 검출기 학습, 얼굴·안경 crop, bbox 및 feature 추출 | `results/phase2a_extract/` | 예정 |
| 2b | 얼굴별 안경 검출 결과를 영상에서 확인 | `results/phase2b_video_check/` | 기존 검출기 기준선 구현; RT-DETRv4 연동 예정 |
| 3a | 일반 안경·스마트글래스 feature 분류 학습 | `results/phase3a_train/` | 예정 |
| 3b | 영상별 최종 분류 결과·오탐·미탐·처리 속도 확인 | `results/phase3b_video_check/` | 예정 |

### Phase 1: 데이터 수집

- **1a 이미지:** 제품 사진과 실제 착용 사진을 수집합니다. 정면·측면·안쪽·안경다리 등 다양한 각도, 두꺼운 일반 안경·선글라스를 포함합니다. 제품 정보로 일반/스마트 라벨을 확인하고 출처를 기록합니다. 검출 bbox는 COCO 형식으로 준비합니다.
- **1b 영상:** 고개 회전·거리 변화·가림·조명 변화 장면을 포함합니다. 일반 안경·미착용 영상도 확보해 오탐을 평가합니다. 학습용 프레임은 일정 간격으로 추출한 뒤 중복을 제거합니다.
- 같은 원본 영상·제품 이미지의 중복본이 학습·검증·시험 데이터에 섞이지 않도록 그룹 단위로 분리합니다.

### Phase 2: 안경 영역 추출

- **2a:** 얼굴 주변 영역을 추출하고 RT-DETRv4-S를 `glasses` 1개 클래스 검출기로 파인튜닝합니다. 검출 bbox와 crop을 저장하고, HybridEncoder의 bbox 대응 영역에서 ROI pooling으로 feature를 추출하도록 구현합니다. 안경다리 등 구분에 필요한 외형이 잘리지 않도록 crop 범위를 비교합니다.
- **2b:** 검출기를 영상에 적용해 원본 좌표의 bbox를 확인합니다. 작은 얼굴·측면·가림·움직임에서 누락과 위치 오류를 기록합니다. 현재 `main.py`는 기존 glasses-detector를 이용하는 기준선이며 RT-DETRv4 추론 코드는 아닙니다.

### Phase 3: 일반 안경 / 스마트글래스 분류

- **3a:** 같은 feature 추출기로 만든 특징 벡터를 사용해 MLP 2-class 분류기를 학습합니다. `0=일반 안경`, `1=스마트글래스`로 정의합니다. 먼저 검출기·encoder를 고정하고 분류기를 학습한 뒤 필요하면 encoder 일부를 미세조정합니다.
- **3b:** 얼굴 검출 → 안경 검출 → feature 추출 → 종류 분류 → 원본 영상 표시를 연결합니다. 연속 프레임의 결과를 안정화하고, 불확실한 결과는 판별 보류로 처리합니다.
- 평가는 스마트글래스 precision·recall, 일반 안경 오탐률, 기종별 성능, 전체 파이프라인의 검출 실패 및 실제 처리 FPS를 포함합니다. 미학습 기종은 별도 평가합니다.

## 모델 정의

| 역할 | 모델 | 적용 상태 |
| --- | --- | --- |
| 얼굴 검출 | OpenCV YuNet | 기준선에서 사용 중 |
| 현재 안경 검출 기준선 | glasses-detector medium: ShuffleNet 기반 착용 여부 분류 + SSDLite 기반 위치 검출 | 사용 중 |
| 목표 안경 검출기 | RT-DETRv4-S, `glasses` 1-class 파인튜닝 | 예정 |
| 목표 feature 추출 | RT-DETRv4 HybridEncoder + bbox 영역 ROI pooling | 추가 구현 예정 |
| 목표 종류 분류기 | 안경 feature를 입력하는 MLP 2-class | 학습 예정 |
| 목표 검출기 학습용 교사 | DINOv3 ViT-B/16-LVD-1689M | RT-DETRv4 증류 학습 시 준비 |

RT-DETRv4의 기본 COCO 가중치에는 안경 클래스가 없으므로 안경 bbox 데이터로 학습해야 합니다.
교사 모델은 학습용이며, 분류기에 사용하는 feature 추출기와 구분합니다.
RT-DETRv4와 DINOv3 소스·가중치는 현재 이 저장소에 포함되어 있지 않습니다. 외부 소스를 추가할 때는 `third_party/`, 가중치는 `ckpt/`에서 관리합니다.

## 현재 기준선 실행

### 설치

Python 3.12 이상, PyTorch·torchvision, OpenCV 4.8 이상 5 미만, NumPy, Pillow, FFmpeg·ffprobe가 필요합니다.

```bash
git clone https://github.com/louppian/BUTAMI.git
cd BUTAMI

conda create -n butam python=3.12 -y
conda activate butam
conda install -c conda-forge ffmpeg -y

# 기존 서버에서 사용한 CUDA 12.8 빌드 예시. 다른 서버는 드라이버에 맞게 선택합니다.
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ./third_party/glasses-detector "opencv-python-headless>=4.8,<5" numpy pillow
```

기존 위치에서 editable 설치한 환경도 이동된 `third_party/glasses-detector` 경로로 다시 설치합니다.
최초 실행 시 YuNet과 glasses-detector 사전학습 가중치를 다운로드하므로 인터넷 연결이 필요합니다.

### 실행

1. 입력 영상을 `data/videos/video1.mp4`에 둡니다.
2. `code/phase2b_video_check/main.py` 상단의 값을 수정합니다.

```python
DEVICE = "cuda"  # GPU가 없으면 "cpu"
FPS = 30
CLASSIFICATION_THRESHOLD = 0.5
DETECTION_THRESHOLD = 0.3
MAX_SECONDS = 10  # 먼저 10초 확인. 전체 영상은 None
```

입력·출력 경로는 같은 파일의 `INPUT_VIDEO`, `OUTPUT_VIDEO`에서 수정합니다.
별도 config·environment 파일이나 CLI 옵션 없이 실행합니다.

```bash
python code/phase2b_video_check/main.py
```

기본 출력은 `results/phase2b_video_check/video1_glasses_30fps.mp4`입니다.
기존 출력 파일은 덮어쓰지 않으므로 다시 실행할 때는 `OUTPUT_VIDEO`를 새 이름으로 바꿉니다.
영상은 지정 FPS로 변환하고 원본 오디오가 있으면 함께 저장합니다. `libx264`가 없으면 `mpeg4`로 저장합니다.
출력 30fps는 재생 프레임레이트이며, 실제 처리 속도를 뜻하지 않습니다.
현재 코드는 입력 영상의 회전 메타데이터를 자동 적용하지 않습니다.

### 현재 영상의 표시값

- `C`: 얼굴 영역에 안경이 있다는 분류 확률. 스마트글래스일 확률이 아닙니다.
- `D`: 안경 위치 검출기의 점수. 실제 정확도나 스마트글래스 분류 점수가 아닙니다.
- 두 임계값을 모두 통과하고 유효한 좌표를 가진 안경만 초록 박스로 표시합니다.
- `faces`: 프레임 전체에서 누적 검출한 얼굴 수. 고유 인원 수가 아닙니다.
- `drawn_boxes`: 실제 그린 안경 박스의 누적 수. 고유 안경 수가 아닙니다.
- 얼굴 미검출·가림·저해상도 등으로 박스가 없을 수 있으며, 박스 없음이 안경 미착용을 보장하지 않습니다.

## 출처

- 목표 검출 모델: [AIMIA-EAI/RT-DETRv4](https://github.com/AIMIA-EAI/RT-DETRv4)
- 기존 기준선: [mantasu/glasses-detector](https://github.com/mantasu/glasses-detector), 포함된 [MIT License](third_party/glasses-detector/LICENSE)와 저작권 표시 유지
- 얼굴 검출: [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- 목표 학습 교사: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

외부 프로젝트의 라이선스는 각 원본에 따릅니다. 포함된 glasses-detector의 라이선스를 BUTAMI 자체 코드 전체의 라이선스로 간주하지 않습니다.
