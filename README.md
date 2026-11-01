# BUTAMI

일반 안경과 스마트글래스의 이미지·영상을 수집하고, 안경 영역의 특징을 추출해 두 종류를 분류하는 프로젝트.

기준 검출 모델: [AIMIA-EAI/RT-DETRv4](https://github.com/AIMIA-EAI/RT-DETRv4)

## 전체 흐름

![BUTAMI 단계별 모델 연결도](phase_flow.png)

- **이미지 경로:** `1a → 2a → 3a` — 이미지 수집 → 안경 bbox 생성·검출기 학습·특징 추출 → 분류기 학습·증류.
- **영상 경로:** `1b → 2b → 3b` — 영상 수집 → 얼굴·안경 검출·특징 추출 → 일반/스마트 분류 결과 표시.
- **모델 전달:** `2a → 2b`는 학습한 검출기·특징 추출기를, `3a → 3b`는 학습한 MLP를 전달한다.

## 폴더 구조

```text
BUTAMI/
├── code/
│   ├── phase1a_images/
│   ├── phase1b_videos/
│   ├── phase2a_extract/
│   ├── phase2b_video_check/
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
├── third_party/
│   ├── glasses-detector/
│   ├── RT-DETRv4/
│   └── dinov3/
└── examples/
    └── temp.py
```

| 폴더 | 관리 대상 |
| --- | --- |
| `code/` | 단계별 코드 |
| `data/` | 원본 이미지·영상, 종류 라벨·bbox 주석 |
| `results/` | 단계별 마스크·crop·특징·영상·평가와 작업 흐름 기록 |
| `ckpt/` | 사전학습 가중치와 학습한 체크포인트 |
| `third_party/` | 외부 소스와 원본 라이선스 |
| `examples/` | 참고 예시 |

`code/<단계>/`와 `results/<단계>/`는 1:1로 대응한다. 원본은 `data/`, 가중치는 `ckpt/`에 저장한다. 각 단계의 `results.md`에는 입력부터 출력까지의 작업 흐름을 기록한다.

## Phase 1. 데이터 수집

### 1a. 일반 안경 / 스마트글래스 이미지

**검색 이미지 → CLIP 수집 필터 → 이미지 저장·라벨 검수 → 2a**

- 일반 안경과 스마트글래스의 제품별 이미지를 수집한다.
- CLIP으로 복수 안경·합성·광고·부분 확대 이미지를 선별한다.
- 동일·유사 중복을 제거하고 출처·검색어·제품 정보를 기록한다.
- 제조사·제품 정보를 확인해 `0=일반 안경`, `1=스마트글래스` 라벨을 검수한다.
- 같은 제품의 사진은 동일한 학습·검증·시험 분할에 배정한다.
- 현재 수집 코드는 클래스별 500장, 총 1,000장을 목표로 다운로드 8개 병렬·CLIP 최대 32장 배치 처리로 구성한다.

**코드:** `code/phase1a_images/image_crawl.py`  
**원본:** `data/images/regular_glasses/`, `data/images/smart_glasses/`  
**기록:** `results/phase1a_images/results.md`, `results/phase1a_images/metadata.csv`

### 1b. 일반 안경 / 스마트글래스 영상

**영상 수집 → 프레임 추출 → 2b**

- 일반 안경·스마트글래스 착용 영상과 비교용 미착용 영상을 확보한다.
- 고개 회전·거리·가림·조명 변화가 포함된 장면을 수집한다.
- 일정 간격으로 프레임을 추출하고 원본 영상·시각·라벨을 기록한다.
- 같은 원본 영상의 프레임은 동일한 데이터 분할에 배정한다.

**원본:** `data/videos/`  
**코드 / 결과:** `code/phase1b_videos/` ↔ `results/phase1b_videos/`

## Phase 2. 안경 영역·특징 추출

### 2a. 이미지 bbox 생성·검출기 학습·특징 추출

**1a 이미지 → GlassesSegmenter → 마스크·bbox → 검수·주석 → RT-DETRv4 학습 → ROI 특징 → 3a**

1. **GlassesSegmenter(`kind="full"`)**로 안경 마스크를 생성한다.
2. 마스크의 최소·최대 좌표로 외접 bbox `[xmin, ymin, xmax, ymax]`를 생성한다.
3. bbox를 검수·수정하고 이미지 ID·종류 라벨·원본 좌표를 함께 기록한다. 빈 마스크는 검수 대상으로 남긴다.
4. 검수한 bbox로 **원본 이미지를 crop**해 안경다리·렌즈·카메라 등 bbox 내부의 원본 픽셀을 보존한다.
5. 검수 bbox를 COCO 형식으로 저장하고, **RT-DETRv4-S를 `glasses` 1-class 검출기**로 학습한다.
6. **DINOv3**를 특징 증류 교사로 사용한다. 동일 이미지의 DINOv3 특징과 RT-DETRv4의 대응 특징을 정렬해 검출 학습에 증류 손실을 결합한다.
7. 학습한 RT-DETRv4의 **HybridEncoder 특징맵과 안경 bbox → ROI pooling → 고정 길이 특징 벡터**를 추출한다.

**3a 전달:** 검수한 원본 안경 crop, 종류 라벨, ROI 특징 벡터.  
**2b 전달:** 학습한 RT-DETRv4 검출기와 동일한 ROI 특징 추출 구성.  
**주석:** `data/annotations/`  
**코드 / 결과:** `code/phase2a_extract/` ↔ `results/phase2a_extract/`  
**체크포인트:** `ckpt/phase2a_extract/`

### 2b. 영상 얼굴·안경 검출·특징 추출

**1b 프레임 → YuNet → RT-DETRv4 → ROI pooling → 3b**

- **YuNet**으로 영상 속 얼굴 영역을 찾는다.
- 얼굴 주변 영역에 **2a에서 학습한 RT-DETRv4**를 적용해 안경 bbox를 검출한다.
- 동일한 HybridEncoder·ROI pooling 구성으로 안경 특징을 추출한다.
- 얼굴 crop 기준 bbox를 원본 프레임 좌표로 변환하고 프레임 ID·시각과 함께 기록한다.
- 안경 bbox와 특징 벡터를 3b에 전달하고, 검출 위치·누락을 확인한다.

**코드 / 결과:** `code/phase2b_video_check/` ↔ `results/phase2b_video_check/`

## Phase 3. 일반 안경 / 스마트글래스 분류

### 3a. 분류 teacher·student 학습과 증류

**2a 원본 crop → CLIP + 분류 헤드 → teacher 예측 분포**  
**2a ROI 특징 → MLP → student 예측**

- **Teacher:** CLIP 이미지 인코더에 일반/스마트 분류 헤드를 연결하고, 검수한 원본 crop과 종류 라벨로 학습한다.
- **Student:** 2a의 RT-DETRv4 ROI 특징을 입력으로 받는 **MLP 2-class 분류기**를 학습한다.
- 먼저 RT-DETRv4 검출기·encoder를 고정하고 MLP를 학습한다.
- Student의 정답 라벨 분류 손실에 teacher의 두 클래스 예측 분포에 대한 증류 손실을 결합한다.
- Teacher와 student에는 동일한 안경 인스턴스에 대응하는 crop·ROI 특징을 입력한다.
- 검증 데이터로 분류 임계값을 결정하고, 학습 기종·미학습 기종을 구분해 평가한다.

**3b 전달:** 학습한 MLP, 클래스 순서, 특징 전처리와 판정 임계값.  
**코드 / 결과:** `code/phase3a_train/` ↔ `results/phase3a_train/`  
**체크포인트:** `ckpt/phase3a_train/`  
**평가:** 혼동행렬, precision·recall, 일반 안경 오탐률, 기종별 성능.

### 3b. 영상 분류·결과 확인

**2b 안경 bbox·ROI 특징 → 3a MLP → 일반/스마트 판정 → 원본 영상 표시**

- 2b에서 전달한 안경 특징을 **3a에서 학습한 MLP**에 입력한다.
- 원본 영상 좌표의 bbox에 일반/스마트 판정과 분류 점수를 표시한다.
- 동일 인물의 연속 예측을 모아 결과를 안정화한다.
- 검출 상태와 분류 상태를 구분하고 불확실한 장면은 판별 보류로 처리한다.
- 영상별 오탐·미탐·실제 처리 FPS를 확인한다.

**코드 / 결과:** `code/phase3b_video_check/` ↔ `results/phase3b_video_check/`

## 모델별 연결

| 모델 | 단계 | 입력 → 출력·전달 대상 |
| --- | --- | --- |
| CLIP 수집 필터 | 1a | 검색 이미지 → 선별 이미지 → 2a |
| GlassesSegmenter(`full`) | 2a | 이미지 → 마스크 → bbox 초안 → 검수·COCO 주석 |
| DINOv3 | 2a | 동일 이미지 → 교사 특징 → RT-DETRv4 특징 증류 |
| RT-DETRv4-S | 2a → 2b | 이미지·프레임 → 안경 bbox·HybridEncoder 특징맵 |
| ROI pooling | 2a → 3a / 2b → 3b | 특징맵 + bbox → 안경별 고정 길이 특징 |
| YuNet | 2b | 영상 프레임 → 얼굴 영역 → RT-DETRv4 |
| CLIP + 분류 헤드 | 3a | 원본 안경 crop → 일반/스마트 예측 분포 → MLP 증류 |
| MLP | 3a → 3b | ROI 특징 → 일반/스마트 점수·판정 |

CLIP의 이미지 선별과 종류 분류는 각각 별도의 역할이다. DINOv3의 특징 증류는 2a 검출 학습에, CLIP teacher의 분류 증류는 3a 분류 학습에 연결한다.

## 체크포인트 관리

| 위치 | 내용 |
| --- | --- |
| `ckpt/pretrained/` | GlassesSegmenter·YuNet·CLIP·RT-DETRv4·DINOv3 사전학습 가중치 |
| `ckpt/phase2a_extract/` | 학습한 안경 검출기와 특징 정렬 모듈 |
| `ckpt/phase3a_train/` | CLIP 분류 teacher, MLP student, 클래스·특징 전처리·임계값 |

YuNet 파일은 `ckpt/pretrained/face_detection_yunet_2023mar.onnx`, 수집 코드의 CLIP 캐시는 `ckpt/pretrained/clip/`에 둔다.

## 원본 출처

- [AIMIA-EAI/RT-DETRv4](https://github.com/AIMIA-EAI/RT-DETRv4)
- [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [mantasu/glasses-detector](https://github.com/mantasu/glasses-detector): 포함된 [MIT License](third_party/glasses-detector/LICENSE) 및 저작권 표시 유지.
