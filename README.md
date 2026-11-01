# BUTAMI

일반 안경과 스마트글래스의 이미지·영상을 수집하고, 안경 영역의 feature를 추출해 두 종류를 분류하는 프로젝트.

기준 모델: [AIMIA-EAI/RT-DETRv4](https://github.com/AIMIA-EAI/RT-DETRv4)

현재는 **폴더 구조와 phase별 개발 계획을 정리한 단계**입니다. 과거의 `temp.py`는 참고 예시이며, phase 구현이나 완성된 실행 프로그램이 아닙니다. RT-DETRv4 학습·feature 추출·일반/스마트글래스 분류는 구현 예정입니다.

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
    └── temp.py.txt
```

| 폴더 | 관리 대상 |
| --- | --- |
| `code/` | 실제 phase별로 구현할 코드 |
| `data/` | 원본 이미지·영상과 정답 라벨·bbox 주석 |
| `results/` | 해당 phase에서 생성한 프레임·crop·feature·영상·평가·로그 |
| `ckpt/` | 사전학습 가중치와 직접 학습한 체크포인트 |
| `third_party/` | 외부 소스 및 원본 라이선스 |
| `examples/` | 과거 참고 예시. 실제 구현·실행 대상에서 제외 |

`code/<단계>/`의 결과는 반드시 같은 이름의 `results/<단계>/`에 저장합니다. 체크포인트만 `ckpt/`에 별도로 저장합니다.

## Phase 1. 데이터 크롤링

### 1a. 스마트글래스 / 일반 안경 이미지

- 제품 이미지와 실제 착용 이미지 수집.
- 정면·측면·안쪽·안경다리 등 다양한 각도 확보.
- 제조사·제품 정보를 기준으로 일반/스마트 라벨 확인.
- 중복 이미지 제거, 출처·기종·촬영 조건 기록.
- 안경 bbox를 COCO 형식으로 주석.
- **원본:** `data/images/`
- **정답 주석:** `data/annotations/`
- **코드 / 결과:** `code/phase1a_images/` ↔ `results/phase1a_images/`

### 1b. 스마트글래스 영상

- 스마트글래스 착용 영상 수집.
- 비교용 일반 안경·미착용 영상 확보.
- 고개 회전·거리 변화·가림·조명 변화 장면 포함.
- 일정 간격으로 프레임 추출 후 중복 제거.
- 같은 원본 영상의 프레임은 학습·검증·시험 데이터 사이에 분산하지 않음.
- **원본:** `data/videos/`
- **코드 / 결과:** `code/phase1b_videos/` ↔ `results/phase1b_videos/`
- **결과:** 추출 프레임, 영상별 라벨·수집 기록.

## Phase 2. 안경 영역 추출

### 2a. 얼굴 / 안경 분리 및 안경 feature 추출

- YuNet으로 얼굴 영역을 찾고 주변 영역 추출.
- RT-DETRv4-S를 일반·스마트 구분 없는 `glasses` 1-class 검출기로 학습.
- 검출 bbox와 안경 crop을 저장하고 원본 좌표 기록.
- 카메라·안경다리 등 구분에 필요한 부분이 잘리지 않도록 crop 범위 확인.
- HybridEncoder의 bbox 대응 영역에서 ROI pooling으로 고정 길이 feature를 추출하도록 구현.
- **코드 / 결과:** `code/phase2a_extract/` ↔ `results/phase2a_extract/`
- **결과:** 얼굴·안경 crop, bbox, feature, 검출 평가.
- **학습 체크포인트:** `ckpt/phase2a_extract/`

### 2b. 영상별 적용 확인

- 학습한 안경 검출기를 영상에 적용.
- 얼굴별 안경 bbox를 원본 영상 좌표에 표시.
- 작은 얼굴·측면·가림·움직임에서 누락 및 위치 오류 확인.
- 실패 장면을 수집해 학습 데이터 보강.
- **코드 / 결과:** `code/phase2b_video_check/` ↔ `results/phase2b_video_check/`
- **결과:** bbox 영상, 영상별 검출 평가, 실패 프레임.

## Phase 3. 일반 안경 / 스마트글래스 분류기

### 3a. 일반 안경 feature vs 스마트글래스 feature 분리 학습

- 동일한 검출기·feature 추출기로 두 종류의 특징 벡터 생성.
- `0=일반 안경`, `1=스마트글래스`로 MLP 분류기 학습.
- 먼저 검출기·encoder를 고정하고 분류기를 학습.
- 검증 데이터에서 임계값 결정.
- 학습에 포함된 기종과 미학습 기종을 구분해 평가.
- **코드 / 결과:** `code/phase3a_train/` ↔ `results/phase3a_train/`
- **결과:** 혼동행렬, precision·recall, 일반 안경 오탐률, 기종별 평가.
- **학습 체크포인트:** `ckpt/phase3a_train/`

### 3b. 영상별 적용 확인

- 얼굴 검출 → 안경 검출 → feature 추출 → 일반/스마트 분류 연결.
- 동일 인물의 연속 예측을 모아 결과 안정화.
- 불확실한 장면은 판별 보류로 처리하고 검출 실패를 일반 안경으로 간주하지 않음.
- bbox·분류 결과·분류 점수를 영상에 표시.
- **코드 / 결과:** `code/phase3b_video_check/` ↔ `results/phase3b_video_check/`
- **결과:** 최종 분류 영상, 예측 기록, 영상별 오탐·미탐·실제 처리 FPS.

## 모델 및 외부 소스

| 역할 | 모델 | 소스 위치 / 상태 |
| --- | --- | --- |
| 얼굴 검출 | OpenCV YuNet | OpenCV 사용. 별도 YuNet 소스 저장소 불필요 |
| 기존 안경 검출 참고 | glasses-detector medium | `third_party/glasses-detector/`에 원본 소스 보관 |
| 목표 안경 검출 | RT-DETRv4-S | `third_party/RT-DETRv4/`에 추가 예정 |
| 목표 feature 추출 | RT-DETRv4 HybridEncoder + ROI pooling | 추가 구현 예정 |
| 목표 종류 분류 | MLP 2-class | `code/phase3a_train/`에 구현 예정 |
| 검출기 증류 학습 교사 | DINOv3 ViT-B/16-LVD-1689M | `third_party/dinov3/`에 추가 예정 |

`third_party/RT-DETRv4/`와 `third_party/dinov3/`는 현재 위치만 확보한 빈 폴더입니다. 소스가 포함되거나 학습 연동이 완료된 상태가 아닙니다. RT-DETRv4의 기본 COCO 가중치에는 안경 클래스가 없으므로 안경 bbox 데이터로 학습해야 합니다.

## 체크포인트 관리

| 위치 | 내용 |
| --- | --- |
| `ckpt/pretrained/` | 직접 준비한 YuNet·RT-DETRv4·DINOv3 등의 사전학습 가중치 |
| `ckpt/phase2a_extract/` | 학습한 안경 검출기 체크포인트 |
| `ckpt/phase3a_train/` | 학습한 일반/스마트글래스 분류기 체크포인트 |

- YuNet 파일 위치: `ckpt/pretrained/face_detection_yunet_2023mar.onnx`
- 가중치는 명시한 로컬 파일을 읽는 방식으로 구현합니다.


## 원본 출처

- [AIMIA-EAI/RT-DETRv4](https://github.com/AIMIA-EAI/RT-DETRv4)
- [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)
- [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [mantasu/glasses-detector](https://github.com/mantasu/glasses-detector): 포함된 [MIT License](third_party/glasses-detector/LICENSE) 및 저작권 표시 유지.
