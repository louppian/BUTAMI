# Phase 2a. 얼굴·안경 bbox 추출

## 목적과 입출력

- 코드: `code/phase2a_extract/extract_bbox.py`
- 입력: `data/images/no_glasses/`, `data/images/regular_glasses/`, `data/images/smart_glasses/`의 JPEG
- 산출물: `data/annotations/face_glasses.json`
- 가중치: `ckpt/pretrained/face_detection_yunet_2023mar.onnx`, `ckpt/pretrained/segmentation_full_lraspp_mobilenet_v3_large.pth`, `ckpt/pretrained/torch/checkpoints/classification_anyglasses_shufflenet_v2_x1_0.pth`

Python 3.10 이상에서 `opencv-python-headless`, `numpy`, `Pillow`, `torch`, `torchvision`을 준비한다. 공식 glasses-detector와 동일한 ShuffleNet 분류기·LR-ASPP 분할기를 torchvision으로 구성하고 같은 사전학습 가중치를 직접 로드한다. 설정은 코드 상단에서 수정하고 IDE에서 파일을 실행한다.

## 추출 흐름

1. 이미지의 EXIF 방향을 반영하고 RGB로 읽는다. 수집기가 저장한 JPEG는 방향이 이미 반영되어 있다.
2. YuNet으로 얼굴 bbox를 찾는다. 얼굴 검출 입력은 긴 변 최대 640px로 축소하고, 결과는 실제 가로·세로 축척으로 원본 이미지 좌표에 복원한다. 기본 얼굴 점수 임계값은 0.5이며 YuNet 점수를 함께 보존한다.
3. 얼굴 bbox 주변을 가로·세로 1.8배 확장해 이미지 경계 안에서 자른다. 이 crop은 안경 착용 분류기와 분할기의 입력으로 메모리에서 사용한다.
4. `anyglasses/medium` ShuffleNet으로 얼굴별 안경 착용 확률을 계산하고, `full/medium` LR-ASPP로 안경테·안경다리·렌즈를 분할한다. 입력은 256×256이며 최대 32개 얼굴 crop을 배치 처리한다.
5. 안경 착용 확률이 0.5 미만이면 해당 얼굴의 안경 bbox를 제외한다. 0.5 이상이면 양수 logit 마스크의 외접 bbox를 구하고 빈 마스크는 제외한다. GPU에서는 bbox 네 좌표와 착용 확률만 배치당 한 번 CPU로 가져온다.
6. 안경 bbox를 얼굴 주변 crop에서 원본 이미지 좌표로 변환한다. 얼굴·안경 좌표는 이미지 경계에 맞춰 자르고 COCO `[x, y, width, height]`로 기록한다.
7. 전체 이미지를 처리한 후 임시 JSON을 최종 파일로 교체한다. 처리 중 실패하면 이전 최종 JSON은 유지된다.

이미지마다 검출된 모든 얼굴과 각 얼굴 주변의 안경 bbox를 기록한다. 얼굴을 검출하지 못한 이미지도 `images`에 포함한다. 클래스 이름에 관계없이 같은 검출·분할 모델을 적용한다. 빈 마스크와 얼굴 검출 실패는 모델의 검출 결과이며 실제 안경 미착용 정답과는 검수로 구분한다.

## COCO JSON

- `images`: `id`, `file_name`, `width`, `height`. `file_name`은 `data/images/` 기준 상대 경로다. 세 종류의 수집 클래스는 이 경로에 유지된다.
- `categories`: `id=1, name=face`, `id=2, name=glasses`.
- `annotations`: `id`, `image_id`, `category_id`, `bbox`, `area`, `iscrowd=0`, `score`.
- 얼굴 주석의 `score`는 YuNet 탐지 점수이고 `glasses_probability`는 해당 얼굴의 안경 착용 확률이다. 미착용으로 판정한 얼굴에도 두 값을 기록한다.
- 안경 주석의 `score`는 해당 얼굴의 안경 착용 확률이다. `face_id`로 같은 이미지의 얼굴 주석 ID를 참조한다. 이 점수는 bbox 위치의 정확도를 평가하는 값과 구분해 사용한다.
- `bbox`: 원본 이미지 픽셀 단위 `[x, y, width, height]`. `area`는 bbox 면적이다.
- 안경 bbox가 있으면 얼굴 주석과 별도의 안경 주석을 같은 `image_id`로 추가한다.
- JSON은 bbox 검출용 COCO 형식이다.

## 가중치 준비

처음 실행할 때 파일이 없으면 OpenCV Zoo의 YuNet과 `mantasu/glasses-detector` v1.0.0의 full-medium·anyglasses-medium 가중치를 공식 배포 URL에서 내려받는다. 이후 로컬 파일을 재사용한다. Git LFS 포인터 텍스트와 100KB 이하의 잘못된 파일을 확인하며, 다운로드는 임시 파일을 거쳐 저장한다.

얼굴 검출은 OpenCV CPU를 사용하고 안경 착용 분류·분할은 CUDA가 있으면 GPU 혼합 정밀도를 사용한다. CPU에서도 동일한 bbox 생성 절차를 사용할 수 있다.

기존 JSON에 점수가 없으면 갱신한 추출 코드를 다시 실행해 `score`, `glasses_probability`, `face_id`를 포함한 JSON을 생성한다.

## 후속 작업

자동 생성한 bbox를 검수·수정한 뒤 안경 검출기 학습에는 `glasses` 주석을 사용한다. RT-DETRv4 학습·DINOv3 증류·특징 추출은 이후 2a 작업으로 이어진다. 현재 추출 코드의 산출물은 얼굴·안경 bbox JSON이다.
