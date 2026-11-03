# Phase 3a. CLIP 특징 + 3-class MLP 학습·평가

## 입력과 실행

- 코드: `code/phase3a_train/train_mlp.py`
- 입력: 2a의 `data/annotations/face_glasses.json`, `data/images/`의 원본 이미지
- 환경: Python 3.10 이상, `torch`, `transformers`, `numpy`, `Pillow`, `matplotlib`
- CLIP: `openai/clip-vit-base-patch32`, `ckpt/pretrained/clip/`의 로컬 캐시 재사용

IDE에서 파일을 실행한다. 설정은 상단에서 수정한다. 기본값은 얼굴 점수 0.5 초과, batch 64, hidden 128, 100 epoch, AdamW 학습률 0.001, 클래스별 80:20 분리, seed 42다.

## 2a와의 경계

2a는 얼굴·안경 bbox, 얼굴 탐지 점수, 안경 착용 확률, 주대상 얼굴 선택·검수 상태를 생성한다. 3a는 `selected_face_id`가 지정되고 얼굴 점수가 0.5를 넘는 이미지에서 선택한 얼굴을 사용한다.

검수 대기·미검출·작은 얼굴 등 선택되지 않은 이미지는 `excluded_images.csv`에 이유를 기록한다. 기존 JSON에 선택 정보가 없으면 갱신한 2a를 먼저 실행한다. 얼굴 ID는 같은 이미지의 얼굴 주석을 참조해야 한다.

원본 폴더의 `no_glasses`, `regular_glasses`, `smart_glasses`를 정답 라벨로 사용한다. 안경 검출 결과와 상관없이 이 라벨을 유지하므로 수집 라벨과 주대상 얼굴은 학습 전에 검수한다.

## 학습

1. 선택된 얼굴 주변을 1.8배 확장해 원본에서 crop한다.
2. 고정한 CLIP 이미지 인코더로 L2 정규화한 특징 벡터를 한 번만 추출한다.
3. 이미지별로 클래스마다 80:20 분리한다. 한 이미지에서 선택한 얼굴 하나만 사용한다.
4. `feature_dim → 128 → ReLU → 3` MLP를 학습한다. 특징·MLP 학습은 GPU 메모리에서 처리한다.
5. 학습이 끝난 뒤 분리한 테스트 데이터에서 원래 softmax 확률로 한 번 평가한다.

현재 실험은 이미지 단위 분리다. 동일 인물·제품 단위의 별도 평가는 이후 검수한 분할로 수행한다.

## 예측과 검출 결과 대조

MLP의 세 클래스 예측과 확률을 그대로 저장한다. 얼굴의 안경 착용 확률을 0.5 기준으로 비교해 상태를 기록한다.

| 조건 | 기록 |
| --- | --- |
| MLP 미착용 + 낮은 착용 확률 | `final_class=no_glasses` |
| MLP 일반/스마트 + 높은 착용 확률 | 해당 MLP 클래스로 `final_class` 기록 |
| 두 모델의 안경 착용 여부가 충돌 | 원래 예측 유지, `final_class=review`, `reason=presence_conflict` |
| 착용 확률이 높은데 안경 bbox 누락 | `reason`에 `glasses_bbox_missing` 추가 |

bbox가 없는 사진도 MLP의 원래 예측을 평가한다. 테스트 정확도·혼동행렬은 충돌 샘플을 포함한 전체 테스트 샘플 기준이다. 판정 충돌과 bbox 누락은 별도 집계한다.

## 산출물

| 경로 | 내용 |
| --- | --- |
| `ckpt/phase3a_train/clip_mlp_3class.pt` | MLP 가중치, 클래스 순서, backbone·특징 차원·crop 설정 |
| `results/phase3a_train/metrics.json` | 원래 MLP 정확도·혼동행렬, 선택/제외 수, 충돌·bbox 누락·보류 수 |
| `results/phase3a_train/predictions.csv` | 원본 경로·얼굴 ID·정답·원래 예측·세 확률·착용 확률·최종 상태 |
| `results/phase3a_train/excluded_images.csv` | 얼굴 선택 조건에서 제외된 이미지와 이유 |
| `results/phase3a_train/confusion_matrix.png` | 원래 MLP 예측의 혼동행렬 |
| `results/phase3a_train/misclassified_001.png` 등 | 오분류 원본 이미지, 선택된 얼굴·연결된 안경 bbox와 점수 |

오분류 그림은 페이지당 최대 15장으로 생성한다. 초록은 선택된 얼굴, 빨강은 해당 얼굴에 연결된 안경 bbox다. 제목에 원래 MLP 예측·확률과 `final_class`, 충돌·누락 이유를 함께 표시한다. 재실행 시 같은 스크립트의 이전 오분류 페이지를 갱신한다.

모델 기반 오탐·미탐과 분류 정확도는 서버의 실제 이미지·가중치로 실행해 확인한다.
