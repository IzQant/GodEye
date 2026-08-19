# 확장 1단계 — 딥러닝 검출: 데이터셋 부트스트랩 (기록)

작성일: 2026-08-30
결정: 데이터셋=오버레이 자동라벨 부트스트랩 / 학습=데스크탑 RTX 5060(GPU) / 프레임워크=YOLOv8(ultralytics)

## 만든 것
- scripts/build_detection_dataset.py
  - data/images/overlay/ + labels.csv → YOLO 형식 데이터셋(data/detect/).
  - 클래스: 0=safe(흰 원), 1=next(파란 원). 원(cx,cy,r)→정규화 bbox, 이미지 밖 클램프.
  - 매치 단위 train/val 분리(seed 42, val 0.2), data.yaml 자동 생성.
  - bbox 계산 검증 완료(정규화·클램프 정확).
- ml/train_yolo.py: YOLOv8s, imgsz 768, batch 16, device 0(GPU), epochs 100, patience 20.
- requirements-detect.txt: ultralytics (검출 전용, 배포 이미지 제외).

## 실행 순서 (데스크탑 GPU)
pip install -r requirements-detect.txt
python scripts/make_map_overlays.py        # data/maps/ 맵 이미지 필요
python scripts/build_detection_dataset.py  # data/detect/ + data.yaml
python ml/train_yolo.py                     # runs/zone_detect*/weights/best.pt

## 정직한 한계 (도메인 갭)
- 부트스트랩 데이터 = "실제 맵 + 우리가 그린 얇은 원". 게임 실제 원과 유사하나
  실제 스크린샷엔 UI(격자·지명·레드존·마커)·촬영 노이즈가 추가됨.
- 오버레이만 학습한 모델은 실제 스크린샷에서 성능 저하 가능.
- 순서: ①오버레이로 부트스트랩(현재) → ②실제 스크린샷 소량 라벨링해 검증/미세조정.
- 개선(구현됨): make_map_overlays.py에 방해요소 증강 추가(ADD_DISTRACTORS).
  격자선·레드존(빨간 채움원)·지명 텍스트·마커·노이즈를 원 그리기 전에 삽입 →
  실제 스크린샷과 유사, 라벨은 그대로 정확(원은 맨 위). ml/figures/overlay_augmented_demo.png 참고.

## 다음 단계
1. (데스크탑) make_map_overlays(증강 포함) → build_detection_dataset → YOLOv8 학습 → mAP 확인.
2. 학습 검출 vs 색상 기반(circle_detector) 성능 비교.
3. 좋으면 검출 결과를 픽셀→맵 변환에 연결(수동 폴백 유지).
4. 이후 실제 스크린샷 소량 라벨링으로 미세조정.
