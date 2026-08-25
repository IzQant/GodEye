# 확장 1단계 — YOLO 검출 적용 인프라 (기록)

작성일: 2026-09-01
상태: 학습된 best.pt는 아직 맥북에 없음. 모델 도착 시 즉시 꽂히도록 서빙 인프라만 선구축.

## 만든 것
- app/services/yolo_detector.py
  - YOLOv8 ONNX를 OpenCV DNN(cv2.dnn.readNetFromONNX)으로 추론. torch/ultralytics 불필요(배포 경량).
  - letterbox 전처리 → decode_yolov8((1,4+nc,N)) → NMS → 클래스별 최고점 박스 → 원(중심, r=(w+h)/4).
  - detect_with_confidence(image): CircleDetector와 동일 인터페이스({safe,next,needs_manual,reasons}).
  - get_yolo_detector(): ml/models/zone_detect.onnx 있으면 로드·캐시, 없으면 None.
  - 검증: 가짜 YOLOv8 출력(N=20)으로 역변환·NMS·클래스 분리 정확 확인.
    (주의: 가짜 테스트에서 N<채널수면 decode의 전치 휴리스틱이 오작동 → 실제 N≈8400은 정상.)
- app/services/pipeline.py
  - _get_detector() = get_yolo_detector() or CircleDetector(). analyze_by_image가 이걸 사용.
  - ONNX 있으면 YOLO, 없으면 색상 폴백. 수동 원 지정 폴백은 그대로.
- ml/export_onnx.py: best.pt → ml/models/zone_detect.onnx (opset12, imgsz=768, 데스크탑 실행).
- ml/compare_detectors.py: 오버레이 라벨로 색상 vs YOLO 성공률·중심/반경 오차 비교.

## 배포 포함
- .gitignore: onnx 미제외(추가 가능). .dockerignore: ml/models/*.joblib만 제외(onnx는 COPY ml/로 포함).
- 서빙 신규 의존성 없음(cv2.dnn은 opencv-python-headless에 포함).

## 데스크탑에서 할 일 (모델 준비)
1. (학습 완료 후) python ml/export_onnx.py            # best.pt → zone_detect.onnx
2. python ml/compare_detectors.py                     # 색상 vs YOLO 비교(오버레이 라벨)
3. 좋으면 git add ml/models/zone_detect.onnx → commit → push (배포 자동 반영)
- imgsz는 학습·내보내기·서빙(INPUT_SIZE=768) 모두 768로 일치시킬 것.

## 다음
- best.pt/onnx 도착 → compare_detectors 수치 확인 → 임계값(CONF_THRESHOLD=0.35) 조정 →
  실제 촬영 사진에서 YOLO 검출 확인 → 필요 시 실사진 소량 라벨링 미세조정(2단계 다리).
