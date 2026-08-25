# 확장 1단계 — YOLO ONNX 검증 결과 (기록)

작성일: 2026-09-01
zone_detect.onnx(44MB) 도착 후 샌드박스 검증.

## 확인된 것
- ONNX 로드·추론 정상. 출력 (1,6,12096), 박스 0~766px(imgsz768), 점수 0~1. 디코드 형식 정확.
- 후처리 버그 발견·수정: 클래스 무관 NMS가 겹친 safe·next를 하나로 합쳐 한쪽 소실.
  → 클래스별 NMS로 수정(yolo_detector._infer). 이제 safe·next 각각 반환.

## 경고 신호 (합성 오버레이 테스트)
- 흰(외곽 500,480,r240) + 파랑(안쪽 470,450,r150)을 그린 오버레이에서:
  safe·next 둘 다 (470,450,r150) = 안쪽 파란 원으로 잡히고, 바깥 흰 원은 완전히 놓침.
  → safe/next 클래스 분리가 약함(둘은 얇은 선 '색'만 다른데 768px 축소 시 뭉개짐 추정).
- 단, 이 테스트는 임의 합성 원(학습 분포와 다름)이라 단정 불가. 실제 오버레이 라벨로
  compare_detectors.py 정량 평가 + 학습 mAP(특히 클래스별) 확인 필요.

## 조치
- get_yolo_detector에 환경변수 게이트: GODEYE_USE_YOLO=1일 때만 YOLO 사용(opt-in).
  검증 전 자동 대체 방지. 기본은 색상(CircleDetector) 유지. 테스트 회귀 없음.
- 파이프라인 _get_detector()가 게이트를 따름.

## 다음 판단에 필요한 것 (사용자/데스크탑)
1. 학습 mAP: mAP50, mAP50-95, 그리고 클래스별(safe/next) AP. results.png.
2. 데스크탑에서 python ml/compare_detectors.py (실제 오버레이 data/images/overlay/labels.csv 기준)
   → 색상 vs YOLO 성공률·중심/반경 오차 비교.
3. 판단:
   - YOLO가 '원 찾기'는 잘하지만 safe/next 분리가 약하면 →
     (a) 색상(HSV)로 클래스만 재판정 + YOLO로 위치, 하이브리드
     (b) imgsz 상향(896/1024) 재학습
     (c) 색상 방식 유지(현재 잘 됨) + YOLO는 보류
   - 좋으면 GODEYE_USE_YOLO=1 배포로 전환.


## YOLO 정량 검증 결과 + 임계값 조정 (2026-09-01)
학습(results.png): mAP50 ~0.96, mAP50-95 ~0.93, precision ~1.0, recall ~0.93. 수렴 우수.

compare_detectors.py (실제 오버레이 2807장):
- 색상(HSV+Contour): safe 100%, next 88% | 중심오차 중앙 278px, 반경 13.5px
  → '100% 성공'은 착시. 항상 원을 내놓지만 278px 빗나감(증강 방해요소가 HSV 파괴).
- YOLO(ONNX, conf0.35): safe 46%, next 55% | 중심오차 중앙 0.1px, 반경 0.0px
  → 검출만 하면 사실상 완벽. 약점은 검출율(recall)뿐.

해석: 정확도는 YOLO 압승(0.1px vs 278px). 색상은 실전 UI 방해요소에 취약.
검출율 46%는 학습 recall 0.93 대비 낮음 → 신뢰도 임계값 0.35가 과도하게 잘라낸 것.
조치: CONF_THRESHOLD 0.35 → 0.25로 낮춤(recall 회복 목적).

다음: 데스크탑에서 compare_detectors.py 재실행해 YOLO 검출율 확인.
- 검출율이 ~85%+로 오르면: YOLO 채택(GODEYE_USE_YOLO=1), 색상/수동은 폴백.
- 여전히 낮으면 confusion_matrix.png로 safe/next 혼동 확인 → 하이브리드(색상으로 클래스 재판정) 검토.
