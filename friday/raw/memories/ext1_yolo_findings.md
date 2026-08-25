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


## YOLO 채택 확정 (2026-09-01, conf 0.25 재측정)
compare_detectors.py (오버레이 2807장):
- 색상: safe 100%, next 2474/2807 | 중심오차 중앙 278px
- YOLO(conf0.25): safe 2799/2807(100%), next 2330/2807(83%) | 중심오차 중앙 0.1px, 반경 0.0px
→ 임계값 0.25로 recall 회복(46%→100%). YOLO 완승 → 기본 검출기로 채택.
조치: get_yolo_detector 게이트 기본값 on(GODEYE_USE_YOLO 기본 "1").
  ONNX 있으면 YOLO, 없으면 색상 폴백. 끄려면 GODEYE_USE_YOLO=0. 수동 폴백은 유지.
배포: ml/models/zone_detect.onnx(44MB)를 git add/commit/push 해야 배포 컨테이너가 YOLO 사용.
남은 확인: 실제 폰 촬영 사진(정렬 후)에서 YOLO 검출 동작(도메인 갭) 최종 점검.


## 배포 OOM 확정 + imgsz 512 축소 (2026-09-01)
- 증상: YOLO 배포 후 결과 안 나옴. GODEYE_USE_YOLO=0 시 정상(→색상으로 예측된 것).
  즉 원인은 YOLO 메모리 OOM 확정(로드+추론 시 ~162MB 추가 → 512MB 컨테이너 초과).
- 조치(재학습 불필요): INPUT_SIZE 768→512, export_onnx IMGSZ 768→512.
  입력 해상도를 낮춰 활성 메모리·연산 대폭 감소.
- 방어: pipeline.analyze_by_image에 try/except — YOLO 추론 예외 시 색상 폴백(결과 안 끊김).
  512 코드 + 768 onnx 불일치도 폴백으로 안전(검증됨).
- 사용자(데스크탑) 할 일:
  1) python ml/export_onnx.py         # 이제 512로 재export → ml/models/zone_detect.onnx
  2) python ml/compare_detectors.py   # 512에서도 정확도 유지되는지 확인(safe 검출율·오차)
  3) git add ml/models/zone_detect.onnx app/services/yolo_detector.py ml/export_onnx.py
     → commit → push
  4) Railway 환경변수 GODEYE_USE_YOLO=0 삭제(또는 1)로 되돌려 YOLO 재활성 → 배포 테스트
- 512로도 OOM이면: yolov8n(nano) 재학습 또는 Railway 메모리 상향.


## 실제 사진 검출 실패 진단 (2026-09-01)
- 배포 512에서 실사진 흰 원 검출 실패 → 수동 폴백 발생.
- 로컬 재현: 사용자의 그 태이고 사진을 512 onnx로 검출 → safe 신뢰도 0.972로 정상 검출!
  즉 512 모델은 실사진에서도 작동. nano 재학습 불필요(크기 문제 아님, 이미 OOM 해결됨).
- 결론: 배포 실패 원인은 검출기가 아니라 그 사진의 map_align(지도 자동 정렬) 실패로
  추정 → 정렬 실패 시 베젤 포함 원본이 YOLO에 들어가 검출 불가.
- 조치: 수동 실패 메시지에 정렬 성공/실패([지도 자동 인식 실패]/[성공]) 표시 추가.
  다음 테스트에서 이 라벨로 정렬 문제인지 검출 문제인지 즉시 구분 가능.
- 후속: 정렬 실패면 → 사진 해상도/각도 문제(정렬 강건화). 정렬 성공인데 검출 실패면
  → YOLO 도메인 갭(실제 스크린샷 소량 라벨링 미세조정이 진짜 해법, nano 아님).


## 로그에 yolo 없음 → 게이트 변수 잔존 유력 (2026-09-01)
- 배포 로그에 [yolo_detector]/[detector] 줄이 전혀 없음.
- 유력 원인: Railway Variables에 GODEYE_USE_YOLO=0 이 아직 남아 있음(디버깅 때 넣은 것).
  → 코드가 YOLO 로드 전에 건너뜀 → 색상 검출기 사용 → 실사진 검출 실패 → 수동.
  '정렬 성공 + 검출 실패 + yolo 로그 없음' 증상과 정확히 일치.
- 조치:
  1) Railway Variables에서 GODEYE_USE_YOLO 삭제(코드 기본값 on). 재배포 후 테스트.
  2) 로그 가시성: print에 flush=True 추가 + '[detector] 사용 검출기: XXX' 매 요청 로그 추가.
     → 배포에서 YoloCircleDetector/CircleDetector 중 뭘 쓰는지 즉시 확인 가능.
- 검증(로컬): 변수 없으면 'YoloCircleDetector', =0이면 'CircleDetector' 로그 정상.
