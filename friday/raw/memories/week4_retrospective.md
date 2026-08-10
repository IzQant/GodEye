# Week 4 회고 (Day 22~28, 08-11 ~ 08-17)

주차 목표: 컴퓨터 비전 원 검출 + /api/detect
종료 산출물: 검출 모듈 + API + v0.4

## 완료한 것
- Day 22: data/images 구조(real/synthetic)+README, check_images.py, make_synthetic_minimaps.py
- Day 23: circle_detector HSV 색상 필터링(BGR→HSV→inRange→모폴로지), show_masks.py
- Day 24: Contour+minEnclosingCircle 검출, eval_detection.py (합성 100장 성공률 100%)
- Day 25: 신뢰도(circularity)+수동입력 폴백 분기, analyze_failures.py (실패율 수치화)
- Day 26: coordinate_transform.py (픽셀→맵 좌표: 어파인+호모그래피), verify_transform.py
- Day 27: CircleDetector 클래스화 + test_detector.py 5개 통과
- Day 28: /api/detect 실구현(이미지 업로드→좌표/반경+needs_manual), 회고

## 데이터 확보 방식 전환 (중요)
- real 스크린샷 수작업 대신 '텔레메트리 + 실제 맵 이미지 오버레이'로 준실사 라벨
  데이터 대량 생성(make_map_overlays.py). 현재 1213장(8개 맵) 확보.
- 장점: 실제 맵 배경 + 실제 자기장 좌표 + 픽셀 단위 정답 라벨.

## 검출 성능
- 합성 100장: 검출 성공률 100%(중심 0px, 반경 ~2px 편향).
- 어려운 변형(저해상도/노이즈/가림/저대비): 신뢰도 하락 → needs_manual 폴백 정상 트리거.
- 주의: 합성/오버레이 기준 수치이며, 실제 스크린샷은 HSV/임계값 재튜닝 필요.

## 좌표 변환
- 기준점 보정 시 어파인 오차 0cm, 호모그래피 <1cm. 실서비스는 맵별 기준점 1회 보정 필요.

## 이슈 및 해결
- 오버레이 생성기 파일명이 실행마다 바뀌어 orphan 누적 → match_id+phase 고정 파일명 +
  시작 시 기존 파일 정리로 해결. (import glob 누락도 수정.)
- joblib sklearn 버전 불일치 → 라우터 503 처리 + 모델 로드 실패 시 테스트 skip.

## API 현황
- /health, /api/predict(matchId→예측), /api/detect(이미지→검출). 통합은 Week 5.

## 다음 주(Week 5)로
- Day 29~: pipeline.py(통합), /api/analyze, Jinja2 프론트엔드, 오버레이 시각화,
  레이트리밋+로깅, E2E 수동 테스트.

## 버전 태그
- v0.4 (본인이 직접 git tag)
