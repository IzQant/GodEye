# 히트맵 예측 연결 + 사진 보정 개선 (기록)

작성일: 2026-08-24 (Week 5 이후 실험/개선)

## 1. 히트맵(분포) 예측 — 서비스 임시 연결
배경: 점(중심 좌표) 예측은 자기장 무작위성으로 한계. 분포 예측이 정직한 개선.

- ml/heatmap_model.py: 이동 오프셋(dx/r, dy/r)을 단계별 가우시안 KDE로 학습.
  predict_grid()로 맵 위 확률 히트맵 생성.
- 평가(전환 목표, 매치 split):
  - Coverage: 50%→56%, 80%→86%, 90%→96% (잘 보정됨, 약간 보수적).
  - 로그가능도: KDE 0.854 vs 균등 -1.145 → +2.0. 균등 무작위보다 정보량 있음.
- 서비스 연결(임시):
  - app/services/heatmap_service.py: CSV로 1회 학습·캐시.
  - app/services/visualize.draw_heatmap: JET 히트맵을 지도에 알파 블렌딩.
  - POST /api/heatmap: analyze와 동일 입력, 점 예측기 불필요(현재 원만 있으면 됨).
  - 프론트: "히트맵으로 보기" 토글 → /api/heatmap 호출해 표시.
- 상태: 실험 기능. 정식화하려면 히트맵 모델 직렬화 필요.

## 2. 사진 보정(원근 변환) — 4모서리 지정
배경: 사진은 지도가 화면을 안 채워 좌표 왜곡 → 지도 네 모서리로 원근 보정 필요.

- app/services/perspective.py: warp_map/warp_point/warp_circle (getPerspectiveTransform).
- 라우터: analyze/visualize/heatmap에 corners 필드 → 보정 후 예측·오버레이.

### 자동 감지 시도와 결론
- app/services/map_detect.py: 에지+최대 사각형 휴리스틱으로 지도 4점 추정.
- 실제 사진 테스트: 정확도 매우 낮음. 원인 = 지도 외곽이 바다색으로 흐릿,
  격자선/UI/베젤/벽/기울어짐이 경쟁 → 엉뚱한 사각형/실패. 파라미터 튜닝으론 한계.
- 개선 결정: **4번 클릭 수동 방식**(좌상→우상→우하→좌하) 채택. 자동 감지는 보조 버튼.
  - 프론트: 캔버스 클릭으로 4점 지정 + 드래그 미세보정 + "다시 찍기" + "자동 감지(보조)".
  - 4점 미만이면 corners 미적용(전체 이미지 폴백).

## 3. 정직한 결론
- 실제 사진에서 "지도 영역 자동 검출"은 일반 CV로는 불안정(원 검출 실패와 같은 부류).
  가장 확실한 개선은 (1) 스크린샷(촬영X) 입력, (2) 쉬운 수동(4클릭). ML 세그멘테이션은
  로드맵 확장급.
- 예측 정확도의 진짜 개선은 히트맵/분포 관점(coverage·LL). 점 정확도는 무작위성 벽.

## 관련 파일
- ml/heatmap_model.py, ml/features.py, ml/compare_algos.py, ml/dataset_pairs.py
- app/services/heatmap_service.py, visualize.py, perspective.py, map_detect.py
- app/routers/analyze.py (/api/heatmap, /api/detect-corners)
- app/templates/index.html (히트맵 토글, 4클릭 보정 UI)
