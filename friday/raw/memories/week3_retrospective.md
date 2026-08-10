# Week 3 회고 (Day 15~21, 08-04 ~ 08-10)

주차 목표: 모델 고도화 + 예측 API 서빙
종료 산출물: /api/predict 엔드포인트 + v0.3

## 완료한 것
- Day 15: train_regression.py (RandomForest). 절대좌표 목표에선 copy-baseline이 강함을 발견
- Day 16: gaussian_model.py (이동량 2D 가우시안 + 신뢰구간)
- Day 17: train_mlp.py (PyTorch MLP, 이동량 목표) — 선택 항목
- Day 18: compare_models.py (4모델 공정 비교), 최종 RF(delta) 선정
- Day 19: train_final.py + model_service.py (joblib 직렬화, 1회 로드 캐시)
- Day 20: /api/predict 실구현 (match_service로 현재 원 추출 → 모델 추론)
- Day 21: pytest 7개(단위4+통합3) 전부 통과, 예외처리, 회고

## 데이터 현황
- 85매치 → 629 phase 행 (Week 3 중 100매치 추가 수집 반영)
- 맵 분포: Taego 220, Erangel 211, Sanhok 60, Miramar 53, Karakin 43, Rondo 22, Vikendi 13, Paramo 7

## 모델 비교 (동일 조건, 이동량 프레이밍, 중심 오차 m)
| 모델 | 평균 | 중앙값 | p90 |
|------|-----:|------:|----:|
| RF(delta) | 4.3 | 1.4 | 11.4 |
| Copy(다음=현재) | 4.7 | 0.6 | 11.2 |
| PhaseMean | 4.9 | 0.7 | 12.0 |
| MLP(delta) | 10.8 | 4.1 | 20.0 |
- 최종: RF(delta) 채택(점 예측) + 가우시안(신뢰구간). Copy 대비 마진은 작음.

## 이슈 및 해결/기록
- 단위 혼동(cm를 m로 서술) 정정 → unit_correction.md, analyze_patterns 출력에 단위 명시
- joblib은 저장/로드 sklearn 버전이 같아야 함. requirements에 1.5.0 고정.
  → 각 환경에서 train_final.py로 predictor.joblib을 새로 생성해야 함(gitignore 대상).
- 향후 모델 업그레이드 방향은 model_upgrade_roadmap.md 참고 (LightGBM/LSTM, 맵별 분리)

## 예외 처리 현황
- 잘못된 matchId → 404, 모델 미준비 → 503, match_id 누락 → 422. 서버는 죽지 않음.

## 다음 주(Week 4)로
- Day 22~: OpenCV 미니맵 원 검출 (테스트 이미지 수집 → HSV 필터 → Contour → 좌표 변환) + /api/detect

## 버전 태그
- v0.3 (본인이 직접 git tag)
