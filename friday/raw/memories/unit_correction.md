# 단위 정정 기록 (cm vs m)

작성일: 2026-08-08

## 무엇이 틀렸나
Day 11 EDA 설명(및 지금은 삭제된 week1 회고 초안, 당시 대화)에서
단계별 중심 이동 거리(move_mean)를 "미터"로 잘못 서술했다.
예: "phase 6 이동 ~2100m"로 적었으나, 실제 값은 cm 단위였다.

## 정확한 사실
- PUBG 텔레메트리 좌표(safety_x/y, poison_x/y)와 그로부터 계산한
  이동량(dx, dy)·이동거리(move)는 모두 **cm 단위**다.
- 미터로 환산하려면 100으로 나눈다.
- 따라서 "phase 6 move_mean ≈ 1388cm" = 약 **13.9m** (2100m 아님).
- 이동이 이렇게 작은 이유: 여기 이동량은 "단계 간 이동"이 아니라
  같은 스냅샷 안의 poison(다음 원) − safety(현재 원) 차이라 원래 작다.
  (초반 단계는 동심원이라 거의 0.)

## 영향 범위 확인
- 코드(evaluate.py, model_service.py 등)는 처음부터 /100으로 m 환산이
  올바르게 되어 있었다. 즉 계산·모델은 정상, **서술만 틀렸다.**
- baseline_eval.md / week2_retrospective.md 등 남은 기록의 오차 수치(예: 5.4m,
  phase 7 27.8m)는 모두 정상 미터 값이다.

## 조치
- analyze_patterns.py 출력에 단위를 명시하고 move를 cm·m 함께 표기하도록 수정
  (재발 방지). 헤더: dx_cm, dy_cm, shrink, move_cm, move_m.
- 코드 주석에도 "좌표/이동량은 cm, ÷100 = m" 명시.
