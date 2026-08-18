# Week 5 회고 (Day 29~35, 08-18 ~ 08-24)

주차 목표: 파이프라인 통합 + 프론트엔드 + 안전장치
종료 산출물: 통합 로컬 데모 + v0.5

## 완료한 것
- Day 29: pipeline.py (matchId/이미지 두 경로 → 동일 결과 스키마)
- Day 30: /api/analyze 통합 엔드포인트(입력검증+친화적 에러)
- Day 31: Jinja2+Bootstrap 프론트엔드(두 탭, 맵 선택, 결과 카드)
- Day 32: 지도 오버레이(visualize.py, /api/visualize PNG) + 프론트 표시
- Day 33: slowapi 레이트리밋(데코레이터) + request_logs 로깅 미들웨어
- Day 34: E2E 3시나리오 테스트 + 버그수정(검출 실패 시 모델 로드 지연)
- Day 35: UI 다듬기(로딩/버튼비활성/429 메시지), README, 발표개요, 회고

## Week 5 중 반영한 설계/개선 (사용자 피드백 기반)
- 이미지+phase 입력 설계 확정·구현 (phase는 사용자 입력, 원은 자동/수동).
- 수동 입력을 클릭+드래그 원 지정으로 변경.
- 사진 보정(지도 네 모서리 → 원근 보정) 추가: 모니터 촬영 사진도 처리.
- 모델 재구성: 예측 목표를 단계 전환(phase N→N+1)으로 → 초반 단계도 의미있는 예측.

## 트러블슈팅 기록
- SlowAPIMiddleware ↔ BaseHTTPMiddleware(로깅) 충돌 → 데코레이터 방식 레이트리밋으로 해결.
- git 대용량 push 사고 → 데이터/생성물 gitignore + 히스토리 정리.
- joblib sklearn 버전 불일치 → 라우터 503 처리 + 테스트 skip.

## 현재 API
- GET / (웹 UI), /health
- POST /api/predict, /api/detect, /api/analyze, /api/visualize
- 모두 레이트리밋(분당 30) + 요청 로깅 대상.

## 테스트
- 총 21개(환경에 따라 일부 skip). 본인 머신(모델·이미지 준비 시) 대부분 통과.

## 다음 주(Week 6)로
- Day 36~: 프로덕션 Docker 정리, Railway/Render 배포, 모니터링(UptimeRobot),
  보안 점검, README/발표자료 완성, 최종 런칭(v1.0).

## 버전 태그
- v0.5 (본인이 직접 git tag)
