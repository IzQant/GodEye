# git 대용량 파일 이슈 및 해결 (기록)

작성일: 2026-08-17
상황: v0.4 첫 push 시도 → 2.25GB 전송 후 실패
      (HTTP 500 / send-pack unexpected disconnect). 원격엔 아무것도 안 올라감.

## 원인
- .gitignore의 `data/images/*.png` 패턴이 하위폴더를 못 걸러서,
  data/images/overlay/(1214장) + synthetic/(102장) + data/maps/(11장) 이미지가
  전부 커밋됨. 이게 히스토리에 쌓여 .git이 2.5GB.
- .venv, *.joblib 은 정상 제외됨(문제 아님).

## 조치
1. .gitignore 수정: `data/images/**/*.png`(하위폴더 포함), data/maps/*.png 등 제외.
2. 히스토리에서 대용량 경로 제거:
   git filter-repo --path data/images --path data/maps --path data/raw --invert-paths --force
   (주차 태그 v0.1~v0.4 보존됨)
   또는 간단히: rm -rf .git 후 새로 init(태그는 사라짐).
3. 원격 등록 후 push + push --tags.

## 교훈 / 원칙
- 생성·수집 데이터(raw JSON, 이미지, 오버레이, joblib)는 git에 넣지 않는다.
  전부 스크립트로 재생성 가능: collect_batch.py, build_dataset.py,
  make_synthetic_minimaps.py, make_map_overlays.py, train_final.py.
- .gitignore에서 하위폴더까지 거르려면 `dir/**/*.ext` 형태를 쓴다.
- GitHub: 단일 파일 100MB 초과 거부, 대용량 push는 실패하기 쉬움.
