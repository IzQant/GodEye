"""
Day 6 작업: 결측치/이상치 처리 + 데이터셋 통합.

data/raw/에 있는 모든 매치 JSON을 파싱해서 phase 요약으로 만들고,
정제(cleaning)한 뒤 하나의 CSV(data/processed/zones_dataset.csv)로 통합한다.

완료 기준(Day 6): 단일 CSV 파일로 통합, 행 수/컬럼 확인.

실행: python scripts/build_dataset.py
"""
import glob
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from app.services.telemetry_parser import parse_zone_events, summarize_phases

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUT_PATH = os.path.join(PROCESSED_DIR, "zones_dataset.csv")

# PUBG 맵 내부 코드명 → 사람이 읽는 이름 매핑.
# 서버 리전 정보는 텔레메트리에 직접 없으므로, 우선 맵 이름 정규화에 집중한다.
MAP_NAME_KO = {
    "Baltic_Main": "Erangel",
    "Desert_Main": "Miramar",
    "Savage_Main": "Sanhok",
    "DihorOtok_Main": "Vikendi",
    "Range_Main": "TrainingRange",
    "Summerland_Main": "Karakin",
    "Chimera_Main": "Paramo",
    "Tiger_Main": "Taego",
    "Kiki_Main": "Deston",
    "Neon_Main": "Rondo",
}


def extract_map_name(telemetry: list[dict]) -> str:
    for event in telemetry:
        if event.get("_T") == "LogMatchStart":
            return event.get("mapName", "unknown")
    return "unknown"


def load_all_phases() -> pd.DataFrame:
    """data/raw/의 모든 매치를 파싱해서 phase 요약을 하나로 합친다."""
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    frames = []

    for path in files:
        match_id = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, encoding="utf-8") as f:
                telemetry = json.load(f)
        except json.JSONDecodeError:
            print(f"[SKIP] {match_id} — JSON 손상")
            continue

        map_name = extract_map_name(telemetry)
        events_df = parse_zone_events(telemetry, match_id, map_name)
        phases_df = summarize_phases(events_df)
        if not phases_df.empty:
            frames.append(phases_df)
            print(f"[OK] {match_id} — {map_name}, phase {len(phases_df)}개")
        else:
            print(f"[SKIP] {match_id} — phase 없음")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """결측치·이상치 처리 + 맵 이름 정규화."""
    before = len(df)

    # 맵 이름 정규화 (코드명 → 읽기 쉬운 이름). 매핑에 없으면 원본 유지.
    df["map"] = df["map_name"].map(MAP_NAME_KO).fillna(df["map_name"])

    # 훈련장(TrainingRange)은 실제 자기장 게임이 아니므로 제외한다.
    df = df[df["map"] != "TrainingRange"]

    # phase 0은 첫 자기장 발표 전이라 poison 좌표/반경이 0인 경우가 많다.
    # 예측 대상(다음 원)이 없는 행이므로 제거한다.
    df = df[df["poison_radius"] > 0]

    # 좌표/반경 핵심 컬럼에 결측치가 있는 행 제거.
    key_cols = ["safety_x", "safety_y", "safety_radius",
                "poison_x", "poison_y", "poison_radius"]
    df = df.dropna(subset=key_cols)

    # 좌표 이상치 제거: PUBG 맵 좌표는 보통 0~816000(cm) 범위.
    # 음수나 과도하게 큰 값은 파싱 오류로 보고 걸러낸다.
    coord_max = 900000
    for col in ["safety_x", "safety_y", "poison_x", "poison_y"]:
        df = df[(df[col] >= 0) & (df[col] <= coord_max)]

    after = len(df)
    print(f"\n정제: {before}행 → {after}행 ({before - after}행 제거)")
    return df.reset_index(drop=True)


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw = load_all_phases()
    if raw.empty:
        print("파싱된 데이터가 없습니다. Day 5 배치 수집을 먼저 실행하세요.")
        return

    cleaned = clean(raw)

    cleaned.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n=== 통합 데이터셋 저장 완료 ===")
    print(f"경로: {OUT_PATH}")
    print(f"행 수: {len(cleaned)}, 컬럼 수: {len(cleaned.columns)}")
    print(f"컬럼: {list(cleaned.columns)}")
    print(f"고유 매치 수: {cleaned['match_id'].nunique()}")
    print(f"\n맵별 phase 분포:")
    print(cleaned["map"].value_counts().to_string())

    if len(cleaned) > 0:
        print("\n✅ Day 6 완료 기준 통과: 단일 CSV로 통합 완료")
    else:
        print("\n❌ 정제 후 데이터가 비어 있음 — clean() 조건 확인 필요")


if __name__ == "__main__":
    main()
