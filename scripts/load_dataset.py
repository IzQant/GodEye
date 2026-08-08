"""
Day 9 작업: zones_dataset.csv를 DB에 적재.

data/processed/zones_dataset.csv를 읽어
- matches 테이블: 매치별 메타데이터(맵, shard)
- circles 테이블: phase별 자기장 정보
에 넣는다.

여러 번 실행해도 중복이 쌓이지 않도록, 적재 전에 두 테이블을 비우고(reset)
새로 채운다. (아직 데이터 규모가 작고 로컬 개발 단계이므로 이 방식이 단순·안전)

완료 기준(Day 9): DB에 데이터 적재 확인 (SELECT count(*)).

실행: python scripts/load_dataset.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from app.config import settings
from app.database import SessionLocal
from app.models import Circle, Match

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "zones_dataset.csv")


def main():
    if not os.path.exists(CSV_PATH):
        print("zones_dataset.csv가 없습니다. scripts/build_dataset.py를 먼저 실행하세요.")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"CSV 로드: {len(df)}행, 고유 매치 {df['match_id'].nunique()}개")

    shard = settings.PUBG_SHARD or "steam"
    db = SessionLocal()
    try:
        # 재적재를 위해 기존 데이터 삭제 (circles가 matches를 참조하므로 circles 먼저)
        deleted_c = db.query(Circle).delete()
        deleted_m = db.query(Match).delete()
        db.commit()
        print(f"기존 데이터 삭제: circles {deleted_c}행, matches {deleted_m}행")

        # matches: 매치당 한 행 (맵 이름은 정규화된 'map' 컬럼 사용)
        for match_id, group in df.groupby("match_id"):
            db.add(Match(
                match_id=match_id,
                map_name=group["map"].iloc[0],
                shard=shard,
            ))

        # circles가 matches를 외래키로 참조하므로, matches를 먼저 DB에 써서
        # (flush) 참조 대상이 존재하게 만든 뒤 circles를 넣는다.
        # (Postgres는 FK를 엄격히 검사하므로 이 순서가 반드시 필요)
        db.flush()

        # circles: CSV 각 행을 그대로 적재
        for _, row in df.iterrows():
            db.add(Circle(
                match_id=row["match_id"],
                map_name=row["map"],
                phase=int(row["phase"]),
                start_time=float(row["start_time"]),
                end_time=float(row["end_time"]),
                safety_x=float(row["safety_x"]),
                safety_y=float(row["safety_y"]),
                safety_radius=float(row["safety_radius"]),
                poison_x=float(row["poison_x"]),
                poison_y=float(row["poison_y"]),
                poison_radius=float(row["poison_radius"]),
            ))

        db.commit()

        match_count = db.query(Match).count()
        circle_count = db.query(Circle).count()
        print(f"\n적재 완료 → matches: {match_count}행, circles: {circle_count}행")

        if circle_count > 0:
            print("✅ Day 9 완료 기준 통과: DB 적재 확인")
        else:
            print("❌ circles가 비어 있음 — CSV/컬럼 확인 필요")
    except Exception as e:
        db.rollback()
        print(f"적재 실패(롤백): {e}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
