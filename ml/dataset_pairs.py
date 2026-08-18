"""
단계 전환쌍(phase N → phase N+1) 생성.

기존 학습 목표의 문제: 같은 스냅샷의 safety(현재)→poison(다음경고)는 초반 단계에서
거의 동일(동심원)이라 예측이 무의미했다.

새 목표: 한 매치 안에서 phase N의 실제 원(safety) → phase N+1의 실제 원(safety).
단계 사이 실제 이동/축소를 예측하므로 초반 단계에서도 의미가 있다.

반환 컬럼(FEATURE_COLS와 이름 일치 → 기존 ZonePredictor/RF 그대로 사용):
- map, safety_x, safety_y, safety_radius, phase   (입력 = 현재 phase의 원)
- dx, dy            : 다음 phase 중심 이동량 (target)
- next_radius       : 다음 phase 반경
- shrink            : next_radius / safety_radius
"""
import pandas as pd


def build_transition_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """zones_dataset(행=phase별 원)에서 매치별 인접 phase 전환쌍을 만든다."""
    rows = []
    for match_id, g in df.groupby("match_id"):
        g = g.sort_values("phase")
        recs = g.to_dict("records")
        for cur, nxt in zip(recs[:-1], recs[1:]):
            # 인접 단계만(중간에 phase가 비면 건너뜀)
            if int(nxt["phase"]) != int(cur["phase"]) + 1:
                continue
            if cur["safety_radius"] <= 0:
                continue
            rows.append({
                "match_id": match_id,
                "map": cur["map"],
                "safety_x": cur["safety_x"],
                "safety_y": cur["safety_y"],
                "safety_radius": cur["safety_radius"],
                "phase": int(cur["phase"]),
                "dx": nxt["safety_x"] - cur["safety_x"],
                "dy": nxt["safety_y"] - cur["safety_y"],
                "next_radius": nxt["safety_radius"],
                "shrink": nxt["safety_radius"] / cur["safety_radius"],
            })
    return pd.DataFrame(rows)
