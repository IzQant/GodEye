"""
Day 4 작업: 텔레메트리 JSON에서 자기장(원) 관련 이벤트만 추출.

PUBG 텔레메트리는 매치 한 판당 수만 개의 이벤트가 들어있는 JSON 배열이다.
그중 "LogGameStatePeriodic" 이벤트만 자기장 정보(현재 안전지대, 다음 자기장)를
담고 있고, 이 이벤트는 게임 진행 중 주기적으로 반복 기록된다.

추출 대상 필드 (gameState 객체 기준):
- elapsedTime            : 매치 시작 후 경과 시간(초)
- safetyZonePosition     : 현재 안전지대(흰 원) 중심좌표
- safetyZoneRadius       : 현재 안전지대 반경
- poisonGasWarningPosition : 다음 자기장(파란 원) 중심좌표 — 우리가 예측하려는 대상
- poisonGasWarningRadius   : 다음 자기장 반경
"""
import pandas as pd


def parse_zone_events(telemetry: list[dict], match_id: str, map_name: str) -> pd.DataFrame:
    """
    텔레메트리 원본(이벤트 리스트)에서 LogGameStatePeriodic 이벤트만 골라
    한 줄(row)씩 DataFrame으로 변환한다.

    반환되는 DataFrame은 "원본 스냅샷" 단위다. 같은 phase 안에서도 여러 번
    기록되므로, 아직 phase별로 정리된 상태는 아니다 (그 작업은 summarize_phases에서 함).
    """
    rows = []
    for event in telemetry:
        # "_T"는 텔레메트리 이벤트의 타입을 나타내는 필드.
        # LogGameStatePeriodic이 아니면 자기장 정보가 없으므로 건너뛴다.
        if event.get("_T") != "LogGameStatePeriodic":
            continue

        state = event.get("gameState", {})

        # safetyZonePosition/poisonGasWarningPosition이 없는 비정상 이벤트는
        # 스킵한다 (매치 극초반/극후반에 필드가 비어있는 경우가 있음).
        if "safetyZonePosition" not in state or "poisonGasWarningPosition" not in state:
            continue

        rows.append({
            "match_id": match_id,
            "map_name": map_name,
            "elapsed_time": state.get("elapsedTime"),
            "safety_x": state["safetyZonePosition"]["x"],
            "safety_y": state["safetyZonePosition"]["y"],
            "safety_radius": state.get("safetyZoneRadius"),
            "poison_x": state["poisonGasWarningPosition"]["x"],
            "poison_y": state["poisonGasWarningPosition"]["y"],
            "poison_radius": state.get("poisonGasWarningRadius"),
        })

    return pd.DataFrame(rows)


def _assign_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    LogGameStatePeriodic에는 "몇 번째 자기장 단계인지"를 알려주는 필드가 없다.
    대신 poison_radius(다음 원의 반경)는 한 단계 동안 고정된 값을 유지하다가
    다음 단계로 넘어갈 때 값이 바뀐다. 이 값이 바뀌는 지점을 기준으로
    phase 번호를 순서대로 매긴다 (0, 1, 2, ...).
    """
    df = df.sort_values("elapsed_time").reset_index(drop=True)
    # poison_radius가 직전 행과 달라지면 새로운 단계로 판단
    is_new_phase = df["poison_radius"].ne(df["poison_radius"].shift())
    df["phase"] = is_new_phase.cumsum() - 1
    return df


def summarize_phases(df: pd.DataFrame) -> pd.DataFrame:
    """
    이벤트 단위 DataFrame을 phase(단계) 단위로 요약한다.

    각 phase마다:
    - start_time / end_time : 그 단계가 관측된 elapsed_time의 최소/최대값
                               (= 축소 시작·종료 시간의 근사값)
    - safety_x/y, safety_radius : 그 단계에서의 현재 안전지대(마지막 관측값)
    - poison_x/y, poison_radius : 그 단계에서 예측 대상인 다음 자기장(고정값이므로 첫 값)

    완료 기준(Day 4): 매치 1개를 넣었을 때 phase별로 정리된 DataFrame이 나오면 통과.
    """
    if df.empty:
        return pd.DataFrame()

    df = _assign_phase(df)

    summary = df.groupby("phase").agg(
        match_id=("match_id", "first"),
        map_name=("map_name", "first"),
        start_time=("elapsed_time", "min"),
        end_time=("elapsed_time", "max"),
        safety_x=("safety_x", "last"),
        safety_y=("safety_y", "last"),
        safety_radius=("safety_radius", "last"),
        poison_x=("poison_x", "first"),
        poison_y=("poison_y", "first"),
        poison_radius=("poison_radius", "first"),
    ).reset_index()

    return summary
