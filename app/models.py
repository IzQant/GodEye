"""
SQLAlchemy ORM 모델 (Day 8).

4개 테이블:
- matches       : 매치 단위 메타데이터 (맵, shard 등)
- circles       : 매치별 단계(phase)별 자기장 정보 (zones_dataset.csv와 대응)
- predictions   : 예측 결과 + 실제값 + 오차 (모델 평가/로그용)
- request_logs  : API 요청 로그 (사용량 확인, 에러 추적)
"""
import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String

from app.database import Base


class Match(Base):
    """매치 한 판의 메타데이터. match_id를 기본키로 사용."""
    __tablename__ = "matches"

    match_id = Column(String, primary_key=True)
    map_name = Column(String)          # 정규화된 맵 이름 (예: Erangel)
    shard = Column(String)             # 플랫폼 (steam, kakao 등)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Circle(Base):
    """
    매치별 단계(phase)별 자기장 스냅샷.
    telemetry_parser.summarize_phases() 결과 한 줄이 여기 한 행에 대응한다.
    """
    __tablename__ = "circles"

    id = Column(Integer, primary_key=True)
    match_id = Column(String, ForeignKey("matches.match_id"), index=True)
    map_name = Column(String)
    phase = Column(Integer)
    start_time = Column(Float)         # 단계 관측 시작 시간(초)
    end_time = Column(Float)           # 단계 관측 종료 시간(초)
    # 현재 안전지대(흰 원)
    safety_x = Column(Float)
    safety_y = Column(Float)
    safety_radius = Column(Float)
    # 다음 자기장(파란 원) — 예측 대상
    poison_x = Column(Float)
    poison_y = Column(Float)
    poison_radius = Column(Float)


class Prediction(Base):
    """
    예측 결과 저장. 예측값과 (알 수 있으면) 실제값, 그 오차 거리를 함께 기록해
    이후 모델 성능을 사후 분석할 수 있게 한다.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    match_id = Column(String, index=True)
    phase = Column(Integer)
    model_name = Column(String)        # 사용한 모델 (baseline, regression 등)
    # 예측한 다음 원
    pred_x = Column(Float)
    pred_y = Column(Float)
    pred_radius = Column(Float)
    # 실제 다음 원 (검증 가능할 때만)
    actual_x = Column(Float, nullable=True)
    actual_y = Column(Float, nullable=True)
    error_dist = Column(Float, nullable=True)   # 예측-실제 중심 거리(오차)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class RequestLog(Base):
    """API 요청 로그. 사용량 확인 및 에러 추적용."""
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True)
    endpoint = Column(String)
    input_type = Column(String)        # "match_id" | "image"
    status = Column(String)            # "success" | "error"
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
