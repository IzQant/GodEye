"""초기 스키마: matches, circles, predictions, request_logs

Revision ID: 0001_init
Revises:
Create Date: 2026-07-29

이 마이그레이션은 app/models.py의 4개 테이블을 그대로 생성한다.
초보자가 `alembic upgrade head` 한 번으로 스키마를 만들 수 있도록 직접 작성했다.
(이후 모델을 바꿀 때는 `alembic revision --autogenerate -m "..."`로 diff 생성)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001_init"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # matches: 매치 메타데이터
    op.create_table(
        "matches",
        sa.Column("match_id", sa.String(), primary_key=True),
        sa.Column("map_name", sa.String()),
        sa.Column("shard", sa.String()),
        sa.Column("created_at", sa.DateTime()),
    )

    # circles: 매치별 phase별 자기장 정보
    op.create_table(
        "circles",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.String(), sa.ForeignKey("matches.match_id")),
        sa.Column("map_name", sa.String()),
        sa.Column("phase", sa.Integer()),
        sa.Column("start_time", sa.Float()),
        sa.Column("end_time", sa.Float()),
        sa.Column("safety_x", sa.Float()),
        sa.Column("safety_y", sa.Float()),
        sa.Column("safety_radius", sa.Float()),
        sa.Column("poison_x", sa.Float()),
        sa.Column("poison_y", sa.Float()),
        sa.Column("poison_radius", sa.Float()),
    )
    op.create_index("ix_circles_match_id", "circles", ["match_id"])

    # predictions: 예측 결과 + 실제값 + 오차
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.String()),
        sa.Column("phase", sa.Integer()),
        sa.Column("model_name", sa.String()),
        sa.Column("pred_x", sa.Float()),
        sa.Column("pred_y", sa.Float()),
        sa.Column("pred_radius", sa.Float()),
        sa.Column("actual_x", sa.Float(), nullable=True),
        sa.Column("actual_y", sa.Float(), nullable=True),
        sa.Column("error_dist", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_predictions_match_id", "predictions", ["match_id"])

    # request_logs: API 요청 로그
    op.create_table(
        "request_logs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("endpoint", sa.String()),
        sa.Column("input_type", sa.String()),
        sa.Column("status", sa.String()),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime()),
    )


def downgrade() -> None:
    op.drop_table("request_logs")
    op.drop_index("ix_predictions_match_id", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_circles_match_id", table_name="circles")
    op.drop_table("circles")
    op.drop_table("matches")
