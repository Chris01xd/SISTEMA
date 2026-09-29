"""CRISP-DM phase 5 (evaluation): technical metrics stay in `model_metrics`;
this table holds the business/operational read-out and the go/no-go decision.
Business figures are explicitly framed as *estimates* pending pilot data."""
import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, ForeignKey, JSON, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPKMixin


class EvaluationDecision(str, enum.Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RETURN_TO_DATA_UNDERSTANDING = "RETURN_TO_DATA_UNDERSTANDING"
    RETURN_TO_DATA_PREPARATION = "RETURN_TO_DATA_PREPARATION"


class ModelEvaluation(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "model_evaluations"

    model_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("ml_models.id"), index=True)
    evaluator_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    # {tiempo_anticipacion_horas, fallos_criticos_detectados, fallos_no_detectados, alertas_falsas}
    operational_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    # {reduccion_paradas_estimada, impacto_mtbf_estimado, impacto_mttr_estimado, ahorro_estimado}
    # — always "_estimado": these are projections, never claimed as measured results.
    business_metrics_estimate: Mapped[dict] = mapped_column(JSON, default=dict)
    specialist_opinion: Mapped[str] = mapped_column(Text, default="")
    decision: Mapped[EvaluationDecision | None] = mapped_column(Enum(EvaluationDecision), nullable=True)
    justification: Mapped[str] = mapped_column(Text, default="")
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
