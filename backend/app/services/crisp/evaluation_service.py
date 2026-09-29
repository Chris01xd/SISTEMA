"""Phase 5 (evaluation): records the specialist's technical review, the
explicitly-estimated business impact, and the go/no-go decision. Technical
metrics themselves live in `model_metrics` (written during training) — this
service only reads them back for the comparison view."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.crisp_evaluation import EvaluationDecision, ModelEvaluation
from app.models.crisp_modeling import MlModel
from app.schemas.crisp_evaluation import ModelEvaluationCreate
from app.services.crisp.audit_service import record_crisp_audit


def create_evaluation(db: Session, payload: ModelEvaluationCreate, actor_id: uuid.UUID) -> ModelEvaluation:
    model = db.get(MlModel, payload.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    evaluation = ModelEvaluation(
        model_id=payload.model_id,
        evaluator_id=actor_id,
        operational_metrics=payload.operational_metrics.model_dump(),
        business_metrics_estimate=payload.business_metrics_estimate.model_dump(),
        specialist_opinion=payload.specialist_opinion,
        decision=payload.decision,
        justification=payload.justification,
        evaluated_at=datetime.now(timezone.utc),
    )
    db.add(evaluation)

    if payload.decision == EvaluationDecision.APPROVED:
        model.is_candidate = True
    elif payload.decision == EvaluationDecision.REJECTED:
        model.is_candidate = False

    db.commit()
    db.refresh(evaluation)

    record_crisp_audit(
        db, user_id=actor_id, action="EVALUATE_MODEL", entity_type="ModelEvaluation", entity_id=str(evaluation.id),
        new_value={"decision": evaluation.decision.value, "model_id": str(model.id)},
    )
    return evaluation


def list_evaluations_for_model(db: Session, model_id: uuid.UUID) -> list[ModelEvaluation]:
    return db.query(ModelEvaluation).filter(ModelEvaluation.model_id == model_id).order_by(ModelEvaluation.created_at.desc()).all()
