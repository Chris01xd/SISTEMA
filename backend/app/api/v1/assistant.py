from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.authentication.dependencies import get_current_user, require_permission
from app.database.session import get_db
from app.models.user import User
from app.schemas.assistant import AskRequest, AskResponse, AssistantStatus
from app.services import assistant_service, dataset_service

router = APIRouter()


@router.get("/status", response_model=AssistantStatus, dependencies=[Depends(get_current_user)])
def get_status() -> dict:
    """Whether the assistant is installed and configured (never exposes the key)."""
    return assistant_service.status()


# Plain `def` on purpose: the LLM round-trip blocks for seconds, so FastAPI runs it in its
# threadpool instead of stalling the event loop (which also serves the WebSocket twin).
@router.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("assistant:ask")),
) -> dict:
    """Natural-language question about an ingested dataset, answered by an LLM that can only
    reach the data through read-only, aggregate-only tools."""
    dataset = dataset_service.get_dataset(db, payload.dataset_id)
    return assistant_service.ask(dataset, payload.question, payload.history, user.id)
