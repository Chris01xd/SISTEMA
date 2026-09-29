from fastapi import APIRouter, Depends

from app.authentication.dependencies import get_current_user, require_permission
from app.models.user import User
from app.schemas.ai import AIInvokeRequest, AIInvokeResponse, AIStatus
from app.services import ai_module_service

router = APIRouter()


@router.get("/status", response_model=AIStatus, dependencies=[Depends(get_current_user)])
def get_status() -> dict:
    """Readiness of the AI module: twin data, prediction model and LLM (never exposes keys)."""
    return ai_module_service.status()


# Plain `def`: the LangGraph run (simulation + LLM round-trip) blocks for seconds, so FastAPI
# runs it in its threadpool instead of stalling the event loop that serves the WebSocket twin.
@router.post("/invoke", response_model=AIInvokeResponse)
def invoke(payload: AIInvokeRequest, user: User = Depends(require_permission("assistant:ask"))) -> dict:
    """Natural-language question answered by the LangGraph workflow over the digital twin.
    Values come from the twin (observed / simulated / predicted); the LLM only interprets."""
    return ai_module_service.invoke(payload.message, payload.parameters, user.id)
