from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from app.authentication.dependencies import get_current_user, require_permission
from app.models.user import User
from app.schemas.ml_analysis import MLAnalysisRequest, MLAnalysisResult
from app.services import ml_analysis_service

router = APIRouter()


# Plain `def`: training + 5-fold CV of 7 models takes tens of seconds; FastAPI runs it in
# its threadpool so the event loop (WebSocket twin) keeps serving.
@router.post("/analysis", response_model=MLAnalysisResult)
def run_analysis(payload: MLAnalysisRequest, user: User = Depends(require_permission("simulation:run"))) -> dict:
    """Trains >=4 normal + 2 hybrid models, cross-validates, cross-references, ranks and selects
    the best model mathematically (Langflow flow `LANGFLOW_FLOW_ID`)."""
    return ml_analysis_service.run_analysis(payload, user.id)


@router.get("/status", dependencies=[Depends(get_current_user)])
def get_status() -> dict:
    return ml_analysis_service.status()


@router.get("/best-model", dependencies=[Depends(get_current_user)])
def get_best_model() -> dict:
    """Metadata of the last selected (persisted) best model."""
    return ml_analysis_service.latest_best_model()


_MEDIA = {"pdf": "application/pdf", "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}


@router.get("/analysis/{run_id}/report", dependencies=[Depends(require_permission("reports:generate"))])
def download_report(run_id: str, format: Literal["pdf", "xlsx"] = Query("pdf")) -> FileResponse:
    """Report of an analysis (CRISP-DM sections; every table/chart with its interpretation)."""
    path = ml_analysis_service.build_report(run_id, format)
    return FileResponse(path, media_type=_MEDIA[format], filename=path.name)
