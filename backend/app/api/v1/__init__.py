from fastapi import APIRouter

from app.api.v1 import ai, assistant, auth, hospitals, simulations, reports, ws, mining, crisp, datasets, digital_twin, ml_analysis

api_router = APIRouter()
api_router.include_router(ai.router, prefix="/ai", tags=["ai-module"])
api_router.include_router(digital_twin.router, prefix="/digital-twin", tags=["ai-module"])
api_router.include_router(ml_analysis.router, prefix="/ml", tags=["ml-comparison"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(assistant.router, prefix="/assistant", tags=["assistant"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(hospitals.router, prefix="/hospitals", tags=["hospitals"])
api_router.include_router(simulations.router, prefix="/simulations", tags=["simulations"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(ws.router, tags=["realtime"])
api_router.include_router(mining.router, prefix="/crisp/mining", tags=["mining"])
api_router.include_router(crisp.router, prefix="/crisp", tags=["crisp-dm"])
