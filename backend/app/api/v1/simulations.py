import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.authentication.dependencies import require_permission
from app.database.session import get_db
from app.models.simulation import Simulation, SimulationResult
from app.models.user import User
from app.schemas.simulation import SimulationCreate, SimulationOut, SimulationResultOut
from app.services.simulation_service import create_simulation, run_simulation

router = APIRouter()


@router.post("", response_model=SimulationOut, dependencies=[Depends(require_permission("simulation:run"))])
def create_simulation_endpoint(
    payload: SimulationCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(require_permission("simulation:run")),
) -> SimulationOut:
    simulation = create_simulation(db, user.id, payload)
    background_tasks.add_task(run_simulation, simulation.id)
    return simulation


@router.get("/{simulation_id}", response_model=SimulationOut)
def get_simulation_endpoint(simulation_id: uuid.UUID, db: Session = Depends(get_db)) -> SimulationOut:
    sim = db.get(Simulation, simulation_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return sim


@router.get("/{simulation_id}/results", response_model=list[SimulationResultOut])
def get_simulation_results(
    simulation_id: uuid.UUID, limit: int = 500, db: Session = Depends(get_db)
) -> list[SimulationResultOut]:
    results = (
        db.query(SimulationResult)
        .filter(SimulationResult.simulation_id == simulation_id)
        .order_by(SimulationResult.tick)
        .limit(limit)
        .all()
    )
    return results
