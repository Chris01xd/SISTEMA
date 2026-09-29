"""Import all ORM models so Base.metadata is complete for Alembic autogenerate / create_all."""
from app.models.audit import AuditEvent  # noqa: F401
from app.models.user import Role, User  # noqa: F401
from app.models.hospital import Hospital, Floor, Room, Bed  # noqa: F401
from app.models.clinical import Patient, MedicalStaff, Resource  # noqa: F401
from app.models.emergency import EmergencyEvent  # noqa: F401
from app.models.simulation import Simulation, SimulationResult, AIPrediction  # noqa: F401
from app.models.dataset import Dataset  # noqa: F401
from app.models.report import Report  # noqa: F401

# Mining digital twin domain (CRISP-DM module)
from app.models.equipment import LoadingUnit, LoadingUnitDigitalTwin  # noqa: F401
from app.models.sensor import Sensor, SensorReading  # noqa: F401
from app.models.maintenance import Alert, MaintenanceOrder  # noqa: F401
from app.models.crisp_project import CrispProject, CrispProjectLoadingUnit, CrispProjectMember  # noqa: F401
from app.models.crisp_phase import CrispPhase, CrispPhaseActivity, CrispPhaseApproval  # noqa: F401
from app.models.crisp_data import (  # noqa: F401
    DataSource,
    CrispDataset,
    DatasetVersion,
    DataPreparationStep,
    EngineeredFeature,
)
from app.models.crisp_modeling import MlExperiment, MlModel, ModelMetric  # noqa: F401
from app.models.crisp_evaluation import ModelEvaluation  # noqa: F401
from app.models.crisp_deployment import (  # noqa: F401
    ModelDeployment,
    ModelDeploymentLoadingUnit,
    ModelPrediction,
    ModelMonitoring,
    DataDriftRecord,
)
from app.models.crisp_governance import CrispDocument, CrispAuditLog  # noqa: F401
