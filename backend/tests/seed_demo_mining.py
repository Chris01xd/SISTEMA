"""One-off DEV-ONLY seed script for the mining digital twin / CRISP-DM module.

Mirrors tests/seed_demo_hospital.py: writes directly via SQLAlchemy (there is
no bulk-import API), and is never imported by application code — production
paths never generate synthetic sensor readings, only this manual script does,
strictly for local development and manual testing.

Usage (from backend/, so the `app` package resolves):
    python -m tests.seed_demo_mining
"""
import random
from datetime import date, datetime, timedelta, timezone

from app.authentication.security import hash_password
from app.database.session import SessionLocal
from app.models.crisp_phase import CrispPhase, PhaseStatus
from app.models.crisp_project import CrispProject, CrispProjectLoadingUnit, ProjectOverallStatus
from app.models.equipment import LoadingUnit, LoadingUnitDigitalTwin, LoadingUnitType, OperationalStatus, TwinState
from app.models.sensor import ReadingQuality, Sensor, SensorReading, SensorSource, SensorType
from app.models.user import Role, User

# Permission sets per the module's RBAC design (Administrador reuses the
# existing "admin" role's {"*": True} wildcard and is not recreated here).
ROLE_PERMISSIONS: dict[str, dict[str, bool]] = {
    "maintenance_engineer": {
        "crisp:project:view": True,
        "crisp:phase:edit:business": True,
        "crisp:phase:approve:business": True,
        "crisp:mining:manage": True,
        "crisp:alert:acknowledge": True,
        "crisp:maintenance:manage": True,
        "crisp:kpi:view": True,
        "crisp:report:download": True,
    },
    "data_scientist": {
        "crisp:project:view": True,
        "crisp:phase:edit:data": True,
        "crisp:phase:edit:model": True,
        "crisp:experiment:manage": True,
        "crisp:model:publish": True,
        "crisp:deployment:manage": True,
        "crisp:kpi:view": True,
        "crisp:report:download": True,
    },
    "supervisor": {
        "crisp:project:view": True,
        "crisp:kpi:view": True,
        "crisp:phase:approve:operational": True,
        "crisp:report:download": True,
    },
    "consultant": {
        "crisp:project:view": True,
    },
}

DEMO_USERS = [
    ("admin@mining-twin.dev", "admin", "Admin Demo"),
    ("mantenimiento@mining-twin.dev", "maintenance_engineer", "Ingeniero de Mantenimiento Demo"),
    ("ciencia-datos@mining-twin.dev", "data_scientist", "Científico de Datos Demo"),
    ("supervisor@mining-twin.dev", "supervisor", "Supervisor Demo"),
    ("consultor@mining-twin.dev", "consultant", "Consultor Demo"),
]

DEMO_PASSWORD = "ChangeMe123!"

SENSOR_SPECS = [
    (SensorType.TEMPERATURE, "°C", 60, 85.0, 6.0),
    (SensorType.VIBRATION, "mm/s", 60, 4.0, 1.2),
    (SensorType.HYDRAULIC_PRESSURE, "bar", 60, 210.0, 15.0),
    (SensorType.FUEL_CONSUMPTION, "L/h", 300, 38.0, 5.0),
    (SensorType.ENGINE_RPM, "rpm", 60, 1800.0, 120.0),
]


def get_or_create_role(db, name: str, permissions: dict[str, bool]) -> Role:
    role = db.query(Role).filter(Role.name == name).first()
    if role is None:
        role = Role(name=name, permissions=permissions)
        db.add(role)
        db.flush()
    return role


def seed() -> None:
    db = SessionLocal()
    try:
        admin_role = db.query(Role).filter(Role.name == "admin").first()
        if admin_role is None:
            admin_role = Role(name="admin", permissions={"*": True})
            db.add(admin_role)
            db.flush()

        roles_by_name = {"admin": admin_role}
        for role_name, perms in ROLE_PERMISSIONS.items():
            roles_by_name[role_name] = get_or_create_role(db, role_name, perms)
        db.commit()

        for email, role_name, full_name in DEMO_USERS:
            if db.query(User).filter(User.email == email).first() is not None:
                continue
            db.add(
                User(
                    email=email,
                    hashed_password=hash_password(DEMO_PASSWORD),
                    full_name=full_name,
                    role_id=roles_by_name[role_name].id,
                )
            )
        db.commit()

        units = []
        for code, unit_type, brand, model_name, capacity in [
            ("CAT-793-001", LoadingUnitType.HAUL_TRUCK, "Caterpillar", "793F", 218.0),
            ("KOM-PC8000-002", LoadingUnitType.SHOVEL, "Komatsu", "PC8000", 42.0),
        ]:
            existing = db.query(LoadingUnit).filter(LoadingUnit.code == code).first()
            if existing is not None:
                units.append(existing)
                continue
            unit = LoadingUnit(
                code=code,
                unit_type=unit_type,
                brand=brand,
                model=model_name,
                manufacture_year=2019,
                capacity_tons=capacity,
                fleet="Flota Piloto CRISP-DM",
                current_location="Tajo Norte - Banco 3240",
                operational_status=OperationalStatus.OPERATIONAL,
            )
            db.add(unit)
            db.flush()
            db.add(
                LoadingUnitDigitalTwin(
                    loading_unit_id=unit.id,
                    model_reference=f"twin-{code.lower()}",
                    calibration_parameters={"source": "manufacturer_defaults"},
                    current_state=TwinState.NORMAL,
                    risk_level=0.05,
                )
            )
            units.append(unit)
        db.commit()

        now = datetime.now(timezone.utc)
        rng = random.Random(42)
        for unit in units:
            for sensor_type, unit_of_measure, freq_seconds, base_value, spread in SENSOR_SPECS:
                sensor = (
                    db.query(Sensor)
                    .filter(Sensor.loading_unit_id == unit.id, Sensor.sensor_type == sensor_type)
                    .first()
                )
                if sensor is None:
                    sensor = Sensor(
                        loading_unit_id=unit.id,
                        sensor_type=sensor_type,
                        source=SensorSource.IOT,
                        unit_of_measure=unit_of_measure,
                        sampling_frequency_seconds=freq_seconds,
                    )
                    db.add(sensor)
                    db.flush()
                else:
                    continue  # already has readings from a previous run

                # SYNTHETIC, DEV-ONLY: 14 days of readings at the sensor's own
                # frequency, purely so the EDA/feature-engineering/training
                # pipeline has something real to run against in a dev
                # environment. Never generated on a production code path.
                n_points = int((14 * 24 * 3600) / freq_seconds)
                for i in range(n_points):
                    ts = now - timedelta(seconds=freq_seconds * (n_points - i))
                    value = base_value + rng.gauss(0, spread)
                    db.add(
                        SensorReading(
                            sensor_id=sensor.id,
                            timestamp=ts,
                            value=round(value, 2),
                            quality=ReadingQuality.GOOD,
                            created_at=ts,
                        )
                    )
                db.commit()

        project = db.query(CrispProject).filter(CrispProject.name == "Piloto Flota Norte - CRISP-DM").first()
        if project is None:
            admin_user = db.query(User).filter(User.email == "admin@mining-twin.dev").first()
            project = CrispProject(
                name="Piloto Flota Norte - CRISP-DM",
                description="Proyecto piloto de mantenimiento predictivo para la flota norte del tajo abierto.",
                digital_twin_id=units[0].digital_twin.id,
                responsible_id=admin_user.id if admin_user else None,
                start_date=date.today(),
                estimated_end_date=date.today() + timedelta(weeks=24),
                current_phase_number=1,
                overall_status=ProjectOverallStatus.IN_PROGRESS,
                created_by_id=admin_user.id if admin_user else None,
            )
            db.add(project)
            db.flush()
            for unit in units:
                db.add(CrispProjectLoadingUnit(project_id=project.id, loading_unit_id=unit.id))
            for phase_number in range(1, 7):
                db.add(
                    CrispPhase(
                        project_id=project.id,
                        phase_number=phase_number,
                        status=PhaseStatus.IN_PROGRESS if phase_number == 1 else PhaseStatus.NOT_STARTED,
                    )
                )
            db.commit()

        print("Seed complete:")
        print(f"  Roles: {list(roles_by_name.keys())}")
        print(f"  Demo users (password: {DEMO_PASSWORD}): {[e for e, *_ in DEMO_USERS]}")
        print(f"  Loading units: {[u.code for u in units]}")
        print(f"  Project: {project.name} ({project.id})")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
