"""Manual end-to-end smoke script for the CRISP-DM module, against a running
server + real Postgres (already seeded via `python -m tests.seed_demo_mining`).
Not part of the pytest suite (requires live infra); run directly:

    python tests/manual_e2e_smoke_crisp.py

Covers: project creation, phase transitions with required-field gating,
per-role permissions, dataset/version registration, experiment training,
model comparison, evaluation approval, deployment, prediction generation,
digital twin update, alert creation, audit logging, and monitoring/drift.
"""
import sys
import time

import httpx

BASE = "http://127.0.0.1:8010/api/v1"
PASSWORD = "ChangeMe123!"


def login(email: str) -> str:
    r = httpx.post(f"{BASE}/auth/login", json={"email": email, "password": PASSWORD})
    r.raise_for_status()
    return r.json()["access_token"]


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def check(condition: bool, message: str) -> None:
    status = "OK" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        raise SystemExit(1)


def main() -> None:
    admin = login("admin@mining-twin.dev")
    engineer = login("mantenimiento@mining-twin.dev")
    scientist = login("ciencia-datos@mining-twin.dev")
    consultant = login("consultor@mining-twin.dev")
    print("Logged in as admin, maintenance_engineer, data_scientist, consultant")

    units = httpx.get(f"{BASE}/crisp/mining/loading-units", headers=auth(admin)).json()
    check(len(units) >= 1, "seeded loading units present")
    unit = units[0]

    r = httpx.post(
        f"{BASE}/crisp/projects",
        headers=auth(admin),
        json={
            "name": f"Smoke Test Project {int(time.time())}",
            "description": "e2e smoke",
            "start_date": "2026-01-01",
            "loading_unit_ids": [unit["id"]],
        },
    )
    check(r.status_code == 200, f"create project ({r.status_code}: {r.text[:200]})")
    project = r.json()
    project_id = project["id"]

    r = httpx.post(f"{BASE}/crisp/projects", headers=auth(consultant), json={"name": "x", "start_date": "2026-01-01"})
    check(r.status_code == 403, f"consultant is blocked from creating projects ({r.status_code})")

    phases = httpx.get(f"{BASE}/crisp/projects/{project_id}/phases", headers=auth(admin)).json()
    check(len(phases) == 6, "project has 6 phases")
    phase1 = next(p for p in phases if p["phase_number"] == 1)

    r = httpx.post(f"{BASE}/crisp/phases/{phase1['id']}/approve", headers=auth(engineer), json={"action": "APPROVE"})
    check(r.status_code == 400, f"approve blocked with missing required fields ({r.status_code})")

    business_data = {
        "business_problem": "Fallos imprevistos en unidades de carguío.",
        "general_objective": "Reducir fallos imprevistos mediante mantenimiento predictivo.",
        "business_success_criteria": ["Adopción por el equipo de mantenimiento"],
        "kpis": [{"indicator": "MTBF", "unit": "horas", "baseline_value": 120.0, "target_value": 180.0}],
    }
    r = httpx.put(f"{BASE}/crisp/phases/{phase1['id']}", headers=auth(engineer), json={"data": business_data})
    check(r.status_code == 200, f"maintenance engineer edits phase 1 ({r.status_code}: {r.text[:200]})")

    r = httpx.put(f"{BASE}/crisp/phases/{phase1['id']}", headers=auth(scientist), json={"data": business_data})
    check(r.status_code == 403, f"data scientist blocked from editing phase 1 ({r.status_code})")

    r = httpx.post(f"{BASE}/crisp/phases/{phase1['id']}/approve", headers=auth(engineer), json={"action": "APPROVE"})
    check(r.status_code == 200, f"approve phase 1 ({r.status_code}: {r.text[:200]})")

    r = httpx.post(
        f"{BASE}/crisp/datasources",
        headers=auth(scientist),
        json={
            "project_id": project_id,
            "name": "Temperatura motor",
            "source_type": "IOT_SENSOR",
            "loading_unit_id": unit["id"],
            "measured_variable": "TEMPERATURE",
        },
    )
    check(r.status_code == 200, f"create data source ({r.status_code}: {r.text[:200]})")
    source = r.json()

    r = httpx.post(f"{BASE}/crisp/datasources/{source['id']}/compute-eda", headers=auth(scientist))
    check(r.status_code == 200, f"compute EDA ({r.status_code}: {r.text[:300]})")
    eda = r.json()
    check(eda["record_count"] > 0, "EDA computed over real seeded readings")

    phase2 = next(p for p in phases if p["phase_number"] == 2)
    r = httpx.post(f"{BASE}/crisp/phases/{phase2['id']}/approve", headers=auth(engineer), json={"action": "APPROVE"})
    check(r.status_code == 200, f"approve phase 2 ({r.status_code}: {r.text[:200]})")

    r = httpx.post(f"{BASE}/crisp/datasets", headers=auth(scientist), json={"project_id": project_id, "name": "Sensores unidad piloto"})
    check(r.status_code == 200, f"create dataset ({r.status_code}: {r.text[:200]})")
    dataset = r.json()

    r = httpx.post(
        f"{BASE}/crisp/datasets/{dataset['id']}/versions",
        headers=auth(scientist),
        json={
            "dataset_id": dataset["id"],
            "name": "v1",
            "operations_applied": [
                {"step_type": "CLEANING"},
                {"step_type": "DEDUPLICATION"},
                {"step_type": "MISSING_VALUE_IMPUTATION"},
                {"step_type": "FAILURE_LABELING", "parameters": {"horizon_hours": 48}},
            ],
        },
    )
    check(r.status_code == 200, f"create dataset version ({r.status_code}: {r.text[:300]})")
    version = r.json()
    check(version["record_count"] > 0 and "failure_within_horizon" in version["variables"], "dataset version materialized with labels")

    phase3 = next(p for p in phases if p["phase_number"] == 3)
    r = httpx.post(f"{BASE}/crisp/phases/{phase3['id']}/approve", headers=auth(engineer), json={"action": "APPROVE"})
    check(r.status_code == 200, f"approve phase 3 ({r.status_code}: {r.text[:200]})")

    input_vars = [v for v in version["variables"] if v != "failure_within_horizon"]
    r = httpx.post(
        f"{BASE}/crisp/experiments",
        headers=auth(scientist),
        json={
            "project_id": project_id,
            "dataset_version_id": version["id"],
            "objective": "FAILURE_CLASSIFICATION",
            "algorithm": "RANDOM_FOREST",
            "input_variables": input_vars,
            "target_variable": "failure_within_horizon",
            "hyperparameters": {"n_estimators": 50, "max_depth": 5},
        },
    )
    check(r.status_code == 200, f"create experiment ({r.status_code}: {r.text[:300]})")
    experiment = r.json()

    r = httpx.post(f"{BASE}/crisp/experiments/{experiment['id']}/train", headers=auth(scientist), timeout=60)
    check(r.status_code == 200, f"train experiment ({r.status_code}: {r.text[:300]})")
    trained = r.json()
    check(trained["status"] == "COMPLETED", "experiment status COMPLETED")

    metrics = httpx.get(f"{BASE}/crisp/experiments/{experiment['id']}/metrics", headers=auth(scientist)).json()
    check(len(metrics) > 0, f"metrics recorded: {[m['metric_name'] for m in metrics]}")

    compare = httpx.get(f"{BASE}/crisp/projects/{project_id}/experiments/compare", headers=auth(scientist)).json()
    check(len(compare) == 1, "comparison table has 1 experiment")

    phase4 = next(p for p in phases if p["phase_number"] == 4)
    r = httpx.post(f"{BASE}/crisp/phases/{phase4['id']}/approve", headers=auth(engineer), json={"action": "APPROVE"})
    check(r.status_code == 200, f"approve phase 4 ({r.status_code}: {r.text[:200]})")

    models = httpx.get(f"{BASE}/crisp/experiments/{experiment['id']}/models", headers=auth(scientist)).json()
    check(len(models) == 1, "model registered from experiment")
    model = models[0]

    r = httpx.post(
        f"{BASE}/crisp/models/{model['id']}/evaluate",
        headers=auth(engineer),
        json={
            "model_id": model["id"],
            "operational_metrics": {"anticipation_time_hours": 24},
            "business_metrics_estimate": {"estimated_mtbf_impact": 15.0},
            "specialist_opinion": "Recall aceptable para fallos críticos.",
            "decision": "APPROVED",
            "justification": "Cumple el criterio mínimo de recall acordado.",
        },
    )
    check(r.status_code == 200, f"evaluate model ({r.status_code}: {r.text[:300]})")

    phase5 = next(p for p in phases if p["phase_number"] == 5)
    r = httpx.post(f"{BASE}/crisp/phases/{phase5['id']}/approve", headers=auth(engineer), json={"action": "APPROVE"})
    check(r.status_code == 200, f"approve phase 5 ({r.status_code}: {r.text[:200]})")

    r = httpx.post(
        f"{BASE}/crisp/models/{model['id']}/deploy",
        headers=auth(scientist),
        json={"model_id": model["id"], "project_id": project_id, "loading_unit_ids": [unit["id"]], "alert_thresholds": {"warning": 0.01, "critical": 0.02}},
    )
    check(r.status_code == 200, f"deploy model ({r.status_code}: {r.text[:300]})")
    deployment = r.json()

    phase6 = next(p for p in phases if p["phase_number"] == 6)
    r = httpx.post(f"{BASE}/crisp/phases/{phase6['id']}/approve", headers=auth(engineer), json={"action": "APPROVE"})
    check(r.status_code == 200, f"approve phase 6 ({r.status_code}: {r.text[:200]})")

    r = httpx.post(f"{BASE}/crisp/predictions", headers=auth(scientist), json={"deployment_id": deployment["id"], "loading_unit_id": unit["id"]})
    check(r.status_code == 200, f"generate prediction ({r.status_code}: {r.text[:300]})")
    prediction = r.json()
    print(f"      prediction: state={prediction['equipment_state']} risk={prediction['risk_level']}")

    twin = httpx.get(f"{BASE}/crisp/mining/loading-units/{unit['id']}/digital-twin", headers=auth(admin)).json()
    check(twin["current_state"] == prediction["equipment_state"], "digital twin state updated from prediction")

    if prediction["linked_alert_id"]:
        alerts = httpx.get(f"{BASE}/crisp/mining/alerts", headers=auth(engineer), params={"loading_unit_id": unit["id"]}).json()
        check(any(a["id"] == prediction["linked_alert_id"] for a in alerts), "alert created and listed")
        r = httpx.post(f"{BASE}/crisp/mining/alerts/{prediction['linked_alert_id']}/acknowledge", headers=auth(engineer), json={"comments": "revisado"})
        check(r.status_code == 200, f"acknowledge alert ({r.status_code})")
    else:
        print("[SKIP] prediction was below alert threshold, no alert to check")

    audit = httpx.get(f"{BASE}/crisp/projects/{project_id}/audit", headers=auth(admin)).json()
    check(len(audit) > 5, f"audit log has entries ({len(audit)})")

    r = httpx.get(f"{BASE}/crisp/projects/{project_id}/report", headers=auth(admin), params={"fmt": "pdf"})
    check(r.status_code == 200 and r.headers["content-type"] == "application/pdf", f"final PDF report generated ({r.status_code})")

    r = httpx.get(f"{BASE}/crisp/projects/{project_id}/report", headers=auth(admin), params={"fmt": "xlsx"})
    check(r.status_code == 200, f"final XLSX report generated ({r.status_code})")

    monitoring = httpx.post(f"{BASE}/crisp/deployments/{deployment['id']}/monitoring/refresh", headers=auth(scientist)).json()
    check("prediction_count" in monitoring, "monitoring snapshot recorded")

    r = httpx.post(f"{BASE}/crisp/deployments/{deployment['id']}/drift/data", headers=auth(scientist))
    check(r.status_code == 200, f"data drift check ({r.status_code}: {r.text[:200]})")
    drift = r.json()
    print(f"      data drift: variable={drift['variable_name']} psi={drift['drift_score']} significant={drift['is_significant']}")

    r = httpx.get(f"{BASE}/crisp/deployments/{deployment['id']}/drift", headers=auth(scientist))
    check(r.status_code == 200 and len(r.json()) >= 1, "drift records listed")

    print("\nAll CRISP-DM smoke checks passed.")


if __name__ == "__main__":
    main()
