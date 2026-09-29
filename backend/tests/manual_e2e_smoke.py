"""Manual end-to-end smoke script against a running server + real Postgres.
Not part of the pytest suite (requires live infra); run directly:
    python tests/manual_e2e_smoke.py
"""
import time

import httpx

BASE = "http://127.0.0.1:8010/api/v1"


def main() -> None:
    email = f"ops_{int(time.time())}@hospital-demo.example.com"
    r = httpx.post(f"{BASE}/auth/register", json={
        "email": email, "password": "S3cure!Pass", "full_name": "Ops Manager", "role_name": "admin",
    })
    print("register:", r.status_code, r.json())

    r = httpx.post(f"{BASE}/auth/login", json={"email": email, "password": "S3cure!Pass"})
    print("login:", r.status_code)
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # grant full permissions to the auto-created "operator" role directly via a second admin call
    # (prototype: the register endpoint seeds simulation:run + reports:generate already)

    r = httpx.post(f"{BASE}/hospitals", json={
        "name": "Hospital General de Prueba", "location": "Ciudad Ejemplo", "total_capacity": 200,
    }, headers=headers)
    print("create hospital:", r.status_code, r.text[:300])
    if r.status_code != 200:
        print("Hospital creation requires twin:admin permission; skipping remaining hospital-scoped steps.")
        return
    hospital_id = r.json()["id"]

    r = httpx.post(f"{BASE}/hospitals/{hospital_id}/emergency-events", json={
        "hospital_id": hospital_id,
        "name": "Terremoto M8.0 - Smoke Test",
        "event_type": "EARTHQUAKE",
        "magnitude": 8.0,
        "expected_victims": 60,
        "severity_distribution": {"minor": 0.4, "delayed": 0.3, "immediate": 0.2, "critical": 0.08, "expectant": 0.02},
        "arrival_curve": "surge_decay",
        "arrival_window_minutes": 30,
    }, headers=headers)
    print("create event:", r.status_code, r.text[:300])
    event_id = r.json()["id"]

    r = httpx.post(f"{BASE}/simulations", json={
        "event_id": event_id, "horizon_minutes": 60, "timestep_minutes": 1,
        "use_ai_optimizer": False, "assignment_strategy": "greedy",
    }, headers=headers)
    print("create simulation:", r.status_code, r.text[:300])
    sim_id = r.json()["id"]

    for _ in range(20):
        r = httpx.get(f"{BASE}/simulations/{sim_id}")
        status = r.json()["status"]
        print("simulation status:", status)
        if status in ("COMPLETED", "FAILED"):
            break
        time.sleep(1)

    r = httpx.get(f"{BASE}/simulations/{sim_id}/results")
    results = r.json()
    print("results ticks:", len(results), "sample:", results[-1] if results else None)

    r = httpx.post(f"{BASE}/reports/generate", json={"simulation_id": sim_id, "format": "PDF"}, headers=headers)
    print("generate report:", r.status_code, r.text[:300])


if __name__ == "__main__":
    main()
