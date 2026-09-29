# Test Plan

## 1. Scope

Validates the hybrid simulation core (ABM + System Dynamics), the AI layer, the
report generators, and the API/integration surface described in
`docs/ARCHITECTURE.md`.

## 2. Automated Test Suite (`backend/tests/`)

Run with: `cd backend && pytest -q` (no external services required — these
tests exercise pure Python simulation/report logic, not the live API/DB).

| File | What it verifies |
|---|---|
| `test_system_dynamics.py` | SD stock/flow invariants: saturation index stays in [0,1]; an empty system reports near-zero saturation; status-band thresholds (GREEN/YELLOW/RED) match the documented cutoffs. |
| `test_abm.py` | ABM invariants under a fixed seed: occupied beds never exceed capacity, no stock goes negative, and patients are eventually discharged/deceased (the model doesn't deadlock). |
| `test_hybrid_engine_smoke.py` | End-to-end tick loop: runs 90 ticks of the coupled ABM+SD engine and asserts every tick's `status_band` and `saturation_index` are well-formed and bed occupancy stays within capacity. |
| `test_reports.py` | DOCX/PDF/XLSX generators produce non-empty files from a sample metrics/timeseries payload. |

**Status: all 10 tests passing** (validated in this delivery against a fresh
virtualenv with `mesa==3.5.1`, confirming the Mesa 3.x AgentSet-based API
migration — `mesa.time.RandomActivation` was removed upstream and the ABM
scheduler was rewritten around an explicit shuffled activation list).

## 3. Manual / Integration Smoke Test (`backend/tests/manual_e2e_smoke.py`)

Requires a live Postgres + running API (`docker compose up postgres backend`,
or a local Postgres + `uvicorn app.main:app`). Exercises the full user-facing
flow:

1. `POST /auth/register` (role `admin`) → `POST /auth/login`
2. `POST /hospitals` → `POST /hospitals/{id}/emergency-events`
3. `POST /simulations` (triggers the background hybrid-engine run)
4. Poll `GET /simulations/{id}` until `COMPLETED`
5. `GET /simulations/{id}/results` — verify tick count and metrics shape
6. `POST /reports/generate` (format=PDF) — verify a real file is written to
   `generated_reports/`

**Status: validated in this delivery** — a 60-minute/60-tick earthquake
scenario (60 expected victims) ran to completion, produced 60
`SimulationResult` rows, and generated a 7 KB PDF report end-to-end against a
containerized PostgreSQL 16 instance.

Known environment gotcha documented here for future runs: `passlib==1.7.4`'s
bcrypt backend probe is incompatible with `bcrypt>=4.1` (raises
`ValueError: password cannot be longer than 72 bytes` during backend
detection, unrelated to actual password length) — `requirements.txt` pins
`bcrypt==4.0.1` to avoid this.

## 4. AI Pipeline Validation

`python -m app.artificial_intelligence.training.train_demand_forecaster`
generates synthetic MCE traces via the hybrid engine itself and trains the
XGBoost demand forecaster end-to-end, saving a model artifact. Validated in
this delivery (3 synthetic traces → 1,044 training samples → artifact saved).
The GA resource optimizer (`genetic_optimize`) and tabular
`QLearningAssignmentPolicy` were smoke-tested independently and return
well-formed `AllocationCandidate`/action outputs.

## 5. Frontend

- `npm run build` (`tsc -b && vite build`) must complete with zero type
  errors — run as part of CI before any deploy.
- Manual UI walkthrough: load `/twin` against a hospital with a seeded
  floor/room/bed snapshot, confirm room/bed colors reflect status, confirm
  OrbitControls (rotate/zoom/pan) work, confirm the WebSocket-driven
  Dashboard KPIs update while a simulation is running.

## 6. Regression Invariants to Always Re-check

- `beds_occupied <= beds_total` and `staff_active <= staff_total` at every tick.
- `0 <= saturation_index <= 1` at every tick.
- `total_patients_seen == patients_deceased + patients_discharged + (all other in-flight states)`.
- Report generation must never crash on an empty `SimulationResult` set (a
  simulation with `FAILED` status and zero ticks) — add a regression test if
  this path changes.

## 7. Verification on 2026-09-28

- Repository-root command: `backend/.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider`.
  Result: **232 passed** in 187.09 seconds, with six Mesa/FastAPI deprecation warnings.
- `pytest.ini` supplies the module paths and importlib import mode so the three
  suites can run together without colliding `tests` package names.
- The backend-key fallback test now explicitly isolates its provider settings
  from the developer's local AI configuration.
- Frontend: `npm run build` passed (TypeScript and Vite). Vite reports the
  existing large-bundle warning (approximately 1.67 MB before gzip).
- Real public-demo ML run: all seven models completed, with results recorded in
  [the comparison report](ml-model-comparison.md#verified-local-run-2026-09-28).
- Langflow 1.12.3 / Python 3.13: `pip check` passed. The exported flow has
  14 nodes and 31 edges. `langflow_integration/scripts/smoke_flow.py` passed
  both error propagation and a real seven-model run through the deserialized
  graph. Run `20260928T204838Z-00c2fba1` completed in 37.23 seconds and selected
  Logistic Regression (final score 0.5627).
- The Langflow check used the local executor; server authentication, HTTP
  deployment and a browser walkthrough were not exercised. Backend-to-Langflow
  HTTP contract coverage in the Python suite uses a mocked transport.

## 8. Future Test Debt (Phase 1+ per roadmap)

- Contract tests for every API router via `httpx.AsyncClient` against an
  in-memory/test Postgres (testcontainers) instead of relying on the manual
  smoke script.
- Property-based tests (Hypothesis) for the SD equations across parameter
  ranges.
- Bias/safety audit test suite for the ML triage re-ranking signal before any
  clinical-facing deployment.
