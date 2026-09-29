# Hospital Digital Twin — MCE Resource Optimization

**Digital Twin-Driven Real-Time Optimization of Hospital Resource Allocation During
Mass Casualty Events: A Hybrid Agent-Based and System Dynamics Model**

A functional prototype: a 3D digital twin of a hospital, driven by a hybrid
Agent-Based / System Dynamics simulation engine, with an AI layer for demand
forecasting and resource-reallocation recommendations, real-time WebSocket
sync between simulation and 3D view, and automated DOCX/PDF/XLSX reporting.

Full architecture, data model, simulation math, and AI model justification:
**[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**. Test plan and validation
results: **[`docs/TEST_PLAN.md`](docs/TEST_PLAN.md)**.

## Stack

- **Backend**: Python 3.12, FastAPI, SQLAlchemy 2.0, Alembic, PostgreSQL, JWT auth, WebSockets
- **Simulation**: Mesa 3.x (ABM) + custom System Dynamics stock/flow model, hybrid-coupled
- **AI**: scikit-learn / XGBoost / PyTorch (LSTM) forecasting, RandomForest saturation classification, DEAP genetic optimization, tabular Q-learning
- **Reports**: python-docx, ReportLab, pandas + openpyxl
- **Frontend**: React 18, TypeScript, Vite, TailwindCSS, Three.js / React Three Fiber, Zustand, React Query

## Quick Start (Docker)

```bash
cp backend/.env.example backend/.env   # edit JWT_SECRET_KEY before any real deployment
docker compose up --build
```

- Backend: http://localhost:8000 (docs at `/docs`, health at `/health`)
- Frontend: http://localhost:5173
- PostgreSQL: localhost:5432 (`twin_user` / `twin_pass` / `hospital_twin`)

Tables are auto-created on backend startup for prototype convenience
(`Base.metadata.create_all`); switch to `alembic upgrade head` migrations for
any environment beyond local development.

**First build is slow, subsequent ones are fast.** The backend Dockerfile
installs `mesa`, `scikit-learn`, `xgboost`, `scipy`, etc., which is a real but
one-time cost (~2-4 min depending on connection); Docker layer caching makes
every later `docker compose up --build` near-instant unless
`requirements.txt` changes. `torch` (only needed for the optional LSTM
forecaster) is deliberately excluded from the default install — see
`backend/requirements-ai-extra.txt` — because its default PyPI wheel resolves
to a CUDA build that alone can add several minutes and 1-2GB to the build.

## Local Development (without Docker)

### Backend

```bash
cd backend
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
cp .env.example .env          # point DATABASE_URL at your local Postgres
uvicorn app.main:app --reload
```

Run the automated test suite (no live DB required — pure simulation/report logic):

```bash
pytest -q
```

Run the end-to-end smoke test (requires a running API + Postgres):

```bash
python tests/manual_e2e_smoke.py
```

Train the demand forecaster against synthetic MCE traces:

```bash
python -m app.artificial_intelligence.training.train_demand_forecaster
```

### Public dataset (MIMIC-IV demo)

The system ships integrated with the open-access **MIMIC-IV Clinical Database Demo**
(100 real, de-identified patients / 275 hospital admissions, with ICU stays, diagnoses and
emergency-department triage; ODbL license, no credentials needed). Log in as an admin, open
**Public data** in the UI (or `POST /api/v1/datasets/ingest {"key": "mimic-iv-demo"}`) and the
backend downloads it from PhysioNet, verifies every file against the published SHA-256
checksums, runs the ETL into the canonical schema and stores it as Parquet under
`backend/data/public/` plus a row in `datasets`. `GET /datasets/{id}/summary` and `/preview`
expose statistics computed from the data. Full credentialed MIMIC-IV works with the same
loader by pointing `raw_path` at the official `hosp/`, `icu/`, `ed/` folders.

### LangChain assistant (standalone module)

`langchain_module/` is an independent LangChain agent that answers natural-language questions about the
imported dataset through read-only, aggregate-only tools. It has its own venv and requirements and imports nothing
from `backend/`. See [`langchain_module/README.md`](langchain_module/README.md).

It is exposed by the backend as **`POST /api/v1/assistant/ask`** (`{"dataset_id", "question", "history"?}` ->
`{"answer", "tool_calls", "model"}`; `GET /assistant/status` reports availability). Enable it with
`pip install -r backend/requirements-assistant.txt` and `ANTHROPIC_API_KEY` in `backend/.env` (already done by
`docker compose up --build`, which mounts the module). Requires the `assistant:ask` permission (admins have it).
Without the dependency or key the endpoint answers 501/503 and the rest of the API is unaffected. The question and
the aggregate tool results are sent to the model provider — fine for the public MIMIC demo, a privacy decision
before using real hospital data.

### AI module: LangChain + LangGraph + public-data digital twin

`ai_module/` is a standalone module (it imports nothing from `backend/`) with:

- a digital twin of ward/ICU bed capacity built end-to-end from the public MIMIC-IV demo
  (checksum-verified ingestion → validation → cleaning → state → Monte Carlo surge
  simulation → ML prediction of ICU need and length of stay);
- LangChain chains and tools with a provider-agnostic LLM;
- a LangGraph `StateGraph` that orchestrates validation → twin context → analysis →
  numerically grounded response;
- a component registry ready for a future Langflow integration.

```bash
pip install -r ai_module/requirements.txt           # from the repo root (the backend venv already has these)
python -m ai_module.digital_twin.prepare            # download + build the twin state
python -m ai_module.digital_twin.train              # versioned model artifact (the API never trains)
python -m ai_module "¿Qué pasa si llegan 60 víctimas en 4 horas?"
```

The backend exposes it as **`POST /api/v1/ai/invoke`**, `GET /api/v1/ai/status`,
**`GET /api/v1/digital-twin/state`**, **`POST /api/v1/digital-twin/simulate`** and
`POST /api/v1/digital-twin/predict`. It is optional: without the module the endpoints
answer 501, and without prepared data they answer 503. Set `AI_LLM_PROVIDER=none` to run
without an LLM. Full documentation: [`ai_module/README.md`](ai_module/README.md); dataset
card: [`ai_module/data/DATASET.md`](ai_module/data/DATASET.md); Langflow plan:
[`docs/langflow-integration.md`](docs/langflow-integration.md).

### ML model comparison

The **ML models** page (`/ml`) compares normal and hybrid classifiers using
patient-grouped cross-validation and a configurable selection score. See the
[protocol and measured results](docs/ml-model-comparison.md) and the
[Langflow setup guide](langflow_integration/README.md). Set
`ML_ANALYSIS_MODE=direct` in `backend/.env` to use the in-process engine, or
configure the Langflow server and import the provided flow for orchestration.

Run all three Python suites from the repository root with:

```powershell
./backend/.venv/Scripts/python.exe -m pytest -q
```

### Frontend development

```bash
cd frontend
npm install
npm run dev
```

## Project Layout

```
backend/app/
  api/v1/            REST + WebSocket routers
  authentication/    JWT + RBAC
  database/          SQLAlchemy engine/session, audit mixin
  models/            ORM models (users, hospitals, patients, simulations, reports, ...)
  schemas/           Pydantic DTOs
  services/          Orchestration: simulation runner, event bus, reports, hospital CRUD
  simulation/
    abm/             Mesa agents (Patient, Clinician, Nurse, Bed, OR, Ambulance) + model
    system_dynamics/ Stock/flow SD model
    hybrid_engine.py Two-way ABM <-> SD coupler
  artificial_intelligence/
    demand_forecasting.py    LSTM (primary) + XGBoost (baseline)
    saturation_classifier.py RandomForest (primary) + SVM (ablation)
    resource_optimizer.py    GA (DEAP) + tabular Q-learning + MILP validation
    training/                 Synthetic-trace training pipeline
  digital_twin/      Live twin state manager + ORM-to-twin serializer
  reports/           DOCX / PDF / XLSX generators
  datasets/          ETL for MIMIC-IV / Kaggle / gov open data + public dataset catalog/downloader

ai_module/            Standalone LangChain + LangGraph module over a public-data digital twin (see ai_module/README.md)
  digital_twin/      Dataset pipeline, DigitalTwinService, Monte Carlo simulator, train/evaluate CLIs
  chains/ agents/    LangChain chains, tools, optional agent
  graphs/            LangGraph StateGraph (typed state, nodes, conditional routes)
  components/        Framework-agnostic component registry (Langflow seam)

frontend/src/
  components/DigitalTwin/  3D scene (React Three Fiber): rooms, beds, status colors
  components/Dashboard/    Executive KPIs, alerts, saturation chart
  components/Simulator/    Scenario builder + run control
  pages/                   Route-level composition
  store/                   Zustand store for the live WS-driven twin state
```

## Validated in This Delivery

- 10/10 backend unit/smoke tests passing (`pytest -q` — see `docs/TEST_PLAN.md`).
- Full end-to-end flow run against a real containerized PostgreSQL 16: user
  registration → hospital → emergency event → simulation (hybrid ABM+SD
  engine, 60 ticks) → results retrieval → PDF report generation (verified
  non-empty output file).
- Synthetic-trace AI training pipeline run end-to-end (hybrid engine →
  sliding-window features → trained XGBoost artifact saved to disk).
- Mesa 3.x API migration verified (`mesa.time.RandomActivation` was removed
  upstream between the 2.x and 3.x lines; the ABM scheduler here uses an
  explicit shuffled activation list instead).

## Roadmap

See `docs/ARCHITECTURE.md` section 11 for the phased plan: real dataset ETL +
trained model artifacts (Phase 1), GA/RL/MILP assignment A/B evaluation
(Phase 2), multi-hospital regional twin (Phase 3), and clinical validation
methodology (Phase 4).
