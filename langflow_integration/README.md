# ML comparison in Langflow

The fourteen custom components in `components/ml_pipeline/` (including EDA and
inferential statistical tests) delegate to
`ai_module.ml`. The exported flow connects Chat Input, these components and
Chat Output, with endpoint name `ml-model-comparison`.

## Windows setup (PowerShell, repository root)

Use the dedicated Python 3.13 environment, separate from the backend:

```powershell
py -3.13 -m venv langflow_integration/.venv
& ./langflow_integration/.venv/Scripts/python.exe -m pip install -r langflow_integration/requirements.txt
$env:PYTHONPATH = (Get-Location).Path
$env:AI_MODULE_ROOT = (Get-Location).Path
$env:LANGFLOW_COMPONENTS_PATH = Join-Path (Get-Location) 'langflow_integration/components'
& ./langflow_integration/.venv/Scripts/python.exe langflow_integration/scripts/generate_flow.py
& ./langflow_integration/.venv/Scripts/python.exe -m langflow run --host 127.0.0.1 --port 7860
```

Prepare the public dataset first if `ai_module/data/processed/` is absent:
`backend/.venv/Scripts/python.exe -m ai_module.digital_twin.prepare`.

Langflow 1.12.3 requires LangChain 1.3.x, while the standalone AI environment
pins 1.4.2. Keep their environments separate and use the integration's own
requirements file; do not install the standalone requirements into Langflow.
The custom components use top-level `lfx` imports because Langflow's source
loader does not resolve imports inside a `try/except` compatibility block.

Shortcut: `scripts/start_langflow.ps1` (or `.sh`) sets all these variables and
starts the server; `scripts/import_flow.py --create-key` imports or updates the
flow in place and stores a local API key in `.langflow/api_key.txt`.

**Only one Langflow per port.** If port 7860 is busy (for example, an old instance
whose `langflow.exe` survived after its terminal was closed), Langflow silently
starts on 7861, while the backend keeps calling the stale server on 7860. That
stale server answers `POST /api/v1/run/...` with HTTP 500. The start scripts now
refuse to start when the port is busy. Stop the old process first
(`Get-NetTCPConnection -LocalPort 7860` → `Stop-Process -Id <PID>`), then start
Langflow and re-import the flow.

Import `flows/ml_model_comparison.json` into Langflow. Generate an API key in
Langflow and set these values in `backend/.env`, then restart the backend:

```dotenv
ML_ANALYSIS_MODE=langflow
LANGFLOW_URL=http://localhost:7860
LANGFLOW_API_KEY=<your local Langflow API key>
LANGFLOW_FLOW_ID=ml-model-comparison
LANGFLOW_TIMEOUT_SECONDS=600
```

If Langflow assigns a different endpoint name, use the imported flow's ID for
`LANGFLOW_FLOW_ID`. For a backend running in Docker and Langflow on Windows,
use `http://host.docker.internal:7860` and configure Langflow's bind address
appropriately.

The Playground accepts a JSON message, for example:

```json
{"dataset":"mimic_icu","cv_folds":5,"explain_with_llm":false,"language":"es"}
```

The final chat message is the analysis JSON consumed by
`POST /api/v1/ml/analysis`. Start with `explain_with_llm=false` to check the ML
pipeline without a provider call. The `/ml` frontend page uses the backend,
so the Langflow API key stays on the backend.

## Without a Langflow server

Set `ML_ANALYSIS_MODE=direct` in `backend/.env` and restart the backend to run
the same ML steps in-process. This does not validate Langflow orchestration.
For a command-line check:

```powershell
& ./backend/.venv/Scripts/python.exe -m ai_module.ml --no-llm
```

## Maintaining the flow

Edit `scripts/generate_components.py` and regenerate the component files with
Python. Then run `scripts/generate_flow.py` with the Langflow environment to
regenerate the importable JSON using Langflow's own serializer. The JSON
embeds the component source; regenerate and reimport after component changes.

To execute the exported graph locally with Langflow's executor, including a
structured-error check and a seven-model run on the prepared public demo:

```powershell
& ./langflow_integration/.venv/Scripts/python.exe langflow_integration/scripts/smoke_flow.py
```

This checks orchestration without starting a server. It creates ML run and
model artifacts, and does not verify server authentication or the browser UI.

Verified on 2026-09-28 with Langflow 1.12.3 / Python 3.13 (earlier 12-component
version): 14 nodes, 31 edges, error propagation passed, and all seven models
completed in 37.23 seconds. The winner was Logistic Regression with final score 0.5627.

Current version (EDA, grid-search tuning, nested CV, statistical tests): 16 nodes,
38 edges. Verified through the real server with Backend → Langflow API → components →
Gemini explanation → Backend. The run returned HTTP 200 in 234 s, all seven models were
evaluated, and the winner was Logistic Regression with final score 0.572. The PDF and
XLSX reports downloaded successfully, and an invalid target returned a structured 422.
Tests: `PYTHONPATH=<repo> langflow_integration/.venv/Scripts/python -m pytest langflow_integration/tests`.

See [the model protocol and measured results](../docs/ml-model-comparison.md).
