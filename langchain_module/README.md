# hospital_assistant — standalone LangChain module

A LangChain (1.x) agent that answers natural-language questions about the hospital
admissions dataset (the MIMIC-IV demo ingested by the backend). It is **independent** of
the backend: it imports nothing from `backend/app`, has its own `requirements.txt` and
virtualenv, and only shares the *file format* (the canonical Parquet the backend's ETL writes).

## How it works

```
question ──► LLM (LangChain agent) ──► tools ──► analytics.py (pandas) ──► aggregates
```

The model never sees the table. It can only call three read-only tools:

| Tool | Purpose |
|---|---|
| `get_dataset_overview` | headline stats + what can be filtered/grouped |
| `aggregate_admissions` | one metric (count, mean age, LOS, ICU rate, mortality…), optional grouping and filters |
| `list_top_diagnoses` | most frequent primary diagnoses |

Guardrails (enforced in code, `analytics.py` / `tools.py`, not just in the prompt): whitelisted
columns, operators, metrics and groupings; nothing is evaluated as code; **only aggregates are
returned, never individual patient rows**; bad requests come back to the model as errors it can
correct; the tool loop is capped. The system prompt additionally requires every number to come
from a tool result, sample size to be reported, and no clinical advice.

## Setup

```bash
cd langchain_module
python -m venv .venv
.venv/Scripts/activate            # Windows; `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # or use another provider, see below
```

Import the dataset first (backend `POST /api/v1/datasets/ingest {"key": "mimic-iv-demo"}`, or
the **Public data** page). By default the module reads
`backend/data/public/canonical/mimic-iv-demo-2.2.parquet`; override with `HOSPITAL_DATASET_PATH`
or `--dataset`.

## Use

```bash
python -m hospital_assistant "¿Cuál es la mortalidad por grupo de edad?"
python -m hospital_assistant -v          # interactive chat; -v shows the tool calls made
```

```python
from pathlib import Path
from hospital_assistant import ask, build_agent, load_dataset

agent = build_agent(load_dataset(Path("mimic.parquet")))
answer, history = ask(agent, "How many ICU admissions had sepsis?")
answer, history = ask(agent, "And their mortality?", history)   # follow-up keeps context
```

## Backend integration

The backend exposes this module as `POST /api/v1/assistant/ask` (see the root README and
`backend/app/services/assistant_service.py`). The module stays independent: the backend imports it lazily
from `langchain_module/` and does not require it to run. Settings there: `ASSISTANT_MODEL`,
`ASSISTANT_TIMEOUT_SECONDS`, `ANTHROPIC_API_KEY`.

## Configuration

| Variable | Default | |
|---|---|---|
| `LLM_MODEL` | `claude-sonnet-5` | bare name = Anthropic; `provider:model` (e.g. `openai:gpt-4o`) uses another provider if its `langchain-<provider>` package is installed |
| `LLM_TEMPERATURE` | `0` | |
| `HOSPITAL_DATASET_PATH` | backend's ingested MIMIC-IV demo | |

## Tests

```bash
python -m pytest -q
```

They run offline: the analytics are checked against hand-computed values, and the agent loop
(tool-argument validation, error recovery, multi-turn history) runs with a scripted fake chat
model. No real LLM call is made by the test suite.
