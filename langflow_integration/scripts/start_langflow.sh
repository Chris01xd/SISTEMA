#!/usr/bin/env bash
# Starts Langflow with the ML custom components (Linux/macOS/Git Bash).
#   bash langflow_integration/scripts/start_langflow.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LF="$REPO/langflow_integration"
# Windows (Git Bash) needs native paths for Python.
if command -v cygpath >/dev/null 2>&1; then REPO_NATIVE="$(cygpath -w "$REPO")"; LF_NATIVE="$(cygpath -w "$LF")"; else REPO_NATIVE="$REPO"; LF_NATIVE="$LF"; fi

# Langflow resolves a component's imports before running its code, so the repository root
# (which contains the ai_module package) must be on PYTHONPATH when the server starts.
export PYTHONPATH="$REPO_NATIVE" AI_MODULE_ROOT="$REPO_NATIVE"
export LANGFLOW_COMPONENTS_PATH="$LF_NATIVE/components" LANGFLOW_CONFIG_DIR="$LF_NATIVE/.langflow"
export LANGFLOW_DO_NOT_TRACK=true LANGFLOW_AUTO_LOGIN=true PYTHONIOENCODING=utf-8
mkdir -p "$LF/.langflow"
# Langflow silently moves to the next free port when the requested one is busy (e.g. an old
# instance still running), and the backend would keep talking to the stale server. Fail instead.
PORT="${LANGFLOW_PORT:-7860}"
if (echo > "/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "ERROR: port $PORT is already in use (another Langflow still running?). Stop it first." >&2
  exit 1
fi
BIN="$LF/.venv/bin/langflow"; [ -x "$BIN" ] || BIN="$LF/.venv/Scripts/langflow.exe"
exec "$BIN" run --host "${LANGFLOW_HOST:-127.0.0.1}" --port "${LANGFLOW_PORT:-7860}" --no-open-browser
