# Starts Langflow with the ML custom components (Windows PowerShell).
#   powershell -ExecutionPolicy Bypass -File langflow_integration\scripts\start_langflow.ps1
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$lf = Join-Path $repo "langflow_integration"

# Langflow resolves a component's imports before running its code, so the repository root
# (which contains the ai_module package) must be on PYTHONPATH when the server starts.
$env:PYTHONPATH = $repo
$env:AI_MODULE_ROOT = $repo
$env:LANGFLOW_COMPONENTS_PATH = Join-Path $lf "components"
$env:LANGFLOW_CONFIG_DIR = Join-Path $lf ".langflow"
$env:LANGFLOW_DO_NOT_TRACK = "true"
$env:LANGFLOW_AUTO_LOGIN = "true"
$env:PYTHONIOENCODING = "utf-8"
if (-not $env:LANGFLOW_PORT) { $env:LANGFLOW_PORT = "7860" }

New-Item -ItemType Directory -Force $env:LANGFLOW_CONFIG_DIR | Out-Null

# Langflow silently moves to the next free port when the requested one is busy (e.g. an old
# instance still running), and the backend would keep talking to the stale server. Fail instead.
$busy = Get-NetTCPConnection -LocalPort ([int]$env:LANGFLOW_PORT) -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Error "Port $($env:LANGFLOW_PORT) is already in use by PID $($busy[0].OwningProcess) (another Langflow still running?). Stop it first: Stop-Process -Id $($busy[0].OwningProcess)"
    exit 1
}
& (Join-Path $lf ".venv\Scripts\langflow.exe") run --host 127.0.0.1 --port $env:LANGFLOW_PORT --no-open-browser
