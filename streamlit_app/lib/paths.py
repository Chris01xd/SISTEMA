"""Bootstraps sys.path and environment variables so the app can import ai_module,
langchain_module and backend.app.reports without installing them as packages."""
from __future__ import annotations

import os
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent
STREAMLIT_APP_ROOT = LIB_DIR.parent
REPO_ROOT = STREAMLIT_APP_ROOT.parent
AI_MODULE_ENV = REPO_ROOT / "ai_module" / ".env"

_DONE = False


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def bootstrap() -> None:
    """Idempotent; call at the top of every page before importing ai_module/hospital_assistant."""
    global _DONE
    if _DONE:
        return
    for path in (REPO_ROOT, REPO_ROOT / "langchain_module", STREAMLIT_APP_ROOT):
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)
    _load_env_file(AI_MODULE_ENV)
    _DONE = True
