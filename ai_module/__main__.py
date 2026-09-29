"""CLI: ``python -m ai_module "message" [--parameters JSON] [--components]``

Runs one request through the LangGraph workflow and prints the JSON response."""
from __future__ import annotations

import argparse
import json
import sys

from ai_module.config.logging import configure_logging
from ai_module.config.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai_module", description=__doc__)
    parser.add_argument("message", nargs="?", help="Question for the digital twin.")
    parser.add_argument("--parameters", default="{}", help='JSON, e.g. \'{"scenario": {"surge_patients": 60}}\'')
    parser.add_argument("--components", action="store_true", help="Print the component registry (Langflow seam) and exit.")
    parser.add_argument("--text", action="store_true", help="Print only the response text.")
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    from ai_module.services.ai_service import AIModule

    if args.components:
        from ai_module.components import describe_components

        print(json.dumps([{k: c[k] for k in ("name", "kind", "description")} for c in describe_components()], indent=2))
        return 0
    if not args.message:
        parser.error("message is required")
    try:
        parameters = json.loads(args.parameters)
    except json.JSONDecodeError as exc:
        parser.error(f"--parameters is not valid JSON: {exc}")
    result = AIModule(settings).invoke(args.message, parameters)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    print(result.response if args.text else result.model_dump_json(indent=2))
    return 0 if result.metadata["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
