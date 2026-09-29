"""CLI: ``python -m ai_module.digital_twin.prepare [--offline]``

Runs the reproducible data pipeline: download (checksum-verified) -> validate -> clean ->
transform -> persist -> build and persist the digital state. Idempotent."""
from __future__ import annotations

import argparse
import json
import sys

from ai_module.config.logging import configure_logging
from ai_module.config.settings import get_settings
from ai_module.digital_twin.errors import DigitalTwinError
from ai_module.digital_twin.service import DigitalTwinService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai_module.digital_twin.prepare", description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Do not download; use files already under data/raw.")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    service = DigitalTwinService(settings)
    try:
        raw = service.load_data(download=not args.offline)
        service.preprocess_data(raw)
        state = service.build_state()
    except DigitalTwinError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    summary = {
        "processed_rows": state.provenance.processed_rows,
        "processed_patients": state.provenance.processed_patients,
        "processed_sha256": state.provenance.processed_sha256,
        "capacity": state.capacity.model_dump(),
        "baseline_status": state.status_color,
        "outputs": [str(service.store.admissions_path), str(service.store.state_path)],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
