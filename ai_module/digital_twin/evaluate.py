"""CLI: ``python -m ai_module.digital_twin.evaluate [--version V] [--bootstrap N]``

Re-scores a stored model on the exact test patients recorded in its metadata, adds
bootstrap 95% intervals, checks that the result reproduces the metrics recorded at
training time, and writes ``evaluation.json`` next to the model."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from ai_module.config.logging import configure_logging
from ai_module.config.settings import get_settings
from ai_module.digital_twin.errors import DigitalTwinError
from ai_module.digital_twin.modeling import evaluate_models, split_from_test_subjects
from ai_module.digital_twin.registry import ModelRegistry
from ai_module.digital_twin.store import TwinDataStore

_COMPARED = {"icu_classifier": ("roc_auc", "brier"), "los_regressor": ("mae_hours",)}


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="ai_module.digital_twin.evaluate", description=__doc__)
    parser.add_argument("--version", help="Model version (default: latest).")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    configure_logging(settings.log_level, settings.log_json)

    registry = ModelRegistry(settings.models_dir)
    store = TwinDataStore(settings.processed_dir)
    try:
        loaded = registry.load(args.version)
        df = store.load_processed()
    except DigitalTwinError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    meta = loaded.metadata
    warnings: list[str] = []
    if store.processed_sha256() != meta["dataset_sha256"]:
        warnings.append("The processed data differs from the data this model was trained on.")

    test = split_from_test_subjects(df, meta["split"]["test_subject_ids"])
    metrics = evaluate_models(loaded.models["icu_classifier"], loaded.models["los_regressor"], test,
                              meta["baselines"], bootstrap=args.bootstrap, seed=meta["seed"])
    reproduced = all(
        abs(metrics[task][m] - meta["test_metrics"][task][m]) < 1e-6
        for task, names in _COMPARED.items() for m in names if m in meta["test_metrics"][task]
    )
    if not reproduced:
        warnings.append("Test metrics differ from those recorded at training time.")
    report = {
        "version": meta["version"],
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_admissions": int(len(test)),
        "metrics": metrics,
        "reproduces_training_metrics": reproduced,
        "warnings": warnings,
    }
    registry.write_json(meta["version"], "evaluation.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
