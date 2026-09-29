"""CLI: ``python -m ai_module.digital_twin.train [--seed N]``

Trains the admission-outcome models (ICU need + length of stay) on the processed public
data and stores a new versioned artifact with its parameters, metrics, date and dataset
checksum. The API never trains: it only loads the latest artifact."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from ai_module.config.logging import configure_logging
from ai_module.config.settings import get_settings
from ai_module.digital_twin.errors import DigitalTwinError
from ai_module.digital_twin.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_features
from ai_module.digital_twin.modeling import TEST_SIZE, VALIDATION_SIZE_OF_REST, train_models
from ai_module.digital_twin.registry import MODEL_NAME, ModelRegistry, library_versions, new_version
from ai_module.digital_twin.store import TwinDataStore


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="ai_module.digital_twin.train", description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.twin_random_seed)
    args = parser.parse_args(argv)
    configure_logging(settings.log_level, settings.log_json)

    store = TwinDataStore(settings.processed_dir)
    try:
        df = store.load_processed()
        provenance = store.load_provenance()
    except DigitalTwinError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    dataset_sha = store.processed_sha256()

    outcome = train_models(df, args.seed)
    train_val = build_features(pd.concat([outcome.split.train, outcome.split.validation]))
    version = new_version(dataset_sha)
    metadata = {
        "name": MODEL_NAME,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": provenance.dataset,
        "dataset_sha256": dataset_sha,
        "seed": args.seed,
        "features": list(FEATURE_COLUMNS),
        "categories": {c: sorted(train_val[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
        "targets": {"icu_classifier": "icu_flag (any ICU stay during the admission)", "los_regressor": "hospital length of stay, hours"},
        "split": {
            "method": "GroupShuffleSplit by subject_id (no patient in two partitions)",
            "test_size": TEST_SIZE,
            "validation_size_of_rest": VALIDATION_SIZE_OF_REST,
            "sizes": outcome.split.sizes(),
            "train_validation_admissions": int(len(outcome.split.train) + len(outcome.split.validation)),
            "test_admissions": int(len(outcome.split.test)),
            "test_subject_ids": sorted(int(s) for s in outcome.split.test["subject_id"].unique()),
        },
        "model_selection": {"criterion": {"icu_classifier": "validation ROC-AUC", "los_regressor": "validation MAE (hours)"},
                            "candidates": outcome.selection, "chosen": outcome.chosen},
        "params": {
            "icu_classifier": {k: repr(v) for k, v in outcome.icu_model.named_steps["clf"].get_params().items()},
            "los_regressor": {k: repr(v) for k, v in outcome.los_model.named_steps["reg"].regressor.get_params().items()},
        },
        "baselines": outcome.baselines,
        "test_metrics": outcome.test_metrics,
        "library_versions": library_versions(),
    }
    path = ModelRegistry(settings.models_dir).save(version, {"icu_classifier": outcome.icu_model, "los_regressor": outcome.los_model}, metadata)
    print(json.dumps({"version": version, "path": str(path), "chosen": outcome.chosen, "test_metrics": outcome.test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
