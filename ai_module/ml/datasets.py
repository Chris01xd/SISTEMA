"""Registry of the datasets the comparison can run on.

Only registered datasets are accepted (never arbitrary file paths from a request). Both
entries use the processed public MIMIC-IV demo table produced by
``python -m ai_module.digital_twin.prepare``; features are the admission-time features
shared with the digital twin (``digital_twin/features.py``), so no outcome leaks in."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ai_module.config.settings import AISettings
from ai_module.digital_twin.features import BOOLEAN_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features
from ai_module.digital_twin.store import TwinDataStore


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    description: str
    default_target: str
    allowed_targets: tuple[str, ...]
    group_column: str | None  # rows sharing a group never fall in two partitions
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    boolean: tuple[str, ...]
    source: str
    loader: Callable[[AISettings], pd.DataFrame]

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric + self.categorical + self.boolean


def _mimic_frame(settings: AISettings) -> pd.DataFrame:
    """Processed admissions + engineered admission-time features (stateless row-wise
    transforms such as hour -> sin/cos, safe to apply before splitting)."""
    raw = TwinDataStore(settings.processed_dir).load_processed()
    features = build_features(raw)
    targets = raw[["subject_id", "icu_flag", "died_in_hospital"]]
    return pd.concat([features, targets], axis=1)


_MIMIC_SOURCE = "MIMIC-IV Clinical Database Demo v2.2 + MIMIC-IV-ED Demo v2.2 (PhysioNet, ODbL)"

DATASETS: dict[str, DatasetSpec] = {
    "mimic_icu": DatasetSpec(
        key="mimic_icu",
        description="ICU need at admission (any ICU stay during the hospital admission).",
        default_target="icu_flag", allowed_targets=("icu_flag", "died_in_hospital"), group_column="subject_id",
        numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES, boolean=BOOLEAN_FEATURES,
        source=_MIMIC_SOURCE, loader=_mimic_frame,
    ),
    "mimic_mortality": DatasetSpec(
        key="mimic_mortality",
        description="In-hospital mortality from admission-time information (strongly imbalanced).",
        default_target="died_in_hospital", allowed_targets=("died_in_hospital", "icu_flag"), group_column="subject_id",
        numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES, boolean=BOOLEAN_FEATURES,
        source=_MIMIC_SOURCE, loader=_mimic_frame,
    ),
}


def get_dataset(key: str) -> DatasetSpec:
    try:
        return DATASETS[key]
    except KeyError:
        from ai_module.ml.errors import InvalidRequestError

        raise InvalidRequestError(f"Unknown dataset '{key}'. Available: {sorted(DATASETS)}") from None
