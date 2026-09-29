"""Feature definition shared by training and inference (single source of truth).

Only information known when the patient is admitted is used. Excluded on purpose:
diagnoses (coded after discharge), discharge location / death time / length of stay
(outcomes -> leakage), and race, insurance, language, marital status (sensitive
attributes with no operational meaning for bed planning)."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = (
    "age", "admit_hour_sin", "admit_hour_cos", "ed_acuity_esi",
    "triage_temperature_f", "triage_heartrate", "triage_resprate", "triage_o2sat", "triage_sbp", "triage_dbp",
)
CATEGORICAL_FEATURES = ("sex", "admission_type", "admission_location", "arrival_transport")
BOOLEAN_FEATURES = ("arrived_via_ed",)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

TARGET_ICU = "icu_flag"
TARGET_LOS = "los_hours"
GROUP_COLUMN = "subject_id"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Maps processed admissions (or ``PatientProfile`` rows) to the model's feature frame."""
    angle = 2 * np.pi * df["admit_hour"].astype("float64") / 24.0
    out = pd.DataFrame(index=df.index)
    for column in NUMERIC_FEATURES:
        if column == "admit_hour_sin":
            out[column] = np.sin(angle)
        elif column == "admit_hour_cos":
            out[column] = np.cos(angle)
        else:
            out[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    for column in CATEGORICAL_FEATURES:
        out[column] = df[column].astype(object).where(df[column].notna(), "UNKNOWN").astype(str)
    for column in BOOLEAN_FEATURES:
        out[column] = df[column].astype(bool).astype("float64")
    return out[list(FEATURE_COLUMNS)]
