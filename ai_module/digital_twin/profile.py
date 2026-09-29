"""Step 6a: the observed operational profile, computed only from the processed data."""
from __future__ import annotations

import pandas as pd

from ai_module.schemas.twin import DistributionStats, ObservedProfile

PROFILE_NOTES = [
    "Source: MIMIC-IV Demo 2.2 (100 real de-identified patients, Beth Israel Deaconess Medical Center). "
    "Small sample: figures are illustrative, not statistically representative of any hospital.",
    "Dates are shifted per patient by the data provider: durations and hour of day are valid, calendar dates are not.",
    "ESI = Emergency Severity Index recorded at ED triage (1 = most urgent); only admissions that came through the ED have one.",
]


def _stats(series: pd.Series) -> DistributionStats | None:
    values = series.dropna()
    if values.empty:
        return None
    return DistributionStats(
        mean=round(float(values.mean()), 1),
        median=round(float(values.median()), 1),
        p90=round(float(values.quantile(0.9)), 1),
        n=int(len(values)),
    )


def _pct(mask: pd.Series) -> float:
    return round(100.0 * float(mask.mean()), 1) if len(mask) else 0.0


def build_observed_profile(df: pd.DataFrame) -> ObservedProfile:
    """Descriptive statistics of the processed admissions table."""
    ed = df[df["arrived_via_ed"]]
    esi = df["ed_acuity_esi"].dropna().astype(int)
    icu_patients = df[df["icu_flag"]]
    by_hour = df["admit_hour"].value_counts().reindex(range(24), fill_value=0)
    icu_by_esi = df.dropna(subset=["ed_acuity_esi"]).groupby(df["ed_acuity_esi"].dropna().astype(int))["icu_flag"].mean()

    los_stats = _stats(df["los_hours"])
    age_stats = _stats(df["age"])
    assert los_stats is not None and age_stats is not None  # validate_processed guarantees rows
    return ObservedProfile(
        admissions=int(len(df)),
        patients=int(df["subject_id"].nunique()),
        age=age_stats,
        sex_counts={str(k): int(v) for k, v in df["sex"].value_counts().sort_index().items()},
        ed_arrival_pct=_pct(df["arrived_via_ed"]),
        ambulance_arrival_pct_of_ed=_pct(ed["arrival_transport"] == "AMBULANCE"),
        esi_distribution={str(k): int(v) for k, v in esi.value_counts().sort_index().items()},
        icu_rate_pct=_pct(df["icu_flag"]),
        icu_rate_pct_by_esi={str(int(k)): round(100.0 * float(v), 1) for k, v in icu_by_esi.items()},
        in_hospital_mortality_pct=_pct(df["died_in_hospital"]),
        los_hours=los_stats,
        los_hours_icu_patients=_stats(icu_patients["los_hours"]),
        icu_hours_icu_patients=_stats(icu_patients["icu_hours"]),
        arrivals_by_hour_pct=[round(100.0 * float(v) / len(df), 2) for v in by_hour.tolist()],
        notes=PROFILE_NOTES,
    )
