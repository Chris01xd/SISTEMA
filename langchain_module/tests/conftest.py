import pandas as pd
import pytest


@pytest.fixture
def admissions() -> pd.DataFrame:
    """8 admissions with hand-computable statistics."""
    return pd.DataFrame(
        {
            "patient_id": [101, 102, 103, 104, 105, 106, 107, 108],
            "age": [30, 45, 50, 66, 70, 82, 85, 17],
            "sex": ["F", "M", "F", "M", "F", "M", "F", "M"],
            "admission_time": pd.to_datetime(
                ["2150-01-01 08:00", "2150-01-02 08:30", "2150-01-03 22:00", "2150-01-04 22:15",
                 "2150-01-05 09:00", "2150-01-06 09:10", "2150-01-07 23:00", "2150-01-08 07:00"]
            ),
            "discharge_time": pd.NaT,
            "diagnosis": ["Sepsis", "Sepsis", "Angina", "Heart failure", "Heart failure", "Heart failure", None, "Asthma"],
            "severity_level": pd.array([4, 3, pd.NA, 2, 3, 4, pd.NA, 1], dtype="Int64"),
            "los_hours": [24.0, 48.0, 72.0, 96.0, 120.0, 144.0, 168.0, 12.0],
            "icu_flag": [True, True, False, False, True, True, False, False],
            "outcome": ["DISCHARGED", "DISCHARGED", "DISCHARGED", "DECEASED", "DISCHARGED", "DECEASED", "DECEASED", "DISCHARGED"],
        }
    )
