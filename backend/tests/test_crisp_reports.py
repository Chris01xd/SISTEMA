"""Pure-logic tests for the CRISP-DM module — no live DB, same pattern as
test_reports.py: exercise real functions with in-memory/tmp_path data."""
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

from app.reports.crisp_report_generator import generate_crisp_excel_report, generate_crisp_pdf_report
from app.services.crisp.monitoring_service import _psi
from app.services.crisp.preparation_service import _validate_temporal_split

SAMPLE_PROJECT = {"name": "Piloto Flota Norte", "responsible": "Ana Ríos", "overall_status": "IN_PROGRESS", "progress_percent": 50.0}
SAMPLE_PHASES = [
    {"phase_number": 1, "name": "Comprensión del negocio", "status": "APPROVED", "approved_at": "2026-01-10"},
    {"phase_number": 2, "name": "Comprensión de los datos", "status": "IN_PROGRESS", "approved_at": None},
]
SAMPLE_KPIS = [{"indicator": "MTBF", "baseline_value": 120.0, "target_value": 180.0, "actual_value": None}]
SAMPLE_EXPERIMENTS = [{"algorithm": "RANDOM_FOREST", "objective": "FAILURE_CLASSIFICATION", "status": "COMPLETED", "metrics": {"f1": 0.81}}]
SAMPLE_DEPLOYMENTS = [{"version": "v1", "service_status": "ACTIVE", "api_endpoint": "/api/v1/crisp/deployments/x/predict", "next_retraining_date": None}]


def test_crisp_pdf_report_generated(tmp_path: Path):
    out = tmp_path / "report.pdf"
    result = generate_crisp_pdf_report(out, SAMPLE_PROJECT, SAMPLE_PHASES, SAMPLE_KPIS, SAMPLE_EXPERIMENTS, SAMPLE_DEPLOYMENTS)
    assert result.exists() and result.stat().st_size > 0


def test_crisp_excel_report_generated(tmp_path: Path):
    out = tmp_path / "report.xlsx"
    result = generate_crisp_excel_report(out, SAMPLE_PROJECT, SAMPLE_PHASES, SAMPLE_KPIS, SAMPLE_EXPERIMENTS, SAMPLE_DEPLOYMENTS)
    assert result.exists() and result.stat().st_size > 0


def test_crisp_report_handles_empty_optional_sections(tmp_path: Path):
    out = tmp_path / "report_minimal.pdf"
    result = generate_crisp_pdf_report(out, SAMPLE_PROJECT, SAMPLE_PHASES, [], [], [])
    assert result.exists() and result.stat().st_size > 0


def test_psi_is_zero_for_identical_distributions():
    baseline = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    assert _psi(baseline, baseline) == pytest.approx(0.0, abs=1e-6)


def test_psi_is_positive_for_shifted_distribution():
    baseline = pd.Series(list(range(100)))
    shifted = pd.Series([x + 500 for x in range(100)])
    assert _psi(baseline, shifted) > 0.2


def test_psi_handles_empty_series_without_crashing():
    assert _psi(pd.Series(dtype=float), pd.Series([1.0, 2.0])) == 0.0


def test_temporal_split_accepts_chronological_ranges():
    _validate_temporal_split(
        {"train": ["2026-01-01", "2026-02-01"], "validation": ["2026-02-01", "2026-02-15"], "test": ["2026-02-15", "2026-03-01"]}
    )  # should not raise


def test_temporal_split_rejects_overlapping_ranges():
    with pytest.raises(HTTPException):
        _validate_temporal_split(
            {"train": ["2026-01-01", "2026-02-15"], "validation": ["2026-02-01", "2026-02-20"], "test": ["2026-02-15", "2026-03-01"]}
        )


def test_temporal_split_ignores_empty_config():
    _validate_temporal_split({})  # should not raise
