"""Report generators should produce non-empty files with the correct extension."""
from pathlib import Path

from app.reports.docx_generator import generate_docx_report
from app.reports.excel_generator import generate_excel_report
from app.reports.pdf_generator import generate_pdf_report

SAMPLE_METRICS = {"Total patients seen": 42, "Peak saturation index": 0.87}
SAMPLE_TIMESERIES = [{"tick": i, "saturation_index": 0.1 * i} for i in range(5)]


def test_docx_report_generated(tmp_path: Path):
    out = tmp_path / "report.docx"
    result = generate_docx_report(out, "Test Hospital", "Test Event", "sim-1", SAMPLE_METRICS, SAMPLE_TIMESERIES, ["Do X"])
    assert result.exists() and result.stat().st_size > 0


def test_pdf_report_generated(tmp_path: Path):
    out = tmp_path / "report.pdf"
    result = generate_pdf_report(out, "Test Hospital", "Test Event", "sim-1", SAMPLE_METRICS, SAMPLE_TIMESERIES)
    assert result.exists() and result.stat().st_size > 0


def test_excel_report_generated(tmp_path: Path):
    out = tmp_path / "report.xlsx"
    result = generate_excel_report(out, SAMPLE_METRICS, SAMPLE_TIMESERIES)
    assert result.exists() and result.stat().st_size > 0
