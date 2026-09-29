"""Thin, cached wrapper around ai_module's real CRISP-DM engine.

Every page reads the same cached `output` dict (one MLAnalysisService.run() per
combination of dataset/target/folds/llm/language), so numbers stay identical between
the screen and the exported PDF/Excel reports."""
from __future__ import annotations

from typing import Any

import streamlit as st

from lib.paths import bootstrap

bootstrap()

from ai_module.config.settings import AISettings, get_settings  # noqa: E402
from ai_module.digital_twin.service import DigitalTwinService  # noqa: E402
from ai_module.ml.config import get_ml_settings  # noqa: E402
from ai_module.ml.datasets import DATASETS  # noqa: E402
from ai_module.ml.errors import MLAnalysisError  # noqa: E402
from ai_module.ml.service import MLAnalysisService, error_output  # noqa: E402

DATASET_LABELS = {key: spec.description for key, spec in DATASETS.items()}


@st.cache_resource(show_spinner=False)
def get_ai_settings() -> AISettings:
    return get_settings()


@st.cache_resource(show_spinner=False)
def get_service() -> MLAnalysisService:
    return MLAnalysisService(get_ai_settings(), get_ml_settings())


@st.cache_data(show_spinner="Ejecutando el pipeline CRISP-DM (EDA, entrenamiento, validación cruzada, estadística inferencial)...")
def run_analysis(dataset: str, cv_folds: int | None, explain_with_llm: bool, language: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"dataset": dataset, "explain_with_llm": explain_with_llm}
    if cv_folds:
        payload["cv_folds"] = cv_folds
    try:
        return get_service().run(payload, language=language)
    except MLAnalysisError as exc:
        return error_output(exc)


def ensure_defaults() -> None:
    st.session_state.setdefault("dataset", "mimic_icu")
    st.session_state.setdefault("cv_folds", 5)
    st.session_state.setdefault("explain_with_llm", True)
    st.session_state.setdefault("language", "es")


def current_output() -> dict[str, Any]:
    ensure_defaults()
    return run_analysis(
        st.session_state["dataset"], st.session_state["cv_folds"],
        st.session_state["explain_with_llm"], st.session_state["language"],
    )


@st.cache_data(show_spinner="Descargando (verificado por checksum) y limpiando el dataset MIMIC-IV...")
def prepare_dataset(offline: bool) -> dict[str, Any]:
    settings = get_ai_settings()
    service = DigitalTwinService(settings)
    raw = service.load_data(download=not offline)
    raw_counts = {"admissions": len(raw.admissions), "patients": len(raw.patients), "icustays": len(raw.icustays),
                  "edstays": len(raw.edstays), "triage": len(raw.triage)}
    service.preprocess_data(raw)
    state = service.build_state()
    return {"raw_counts": raw_counts, "processed_rows": state.provenance.processed_rows,
            "processed_patients": state.provenance.processed_patients, "processed_sha256": state.provenance.processed_sha256}
