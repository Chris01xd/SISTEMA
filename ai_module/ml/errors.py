"""Explicit errors of the model-comparison analysis (messages are safe to show to clients)."""
from __future__ import annotations


class MLAnalysisError(RuntimeError):
    """Base class. ``code`` is a stable identifier for API clients and Langflow outputs."""

    code = "analysis_error"


class InvalidRequestError(MLAnalysisError):
    code = "invalid_request"


class EmptyDatasetError(MLAnalysisError):
    code = "empty_dataset"


class TargetNotFoundError(MLAnalysisError):
    code = "target_not_found"


class InvalidColumnsError(MLAnalysisError):
    code = "invalid_columns"


class InsufficientDataError(MLAnalysisError):
    code = "insufficient_data"


class DatasetUnavailableError(MLAnalysisError):
    code = "dataset_unavailable"


class NotEnoughModelsError(MLAnalysisError):
    code = "not_enough_models"


class RunNotFoundError(MLAnalysisError):
    code = "run_not_found"


class IncompleteResultError(MLAnalysisError):
    code = "incomplete_result"
