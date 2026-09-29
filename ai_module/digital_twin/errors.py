"""Explicit error types of the digital twin. Messages are safe to show to API clients
(no paths to secrets, no stack traces) and tell the operator how to fix the problem."""
from __future__ import annotations


class DigitalTwinError(RuntimeError):
    """Base class for every digital-twin failure."""


class DatasetDownloadError(DigitalTwinError):
    pass


class DataValidationError(DigitalTwinError):
    pass


class TwinDataNotPreparedError(DigitalTwinError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "The digital twin data has not been prepared. Run `python -m ai_module.digital_twin.prepare`"
            + (f" ({detail})" if detail else "")
        )


class ModelNotTrainedError(DigitalTwinError):
    def __init__(self, detail: str = "") -> None:
        super().__init__(
            "No trained prediction model is available. Run `python -m ai_module.digital_twin.train`"
            + (f" ({detail})" if detail else "")
        )


class ModelArtifactError(DigitalTwinError):
    pass
