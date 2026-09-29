"""Model-comparison engine: 5 normal + 2 hybrid classifiers, real cross-validation,
cross-referenced results, configurable final_score, ranking, automatic best-model
selection, persistence and explanation. Framework-agnostic: the Langflow custom
components in ``langflow_integration/components`` call ``MLAnalysisService`` step by step."""
from ai_module.ml.service import MLAnalysisService, error_output, parse_request

__all__ = ["MLAnalysisService", "error_output", "parse_request"]
