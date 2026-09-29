"""MLAnalysisService: one method per step of the model-comparison analysis.

The Langflow custom components call these methods one per node (passing the JSON reports
between nodes). ``run()`` chains the same methods in-process (CLI, tests, backend direct
mode). Both paths therefore execute identical code and produce the same output contract."""
from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from ai_module.config.settings import AISettings, get_settings
from ai_module.ml.comparator import ModelComparator
from ai_module.ml.config import MLSettings, get_ml_settings
from ai_module.ml.cross_reference import cross_reference
from ai_module.ml.cross_validation import run_cross_validation
from ai_module.ml.errors import IncompleteResultError, InvalidRequestError, MLAnalysisError
from ai_module.ml.eda import run_eda
from ai_module.ml.evaluation import ModelEvaluationService
from ai_module.ml.explanation import llm_analysis, selection_reason
from ai_module.ml.interpretation import crisp_dm, interpret
from ai_module.ml.log import ml_log
from ai_module.ml.preprocessing import preprocess
from ai_module.ml.run_store import RunStore
from ai_module.ml.schemas import AnalysisRequest
from ai_module.ml.selector import BestModelSelector
from ai_module.ml.statistics import statistical_tests
from ai_module.ml.training import train_family
from ai_module.ml.validation import validate_input

OUTPUT_KEYS = ("status", "models_evaluated", "normal_models", "hybrid_models", "ranking", "best_model",
               "cross_validation", "cross_reference_analysis", "analysis", "selection_reason",
               "eda", "statistical_tests", "hyperparameters", "interpretations", "crisp_dm")


def parse_request(payload: dict[str, Any] | AnalysisRequest | None) -> AnalysisRequest:
    if isinstance(payload, AnalysisRequest):
        return payload
    try:
        return AnalysisRequest.model_validate(payload or {})
    except ValidationError as exc:
        details = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors(include_url=False, include_input=False))
        raise InvalidRequestError(f"Invalid analysis request: {details}") from None


class MLAnalysisService:
    def __init__(self, ai_settings: AISettings | None = None, ml_settings: MLSettings | None = None) -> None:
        self.ai_settings = ai_settings or get_settings()
        self.ml = ml_settings or get_ml_settings()
        self.store = RunStore(self.ml.runs_dir, self.ml.keep_runs)
        self.best_dir = self.ml.runs_dir.parent / "ml_best"

    # ---------------------------------------------------------------- steps (one per Langflow node)

    def validate(self, payload: dict[str, Any] | AnalysisRequest | None) -> dict[str, Any]:
        return validate_input(parse_request(payload), self.ai_settings, self.ml, self.store)

    def eda(self, validation: dict[str, Any]) -> dict[str, Any]:
        return run_eda(validation["run_id"], self.store, self.ml.alpha)

    def preprocess(self, validation: dict[str, Any]) -> dict[str, Any]:
        return preprocess(validation["run_id"], self.store)

    def train_normal(self, preprocessing: dict[str, Any]) -> dict[str, Any]:
        return train_family(preprocessing["run_id"], "normal", self.store, self.ml)

    def train_hybrid(self, preprocessing: dict[str, Any]) -> dict[str, Any]:
        return train_family(preprocessing["run_id"], "hybrid", self.store, self.ml)

    def cross_validate(self, normal: dict[str, Any], hybrid: dict[str, Any]) -> dict[str, Any]:
        _same_run(normal, hybrid)
        return run_cross_validation(normal["run_id"], [normal, hybrid], self.store, self.ml)

    def evaluate(self, normal: dict[str, Any], hybrid: dict[str, Any], cv: dict[str, Any]) -> dict[str, Any]:
        _same_run(normal, hybrid, cv)
        return ModelEvaluationService().evaluate(normal["run_id"], [normal, hybrid], cv)

    def cross_reference(self, evaluation: dict[str, Any], cv: dict[str, Any]) -> dict[str, Any]:
        _same_run(evaluation, cv)
        return cross_reference(evaluation["run_id"], evaluation, cv, self.ml)

    def compare(self, evaluation: dict[str, Any], cross_ref: dict[str, Any]) -> dict[str, Any]:
        _same_run(evaluation, cross_ref)
        cfg = self.store.load_json(evaluation["run_id"], "config")
        return ModelComparator(self.ml).compare(evaluation["run_id"], evaluation, cfg["imbalanced"])

    def statistical_tests(self, evaluation: dict[str, Any], cv: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
        _same_run(evaluation, cv, comparison)
        cfg = self.store.load_json(evaluation["run_id"], "config")
        return statistical_tests(evaluation["run_id"], evaluation, cv, comparison, self.store, self.ml.alpha,
                                 self.ml.bootstrap_resamples, cfg["random_state"])

    def select_best(self, comparison: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
        _same_run(comparison, evaluation)
        return BestModelSelector(self.store, self.best_dir).select(comparison["run_id"], comparison, evaluation)

    def explain(self, selection: dict[str, Any], comparison: dict[str, Any], cross_ref: dict[str, Any],
                statistics: dict[str, Any] | None = None, language: str = "es", use_llm: bool | None = None,
                llm: Any = None) -> dict[str, Any]:
        _same_run(selection, comparison, cross_ref, *([statistics] if statistics else []))
        run_id = selection["run_id"]
        stats_summary = None
        if statistics:
            stats_summary = {"friedman": statistics["friedman"], "summary": statistics["summary"], "alpha": statistics["alpha"]}
        cfg = {**self.store.load_json(run_id, "config"), "run_id": run_id, "statistics_summary": stats_summary}
        best = selection["best_model"]
        reason = selection_reason(best, comparison, cross_ref, cfg, language, statistics)
        want_llm = cfg["explain_with_llm"] if use_llm is None else use_llm
        if want_llm:
            text, problem = llm_analysis(best, comparison, cross_ref, cfg, self.ai_settings, language, llm)
        else:
            text, problem = None, None
        if problem:
            ml_log("LLM", f"Deterministic explanation used: {problem}", run_id=run_id)
        return {"run_id": run_id, "selection_reason": reason, "analysis": text or reason,
                "analysis_source": "llm" if text else "deterministic", "llm_problem": problem}

    def build_output(self, validation: dict[str, Any], eda: dict[str, Any], preprocessing: dict[str, Any], normal: dict[str, Any],
                     hybrid: dict[str, Any], cv: dict[str, Any], evaluation: dict[str, Any], cross_ref: dict[str, Any],
                     comparison: dict[str, Any], statistics: dict[str, Any], selection: dict[str, Any], explanation: dict[str, Any],
                     duration_s: float | None = None, language: str | None = None) -> dict[str, Any]:
        """The flow's final JSON (contract shared by Langflow, backend and frontend)."""
        _same_run(validation, eda, preprocessing, normal, hybrid, cv, evaluation, cross_ref, comparison, statistics, selection, explanation)
        language = language or validation.get("language", "es")
        rows = {r["key"]: r for r in evaluation["models"]}
        scores = {e["key"]: e["final_score"] for e in comparison["ranking"]}

        def summary(report: dict[str, Any]) -> list[dict[str, Any]]:
            keys = ("key", "name", "family", "status", "error", "description", "components", "training_time_s",
                    "inference_time_s", "inference_ms_per_sample", "params", "tuning")
            return [{**{k: m[k] for k in keys}, "train_metrics": m["train_metrics"], "test_metrics": m["test_metrics"],
                     "final_score": scores.get(m["key"]), "evaluation": rows.get(m["key"])} for m in report["models"]]

        trained = [m for report in (normal, hybrid) for m in report["models"] if m["status"] == "ok"]
        hyperparameters = {}
        for m in trained:
            cv_model = cv["models"].get(m["key"], {})
            hyperparameters[m["name"]] = {**m["tuning"], "fold_choices": [f["best_params"] for f in cv_model.get("folds", [])],
                                          "stability": cv_model.get("tuning_stability")}
        warnings = list(validation["warnings"])
        if explanation.get("llm_problem"):
            warnings.append(explanation["llm_problem"])
        output = {
            "status": "success",
            "run_id": validation["run_id"],
            "language": language,
            "models_evaluated": comparison["models_ranked"],
            "models_attempted": comparison["models_evaluated"],
            "normal_models": summary(normal),
            "hybrid_models": summary(hybrid),
            "comparison_table": evaluation["models"],
            "ranking": comparison["ranking"],
            "best_model": selection["best_model"],
            "best_model_metadata": selection["metadata"],
            "cross_validation": cv,
            "cross_reference_analysis": cross_ref,
            "statistical_tests": statistics,
            "hyperparameters": hyperparameters,
            "curves": {m["name"]: m["curves"] for m in trained},
            "eda": eda,
            "score_config": comparison["score_config"],
            "analysis": explanation["analysis"],
            "analysis_source": explanation["analysis_source"],
            "selection_reason": explanation["selection_reason"],
            "dataset": {k: validation[k] for k in ("dataset", "dataset_description", "source", "target", "problem_type", "rows",
                                                    "class_counts", "minority_share", "imbalanced", "primary_metric", "groups")},
            "preprocessing": preprocessing,
            "warnings": warnings,
            "errors": evaluation["failed_models"],
            "duration_s": duration_s,
        }
        output["interpretations"] = interpret(output, language)
        output["crisp_dm"] = crisp_dm(language)
        missing = [k for k in OUTPUT_KEYS if output.get(k) in (None, "", [])]
        if missing:
            raise IncompleteResultError(f"Analysis output is incomplete: {missing}")
        self.store.save_json(validation["run_id"], "output", output)
        return output

    # ---------------------------------------------------------------- whole pipeline in-process

    def run(self, payload: dict[str, Any] | AnalysisRequest | None = None, language: str = "es", llm: Any = None) -> dict[str, Any]:
        start = time.perf_counter()
        ml_log("PIPELINE", "Analysis started")
        validation = self.validate(payload)
        eda = self.eda(validation)
        preprocessing = self.preprocess(validation)
        normal = self.train_normal(preprocessing)
        hybrid = self.train_hybrid(preprocessing)
        cv = self.cross_validate(normal, hybrid)
        evaluation = self.evaluate(normal, hybrid, cv)
        cross_ref = self.cross_reference(evaluation, cv)
        comparison = self.compare(evaluation, cross_ref)
        statistics = self.statistical_tests(evaluation, cv, comparison)
        selection = self.select_best(comparison, evaluation)
        explanation = self.explain(selection, comparison, cross_ref, statistics, language, llm=llm)
        output = self.build_output(validation, eda, preprocessing, normal, hybrid, cv, evaluation, cross_ref, comparison,
                                   statistics, selection, explanation, round(time.perf_counter() - start, 2), language)
        ml_log("PIPELINE", f"Analysis completed: {output['best_model']['name']} selected", run_id=output["run_id"])
        return output


def _same_run(*reports: dict[str, Any]) -> None:
    ids = {r.get("run_id") for r in reports}
    if len(ids) != 1 or None in ids:
        raise IncompleteResultError(f"Step inputs belong to different or missing runs: {sorted(str(i) for i in ids)}")


def error_output(exc: Exception) -> dict[str, Any]:
    """Uniform failure payload (same top-level keys as a success)."""
    code = exc.code if isinstance(exc, MLAnalysisError) else "internal_error"
    message = str(exc) if isinstance(exc, MLAnalysisError) else f"Unexpected error ({type(exc).__name__})"
    return {"status": "error", "error": {"code": code, "message": message}, **{k: None for k in OUTPUT_KEYS if k != "status"}}
