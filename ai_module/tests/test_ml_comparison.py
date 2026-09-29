"""Model-comparison engine: catalog, real training + CV, cross-reference, comparator,
automatic best-model selection, persistence, explanation and error handling."""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline

from ai_module.config.settings import AISettings
from ai_module.digital_twin.service import DigitalTwinService
from ai_module.ml import catalog
from ai_module.ml.catalog import CATALOG, BuildContext, FeatureSpec, get_specs
from ai_module.ml.comparator import ModelComparator
from ai_module.ml.config import MLSettings, ScoreWeights
from ai_module.ml.errors import (
    DatasetUnavailableError,
    EmptyDatasetError,
    IncompleteResultError,
    InsufficientDataError,
    InvalidColumnsError,
    InvalidRequestError,
    NotEnoughModelsError,
    RunNotFoundError,
    TargetNotFoundError,
)
from ai_module.ml.service import OUTPUT_KEYS, MLAnalysisService, error_output

from .conftest import make_raw_tables, scripted


def _settings(root: Path) -> tuple[AISettings, MLSettings]:
    ai = AISettings(_env_file=None, data_dir=root / "data", artifacts_dir=root / "artifacts", llm_provider="none",
                    twin_sim_runs=3, twin_sim_horizon_hours=24, twin_sim_warmup_hours=24)
    # Tuning off for the per-test runs (speed); the shared `ml_analysis` fixture runs it for real.
    ml = MLSettings(_env_file=None, runs_dir=root / "artifacts" / "ml_runs", explain_with_llm=False, tuning_enabled=False,
                    bootstrap_resamples=100)
    return ai, ml


def _prepared(root: Path, n_patients: int = 60) -> tuple[AISettings, MLSettings]:
    ai, ml = _settings(root)
    DigitalTwinService(ai).preprocess_data(make_raw_tables(n_patients=n_patients))
    return ai, ml


@pytest.fixture
def analysis(ml_analysis):
    return ml_analysis


# ---------------------------------------------------------------- catalog


def test_catalog_has_at_least_four_normal_and_exactly_two_hybrid_models():
    assert len(get_specs("normal")) >= 4
    assert len(get_specs("hybrid")) == 2


def test_every_model_is_a_leak_free_pipeline_and_hybrids_are_real_ensembles():
    ctx = BuildContext(FeatureSpec(("age",), ("sex",), ("arrived_via_ed",)), random_state=0, imbalanced=False)
    for spec in CATALOG:
        if not spec.available():
            continue
        model = spec.build(ctx)
        assert isinstance(model, Pipeline) and list(model.named_steps) == ["preprocess", "model"]
    voting = get_specs(keys=["voting_ensemble"])[0].build(ctx).named_steps["model"]
    stacking = get_specs(keys=["stacking_ensemble"])[0].build(ctx).named_steps["model"]
    assert isinstance(voting, VotingClassifier) and voting.voting == "soft" and len(voting.estimators) == 3
    assert isinstance(stacking, StackingClassifier) and len(stacking.estimators) == 3 and stacking.final_estimator is not None


# ---------------------------------------------------------------- end to end


def test_output_contract_and_counts(analysis):
    _, out = analysis
    assert out["status"] == "success"
    assert all(out[k] not in (None, "", []) for k in OUTPUT_KEYS)
    expected = len([s for s in CATALOG if s.available()])
    assert out["models_evaluated"] == expected >= 6
    assert len(out["normal_models"]) >= 4 and len(out["hybrid_models"]) == 2
    assert all(m["status"] == "ok" for m in out["normal_models"] + out["hybrid_models"] if m["key"] != "xgboost" or catalog.XGBOOST_AVAILABLE)


def test_every_model_was_really_trained_and_has_all_metrics(analysis):
    _, out = analysis
    for row in out["comparison_table"]:
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "cv_mean", "cv_std", "train_score",
                       "test_score", "overfitting_gap", "training_time_s", "inference_time_s"):
            assert row[metric] is not None, (row["model"], metric)
        assert row["training_time_s"] > 0 and 0 <= row["accuracy"] <= 1
        assert row["overfitting_gap"] == pytest.approx(row["train_score"] - row["test_score"], abs=1e-3)


def test_cross_validation_is_real_k_fold(analysis):
    _, out = analysis
    cv = out["cross_validation"]
    assert cv["folds"] == 5 and "Group" in cv["strategy"]
    for key, result in cv["models"].items():
        assert len(result["folds"]) == 5 and len(result["cv_scores"]) == 5
        assert result["cv_mean"] == pytest.approx(np.mean(result["cv_scores"]), abs=1e-3)
        assert result["cv_std"] == pytest.approx(np.std(result["cv_scores"]), abs=1e-3)
        assert len(set(result["cv_scores"])) > 1  # distinct folds, not a copied number
        assert sum(f["validation_size"] for f in result["folds"]) == out["preprocessing"]["split"]["train_rows"]


def test_hold_out_and_folds_never_share_patients(analysis):
    service, out = analysis
    assert out["preprocessing"]["split"]["group_overlap"] == 0
    from sklearn.model_selection import StratifiedGroupKFold

    from ai_module.ml.preprocessing import load_partitions

    _, _, x_train, y_train, _, _, groups = load_partitions(out["run_id"], service.store)
    for train_idx, val_idx in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42).split(x_train, y_train, groups):
        assert not set(groups.iloc[train_idx]) & set(groups.iloc[val_idx])


def test_ranking_is_ordered_by_final_score_and_includes_hybrids(analysis):
    _, out = analysis
    scores = [e["final_score"] for e in out["ranking"]]
    assert scores == sorted(scores, reverse=True)
    assert [e["rank"] for e in out["ranking"]] == list(range(1, len(scores) + 1))
    assert {e["type"] for e in out["ranking"]} == {"normal", "hybrid"}
    assert sum(e["type"] == "hybrid" for e in out["ranking"]) == 2


def test_final_score_is_recomputed_from_the_real_metrics(analysis):
    service, out = analysis
    rows = {r["key"]: r for r in out["comparison_table"]}
    for entry in out["ranking"]:
        row = rows[entry["key"]]
        w = service.ml.weights_balanced
        expected = (w.roc_auc * row["cv_metrics"]["roc_auc"] + w.f1 * row["cv_metrics"]["f1"] + w.pr_auc * row["cv_metrics"]["pr_auc"]
                    + w.accuracy * row["cv_metrics"]["accuracy"] - service.ml.instability_penalty * row["cv_std"]
                    - service.ml.overfitting_penalty * max(0, row["cv_overfitting_gap"]))
        assert entry["final_score"] == pytest.approx(expected, abs=2e-4)


def test_best_model_is_the_argmax_of_final_score_and_is_persisted(analysis):
    _, out = analysis
    best = out["best_model"]
    top = max(out["ranking"], key=lambda e: e["final_score"])
    assert best["name"] == top["model"] == out["ranking"][0]["model"] and best["final_score"] == top["final_score"]
    model = joblib.load(best["artifact_path"])
    meta = json.loads(Path(best["metadata_path"]).read_text())
    for key in ("model_name", "model_type", "final_score", "metrics", "features", "target", "training_date"):
        assert meta[key] not in (None, "", [])
    assert meta["model_name"] == best["name"]
    x = pd.DataFrame([{c: 0 for c in meta["features"]} | {"sex": "F", "admission_type": "EW EMER.", "admission_location": "EMERGENCY ROOM",
                                                            "arrival_transport": "AMBULANCE"}])
    assert 0 <= model.predict_proba(x[meta["features"]])[0, 1] <= 1
    latest = json.loads((Path(best["artifact_path"]).parents[1] / "latest.json").read_text())
    assert latest["run_id"] == out["run_id"]


def test_cross_reference_crosses_train_cv_and_test(analysis):
    _, out = analysis
    xr = out["cross_reference_analysis"]
    for key, item in xr["models"].items():
        assert item["cv_vs_test_difference"] == pytest.approx(item["test_score"] - item["cv_mean"], abs=1e-3)
        assert item["stability"] in ("stable", "unstable") and item["generalization"] in ("consistent", "inconsistent")
    assert -1 <= xr["cross_model"]["cv_vs_holdout_rank_spearman"] <= 1
    assert set(xr["cross_model"]["metric_leaders"]) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}


def test_selection_reason_is_built_from_the_real_results(analysis):
    _, out = analysis
    best = out["best_model"]
    assert best["name"] in out["selection_reason"] and f"{best['final_score']:.4f}" in out["selection_reason"]
    assert out["analysis_source"] == "deterministic"


def test_no_hardcoded_winner_in_the_engine():
    source = "\n".join(p.read_text(encoding="utf-8") for p in (Path(__file__).parents[1] / "ml").glob("*.py"))
    names = "|".join(re.escape(s.name) for s in CATALOG) + "|" + "|".join(s.key for s in CATALOG)
    assert not re.search(rf"(best_model|winner|best)\w*\s*=\s*[\"'](?:{names})[\"']", source)


# ---------------------------------------------------------------- comparator / selector units


def _row(key, family, cv, std=0.02, gap=0.0, train_time=1.0):
    return {"model": key.title(), "key": key, "type": family, "status": "ok", "cv_std": std, "cv_overfitting_gap": gap,
            "training_time_s": train_time, "cv_metrics": {"roc_auc": cv, "f1": cv, "pr_auc": cv, "accuracy": cv}}


def test_comparator_picks_a_hybrid_when_its_metrics_are_best():
    ml = MLSettings(_env_file=None)
    evaluation = {"primary_metric": "roc_auc", "failed_models": [], "models": [
        _row("a", "normal", 0.80), _row("b", "normal", 0.78), _row("c", "normal", 0.75), _row("stack", "hybrid", 0.83)]}
    ranking = ModelComparator(ml).compare("r", evaluation, imbalanced=False)["ranking"]
    assert ranking[0]["key"] == "stack" and ranking[0]["type"] == "hybrid"


def test_penalties_can_change_the_winner():
    ml = MLSettings(_env_file=None)
    evaluation = {"primary_metric": "roc_auc", "failed_models": [], "models": [
        _row("overfit", "normal", 0.82, gap=0.40), _row("steady", "hybrid", 0.80, gap=0.01), _row("c", "normal", 0.60)]}
    assert ModelComparator(ml).compare("r", evaluation, False)["ranking"][0]["key"] == "steady"
    lenient = ml.model_copy(update={"overfitting_penalty": 0.0})
    assert ModelComparator(lenient).compare("r", evaluation, False)["ranking"][0]["key"] == "overfit"


def test_imbalanced_weights_drop_accuracy():
    ml = MLSettings(_env_file=None)
    assert ModelComparator(ml).weights(True)["accuracy"] == 0
    with pytest.raises(ValueError, match="add up to 1"):
        ScoreWeights(roc_auc=0.9, f1=0.9, pr_auc=0, accuracy=0)


def test_too_few_valid_models_fails_with_the_recorded_errors():
    ml = MLSettings(_env_file=None)
    evaluation = {"primary_metric": "roc_auc", "failed_models": [{"model": "X", "status": "failed", "error": "boom"}],
                  "models": [_row("a", "normal", 0.8), {**_row("b", "normal", 0.7), "status": "failed"}]}
    with pytest.raises(NotEnoughModelsError, match="boom"):
        ModelComparator(ml).compare("r", evaluation, False)


# ---------------------------------------------------------------- failures and errors


def test_a_failing_model_does_not_cancel_the_analysis(tmp_path, monkeypatch):
    ai, ml = _prepared(tmp_path)
    broken = catalog.ModelSpec("svm", "SVM (RBF)", "normal", "broken", lambda ctx: (_ for _ in ()).throw(RuntimeError("cannot train")))
    monkeypatch.setattr(catalog, "CATALOG", tuple(broken if s.key == "svm" else s for s in catalog.CATALOG))
    out = MLAnalysisService(ai, ml).run({"models": ["logistic_regression", "svm", "random_forest", "gradient_boosting",
                                                    "voting_ensemble", "stacking_ensemble"]})
    assert out["status"] == "success" and out["models_evaluated"] == 5
    assert out["errors"] == [{"model": "SVM (RBF)", "status": "failed", "error": "RuntimeError: cannot train"}]
    assert "SVM (RBF)" not in [e["model"] for e in out["ranking"]]


@pytest.mark.parametrize(
    "payload, error",
    [
        ({"dataset": "nope"}, InvalidRequestError),
        ({"target": "los_hours"}, TargetNotFoundError),
        ({"features": ["age", "not_a_column"]}, InvalidColumnsError),
        ({"features": ["age"], "target": "icu_flag", "models": ["deep_net"]}, InvalidRequestError),
        ({"unexpected": 1}, InvalidRequestError),
        ({"problem_type": "regression"}, InvalidRequestError),
    ],
)
def test_invalid_requests(tmp_path, payload, error):
    ai, ml = _prepared(tmp_path, n_patients=20)
    with pytest.raises(error):
        MLAnalysisService(ai, ml).validate(payload)


def test_data_problems(tmp_path, monkeypatch):
    ai, ml = _settings(tmp_path)
    with pytest.raises(DatasetUnavailableError, match="prepare"):
        MLAnalysisService(ai, ml).validate({})
    ai, ml = _prepared(tmp_path, n_patients=20)
    with pytest.raises(InsufficientDataError):
        MLAnalysisService(ai, ml.model_copy(update={"min_rows": 500})).validate({})
    from ai_module.ml import datasets

    spec = datasets.DATASETS["mimic_icu"]
    monkeypatch.setitem(datasets.DATASETS, "mimic_icu", spec.__class__(**{**spec.__dict__, "loader": lambda s: pd.DataFrame()}))
    with pytest.raises(EmptyDatasetError):
        MLAnalysisService(ai, ml).validate({})


def test_steps_refuse_reports_from_different_runs_and_bad_run_ids(tmp_path):
    ai, ml = _prepared(tmp_path, n_patients=20)
    service = MLAnalysisService(ai, ml)
    with pytest.raises(IncompleteResultError):
        service.cross_validate({"run_id": "a"}, {"run_id": "b"})
    with pytest.raises(RunNotFoundError):
        service.preprocess({"run_id": "../../etc"})


def test_error_output_keeps_the_contract():
    out = error_output(TargetNotFoundError("no target"))
    assert out["status"] == "error" and out["error"] == {"code": "target_not_found", "message": "no target"}
    assert set(OUTPUT_KEYS) <= set(out)
    assert error_output(ZeroDivisionError("secret detail"))["error"]["message"] == "Unexpected error (ZeroDivisionError)"


# ---------------------------------------------------------------- LLM explanation cannot change the winner


def _explain(analysis, reply: str):
    service, out = analysis
    store = service.store
    run = out["run_id"]
    selection = {"run_id": run, "best_model": out["best_model"]}
    comparison = {"run_id": run, "ranking": out["ranking"], "score_config": out["score_config"]}
    xr = out["cross_reference_analysis"]
    return service.explain(selection, comparison, xr, use_llm=True, llm=scripted(AIMessage(content=reply))), store


def test_grounded_llm_explanation_is_used(analysis):
    _, out = analysis
    best = out["best_model"]
    result, _ = _explain(analysis, f"{best['name']} quedó primero con final_score {best['final_score']}.")
    assert result["analysis_source"] == "llm" and best["name"] in result["analysis"]


@pytest.mark.parametrize("reply", ["Random Forest es claramente el mejor modelo.", "{name} ganó con un score de 0.999123."])
def test_llm_explanations_that_contradict_or_invent_are_discarded(analysis, reply):
    _, out = analysis
    best = out["best_model"]
    if "{name}" not in reply and best["name"] == "Random Forest":
        reply = "Logistic Regression es claramente el mejor modelo."
    result, _ = _explain(analysis, reply.replace("{name}", best["name"]))
    assert result["analysis_source"] == "deterministic" and result["llm_problem"]
    assert result["selection_reason"].startswith(best["name"])
