"""Langflow side of the ML analysis. Runs in the Langflow environment:

    PYTHONPATH=<repo> langflow_integration/.venv/Scripts/python -m pytest langflow_integration/tests -q

Covers: every custom component loads with its inputs/outputs; the exported flow is well
formed; the component chain produces a complete result on synthetic MIMIC-shaped data;
errors are propagated as a JSON error payload; and the exported flow JSON runs end-to-end
with Langflow's own runner (lfx.load.run_flow_from_json)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

pytest.importorskip("lfx")
from lfx.schema.data import Data  # noqa: E402

COMPONENTS = {
    "dataset_input": ("MLDatasetInputComponent", ["request"]),
    "data_validation": ("MLDataValidationComponent", ["validation"]),
    "eda": ("MLEDAComponent", ["eda"]),
    "preprocessing": ("MLPreprocessingComponent", ["preprocessing"]),
    "normal_models": ("MLNormalModelsTrainingComponent", ["normal_models"]),
    "hybrid_models": ("MLHybridModelsTrainingComponent", ["hybrid_models"]),
    "cross_validation": ("MLCrossValidationComponent", ["cross_validation"]),
    "model_evaluation": ("MLModelEvaluationComponent", ["evaluation"]),
    "cross_reference": ("MLCrossReferenceComponent", ["cross_reference"]),
    "model_comparator": ("MLModelComparatorComponent", ["comparison"]),
    "statistical_tests": ("MLStatisticalTestsComponent", ["statistics"]),
    "best_model_selector": ("MLBestModelSelectorComponent", ["selection"]),
    "explanation": ("MLExplanationComponent", ["explanation"]),
    "analysis_output": ("MLAnalysisOutputComponent", ["message"]),
}


def load(stem: str):
    path = HERE / "components" / "ml_pipeline" / f"{stem}.py"
    name = f"ml_pipeline_{stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return getattr(module, COMPONENTS[stem][0])()


@pytest.fixture
def ml_env(tmp_path, monkeypatch):
    """Components build MLAnalysisService() from AI_* settings: point them at a temp dir with
    synthetic data, no LLM and no tuning (speed; tuning is covered by ai_module's tests)."""
    from ai_module.config import settings as ai_settings_module
    from ai_module.digital_twin.service import DigitalTwinService
    from ai_module.ml import config as ml_config
    from ai_module.tests.conftest import make_raw_tables

    for key, value in {"AI_DATA_DIR": tmp_path / "data", "AI_ARTIFACTS_DIR": tmp_path / "artifacts", "AI_LLM_PROVIDER": "none",
                       "AI_ML_RUNS_DIR": tmp_path / "artifacts" / "ml_runs", "AI_ML_TUNING_ENABLED": "false",
                       "AI_ML_BOOTSTRAP_RESAMPLES": "100", "AI_ENV_FILE": tmp_path / "none.env"}.items():
        monkeypatch.setenv(key, str(value))
    ai_settings_module.get_settings.cache_clear()
    ml_config.get_ml_settings.cache_clear()
    DigitalTwinService(ai_settings_module.get_settings()).preprocess_data(make_raw_tables(n_patients=50))
    yield
    ai_settings_module.get_settings.cache_clear()
    ml_config.get_ml_settings.cache_clear()


def test_every_component_loads_with_its_outputs():
    for stem, (_, outputs) in COMPONENTS.items():
        component = load(stem)
        assert [o.name for o in component.outputs] == outputs, stem
        assert component.display_name.startswith("ML ")


def test_exported_flow_is_complete_and_consistent():
    flow = json.loads((HERE / "flows" / "ml_model_comparison.json").read_text(encoding="utf-8"))
    assert flow["endpoint_name"] == "ml-model-comparison"
    nodes = {n["id"]: n for n in flow["data"]["nodes"]}
    types = {n["data"]["type"] for n in nodes.values()}
    assert {"ChatInput", "ChatOutput"} <= types
    assert {cls.removesuffix("Component") for cls, _ in COMPONENTS.values()} <= types
    for edge in flow["data"]["edges"]:
        assert edge["source"] in nodes and edge["target"] in nodes
    for node in nodes.values():  # each node carries its component source code
        if node["data"]["type"].startswith("ML"):
            assert "ai_module.ml.langflow_support" in node["data"]["node"]["template"]["code"]["value"]


def _chain(input_value: str) -> dict:
    c = {stem: load(stem) for stem in COMPONENTS}
    c["dataset_input"].set(input_value=input_value, explain_with_llm=False)
    req = c["dataset_input"].build_request()
    c["data_validation"].set(request=req)
    val = c["data_validation"].validate_data()
    c["eda"].set(validation=val)
    c["preprocessing"].set(validation=val)
    eda, pre = c["eda"].run_eda(), c["preprocessing"].preprocess()
    c["normal_models"].set(preprocessing=pre)
    c["hybrid_models"].set(preprocessing=pre)
    normal, hybrid = c["normal_models"].train_normal(), c["hybrid_models"].train_hybrid()
    c["cross_validation"].set(normal_models=normal, hybrid_models=hybrid)
    cv = c["cross_validation"].cross_validate()
    c["model_evaluation"].set(normal_models=normal, hybrid_models=hybrid, cross_validation=cv)
    ev = c["model_evaluation"].evaluate()
    c["cross_reference"].set(evaluation=ev, cross_validation=cv)
    xr = c["cross_reference"].cross_reference()
    c["model_comparator"].set(evaluation=ev, cross_reference=xr)
    comp = c["model_comparator"].compare()
    c["statistical_tests"].set(evaluation=ev, cross_validation=cv, comparison=comp)
    st = c["statistical_tests"].run_tests()
    c["best_model_selector"].set(comparison=comp, evaluation=ev)
    sel = c["best_model_selector"].select_best()
    c["explanation"].set(validation=val, selection=sel, comparison=comp, cross_reference=xr, statistics=st)
    exp = c["explanation"].explain()
    c["analysis_output"].set(validation=val, eda=eda, preprocessing=pre, normal_models=normal, hybrid_models=hybrid,
                             cross_validation=cv, evaluation=ev, cross_reference=xr, comparison=comp, statistics=st,
                             selection=sel, explanation=exp)
    return json.loads(c["analysis_output"].build_message().text)


def test_component_chain_produces_the_full_result(ml_env):
    out = _chain(json.dumps({"dataset": "mimic_icu", "target": "icu_flag", "cv_folds": 3}))
    assert out["status"] == "success", out.get("error")
    assert len(out["normal_models"]) >= 4 and len(out["hybrid_models"]) == 2
    assert out["best_model"]["name"] == out["ranking"][0]["model"]
    assert {e["type"] for e in out["ranking"]} == {"normal", "hybrid"}
    assert out["cross_validation"]["folds"] == 3 and out["statistical_tests"]["pairwise_vs_best"]
    assert out["interpretations"]["best_model"] and len(out["crisp_dm"]) == 6


@pytest.mark.parametrize("input_value, code", [
    ("not json", "invalid_request"),
    (json.dumps({"target": "nope"}), "target_not_found"),
    (json.dumps({"features": ["age", "bogus"]}), "invalid_columns"),
])
def test_errors_travel_through_the_flow_as_json(ml_env, input_value, code):
    c = {stem: load(stem) for stem in ("dataset_input", "data_validation", "preprocessing", "analysis_output")}
    c["dataset_input"].set(input_value=input_value)
    c["data_validation"].set(request=c["dataset_input"].build_request())
    val = c["data_validation"].validate_data()
    c["preprocessing"].set(validation=val)
    pre = c["preprocessing"].preprocess()  # a downstream step passes the error through unchanged
    assert pre.data["error"]["code"] == code
    c["analysis_output"].set(validation=val, preprocessing=pre)
    out = json.loads(c["analysis_output"].build_message().text)
    assert out["status"] == "error" and out["error"]["code"] == code and "ranking" in out


def test_missing_connections_are_reported_not_crashing(ml_env):
    component = load("analysis_output")
    out = json.loads(component.build_message().text)
    assert out["status"] == "error" and out["error"]["code"] == "incomplete_result"


def test_exported_flow_runs_end_to_end_with_langflows_runner(ml_env):
    from lfx.load import run_flow_from_json

    results = run_flow_from_json(str(HERE / "flows" / "ml_model_comparison.json"),
                                 input_value=json.dumps({"dataset": "mimic_icu", "cv_folds": 3, "explain_with_llm": False}),
                                 disable_logs=True)
    text = results[0].outputs[0].results["message"].text
    out = json.loads(text)
    assert out["status"] == "success", out.get("error")
    assert out["models_evaluated"] >= 6 and out["best_model"]["name"] == out["ranking"][0]["model"]
