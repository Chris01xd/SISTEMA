"""Inferential statistics, EDA, hyper-parameter tuning (nested CV), interpretations and CRISP-DM."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from ai_module.ml.statistics import delong, holm, mcnemar, nadeau_bengio

# ---------------------------------------------------------------- statistics units


def test_holm_adjustment_matches_the_textbook_procedure():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_delong_aucs_match_sklearn_and_identical_models_are_not_different():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    good, noisy = y * 0.6 + rng.random(200) * 0.7, rng.random(200)
    result = delong(y, good, noisy)
    assert result["auc_a"] == pytest.approx(roc_auc_score(y, good), abs=1e-4)
    assert result["auc_b"] == pytest.approx(roc_auc_score(y, noisy), abs=1e-4)
    assert result["p_value"] < 0.001  # a clearly better model is detected
    assert delong(y, good, good)["p_value"] == 1.0


def test_nadeau_bengio_uses_the_corrected_variance():
    a, b = [0.80, 0.82, 0.78, 0.81, 0.79], [0.75, 0.77, 0.76, 0.74, 0.78]
    d = np.array(a) - np.array(b)
    expected_t = d.mean() / np.sqrt((1 / 5 + 44 / 176) * d.var(ddof=1))
    result = nadeau_bengio(a, b, n_train=176, n_test=44)
    assert result["t"] == pytest.approx(expected_t, abs=1e-3)
    assert result["p_value"] == pytest.approx(2 * stats.t.sf(abs(expected_t), 4), abs=1e-5)
    naive_t = d.mean() / np.sqrt(d.var(ddof=1) / 5)
    assert abs(result["t"]) < abs(naive_t)  # the correction makes the test more conservative


def test_mcnemar_counts_discordant_pairs():
    y = np.array([1, 1, 0, 0, 1, 0])
    a = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])  # all right
    b = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.3])  # two wrong
    result = mcnemar(y, a, b)
    assert (result["a_right_b_wrong"], result["a_wrong_b_right"]) == (2, 0)
    assert result["p_value"] == pytest.approx(0.5)


# ---------------------------------------------------------------- integrated output


def test_statistical_tests_in_the_analysis(ml_analysis):
    _, out = ml_analysis
    st = out["statistical_tests"]
    n_models = len(out["ranking"])
    assert st["best_model"] == out["best_model"]["name"]
    assert len(st["pairwise_vs_best"]) == n_models - 1
    assert 0 <= st["friedman"]["p_value"] <= 1 and st["nemenyi"]["critical_difference"] > 0
    for row in st["pairwise_vs_best"]:
        for test in ("nadeau_bengio", "delong", "mcnemar"):
            if row[test]["p_value"] is not None:
                assert row[test]["p_value_holm"] >= row[test]["p_value"] - 1e-9
    for ci in st["bootstrap_ci_95"].values():
        assert ci["roc_auc"][0] <= ci["roc_auc"][1]
    assert st["summary"]["evidence"] in ("strong", "partial", "weak")


def test_eda_tests_match_scipy(ml_analysis):
    service, out = ml_analysis
    frame = service.store.load_object(out["run_id"], "frame")
    item = next(n for n in out["eda"]["numeric"] if n["association"]["p_value"] is not None)
    y = frame["icu_flag"].astype(int)
    col = frame[item["feature"]]
    _, p = stats.mannwhitneyu(col[y == 1].dropna(), col[y == 0].dropna(), alternative="two-sided")
    assert item["association"]["p_value"] == pytest.approx(p, abs=1e-6)
    assert out["eda"]["overview"]["rows"] == len(frame)
    assert len(out["eda"]["correlations"]["matrix"]) == len(out["eda"]["correlations"]["features"])


def test_hyperparameters_are_tuned_with_nested_cv(ml_analysis):
    _, out = ml_analysis
    cv = out["cross_validation"]
    assert cv["nested"] is True and cv["inner_folds"] == 3
    for name, h in out["hyperparameters"].items():
        assert h["grid_size"] >= 2 and h["best_params"], name
        assert len(h["candidates"]) == h["grid_size"]
        assert len(h["fold_choices"]) == cv["folds"]  # a separate search inside every outer fold
        assert 0 < h["stability"] <= 1


def test_every_table_and_chart_has_an_interpretation(ml_analysis):
    _, out = ml_analysis
    expected = {"eda_overview", "eda_missing", "eda_numeric", "eda_categorical", "eda_correlations", "preprocessing",
                "hyperparameters", "cross_validation", "comparison_table", "ranking", "cross_reference", "statistical_tests",
                "roc_curves", "bootstrap_ci", "confusion_matrix", "best_model"}
    assert expected <= set(out["interpretations"])
    assert all(len(text) > 40 for text in out["interpretations"].values())
    assert out["best_model"]["name"] in out["interpretations"]["statistical_tests"]
    assert len(out["crisp_dm"]) == 6 and out["crisp_dm"][0]["phase"].startswith("1.")


def test_curves_exist_for_every_trained_model(ml_analysis):
    _, out = ml_analysis
    for name, curve in out["curves"].items():
        assert curve["roc"]["x"][0] == 0 and curve["roc"]["x"][-1] == 1, name
