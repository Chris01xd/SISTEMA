"""Interpretation of every table and chart of the analysis, generated from the real values.

One entry per artefact shown in the UI and in the reports. The texts are deterministic,
so they are always consistent with the numbers and need no LLM. The CRISP-DM summary
maps each step of the analysis to its methodology phase."""
from __future__ import annotations

from typing import Any

T: dict[str, dict[str, str]] = {
    "es": {
        "eda_overview": "El dataset tiene {rows} filas y {features} variables ({numeric} numéricas, {categorical} categóricas, {boolean} binarias). "
                        "La clase positiva ('{target}') representa el {rate:.1f}% ({pos} casos), {balance}. {complete:.1f}% de las filas no tiene valores faltantes{dups}.",
        "balanced": "por lo que el problema está razonablemente balanceado y se usa ROC-AUC como métrica principal",
        "imbalanced": "por lo que el problema está desbalanceado: se usa PR-AUC como métrica principal, pesos por clase y la exactitud no puntúa",
        "dups": " y hay {n} filas duplicadas", "eda_missing": "{n} variables tienen valores faltantes; la más afectada es '{top}' ({pct:.1f}%). "
                        "Los faltantes se imputan con la mediana dentro del Pipeline (ajustada solo con datos de entrenamiento) y se añade un indicador de faltante, porque en triage la ausencia de un signo vital puede ser informativa.",
        "eda_missing_none": "Ninguna variable tiene valores faltantes.",
        "eda_numeric": "{sig} de {n} variables numéricas o binarias difieren significativamente entre clases (Mann-Whitney, Holm, α={alpha}). "
                       "La asociación más fuerte es '{top}' (efecto rank-biserial {eff:+.2f}; media {mpos} en positivos vs {mneg} en negativos). "
                       "{skew}",
        "skew": "Las variables con asimetría fuerte ({vars}) justifican el uso de pruebas no paramétricas y modelos robustos a escala.",
        "eda_categorical": "{sig} de {n} variables categóricas están asociadas al objetivo (chi-cuadrado, Holm). La más asociada es '{top}' (V de Cramér {v:.2f}); "
                           "su nivel con mayor tasa de positivos es '{level}' ({rate:.1f}%).{low}",
        "low": " Algunas tablas tienen frecuencias esperadas < 5, así que su p-valor es aproximado.",
        "eda_correlations": "La correlación de Spearman más alta entre predictores es '{a}'–'{b}' ({r:+.2f}){multi}. "
                            "El predictor numérico más correlacionado con el objetivo es '{t}' ({rt:+.2f}).",
        "multi": "; al superar |0.7| conviene vigilar la multicolinealidad (afecta a la interpretación de la regresión logística, no a los árboles)",
        "preprocessing": "Hold-out del {test_pct:.0f}% ({test} filas) separado por paciente (solapamiento {overlap}); el entrenamiento usa {train} filas. "
                         "Toda la preparación que aprende de los datos está dentro de cada Pipeline, por lo que no hay fuga de información hacia validación ni prueba.",
        "hyperparameters": "Cada modelo se ajustó con búsqueda en rejilla ({inner} folds internos agrupados) optimizando {primary}. "
                           "{stable} Una estabilidad baja indica que la mejor configuración depende de la muestra, algo esperable con pocos datos.",
        "hyper_stable": "Las configuraciones elegidas fueron estables entre folds externos en {n} de {total} modelos (≥60% de coincidencia).",
        "cross_validation": "Validación cruzada {nested}de {folds} folds agrupada por paciente. El mejor {primary} medio lo obtuvo {best} ({mean:.3f} ± {std:.3f}); "
                            "el más estable fue {stable} (desviación {sstd:.3f}). Las desviaciones de {lo:.3f}–{hi:.3f} reflejan la variabilidad de muestras pequeñas: "
                            "diferencias entre modelos menores a esa magnitud no son concluyentes por sí solas.",
        "nested": "anidada ",
        "comparison_table": "Todos los modelos se evaluaron con las mismas métricas. En el conjunto de prueba el mejor ROC-AUC fue {auc_best} ({auc:.3f}) y el mejor F1 {f1_best} ({f1:.3f}). "
                            "El mayor gap de sobreajuste (train−test) lo tiene {gap_model} ({gap:+.3f}). Las métricas de prueba provienen de un conjunto pequeño y sirven para confirmar, no para elegir.",
        "ranking": "El ranking ordena por final_score (métricas de CV ponderadas menos penalizaciones por inestabilidad y sobreajuste). {best} lidera con {s1:.4f}, "
                   "{margin:.4f} por encima de {second}. El mejor híbrido es {hybrid} (puesto {hrank}); {hybrid_note}",
        "hybrid_better": "los ensambles superan a los modelos individuales en este análisis.",
        "hybrid_worse": "con tan pocos datos, combinar modelos no aporta ventaja sobre el mejor modelo individual.",
        "cross_reference": "Correlación de Spearman entre el orden por CV y por prueba: {rho}. {agree} {flags}",
        "agree_yes": "CV y prueba coinciden en el mejor modelo.",
        "agree_no": "CV ({cvb}) y prueba ({hb}) no coinciden en el mejor modelo, lo que indica que las diferencias entre los primeros puestos son pequeñas.",
        "xr_flags": "Modelos con alertas: {items}.", "xr_none": "Ningún modelo presenta alertas de sobreajuste, inestabilidad o incoherencia.",
        "statistical_tests": "Friedman sobre los folds: χ²={chi}, p={p} ({fried}). Diferencia crítica de Nemenyi {cd:.2f} rangos. "
                             "Frente al ganador ({best}), tras la corrección de Holm: {beaten}. Evidencia de superioridad del ganador: {evidence}. "
                             "Con {folds} folds y {holdout} casos de prueba la potencia estadística es baja: la falta de significancia no demuestra que los modelos sean equivalentes.",
        "fried_sig": "hay diferencias significativas entre modelos", "fried_ns": "no hay diferencias significativas entre modelos",
        "beaten_some": "es significativamente mejor que {models}", "beaten_none": "ninguna diferencia es estadísticamente significativa",
        "evidence": {"strong": "fuerte", "partial": "parcial", "weak": "débil"},
        "flag_names": {"overfitting": "sobreajuste", "unstable": "inestable entre folds", "inconsistent": "CV y prueba discrepan"},
        "roc_curves": "Curvas ROC en el conjunto de prueba. Cuanto más se acerca una curva a la esquina superior izquierda, mejor separa las clases a cualquier umbral; "
                      "el área (ROC-AUC) del ganador es {auc:.3f} con IC95% bootstrap [{lo}, {hi}].",
        "bootstrap_ci": "Intervalos de confianza del 95% por bootstrap ({n} remuestreos del conjunto de prueba). Los intervalos de ROC-AUC tienen una "
                        "amplitud media de {width:.2f}; {overlap} Esto cuantifica la incertidumbre de las métricas de prueba con {holdout} casos.",
        "ci_overlap": "el intervalo del ganador se solapa con el de {n} de {total} modelos, por lo que sus diferencias en prueba no son concluyentes.",
        "confusion_matrix": "Matriz de confusión del ganador en prueba (umbral 0.5): {tp} verdaderos positivos, {fn} falsos negativos, {fp} falsos positivos y {tn} verdaderos negativos. "
                            "Recall {recall:.2f}: proporción de casos positivos ('{target}') detectados; precisión {precision:.2f}: aciertos entre los casos señalados. "
                            "En planificación de camas los falsos negativos (necesidades no anticipadas) suelen ser el error más costoso.",
    },
    "en": {
        "eda_overview": "The dataset has {rows} rows and {features} features ({numeric} numeric, {categorical} categorical, {boolean} binary). "
                        "The positive class ('{target}') is {rate:.1f}% ({pos} cases), {balance}. {complete:.1f}% of rows have no missing values{dups}.",
        "balanced": "so the problem is reasonably balanced and ROC-AUC is the primary metric",
        "imbalanced": "so the problem is imbalanced: PR-AUC is the primary metric, class weights are used and accuracy gets no weight",
        "dups": " and there are {n} duplicated rows", "eda_missing": "{n} features have missing values; the most affected is '{top}' ({pct:.1f}%). "
                        "Missing values are median-imputed inside the Pipeline (fitted on training data only) and a missing indicator is added, since a missing triage vital can be informative.",
        "eda_missing_none": "No feature has missing values.",
        "eda_numeric": "{sig} of {n} numeric or binary features differ significantly between classes (Mann-Whitney, Holm, α={alpha}). "
                       "The strongest association is '{top}' (rank-biserial effect {eff:+.2f}; mean {mpos} in positives vs {mneg} in negatives). {skew}",
        "skew": "Strongly skewed features ({vars}) justify non-parametric tests and scale-robust models.",
        "eda_categorical": "{sig} of {n} categorical features are associated with the target (chi-square, Holm). The most associated is '{top}' (Cramer's V {v:.2f}); "
                           "its level with the highest positive rate is '{level}' ({rate:.1f}%).{low}",
        "low": " Some tables have expected counts < 5, so their p-value is approximate.",
        "eda_correlations": "The highest Spearman correlation between predictors is '{a}'–'{b}' ({r:+.2f}){multi}. "
                            "The numeric predictor most correlated with the target is '{t}' ({rt:+.2f}).",
        "multi": "; above |0.7| multicollinearity should be watched (it affects logistic-regression interpretation, not trees)",
        "preprocessing": "{test_pct:.0f}% hold-out ({test} rows) split by patient (overlap {overlap}); training uses {train} rows. "
                         "Every data-dependent preparation step lives inside each Pipeline, so nothing leaks into validation or test data.",
        "hyperparameters": "Each model was tuned with grid search ({inner} grouped inner folds) optimizing {primary}. {stable} "
                           "Low stability means the best configuration depends on the sample, which is expected with little data.",
        "hyper_stable": "The chosen configurations were stable across outer folds for {n} of {total} models (≥60% agreement).",
        "cross_validation": "{nested}{folds}-fold cross-validation grouped by patient. The best mean {primary} was {best} ({mean:.3f} ± {std:.3f}); "
                            "the most stable was {stable} (std {sstd:.3f}). Standard deviations of {lo:.3f}–{hi:.3f} reflect small-sample variability: "
                            "differences between models smaller than that are not conclusive on their own.",
        "nested": "Nested ",
        "comparison_table": "All models were evaluated with the same metrics. On the test set the best ROC-AUC was {auc_best} ({auc:.3f}) and the best F1 {f1_best} ({f1:.3f}). "
                            "The largest overfitting gap (train−test) is {gap_model} ({gap:+.3f}). Test metrics come from a small set and confirm rather than choose.",
        "ranking": "The ranking orders by final_score (weighted CV metrics minus instability and overfitting penalties). {best} leads with {s1:.4f}, "
                   "{margin:.4f} above {second}. The best hybrid is {hybrid} (rank {hrank}); {hybrid_note}",
        "hybrid_better": "the ensembles beat the individual models in this analysis.",
        "hybrid_worse": "with so little data, combining models gives no advantage over the best individual model.",
        "cross_reference": "Spearman correlation between the CV and test orderings: {rho}. {agree} {flags}",
        "agree_yes": "CV and test agree on the best model.",
        "agree_no": "CV ({cvb}) and test ({hb}) disagree on the best model, which shows the differences at the top are small.",
        "xr_flags": "Models with flags: {items}.", "xr_none": "No model shows overfitting, instability or inconsistency flags.",
        "statistical_tests": "Friedman over folds: χ²={chi}, p={p} ({fried}). Nemenyi critical difference {cd:.2f} ranks. "
                             "Against the winner ({best}), after Holm correction: {beaten}. Evidence that the winner is better: {evidence}. "
                             "With {folds} folds and {holdout} test cases statistical power is low: non-significance does not prove the models are equivalent.",
        "fried_sig": "models differ significantly", "fried_ns": "no significant differences between models",
        "beaten_some": "it is significantly better than {models}", "beaten_none": "no difference is statistically significant",
        "evidence": {"strong": "strong", "partial": "partial", "weak": "weak"},
        "flag_names": {"overfitting": "overfitting", "unstable": "unstable across folds", "inconsistent": "CV and test disagree"},
        "roc_curves": "ROC curves on the test set. The closer a curve is to the top-left corner, the better it separates the classes at any threshold; "
                      "the winner's area (ROC-AUC) is {auc:.3f} with bootstrap 95% CI [{lo}, {hi}].",
        "bootstrap_ci": "Bootstrap 95% confidence intervals ({n} resamples of the test set). ROC-AUC intervals have a mean width of {width:.2f}; "
                        "{overlap} This quantifies the uncertainty of test metrics with {holdout} cases.",
        "ci_overlap": "the winner's interval overlaps with that of {n} of {total} models, so their test differences are not conclusive.",
        "confusion_matrix": "Winner's confusion matrix on test (threshold 0.5): {tp} true positives, {fn} false negatives, {fp} false positives and {tn} true negatives. "
                            "Recall {recall:.2f}: share of positive cases ('{target}') detected; precision {precision:.2f}: hits among those flagged. "
                            "For bed planning, false negatives (unanticipated needs) are usually the costliest error.",
    },
}

CRISP = {
    "es": [
        ("1. Comprensión del negocio", "Anticipar la necesidad de UCI o la mortalidad hospitalaria al ingreso para planificar camas en eventos de víctimas masivas; criterio de éxito: mejor final_score con evidencia estadística."),
        ("2. Comprensión de los datos", "Carga del dataset público MIMIC-IV Demo, validación y EDA con pruebas univariadas."),
        ("3. Preparación de los datos", "Selección de variables disponibles al ingreso, ingeniería de variables sin estado, Pipelines sin fuga y hold-out agrupado por paciente."),
        ("4. Modelado", "5 modelos normales + 2 híbridos con búsqueda de hiperparámetros en rejilla."),
        ("5. Evaluación", "CV anidada, métricas unificadas, referencias cruzadas, pruebas estadísticas, ranking y selección automática del mejor modelo."),
        ("6. Despliegue", "Modelo ganador persistido con metadatos, expuesto vía API del backend y orquestado por el flujo de Langflow."),
    ],
    "en": [
        ("1. Business understanding", "Anticipate ICU need (or mortality) at admission to plan beds during mass casualty events; success criterion: best final_score with statistical evidence."),
        ("2. Data understanding", "Loading of the public MIMIC-IV Demo dataset, validation and EDA with univariate tests."),
        ("3. Data preparation", "Admission-time features, stateless feature engineering, leak-free Pipelines and a patient-grouped hold-out."),
        ("4. Modeling", "5 normal + 2 hybrid models with grid-search hyper-parameter tuning."),
        ("5. Evaluation", "Nested CV, unified metrics, cross-reference, statistical tests, ranking and automatic best-model selection."),
        ("6. Deployment", "Winner persisted with metadata, exposed through the backend API and orchestrated by the Langflow flow."),
    ],
}


def _fmt(v: Any, d: int = 2) -> str:
    return "—" if v is None else f"{v:.{d}f}"


def interpret(out: dict[str, Any], language: str = "es") -> dict[str, str]:
    t = T.get(language, T["es"])
    eda, ds, pre = out["eda"], out["dataset"], out["preprocessing"]
    ov = eda["overview"]
    texts: dict[str, str] = {}

    texts["eda_overview"] = t["eda_overview"].format(
        rows=ov["rows"], features=ov["features"], numeric=ov["numeric"], categorical=ov["categorical"], boolean=ov["boolean"],
        target=ov["target"], rate=100 * ov["positive_rate"], pos=ov["class_counts"]["1"],
        balance=t["imbalanced"] if ds["imbalanced"] else t["balanced"], complete=ov["complete_rows_pct"],
        dups=t["dups"].format(n=ov["duplicated_rows"]) if ov["duplicated_rows"] else "")
    missing = [m for m in eda["missing"] if m["missing"]]
    texts["eda_missing"] = (t["eda_missing"].format(n=len(missing), top=missing[0]["feature"], pct=missing[0]["missing_pct"])
                            if missing else t["eda_missing_none"])

    numeric = [n for n in eda["numeric"] if n["association"]["p_value"] is not None]
    if numeric:
        top = max(numeric, key=lambda n: abs(n["association"]["effect_size_rank_biserial"] or 0))
        skewed = [n["feature"] for n in eda["numeric"] if n["skewness"] is not None and abs(n["skewness"]) > 1]
        texts["eda_numeric"] = t["eda_numeric"].format(
            sig=sum(bool(n["association"].get("significant")) for n in numeric), n=len(eda["numeric"]), alpha=eda["alpha"],
            top=top["feature"], eff=top["association"]["effect_size_rank_biserial"] or 0,
            mpos=_fmt(top["mean_positive"]), mneg=_fmt(top["mean_negative"]),
            skew=t["skew"].format(vars=", ".join(skewed[:4])) if skewed else "").strip()
    cats = [c for c in eda["categorical"] if c["association"]["p_value"] is not None]
    if cats:
        top = max(cats, key=lambda c: c["association"]["cramers_v"] or 0)
        level = max(top["levels"], key=lambda lv: (lv["positive_rate"] or 0) if lv["count"] >= 5 else -1)
        texts["eda_categorical"] = t["eda_categorical"].format(
            sig=sum(bool(c["association"].get("significant")) for c in cats), n=len(eda["categorical"]), top=top["feature"],
            v=top["association"]["cramers_v"] or 0, level=level["level"], rate=100 * (level["positive_rate"] or 0),
            low=t["low"] if any(c["association"]["low_expected_counts"] for c in cats) else "")
    corr = eda["correlations"]
    if corr["top_pairs"]:
        pair = corr["top_pairs"][0]
        with_t = max(corr["with_target"].items(), key=lambda kv: abs(kv[1] or 0))
        texts["eda_correlations"] = t["eda_correlations"].format(
            a=pair["a"], b=pair["b"], r=pair["spearman"], multi=t["multi"] if abs(pair["spearman"]) > 0.7 else "", t=with_t[0], rt=with_t[1] or 0)

    split = pre["split"]
    texts["preprocessing"] = t["preprocessing"].format(test_pct=100 * split["test_rows"] / (split["test_rows"] + split["train_rows"]),
                                                       test=split["test_rows"], overlap=split["group_overlap"], train=split["train_rows"])
    cv = out["cross_validation"]
    ok = [m for m in cv["models"].values() if m["status"] == "ok"]
    stable = [m for m in ok if (m.get("tuning_stability") or 0) >= 0.6]
    texts["hyperparameters"] = t["hyperparameters"].format(
        inner=cv.get("inner_folds") or "-", primary=ds["primary_metric"], stable=t["hyper_stable"].format(n=len(stable), total=len(ok)))
    best_cv = max(ok, key=lambda m: m["cv_mean"])
    most_stable = min(ok, key=lambda m: m["cv_std"])
    stds = [m["cv_std"] for m in ok]
    texts["cross_validation"] = t["cross_validation"].format(
        nested=t["nested"] if cv.get("nested") else "", folds=cv["folds"], primary=ds["primary_metric"], best=best_cv["name"],
        mean=best_cv["cv_mean"], std=best_cv["cv_std"], stable=most_stable["name"], sstd=most_stable["cv_std"], lo=min(stds), hi=max(stds))

    rows = [r for r in out["comparison_table"] if r["status"] == "ok"]
    auc = max(rows, key=lambda r: r["roc_auc"] or 0)
    f1 = max(rows, key=lambda r: r["f1"] or 0)
    gap = max(rows, key=lambda r: r["overfitting_gap"] or 0)
    texts["comparison_table"] = t["comparison_table"].format(auc_best=auc["model"], auc=auc["roc_auc"], f1_best=f1["model"], f1=f1["f1"],
                                                             gap_model=gap["model"], gap=gap["overfitting_gap"])
    ranking = out["ranking"]
    hybrid = next(e for e in ranking if e["type"] == "hybrid")
    texts["ranking"] = t["ranking"].format(
        best=ranking[0]["model"], s1=ranking[0]["final_score"], margin=ranking[0]["final_score"] - ranking[1]["final_score"],
        second=ranking[1]["model"], hybrid=hybrid["model"], hrank=hybrid["rank"],
        hybrid_note=t["hybrid_better"] if ranking[0]["type"] == "hybrid" else t["hybrid_worse"])

    xr = out["cross_reference_analysis"]
    cm = xr["cross_model"]
    names = t["flag_names"]
    flagged = [f"{m['model']} ({', '.join(names[c] for c in m.get('flag_codes', []))})" for m in xr["models"].values() if m.get("flag_codes")]
    texts["cross_reference"] = t["cross_reference"].format(
        rho=_fmt(cm.get("cv_vs_holdout_rank_spearman")),
        agree=t["agree_yes"] if cm.get("cv_and_holdout_agree_on_best") else t["agree_no"].format(cvb=cm.get("cv_best"), hb=cm.get("holdout_best")),
        flags=t["xr_flags"].format(items="; ".join(flagged)) if flagged else t["xr_none"])

    st = out["statistical_tests"]
    fr = st["friedman"]
    beaten = st["summary"]["models_significantly_worse_than_best"]
    texts["statistical_tests"] = t["statistical_tests"].format(
        chi=_fmt(fr["statistic"]), p=_fmt(fr["p_value"], 4), fried=t["fried_sig"] if fr.get("significant") else t["fried_ns"],
        cd=st["nemenyi"]["critical_difference"], best=st["best_model"],
        beaten=t["beaten_some"].format(models=", ".join(beaten)) if beaten else t["beaten_none"],
        evidence=t["evidence"][st["summary"]["evidence"]], folds=st["cv_folds"], holdout=st["holdout_size"])

    best = out["best_model"]
    ci = st["bootstrap_ci_95"].get(best["name"], {}).get("roc_auc", [None, None])
    intervals = {name: c["roc_auc"] for name, c in st["bootstrap_ci_95"].items() if None not in c["roc_auc"]}
    if intervals and None not in ci:
        others = {n: iv for n, iv in intervals.items() if n != best["name"]}
        overlapping = sum(1 for lo, hi in others.values() if lo <= ci[1] and hi >= ci[0])
        texts["bootstrap_ci"] = t["bootstrap_ci"].format(
            n=st.get("bootstrap_resamples", "B"), width=sum(hi - lo for lo, hi in intervals.values()) / len(intervals),
            overlap=t["ci_overlap"].format(n=overlapping, total=len(others)), holdout=st["holdout_size"])
    texts["roc_curves"] = t["roc_curves"].format(auc=best["metrics"]["roc_auc"] or 0, lo=_fmt(ci[0], 3), hi=_fmt(ci[1], 3))
    (tn, fp), (fn, tp) = best["metrics"]["confusion_matrix"]
    texts["confusion_matrix"] = t["confusion_matrix"].format(tp=tp, fn=fn, fp=fp, tn=tn, recall=best["metrics"]["recall"], target=ds["target"],
                                                             precision=best["metrics"]["precision"])
    texts["best_model"] = out["selection_reason"]
    return texts


def crisp_dm(language: str = "es") -> list[dict[str, str]]:
    return [{"phase": phase, "summary": summary} for phase, summary in CRISP.get(language, CRISP["es"])]
