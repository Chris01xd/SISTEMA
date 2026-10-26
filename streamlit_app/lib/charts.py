"""Plotly figures built directly from ai_module.ml's output dict — no numbers computed here."""
from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#ca8a04", "#4b5563"]


def numeric_histogram(feature: dict[str, Any]) -> go.Figure:
    hist = feature["histogram"]
    edges = hist["edges"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    width = [(edges[i + 1] - edges[i]) * 0.9 for i in range(len(edges) - 1)]
    fig = go.Figure()
    fig.add_bar(x=centers, y=hist["positive"], width=width, name="Clase positiva", marker_color=PALETTE[1], opacity=0.75)
    fig.add_bar(x=centers, y=hist["negative"], width=width, name="Clase negativa", marker_color=PALETTE[2], opacity=0.6)
    fig.update_layout(barmode="overlay", title=feature["feature"], xaxis_title=feature["feature"], yaxis_title="admisiones",
                      height=320, margin=dict(t=40, b=10, l=10, r=10), legend=dict(orientation="h", y=-0.25))
    return fig


def categorical_bar(feature: dict[str, Any]) -> go.Figure:
    levels = feature["levels"]
    fig = go.Figure()
    fig.add_bar(x=[lv["level"] for lv in levels], y=[lv["count"] for lv in levels], name="admisiones", marker_color=PALETTE[0])
    fig.add_scatter(x=[lv["level"] for lv in levels], y=[(lv["positive_rate"] or 0) * 100 for lv in levels], name="% clase positiva",
                    mode="lines+markers", yaxis="y2", marker_color=PALETTE[2])
    fig.update_layout(title=feature["feature"], height=320, margin=dict(t=40, b=10, l=10, r=10),
                      yaxis=dict(title="admisiones"), yaxis2=dict(title="% positivos", overlaying="y", side="right", range=[0, 100]),
                      legend=dict(orientation="h", y=-0.3))
    return fig


def correlation_heatmap(correlations: dict[str, Any]) -> go.Figure:
    features = correlations["features"]
    matrix = correlations["matrix"]
    fig = go.Figure(go.Heatmap(z=matrix, x=features, y=features, zmin=-1, zmax=1, colorscale="RdBu_r",
                                text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in matrix], texttemplate="%{text}"))
    fig.update_layout(title="Correlación de Spearman", height=max(400, 28 * len(features)), margin=dict(t=40, b=10, l=10, r=10))
    return fig


def roc_curves(curves: dict[str, Any], best_model: str) -> go.Figure:
    fig = go.Figure()
    for i, (name, c) in enumerate(curves.items()):
        roc = c.get("roc")
        if not roc:
            continue
        fig.add_scatter(x=roc["x"], y=roc["y"], mode="lines", name=name,
                        line=dict(width=3 if name == best_model else 1.5, color=PALETTE[i % len(PALETTE)]))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="azar", line=dict(dash="dash", color="#9ca3af"))
    fig.update_layout(title="Curvas ROC (conjunto de prueba)", xaxis_title="FPR", yaxis_title="TPR", height=420,
                      margin=dict(t=40, b=10, l=10, r=10))
    return fig


def ranking_bar(ranking: list[dict[str, Any]]) -> go.Figure:
    ordered = list(reversed(ranking))
    colors = [PALETTE[1] if e["rank"] == 1 else PALETTE[3] if e["type"] == "hybrid" else PALETTE[0] for e in ordered]
    fig = go.Figure(go.Bar(x=[e["final_score"] for e in ordered], y=[e["model"] for e in ordered], orientation="h",
                            marker_color=colors, text=[f"{e['final_score']:.4f}" for e in ordered], textposition="outside"))
    fig.update_layout(title="Ranking por final_score", height=60 + 40 * len(ordered), margin=dict(t=40, b=10, l=10, r=60))
    return fig


def cv_fold_bars(cv_models: dict[str, Any], primary_metric: str) -> go.Figure:
    fig = go.Figure()
    for name, m in cv_models.items():
        if m["status"] != "ok":
            continue
        fig.add_bar(name=name, x=[f["fold"] for f in m["folds"]], y=[f["validation"].get(primary_metric) for f in m["folds"]])
    fig.update_layout(barmode="group", title=f"{primary_metric} por fold (validación cruzada)", xaxis_title="fold",
                      yaxis_title=primary_metric, height=380, margin=dict(t=40, b=10, l=10, r=10))
    return fig


def cv_mean_std(cv_models: dict[str, Any]) -> go.Figure:
    ok = {name: m for name, m in cv_models.items() if m["status"] == "ok"}
    fig = go.Figure(go.Bar(x=list(ok.keys()), y=[m["cv_mean"] for m in ok.values()],
                            error_y=dict(type="data", array=[m["cv_std"] for m in ok.values()]), marker_color=PALETTE[0]))
    fig.update_layout(title="Media ± desviación estándar por modelo (CV)", height=380, margin=dict(t=40, b=10, l=10, r=10))
    return fig


def confusion_matrix_fig(cm: list[list[int]], target: str) -> go.Figure:
    (tn, fp), (fn, tp) = cm
    z = [[tn, fp], [fn, tp]]
    fig = go.Figure(go.Heatmap(z=z, x=[f"Pred. 0", f"Pred. 1"], y=[f"Real 0", f"Real 1"], colorscale="Blues",
                                text=z, texttemplate="%{text}", showscale=False))
    fig.update_layout(title=f"Matriz de confusión — {target} (umbral 0.5)", height=340, margin=dict(t=40, b=10, l=10, r=10))
    return fig


def all_figures(out: dict[str, Any]) -> dict[str, go.Figure]:
    """Every chart shown across the pages, keyed by a file-friendly name (used for the PDF annex and the ZIP export)."""
    eda, cv = out["eda"], out["cross_validation"]
    figs: dict[str, go.Figure] = {}
    for f in eda["numeric"]:
        figs[f"eda_hist_{f['feature']}"] = numeric_histogram(f)
    for f in eda["categorical"]:
        figs[f"eda_cat_{f['feature']}"] = categorical_bar(f)
    figs["eda_correlaciones_spearman"] = correlation_heatmap(eda["correlations"])
    figs["curvas_roc"] = roc_curves(out["curves"], out["best_model"]["name"])
    figs["ranking_final_score"] = ranking_bar(out["ranking"])
    figs["matriz_confusion_ganador"] = confusion_matrix_fig(out["best_model"]["metrics"]["confusion_matrix"], out["dataset"]["target"])
    figs["cv_por_fold"] = cv_fold_bars(cv["models"], cv["primary_metric"])
    figs["cv_media_desviacion"] = cv_mean_std(cv["models"])
    return figs
