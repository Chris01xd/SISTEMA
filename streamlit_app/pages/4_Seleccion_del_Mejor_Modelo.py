"""Módulo 4 — Selección del mejor modelo, CRISP-DM fase 5-6."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.paths import bootstrap

bootstrap()

from lib.charts import confusion_matrix_fig, ranking_bar  # noqa: E402
from lib.pipeline import current_output  # noqa: E402
from lib.report import render_export_buttons  # noqa: E402

st.set_page_config(page_title="4. Selección del Mejor Modelo", page_icon="🏆", layout="wide")
st.title("🏆 Módulo 4 — Selección del mejor modelo")

out = current_output()
if out["status"] != "success":
    st.error("Ejecutá el análisis desde la portada (Home) primero.")
    st.stop()

best = out["best_model"]
st.success(f"Ganador: **{best['name']}** ({best['type']}) — final_score = {best['final_score']:.4f}")
st.caption(f"Fórmula: {out['score_config']['formula']}")
st.write("Pesos usados (dataset " + ("desbalanceado" if out["dataset"]["imbalanced"] else "balanceado") + "):",
        out["score_config"]["weights"])

st.subheader("Ranking")
st.plotly_chart(ranking_bar(out["ranking"]), width='stretch')
rank_df = pd.DataFrame([{"#": e["rank"], "modelo": e["model"], "tipo": e["type"], "final_score": e["final_score"],
                          **e["score_breakdown"]} for e in out["ranking"]])
st.dataframe(rank_df, width='stretch')
st.info(out["interpretations"]["ranking"])
st.info(out["interpretations"]["comparison_table"])

st.subheader("Matriz de confusión del ganador")
st.plotly_chart(confusion_matrix_fig(best["metrics"]["confusion_matrix"], out["dataset"]["target"]), width='content')
st.info(out["interpretations"]["confusion_matrix"])

st.subheader("Referencia cruzada CV vs. conjunto de prueba")
xr_df = pd.DataFrame([{"modelo": m["model"], "train": m["train_score"], "cv": m["cv_mean"], "test": m["test_score"],
                        "cv→test": m["cv_vs_test_difference"], "alertas": ", ".join(m.get("flag_codes") or []) or "—"}
                      for m in out["cross_reference_analysis"]["models"].values()])
st.dataframe(xr_df, width='stretch')
st.info(out["interpretations"]["cross_reference"])

st.subheader("¿Por qué se seleccionó este modelo?")
st.write(out["selection_reason"])
if out.get("analysis_source") == "llm":
    st.markdown("**Análisis adicional (LLM, no cambia la selección):**")
    st.write(out["analysis"])

st.divider()
render_export_buttons(out, key="best")
