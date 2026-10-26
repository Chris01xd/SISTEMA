"""Módulo 2 — EDA (análisis exploratorio de datos), CRISP-DM fase 2."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.paths import bootstrap

bootstrap()

from lib.export import chart, table  # noqa: E402
from lib.charts import categorical_bar, correlation_heatmap, numeric_histogram  # noqa: E402
from lib.pipeline import DATASET_LABELS, current_output  # noqa: E402
from lib.report import render_export_buttons  # noqa: E402

st.set_page_config(page_title="2. EDA", page_icon="🔍", layout="wide")
st.title("🔍 Módulo 2 — Análisis exploratorio de datos (EDA)")

out = current_output()
if out["status"] != "success":
    st.error("Ejecutá el análisis desde la portada (Home) primero.")
    st.stop()

eda, it = out["eda"], out["interpretations"]
st.caption(f"Dataset: {out['dataset']['dataset']} — {DATASET_LABELS[out['dataset']['dataset']]} · objetivo: {out['dataset']['target']}")

st.subheader("Resumen del dataset")
ov = eda["overview"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Filas (admisiones)", ov["rows"])
c2.metric("Pacientes (grupos)", ov["groups"])
c3.metric("Variables", ov["features"])
c4.metric("Clase positiva", f"{ov['positive_rate'] * 100:.1f}%")
st.info(it["eda_overview"])

st.subheader("Valores faltantes")
missing_df = pd.DataFrame(eda["missing"])
table(missing_df, 'Valores faltantes')
st.info(it["eda_missing"])

st.subheader("Variables numéricas y binarias")
st.caption("Prueba de Mann-Whitney U (positiva vs. negativa), corrección de Holm por comparaciones múltiples.")
for feature in eda["numeric"]:
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        chart(numeric_histogram(feature), key=f"hist_{feature['feature']}")
    with col_table:
        a = feature["association"]
        st.markdown(f"**{feature['feature']}**")
        st.write({"media": feature["mean"], "mediana": feature["median"], "std": feature["std"], "faltantes %": feature["missing_pct"],
                  "p-valor (Holm)": a.get("p_value_holm"), "significativo": a.get("significant"),
                  "efecto rank-biserial": a.get("effect_size_rank_biserial")})
st.info(it["eda_numeric"])

st.subheader("Variables categóricas")
st.caption("Prueba chi-cuadrado con V de Cramér, corrección de Holm.")
for feature in eda["categorical"]:
    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        chart(categorical_bar(feature), key=f"cat_{feature['feature']}")
    with col_table:
        a = feature["association"]
        st.markdown(f"**{feature['feature']}** ({feature['n_levels']} niveles)")
        st.write({"p-valor (Holm)": a.get("p_value_holm"), "V de Cramér": a.get("cramers_v"), "significativo": a.get("significant")})
st.info(it["eda_categorical"])

st.subheader("Correlaciones (Spearman)")
chart(correlation_heatmap(eda["correlations"]))
table(pd.DataFrame(eda["correlations"]["top_pairs"]), 'Correlaciones (Spearman)')
st.info(it["eda_correlations"])

st.divider()
render_export_buttons(out, key="eda")
