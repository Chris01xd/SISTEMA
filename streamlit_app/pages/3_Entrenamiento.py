"""Módulo 3 — Entrenamiento (5 modelos normales + 2 ensambles híbridos), CRISP-DM fase 4."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.paths import bootstrap

bootstrap()

from lib.charts import roc_curves  # noqa: E402
from lib.pipeline import current_output  # noqa: E402
from lib.report import render_export_buttons  # noqa: E402

st.set_page_config(page_title="3. Entrenamiento", page_icon="🤖", layout="wide")
st.title("🤖 Módulo 3 — Entrenamiento de modelos")

out = current_output()
if out["status"] != "success":
    st.error("Ejecutá el análisis desde la portada (Home) primero.")
    st.stop()

st.info(out["interpretations"]["preprocessing"])

models = out["normal_models"] + out["hybrid_models"]
rows = [{
    "modelo": m["name"], "familia": m["family"], "estado": m["status"],
    "componentes": ", ".join(m.get("components") or []) or "—",
    "tiempo entrenamiento (s)": m["training_time_s"], "inferencia (ms/muestra)": m["inference_ms_per_sample"],
    "accuracy (test)": (m["test_metrics"] or {}).get("accuracy"), "ROC-AUC (test)": (m["test_metrics"] or {}).get("roc_auc"),
    "F1 (test)": (m["test_metrics"] or {}).get("f1"),
} for m in models]
st.subheader("Modelos entrenados (5 normales + 2 híbridos)")
st.dataframe(pd.DataFrame(rows), width='stretch')

st.subheader("Curvas ROC (conjunto de prueba)")
st.plotly_chart(roc_curves(out["curves"], out["best_model"]["name"]), width='stretch')
st.info(out["interpretations"]["roc_curves"])

st.subheader("Descripción de cada modelo")
for m in models:
    with st.expander(f"{m['name']} ({m['family']}) — {m['status']}"):
        st.write(m["description"])
        if m.get("components"):
            st.write("Componentes del ensamble:", ", ".join(m["components"]))
        if m.get("error"):
            st.error(m["error"])
        st.write("Métricas de entrenamiento:", m["train_metrics"])
        st.write("Métricas de prueba:", m["test_metrics"])

st.divider()
render_export_buttons(out, key="train")
