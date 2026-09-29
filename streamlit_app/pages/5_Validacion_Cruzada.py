"""Módulo 5 — Validación cruzada (anidada, agrupada por paciente), CRISP-DM fase 5."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.paths import bootstrap

bootstrap()

from lib.charts import cv_fold_bars, cv_mean_std  # noqa: E402
from lib.pipeline import current_output  # noqa: E402
from lib.report import render_export_buttons  # noqa: E402

st.set_page_config(page_title="5. Validación Cruzada", page_icon="🔁", layout="wide")
st.title("🔁 Módulo 5 — Validación cruzada")

out = current_output()
if out["status"] != "success":
    st.error("Ejecutá el análisis desde la portada (Home) primero.")
    st.stop()

cv = out["cross_validation"]
st.caption(f"Estrategia: {cv['strategy']} · {cv['folds']} folds externos"
          + (f" · {cv['inner_folds']} folds internos (búsqueda de hiperparámetros anidada)" if cv["nested"] else "")
          + f" · métrica primaria: {cv['primary_metric']}")

st.subheader("Métrica primaria por modelo y por fold")
st.plotly_chart(cv_fold_bars(cv["models"], cv["primary_metric"]), width='stretch')

st.subheader("Media ± desviación estándar por modelo")
st.plotly_chart(cv_mean_std(cv["models"]), width='stretch')
st.info(out["interpretations"]["cross_validation"])

st.subheader("Detalle por fold")
model_name = st.selectbox("Modelo", options=[m["name"] for m in cv["models"].values() if m["status"] == "ok"])
model = next(m for m in cv["models"].values() if m["name"] == model_name)
fold_rows = [{"fold": f["fold"], "train (n)": f["train_size"], "validación (n)": f["validation_size"],
              **{f"val_{k}": v for k, v in f["validation"].items()}, "tiempo ajuste (s)": f["fit_time_s"]}
            for f in model["folds"]]
st.dataframe(pd.DataFrame(fold_rows), width='stretch')
c1, c2, c3 = st.columns(3)
c1.metric("cv_mean", model["cv_mean"])
c2.metric("cv_std", model["cv_std"])
c3.metric("Estabilidad de hiperparámetros", model.get("tuning_stability"))

st.divider()
render_export_buttons(out, key="cv")
