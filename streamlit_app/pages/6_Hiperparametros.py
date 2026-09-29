"""Módulo 6 — Hiperparámetros (búsqueda en rejilla anidada), CRISP-DM fase 4."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.paths import bootstrap

bootstrap()

from lib.pipeline import current_output  # noqa: E402
from lib.report import render_export_buttons  # noqa: E402

st.set_page_config(page_title="6. Hiperparámetros", page_icon="⚙️", layout="wide")
st.title("⚙️ Módulo 6 — Hiperparámetros")

out = current_output()
if out["status"] != "success":
    st.error("Ejecutá el análisis desde la portada (Home) primero.")
    st.stop()

hp = out["hyperparameters"]
st.subheader("Resumen por modelo")
summary_rows = [{"modelo": name, "tamaño de la rejilla": h.get("grid_size"), "mejores parámetros (refit final)": h.get("best_params"),
                  "score interno": h.get("best_inner_score"), "estabilidad entre folds externos": h.get("stability")}
                for name, h in hp.items()]
st.dataframe(pd.DataFrame(summary_rows), width='stretch')
st.info(out["interpretations"]["hyperparameters"])

st.subheader("Detalle por modelo")
model_name = st.selectbox("Modelo", options=list(hp.keys()))
h = hp[model_name]
if not h.get("enabled"):
    st.warning("Este modelo no tiene rejilla de hiperparámetros (usa sus valores por defecto).")
else:
    st.write(f"Búsqueda: GridSearchCV, {h['inner_folds']} folds internos agrupados por paciente, "
            f"optimizando **{h['scoring']}**, {h['grid_size']} configuraciones evaluadas.")
    st.markdown("**Todas las configuraciones probadas (ordenadas por rank):**")
    st.dataframe(pd.DataFrame(h.get("candidates") or []), width='stretch')
    st.markdown("**Configuración elegida en cada fold externo de la validación cruzada:**")
    st.dataframe(pd.DataFrame(h.get("fold_choices") or []), width='stretch')

st.divider()
render_export_buttons(out, key="hyper")
