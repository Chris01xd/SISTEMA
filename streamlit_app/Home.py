"""Portada: metodología CRISP-DM, selección de dataset y ejecución del motor."""
from __future__ import annotations

import streamlit as st

from lib.paths import bootstrap

bootstrap()

from lib.pipeline import DATASET_LABELS, current_output, ensure_defaults  # noqa: E402

st.set_page_config(page_title="Motor CRISP-DM — MIMIC-IV", page_icon="🏥", layout="wide")
ensure_defaults()

st.title("Motor de Machine Learning — Metodología CRISP-DM")
st.caption("Dataset real: MIMIC-IV Clinical Database Demo v2.2 + MIMIC-IV-ED Demo v2.2 (PhysioNet, licencia ODbL)")

st.markdown(
    """
Esta app tiene **9 módulos independientes**, uno por página en la barra lateral, cada uno
con sus propias tablas, gráficos e **interpretación/explicabilidad** en pantalla:

1. 📥 Carga del dataset (datos reales)
2. 🔍 EDA (análisis exploratorio)
3. 🤖 Entrenamiento
4. 🏆 Selección del mejor modelo
5. 🔁 Validación cruzada
6. ⚙️ Hiperparámetros
7. 📊 Pruebas estadísticas inferenciales robustas
8. 💬 Módulo LangChain (asistente conversacional sobre el dataset)
9. 🔗 Módulo LangFlow (el mismo motor como flujo visual)

Cada página de análisis (2–7) exporta el mismo contenido como **reporte PDF/Excel**.
"""
)

with st.expander("Cita del dataset (obligatoria por licencia ODbL)"):
    st.markdown(
        """
- Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023).
  *MIMIC-IV Clinical Database Demo* (v2.2). PhysioNet. https://doi.org/10.13026/dp1f-ex47
- Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., & Horng, S. (2023).
  *MIMIC-IV-ED Demo* (v2.2). PhysioNet. https://doi.org/10.13026/jzz5-vs76
- Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23), e215–e220.

100 pacientes reales, de-identificados, del Beth Israel Deaconess Medical Center. Muestra
pequeña: los resultados son ilustrativos, no representativos de un hospital real.
"""
    )

st.divider()
st.subheader("Configuración del análisis")
col1, col2, col3 = st.columns(3)
with col1:
    dataset = st.selectbox("Dataset / objetivo", options=list(DATASET_LABELS), format_func=lambda k: f"{k} — {DATASET_LABELS[k]}",
                           index=list(DATASET_LABELS).index(st.session_state["dataset"]))
with col2:
    cv_folds = st.slider("Folds de validación cruzada", min_value=3, max_value=10, value=st.session_state["cv_folds"])
with col3:
    explain_with_llm = st.checkbox("Explicación adicional con LLM (Claude)", value=st.session_state["explain_with_llm"],
                                   help="Si no hay ANTHROPIC_API_KEY configurada, se usa igualmente la explicación "
                                        "determinística (la misma que interpreta cada tabla y gráfico).")

st.session_state["dataset"], st.session_state["cv_folds"], st.session_state["explain_with_llm"] = dataset, cv_folds, explain_with_llm

if st.button("▶ Ejecutar / actualizar el análisis", type="primary"):
    st.cache_data.clear()

with st.spinner("Ejecutando validación → EDA → preprocesamiento → entrenamiento → CV → estadística → selección..."):
    out = current_output()

if out["status"] != "success":
    st.error(f"El pipeline falló: {out['error']['message']}")
    st.stop()

st.success(f"Análisis '{out['run_id']}' listo — {out['dataset']['rows']} admisiones, objetivo '{out['dataset']['target']}'.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelos evaluados", out["models_evaluated"])
c2.metric("Mejor modelo", out["best_model"]["name"])
c3.metric("final_score", f"{out['best_model']['final_score']:.4f}")
c4.metric("Duración (s)", out.get("duration_s"))

st.divider()
st.subheader("Mapeo a las fases CRISP-DM")
for phase in out["crisp_dm"]:
    st.markdown(f"**{phase['phase']}** — {phase['summary']}")

st.info("Usá la barra lateral para recorrer los 9 módulos. Los módulos 2 a 7 comparten este "
        "mismo análisis (así los números coinciden entre pantalla y reporte); cambiá la "
        "configuración de arriba y volvé a ejecutar para regenerarlos todos.")
