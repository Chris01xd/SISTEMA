"""Módulo 9 — El mismo motor de ai_module.ml expuesto como flujo visual en LangFlow."""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from lib.paths import REPO_ROOT, bootstrap

bootstrap()

from lib.export import table  # noqa: E402
from lib.langflow_client import DEFAULT_BASE_URL, extract_json_payload, is_reachable, run_flow  # noqa: E402

st.set_page_config(page_title="9. LangFlow", page_icon="🔗", layout="wide")
st.title("🔗 Módulo 9 — Flujo LangFlow")

FLOW_PATH = REPO_ROOT / "langflow_integration" / "flows" / "ml_model_comparison.json"

st.markdown(
    """
El mismo pipeline CRISP-DM de los módulos 2 a 7 (`ai_module.ml`) también está expuesto
como un **flujo visual de LangFlow**: cada paso (validación, EDA, preprocesamiento,
entrenamiento, CV, evaluación, comparación, estadística, selección, explicación) es un
nodo del flujo, y los componentes de `langflow_integration/components/ml_pipeline/` son
adaptadores delgados que llaman exactamente a las mismas funciones de `ai_module.ml` que
usan las páginas anteriores — no hay una segunda implementación.
"""
)

if not FLOW_PATH.exists():
    st.error(f"No se encontró el archivo del flujo en {FLOW_PATH}.")
    st.stop()

flow = json.loads(FLOW_PATH.read_text(encoding="utf-8"))
nodes = flow["data"]["nodes"]
edges = flow["data"]["edges"]
endpoint = flow.get("endpoint_name", "ml-model-comparison")

st.subheader(f"Estructura del flujo: {flow.get('name', 'ML Model Comparison')}")
st.caption(flow.get("description", ""))
st.write(f"{len(nodes)} nodos, {len(edges)} conexiones. Endpoint: `{endpoint}`")
node_labels = [n.get("data", {}).get("type", n.get("id")) for n in nodes]
table(pd.DataFrame({"nodo": [n["id"] for n in nodes], "tipo de componente": node_labels}), "estructura_del_flujo")

st.divider()
st.subheader("Ejecutar el flujo")
base_url = st.text_input("URL del servidor LangFlow", value=os.environ.get("LANGFLOW_URL", DEFAULT_BASE_URL))
reachable = is_reachable(base_url)

if not reachable:
    st.warning(
        "No hay un servidor LangFlow corriendo en esa URL. Es el mismo motor que ya viste en los "
        "módulos 2–7 (por eso esta página no bloquea el resto de la app), pero si querés verlo "
        "correr como flujo visual, abrí una terminal y ejecutá:\n\n"
        "```powershell\npowershell -ExecutionPolicy Bypass -File langflow_integration\\scripts\\start_langflow.ps1\n```\n\n"
        "Esto levanta LangFlow en http://127.0.0.1:7860 con los componentes de `ml_pipeline` ya registrados "
        "(el flujo debe importarse una vez con `python langflow_integration/scripts/import_flow.py`)."
    )
else:
    st.success("Servidor LangFlow detectado.")
    dataset = st.selectbox("Dataset", options=["mimic_icu", "mimic_mortality"], key="langflow_dataset")
    if st.button("▶ Ejecutar el flujo vía HTTP", type="primary"):
        with st.spinner("Ejecutando el flujo completo en LangFlow (puede tardar, corre el mismo pipeline de punta a punta)..."):
            try:
                response = run_flow(endpoint, json.dumps({"dataset": dataset}), base_url=base_url)
                payload = extract_json_payload(response)
            except Exception as exc:
                st.error(f"El flujo falló: {exc}")
            else:
                st.success(f"Flujo ejecutado. Mejor modelo: {payload.get('best_model', {}).get('name')}")
                st.json(payload, expanded=False)
                st.caption("Mismo formato de salida que MLAnalysisService.run() (módulos 2–7): "
                          "podés compararlo campo a campo.")
