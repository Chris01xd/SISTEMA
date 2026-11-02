"""Módulo 9 — El mismo motor de ai_module.ml expuesto como flujo visual en LangFlow."""
from __future__ import annotations

import json
import os
import time

import pandas as pd
import streamlit as st

from lib.paths import REPO_ROOT, bootstrap

bootstrap()

from lib.charts import ranking_bar, roc_curves  # noqa: E402
from lib.export import _stretch, chart, table  # noqa: E402
from lib.langflow_client import (  # noqa: E402
    DEFAULT_BASE_URL, LangflowError, extract_json_payload, flow_status, import_flow, langflow_installed, run_flow,
    server_log_tail, start_server,
)
from lib.pipeline import DATASET_LABELS, run_analysis  # noqa: E402
from lib.report import render_export_buttons  # noqa: E402

st.set_page_config(page_title="9. LangFlow", page_icon="🔗", layout="wide")
st.title("🔗 Módulo 9 — Flujo LangFlow")

FLOW_PATH = REPO_ROOT / "langflow_integration" / "flows" / "ml_model_comparison.json"

st.markdown(
    """
El mismo pipeline CRISP-DM de los módulos 2 a 7 (`ai_module.ml`) también está expuesto como un
**flujo visual de LangFlow**: cada paso (validación, EDA, preprocesamiento, entrenamiento, CV,
evaluación, comparación, estadística, selección, explicación) es un nodo, y los componentes de
`langflow_integration/components/ml_pipeline/` llaman a las mismas funciones de `ai_module.ml`
que usan las páginas anteriores — no hay una segunda implementación.
"""
)

if not FLOW_PATH.exists():
    st.error(f"No se encontró el archivo del flujo en {FLOW_PATH}.")
    st.stop()

flow = json.loads(FLOW_PATH.read_text(encoding="utf-8"))
nodes, edges = flow["data"]["nodes"], flow["data"]["edges"]
endpoint = flow.get("endpoint_name", "ml-model-comparison")

# --------------------------------------------------------------------------------- 1. server / flow status

st.subheader("1. Estado de LangFlow")
base_url = st.text_input("URL del servidor LangFlow", value=os.environ.get("LANGFLOW_URL", DEFAULT_BASE_URL)).rstrip("/")


@st.fragment(run_every=5)
def wait_for_server() -> None:
    """While a server we started is booting: poll /health and show its log, then rerun the page."""
    started = st.session_state.get("lf_starting")
    if not started:
        return
    status = flow_status(endpoint, base_url)
    if status.reachable:
        st.session_state.pop("lf_starting", None)
        st.rerun()
    elapsed = int(time.time() - started)
    st.info(f"Iniciando LangFlow… {elapsed} s (suele tardar 1–2 minutos la primera vez).")
    log = server_log_tail()
    if log:
        st.code(log, language="text")
    if elapsed > 300:
        st.session_state.pop("lf_starting", None)
        st.error("LangFlow no arrancó en 5 minutos. Revisá el registro de arriba.")


status = flow_status(endpoint, base_url)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Servidor", "activo" if status.reachable else "apagado")
c2.metric("API key", "válida" if status.auth_ok else "—" if not status.reachable else "falta / inválida")
c3.metric("Flujo importado", "sí" if status.flow else "—" if not status.auth_ok else "no")
c4.metric("Versión", status.version or "—")

if st.session_state.get("lf_starting"):
    wait_for_server()
elif not status.reachable:
    st.warning(
        "No hay un servidor LangFlow en esa URL. Los módulos 2–7 no lo necesitan (usan el mismo motor directamente); "
        "solo hace falta para verlo correr como flujo."
    )
    if langflow_installed() and base_url in (DEFAULT_BASE_URL, "http://localhost:7860"):
        if st.button("▶ Iniciar servidor LangFlow", type="primary"):
            start_server()
            st.session_state["lf_starting"] = time.time()
            st.rerun()
    else:
        st.markdown("Para iniciarlo manualmente:")
    st.code("powershell -ExecutionPolicy Bypass -File langflow_integration\\scripts\\start_langflow.ps1", language="powershell")
    if not langflow_installed():
        st.caption("LangFlow no está instalado en `langflow_integration/.venv` (ver langflow_integration/README.md).")

if status.reachable and not status.ready:
    st.warning(status.problem)
if status.reachable:
    b1, b2, _ = st.columns([1.3, 1, 3])
    if b1.button("⬆ Importar / actualizar flujo", type="secondary" if status.ready else "primary",
                 help="Sube flows/ml_model_comparison.json al servidor (lo actualiza si ya existe) y crea la API key si falta."):
        try:
            res = import_flow(base_url, create_key=not status.auth_ok)
        except LangflowError as exc:
            st.error(str(exc))
        else:
            st.success(f"Flujo {'actualizado' if res.get('action') == 'updated' else 'importado'} (id {res.get('flow_id')}).")
            time.sleep(0.8)
            st.rerun()
    if b2.button("🔄 Volver a comprobar"):
        st.rerun()
    if status.ready:
        st.success(f"Listo para ejecutar: `{base_url}/api/v1/run/{endpoint}` · editor visual: {base_url}/flow/{status.flow['id']}")

# --------------------------------------------------------------------------------- 2. flow structure

st.divider()
st.subheader(f"2. Estructura del flujo: {flow.get('name', 'ML Model Comparison')}")
st.caption(flow.get("description", ""))
st.write(f"{len(nodes)} nodos, {len(edges)} conexiones. Endpoint: `{endpoint}`")

labels = {n["id"]: n.get("data", {}).get("node", {}).get("display_name") or n.get("data", {}).get("type", n["id"]) for n in nodes}
dot = ["digraph G {", 'rankdir=LR; bgcolor="transparent";',
       'node [shape=box, style="rounded,filled", fillcolor="#eef2ff", color="#2563eb", fontname="Helvetica", fontsize=11];',
       'edge [color="#9ca3af", arrowsize=0.6];']
for nid, label in labels.items():
    fill = ' fillcolor="#dcfce7", color="#16a34a"' if nid.startswith("Chat") else ""
    dot.append(f'"{nid}" [label="{label}"{fill}];')
dot += [f'"{e["source"]}" -> "{e["target"]}";' for e in edges] + ["}"]
with st.expander("Diagrama del flujo", expanded=True):
    _stretch(st.graphviz_chart, "\n".join(dot))

incoming = pd.Series([e["target"] for e in edges]).value_counts()
outgoing = pd.Series([e["source"] for e in edges]).value_counts()
table(pd.DataFrame([{
    "nodo": labels[n["id"]], "componente": n.get("data", {}).get("type"),
    "descripción": n.get("data", {}).get("node", {}).get("description", ""),
    "entradas": int(incoming.get(n["id"], 0)), "salidas": int(outgoing.get(n["id"], 0)),
} for n in nodes]), "Estructura del flujo LangFlow")

# --------------------------------------------------------------------------------- 3. run

st.divider()
st.subheader("3. Ejecutar el flujo")
if not status.ready:
    st.info("Cuando el servidor esté activo y el flujo importado (sección 1) vas a poder ejecutarlo desde acá.")
else:
    with st.form("langflow_run"):
        f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
        dataset = f1.selectbox("Dataset", options=list(DATASET_LABELS), format_func=lambda k: f"{k} — {DATASET_LABELS[k]}")
        cv_folds = f2.number_input("Folds de CV", min_value=2, max_value=10, value=5)
        language = f3.selectbox("Idioma", ["es", "en"])
        use_llm = f4.checkbox("Explicación con LLM", value=True, help="Si el proveedor falla, el flujo sigue y usa la explicación determinista.")
        submitted = st.form_submit_button("▶ Ejecutar el flujo vía HTTP", type="primary")
    if submitted:
        request = {"dataset": dataset, "cv_folds": int(cv_folds), "explain_with_llm": use_llm, "language": language}
        with st.spinner("Ejecutando el flujo completo en LangFlow (3–4 min: EDA, búsqueda de hiperparámetros, CV anidada, pruebas)…"):
            t0 = time.time()
            try:
                payload = extract_json_payload(run_flow(endpoint, json.dumps(request), base_url=base_url))
            except LangflowError as exc:
                st.session_state["lf_result"] = {"request": request, "error": str(exc)}
            else:
                st.session_state["lf_result"] = {"request": request, "payload": payload, "seconds": time.time() - t0}

# --------------------------------------------------------------------------------- 4. results (kept across reruns)

result = st.session_state.get("lf_result")
if result:
    st.divider()
    st.subheader("4. Resultado del flujo")
    st.caption("Solicitud enviada: `" + json.dumps(result["request"]) + "`")
    if "error" in result:
        st.error(result["error"])
        st.stop()
    out = result["payload"]
    if out.get("status") != "success":
        err = out.get("error") or {}
        st.error(f"El flujo terminó con error `{err.get('code', 'desconocido')}`: {err.get('message', 'sin detalle')}")
        st.download_button("⬇ Respuesta JSON", json.dumps(out, ensure_ascii=False, indent=2), "langflow_respuesta.json", "application/json")
        st.stop()

    best = out["best_model"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mejor modelo", best["name"])
    m2.metric("final_score", f"{best['final_score']:.4f}")
    m3.metric("Modelos evaluados", f"{out.get('models_evaluated')}/{out.get('models_attempted')}")
    m4.metric("Duración", f"{result['seconds']:.0f} s")
    for w in out.get("warnings") or []:
        st.warning(w)
    if out.get("analysis_source") != "llm" and result["request"]["explain_with_llm"]:
        st.caption("La explicación del LLM no estuvo disponible en esta ejecución; se muestra la explicación determinista.")

    st.markdown("**¿Por qué se seleccionó este modelo?**")
    st.info(out["selection_reason"])
    if out.get("analysis_source") == "llm":
        st.markdown("**Análisis del LLM** (no altera la selección)")
        st.info(out["analysis"])

    chart(ranking_bar(out["ranking"]), key="lf_ranking")
    table(pd.DataFrame([{"#": e["rank"], "modelo": e["model"], "tipo": e["type"], "final_score": e["final_score"],
                         **e["score_breakdown"]} for e in out["ranking"]]), "Ranking (LangFlow)")
    chart(roc_curves(out["curves"], best["name"]), key="lf_roc")

    with st.expander("Comparar con el motor directo (módulos 2–7)"):
        st.caption("Corre el mismo análisis sin LangFlow y compara el ranking. Mismos datos y semillas → mismos números. "
                   "Las diferencias solo pueden venir de la explicación del LLM, que no altera la selección.")
        if st.button("Comparar", key="lf_compare"):
            req = result["request"]
            direct = run_analysis(req["dataset"], req["cv_folds"], False, req["language"])
            if direct.get("status") != "success":
                st.error(f"El motor directo falló: {(direct.get('error') or {}).get('message')}")
            else:
                d = {e["model"]: e["final_score"] for e in direct["ranking"]}
                cmp = pd.DataFrame([{"modelo": e["model"], "final_score LangFlow": e["final_score"],
                                     "final_score directo": d.get(e["model"]),
                                     "diferencia": None if d.get(e["model"]) is None else e["final_score"] - d[e["model"]]}
                                    for e in out["ranking"]])
                same = direct["best_model"]["name"] == best["name"] and cmp["diferencia"].abs().max() < 1e-9
                (st.success if same else st.warning)(
                    "Resultados idénticos: mismo ganador y mismos final_score." if same else
                    f"Hay diferencias (ganador directo: {direct['best_model']['name']}).")
                table(cmp, "Comparación LangFlow vs. motor directo")

    st.divider()
    st.download_button("⬇ Respuesta completa del flujo (JSON)", json.dumps(out, ensure_ascii=False, indent=2),
                       f"langflow_{out['run_id']}.json", "application/json")
    render_export_buttons(out, key="langflow")
