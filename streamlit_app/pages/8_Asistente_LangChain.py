"""Módulo 8 — Asistente conversacional con LangChain sobre el dataset MIMIC-IV.

Reutiliza langchain_module/hospital_assistant tal cual: un agente "tool-only" que nunca
recibe los datos crudos, solo puede llamar herramientas que agregan el DataFrame con pandas."""
from __future__ import annotations

import os

import streamlit as st

from lib.paths import bootstrap

bootstrap()

from hospital_assistant.agent import ask, build_agent  # noqa: E402
from hospital_assistant.config import Settings  # noqa: E402
from hospital_assistant.data import DatasetError, load_dataset  # noqa: E402

st.set_page_config(page_title="8. Asistente LangChain", page_icon="💬", layout="wide")
st.title("💬 Módulo 8 — Asistente LangChain")
st.caption("Agente LangChain 'tool-only': el LLM nunca ve los datos crudos, solo resultados agregados con pandas.")

if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("AI_LLM_API_KEY")):
    st.warning("No se detectó ANTHROPIC_API_KEY en el entorno. Configurala en ai_module/.env (AI_LLM_API_KEY) "
              "o como variable de entorno del sistema para usar este módulo.")
    st.stop()


@st.cache_resource(show_spinner="Cargando el dataset y construyendo el agente...")
def _agent():
    settings = Settings.from_env()
    df = load_dataset(settings.dataset_path)
    # Usa el proveedor configurado en ai_module/.env (p. ej. google_genai) en vez de Anthropic fijo.
    from ai_module.config.settings import AISettings
    from ai_module.services.llm_factory import build_chat_model
    ai_settings = AISettings()
    model = build_chat_model(ai_settings) if ai_settings.llm_enabled else None
    return build_agent(df, model=model, settings=settings), len(df)


try:
    agent, n_rows = _agent()
except DatasetError as exc:
    st.error(str(exc))
    st.stop()

st.caption(f"Dataset canónico cargado: {n_rows} ingresos.")

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("chat_display", [])

for role, text in st.session_state["chat_display"]:
    with st.chat_message(role):
        st.write(text)

question = st.chat_input("Preguntá algo sobre el dataset de ingresos hospitalarios...")
if question:
    st.session_state["chat_display"].append(("user", question))
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            answer, history = ask(agent, question, st.session_state["chat_history"])
        st.write(answer)
    st.session_state["chat_history"] = history
    st.session_state["chat_display"].append(("assistant", answer))

if st.button("🗑️ Reiniciar conversación"):
    st.session_state["chat_history"] = []
    st.session_state["chat_display"] = []
    st.rerun()
