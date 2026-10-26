"""Módulo 1 — Carga del dataset (datos reales), independiente de los demás."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from lib.paths import bootstrap

bootstrap()

from lib.export import table  # noqa: E402
from ai_module.digital_twin.store import TwinDataStore  # noqa: E402
from lib.pipeline import get_ai_settings, prepare_dataset  # noqa: E402

st.set_page_config(page_title="1. Carga de Datos", page_icon="📥", layout="wide")
st.title("📥 Módulo 1 — Carga del dataset (datos reales)")

st.markdown(
    """
**Fuente**: MIMIC-IV Clinical Database Demo v2.2 + MIMIC-IV-ED Demo v2.2 (PhysioNet, licencia
ODbL — acceso abierto, sin credenciales). 100 pacientes reales, de-identificados: 275
ingresos hospitalarios, 140 estancias en UCI y 222 visitas a urgencias con triaje.

Este módulo descarga los 5 archivos crudos (`admissions`, `patients`, `icustays`,
`edstays`, `triage`), verifica su **checksum SHA-256** publicado, y los limpia/transforma
a una tabla de una fila por ingreso, contando cada regla de limpieza para que sea
auditable contra los datos crudos.
"""
)

offline = st.checkbox("Modo offline (usar archivos ya descargados en ai_module/data/raw)", value=True)
if st.button("▶ Ejecutar carga y limpieza", type="primary"):
    st.cache_data.clear()
    result = prepare_dataset(offline)
    st.session_state["prepare_result"] = result

settings = get_ai_settings()
store = TwinDataStore(settings.processed_dir)

if "prepare_result" in st.session_state:
    r = st.session_state["prepare_result"]
    st.subheader("Tablas crudas leídas")
    table(pd.DataFrame([r["raw_counts"]]).T.rename(columns={0: "filas"}), 'Tablas crudas leídas')
else:
    st.info("Todavía no ejecutaste la carga en esta sesión — mostrando el último resultado ya persistido en disco.")

try:
    provenance = store.load_provenance()
    report = json.loads((settings.processed_dir / "data_report.json").read_text(encoding="utf-8"))
except Exception:
    st.warning("No hay datos procesados en disco todavía. Ejecutá la carga con el botón de arriba.")
    st.stop()

st.subheader("Procedencia (provenance.json)")
st.json(provenance.model_dump(), expanded=False)

st.subheader("Auditoría de limpieza (data_report.json)")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Filas crudas por tabla**")
    table(pd.DataFrame([report["raw_rows"]]).T.rename(columns={0: "filas"}), 'Auditoría de limpieza (data_report.json)')
    st.markdown("**Ingresos descartados**")
    table(pd.DataFrame([report["dropped"]]).T.rename(columns={0: "ingresos"}), 'Auditoría de limpieza (data_report.json) 2')
with col2:
    st.markdown("**Signos vitales anulados por estar fuera de rango fisiológico**")
    table(pd.DataFrame([report["vitals_nulled_out_of_range"]]).T.rename(columns={0: "valores anulados"}), 'Auditoría de limpieza (data_report.json) 3')
    st.metric("Temperaturas convertidas de °C a °F", report["temperature_celsius_converted"])
    st.metric("Horas de UCI recortadas a la estancia total", report["icu_hours_clipped_to_los"])

st.metric("Ingresos procesados (final)", f"{report['processed_rows']} ({report['processed_patients']} pacientes)")

st.divider()
st.subheader("Interpretación")
st.info(
    f"De los {sum(report['raw_rows'].values())} registros crudos combinados, quedan "
    f"**{report['processed_rows']} ingresos de {report['processed_patients']} pacientes** tras la limpieza: se "
    f"descartaron {report['dropped'].get('invalid_or_non_positive_stay', 0)} ingresos con horarios inválidos, "
    f"{report['dropped'].get('age_out_of_range', 0)} con edad fuera de 0–120 años y "
    f"{report['dropped'].get('duplicated_admission', 0)} duplicados. Se anularon "
    f"{sum(report['vitals_nulled_out_of_range'].values())} lecturas de signos vitales fuera de rango fisiológico "
    "plausible (posibles errores de registro) y se convirtieron "
    f"{report['temperature_celsius_converted']} temperaturas mal tipeadas en Celsius. Estas reglas están "
    "documentadas y contadas explícitamente (no son limpieza silenciosa), por lo que la tabla procesada es "
    "auditable contra los archivos crudos originales. Al ser una muestra pequeña de 100 pacientes, las cifras "
    "son ilustrativas del método, no representativas de un hospital real."
)
st.caption("Checksum SHA-256 del archivo procesado: " + provenance.processed_sha256)
