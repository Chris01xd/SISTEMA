"""Módulo 7 — Pruebas estadísticas inferenciales robustas, CRISP-DM fase 5."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.paths import bootstrap

bootstrap()

from lib.export import table  # noqa: E402
from lib.pipeline import current_output  # noqa: E402
from lib.report import render_export_buttons  # noqa: E402

st.set_page_config(page_title="7. Pruebas Estadísticas", page_icon="📊", layout="wide")
st.title("📊 Módulo 7 — Pruebas estadísticas inferenciales robustas")

out = current_output()
if out["status"] != "success":
    st.error("Ejecutá el análisis desde la portada (Home) primero.")
    st.stop()

st_out = out["statistical_tests"]
st.caption(f"α = {st_out['alpha']} · {st_out['bootstrap_resamples']} remuestreos bootstrap · "
          f"{st_out['cv_folds']} folds · {st_out['holdout_size']} casos de prueba · mejor modelo: {st_out['best_model']}")

st.subheader("1. Friedman + Nemenyi (todos los modelos, sobre los folds de CV)")
fr = st_out["friedman"]
c1, c2, c3 = st.columns(3)
c1.metric("χ² (Friedman)", fr["statistic"])
c2.metric("p-valor", fr["p_value"])
c3.metric("¿Diferencia significativa?", "Sí" if fr["significant"] else "No")
st.write("Rango medio por modelo (menor = mejor):", st_out["nemenyi"]["mean_ranks"])
st.write(f"Diferencia crítica de Nemenyi: {st_out['nemenyi']['critical_difference']:.3f} rangos")
if st_out["nemenyi"]["different_from_best"]:
    st.write("Modelos cuyo rango difiere del ganador más allá de la diferencia crítica:",
            st_out["nemenyi"]["different_from_best"])

st.subheader("2. Mejor modelo vs. cada uno de los demás (Holm-corregido)")
st.caption("Nadeau-Bengio: t pareado corregido sobre folds de CV · DeLong: diferencia de ROC-AUC en el hold-out · "
          "McNemar: diferencia de aciertos al umbral 0.5 en el hold-out.")
pairwise_rows = []
for p in st_out["pairwise_vs_best"]:
    pairwise_rows.append({
        "modelo": p["model"],
        "Nadeau-Bengio p (Holm)": p["nadeau_bengio"].get("p_value_holm"), "NB sig.": p["nadeau_bengio"].get("significant"),
        "DeLong p (Holm)": p["delong"].get("p_value_holm"), "DeLong sig.": p["delong"].get("significant"),
        "McNemar p (Holm)": p["mcnemar"].get("p_value_holm"), "McNemar sig.": p["mcnemar"].get("significant"),
    })
table(pd.DataFrame(pairwise_rows), '2. Mejor modelo vs. cada uno de los demás (Holm-corregido)')
st.info(out["interpretations"]["statistical_tests"])

st.subheader("3. Intervalos de confianza 95% (bootstrap, hold-out)")
ci_rows = [{"modelo": name, "ROC-AUC 95% CI": c["roc_auc"], "F1 95% CI": c["f1"]} for name, c in st_out["bootstrap_ci_95"].items()]
table(pd.DataFrame(ci_rows), '3. Intervalos de confianza 95% (bootstrap, hold-out)')
if "bootstrap_ci" in out["interpretations"]:
    st.info(out["interpretations"]["bootstrap_ci"])

st.subheader("Evidencia global")
st.write(f"Modelos significativamente peores que el ganador: {st_out['summary']['models_significantly_worse_than_best'] or 'ninguno'}")
st.write(f"Nivel de evidencia: **{st_out['summary']['evidence']}**")

st.divider()
render_export_buttons(out, key="stats")
