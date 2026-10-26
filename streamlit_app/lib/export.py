"""Downloadable charts and tables: every figure/table shown on screen gets HD PNG / SVG / HTML (charts) or CSV / Excel (tables) buttons,
and `build_assets_zip` bundles all of them (plus the PDF and Excel report) in one file."""
from __future__ import annotations

import io
import re
import zipfile
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

EXPORT_W, EXPORT_H, EXPORT_SCALE = 1600, 900, 3  # 4800x2700 px PNG, suitable for print / slides

_PLOT_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "width": EXPORT_W, "height": EXPORT_H, "scale": EXPORT_SCALE},
}


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "figura"


def _export_layout(fig: go.Figure) -> go.Figure:
    """Copy of `fig` sized for export (the on-screen height is tuned for the page, not for a standalone image)."""
    exp = go.Figure(fig)
    exp.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white", width=EXPORT_W,
                      height=max(EXPORT_H, int(fig.layout.height or 0)), margin=dict(t=80, b=80, l=80, r=80), font=dict(size=18))
    return exp


@st.cache_resource(show_spinner=False)
def _start_kaleido() -> bool:
    """Keep one Chrome alive for all exports (~0.2 s per image instead of ~3 s)."""
    try:
        import kaleido

        kaleido.start_sync_server(silence_warnings=True)
    except Exception:  # kaleido<1 (bundled engine) or not installed: to_image still works / fails per call
        return False
    return True


@st.cache_data(show_spinner=False, max_entries=256)
def _render(fig_json: str, fmt: str) -> bytes:
    _start_kaleido()
    exp = _export_layout(pio.from_json(fig_json))
    if fmt == "png":
        return exp.to_image(format="png", scale=EXPORT_SCALE)
    if fmt == "svg":
        return exp.to_image(format="svg")
    return exp.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")


def figure_bytes(fig: go.Figure, fmt: str) -> bytes | None:
    try:
        return _render(fig.to_json(), fmt)
    except Exception:  # kaleido missing / no chromium: the HTML export and the modebar camera still work
        return None


def _stretch(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """`width="stretch"` on recent Streamlit, `use_container_width` on older ones."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except (TypeError, st.errors.StreamlitAPIException):
        return fn(*args, use_container_width=True, **kwargs)


def chart(fig: go.Figure, name: str | None = None, key: str | None = None, **kwargs: Any) -> None:
    """Plotly chart + HD PNG, vector SVG and interactive HTML downloads."""
    name = name or (fig.layout.title.text if fig.layout.title and fig.layout.title.text else "grafico")
    key = key or f"chart_{slug(name)}"
    cfg = {**_PLOT_CONFIG, "toImageButtonOptions": {**_PLOT_CONFIG["toImageButtonOptions"], "filename": slug(name)}}
    _stretch(st.plotly_chart, fig, config=cfg, key=key, **kwargs)
    cols = st.columns([1, 1, 1, 3])
    for col, (fmt, label, mime) in zip(cols, [("png", "PNG HD", "image/png"), ("svg", "SVG", "image/svg+xml"), ("html", "HTML", "text/html")]):
        data = figure_bytes(fig, fmt)
        if data:
            col.download_button(f"⬇ {label}", data, file_name=f"{slug(name)}.{fmt}", mime=mime, key=f"{key}_{fmt}")


def table_bytes(df: pd.DataFrame, fmt: str) -> bytes:
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8-sig")  # BOM so Excel opens accents correctly
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def table(df: pd.DataFrame, name: str, key: str | None = None, **kwargs: Any) -> None:
    """DataFrame + CSV / Excel downloads."""
    key = key or f"table_{slug(name)}"
    _stretch(st.dataframe, df, **kwargs)
    c1, c2, _ = st.columns([1, 1, 4])
    c1.download_button("⬇ CSV", table_bytes(df, "csv"), file_name=f"{slug(name)}.csv", mime="text/csv", key=f"{key}_csv")
    c2.download_button("⬇ Excel", table_bytes(df, "xlsx"), file_name=f"{slug(name)}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"{key}_xlsx")


def build_assets_zip(figures: dict[str, go.Figure], tables: dict[str, pd.DataFrame], extra_files: dict[str, bytes] | None = None) -> bytes:
    """ZIP with graficos/ (PNG HD + SVG + HTML), tablas/ (CSV + XLSX) and any extra files (PDF, Excel report)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, fig in figures.items():
            for fmt in ("png", "svg", "html"):
                data = figure_bytes(fig, fmt)
                if data:
                    z.writestr(f"graficos/{slug(name)}.{fmt}", data)
        for name, df in tables.items():
            z.writestr(f"tablas/{slug(name)}.csv", table_bytes(df, "csv"))
            z.writestr(f"tablas/{slug(name)}.xlsx", table_bytes(df, "xlsx"))
        for fname, data in (extra_files or {}).items():
            z.writestr(fname, data)
    return buf.getvalue()
