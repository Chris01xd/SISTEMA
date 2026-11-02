"""Downloadable charts and tables.

Every chart shown on screen gets PNG HD / SVG / HTML / Word / Excel buttons, every table gets
CSV / Excel / Word / PNG buttons, and `build_assets_zip` bundles all of them (plus the report files).
Excel and Word files share the look defined in backend/app/reports/office_style.py."""
from __future__ import annotations

import io
import re
import zipfile
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from lib.paths import bootstrap

bootstrap()

from backend.app.reports.office_style import (  # noqa: E402
    HEADER_HEX, ZEBRA_HEX, add_docx_title, clean_table, fmt_value, generated_stamp, new_document,
    style_excel_table, table_to_docx, table_to_xlsx, usable_width, wrap, write_excel_title,
)

EXPORT_W, EXPORT_H, EXPORT_SCALE = 1600, 900, 3  # 4800x2700 px PNG, suitable for print / slides
TABLE_IMG_MAX_ROWS = 60  # longer tables are cut in the PNG (CSV / Excel / Word always carry every row)

MIME = {
    "png": "image/png", "svg": "image/svg+xml", "html": "text/html", "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_PLOT_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "width": EXPORT_W, "height": EXPORT_H, "scale": EXPORT_SCALE},
}


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower() or "figura"


def _title_of(fig: go.Figure, default: str = "Gráfico") -> str:
    return fig.layout.title.text if fig.layout.title and fig.layout.title.text else default


# --------------------------------------------------------------------------------- charts


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
        if fmt in ("docx", "xlsx"):
            return _render_office(fig.to_json(), fmt)
        return _render(fig.to_json(), fmt)
    except Exception:  # kaleido missing / no chromium: the HTML export and the modebar camera still work
        return None


def figure_data(fig: go.Figure) -> pd.DataFrame:
    """The numbers behind a chart, one row per point (bars / lines) or the matrix itself (heatmaps)."""
    rows: list[dict[str, Any]] = []
    for i, tr in enumerate(fig.data):
        name = tr.name or f"serie {i + 1}"
        if tr.type == "heatmap":
            z = [list(r) for r in tr.z]
            xs = list(tr.x) if tr.x is not None else list(range(len(z[0]) if z else 0))
            ys = list(tr.y) if tr.y is not None else list(range(len(z)))
            return pd.DataFrame(z, index=pd.Index([str(y) for y in ys], name=" "), columns=[str(x) for x in xs]).reset_index()
        x = list(tr.x) if getattr(tr, "x", None) is not None else []
        y = list(tr.y) if getattr(tr, "y", None) is not None else []
        if not y and not x:
            continue
        horizontal = getattr(tr, "orientation", None) == "h"
        cat, val = (y, x) if horizontal else (x, y)
        if not cat:
            cat = list(range(1, len(val) + 1))
        err = getattr(tr, "error_y", None)
        err_vals = list(err.array) if err is not None and err.array is not None else None
        for k, (c, v) in enumerate(zip(cat, val)):
            row = {"serie": name, "eje X" if not horizontal else "categoría": c, "valor": v}
            if err_vals is not None and k < len(err_vals):
                row["± desviación"] = err_vals[k]
            rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, max_entries=256)
def _render_office(fig_json: str, fmt: str) -> bytes:
    fig = pio.from_json(fig_json)
    title = _title_of(fig)
    png = _render(fig_json, "png")
    data = figure_data(fig)
    return _chart_docx(title, png, data) if fmt == "docx" else _chart_xlsx(title, png, data)


def _chart_docx(title: str, png: bytes, data: pd.DataFrame) -> bytes:
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    doc = new_document(landscape=True)
    add_docx_title(doc, title, generated_stamp())
    doc.add_picture(io.BytesIO(png), width=int(usable_width(doc.sections[-1]) * 0.92))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _chart_xlsx(title: str, png: bytes, data: pd.DataFrame) -> bytes:
    """Sheet 'Gráfico' with the HD image, sheet 'Datos' with the plotted values in a formatted table."""
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage

    wb = Workbook()
    ws = wb.active
    ws.title = "Gráfico"
    for col in "ABCDEFGHIJKLMNOP":
        ws.column_dimensions[col].width = 12
    start = write_excel_title(ws, title, generated_stamp(), 14)
    img = XLImage(io.BytesIO(png))
    img.width, img.height = 1120, 630  # displayed size; the embedded PNG stays at full resolution
    ws.add_image(img, f"A{start}")
    ws.sheet_view.showGridLines = False
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = ws.page_setup.fitToHeight = 1
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    if not data.empty:
        ds = wb.create_sheet("Datos")
        clean = clean_table(data)
        header_row = write_excel_title(ds, f"Datos — {title}", "Valores usados para dibujar el gráfico", len(clean.columns))
        for i, row in enumerate(clean.itertuples(index=False)):
            for j, v in enumerate(row):
                ds.cell(row=header_row + 1 + i, column=1 + j, value=v)
        style_excel_table(ds, clean, header_row)
        ds.print_title_rows = f"{header_row}:{header_row}"
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _stretch(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """`width="stretch"` on recent Streamlit, `use_container_width` on older ones."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except (TypeError, st.errors.StreamlitAPIException):
        return fn(*args, use_container_width=True, **kwargs)


def _buttons(items: list[tuple[str, bytes | None, str]], key: str) -> None:
    """Row of download buttons; formats that could not be generated are skipped."""
    items = [it for it in items if it[1]]
    if not items:
        return
    cols = st.columns(len(items) + max(0, 6 - len(items)))
    for col, (label, data, fname) in zip(cols, items):
        ext = fname.rsplit(".", 1)[1]
        col.download_button(f"⬇ {label}", data, file_name=fname, mime=MIME[ext], key=f"{key}_{ext}")


def chart(fig: go.Figure, name: str | None = None, key: str | None = None, **kwargs: Any) -> None:
    """Plotly chart + PNG HD, SVG, HTML, Word (image) and Excel (image + data) downloads."""
    name = name or _title_of(fig, "grafico")
    key = key or f"chart_{slug(name)}"
    cfg = {**_PLOT_CONFIG, "toImageButtonOptions": {**_PLOT_CONFIG["toImageButtonOptions"], "filename": slug(name)}}
    _stretch(st.plotly_chart, fig, config=cfg, key=key, **kwargs)
    s = slug(name)
    _buttons([(label, figure_bytes(fig, fmt), f"{s}.{fmt}") for fmt, label in
              [("png", "PNG"), ("svg", "SVG"), ("html", "HTML"), ("docx", "Word"), ("xlsx", "Excel")]], key)


# --------------------------------------------------------------------------------- tables


def _table_figure(df: pd.DataFrame, title: str) -> go.Figure:
    """Plotly table styled like the Word / Excel exports (header dark, zebra rows, numbers right-aligned)."""
    total = len(df)
    shown = df.head(TABLE_IMG_MAX_ROWS)
    numeric = [bool(len(shown[c].dropna())) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in shown[c].dropna())
               for c in shown.columns]
    widths = [min(40, max(len(str(c)), *(len(fmt_value(v)) for v in shown[c]), 6)) for c in shown.columns] if len(shown) else \
        [max(len(str(c)), 6) for c in shown.columns]
    cells = [[wrap(fmt_value(v), w) for v in shown[c]] for c, w in zip(shown.columns, widths)]
    headers = [f"<b>{wrap(str(c), max(w, 10))}</b>" for c, w in zip(shown.columns, widths)]
    lines = [max([1, *(cells[j][i].count("<br>") + 1 for j in range(len(cells)))]) for i in range(len(shown))]
    head_lines = max([1, *(h.count("<br>") + 1 for h in headers)])
    row_px, line_px = 30, 19
    body_h = sum(row_px + (n - 1) * line_px for n in lines)
    width = int(min(2600, max(900, sum(w * 10 + 30 for w in widths))))
    fills = [["#FFFFFF" if i % 2 == 0 else f"#{ZEBRA_HEX}" for i in range(len(shown))]] * len(shown.columns)
    fig = go.Figure(go.Table(
        columnwidth=widths,
        header=dict(values=headers, fill_color=f"#{HEADER_HEX}", font=dict(color="white", size=14), align="center",
                    line_color="#D1D5DB", height=row_px + (head_lines - 1) * line_px),
        cells=dict(values=cells, fill_color=fills, font=dict(color="#111827", size=13), height=row_px,
                   align=["right" if n else "left" for n in numeric], line_color="#D1D5DB"),
    ))
    note = f"Primeras {TABLE_IMG_MAX_ROWS} de {total} filas (el CSV / Excel / Word incluyen todas)" if total > TABLE_IMG_MAX_ROWS else None
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>" + (f"<br><sup>{note}</sup>" if note else ""), x=0.01, font=dict(size=20, color="#2563EB")),
        width=width, height=110 + (row_px + (head_lines - 1) * line_px) + body_h + (20 if note else 0),
        margin=dict(t=80 if not note else 95, b=20, l=20, r=20), paper_bgcolor="white",
    )
    return fig


@st.cache_data(show_spinner=False, max_entries=512)
def _table_bytes_cached(df: pd.DataFrame, fmt: str, title: str) -> bytes | None:
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8-sig")  # BOM so Excel opens accents correctly
    if fmt == "xlsx":
        return table_to_xlsx(df, title, sheet_name=title)
    if fmt == "docx":
        return table_to_docx(df, title)
    try:
        _start_kaleido()
        return _table_figure(df, title).to_image(format="png", scale=2)
    except Exception:  # kaleido / Chrome unavailable
        return None


def table_bytes(df: pd.DataFrame, fmt: str, title: str = "Tabla") -> bytes | None:
    return _table_bytes_cached(clean_table(df), fmt, title)


def table(df: pd.DataFrame, name: str, key: str | None = None, **kwargs: Any) -> None:
    """DataFrame + CSV / Excel / Word / PNG downloads (Excel and Word come fully formatted)."""
    key = key or f"table_{slug(name)}"
    _stretch(st.dataframe, df, **kwargs)
    s = slug(name)
    _buttons([(label, table_bytes(df, fmt, name), f"{s}.{fmt}") for fmt, label in
              [("xlsx", "Excel"), ("docx", "Word"), ("png", "PNG"), ("csv", "CSV")]], key)


def build_assets_zip(figures: dict[str, go.Figure], tables: dict[str, pd.DataFrame], extra_files: dict[str, bytes] | None = None) -> bytes:
    """ZIP with graficos/ (PNG HD + SVG + HTML + DOCX + XLSX), tablas/ (CSV + XLSX + DOCX + PNG) and any extra files."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, fig in figures.items():
            for fmt in ("png", "svg", "html", "docx", "xlsx"):
                data = figure_bytes(fig, fmt)
                if data:
                    z.writestr(f"graficos/{slug(name)}.{fmt}", data)
        for name, df in tables.items():
            for fmt in ("csv", "xlsx", "docx", "png"):
                data = table_bytes(df, fmt, name)
                if data:
                    z.writestr(f"tablas/{slug(name)}.{fmt}", data)
        for fname, data in (extra_files or {}).items():
            z.writestr(fname, data)
    return buf.getvalue()
