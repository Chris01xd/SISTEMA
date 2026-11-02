"""Shared look for every Excel / Word export (single tables, charts and the full report).

Pure openpyxl / python-docx, no Streamlit or FastAPI, so the backend report and the
Streamlit download buttons produce identical-looking files:
dark header, zebra rows, thin gray borders, numbers right-aligned with a fixed number
of decimals (scientific notation for tiny p-values), widths fitted to the content,
frozen header and autofilter in Excel, repeated header row and landscape pages in Word."""
from __future__ import annotations

import io
import math
import textwrap
from datetime import datetime
from typing import Any

import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

HEADER_HEX = "1F2937"   # gray-800
ZEBRA_HEX = "F3F4F6"    # gray-100
BORDER_HEX = "D1D5DB"   # gray-300
ACCENT_HEX = "2563EB"   # blue-600 (titles)
MUTED_HEX = "6B7280"    # gray-500 (captions)
FONT = "Calibri"
DECIMALS = 4

_THIN = Side(style="thin", color=BORDER_HEX)
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)


# --------------------------------------------------------------------------------- values


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):  # lists / dicts
        return False


def _compact(v: Any) -> str:
    """Values inside dicts / lists (hyper-parameters, CIs): 10 instead of 10.0000, 0.03 instead of 0.0300."""
    if isinstance(v, float) and not isinstance(v, bool):
        return f"{v:.4g}"
    return fmt_value(v)


def fmt_value(v: Any, decimals: int = DECIMALS) -> str:
    """Text shown in Word / PNG tables (Excel keeps real numbers, see `excel_number_format`)."""
    if _is_missing(v):
        return "—"
    if isinstance(v, bool):
        return "Sí" if v else "No"
    if isinstance(v, int):
        return f"{v:,}".replace(",", " ") if abs(v) >= 10_000 else str(v)
    if isinstance(v, float):
        if math.isinf(v):
            return "∞" if v > 0 else "-∞"
        if v != 0 and abs(v) < 10 ** -(decimals - 1):
            return f"{v:.2e}"
        if v.is_integer() and abs(v) >= 1000:
            return f"{int(v):,}".replace(",", " ")
        return f"{v:.{decimals}f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_compact(x)}" for k, x in v.items()) or "—"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_compact(x) for x in v) + "]"
    return str(v)


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Copy safe for every format: a meaningful index becomes a column, dict/list cells become text,
    numpy scalars become Python ones, booleans become Sí/No."""
    out = df.copy()
    if not isinstance(out.index, pd.RangeIndex) or out.index.name is not None:
        out = out.reset_index().rename(columns={"index": out.index.name or "concepto"})
    out.columns = [str(c) for c in out.columns]

    def cell(v: Any) -> Any:
        if hasattr(v, "item") and not isinstance(v, (str, bytes)):
            try:
                v = v.item()
            except (ValueError, AttributeError):
                pass
        if isinstance(v, bool):
            return "Sí" if v else "No"
        if isinstance(v, (dict, list, tuple)):
            return fmt_value(v)
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    for col in out.columns:
        out[col] = out[col].astype(object).map(cell)
    return out.reset_index(drop=True)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def excel_number_format(v: Any) -> str:
    if isinstance(v, int):
        return "#,##0"
    if isinstance(v, float):
        if v != 0 and abs(v) < 10 ** -(DECIMALS - 1):
            return "0.00E+00"
        if v.is_integer() and abs(v) >= 1000:
            return "#,##0"
        return "0." + "0" * DECIMALS
    return "General"


def _numeric_columns(df: pd.DataFrame) -> set[str]:
    """Columns whose non-empty cells are all numbers (right-aligned everywhere)."""
    cols = set()
    for c in df.columns:
        vals = [v for v in df[c] if not _is_missing(v)]
        if vals and all(_is_number(v) for v in vals):
            cols.add(c)
    return cols


def _header_need(header: str) -> int:
    """Characters a header needs: short ones on one line, long ones over ~3 lines, never splitting a word."""
    words = str(header).split() or [""]
    longest = max(len(w) for w in words) + 2  # bold text is wider
    return len(str(header)) + 1 if len(str(header)) <= 16 else max(longest, math.ceil(len(str(header)) / 3))


def _col_widths_chars(df: pd.DataFrame, max_chars: int = 60) -> list[int]:
    return [min(max_chars, max([_header_need(c), *(len(fmt_value(v)) for v in df[c]), 6])) for c in df.columns]


def _header_lines(header: str, width_chars: float) -> int:
    return len(textwrap.wrap(str(header), width=max(int(width_chars), 4), break_long_words=False)) or 1


def generated_stamp() -> str:
    return datetime.now().strftime("Generado el %d/%m/%Y %H:%M")


# --------------------------------------------------------------------------------- Excel


def style_excel_table(ws: Worksheet, df: pd.DataFrame, header_row: int, first_col: int = 1,
                      autofilter: bool = True, freeze: bool = True) -> None:
    """Format a table already written at (header_row, first_col) — e.g. by `DataFrame.to_excel(startrow=header_row-1)`."""
    n_rows, n_cols = len(df), len(df.columns)
    numeric = _numeric_columns(df)
    for j, col in enumerate(df.columns):
        c = ws.cell(row=header_row, column=first_col + j)
        c.value = str(col)
        c.font = Font(name=FONT, bold=True, color="FFFFFF", size=11)
        c.fill = PatternFill("solid", fgColor=HEADER_HEX)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = _BORDER
    for i in range(n_rows):
        fill = PatternFill("solid", fgColor=ZEBRA_HEX) if i % 2 else None
        for j, col in enumerate(df.columns):
            c = ws.cell(row=header_row + 1 + i, column=first_col + j)
            v = df.iat[i, j]
            if _is_missing(v):
                c.value = "—"
                c.alignment = Alignment(horizontal="center", vertical="center")
            else:
                if isinstance(v, (dict, list, tuple)):
                    c.value = fmt_value(v)
                c.number_format = excel_number_format(c.value)
                is_num = col in numeric
                c.alignment = Alignment(horizontal="right" if is_num else "left", vertical="center", wrap_text=not is_num, indent=1)
            c.font = Font(name=FONT, size=10)
            c.border = _BORDER
            if fill:
                c.fill = fill
    widths = [max(10, min(w + 4, 62)) for w in _col_widths_chars(df)]
    for j, w in enumerate(widths):
        ws.column_dimensions[get_column_letter(first_col + j)].width = w
    lines = max(_header_lines(c, w - 2) for c, w in zip(df.columns, widths))
    ws.row_dimensions[header_row].height = 16 * lines + 8
    if freeze:
        ws.freeze_panes = ws.cell(row=header_row + 1, column=first_col)
    if autofilter and n_rows:
        ws.auto_filter.ref = (f"{get_column_letter(first_col)}{header_row}:"
                              f"{get_column_letter(first_col + n_cols - 1)}{header_row + n_rows}")
    ws.sheet_view.showGridLines = False


def write_excel_title(ws: Worksheet, title: str, subtitle: str | None, n_cols: int, row: int = 1) -> int:
    """Title (+ optional wrapped subtitle) merged across the table width. Returns the next free row."""
    span = max(n_cols, 4)
    ws.cell(row=row, column=1, value=title).font = Font(name=FONT, bold=True, size=14, color=ACCENT_HEX)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
    ws.row_dimensions[row].height = 24
    row += 1
    if subtitle:
        c = ws.cell(row=row, column=1, value=subtitle)
        c.font = Font(name=FONT, italic=True, size=10, color=MUTED_HEX)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=span)
        approx_width = sum(ws.column_dimensions[get_column_letter(k)].width or 12 for k in range(1, span + 1)) or 60
        ws.row_dimensions[row].height = max(18, 15 * math.ceil(len(subtitle) / max(approx_width * 1.1, 40)))
        row += 1
    return row + 1  # one blank spacer row


def sheet_print_setup(ws: Worksheet, n_cols: int) -> None:
    ws.page_setup.orientation = "landscape" if n_cols > 5 else "portrait"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_options.horizontalCentered = True


def table_to_xlsx(df: pd.DataFrame, title: str, sheet_name: str = "Tabla", note: str | None = None) -> bytes:
    from openpyxl import Workbook

    df = clean_table(df)
    wb = Workbook()
    ws = wb.active
    ws.title = safe_sheet_name(sheet_name)
    for j, w in enumerate(_col_widths_chars(df)):  # widths first so the subtitle height estimate is right
        ws.column_dimensions[get_column_letter(1 + j)].width = max(10, min(w + 3, 62))
    header_row = write_excel_title(ws, title, " · ".join(x for x in (note, generated_stamp()) if x), len(df.columns))
    for i, row in enumerate(df.itertuples(index=False)):
        for j, v in enumerate(row):
            ws.cell(row=header_row + 1 + i, column=1 + j, value=None if _is_missing(v) else v)
    style_excel_table(ws, df, header_row)
    ws.print_title_rows = f"{header_row}:{header_row}"
    sheet_print_setup(ws, len(df.columns))
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


_INVALID_SHEET_CHARS = set("[]:*?/\\")


def safe_sheet_name(name: str, used: set[str] | None = None) -> str:
    base = "".join(ch for ch in name if ch not in _INVALID_SHEET_CHARS).strip()[:31] or "Hoja"
    out, k = base, 2
    while used is not None and out in used:
        out = f"{base[:28]}_{k}"
        k += 1
    if used is not None:
        used.add(out)
    return out


# --------------------------------------------------------------------------------- Word


def _shade(cell: Any, hex_color: str) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tc_pr = cell._tc.get_or_add_tcPr()
    for old in tc_pr.findall(qn("w:shd")):
        tc_pr.remove(old)
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def _table_borders(table: Any) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tbl_pr = table._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), BORDER_HEX)
        borders.append(el)
    tbl_pr.append(borders)


def _repeat_header(row: Any) -> None:
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tr_pr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    tr_pr.append(el)
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def new_document(landscape: bool = False) -> Any:
    from docx import Document
    from docx.shared import Cm, Pt

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(4)
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)  # A4 (python-docx default is US Letter)
    if landscape:
        set_landscape(sec)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(1.8))
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    return doc


def set_landscape(section: Any) -> None:
    from docx.enum.section import WD_ORIENT

    if section.orientation != WD_ORIENT.LANDSCAPE:
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = section.page_height, section.page_width


def usable_width(section: Any) -> int:
    return section.page_width - section.left_margin - section.right_margin


def add_docx_title(doc: Any, title: str, subtitle: str | None = None, level: int = 1) -> None:
    from docx.shared import Pt, RGBColor

    h = doc.add_heading(title, level=level)
    for r in h.runs:
        r.font.name = FONT
        r.font.color.rgb = RGBColor.from_string(ACCENT_HEX if level <= 1 else HEADER_HEX)
    if subtitle:
        p = doc.add_paragraph()
        run = p.add_run(subtitle)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor.from_string(MUTED_HEX)


def add_docx_table(doc: Any, df: pd.DataFrame, font_pt: float | None = None) -> Any:
    """Styled table that fits the current section width (shrinks the font for wide tables)."""
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, RGBColor

    df = clean_table(df)
    n_cols = len(df.columns)
    numeric = _numeric_columns(df)
    font_pt = font_pt or (10 if n_cols <= 5 else 9 if n_cols <= 8 else 8 if n_cols <= 11 else 7)
    table = doc.add_table(rows=1, cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    _table_borders(table)

    chars = [max(4, w) for w in _col_widths_chars(df, max_chars=45)]
    total = usable_width(doc.sections[-1])
    widths = [int(total * c / sum(chars)) for c in chars]

    def write(cell: Any, text: str, *, header: bool, right: bool) -> None:
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if header else WD_ALIGN_PARAGRAPH.RIGHT if right else WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.font.size = Pt(font_pt)
        run.font.name = FONT
        if header:
            run.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    hdr = table.rows[0]
    _repeat_header(hdr)
    for j, col in enumerate(df.columns):
        write(hdr.cells[j], str(col), header=True, right=False)
        _shade(hdr.cells[j], HEADER_HEX)
    for i, row in enumerate(df.itertuples(index=False)):
        cells = table.add_row().cells
        for j, v in enumerate(row):
            write(cells[j], fmt_value(v), header=False, right=df.columns[j] in numeric)
            if i % 2:
                _shade(cells[j], ZEBRA_HEX)
    for row in table.rows:
        for j, cell in enumerate(row.cells):
            cell.width = widths[j]
    return table


def table_to_docx(df: pd.DataFrame, title: str, note: str | None = None) -> bytes:
    doc = new_document(landscape=len(clean_table(df).columns) > 5)
    add_docx_title(doc, title, " · ".join(x for x in (note, generated_stamp()) if x))
    add_docx_table(doc, df)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def wrap(text: str, width: int) -> str:
    """Line breaks for fixed-width renderers (Plotly tables do not wrap on their own)."""
    return "<br>".join(textwrap.wrap(text, width=width, break_long_words=True)) or text
