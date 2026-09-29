"""Spanish display labels shared by the DOCX/PDF/XLSX report generators.

Simulation data keeps its English field names everywhere else (WebSocket
payloads, the frontend, the DB) — this module only maps them to Spanish text
at the point reports are rendered, so nothing else needs to change.
"""
from __future__ import annotations

COLUMN_LABELS_ES: dict[str, str] = {
    "tick": "Tick",
    "patients_waiting": "Pacientes en Espera",
    "patients_in_treatment": "Pacientes en Tratamiento",
    "patients_icu": "Pacientes en UCI",
    "patients_ward": "Pacientes en Hospitalización",
    "patients_deceased": "Pacientes Fallecidos",
    "patients_discharged": "Pacientes Dados de Alta",
    "beds_total": "Camas Totales",
    "beds_occupied": "Camas Ocupadas",
    "staff_total": "Personal Total",
    "staff_active": "Personal Activo",
    "total_patients_seen": "Total de Pacientes Atendidos",
    "saturation_index": "Índice de Saturación",
    "status_band": "Estado",
    "staff_fatigue": "Fatiga del Personal",
}

STATUS_LABELS_ES: dict[str, str] = {
    "GREEN": "VERDE",
    "YELLOW": "AMARILLO",
    "RED": "ROJO",
}

# Bare hex (no leading "#") so each format can prefix what its library
# expects — ReportLab wants "#rrggbb", python-docx shading wants "rrggbb",
# openpyxl fills want the ARGB form "FFrrggbb". Matches the frontend's
# status.green/yellow/red in tailwind.config.js and statusColors.ts.
STATUS_COLORS_HEX: dict[str, str] = {
    "GREEN": "16a34a",
    "YELLOW": "eab308",
    "RED": "dc2626",
}


def column_label(name: str) -> str:
    return COLUMN_LABELS_ES.get(name, name.replace("_", " ").strip().title())


def format_value(column: str, value: object) -> str:
    if column == "status_band" and isinstance(value, str):
        return STATUS_LABELS_ES.get(value, value)
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
