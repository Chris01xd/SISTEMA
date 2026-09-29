"""Tiny language heuristic for response labels (Spanish vs English); no external dependency."""
from __future__ import annotations

import re

_SPANISH = re.compile(r"[áéíóúñ¿¡]|\b(el|la|los|las|de|que|qué|cu[aá]l|cu[aá]nt[oa]s?|c[oó]mo|pasa|si|camas|hospital(es)?|pacientes|estado|es|en|con|por|para|una?)\b", re.I)
_ENGLISH = re.compile(r"\b(the|what|which|how|many|is|are|of|if|beds|patients|state|with|for|an?)\b", re.I)


def detect_language(text: str) -> str:
    """Returns "es" or "en" (default "es" on ties, the team's working language)."""
    return "en" if len(_ENGLISH.findall(text)) > len(_SPANISH.findall(text)) else "es"
