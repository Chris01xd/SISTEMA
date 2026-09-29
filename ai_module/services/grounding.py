"""Numeric grounding check: every number the LLM writes must exist in the twin context
(or in the user's own input). This is what stops the model from inventing twin values."""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

# Integers that are almost always structural ("0 patients", "1 unit", "100%") rather than data.
_TRIVIAL = (0.0, 1.0, 100.0)
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?![\w])")


def _decimals(candidate: str) -> int:
    return len(candidate.split(".", 1)[1]) if "." in candidate else 0


def _interpretations(token: str) -> list[tuple[float, int]]:
    """'1.234' / '1,234' / '12,5' are ambiguous between locales: keep every reading, each
    with the number of decimals it was written with."""
    readings: set[tuple[float, int]] = set()
    for candidate in (token, token.replace(",", "."), token.replace(",", ""), token.replace(".", "").replace(",", ".")):
        try:
            readings.add((float(candidate), _decimals(candidate)))
        except ValueError:
            continue
    return sorted(readings)


def numbers_in_text(text: str) -> list[tuple[str, list[tuple[float, int]]]]:
    return [(m.group(0), _interpretations(m.group(0))) for m in _NUMBER.finditer(text)]


def collect_reference_values(obj: Any, key: str = "") -> list[float]:
    """Every number in a JSON-like structure (values, numeric keys, numbers inside strings),
    plus the derived forms a correct answer may legitimately use: a fraction as a percent,
    and hours as days."""
    values: list[float] = []
    if isinstance(obj, bool):
        return values
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            values.append(float(obj))
            if 0 <= obj <= 1:
                values.append(float(obj) * 100)
            if "hour" in key:
                values.append(float(obj) / 24)
        return values
    if isinstance(obj, str):
        for _, readings in numbers_in_text(obj):
            values.extend(value for value, _ in readings)
        return values
    if isinstance(obj, dict):
        for k, v in obj.items():
            values.extend(collect_reference_values(str(k)))
            values.extend(collect_reference_values(v, str(k)))
        return values
    if isinstance(obj, (list, tuple)):
        for item in obj:
            values.extend(collect_reference_values(item, key))
    return values


def _matches(value: float, decimals: int, references: Iterable[float]) -> bool:
    """A number written with d decimals must be a rounding of some reference value
    (|value - ref| <= 0.5 * 10^-d): "86%" matches 86.4, "0.73" matches 0.7291, but an
    invented "0.999123" matches nothing."""
    tolerance = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(value - ref) <= tolerance for ref in references)


def ungrounded_numbers(texts: Iterable[str], *sources: Any) -> list[str]:
    """Numbers in ``texts`` that match nothing in ``sources`` (sorted, de-duplicated)."""
    references = sorted(set(_TRIVIAL).union(*(collect_reference_values(s) for s in sources)))
    missing: set[str] = set()
    for text in texts:
        for token, readings in numbers_in_text(text):
            if not any(_matches(value, decimals, references) for value, decimals in readings):
                missing.add(token)
    return sorted(missing)
