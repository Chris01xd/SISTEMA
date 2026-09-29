"""START/SALT-inspired rule-based triage. Primary and authoritative per
docs/ARCHITECTURE.md section 7 — patient safety requires this logic to stay
transparent and clinically auditable; ML only re-ranks within the queue that
this function produces, it never overrides it outright."""
from __future__ import annotations

from app.models.clinical import SeverityLevel


def classify_severity(can_walk: bool, breathing: bool, breaths_per_min: int | None,
                       radial_pulse_present: bool, mental_status_follows_commands: bool) -> SeverityLevel:
    """Simplified START algorithm decision tree."""
    if can_walk:
        return SeverityLevel.MINOR

    if not breathing:
        return SeverityLevel.EXPECTANT

    if breaths_per_min is not None and breaths_per_min > 30:
        return SeverityLevel.CRITICAL

    if not radial_pulse_present:
        return SeverityLevel.CRITICAL

    if not mental_status_follows_commands:
        return SeverityLevel.CRITICAL

    return SeverityLevel.DELAYED


def base_priority_score(severity: SeverityLevel, wait_minutes: int) -> float:
    base = {
        SeverityLevel.CRITICAL: 90,
        SeverityLevel.IMMEDIATE: 60,
        SeverityLevel.DELAYED: 30,
        SeverityLevel.MINOR: 10,
        SeverityLevel.EXPECTANT: 0,
    }[severity]
    return base + min(wait_minutes, 120) * 0.15
