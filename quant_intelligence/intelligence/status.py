"""One data-quality status shared by every stage of the intelligence pipeline.

A stage that cannot trust its input returns a CRITICAL status naming the component and the reason; the next stage reads
it and stops (fail closed) instead of continuing on a default. WARNING means usable with a stated weakness.
`unavailable` lists fields that were not provided by the data source - they are reported as missing, never filled in.
"""
from __future__ import annotations

from dataclasses import dataclass

OK, WARNING, CRITICAL = "OK", "WARNING", "CRITICAL"
_ORDER = {OK: 0, WARNING: 1, CRITICAL: 2}


@dataclass(frozen=True)
class DataStatus:
    component: str
    level: str = OK
    reason: str = ""
    unavailable: tuple[str, ...] = ()

    def __post_init__(self):
        if self.level not in _ORDER:
            raise ValueError(f"unknown status level {self.level!r}")

    @property
    def ok(self) -> bool:
        """Usable (OK or WARNING). CRITICAL means a later stage must not proceed on this data."""
        return self.level != CRITICAL

    @property
    def clean(self) -> bool:
        return self.level == OK

    def to_dict(self) -> dict:
        return {"component": self.component, "level": self.level, "reason": self.reason, "unavailable": list(self.unavailable)}

    def __str__(self) -> str:
        bits = [f"{self.component}: {self.level}"]
        if self.reason:
            bits.append(self.reason)
        if self.unavailable:
            bits.append("unavailable: " + ", ".join(self.unavailable))
        return " - ".join(bits)


def good(component: str, unavailable: tuple[str, ...] = ()) -> DataStatus:
    return DataStatus(component, OK, "", unavailable)


def warning(component: str, reason: str, unavailable: tuple[str, ...] = ()) -> DataStatus:
    return DataStatus(component, WARNING, reason, unavailable)


def critical(component: str, reason: str, unavailable: tuple[str, ...] = ()) -> DataStatus:
    return DataStatus(component, CRITICAL, reason, unavailable)


def worst(component: str, statuses: list[DataStatus]) -> DataStatus:
    """The combined status of several stages: the most severe level, every reason, every missing field."""
    if not statuses:
        return critical(component, "no status was reported")
    level = max((s.level for s in statuses), key=_ORDER.__getitem__)
    reasons = "; ".join(f"{s.component}: {s.reason}" for s in statuses if s.reason)
    missing = tuple(dict.fromkeys(f for s in statuses for f in s.unavailable))
    return DataStatus(component, level, reasons, missing)
