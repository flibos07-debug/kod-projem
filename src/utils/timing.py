"""Timeframe-boundary timing helpers for the live scanner.

The scanner must act on *closed* candles, so it wakes just after each
timeframe boundary (e.g. every 5 minutes on the clock). These helpers are pure
functions of an input time, which keeps the scheduling logic testable without
sleeping.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

INTERVAL_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
}


def _interval_minutes(interval: str) -> int:
    if interval not in INTERVAL_MINUTES:
        raise ValueError(f"unsupported interval {interval!r}")
    return INTERVAL_MINUTES[interval]


def floor_to_interval(moment: datetime, interval: str) -> datetime:
    """Round ``moment`` down to the start of its interval bucket (UTC)."""
    minutes = _interval_minutes(interval)
    moment = _as_utc(moment)
    day_start = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    elapsed = int((moment - day_start).total_seconds() // 60)
    bucket = (elapsed // minutes) * minutes
    return day_start + timedelta(minutes=bucket)


def next_boundary(moment: datetime, interval: str) -> datetime:
    """The next interval boundary strictly after ``moment``."""
    return floor_to_interval(moment, interval) + timedelta(minutes=_interval_minutes(interval))


def seconds_until(target: datetime, *, now: datetime | None = None) -> float:
    """Non-negative seconds from ``now`` (default: current UTC) until ``target``."""
    now = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    return max(0.0, (_as_utc(target) - now).total_seconds())


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)
