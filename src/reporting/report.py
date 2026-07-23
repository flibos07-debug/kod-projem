"""Render, persist and prune scan reports.

Terminal output is a plain monospaced table (no third-party dependency). Each
scan is also written to the report directory as CSV for later analysis, and old
scans beyond the configured retention window are pruned.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)

_SCAN_PREFIX = "scan_"
_TS_PATTERN = re.compile(r"scan_(\d{8}T\d{6}Z)\.csv$")


def render_table(df: pd.DataFrame, *, columns: list[str] | None = None, float_fmt: str = "{:.4f}") -> str:
    """Render a DataFrame as an aligned monospaced table."""
    if df.empty:
        return "(no rows)"
    columns = columns or list(df.columns)

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return float_fmt.format(value)
        return str(value)

    rows = [[fmt(v) for v in row] for row in df[columns].itertuples(index=False, name=None)]
    widths = [len(c) for c in columns]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([line(columns), sep, *[line(r) for r in rows]])


def _fmt_price(x: float) -> str:
    if x != x:  # NaN
        return "-"
    if abs(x) >= 100:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


def render_signals(signals: dict[str, pd.DataFrame]) -> str:
    """Render the LONG/SHORT trade-plan tables (entry zone, SL, TP1, TP2)."""
    out: list[str] = []
    for side in ("long", "short"):
        df = signals.get(side)
        title = f"{side.upper()} SİNYALLERİ"
        if df is None or df.empty:
            out.append(f"=== {title} ===\n(uygun sinyal yok)")
            continue
        rows = []
        for _, r in df.iterrows():
            vol = r.get("quote_volume", float("nan"))
            funding = r.get("funding", float("nan"))
            rows.append({
                "symbol": r["symbol"],
                "prob": f"{r['prob']:.3f}",
                "conf": "evet" if r["confident"] else "-",
                "hacim(M$)": "-" if vol != vol else f"{vol/1e6:,.0f}",
                "funding%": "-" if funding != funding else f"{funding:+.3f}",
                "giriş": f"{_fmt_price(r['entry_low'])} - {_fmt_price(r['entry_high'])}",
                "SL": f"{_fmt_price(r['stop_loss'])} ({r['sl_pct']:+.2f}%)",
                "TP1": f"{_fmt_price(r['tp1'])} ({r['tp1_pct']:+.2f}%)",
                "TP2": f"{_fmt_price(r['tp2'])} ({r['tp2_pct']:+.2f}%)",
            })
        table = render_table(pd.DataFrame(rows))
        out.append(f"=== {title} (en iyi {len(df)}) ===\n{table}")
    return "\n\n".join(out)


def save_signals(signals: dict[str, pd.DataFrame], report_dir: str | Path, *, timestamp: datetime | None = None) -> Path:
    """Persist LONG+SHORT signals (with levels) to one CSV."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    frames = [df for df in (signals.get("long"), signals.get("short")) if df is not None and not df.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = report_dir / f"signals_{ts.strftime('%Y%m%dT%H%M%SZ')}.csv"
    combined.to_csv(path, index=False)
    logger.info("Saved signals to %s", path)
    return path


def save_scan(df: pd.DataFrame, report_dir: str | Path, *, timestamp: datetime | None = None) -> Path:
    """Write a scan to ``report_dir/scan_<UTC timestamp>.csv`` and return the path."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = (timestamp or datetime.now(timezone.utc)).astimezone(timezone.utc)
    path = report_dir / f"{_SCAN_PREFIX}{ts.strftime('%Y%m%dT%H%M%SZ')}.csv"
    df.to_csv(path, index=False)
    logger.info("Saved scan report to %s", path)
    return path


def cleanup_old_reports(
    report_dir: str | Path, retention_days: int, *, now: datetime | None = None
) -> int:
    """Delete scan CSVs older than ``retention_days``; return the count removed."""
    report_dir = Path(report_dir)
    if not report_dir.exists():
        return 0
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff = now - timedelta(days=retention_days)

    removed = 0
    for path in report_dir.glob(f"{_SCAN_PREFIX}*.csv"):
        match = _TS_PATTERN.search(path.name)
        if not match:
            continue
        stamp = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        if stamp < cutoff:
            path.unlink()
            removed += 1
    if removed:
        logger.info("Pruned %d scan report(s) older than %d days", removed, retention_days)
    return removed
