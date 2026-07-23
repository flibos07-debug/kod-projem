"""Render the LONG/SHORT signals into a self-contained, colored HTML page.

The output is a single standalone ``.html`` file (inline CSS, no external
requests) the user opens in a browser. LONG rows are green-accented, SHORT rows
red; each carries a verdict badge (UYGUN / DİKKATLİ / ZAYIF) and the futures
context (volume, funding, long/short ratio, open interest) alongside the trade
plan (entry zone, SL, TP1, TP2).
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_VERDICT_CLASS = {"UYGUN": "v-ok", "DİKKATLİ": "v-warn", "ZAYIF": "v-bad"}

_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
  background: #0f1420; color: #e6e9ef; padding: 20px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: #9aa4b2; font-size: 13px; margin-bottom: 18px; }
.section { margin-bottom: 26px; }
.section h2 { font-size: 16px; margin: 0 0 10px; padding-left: 10px; border-left: 4px solid; }
.long h2 { border-color: #2ecc71; color: #6ee7a8; }
.short h2 { border-color: #e74c3c; color: #ff9b8f; }
.wrap { overflow-x: auto; border-radius: 10px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; min-width: 900px; }
th, td { padding: 8px 10px; text-align: right; white-space: nowrap; }
th { background: #1a2030; color: #aab3c2; font-weight: 600; position: sticky; top: 0; }
td.sym, th.sym { text-align: left; font-weight: 700; }
tbody tr:nth-child(odd) { background: #161c29; }
tbody tr:nth-child(even) { background: #131824; }
.long tbody tr { border-left: 3px solid rgba(46,204,113,.5); }
.short tbody tr { border-left: 3px solid rgba(231,76,60,.5); }
.pos { color: #4ade80; } .neg { color: #f87171; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }
.v-ok { background: #16351f; color: #4ade80; border: 1px solid #2ecc71; }
.v-warn { background: #3a300f; color: #fcd34d; border: 1px solid #d4a017; }
.v-bad { background: #3a1717; color: #f87171; border: 1px solid #e74c3c; }
.conf { font-size: 11px; }
.note { color: #9aa4b2; font-size: 11px; text-align: left; white-space: normal; max-width: 220px; }
.empty { color: #9aa4b2; font-style: italic; padding: 8px 10px; }
.foot { color: #6b7280; font-size: 11px; margin-top: 20px; border-top: 1px solid #232a3a; padding-top: 10px; }
"""


def _fmt_price(x: float) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    if abs(x) >= 100:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


def _pct_span(pct: float) -> str:
    if pct != pct:
        return "-"
    cls = "pos" if pct >= 0 else "neg"
    return f'<span class="{cls}">{pct:+.2f}%</span>'


def _num(x: float, div: float = 1.0, fmt: str = "{:,.0f}") -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return fmt.format(x / div)


_HEADERS = [
    ("sym", "Coin"), ("prob", "Olasılık"), ("verdict", "Değerlendirme"),
    ("vol", "Hacim(M$)"), ("funding", "Funding%"), ("ls", "L/S"), ("oi", "OI"),
    ("entry", "Giriş aralığı"), ("sl", "SL"), ("tp1", "TP1"), ("tp2", "TP2"),
    ("note", "Not"),
]


def _row_html(r: pd.Series) -> str:
    verdict = str(r.get("verdict", ""))
    vclass = _VERDICT_CLASS.get(verdict, "v-warn")
    conf = "✓" if r.get("confident") else ""
    cells = [
        f'<td class="sym">{html.escape(str(r["symbol"]))}</td>',
        f'<td>{r["prob"]:.3f} <span class="conf pos">{conf}</span></td>',
        f'<td><span class="badge {vclass}">{html.escape(verdict)}</span></td>',
        f'<td>{_num(r.get("quote_volume"), 1e6)}</td>',
        f'<td>{_num(r.get("funding"), fmt="{:+.3f}")}</td>',
        f'<td>{_num(r.get("ls_ratio"), fmt="{:.2f}")}</td>',
        f'<td>{_num(r.get("open_interest"), fmt="{:,.0f}")}</td>',
        f'<td>{_fmt_price(r["entry_low"])} - {_fmt_price(r["entry_high"])}</td>',
        f'<td>{_fmt_price(r["stop_loss"])} {_pct_span(r["sl_pct"])}</td>',
        f'<td>{_fmt_price(r["tp1"])} {_pct_span(r["tp1_pct"])}</td>',
        f'<td>{_fmt_price(r["tp2"])} {_pct_span(r["tp2_pct"])}</td>',
        f'<td class="note">{html.escape(str(r.get("note", "")))}</td>',
    ]
    return "<tr>" + "".join(cells) + "</tr>"


def _table_html(df: pd.DataFrame, side: str) -> str:
    title = "LONG (AL)" if side == "long" else "SHORT (SAT)"
    if df is None or df.empty:
        return f'<div class="section {side}"><h2>{title}</h2><div class="empty">Uygun sinyal yok</div></div>'
    head = "".join(
        f'<th class="{ "sym" if key=="sym" else "" }">{html.escape(label)}</th>'
        for key, label in _HEADERS
    )
    body = "".join(_row_html(r) for _, r in df.iterrows())
    return (
        f'<div class="section {side}"><h2>{title} — en iyi {len(df)}</h2>'
        f'<div class="wrap"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{body}</tbody></table></div></div>'
    )


def render_html(signals: dict[str, pd.DataFrame], *, meta: dict | None = None) -> str:
    meta = meta or {}
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    scanned = meta.get("scanned", "?")
    max_move = meta.get("max_move", "?")
    min_vol = meta.get("min_volume", "?")
    body = _table_html(signals.get("long"), "long") + _table_html(signals.get("short"), "short")
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Futures Sinyalleri</title><style>{_CSS}</style></head>
<body>
<h1>🎯 Futures Sinyalleri — LONG / SHORT</h1>
<div class="sub">{ts} · taranan: {scanned} coin · TP2 ≤ %{max_move} · min hacim: {min_vol}M$</div>
{body}
<div class="foot">Bu bir karar-destek aracıdır, yatırım tavsiyesi değildir. Değerlendirme
= model güveni + funding + long/short oranı. Gerçek işlemden önce demo/paper ile test edin.
Olasılıklar düşükse liste bir izleme/sıralama listesidir.</div>
</body></html>"""


def save_html(signals: dict[str, pd.DataFrame], path: str | Path, *, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(signals, meta=meta), encoding="utf-8")
    return path
