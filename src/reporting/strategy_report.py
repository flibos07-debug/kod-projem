"""Terminal + colored-HTML rendering for the BB mid-band cross strategy."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .report import render_table

_STAGE_CLASS = {"GİRİŞ": "st-early", "BEKLE": "st-mid"}


def _n(x, fmt="{:.2f}"):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return fmt.format(x)


def _price(x):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    if abs(x) >= 100:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


_SECTIONS = [("long", "LONG (AL)"), ("short", "SHORT (SAT)")]


def render_strategy(signals: dict[str, pd.DataFrame]) -> str:
    out = []
    for key, title in _SECTIONS:
        df = signals.get(key)
        if df is None or df.empty:
            continue
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "coin": r["symbol"],
                "aşama": r["stage"],
                "TF": r.get("tf", "-"),
                "kesişim": f"{int(r['break_ago'])} bar önce" if r.get("break_ago") == r.get("break_ago") else "-",
                "fiyat": _price(r["price"]),
                "hacim": _n(r.get("vol_ratio"), "{:.1f}x"),
                "giriş": _price(r["entry"]),
                "SL": f"{_price(r['stop_loss'])} ({_n(r.get('sl_pct'), '{:+.1f}')}%)",
                "TP": f"{_price(r['tp'])} ({_n(r.get('tp_pct'), '{:+.1f}')}%)",
                "24s%": _n(r.get("chg24h"), "{:+.1f}"),
                "not": r.get("note", "-"),
            })
        out.append(f"=== {title} ({len(df)}) ===\n{render_table(pd.DataFrame(rows))}")
    return "\n\n".join(out) if out else "(uygun kurulum yok)"


_CSS = """
:root{color-scheme:light dark;}*{box-sizing:border-box;}
body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#0f1420;color:#e6e9ef;padding:18px;}
h1{font-size:19px;margin:0 0 3px;}.sub{color:#9aa4b2;font-size:12px;margin-bottom:16px;}
.section{margin-bottom:22px;}.section h2{font-size:15px;margin:0 0 8px;padding-left:9px;border-left:4px solid;}
.long h2{border-color:#2ecc71;color:#6ee7a8;}.short h2{border-color:#e74c3c;color:#ff9b8f;}.watch h2{border-color:#d4a017;color:#fcd34d;}
.wrap{overflow-x:auto;border-radius:9px;}table{border-collapse:collapse;width:100%;font-size:12px;min-width:900px;}
th,td{padding:7px 9px;text-align:right;white-space:nowrap;}th{background:#1a2030;color:#aab3c2;position:sticky;top:0;}
td.sym,th.sym{text-align:left;font-weight:700;}tbody tr:nth-child(odd){background:#161c29;}tbody tr:nth-child(even){background:#131824;}
.up{color:#4ade80;}.dn{color:#f87171;}
.badge{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11px;font-weight:700;}
.st-early{background:#16351f;color:#4ade80;border:1px solid #2ecc71;}
.st-mid{background:#232a3a;color:#cbd5e1;border:1px solid #3a4358;}
.st-late{background:#3a300f;color:#fcd34d;border:1px solid #d4a017;}
.note{color:#fcd34d;text-align:left;white-space:normal;max-width:320px;font-size:11px;}
.foot{color:#6b7280;font-size:11px;margin-top:18px;border-top:1px solid #232a3a;padding-top:9px;}
"""

_HEAD = [("sym", "Coin"), ("stage", "Aşama"), ("side", "Yön"), ("tf", "TF"),
         ("brk", "Kesişim"), ("price", "Fiyat"),
         ("vol", "Hacim"), ("entry", "Giriş"), ("sl", "SL"), ("tp", "TP"),
         ("chg", "24s%"), ("fund", "Funding%"), ("ls", "L/S"), ("note", "Kural / Not")]


def _brk(r):
    v = r.get("break_ago")
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{int(v)} bar önce"


def _row(r):
    stage = str(r["stage"])
    badge = f'<span class="badge {_STAGE_CLASS.get(stage, "st-mid")}">{html.escape(stage)}</span>'
    return "<tr>" + "".join([
        f'<td class="sym">{html.escape(str(r["symbol"]))}</td>',
        f'<td>{badge}</td>',
        f'<td>{html.escape(str(r["side"]))}</td>',
        f'<td>{html.escape(str(r.get("tf", "-")))}</td>',
        f'<td>{_brk(r)}</td>',
        f'<td>{_price(r["price"])}</td>',
        f'<td>{_n(r.get("vol_ratio"), "{:.1f}x")}</td>',
        f'<td>{_price(r["entry"])}</td>',
        f'<td>{_price(r["stop_loss"])} <span class="dn">({_n(r.get("sl_pct"), "{:+.1f}")}%)</span></td>',
        f'<td>{_price(r["tp"])} <span class="up">({_n(r.get("tp_pct"), "{:+.1f}")}%)</span></td>',
        f'<td class="{"up" if (r.get("chg24h") or 0) >= 0 else "dn"}">{_n(r.get("chg24h"), "{:+.1f}")}</td>',
        f'<td>{_n(r.get("funding"), "{:+.3f}")}</td>',
        f'<td>{_n(r.get("ls_ratio"))}</td>',
        f'<td class="note">{html.escape(str(r.get("note", "")))}</td>',
    ]) + "</tr>"


def _table(df, key, title):
    if df is None or df.empty:
        return ""
    head = "".join(f'<th class="{"sym" if k == "sym" else ""}">{html.escape(l)}</th>' for k, l in _HEAD)
    body = "".join(_row(r) for _, r in df.iterrows())
    return (f'<div class="section {key}"><h2>{title} — {len(df)}</h2>'
            f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>')


def render_strategy_html(signals: dict[str, pd.DataFrame], *, meta: dict | None = None) -> str:
    meta = meta or {}
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    body = "".join(_table(signals.get(k), k, t) for k, t in _SECTIONS)
    return f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>BB Orta Bant Kırılım Stratejisi</title>
<style>{_CSS}</style></head><body>
<h1>🎯 Bollinger Orta Bant Kırılımı (retest yok)</h1>
<div class="sub">{ts} · taranan: {meta.get('scanned','?')} coin · 15m/1h orta bant kesişimi + 5m onay · GİRİŞ=5m onayladı gir, BEKLE=kesişti 5m onayı bekliyor</div>
{body or '<div style="color:#9aa4b2">Uygun kurulum yok.</div>'}
<div class="foot">Strateji: 15m orta bant (20-SMA) YUKARI kesişimi (son 0-2 mum) + 5m fiyat orta bandın ÜSTÜNDE → LONG (tam tersi SHORT). 15m'de yoksa 1h'e bakılır.
Repaint önleme: veriler son tamamlanmış mumdan. SL: son 5m mumunun dibi/tepesi ± ATR. TP: karşı dış bant (long üst, short alt). Karar sana ait.</div>
</body></html>"""


def save_strategy_html(signals, path, *, meta=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_strategy_html(signals, meta=meta), encoding="utf-8")
    return path
