"""Terminal + colored-HTML rendering for the multi-timeframe screener."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .report import render_table

_TREND_ARROW = {"up": "▲", "down": "▼", "flat": "→"}


def _n(x, fmt="{:.1f}"):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return fmt.format(x)


def _pct100(x):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    return f"{x * 100:.2f}"


def _price(x):
    if x is None or (isinstance(x, float) and x != x):
        return "-"
    if abs(x) >= 100:
        return f"{x:,.2f}"
    if abs(x) >= 1:
        return f"{x:.4f}"
    return f"{x:.6f}"


def render_screener(signals: dict[str, pd.DataFrame]) -> str:
    """Compact terminal view of the LONG/SHORT screener candidates."""
    out = []
    for side in ("long", "short"):
        df = signals.get(side)
        title = "LONG ADAYLARI" if side == "long" else "SHORT ADAYLARI"
        if df is None or df.empty:
            out.append(f"=== {title} ===\n(yok)")
            continue
        rows = []
        for _, r in df.iterrows():
            bias = f"{_TREND_ARROW.get(r.get('ema200','na'),'·')}{_TREND_ARROW.get(r.get('supertrend','na'),'·')}{_TREND_ARROW.get(r.get('vwap','na'),'·')}"
            rows.append({
                "coin": r["symbol"],
                "durum": r.get("state", "-"),
                "uzama%": _n(r.get("ext_pct"), "{:+.1f}"),
                "setup": r.get("setup", "-"),
                "skor": _n(r["score"], "{:.2f}"),
                "24s%": _n(r.get("chg24h"), "{:+.1f}"),
                "RSI 5/15/1h": f"{_n(r['rsi_5m'],'{:.0f}')}/{_n(r['rsi_15m'],'{:.0f}')}/{_n(r['rsi_1h'],'{:.0f}')}",
                "trend": f"{_TREND_ARROW.get(r['trend_5m'],'?')}{_TREND_ARROW.get(r['trend_15m'],'?')}{_TREND_ARROW.get(r['trend_1h'],'?')}",
                "E200/ST/VW": bias,
                "ADX": _n(r.get("adx"), "{:.0f}"),
                "MACD": r["macd_1h"],
                "hacim": _n(r.get("vol_ratio"), "{:.1f}x"),
                "OIΔ%": _n(r.get("oi_change"), "{:+.1f}"),
                "fund%": _n(r.get("funding"), "{:+.3f}"),
                "L/S": _n(r.get("ls_ratio"), "{:.2f}"),
                "öncü sinyal": r.get("leading", "-"),
            })
        out.append(f"=== {title} (en iyi {len(df)}) ===\n{render_table(pd.DataFrame(rows))}")
    return "\n\n".join(out)


_CSS = """
:root { color-scheme: light dark; } * { box-sizing: border-box; }
body { margin:0; font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif; background:#0f1420; color:#e6e9ef; padding:18px; }
h1 { font-size:19px; margin:0 0 3px; } .sub { color:#9aa4b2; font-size:12px; margin-bottom:16px; }
.section { margin-bottom:24px; } .section h2 { font-size:15px; margin:0 0 8px; padding-left:9px; border-left:4px solid; }
.long h2 { border-color:#2ecc71; color:#6ee7a8; } .short h2 { border-color:#e74c3c; color:#ff9b8f; }
.wrap { overflow-x:auto; border-radius:9px; }
table { border-collapse:collapse; width:100%; font-size:12px; min-width:1100px; }
th,td { padding:6px 8px; text-align:right; white-space:nowrap; }
th { background:#1a2030; color:#aab3c2; position:sticky; top:0; }
td.sym,th.sym { text-align:left; font-weight:700; }
tbody tr:nth-child(odd){ background:#161c29; } tbody tr:nth-child(even){ background:#131824; }
.long tbody tr { border-left:3px solid rgba(46,204,113,.5);} .short tbody tr{ border-left:3px solid rgba(231,76,60,.5);}
.up{color:#4ade80;} .dn{color:#f87171;} .flat{color:#9aa4b2;}
.rsi-lo{color:#4ade80;font-weight:700;} .rsi-hi{color:#f87171;font-weight:700;}
.lead{ color:#fcd34d; text-align:left; white-space:normal; max-width:240px; font-size:11px;}
.badge{display:inline-block;padding:1px 6px;border-radius:6px;background:#3a300f;color:#fcd34d;border:1px solid #d4a017;font-size:10px;}
.s-trend{background:#10233a;color:#7cc4ff;border-color:#2b6cb0;}
.s-rev{background:#2a1533;color:#d6a8ff;border-color:#805ad5;}
.s-break{background:#3a300f;color:#fcd34d;border-color:#d4a017;}
.s-flat{background:#232a3a;color:#9aa4b2;border-color:#3a4358;}
.st-early{background:#16351f;color:#4ade80;border-color:#2ecc71;}
.st-mid{background:#232a3a;color:#cbd5e1;border-color:#3a4358;}
.st-late{background:#3a1717;color:#f87171;border-color:#e74c3c;}
.foot{color:#6b7280;font-size:11px;margin-top:18px;border-top:1px solid #232a3a;padding-top:9px;}
"""

_HEAD = [
    ("sym", "Coin"), ("state", "Durum"), ("ext", "Uzama%"), ("setup", "Setup"),
    ("score", "Skor"), ("price", "Fiyat"), ("chg", "24s%"),
    ("rsi", "RSI 5m/15m/1h"), ("trend", "Trend 5/15/1h"),
    ("bias", "E200/ST/VWAP"), ("adx", "ADX"), ("macd", "MACD"),
    ("stoch", "StochRSI"), ("bb", "BB %B"), ("vol", "Hacim"), ("atr", "ATR%"),
    ("oichg", "OIΔ%"), ("fund", "Funding%"), ("ls", "L/S"),
    ("entry", "Giriş"), ("sl", "SL"), ("tp1", "TP1"), ("tp2", "TP2"),
    ("lead", "Öncü sinyal"),
]

_SETUP_CLASS = {"TREND": "s-trend", "DÖNÜŞ": "s-rev", "KIRILIM": "s-break", "NÖTR": "s-flat"}
_STATE_CLASS = {"BAŞLANGIÇ": "st-early", "ORTA": "st-mid", "UZAMIŞ": "st-late"}


def _trend_html(t):
    cls = {"up": "up", "down": "dn"}.get(t, "flat")
    return f'<span class="{cls}">{_TREND_ARROW.get(t, "?")}</span>'


def _rsi_html(v):
    if v != v:
        return "-"
    cls = "rsi-lo" if v < 35 else ("rsi-hi" if v > 65 else "")
    return f'<span class="{cls}">{v:.0f}</span>'


def _row_html(r):
    macd_cls = "up" if r["macd_1h"] == "up" else "dn"
    lead = html.escape(str(r.get("leading", "")))
    lead_html = f'<span class="badge">{lead}</span>' if lead and lead != "-" else "-"
    setup = str(r.get("setup", "-"))
    setup_html = f'<span class="badge {_SETUP_CLASS.get(setup, "s-flat")}">{html.escape(setup)}</span>'
    state = str(r.get("state", "-"))
    state_html = f'<span class="badge {_STATE_CLASS.get(state, "s-flat")}">{html.escape(state)}</span>'
    ext = r.get("ext_pct")
    ext_html = "-" if ext is None or ext != ext else f'{ext:+.1f}'
    oichg = r.get("oi_change")
    oichg_html = "-" if oichg is None or oichg != oichg else f'<span class="{"up" if oichg >= 0 else "dn"}">{oichg:+.1f}</span>'
    return "<tr>" + "".join([
        f'<td class="sym">{html.escape(str(r["symbol"]))}</td>',
        f'<td>{state_html}</td>',
        f'<td>{ext_html}</td>',
        f'<td>{setup_html}</td>',
        f'<td>{r["score"]:.2f}</td>',
        f'<td>{_price(r["price"])}</td>',
        f'<td class="{"up" if (r.get("chg24h") or 0) >= 0 else "dn"}">{_n(r.get("chg24h"), "{:+.1f}")}</td>',
        f'<td>{_rsi_html(r["rsi_5m"])}/{_rsi_html(r["rsi_15m"])}/{_rsi_html(r["rsi_1h"])}</td>',
        f'<td>{_trend_html(r["trend_5m"])}{_trend_html(r["trend_15m"])}{_trend_html(r["trend_1h"])}</td>',
        f'<td>{_trend_html(r.get("ema200","na"))}{_trend_html(r.get("supertrend","na"))}{_trend_html(r.get("vwap","na"))}</td>',
        f'<td>{_n(r.get("adx"), "{:.0f}")}</td>',
        f'<td class="{macd_cls}">{r["macd_1h"]}</td>',
        f'<td>{_n(r["stoch_1h"], "{:.0f}")}</td>',
        f'<td>{_n(r["bb_1h"], "{:.2f}")}</td>',
        f'<td>{_n(r.get("vol_ratio"), "{:.1f}x")}</td>',
        f'<td>{_pct100(r.get("atr_pct"))}</td>',
        f'<td>{oichg_html}</td>',
        f'<td>{_n(r.get("funding"), "{:+.3f}")}</td>',
        f'<td>{_n(r.get("ls_ratio"), "{:.2f}")}</td>',
        f'<td>{_price(r.get("entry_low"))}-{_price(r.get("entry_high"))}</td>',
        f'<td>{_price(r.get("stop_loss"))}</td>',
        f'<td>{_price(r.get("tp1"))}</td>',
        f'<td>{_price(r.get("tp2"))}</td>',
        f'<td class="lead">{lead_html}</td>',
    ]) + "</tr>"


def _table_html(df, side):
    title = "LONG ADAYLARI" if side == "long" else "SHORT ADAYLARI"
    if df is None or df.empty:
        return f'<div class="section {side}"><h2>{title}</h2><div style="color:#9aa4b2">(yok)</div></div>'
    head = "".join(f'<th class="{"sym" if k == "sym" else ""}">{html.escape(lbl)}</th>' for k, lbl in _HEAD)
    body = "".join(_row_html(r) for _, r in df.iterrows())
    return (f'<div class="section {side}"><h2>{title} — {len(df)}</h2>'
            f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></div>')


def render_screener_html(signals: dict[str, pd.DataFrame], *, meta: dict | None = None) -> str:
    meta = meta or {}
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    body = _table_html(signals.get("long"), "long") + _table_html(signals.get("short"), "short")
    return f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Futures Screener</title>
<style>{_CSS}</style></head><body>
<h1>📊 Futures Screener — 5m / 15m / 1h</h1>
<div class="sub">{ts} · taranan: {meta.get('scanned','?')} coin · RSI &lt;35 yeşil (aşırı satım), &gt;65 kırmızı · öncü sinyaller sarı</div>
{body}
<div class="foot">Karar-destek/tarama aracı; giriş-çıkış kararı sana ait. Skor = 5m/15m/1h teknik hizalanma
(1h ağırlıklı). Öncü sinyaller: Bollinger sıkışması, hacim spike'ı, StochRSI/MACD dönüşü.</div>
</body></html>"""


def save_screener_html(signals, path, *, meta=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_screener_html(signals, meta=meta), encoding="utf-8")
    return path
