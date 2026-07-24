"""Turn a signal + market context into a plain-language entry verdict.

Combines the model's confidence with futures-market context (funding rate and
the crowd long/short ratio) into a transparent score and a Turkish verdict:

- ``UYGUN``   — favourable: confident and the crowd/funding are not against it;
- ``DİKKATLİ`` — mixed: tradeable but with caveats;
- ``ZAYIF``   — weak: low conviction or crowded against the trade.

The rules are deliberately simple and explainable — this is decision *support*,
not a black box. Funding is expressed in percent; the L/S ratio is the global
account ratio (>1 means the crowd is net long).
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Suitability:
    verdict: str        # "UYGUN" | "DİKKATLİ" | "ZAYIF"
    score: int
    note: str


def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def compute_suitability(
    side: str,
    *,
    prob: float,
    confident: bool,
    funding_pct: float,
    ls_ratio: float,
    fresh_cross: bool = False,
    wide_band: bool = False,
) -> Suitability:
    """Score a setup and produce a verdict with a short reason (Turkish)."""
    score = 0
    notes: list[str] = []

    if confident:
        score += 2
        notes.append("model emin")

    # A fresh Bollinger mid-band crossover in the trade direction — stronger
    # still with wide bands (a real move, not chop).
    if fresh_cross:
        score += 1
        notes.append("taze orta-bant kesişimi")
        if wide_band:
            score += 1
            notes.append("geniş bant")
    if prob >= 0.5:
        score += 2
    elif prob >= 0.4:
        score += 1
    elif prob < 0.3:
        score -= 1

    # Funding alignment: a long pays funding when it is positive (costly, crowded
    # longs); a short benefits from positive funding, and vice-versa.
    if not _is_nan(funding_pct):
        if side == "long":
            if funding_pct <= 0.0:
                score += 1
                notes.append("funding uygun")
            elif funding_pct > 0.05:
                score -= 1
                notes.append("funding yüksek (kalabalık long)")
        else:  # short
            if funding_pct >= 0.0:
                score += 1
                notes.append("funding uygun")
            elif funding_pct < -0.05:
                score -= 1
                notes.append("funding negatif (kalabalık short)")

    # Crowd positioning (contrarian caution when the crowd is already piled in).
    if not _is_nan(ls_ratio):
        if side == "long":
            if ls_ratio > 2.0:
                score -= 1
                notes.append("kalabalık long (dikkat)")
            elif ls_ratio < 1.0:
                score += 1
        else:  # short
            if ls_ratio < 0.8:
                score -= 1
                notes.append("kalabalık short (dikkat)")
            elif ls_ratio > 1.5:
                score += 1
                notes.append("long kalabalık, düşüşe yer var")

    if score >= 3:
        verdict = "UYGUN"
    elif score >= 1:
        verdict = "DİKKATLİ"
    else:
        verdict = "ZAYIF"

    note = ", ".join(notes) if notes else "nötr"
    return Suitability(verdict=verdict, score=score, note=note)
