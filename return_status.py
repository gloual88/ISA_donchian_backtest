# -*- coding: utf-8 -*-
"""
ISA 추세전략 — 수익률 현황 (기준일 이후 누적 + 당일)
======================================================
data/signals.json(사전계산) 또는 라이브에서 자본곡선을 읽어
지정 기준일 이후 누적수익률 / 당일수익률 / 기준일 이후 MDD를 산출.
전략 + KOSPI200 벤치마크 함께.

기준일 변경: 환경변수 ANCHOR_DATE (기본 2026-03-31)
실행: PYTHONIOENCODING=utf-8 python return_status.py
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
ANCHOR = os.getenv("ANCHOR_DATE", "2026-03-31")


def load_curves():
    cache = BASE / "data" / "signals.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        eq = pd.Series(d["equity"]["values"],
                       index=pd.to_datetime(d["equity"]["dates"]))
        ks = pd.Series(d["kospi"]["values"],
                       index=pd.to_datetime(d["kospi"]["dates"]))
        return d.get("asof"), eq, ks
    from isa_signals import get_isa_signals
    s = get_isa_signals()
    return s["asof"], s["equity"], s["kospi"]


def compute_status(eq: pd.Series, anchor: str) -> dict:
    """기준일 이후 누적·당일·MDD."""
    eq = eq.dropna()
    before = eq[eq.index <= pd.Timestamp(anchor)]
    if before.empty:
        a_val, a_date = eq.iloc[0], eq.index[0]
    else:
        a_val, a_date = before.iloc[-1], before.index[-1]
    seg = eq[eq.index >= a_date]
    cum = eq.iloc[-1] / a_val - 1.0
    daily = eq.iloc[-1] / eq.iloc[-2] - 1.0 if len(eq) >= 2 else 0.0
    mdd = (seg / seg.cummax() - 1.0).min()
    return {
        "anchor_date": str(a_date.date()),
        "last_date": str(eq.index[-1].date()),
        "trading_days": len(seg) - 1,
        "cum_return": float(cum),
        "daily_return": float(daily),
        "mdd_since": float(mdd),
    }


def main():
    asof, eq, ks = load_curves()
    st = compute_status(eq, ANCHOR)
    sk = compute_status(ks, ANCHOR)

    print("=" * 60)
    print(f"  ISA 추세전략 — 수익률 현황")
    print(f"  기준일 {st['anchor_date']}  →  현재 {st['last_date']}  "
          f"({st['trading_days']} 거래일)")
    print("=" * 60)
    print(f"{'':14}{'당일':>12}{'기준일 이후 누적':>18}{'기준후 MDD':>12}")
    print(f"{'전략':14}{st['daily_return']*100:>11.2f}%"
          f"{st['cum_return']*100:>17.2f}%{st['mdd_since']*100:>11.1f}%")
    print(f"{'KOSPI200':14}{sk['daily_return']*100:>11.2f}%"
          f"{sk['cum_return']*100:>17.2f}%{sk['mdd_since']*100:>11.1f}%")
    print("-" * 60)
    diff = (st['cum_return'] - sk['cum_return']) * 100
    print(f"  기준일 이후 초과수익(전략−KOSPI): {diff:+.2f}%p")


if __name__ == "__main__":
    main()
