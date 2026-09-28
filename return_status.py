# -*- coding: utf-8 -*-
"""
ISA 추세전략 — 수익률 현황 (기준일 이후 누적 + 당일)
======================================================
data/signals.json(사전계산) 또는 라이브에서 자본곡선을 읽어
지정 기준일 이후 누적수익률 / 당일수익률 / 기준일 이후 MDD를 산출.
전략 + KOSPI200 벤치마크 함께.

기준일 변경: 환경변수 ANCHOR_DATE (기본 2026-06-02 · 포트폴리오 시작 시점)
실행: PYTHONIOENCODING=utf-8 python return_status.py
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
ANCHOR = os.getenv("ANCHOR_DATE", "2026-06-02")


CURVE_KEYS = [("strategy", "equity"), ("sixty_forty", "sixty_forty"),
              ("ew_basket", "ew_basket"), ("kospi", "kospi")]


def load_curves():
    """반환: (asof, {name: Series})  name∈strategy/sixty_forty/ew_basket/kospi"""
    cache = BASE / "data" / "signals.json"
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        curves = {}
        for name, key in CURVE_KEYS:
            if key in d:
                curves[name] = pd.Series(
                    d[key]["values"], index=pd.to_datetime(d[key]["dates"]))
        return d.get("asof"), curves
    from isa_signals import get_isa_signals
    s = get_isa_signals()
    return s["asof"], {n: s[k] for n, k in CURVE_KEYS if k in s}


def load_us_last():
    """signals.json의 미국 종가 확정일(us_last_date). 없으면 None(구버전 파일)."""
    cache = BASE / "data" / "signals.json"
    if not cache.exists():
        return None
    return json.loads(cache.read_text(encoding="utf-8")).get("us_last_date")


def compute_status(eq: pd.Series, anchor: str, end=None) -> dict:
    """기준일 이후 누적·당일·MDD.

    end: 집계 종료일(포함). 한국 발송시각(18:30/20:00 KST)에는 미국장이 아직
    열리기 전이라, 마지막 한국 거래일 행의 미국 ETF 수익률은 직전 종가 ffill(=0)이다.
    이 행까지 '당일'로 잡으면 미국 자산 위주 포트폴리오가 현금이자(≈+0.005%)만
    찍혀 항상 0%로 보이므로, 미국 종가 확정일(us_last_date)에서 잘라 계산한다.
    """
    eq = eq.dropna()
    if end:
        eq = eq[eq.index <= pd.Timestamp(end)]
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


LABELS = {"strategy": "전략", "sixty_forty": "60/40",
          "ew_basket": "동일가중바스켓", "kospi": "KOSPI200(참고)"}


def main():
    asof, curves = load_curves()
    end = load_us_last()
    stats = {n: compute_status(s, ANCHOR, end) for n, s in curves.items()}
    st = stats["strategy"]

    print("=" * 64)
    print("  ISA 추세전략 — 수익률 현황")
    print(f"  기준일 {st['anchor_date']}  →  현재 {st['last_date']}  "
          f"({st['trading_days']} 거래일)")
    print("=" * 64)
    print(f"{'':16}{'당일':>11}{'기준일이후누적':>16}{'기준후MDD':>12}")
    for n in ["strategy", "sixty_forty", "ew_basket", "kospi"]:
        if n not in stats:
            continue
        m = stats[n]
        print(f"{LABELS[n]:16}{m['daily_return']*100:>10.2f}%"
              f"{m['cum_return']*100:>15.2f}%{m['mdd_since']*100:>11.1f}%")
    print("-" * 64)
    if "sixty_forty" in stats:
        d = (st['cum_return'] - stats['sixty_forty']['cum_return']) * 100
        print(f"  기준일 이후 초과수익(전략−60/40): {d:+.2f}%p")


if __name__ == "__main__":
    main()
