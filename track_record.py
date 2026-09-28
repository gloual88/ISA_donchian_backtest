# -*- coding: utf-8 -*-
"""
ISA 추세전략 — 발행 비중 기록(track record) & 실적 곡선
=======================================================
기존 전략 곡선(equity)은 매 실행마다 '현재 규칙으로 2001년부터 다시 돌린 백테스트'라
과거 값이 바뀐다(규칙 변경·데이터 수정 시 6/2 이후 수익률이 수%p 달라짐).
'실제로 그날 메일/대시보드가 알려준 비중을 들고 있었다면'의 성과는 재계산하지 않고
기록으로 고정한다.

  data/track_record.json : {asof: {종목라벨: 비중%}}  — 매 실행마다 '당일 asof' 항목만
                           덮어쓰고, 지난 날짜는 절대 수정하지 않는다(append-only).
  live_curve()           : 전일 발행 비중 × 당일 종목 수익률 + 잔여현금 × 무위험수익률

시작점은 기록의 첫 날짜(2026-05-29, git 이력으로 시드). 이후는 매일 누적된다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RECORD = Path(__file__).resolve().parent / "data" / "track_record.json"


def load_record() -> dict:
    if not RECORD.exists():
        return {}
    return json.loads(RECORD.read_text(encoding="utf-8")).get("weights", {})


def save_record(weights: dict) -> None:
    RECORD.parent.mkdir(exist_ok=True)
    payload = {
        "note": "asof별 발행 비중(%). 당일 항목만 갱신, 과거 날짜는 수정 금지.",
        "weights": {k: weights[k] for k in sorted(weights)},
    }
    RECORD.write_text(json.dumps(payload, ensure_ascii=False, indent=0),
                      encoding="utf-8")


def update_record(asof: str, positions: list) -> dict:
    """오늘 발행 비중을 기록(같은 asof 재실행이면 그 항목만 갱신)."""
    w = load_record()
    w[str(asof)] = {p["label"]: float(p["isa_weight_pct"]) for p in positions}
    save_record(w)
    return w


def live_curve(prices: pd.DataFrame, cash_rate, record: dict) -> pd.Series:
    """발행 비중 기록으로 만든 실적 곡선(첫 기록일=1.0).

    t일 수익 = (t-1일 이전 마지막 발행 비중) × (t일 종목 수익률)
              + 잔여현금 × 무위험수익률.  발행이 없는 날은 직전 발행 비중 유지.
    """
    if not record:
        return pd.Series(dtype=float)
    keys = sorted(record)
    ret = prices.pct_change()
    crv = pd.Series(np.asarray(cash_rate, dtype=float), index=prices.index)
    dates = [d for d in prices.index if d >= pd.Timestamp(keys[0])]
    val, out = 1.0, {dates[0]: 1.0}
    for prev, t in zip(dates[:-1], dates[1:]):
        k = max(x for x in keys if x <= str(prev.date()))
        w = pd.Series(record[k], dtype=float).div(100.0)
        w = w.reindex(prices.columns).fillna(0.0)
        r = (w * ret.loc[t].fillna(0.0)).sum() \
            + max(1.0 - w.sum(), 0.0) * crv.loc[t]
        val *= 1.0 + float(r)
        out[t] = val
    return pd.Series(out)
