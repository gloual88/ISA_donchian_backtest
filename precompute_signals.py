# -*- coding: utf-8 -*-
"""
ISA 추세전략 — 매일 시그널 사전계산 (GitHub Actions cron 용)
================================================================
isa_signals.get_isa_signals()를 실행해 결과를 data/signals.json 으로 저장.
대시보드는 이 파일을 즉시 로드(느린 데이터 다운로드 생략) → 접속 즉시 표시.

실행: python precompute_signals.py
"""
import json
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd

from isa_signals import get_isa_signals

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "signals.json"


def _series_to_obj(s: pd.Series) -> dict:
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in s.index],
        "values": [float(v) for v in s.values],
    }


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def main():
    print("[1/2] 시그널 계산...")
    s = get_isa_signals()

    payload = {k: s[k] for k in (
        "asof", "params", "cash_pct", "n_positions", "metrics",
        "positions", "buy_today", "stop_today", "near_stop")}
    payload["equity"] = _series_to_obj(s["equity"])
    payload["kospi"] = _series_to_obj(s["kospi"])
    payload["_generated_kst"] = pd.Timestamp.now(
        tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M KST")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        json.dumps(payload, ensure_ascii=False, default=_default),
        encoding="utf-8")
    print(f"[2/2] 저장: {OUT}  (기준일 {s['asof']}, 보유 {s['n_positions']})")


if __name__ == "__main__":
    main()
