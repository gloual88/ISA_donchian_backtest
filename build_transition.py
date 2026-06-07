# -*- coding: utf-8 -*-
"""
build_transition.py — 전략 전환 리밸런싱(1회성) 산출 → data/transition.json

전략 파라미터를 교체하면, 구전략 포트폴리오를 신전략으로 갈아탈 때 '오늘 실제로
조정할 매매'(비중 델타)가 발생한다. 이건 대시보드의 '오늘의 액션'(신전략 자생적
신호)과 다르다. 구/신 파라미터를 **같은 기준일·같은 패널**로 계산해 종목별 비중
델타를 transition.json에 저장하면, 대시보드·이메일이 파일이 있을 때만 '전환' 섹션을
표시한다. 전환 반영이 끝나면 파일을 삭제하면 섹션이 사라진다.

실행: PYTHONIOENCODING=utf-8 python build_transition.py
  (기본: 구 N=252/lag=1 → 신 = isa_signals.PARAMS. --from 으로 구 설정 변경 가능)
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd

import mulvaney_replica as M
from mulvaney_isa_backtest import ISA_DEF, build_krw_panel
from isa_signals import (KRX_NAME, _compute_nolev,
                         PARAMS as NEW_PARAMS, SCHEME as NEW_SCHEME)

BASE = Path(__file__).resolve().parent
OUT = BASE / "data" / "transition.json"

# 직전 라이브 = 크로스애셋 12종 + LP (업종 확장·HLP 채택 전)
ORIGINAL_12 = ["S&P500(미국)", "나스닥100", "러셀2000(H)", "신흥국(H)", "KOSPI200",
               "미국채30년", "미국채10년", "금(H)", "은(H)", "WTI원유(H)",
               "농산물(H)", "미국달러선물"]
OLD_DEFAULT = dict(N=189, p=0.4, lag=2, cap=2, K=1, short=0.0)


def _weights(high, low, close, cash_rate, cfg, labels, scheme):
    """labels 서브셋 + scheme(0=LP/1=HLP)로 비중 산출. 패널을 labels로 슬라이스해
    TICKERS/컬럼 정합을 맞춘다(구=12종/LP, 신=27종/HLP)."""
    h, lo, c = high[labels], low[labels], close[labels]
    M.N_LOOKBACK = cfg["N"]
    M.STOP_P = cfg["p"]
    M.EXEC_LAG = cfg["lag"]
    M.PYR_CAP = cfg["cap"]
    M.PYR_K = cfg["K"]
    M.SHORT_WEIGHT = cfg["short"]
    M.SCHEME = scheme
    sectors = {}
    for lab in labels:
        sectors.setdefault(ISA_DEF[lab][2], []).append(lab)
    M.UNIVERSE = sectors
    M.TICKERS = labels
    sig = M.precompute_signals(h, lo, c)
    res = M.backtest(h, lo, c, sig, cash_rate)
    w, cash_pct = _compute_nolev(res["positions"])
    asof = str(pd.Timestamp(res["asof"]).date())
    return {t: float(w.get(t, 0.0)) for t in w.index}, float(cash_pct), asof


def _action(old_w: float, new_w: float, delta: float) -> str:
    if old_w <= 0 < new_w:
        return "신규편입"
    if new_w <= 0 < old_w:
        return "전량매도"
    if delta >= 0.5:
        return "매수"
    if delta <= -0.5:
        return "매도"
    return "유지"


def main(old_cfg: dict):
    print("[1/2] KRW 패널 + 구/신 비중 계산...")
    high, low, close, ksC = build_krw_panel()
    cash_rate = M.load_cash_rate(close.index)
    # 구 = 12종/LP, 신 = 27종(전체)/HLP
    old_w, old_cash, _ = _weights(high, low, close, cash_rate, old_cfg,
                                  ORIGINAL_12, 0)
    new_w, new_cash, asof = _weights(high, low, close, cash_rate, NEW_PARAMS,
                                     list(ISA_DEF.keys()), NEW_SCHEME)

    tickers = set(old_w) | set(new_w)
    rows = []
    for t in tickers:
        ow, nw = round(old_w.get(t, 0.0), 1), round(new_w.get(t, 0.0), 1)
        if ow == 0 and nw == 0:
            continue
        d = round(nw - ow, 1)
        name, code = KRX_NAME.get(t, (t, ""))
        rows.append(dict(ticker=t, etf=name, code=code,
                         sector=ISA_DEF[t][2] if t in ISA_DEF else "",
                         old_w=ow, new_w=nw, delta=d, action=_action(ow, nw, d)))
    rows.sort(key=lambda r: -abs(r["delta"]))

    sch = {0: "LP", 1: "HLP"}
    payload = dict(
        from_params={**{k: old_cfg[k] for k in ("N", "p", "lag", "cap", "K")},
                     "scheme": "LP(12종)"},
        to_params={**{k: NEW_PARAMS[k] for k in ("N", "p", "lag", "cap", "K")},
                   "scheme": f"{sch.get(NEW_SCHEME, '?')}(27종)"},
        asof=asof,
        old_cash=round(max(old_cash, 0), 1),
        new_cash=round(max(new_cash, 0), 1),
        generated_kst=pd.Timestamp.now(tz="Asia/Seoul").strftime(
            "%Y-%m-%d %H:%M KST"),
        rows=rows)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"[2/2] 저장: {OUT}  (기준일 {asof}, {len(rows)}종목)")
    print(f"  전환: N={old_cfg['N']}/lag={old_cfg['lag']} → "
          f"N={NEW_PARAMS['N']}/lag={NEW_PARAMS['lag']}")
    for r in rows:
        if r["action"] != "유지":
            print(f"  {r['action']:6} {r['etf']:24} {r['old_w']:5.1f}% → "
                  f"{r['new_w']:5.1f}%  ({r['delta']:+.1f}%p)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-N", type=int, default=OLD_DEFAULT["N"])
    ap.add_argument("--from-lag", type=int, default=OLD_DEFAULT["lag"])
    args = ap.parse_args()
    old = dict(OLD_DEFAULT)
    old["N"], old["lag"] = args.from_N, args.from_lag
    main(old)
