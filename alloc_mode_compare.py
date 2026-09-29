# -*- coding: utf-8 -*-
"""
배분 방식 비교: hold_cash vs redeploy vs raw(엔진 원비중)
=========================================================
질문: 현재 현금 77%는 의도된 결과인가? hold_cash 채택(2026-09-28)이
6~7월 KOSPI 급락 '사후 선택'은 아닌가?

비교
  hold_cash : 청산분 현금 유지, 비중 진입시점 확정 (현행)
  redeploy  : 청산분 남은 종목에 재분배, 종목 20% 상한 (7/9~9/25)
  raw       : 엔진 원시 리스크패리티 명목비중(합>1이면 축소) — 락/재분배 없음
  redeploy_vm: redeploy를 hold_cash와 같은 변동성으로 축소(나머지 현금) → 공정 MDD 비교
출력: 구간별/연도별 지표, 현금비중 분포, 현재 비중.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import isa_signals as S
import mulvaney_replica as M
from mulvaney_isa_backtest import ISA_DEF, build_krw_panel

TD = 252


def setup():
    p = S.PARAMS
    M.N_LOOKBACK, M.STOP_P, M.EXEC_LAG = p["N"], p["p"], p["lag"]
    M.PYR_CAP, M.PYR_K, M.SHORT_WEIGHT = p["cap"], p["K"], p["short"]
    M.SCHEME = S.SCHEME
    labels = list(ISA_DEF.keys())
    sectors = {}
    for lab in labels:
        sectors.setdefault(ISA_DEF[lab][2], []).append(lab)
    M.UNIVERSE, M.TICKERS = sectors, labels


def port_eq(Pm, Rm, crv):
    Ps = Pm.sum(axis=1)
    r = np.zeros(len(Pm))
    r[1:] = (Pm[:-1] * Rm[1:]).sum(axis=1) + np.clip(1 - Ps[:-1], 0, None) * crv[1:]
    return np.nan_to_num(r)


def stats(r):
    r = pd.Series(r)
    eq = (1 + r).cumprod()
    n = len(r)
    cagr = eq.iloc[-1] ** (TD / n) - 1
    vol = r.std() * np.sqrt(TD)
    sh = r.mean() * TD / vol if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return dict(CAGR=cagr * 100, vol=vol * 100, Sharpe=sh, MDD=mdd * 100)


def main():
    setup()
    high, low, close, _ = build_krw_panel()
    cash_rate = M.load_cash_rate(close.index)
    sig = M.precompute_signals(high, low, close)
    res = M.backtest(high, low, close, sig, cash_rate, record_weights=True)
    valid = (res["n_active"] > 0)
    valid = np.asarray(valid.values if hasattr(valid, "values") else valid)

    order = list(M.TICKERS)
    secs = sorted({ISA_DEF[lb][2] for lb in order})
    sid = {s: i for i, s in enumerate(secs)}
    sec_ids = np.array([sid[ISA_DEF[lb][2]] for lb in order])
    Wm = np.clip(res["weights"][order].values, 0, None)
    Sm = res["shares"][order].values
    Rm = close[order].pct_change().fillna(0.0).values
    crv = np.asarray(getattr(cash_rate, "values", cash_rate), dtype=float)

    g = Wm.sum(axis=1, keepdims=True)
    P = {
        "hold_cash": S._allocate_hold_cash(Wm, Sm, S.P_CAP),
        "redeploy": np.vstack([S._allocate(Wm[t], S.P_CAP, S.S_CAP, sec_ids, len(secs))
                               for t in range(len(Wm))]),
        "raw": np.where(g > 1, Wm / np.where(g > 0, g, 1.0), Wm),
    }
    idx = close.index[valid]
    R = {k: pd.Series(port_eq(v, Rm, crv)[valid], index=idx) for k, v in P.items()}
    cash = {k: pd.Series(1 - v.sum(axis=1), index=close.index)[valid] for k, v in P.items()}

    # 변동성 매칭 redeploy: 위험자산 비중을 k배로 축소, 나머지 현금
    k = R["hold_cash"].std() / R["redeploy"].std()
    Pvm = P["redeploy"] * k
    R["redeploy_vm"] = pd.Series(port_eq(Pvm, Rm, crv)[valid], index=idx)
    cash["redeploy_vm"] = pd.Series(1 - Pvm.sum(axis=1), index=close.index)[valid]
    print(f"변동성매칭 계수 k={k:.2f}")

    cols = ["hold_cash", "redeploy", "redeploy_vm", "raw"]
    periods = {
        "전체": (idx[0], idx[-1]),
        "~2025말(규칙변경 전 표본)": (idx[0], pd.Timestamp("2025-12-31")),
        "2001-2012": (idx[0], pd.Timestamp("2012-12-31")),
        "2013-2025": (pd.Timestamp("2013-01-01"), pd.Timestamp("2025-12-31")),
        "2026 YTD": (pd.Timestamp("2026-01-01"), idx[-1]),
    }
    rows = []
    for pn, (a, b) in periods.items():
        for c in cols:
            s = stats(R[c].loc[a:b])
            s.update(구간=pn, 방식=c, 평균현금=cash[c].loc[a:b].mean() * 100)
            rows.append(s)
    tab = pd.DataFrame(rows).set_index(["구간", "방식"]).round(2)
    pd.set_option("display.width", 200)
    print(tab.to_string())

    # 연도별 Sharpe 승패 (hold_cash vs redeploy)
    yr = pd.DataFrame({c: R[c].groupby(R[c].index.year).apply(
        lambda x: x.mean() / x.std() * np.sqrt(TD) if x.std() > 0 else np.nan) for c in cols})
    yret = pd.DataFrame({c: R[c].groupby(R[c].index.year).apply(
        lambda x: ((1 + x).prod() - 1) * 100) for c in cols})
    print("\n연도별 수익률(%)\n", yret.round(1).to_string())
    win = (yr["hold_cash"] > yr["redeploy"]).sum()
    print(f"\n연도별 Sharpe: hold_cash 우위 {win}/{len(yr)}년 (vs redeploy)")
    win_vm = (yret["hold_cash"] > yret["redeploy_vm"]).sum()
    print(f"연도별 수익률: hold_cash > redeploy_vm {win_vm}/{len(yr)}년")

    # 현금비중 분포
    q = pd.DataFrame({c: cash[c].describe(percentiles=[.1, .25, .5, .75, .9]) for c in cols}).T
    print("\n현금비중 분포(%)\n", (q[["mean", "10%", "25%", "50%", "75%", "90%"]] * 100).round(1).to_string())
    pct77 = (cash["hold_cash"] >= 0.77).mean() * 100
    print(f"hold_cash 현금 ≥77% 인 날 비율: {pct77:.1f}%")

    # 현재 비중
    last = pd.DataFrame({c: pd.Series(P[c][-1] * 100, index=order) for c in ["hold_cash", "redeploy", "raw"]})
    last = last[(last > 0.05).any(axis=1)].round(1)
    print("\n현재 비중(%)\n", last.to_string())
    print("현재 현금(%):", {c: round(cash[c].iloc[-1] * 100, 1) for c in cols})

    tab.to_csv("alloc_mode_compare.csv", encoding="utf-8-sig")
    yret.round(2).to_csv("alloc_mode_compare_yearly.csv", encoding="utf-8-sig")
    pd.DataFrame({c: (1 + R[c]).cumprod() for c in cols}).join(
        pd.DataFrame({f"cash_{c}": cash[c] for c in cols})).to_csv(
        "alloc_mode_compare_daily.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
