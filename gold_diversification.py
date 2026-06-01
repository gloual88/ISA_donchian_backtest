# -*- coding: utf-8 -*-
"""
분산의 힘 — 금 단독 vs 분산 바스켓 (같은 추세 농법)
======================================================
"분산이 좋다"를 보여주는 비교. 동일한 추세 방법(돈치안 N=252, p=0.4,
추적손절, 롱온리)을 (1)금 단독 vs (2)분산 바스켓에 적용 → 차이 = 분산효과.
금 단순보유(B&H)는 기준선.

세 곡선:
  ① 금 단순보유 (Buy & Hold)
  ② 금 단독 + 추세 농법 (단일자산)
  ③ 분산 바스켓 + 추세 농법 (헤지펀드 추세전략)
모두 동일 KRW 프레임워크·동일 기간. 헤지펀드 비교를 위해 '같은 위험'
(금 보유 변동성)으로 맞춘 패널도 함께.

실행: PYTHONIOENCODING=utf-8 python gold_diversification.py
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mulvaney_replica as M
from mulvaney_isa_backtest import ISA_DEF, build_krw_panel

BASE = Path(__file__).resolve().parent
TD = 252
N, P = 252, 0.4
GOLD = "금(H)"


def gold_only_trend(high, low, close):
    """금 단일자산 롱온리 추세(보유=금100%, 청산=현금0%)."""
    up = high[GOLD].rolling(N).max().shift(1).values
    lo = low[GOLD].rolling(N).min().shift(1).values
    wd = up - lo
    mid = (up + lo) / 2
    c = close[GOLD].values
    pos = np.zeros(len(c))
    hold, stop, entry = 0, np.nan, np.nan
    for i in range(len(c)):
        if hold == 0:
            if not np.isnan(up[i]) and c[i] >= up[i] and wd[i] > 0:
                hold, entry, stop = 1, c[i], up[i] - P * wd[i]
        else:
            if c[i] > entry and not np.isnan(mid[i]):
                stop = max(stop, mid[i])
            if c[i] <= stop:
                hold = 0
        pos[i] = hold
    return pd.Series(pos, index=close.index)


def met(eq):
    r = eq.pct_change().fillna(0)
    n = len(eq)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (TD / n) - 1
    vol = r.std() * np.sqrt(TD)
    sh = (r.mean() * TD) / vol if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return dict(tot=eq.iloc[-1] / eq.iloc[0] - 1, cagr=cagr, vol=vol,
                sharpe=sh, mdd=mdd)


def volmatch(ret, target_vol):
    v = ret.std() * np.sqrt(TD)
    s = target_vol / v if v > 0 else 1.0
    return (1 + ret * s).cumprod()


def main():
    # 엔진: 분산 바스켓 롱온리
    M.N_LOOKBACK, M.STOP_P, M.EXEC_LAG = N, P, 1
    M.PYR_CAP, M.PYR_K, M.SHORT_WEIGHT = 2, 1, 0.0
    labels = list(ISA_DEF.keys())
    sectors = {}
    for lab in labels:
        sectors.setdefault(ISA_DEF[lab][2], []).append(lab)
    M.UNIVERSE, M.TICKERS = sectors, labels

    print("[1/3] 패널 + 백테스트...")
    high, low, close, ksC = build_krw_panel()
    cash_rate = M.load_cash_rate(close.index)
    sig = M.precompute_signals(high, low, close)
    res = M.backtest(high, low, close, sig, cash_rate)

    div = res["equity"][res["n_active"] > 0]
    gpos = gold_only_trend(high, low, close)
    gret = close[GOLD].pct_change()

    # 공통 기간 = 금 데이터 시작 이후 (분산도 유효한 구간)
    start = max(div.index[0], close[GOLD].dropna().index[0])
    idx = close.loc[start:].index
    idx = idx.intersection(div.index)

    bh_eq = (1 + gret.reindex(idx).fillna(0)).cumprod()
    gtr_eq = (1 + (gpos.shift(1) * gret).reindex(idx).fillna(0)).cumprod()
    div_eq = (div.reindex(idx) / div.reindex(idx).iloc[0])
    for e in (bh_eq, gtr_eq, div_eq):
        e /= e.iloc[0]

    mb, mg, md = met(bh_eq), met(gtr_eq), met(div_eq)
    print(f"\n      기간 {idx[0].date()} ~ {idx[-1].date()}")
    print(f"{'':22}{'총수익':>9}{'CAGR':>8}{'변동성':>8}"
          f"{'Sharpe':>8}{'MDD':>8}")
    for nm, m in [("금 단순보유", mb), ("금 단독 추세", mg),
                  ("분산 바스켓 추세", md)]:
        print(f"{nm:22}{m['tot']*100:>8.0f}%{m['cagr']*100:>7.1f}%"
              f"{m['vol']*100:>7.1f}%{m['sharpe']:>8.2f}{m['mdd']*100:>7.0f}%")

    # 같은 위험(금 보유 변동성)으로 맞춘 누적
    tv = mb["vol"]
    bh_v = bh_eq
    gtr_v = volmatch(gtr_eq.pct_change().fillna(0), tv)
    div_v = volmatch(div_eq.pct_change().fillna(0), tv)

    print("\n[2/3] 차트...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                   gridspec_kw={"height_ratios": [3, 1.2]},
                                   sharex=True)
    C = {"bh": "#9ca3af", "g": "#D4AF37", "d": "#7c3aed"}
    ax1.plot(div_v.index, div_v, color=C["d"], lw=2.4,
             label="③ 분산 바스켓 + 추세 농법 (헤지펀드 추세전략)")
    ax1.plot(bh_v.index, bh_v, color=C["bh"], lw=1.7,
             label="① 금 단순보유 (Buy & Hold)")
    ax1.plot(gtr_v.index, gtr_v, color=C["g"], lw=1.7,
             label="② 금 단독 + 추세 농법")
    ax1.set_yscale("log")
    ax1.set_title("분산의 힘 — '같은 위험'으로 맞췄을 때 누적수익 "
                  "(금 보유 변동성 기준)", fontsize=14, fontweight="bold")
    ax1.set_ylabel("누적 배수 (로그, 시작=1.0)")
    ax1.legend(fontsize=11.5, loc="upper left")
    ax1.grid(alpha=0.3)
    box = (f"같은 위험 기준 위험대비수익(Sharpe)\n"
           f"  ③ 분산 추세 : {md['sharpe']:.2f}\n"
           f"  ① 금 단순보유: {mb['sharpe']:.2f}\n"
           f"  ② 금만 추세  : {mg['sharpe']:.2f}")
    ax1.text(0.985, 0.04, box, transform=ax1.transAxes, va="bottom",
             ha="right", fontsize=11,
             bbox=dict(boxstyle="round", fc="#f5f3ff", ec=C["d"]))

    for nm, eq, c in [("③ 분산 추세", div_eq, C["d"]),
                      ("① 금 단순보유", bh_eq, C["bh"]),
                      ("② 금만 추세", gtr_eq, C["g"])]:
        dd = (eq / eq.cummax() - 1) * 100
        ax2.plot(dd.index, dd, color=c, lw=1.4, label=nm)
    ax2.set_title("낙폭(Drawdown) — 분산이 하락을 얼마나 줄이나",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("낙폭 (%)")
    ax2.legend(fontsize=10, ncol=3)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = BASE / "gold_diversification.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"      저장: {out}")
    print("  ※ KRW 프레임·프록시 가정. '같은 위험'=금 변동성에 스케일. "
          "투자권유 아님.")


if __name__ == "__main__":
    main()
