# -*- coding: utf-8 -*-
"""
손절 후 재투자 정책 비교 백테스트 (ISA 무레버리지)
=====================================================
같은 돈치안 진입/트레일링스톱/피라미딩(리스크패리티 목표비중)에 대해,
'비운 비중을 어떻게 처리하는가'만 바꿔 3정책을 비교한다.

  C (현행)      : 무레버리지 상한만 — gross>100%면 100%로 축소, gross<100%면
                  남은 비중은 현금(재투자 안 함).  ← 손절 시 현금 드래그
  A (전액재투자): 보유 종목을 항상 합 100%로 재정규화(빈 비중을 남은 종목에 비례 배분).
  B (재투자+상한): A와 같되 종목별 비중 상한(기본 25%)을 두고 초과분을 나머지에 재분배.

정책은 '비중 배분'만 바꾸며, 진입/청산/스톱 트리거는 가격·신호로 결정되어
정책과 무관하다(따라서 같은 보유셋·수익 시계열에 배분만 달리 적용 = 공정 비교).

출력: mulvaney_isa_redeploy.png, mulvaney_isa_redeploy_metrics.csv
실행: PYTHONIOENCODING=utf-8 python redeploy_backtest.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import mulvaney_replica as M
from mulvaney_isa_backtest import ISA_DEF, build_krw_panel
from isa_signals import PARAMS, SCHEME, _metrics

# 한글 폰트
for fp in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CAP_B = 0.25   # B 정책 종목별 비중 상한


def configure_engine():
    M.N_LOOKBACK = PARAMS["N"]; M.STOP_P = PARAMS["p"]
    M.EXEC_LAG = PARAMS["lag"]; M.PYR_CAP = PARAMS["cap"]
    M.PYR_K = PARAMS["K"]; M.SHORT_WEIGHT = PARAMS["short"]; M.SCHEME = SCHEME
    labels = list(ISA_DEF.keys())
    sectors = {}
    for lab in labels:
        sectors.setdefault(ISA_DEF[lab][2], []).append(lab)
    M.UNIVERSE = sectors
    M.TICKERS = labels
    return labels


def cap_norm(w, cap):
    """합 1로 정규화 후 종목별 상한 water-filling."""
    w = np.clip(w, 0, None).astype(float)
    s = w.sum()
    if s <= 0:
        return w
    w = w / s
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = (~over) & (w > 1e-12)
        us = w[under].sum()
        if under.sum() == 0 or us <= 0:
            break
        w[under] = w[under] + excess * (w[under] / us)
    return w


def pol_C(w):
    g = w.sum()
    return w / g if g > 1 else w            # 초과만 축소, 미달은 현금
def pol_A(w):
    g = w.sum()
    return w / g if g > 0 else w            # 항상 100%
def pol_B(w):
    g = w.sum()
    base = w / g if g > 0 else w
    return cap_norm(base, CAP_B)


def equity_from_policy(W, R, cr, pol):
    """W: 일별 원시 목표비중(DF), R: 일별 수익(DF), cr: 일별 현금금리(Series)."""
    P = np.vstack([pol(W.iloc[t].values) for t in range(len(W))])
    P = pd.DataFrame(P, index=W.index, columns=W.columns)
    sumP = P.sum(axis=1)
    port = (P.shift(1) * R).sum(axis=1) + (1 - sumP.shift(1)).clip(lower=0) * cr
    return (1 + port.fillna(0.0)).cumprod()


def main():
    labels = configure_engine()
    print("· 데이터 로드 + 시그널...")
    high, low, close, ksC = build_krw_panel()
    cr = M.load_cash_rate(close.index)
    sig = M.precompute_signals(high, low, close)
    print("· 백테스트(가중치 로깅)...")
    res = M.backtest(high, low, close, sig, cr, record_weights=True)

    W = res["weights"][labels]
    R = close[labels].pct_change().fillna(0.0)
    crs = pd.Series(cr, index=close.index)
    valid = res["n_active"] > 0
    idx = close.index[valid]

    curves = {}
    for name, pol in [("C 현행(현금드래그)", pol_C),
                      ("A 전액재투자", pol_A),
                      (f"B 재투자+상한{int(CAP_B*100)}%", pol_B)]:
        eq = equity_from_policy(W, R, crs, pol)
        eq = eq[valid]
        eq = eq / eq.iloc[0]
        curves[name] = eq
    # 벤치마크 KOSPI200
    ks = ksC.reindex(close.index).ffill()[valid]
    ks = ks / ks.dropna().iloc[0]
    curves["KOSPI200(참고)"] = ks

    # ── 지표(전체 + 최근1년) ──
    rows = []
    last1y = idx[-252:] if len(idx) > 252 else idx
    for name, eq in curves.items():
        m = _metrics(eq.dropna())
        eq1 = eq.loc[last1y].dropna()
        eq1 = eq1 / eq1.iloc[0]
        m1 = _metrics(eq1)
        rows.append({
            "정책": name,
            "총수익배수": round(eq.iloc[-1] / eq.iloc[0], 2),
            "CAGR": round(m["CAGR"] * 100, 1),
            "Sharpe": round(m["Sharpe"], 3),
            "MDD": round(m["MDD"] * 100, 1),
            "Calmar": round(m["Calmar"], 3),
            "최근1y수익%": round((eq1.iloc[-1] - 1) * 100, 1),
            "최근1y_MDD%": round(m1["MDD"] * 100, 1),
        })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 92)
    print(f"재투자 정책 비교 — 기간 {idx[0].date()} ~ {idx[-1].date()} "
          f"({len(idx)} 거래일)")
    print("=" * 92)
    print(df.to_string(index=False))
    df.to_csv("mulvaney_isa_redeploy_metrics.csv", index=False,
              encoding="utf-8-sig")

    # ── 차트: 자본곡선(로그) + 낙폭 ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                   gridspec_kw={"height_ratios": [2, 1]})
    colmap = {"C 현행(현금드래그)": "#8894a3",
              "A 전액재투자": "#2a78d6",
              f"B 재투자+상한{int(CAP_B*100)}%": "#1a8f5a",
              "KOSPI200(참고)": "#c9820a"}
    for name, eq in curves.items():
        ls = "--" if "KOSPI" in name else "-"
        lw = 1.6 if "KOSPI" in name else 2.1
        ax1.plot(eq.index, eq.values, label=name,
                 color=colmap[name], ls=ls, lw=lw)
    ax1.set_yscale("log")
    ax1.set_title("손절 후 재투자 정책 비교 — 자본곡선(로그) · ISA 무레버리지",
                  fontsize=15, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)
    for name, eq in curves.items():
        if "KOSPI" in name:
            continue
        dd = (eq / eq.cummax() - 1) * 100
        ax2.plot(dd.index, dd.values, label=name, color=colmap[name], lw=1.6)
    ax2.set_title("낙폭(Drawdown, %)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=10)
    fig.tight_layout()
    fig.savefig("mulvaney_isa_redeploy.png", dpi=130)
    print("\n저장: mulvaney_isa_redeploy.png, mulvaney_isa_redeploy_metrics.csv")


if __name__ == "__main__":
    main()
