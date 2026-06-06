# -*- coding: utf-8 -*-
"""
Mulvaney 복제 — 한국 ISA 롱온리 백테스트 (손절 청산) + 손절선 시각화
=======================================================================
한국 개인투자자 실현 가능 형태:
  - 롱 포지션만 실행 (숏 미실행, short_weight=0)
  - 청산은 손절선(초기 고정 → 미드라인 추적)으로만
  - 무레버리지(현금계좌) 비중 + 각 자산 손절매 기준선 표시

파라미터(RSI 강건설정 선택값, 2026-06-06):
  N=189, p=0.4, lag=2, cap=2, K=1, short=0, LP

벤치마크: KOSPI200 매수보유(KRW). 환처리는 isa_backtest와 동일.
실행: PYTHONIOENCODING=utf-8 python mulvaney_isa_longonly.py
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
from mulvaney_current_portfolio import compute_nolev, CLASS_COLORS

BASE_DIR = Path(__file__).resolve().parent
TRADING_DAYS = 252
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── ISA 롱온리 강건 파라미터 ──
PARAMS = dict(N=252, p=0.4, lag=1, cap=2, K=1, short=0.0)

# ISA 라벨 → 국내상장 ETF (실행용)
KRX_NAME = {
    "S&P500(미국)": "KODEX 미국S&P500(379800)",
    "나스닥100": "TIGER 미국나스닥100(133690)",
    "러셀2000(H)": "KODEX 미국러셀2000H(280930)",
    "신흥국(H)": "PLUS 신흥국MSCI합성H(195980)",
    "KOSPI200": "KODEX 200(069500)",
    "미국채30년": "PLUS 미국채30년액티브(464470)",
    "미국채10년": "TIGER 미국채10년선물(305080)",
    "금(H)": "KODEX 골드선물H(132030)",
    "은(H)": "KODEX 은선물H(144600)",
    "WTI원유(H)": "KODEX WTI원유선물H(261220)",
    "농산물(H)": "KODEX 3대농산물선물H(271060)",
    "미국달러선물": "KODEX 미국달러선물(261240)",
}


def metrics(eq):
    r = eq.pct_change().fillna(0)
    n = len(eq)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (TRADING_DAYS / n) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sh = (r.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return {"총수익률": eq.iloc[-1] / eq.iloc[0] - 1, "CAGR": cagr,
            "변동성(연)": vol, "Sharpe": sh, "MDD": mdd,
            "Calmar": cagr / abs(mdd) if mdd else np.nan}


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v*100:.1f}%" if pct else f"{v:.2f}"


def main():
    # 엔진 파라미터 오버라이드 (롱온리 강건값)
    M.N_LOOKBACK = PARAMS["N"]
    M.STOP_P = PARAMS["p"]
    M.EXEC_LAG = PARAMS["lag"]
    M.PYR_CAP = PARAMS["cap"]
    M.PYR_K = PARAMS["K"]
    M.SHORT_WEIGHT = PARAMS["short"]

    labels = list(ISA_DEF.keys())
    sectors = {}
    for lab in labels:
        sectors.setdefault(ISA_DEF[lab][2], []).append(lab)
    M.UNIVERSE = sectors
    M.TICKERS = labels

    print("[1/3] KRW 패널 + 롱온리 백테스트...")
    print(f"      파라미터: {PARAMS}")
    high, low, close, ksC = build_krw_panel()
    cash_rate = M.load_cash_rate(close.index)
    sig = M.precompute_signals(high, low, close)
    res = M.backtest(high, low, close, sig, cash_rate)

    valid = res["n_active"] > 0
    eq = res["equity"][valid]
    eq = eq / eq.iloc[0]
    asof = pd.Timestamp(res["asof"]).date()

    kospi = ksC.reindex(eq.index).ffill()
    kospi = kospi / kospi.iloc[0]
    sr, kr = eq.pct_change().fillna(0), kospi.pct_change().fillna(0)
    scal = kr.std() / sr.std() if sr.std() > 0 else 1.0
    eq_sc = (1 + sr * scal).cumprod()

    print("[2/3] 성과")
    curves = {"ISA 롱온리(raw)": eq, "ISA 롱온리(변동성매칭)": eq_sc,
              "KOSPI200 매수보유": kospi}
    metr = {k: metrics(v) for k, v in curves.items()}
    print(f"      평가기간 {eq.index[0].date()}~{eq.index[-1].date()} | "
          f"매칭 ×{scal:.2f}")
    cols = ["총수익률", "CAGR", "변동성(연)", "Sharpe", "MDD", "Calmar"]
    pct = {"총수익률", "CAGR", "변동성(연)", "MDD"}
    print("=" * 76)
    print(f"{'전략':<22}" + "".join(f"{c:>9}" for c in cols))
    for name, m in metr.items():
        print(f"{name:<22}" +
              "".join(f"{fmt(m[c], c in pct):>9}" for c in cols))
    print("=" * 76)
    mv, bh = metr["ISA 롱온리(변동성매칭)"], metr["KOSPI200 매수보유"]
    print(f"  [KOSPI200 대비] Sharpe {'이김' if mv['Sharpe']>bh['Sharpe'] else '짐'}"
          f" ({fmt(mv['Sharpe'])} vs {fmt(bh['Sharpe'])}) | "
          f"MDD {'우위' if mv['MDD']>bh['MDD'] else '열위'} "
          f"({fmt(mv['MDD'],True)} vs {fmt(bh['MDD'],True)})")
    tr = res["trades"]
    if len(tr):
        print(f"  거래 {len(tr)} | 승률 {(tr>0).mean()*100:.1f}% | "
              f"평균손익 {tr.mean()*100:.2f}% (손절로만 청산)")

    # ── 현재 보유 + 손절선 ──
    pos = res["positions"].copy()
    if pos.empty:
        print("\n  현재 보유 없음 (전 시장 관망).")
        return
    nl, cash_pct, _ = compute_nolev(pos)
    nl = nl.set_index("티커")["무레버리지비중%"]
    pos["ISA비중%"] = pos["티커"].map(nl).fillna(0)
    pos["국내ETF"] = pos["티커"].map(KRX_NAME)
    pos["진입대비%"] = ((pos["진입가"] / pos["현재가"] - 1) * 100).round(1)
    pos = pos.sort_values("ISA비중%", ascending=False).reset_index(drop=True)

    print(f"\n  [현재 롱 보유 + 손절선]  기준일 {asof}  현금 {cash_pct:.0f}%")
    cc = ["국내ETF", "유닛(m)", "진입일", "현재가", "현재스톱",
          "스톱여유%", "미실현%", "ISA비중%"]
    print(pos[cc].to_string(index=False))

    print("\n[3/3] 차트 저장...")
    pos.to_csv(BASE_DIR / "mulvaney_isa_lo_positions.csv",
               index=False, encoding="utf-8-sig")
    pd.DataFrame(metr).T.to_csv(BASE_DIR / "mulvaney_isa_lo_metrics.csv",
                               encoding="utf-8-sig")
    _plot(curves, pos, cash_pct, asof, BASE_DIR / "mulvaney_isa_longonly.png")
    print(f"      저장: {BASE_DIR}")
    print("  ※ 손절선=추적손절(미드라인). 종가기준·무슬리피지·프록시 가정. 투자권유 아님.")


def _plot(curves, pos, cash_pct, asof, path):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.3,
                          wspace=0.22)

    # (상단 전체) 누적성과
    ax0 = fig.add_subplot(gs[0, :])
    sty = {"ISA 롱온리(raw)": ("#c4b5fd", 1.1),
           "ISA 롱온리(변동성매칭)": ("#7c3aed", 2.0),
           "KOSPI200 매수보유": ("#dc2626", 1.5)}
    for name, e in curves.items():
        c, lw = sty.get(name, ("#333", 1.2))
        ax0.plot(e.index, e / e.iloc[0], lw=lw, label=name, color=c)
    ax0.set_yscale("log")
    ax0.set_title("ISA 롱온리(손절 청산) vs KOSPI200 — 누적성과(로그)",
                  fontsize=13, fontweight="bold")
    ax0.set_ylabel("누적 배수"); ax0.legend(fontsize=10); ax0.grid(alpha=0.3)

    # (하단좌) ISA 비중
    ax1 = fig.add_subplot(gs[1, 0])
    d = pos.sort_values("ISA비중%", ascending=True)
    colors = [CLASS_COLORS.get(c, "#999") for c in d["자산군"]]
    ax1.barh(d["티커"], d["ISA비중%"], color=colors)
    for i, v in enumerate(d["ISA비중%"]):
        ax1.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
    ax1.set_title(f"ISA 실행 비중 (무레버리지, 현금 {cash_pct:.0f}%)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("% of 자산"); ax1.grid(alpha=0.3, axis="x")

    # (하단우) 손절선 맵 — 현재가 대비 손절선 거리(%)
    ax2 = fig.add_subplot(gs[1, 1])
    ps = pos.sort_values("스톱여유%", ascending=False).reset_index(drop=True)
    for i, r in ps.iterrows():
        gap = r["스톱여유%"]
        col = ("#dc2626" if gap < 5 else "#f59e0b" if gap < 10 else "#16a34a")
        ax2.barh(i, -gap, color=col, alpha=0.85, height=0.6)
        ax2.text(-gap - 0.3, i, f"손절 {gap:.1f}%↓", va="center",
                 ha="right", fontsize=8.5)
        er = r["진입대비%"]
        ax2.plot(er, i, "o", color="#111", ms=5)
        ax2.text(er + 0.3, i, "진입", va="center", fontsize=7.5,
                 color="#111")
    ax2.set_yticks(range(len(ps)))
    ax2.set_yticklabels(ps["티커"], fontsize=9)
    ax2.invert_yaxis()
    ax2.axvline(0, color="#111", lw=1.2)
    ax2.text(0, -0.7, "현재가", ha="center", fontsize=8.5,
             fontweight="bold")
    ax2.set_title("손절매 기준 — 현재가 대비 손절선까지 거리(%)\n"
                  "(빨강<5% 임박 · 주황<10% · 초록 여유)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("현재가 대비 % (음수=손절선)")
    ax2.grid(alpha=0.3, axis="x")

    fig.suptitle(f"Mulvaney 복제 한국 ISA 롱온리  |  {asof}  |  "
                 f"N={PARAMS['N']} p={PARAMS['p']} 손절청산 only",
                 fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
