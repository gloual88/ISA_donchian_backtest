# -*- coding: utf-8 -*-
"""
ISA 추세전략 — 강건성 검증 (walk-forward + 거래비용 민감도)
================================================================
1) Walk-forward: 매 테스트연도마다 '직전 데이터'로 파라미터를 재선택(in-sample)
   → 그 파라미터로 '다음 1년'을 거래(out-of-sample) → OOS 수익 누적.
   과최적화면 OOS 성과가 in-sample보다 크게 무너진다.
2) 거래비용 민감도: 고정 파라미터로 COST_BPS를 0~20bp 변화시켜 Sharpe/MDD 추적.
   엣지가 현실적 비용에서도 살아남는지 확인.

모두 KRW ISA 패널(롱온리). 합성 아님 — 실제 프록시 데이터.
실행: PYTHONIOENCODING=utf-8 python validate_robustness.py
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
from mulvaney_gridsearch import run_core, signals_for_N, RISK_BUDGET

BASE = Path(__file__).resolve().parent
TD = 252
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 롱온리 walk-forward 재선택 그리드 (작게 — 과최적화 자체를 검증)
WF_N = [126, 168, 210, 252, 294]
WF_P = [0.3, 0.4]


def sharpe(ret):
    v = ret.std() * np.sqrt(TD)
    return (ret.mean() * TD) / v if v > 0 else np.nan


def met(eq):
    r = eq.pct_change().fillna(0)
    n = len(eq)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (TD / n) - 1
    mdd = (eq / eq.cummax() - 1).min()
    return dict(CAGR=cagr, vol=r.std()*np.sqrt(TD), Sharpe=sharpe(r),
                MDD=mdd, Calmar=cagr/abs(mdd) if mdd else np.nan)


def setup_engine(cost_bps):
    M.N_LOOKBACK, M.STOP_P, M.EXEC_LAG = 252, 0.4, 1
    M.PYR_CAP, M.PYR_K, M.SHORT_WEIGHT = 2, 1, 0.0
    M.COST_BPS = cost_bps
    labels = list(ISA_DEF.keys())
    sectors = {}
    for lab in labels:
        sectors.setdefault(ISA_DEF[lab][2], []).append(lab)
    M.UNIVERSE, M.TICKERS = sectors, labels
    return labels


# ──────────────────────────────────────────────────────────────
def cost_sensitivity(high, low, close, cash_rate):
    print("\n[거래비용 민감도] (고정 N=252 p=0.4 롱온리)")
    print(f"{'비용(bp)':>8}{'CAGR':>9}{'Sharpe':>9}{'MDD':>9}{'총수익':>10}")
    rows = []
    for bps in [0, 1, 2, 5, 10, 20]:
        setup_engine(bps)
        sig = M.precompute_signals(high, low, close)
        res = M.backtest(high, low, close, sig, cash_rate)
        eq = res["equity"][res["n_active"] > 0]
        eq = eq / eq.iloc[0]
        m = met(eq)
        rows.append((bps, m))
        print(f"{bps:>8}{m['CAGR']*100:>8.1f}%{m['Sharpe']:>9.2f}"
              f"{m['MDD']*100:>8.0f}%{(eq.iloc[-1]-1)*100:>9.0f}%")
    return rows


# ──────────────────────────────────────────────────────────────
def walk_forward(high, low, close, ksC, sector_ids, n_sectors, cash):
    print("\n[Walk-forward] 직전 데이터로 파라미터 재선택 → 다음 해 OOS 거래")
    idx = close.index
    years = sorted(set(idx.year))
    # 최소 5년 학습 후 테스트 시작
    test_years = [y for y in years if y >= years[0] + 5]
    C = close.values

    oos_ret = pd.Series(0.0, index=idx)
    picks = []
    for ty in test_years:
        train_mask = idx.year < ty
        if train_mask.sum() < 252 * 3:
            continue
        # 학습구간에서 최적 (N,p) by Sharpe
        best, best_sh = None, -9
        for N in WF_N:
            sg = signals_for_N(high, low, close, N, 1)
            for p in WF_P:
                eq = run_core(C, *sg, cash, sector_ids, n_sectors,
                              float(p), 2.0, 1.0, 0.0, 0,
                              RISK_BUDGET, 1.0, 1.0)
                eqs = pd.Series(eq, index=idx)[train_mask]
                sh = sharpe(eqs.pct_change().fillna(0))
                if sh > best_sh:
                    best_sh, best = sh, (N, p)
        # 그 파라미터로 전체 실행 후 테스트연도 수익만 추출 (OOS)
        N, p = best
        sg = signals_for_N(high, low, close, N, 1)
        eqf = pd.Series(run_core(C, *sg, cash, sector_ids, n_sectors,
                                 float(p), 2.0, 1.0, 0.0, 0,
                                 RISK_BUDGET, 1.0, 1.0), index=idx)
        ty_mask = idx.year == ty
        oos_ret[ty_mask] = eqf.pct_change().fillna(0)[ty_mask].values
        picks.append((ty, N, p))

    valid = oos_ret.index.year >= test_years[0]
    oos_eq = (1 + oos_ret[valid]).cumprod()
    oos_eq = oos_eq / oos_eq.iloc[0]

    # 비교: 고정 파라미터(252,0.4)를 같은 OOS 구간에
    sgf = signals_for_N(high, low, close, 252, 1)
    fixed = pd.Series(run_core(C, *sgf, cash, sector_ids, n_sectors,
                               0.4, 2.0, 1.0, 0.0, 0, RISK_BUDGET, 1.0, 1.0),
                      index=idx)
    fixed_eq = (1 + fixed.pct_change().fillna(0)[valid]).cumprod()
    fixed_eq = fixed_eq / fixed_eq.iloc[0]
    kospi = ksC.reindex(oos_eq.index).ffill()
    kospi = kospi / kospi.iloc[0]

    print(f"  테스트구간: {oos_eq.index[0].date()} ~ {oos_eq.index[-1].date()}")
    print(f"  연도별 선택(N,p): " +
          " ".join(f"{y}:{N}/{p}" for y, N, p in picks[:8]) + " ...")
    for nm, eq in [("Walk-forward(OOS)", oos_eq),
                   ("고정 N252/p0.4", fixed_eq),
                   ("KOSPI200", kospi)]:
        m = met(eq)
        print(f"  {nm:20} Sharpe {m['Sharpe']:.2f} | CAGR "
              f"{m['CAGR']*100:.1f}% | MDD {m['MDD']*100:.0f}%")
    return oos_eq, fixed_eq, kospi, picks


def main():
    print("[1/3] 패널 로드...")
    setup_engine(1.0)
    high, low, close, ksC = build_krw_panel()
    cash_rate = M.load_cash_rate(close.index)
    cash = cash_rate.values
    labels = list(ISA_DEF.keys())
    sec_names = []
    for lab in labels:
        if ISA_DEF[lab][2] not in sec_names:
            sec_names.append(ISA_DEF[lab][2])
    sector_ids = np.array([sec_names.index(ISA_DEF[lab][2])
                           for lab in labels], dtype=np.int64)

    print("[2/3] 검증...")
    cost_rows = cost_sensitivity(high, low, close, cash_rate)
    oos_eq, fixed_eq, kospi, picks = walk_forward(
        high, low, close, ksC, sector_ids, len(sec_names), cash)

    print("\n[3/3] 차트...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={"wspace": 0.25})
    ax1.plot(oos_eq.index, oos_eq, color="#7c3aed", lw=2,
             label="Walk-forward (OOS·매년 재선택)")
    ax1.plot(fixed_eq.index, fixed_eq, color="#16a34a", lw=1.5,
             label="고정 파라미터")
    ax1.plot(kospi.index, kospi, color="#dc2626", lw=1.3, label="KOSPI200")
    ax1.set_yscale("log")
    ax1.set_title("Walk-forward 검증 — OOS vs 고정 vs KOSPI200",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("누적배수(로그)"); ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    bps = [r[0] for r in cost_rows]
    shs = [r[1]["Sharpe"] for r in cost_rows]
    cagrs = [r[1]["CAGR"]*100 for r in cost_rows]
    ax2.plot(bps, shs, "o-", color="#7c3aed", label="Sharpe")
    ax2b = ax2.twinx()
    ax2b.plot(bps, cagrs, "s--", color="#f59e0b", label="CAGR%")
    ax2.set_xlabel("편도 거래비용 (bp)")
    ax2.set_ylabel("Sharpe", color="#7c3aed")
    ax2b.set_ylabel("CAGR %", color="#f59e0b")
    ax2.set_title("거래비용 민감도", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.3)
    fig.savefig(BASE / "validate_robustness.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print(f"      저장: {BASE/'validate_robustness.png'}")


if __name__ == "__main__":
    main()
