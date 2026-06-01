# -*- coding: utf-8 -*-
"""
한국 ISA 유니버스 전용 그리드서치 (손절 기준 재최적화)
==========================================================
기존 그리드서치는 US 유니버스 기준이었음. ISA 백테스트엔 US best-fit
(p=0.3 등)을 그대로 적용했을 뿐 ISA 환경에서 손절·파라미터를 재최적화한
적이 없음. → KRW·ISA 유니버스(12종목)에서 4,320 그리드를 다시 스윕.

핵심 질문:
  1) ISA에서 최적 손절폭 p는? (US는 평균적으로 p=0.4가 우월했음)
  2) 재최적화하면 KOSPI200(Sharpe 0.49)을 넘을 수 있나, 아니면 구조적으로 막혔나
  3) 상위 구성이 일관된가(강건) vs 흩어지나(과최적화)

데이터/엔진/환처리는 mulvaney_isa_backtest 와 동일. numba 코어 재사용.
실행: PYTHONIOENCODING=utf-8 python mulvaney_isa_gridsearch.py
"""
import time
import itertools
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
from mulvaney_gridsearch import (
    run_core, signals_for_N, metrics_from_eq,
    N_VALUES, P_VALUES, LAG_VALUES, CAP_VALUES, K_VALUES,
    SCHEME_VALUES, RISK_BUDGET, COST_BPS,
)

# 한국 개인은 공매도 불가 → short=0(롱온리) 추가.
# short=0: 하락신호 시 0주(현금)로 대기, 진짜 숏 이익 없음(가공수익 X).
SHORT_VALUES = [0.0, 0.5, 1.0]

BASE_DIR = Path(__file__).resolve().parent
TRADING_DAYS = 252
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def kospi_sharpe(ksC, index):
    k = ksC.reindex(index).ffill()
    r = k.pct_change().dropna()
    return (r.mean() * TRADING_DAYS) / (r.std() * np.sqrt(TRADING_DAYS))


def main():
    print("[1/4] KRW ISA 패널 구성...")
    labels = list(ISA_DEF.keys())
    sec_names = []
    for lab in labels:
        s = ISA_DEF[lab][2]
        if s not in sec_names:
            sec_names.append(s)
    sector_ids = np.array([sec_names.index(ISA_DEF[l][2])
                           for l in labels], dtype=np.int64)
    n_sectors = len(sec_names)

    high, low, close, ksC = build_krw_panel()
    C = close.values
    cash = M.load_cash_rate(close.index).values
    print(f"      {close.index[0].date()}~{close.index[-1].date()}, "
          f"{len(labels)} 시장, {n_sectors} 섹터")

    # warmup per N
    warmup_of = {}
    for N in N_VALUES:
        dh = high.rolling(N).max().shift(1).values
        warmup_of[N] = int(np.argmax(~np.all(np.isnan(dh), axis=1)))

    # KOSPI200 벤치마크 Sharpe (평가창)
    eval_idx = close.index[warmup_of[min(N_VALUES)]:]
    ks_sh = kospi_sharpe(ksC, eval_idx)
    print(f"      KOSPI200 매수보유 Sharpe(벤치마크): {ks_sh:.3f}")

    print("[2/4] 컴파일 + 일관성 체크...")
    sig0 = signals_for_N(high, low, close, 126, 1)
    eq0 = run_core(C, *sig0, cash, sector_ids, n_sectors,
                   0.3, 2.0, 1.0, 1.0, 0, RISK_BUDGET, COST_BPS, 1.0)
    chk = metrics_from_eq(eq0, warmup_of[126])
    print(f"      p=0.3 default Sharpe={chk[3]:.3f} "
          f"(isa_backtest 0.35 기대)")

    print("[3/4] 4,320 그리드 스윕 (ISA 유니버스)...")
    combos = list(itertools.product(
        N_VALUES, P_VALUES, LAG_VALUES, CAP_VALUES,
        K_VALUES, SHORT_VALUES, SCHEME_VALUES))
    sigcache = {}
    rows = []
    t0 = time.time()
    for i, (N, p, lag, cap, K, sw, sch) in enumerate(combos, 1):
        key = (N, lag)
        if key not in sigcache:
            sigcache[key] = signals_for_N(high, low, close, N, lag)
        eq = run_core(C, *sigcache[key], cash, sector_ids, n_sectors,
                      float(p), float(cap), float(K), float(sw),
                      int(sch), RISK_BUDGET, COST_BPS, 1.0)
        tot, cagr, vol, sh, mdd, cal = metrics_from_eq(eq, warmup_of[N])
        rows.append(dict(N=N, p=p, lag=lag, cap=cap, K=K, short=sw,
                         scheme="HLP" if sch else "LP", CAGR=cagr,
                         vol=vol, Sharpe=sh, MDD=mdd, Calmar=cal))
        if i % 1440 == 0:
            print(f"      {i}/{len(combos)} ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows).sort_values(
        "Sharpe", ascending=False).reset_index(drop=True)
    df.to_csv(BASE_DIR / "mulvaney_isa_grid_results.csv",
              index=False, encoding="utf-8-sig")
    print(f"      완료 {len(df)}개, {time.time()-t0:.0f}s")

    # ── 분석 ──
    print("\n[4/4] 분석")
    print("=" * 76)
    best = df.iloc[0]
    n_beat = int((df["Sharpe"] > ks_sh).sum())
    print(f"  KOSPI200 벤치마크 Sharpe = {ks_sh:.3f}")
    print(f"  최고 ISA 구성 Sharpe = {best['Sharpe']:.3f}  "
          f"(N={best['N']} p={best['p']} lag={best['lag']} "
          f"cap={best['cap']} K={best['K']} short={best['short']} "
          f"{best['scheme']})")
    print(f"  → KOSPI200를 넘는 구성: {n_beat}/{len(df)} "
          f"({n_beat/len(df)*100:.1f}%)")
    print(f"  Sharpe 분포: 최고 {df.Sharpe.max():.3f} / 중앙 "
          f"{df.Sharpe.median():.3f} / 최저 {df.Sharpe.min():.3f}")

    print("\n  [숏비중 별 — 롱온리(0) vs 반숏(0.5) vs 풀숏(1.0)]")
    for sw in SHORT_VALUES:
        sub = df[df["short"] == sw]["Sharpe"]
        tag = "롱온리(개인가능)" if sw == 0 else ("반숏" if sw == 0.5
                                                  else "풀숏")
        print(f"   short={sw} {tag:14}: 평균 {sub.mean():.3f} | "
              f"최고 {sub.max():.3f} | >KOSPI {int((sub>ks_sh).sum())}개")

    # 롱온리 최적 구성
    lo = df[df["short"] == 0.0].sort_values(
        "Sharpe", ascending=False)
    b = lo.iloc[0]
    print(f"\n  ★ 롱온리 최적: Sharpe {b['Sharpe']:.3f} "
          f"(N={b['N']} p={b['p']} lag={b['lag']} cap={b['cap']} "
          f"K={b['K']} {b['scheme']})  vs KOSPI200 {ks_sh:.3f} → "
          f"{'이김' if b['Sharpe']>ks_sh else '짐'}")
    print(f"     롱온리 중 KOSPI200 초과: "
          f"{int((lo['Sharpe']>ks_sh).sum())}/{len(lo)}개")
    best_half = df[df['short'] == 0.5]['Sharpe'].max()
    print(f"     숏 제거 비용: 반숏최고 {best_half:.3f} → 롱온리최고 "
          f"{b['Sharpe']:.3f} (차이 {best_half-b['Sharpe']:+.3f})")

    print("\n  [손절폭 p 별 — ISA에서 최적 손절 기준]")
    for p in P_VALUES:
        sub = df[df.p == p]["Sharpe"]
        print(f"   p={p}: 평균 {sub.mean():.3f} | 최고 {sub.max():.3f} | "
              f">KOSPI {int((sub>ks_sh).sum())}개")

    print("\n  [전 파라미터 한계효과 (값별 평균 Sharpe)]")
    for col in ["N", "p", "lag", "cap", "K", "short", "scheme"]:
        g = df.groupby(col)["Sharpe"].mean()
        print(f"   {col:7}: 최적={g.idxmax()} 스프레드={g.max()-g.min():.3f}"
              f"  " + " ".join(f"{k}:{v:.2f}" for k, v in g.items()))

    print("\n  [Top 8]")
    show = ["N", "p", "lag", "cap", "K", "short", "scheme",
            "Sharpe", "CAGR", "MDD"]
    t = df.head(8)[show].copy()
    t["CAGR"] = (t.CAGR * 100).round(1)
    t["MDD"] = (t.MDD * 100).round(1)
    t["Sharpe"] = t.Sharpe.round(3)
    print(t.to_string(index=False))

    _plot(df, ks_sh, BASE_DIR / "mulvaney_isa_grid.png")
    print(f"\n  저장: {BASE_DIR}")


def _plot(df, ks_sh, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5),
                                   gridspec_kw={"wspace": 0.25})
    ax1.hist(df["Sharpe"].dropna(), bins=60, color="#7c3aed", alpha=0.8)
    ax1.axvline(ks_sh, color="#dc2626", lw=2,
                label=f"KOSPI200 매수보유 ({ks_sh:.2f})")
    ax1.axvline(df["Sharpe"].median(), color="#111", lw=1.2, ls="--",
                label=f"그리드 중앙 ({df['Sharpe'].median():.2f})")
    ax1.set_title("ISA 4,320 그리드 Sharpe 분포 vs KOSPI200",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Sharpe"); ax1.set_ylabel("조합 수")
    ax1.legend(); ax1.grid(alpha=0.3)

    g = df.groupby("short")["Sharpe"].agg(["mean", "max"])
    x = np.arange(len(g))
    ax2.bar(x - 0.2, g["mean"], 0.4, label="평균", color="#7c3aed")
    ax2.bar(x + 0.2, g["max"], 0.4, label="최고", color="#a78bfa")
    ax2.axhline(ks_sh, color="#dc2626", ls="--", lw=1.5,
                label=f"KOSPI200 ({ks_sh:.2f})")
    ax2.set_xticks(x)
    lbl = {0.0: "롱온리\n(개인가능)", 0.5: "반숏", 1.0: "풀숏"}
    ax2.set_xticklabels([lbl.get(s, str(s)) for s in g.index])
    ax2.set_title("숏비중 별 Sharpe (평균/최고)", fontsize=12,
                  fontweight="bold")
    ax2.legend(); ax2.grid(alpha=0.3, axis="y")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
