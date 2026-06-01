# -*- coding: utf-8 -*-
"""
Mulvaney 복제 — 4,320 그리드서치 (과최적화/강건성 검증)
==========================================================
원본 연구의 4,320 합성 CTA 그리드를 그대로 스윕하되, Mulvaney 실수익률이
없으므로 R² 대신 **크로스애셋 ETF 바스켓에서의 Sharpe**로 랭킹.

목적:
  1) Mulvaney best-fit(N=126,p0.3,lag1,cap2,K1,short1.0,LP)이 상위에
     클러스터되는가(=강건) vs 외톨이 피크인가(=과최적화 위험)
  2) 어떤 파라미터가 성과를 실제로 좌우하는가(한계효과)
  3) 상위 구성들이 일관된 값으로 모이는가(=신호) vs 흩어지는가(=노이즈)

그리드 4,320 = N(10) × p(3) × lag(3) × cap(3) × K(4) × short(2) × scheme(2)
  N      = 126..315 (21일 간격)
  p      = 0.2/0.3/0.4
  lag    = 0/1/2
  cap    = 2/3/4
  K      = 1/2/3/4
  short  = 0.5/1.0
  scheme = LP(Equal-Market) / HLP(Hierarchical)

속도: numba JIT 코어 → 4,320 런 ~1분
산출물: 랭킹 CSV + 강건성 차트(분포/한계효과/히트맵)

실행: PYTHONIOENCODING=utf-8 python mulvaney_gridsearch.py
"""
import os
import time
import itertools
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numba import njit

import mulvaney_replica as M

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
TRADING_DAYS = 252

# ── 그리드 정의 ──
N_VALUES = list(range(126, 316, 21))      # 10
P_VALUES = [0.2, 0.3, 0.4]                # 3
LAG_VALUES = [0, 1, 2]                    # 3
CAP_VALUES = [2, 3, 4]                    # 3
K_VALUES = [1, 2, 3, 4]                   # 4
SHORT_VALUES = [0.5, 1.0]                 # 2
SCHEME_VALUES = [0, 1]                    # 0=LP, 1=HLP   2

RISK_BUDGET = 0.15
CASH_SPREAD = 0.01
COST_BPS = 1.0

BESTFIT = dict(N=126, p=0.3, lag=1, cap=2, K=1, short=1.0, scheme=0)


# ──────────────────────────────────────────────────────────────
# numba 백테스트 코어 (mulvaney_replica.backtest 와 동일 로직)
# ──────────────────────────────────────────────────────────────
@njit(cache=True, fastmath=False)
def run_core(C, dHigh, dLow, width, midline, eL, eS, cash_rate,
             sector_ids, n_sectors, p, cap, K, short_w, scheme,
             risk_budget, cost_bps, start_equity):
    T, Mk = C.shape
    direction = np.zeros(Mk)
    entry_px = np.zeros(Mk)
    init_stop = np.zeros(Mk)
    trail_stop = np.zeros(Mk)
    base_shares = np.zeros(Mk)
    mult = np.ones(Mk)
    signed_prev = np.zeros(Mk)
    sec_count = np.zeros(n_sectors)

    equity = start_equity
    eq = np.empty(T)

    for t in range(T):
        # 1) 전일 포지션 손익 + 담보현금
        if t > 0:
            pnl = 0.0
            for j in range(Mk):
                if signed_prev[j] != 0.0:
                    dpx = C[t, j] - C[t - 1, j]
                    if not np.isnan(dpx):
                        pnl += signed_prev[j] * dpx
            equity += pnl + equity * cash_rate[t]
        eq[t] = equity

        # 2) 활성 시장 + 섹터별 활성 카운트
        Nm = 0
        for s in range(n_sectors):
            sec_count[s] = 0.0
        for j in range(Mk):
            if not np.isnan(dHigh[t, j]) and not np.isnan(C[t, j]):
                Nm += 1
                sec_count[sector_ids[j]] += 1.0
        if Nm == 0:
            for j in range(Mk):
                signed_prev[j] = 0.0
            continue
        n_sec_active = 0
        for s in range(n_sectors):
            if sec_count[s] > 0:
                n_sec_active += 1

        # 3) 시장별 신호/스톱/피라미딩
        for j in range(Mk):
            if np.isnan(dHigh[t, j]) or np.isnan(C[t, j]):
                continue
            c = C[t, j]
            midl = midline[t, j]
            # 사이징 분모 (LP vs HLP)
            if scheme == 0:
                divisor = Nm
            else:
                divisor = n_sec_active * sec_count[sector_ids[j]]

            if direction[j] == 0.0:
                # 진입 (지연 적용)
                tl = t - 0  # placeholder
                long_sig = False
                short_sig = False
                # lag 는 외부에서 eL/eS 가 이미 시프트됨
                if eL[t, j] > 0:
                    long_sig = True
                elif eS[t, j] > 0:
                    short_sig = True
                if long_sig and width[t, j] > 0:
                    risk = c - (dHigh[t, j] - p * width[t, j])
                    if risk > 0:
                        direction[j] = 1.0
                        entry_px[j] = c
                        init_stop[j] = dHigh[t, j] - p * width[t, j]
                        trail_stop[j] = init_stop[j]
                        base_shares[j] = (risk_budget * equity / divisor) / risk
                        mult[j] = 1.0
                elif short_sig and width[t, j] > 0:
                    risk = (dLow[t, j] + p * width[t, j]) - c
                    if risk > 0:
                        direction[j] = -1.0
                        entry_px[j] = c
                        init_stop[j] = dLow[t, j] + p * width[t, j]
                        trail_stop[j] = init_stop[j]
                        base_shares[j] = (short_w * risk_budget
                                          * equity / divisor) / risk
                        mult[j] = 1.0
            elif direction[j] == 1.0:
                if c > entry_px[j] and not np.isnan(midl):
                    if midl > trail_stop[j]:
                        trail_stop[j] = midl
                if c <= trail_stop[j]:
                    direction[j] = 0.0
                    base_shares[j] = 0.0
                    mult[j] = 1.0
                else:
                    denom = entry_px[j] - init_stop[j]
                    r = (c - entry_px[j]) / denom if denom > 0 else 0.0
                    if r < 0:
                        r = 0.0
                    m_new = 1.0 + np.floor(r / K)
                    if m_new > cap:
                        m_new = cap
                    if m_new > mult[j]:
                        mult[j] = m_new
            else:
                if c < entry_px[j] and not np.isnan(midl):
                    if midl < trail_stop[j]:
                        trail_stop[j] = midl
                if c >= trail_stop[j]:
                    direction[j] = 0.0
                    base_shares[j] = 0.0
                    mult[j] = 1.0
                else:
                    denom = init_stop[j] - entry_px[j]
                    r = (entry_px[j] - c) / denom if denom > 0 else 0.0
                    if r < 0:
                        r = 0.0
                    m_new = 1.0 + np.floor(r / K)
                    if m_new > cap:
                        m_new = cap
                    if m_new > mult[j]:
                        mult[j] = m_new

        # 4) 보유 갱신 + 체결비용
        for j in range(Mk):
            new_signed = direction[j] * base_shares[j] * mult[j]
            if cost_bps > 0 and not np.isnan(C[t, j]):
                dsh = new_signed - signed_prev[j]
                if dsh < 0:
                    dsh = -dsh
                if dsh > 0:
                    equity -= (cost_bps / 1e4) * dsh * C[t, j]
            signed_prev[j] = new_signed

    return eq


# ──────────────────────────────────────────────────────────────
# 준비 + 지표
# ──────────────────────────────────────────────────────────────
def prep():
    high, low, close, idx = M.load_universe()
    cash_rate = M.load_cash_rate(idx)
    tickers = M.TICKERS
    # 섹터 매핑
    sec_of = {}
    sec_names = list(M.UNIVERSE.keys())
    for si, (sec, tks) in enumerate(M.UNIVERSE.items()):
        for tk in tks:
            sec_of[tk] = si
    sector_ids = np.array([sec_of[t] for t in tickers], dtype=np.int64)
    return high, low, close, idx, cash_rate, sector_ids, len(sec_names)


def signals_for_N(high, low, close, N, lag):
    dHigh = high.rolling(N).max().shift(1)
    dLow = low.rolling(N).min().shift(1)
    width = dHigh - dLow
    midline = (dHigh + dLow) / 2.0
    eL = ((close >= dHigh) & (width > 0)).shift(lag).fillna(False)
    eS = ((close <= dLow) & (width > 0)).shift(lag).fillna(False)
    return (dHigh.values, dLow.values, width.values, midline.values,
            eL.values.astype(np.int8), eS.values.astype(np.int8))


def metrics_from_eq(eq_arr, warmup):
    eq = eq_arr[warmup:]
    eq = eq / eq[0]
    r = np.diff(eq) / eq[:-1]
    n = len(eq)
    total = eq[-1] - 1.0
    cagr = eq[-1] ** (TRADING_DAYS / n) - 1.0
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    peak = np.maximum.accumulate(eq)
    mdd = (eq / peak - 1.0).min()
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    return total, cagr, vol, sharpe, mdd, calmar


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    print("[1/5] 데이터 로드...")
    high, low, close, idx, cash_rate, sector_ids, n_sectors = prep()
    C = close.values
    cash = cash_rate.values
    print(f"      {idx[0].date()} ~ {idx[-1].date()}, {len(idx)}일, "
          f"{len(M.TICKERS)} 시장, {n_sectors} 섹터")

    # N별 warmup(첫 활성일) 사전계산
    warmup_of = {}
    for N in N_VALUES:
        dh = high.rolling(N).max().shift(1).values
        any_active = ~np.all(np.isnan(dh), axis=1)
        warmup_of[N] = int(np.argmax(any_active))

    print("[2/5] numba 코어 컴파일(첫 호출)...")
    sig0 = signals_for_N(high, low, close, 126, 1)
    t0 = time.time()
    _ = run_core(C, *sig0, cash, sector_ids, n_sectors,
                 0.3, 2.0, 1.0, 1.0, 0, RISK_BUDGET, COST_BPS, 1.0)
    print(f"      컴파일+1런: {time.time()-t0:.1f}s")

    # 일관성 체크: best-fit Sharpe ≈ mulvaney_replica raw(0.51)
    bf_eq = run_core(C, *signals_for_N(high, low, close, 126, 1), cash,
                     sector_ids, n_sectors, 0.3, 2.0, 1.0, 1.0, 0,
                     RISK_BUDGET, COST_BPS, 1.0)
    chk = metrics_from_eq(bf_eq, warmup_of[126])
    print(f"      일관성 체크 best-fit Sharpe={chk[3]:.2f} "
          f"CAGR={chk[1]*100:.1f}% (mulvaney_replica raw 0.51/14.3% 기대)")

    print("[3/5] 4,320 그리드 스윕...")
    rows = []
    t0 = time.time()
    cnt = 0
    # N,lag 별로 시그널 캐시 (30개)
    sigcache = {}
    combos = list(itertools.product(
        N_VALUES, P_VALUES, LAG_VALUES, CAP_VALUES,
        K_VALUES, SHORT_VALUES, SCHEME_VALUES))
    for (N, p, lag, cap, K, short_w, scheme) in combos:
        key = (N, lag)
        if key not in sigcache:
            sigcache[key] = signals_for_N(high, low, close, N, lag)
        sg = sigcache[key]
        eq = run_core(C, *sg, cash, sector_ids, n_sectors,
                      float(p), float(cap), float(K), float(short_w),
                      int(scheme), RISK_BUDGET, COST_BPS, 1.0)
        tot, cagr, vol, sh, mdd, cal = metrics_from_eq(eq, warmup_of[N])
        rows.append(dict(N=N, p=p, lag=lag, cap=cap, K=K, short=short_w,
                         scheme="HLP" if scheme else "LP",
                         total=tot, CAGR=cagr, vol=vol, Sharpe=sh,
                         MDD=mdd, Calmar=cal))
        cnt += 1
        if cnt % 720 == 0:
            print(f"      {cnt}/{len(combos)}  "
                  f"({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    print(f"      완료: {len(df)} 조합, {time.time()-t0:.0f}s")

    df = df.sort_values("Sharpe", ascending=False).reset_index(drop=True)
    df.to_csv(BASE_DIR / "mulvaney_grid_results.csv",
              index=False, encoding="utf-8-sig")

    # ── 분석 ──
    print("\n[4/5] 강건성 분석")
    print("=" * 74)
    # best-fit 순위
    bf = df[(df.N == 126) & (df.p == 0.3) & (df.lag == 1) & (df.cap == 2)
            & (df.K == 1) & (df.short == 1.0) & (df.scheme == "LP")]
    if len(bf):
        rank = bf.index[0] + 1
        print(f"  Mulvaney best-fit 순위: {rank}/{len(df)} "
              f"(상위 {rank/len(df)*100:.1f}%) | "
              f"Sharpe {bf.iloc[0]['Sharpe']:.3f}")
    print(f"  Sharpe 분포: 최고 {df.Sharpe.max():.3f} | "
          f"중앙 {df.Sharpe.median():.3f} | 최저 {df.Sharpe.min():.3f} | "
          f"표준편차 {df.Sharpe.std():.3f}")

    print("\n  [Top 10]")
    show = ["N", "p", "lag", "cap", "K", "short", "scheme",
            "Sharpe", "CAGR", "MDD"]
    top = df.head(10)[show].copy()
    top["CAGR"] = (top["CAGR"] * 100).round(1)
    top["MDD"] = (top["MDD"] * 100).round(1)
    top["Sharpe"] = top["Sharpe"].round(3)
    print(top.to_string(index=False))

    print("\n  [파라미터 한계효과 — 값별 평균 Sharpe]")
    for col in ["N", "p", "lag", "cap", "K", "short", "scheme"]:
        g = df.groupby(col)["Sharpe"].mean()
        spread = g.max() - g.min()
        best = g.idxmax()
        print(f"   {col:7}: 최적={best}  스프레드={spread:.3f}  "
              + " ".join(f"{k}:{v:.2f}" for k, v in g.items()))

    # 상위 5% 클러스터 일관성
    topq = df.head(max(1, len(df) // 20))
    print(f"\n  [상위 5%({len(topq)}개) 구성 빈도 — 클러스터 강건성]")
    for col in ["N", "p", "cap", "K", "short", "scheme"]:
        vc = topq[col].value_counts(normalize=True)
        dom = vc.index[0]
        print(f"   {col:7}: 최빈값 {dom} ({vc.iloc[0]*100:.0f}%)")

    print("\n[5/5] 차트 저장...")
    _plots(df, BASE_DIR)
    print(f"      저장: {BASE_DIR}")
    print("\n완료.")


def _plots(df, outdir):
    # 1) Sharpe 분포 + best-fit
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(df["Sharpe"].dropna(), bins=60, color="#7c3aed", alpha=0.8)
    bf = df[(df.N == 126) & (df.p == 0.3) & (df.lag == 1) & (df.cap == 2)
            & (df.K == 1) & (df.short == 1.0) & (df.scheme == "LP")]
    if len(bf):
        ax.axvline(bf.iloc[0]["Sharpe"], color="#dc2626", lw=2,
                   label=f"Mulvaney best-fit ({bf.iloc[0]['Sharpe']:.2f})")
    ax.axvline(df["Sharpe"].median(), color="#111", lw=1.2, ls="--",
               label=f"중앙값 ({df['Sharpe'].median():.2f})")
    ax.set_title("4,320 그리드 Sharpe 분포 — best-fit 위치",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Sharpe"); ax.set_ylabel("조합 수")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "mulvaney_grid_dist.png", dpi=130)
    plt.close(fig)

    # 2) 파라미터별 평균 Sharpe (한계효과)
    params = ["N", "p", "lag", "cap", "K", "short", "scheme"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, params):
        g = df.groupby(col)["Sharpe"].mean()
        ax.bar([str(x) for x in g.index], g.values, color="#7c3aed")
        ax.set_title(f"{col} 별 평균 Sharpe", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(labelsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle("파라미터 한계효과 (값별 평균 Sharpe)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "mulvaney_grid_marginal.png", dpi=130)
    plt.close(fig)

    # 3) N × p 히트맵 (나머지 평균)
    piv = df.pivot_table(index="N", columns="p", values="Sharpe",
                         aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis",
                   origin="lower")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel("Stop Fraction p"); ax.set_ylabel("Donchian Lookback N")
    ax.set_title("N × p 평균 Sharpe 히트맵", fontsize=12, fontweight="bold")
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center",
                    va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "mulvaney_grid_heatmap.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
