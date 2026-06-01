# -*- coding: utf-8 -*-
"""
Mulvaney CTA 역설계 복제 — 크로스애셋 ETF 바스켓
========================================================
사용자 제공 역설계 스펙(Mulvaney Capital 합성 복제)을 그대로 구현.
단일 종목이 아니라 자산군 분산 ETF 바스켓에 적용 → loss-parity·숏·
피라미딩이 본연대로 작동.

[트레이딩 룰 — 문서 그대로]
  ENTRY        Long: Close >= N일 Donchian High / Short: Close <= N일 Low
  INIT STOP    Width = High - Low
               Long  stop = High - p*Width  /  Short stop = Low + p*Width
  TRAIL STOP   수익 전환 후 Midline=(High+Low)/2 가 스톱 인계 (래칫)
  SIZING       Contracts = (15% * Equity / 시장수) / DollarStopRisk  (Equal-Market LP)
  PYRAMIDING   r = 미실현이익 / 초기스톱리스크(=R배수)
               m = min(cap, 1 + floor(max(0,r)/K)),  포지션은 증가만
  CASH         포지션 P&L + 담보현금(Fama-French RF - 100bps)

[Mulvaney best-fit 기본 파라미터 (top-10 클러스터)]
  N=126, p=0.3, ExecLag=1, PyrCap=2, PyrK=1, ShortWeight=1.0, Equal-Market LP

벤치마크: SPY Buy&Hold, 동일가중 바스켓 Buy&Hold
산출물: 자본곡선/낙폭/노출 PNG + 지표 CSV

실행: PYTHONIOENCODING=utf-8 python mulvaney_replica.py
필요: yfinance pandas_datareader pandas numpy matplotlib certifi
"""
import os
import shutil
import tempfile
import math
from pathlib import Path

# ── SSL 우회 (yfinance import 전) ──
try:
    import certifi
    _ca = os.path.join(tempfile.gettempdir(), "cacert_mulv.pem")
    shutil.copy(certifi.where(), _ca)
    for _k in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        os.environ[_k] = _ca
except Exception as _e:
    print(f"[경고] SSL 우회 실패: {_e}")

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
TRADING_DAYS = 252

# ── 유니버스 (자산군 분산) ──
UNIVERSE = {
    "주식": ["SPY", "QQQ", "EFA", "EEM", "IWM", "EWY"],
    "금리": ["TLT", "IEF"],
    "원자재": ["GLD", "SLV", "USO", "DBA", "DBB", "DBC"],
    "통화": ["UUP", "FXE"],
}
TICKERS = [t for v in UNIVERSE.values() for t in v]

# ── Mulvaney best-fit 파라미터 ──
N_LOOKBACK = 126          # Donchian 룩백 (≈6개월)
STOP_P = 0.30             # 초기 고정스톱 분율
EXEC_LAG = 1              # 신호 후 실행 지연(일)
PYR_CAP = 2               # 피라미딩 상한 배수
PYR_K = 1                 # 피라미딩 K factor
SHORT_WEIGHT = 1.0        # 숏 비중 (100%)
RISK_BUDGET = 0.15        # 15% * Equity / 시장수
CASH_SPREAD = 0.01        # 담보현금 = RF - 100bps
COST_BPS = 1.0            # 체결비용 (편도, bp) — 조정 가능


# ──────────────────────────────────────────────────────────────
# 데이터
# ──────────────────────────────────────────────────────────────
def load_universe():
    raw = yf.download(TICKERS, start="1999-01-01", auto_adjust=True,
                      progress=False)
    high = raw["High"][TICKERS]
    low = raw["Low"][TICKERS]
    close = raw["Close"][TICKERS]
    idx = close.index
    return high, low, close, idx


def load_cash_rate(index) -> pd.Series:
    """
    담보현금 일수익률(소수) = 무위험수익률 - 100bps/252.
    FRED 3개월 국채(DTB3)를 CSV로 직접 호출(키 불필요). 실패 시 상수(연 3%).
    pandas_datareader 의존 제거 — Python 3.12+/배포환경 호환.
    """
    daily_spread = CASH_SPREAD / TRADING_DAYS
    const = pd.Series(0.03 / TRADING_DAYS, index=index)  # 폴백: 연 3%
    rf = None
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3"
        raw = pd.read_csv(url)
        raw.columns = ["date", "rate"]
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
        raw["rate"] = pd.to_numeric(raw["rate"], errors="coerce")
        s = raw.dropna().set_index("date")["rate"]
        if not s.empty:
            rf = (s / 100.0) / TRADING_DAYS    # 연율% → 일별 소수
    except Exception:
        rf = None
    if rf is None:
        rf = const
    rf = (rf.reindex(index.union(rf.index)).ffill()
          .reindex(index).fillna(0.03 / TRADING_DAYS))
    return (rf - daily_spread).rename("cash_rate")


# ──────────────────────────────────────────────────────────────
# 시그널 전처리 (벡터화)
# ──────────────────────────────────────────────────────────────
def precompute_signals(high, low, close):
    dHigh = high.rolling(N_LOOKBACK).max().shift(1)
    dLow = low.rolling(N_LOOKBACK).min().shift(1)
    width = dHigh - dLow
    midline = (dHigh + dLow) / 2.0
    entry_long = (close >= dHigh) & (width > 0)
    entry_short = (close <= dLow) & (width > 0)
    # 실행 지연
    entry_long = entry_long.shift(EXEC_LAG).fillna(False)
    entry_short = entry_short.shift(EXEC_LAG).fillna(False)
    return dict(dHigh=dHigh, dLow=dLow, width=width, midline=midline,
                entry_long=entry_long, entry_short=entry_short)


# ──────────────────────────────────────────────────────────────
# 포트폴리오 백테스트 (마크투마켓)
# ──────────────────────────────────────────────────────────────
def backtest(high, low, close, sig, cash_rate, start_equity=1.0):
    dates = close.index
    T = len(dates)
    cols = TICKERS

    C = close.values
    dHigh = sig["dHigh"].values
    dLow = sig["dLow"].values
    width = sig["width"].values
    midline = sig["midline"].values
    eL = sig["entry_long"].values
    eS = sig["entry_short"].values
    crate = cash_rate.values
    col_idx = {t: j for j, t in enumerate(cols)}

    # 시장별 상태
    direction = np.zeros(len(cols))     # +1/-1/0
    entry_px = np.full(len(cols), np.nan)
    init_stop = np.full(len(cols), np.nan)
    trail_stop = np.full(len(cols), np.nan)
    base_shares = np.zeros(len(cols))
    mult = np.ones(len(cols))
    signed_prev = np.zeros(len(cols))   # 전일 보유 (P&L 계산용)
    entry_idx = np.full(len(cols), -1, dtype=int)  # 진입일 인덱스

    equity = start_equity
    eq_curve = np.empty(T)
    gross_exp = np.zeros(T)
    n_long = np.zeros(T, dtype=int)
    n_short = np.zeros(T, dtype=int)
    n_active = np.zeros(T, dtype=int)
    trade_returns = []                  # 청산된 거래 수익(R 아님, % of entry notional)
    exits_today = []                    # 최종 바에서 손절 청산된 종목

    for t in range(T):
        # ── 1) 전일 포지션 손익 + 담보현금 ──
        if t > 0:
            pnl = 0.0
            for j in range(len(cols)):
                if signed_prev[j] != 0:
                    dpx = C[t, j] - C[t - 1, j]
                    if not math.isnan(dpx):
                        pnl += signed_prev[j] * dpx
            equity += pnl + equity * crate[t]
        eq_curve[t] = equity

        # ── 2) 활성 시장 수 (채널 형성된 시장) ──
        active = [j for j in range(len(cols))
                  if not math.isnan(dHigh[t, j]) and not math.isnan(C[t, j])]
        Nm = len(active)
        n_active[t] = Nm
        if Nm == 0:
            signed_prev[:] = 0
            continue

        # ── 3) 시장별 신호/스톱/피라미딩 갱신 ──
        for j in active:
            c = C[t, j]
            midl = midline[t, j]
            if direction[j] == 0:
                # 진입 (지연 적용된 신호)
                if eL[t, j]:
                    direction[j] = 1
                    entry_px[j] = c
                    init_stop[j] = dHigh[t, j] - STOP_P * width[t, j]
                    trail_stop[j] = init_stop[j]
                    risk = c - init_stop[j]
                    if risk > 0:
                        base_shares[j] = (RISK_BUDGET * equity / Nm) / risk
                        mult[j] = 1
                        entry_idx[j] = t
                    else:
                        direction[j] = 0
                elif eS[t, j]:
                    direction[j] = -1
                    entry_px[j] = c
                    init_stop[j] = dLow[t, j] + STOP_P * width[t, j]
                    trail_stop[j] = init_stop[j]
                    risk = init_stop[j] - c
                    if risk > 0:
                        base_shares[j] = (SHORT_WEIGHT * RISK_BUDGET
                                          * equity / Nm) / risk
                        mult[j] = 1
                        entry_idx[j] = t
                    else:
                        direction[j] = 0
            elif direction[j] == 1:
                # 롱 관리: 트레일링(수익 시 midline 래칫) → 청산 → 피라미딩
                if c > entry_px[j] and not math.isnan(midl):
                    trail_stop[j] = max(trail_stop[j], midl)
                if c <= trail_stop[j]:
                    trade_returns.append(c / entry_px[j] - 1.0)
                    if t == T - 1 and base_shares[j] > 0:
                        exits_today.append({
                            "ticker": cols[j],
                            "ret_pct": round((c / entry_px[j] - 1) * 100, 1),
                            "진입일": (dates[entry_idx[j]].date()
                                     if entry_idx[j] >= 0 else None),
                        })
                    direction[j] = 0
                    base_shares[j] = 0.0
                    mult[j] = 1
                    entry_idx[j] = -1
                else:
                    denom = entry_px[j] - init_stop[j]
                    r = (c - entry_px[j]) / denom if denom > 0 else 0.0
                    m_new = min(PYR_CAP, 1 + math.floor(max(0.0, r) / PYR_K))
                    mult[j] = max(mult[j], m_new)
            else:
                # 숏 관리
                if c < entry_px[j] and not math.isnan(midl):
                    trail_stop[j] = min(trail_stop[j], midl)
                if c >= trail_stop[j]:
                    trade_returns.append(entry_px[j] / c - 1.0)
                    direction[j] = 0
                    base_shares[j] = 0.0
                    mult[j] = 1
                    entry_idx[j] = -1
                else:
                    denom = init_stop[j] - entry_px[j]
                    r = (entry_px[j] - c) / denom if denom > 0 else 0.0
                    m_new = min(PYR_CAP, 1 + math.floor(max(0.0, r) / PYR_K))
                    mult[j] = max(mult[j], m_new)

        # ── 4) 보유 갱신 + 체결비용 + 노출 집계 ──
        gross = 0.0
        nl = ns = 0
        for j in range(len(cols)):
            new_signed = direction[j] * base_shares[j] * mult[j]
            # 체결비용 (보유주식 변화분)
            if COST_BPS > 0 and not math.isnan(C[t, j]):
                dshares = abs(new_signed - signed_prev[j])
                if dshares > 0:
                    equity -= (COST_BPS / 1e4) * dshares * C[t, j]
            signed_prev[j] = new_signed
            if new_signed != 0 and not math.isnan(C[t, j]):
                gross += abs(new_signed) * C[t, j]
                if new_signed > 0:
                    nl += 1
                else:
                    ns += 1
        gross_exp[t] = gross / equity if equity > 0 else 0.0
        n_long[t] = nl
        n_short[t] = ns

    # ── 최종일 보유 포지션 스냅샷 ──
    last_t = T - 1
    pos_rows = []
    sec_of = {tk: sec for sec, tks in UNIVERSE.items() for tk in tks}
    for j in range(len(cols)):
        # 실제 보유분만 (롱온리 short_weight=0 의 0주 대기 숏은 제외)
        if direction[j] == 0 or base_shares[j] == 0:
            continue
        cur = C[last_t, j]
        ep = entry_px[j]
        d = int(direction[j])
        signed = d * base_shares[j] * mult[j]
        denom = (ep - init_stop[j]) if d == 1 else (init_stop[j] - ep)
        r_mult = ((cur - ep) / denom if d == 1 else (ep - cur) / denom) \
            if denom > 0 else np.nan
        # 스톱까지 거리(현재가 대비 %): 청산 트리거까지 여유
        stop_dist = ((cur - trail_stop[j]) / cur if d == 1
                     else (trail_stop[j] - cur) / cur) * 100
        ei = entry_idx[j]
        pos_rows.append({
            "티커": cols[j],
            "자산군": sec_of.get(cols[j], ""),
            "방향": "롱" if d == 1 else "숏",
            "유닛(m)": mult[j],
            "진입일": dates[ei].date() if ei >= 0 else None,
            "진입가": round(ep, 2),
            "현재가": round(cur, 2),
            "현재스톱": round(trail_stop[j], 2),
            "스톱여유%": round(stop_dist, 1),
            "미실현%": round(((cur / ep - 1) * 100) * d, 1),
            "R배수": round(r_mult, 2) if not np.isnan(r_mult) else None,
            "명목비중%": round(signed * cur / equity * 100, 1),
        })
    positions = pd.DataFrame(pos_rows)

    return dict(
        equity=pd.Series(eq_curve, index=dates),
        gross=pd.Series(gross_exp, index=dates),
        n_long=pd.Series(n_long, index=dates),
        n_short=pd.Series(n_short, index=dates),
        n_active=pd.Series(n_active, index=dates),
        trades=np.array(trade_returns),
        positions=positions,
        equity_final=equity,
        asof=dates[last_t],
        exits_today=exits_today,
    )


# ──────────────────────────────────────────────────────────────
# 지표 / 벤치마크
# ──────────────────────────────────────────────────────────────
def metrics(eq: pd.Series) -> dict:
    r = eq.pct_change().fillna(0)
    n = len(eq)
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (TRADING_DAYS / n) - 1.0
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    dd = eq / eq.cummax() - 1.0
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else np.nan
    return {"총수익률": total, "CAGR": cagr, "변동성(연)": vol,
            "Sharpe": sharpe, "MDD": mdd, "Calmar": calmar}


def fmt(v, pct=False):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v*100:.1f}%" if pct else f"{v:.2f}"


def benchmarks(close):
    spy = close["SPY"].dropna()
    spy_eq = (spy / spy.iloc[0])
    rets = close[TICKERS].pct_change()
    ew = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    return spy_eq, ew


# ──────────────────────────────────────────────────────────────
# 차트
# ──────────────────────────────────────────────────────────────
def plot_equity(curves, path):
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sty = {"Mulvaney 복제(raw)": ("#c4b5fd", 1.2),
           "Mulvaney 복제(변동성매칭)": ("#7c3aed", 2.0),
           "SPY Buy&Hold": ("#9ca3af", 1.4),
           "동일가중 바스켓 B&H": ("#16a34a", 1.4)}
    for name, eq in curves.items():
        c, lw = sty.get(name, ("#333", 1.3))
        ax.plot(eq.index, eq / eq.iloc[0], lw=lw, label=name, color=c)
    ax.set_yscale("log")
    ax.set_title("Mulvaney 복제(크로스애셋 ETF) vs 벤치마크 — 누적성과(로그)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("누적 배수 (시작=1.0)")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_drawdown(curves, path):
    fig, ax = plt.subplots(figsize=(13, 5))
    sty = {"Mulvaney 복제(raw)": "#c4b5fd",
           "Mulvaney 복제(변동성매칭)": "#7c3aed",
           "SPY Buy&Hold": "#9ca3af", "동일가중 바스켓 B&H": "#16a34a"}
    for name, eq in curves.items():
        dd = eq / eq.cummax() - 1.0
        ax.plot(dd.index, dd * 100, lw=1.3, label=name,
                color=sty.get(name, "#333"))
    ax.set_title("낙폭(Drawdown) 비교", fontsize=13, fontweight="bold")
    ax.set_ylabel("낙폭 (%)"); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_exposure(res, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    ax1.plot(res["gross"].index, res["gross"] * 100, lw=1.0, color="#7c3aed")
    ax1.set_title("총 노출(Gross Exposure)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("% of Equity"); ax1.grid(alpha=0.3)
    ax2.plot(res["n_long"].index, res["n_long"], lw=1.0, color="#16a34a",
             label="롱 시장수")
    ax2.plot(res["n_short"].index, -res["n_short"], lw=1.0, color="#dc2626",
             label="숏 시장수")
    ax2.plot(res["n_active"].index, res["n_active"], lw=0.8, ls="--",
             color="#9ca3af", label="활성 시장수")
    ax2.axhline(0, color="#555", lw=0.6)
    ax2.set_title("롱/숏 시장 수", fontsize=12, fontweight="bold")
    ax2.set_ylabel("시장 수"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ──────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────
def main():
    print("[1/5] 데이터 로드 (15 ETF + Fama-French RF)...")
    high, low, close, idx = load_universe()
    cash_rate = load_cash_rate(idx)
    print(f"      기간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)} 일)")
    print(f"      유니버스: {len(TICKERS)} 시장 "
          f"({', '.join(TICKERS)})")

    print("[2/5] 시그널 전처리...")
    sig = precompute_signals(high, low, close)

    print("[3/5] 포트폴리오 백테스트 (롱숏·트레일링·피라미딩·현금담보)...")
    res = backtest(high, low, close, sig, cash_rate)

    # 채널 형성 후 구간만 평가 (워밍업 제거)
    valid = res["n_active"] > 0
    eq = res["equity"][valid]
    eq = eq / eq.iloc[0]

    print("[4/5] 벤치마크 + 지표...")
    spy_eq, ew_eq = benchmarks(close)
    spy_eq = spy_eq[spy_eq.index >= eq.index[0]]
    spy_eq = spy_eq / spy_eq.iloc[0]
    ew_eq = ew_eq[ew_eq.index >= eq.index[0]]
    ew_eq = ew_eq / ew_eq.iloc[0]

    # 변동성 매칭 (원본 연구의 vol-scaling 단계): 전략 일별수익률을
    # SPY 실현변동성에 맞춰 상수 스케일 → 레버리지 제거한 like-for-like
    strat_r = eq.pct_change().fillna(0)
    spy_r = spy_eq.pct_change().reindex(eq.index).fillna(0)
    sv = strat_r.std()
    target_v = spy_r.std()
    scalar = target_v / sv if sv > 0 else 1.0
    eq_scaled = (1 + strat_r * scalar).cumprod()

    curves = {"Mulvaney 복제(raw)": eq,
              "Mulvaney 복제(변동성매칭)": eq_scaled,
              "SPY Buy&Hold": spy_eq,
              "동일가중 바스켓 B&H": ew_eq}
    metr = {k: metrics(v) for k, v in curves.items()}
    print(f"      변동성매칭 스케일: ×{scalar:.2f} "
          f"(raw {sv*np.sqrt(TRADING_DAYS)*100:.0f}% → "
          f"SPY {target_v*np.sqrt(TRADING_DAYS)*100:.0f}%)")

    print(f"\n      평가기간: {eq.index[0].date()} ~ {eq.index[-1].date()}")
    print("=" * 78)
    cols = ["총수익률", "CAGR", "변동성(연)", "Sharpe", "MDD", "Calmar"]
    pct = {"총수익률", "CAGR", "변동성(연)", "MDD"}
    print(f"{'전략':<20}" + "".join(f"{c:>10}" for c in cols))
    print("-" * 78)
    for name, m in metr.items():
        print(f"{name:<20}" +
              "".join(f"{fmt(m[c], c in pct):>10}" for c in cols))
    print("-" * 78)

    tr = res["trades"]
    if len(tr):
        wr = (tr > 0).mean()
        print(f"  거래수 {len(tr)} | 승률 {wr*100:.1f}% | "
              f"평균손익 {tr.mean()*100:.2f}% | "
              f"평균 총노출 {res['gross'][valid].mean()*100:.0f}% | "
              f"최대 활성시장 {int(res['n_active'].max())}")

    bh = metr["SPY Buy&Hold"]
    mv = metr["Mulvaney 복제(변동성매칭)"]
    # 2008 위기 알파: SPY 최악 구간 동안 전략 수익
    crisis = (eq.index >= "2008-09-01") & (eq.index <= "2009-03-31")
    if crisis.any():
        ce = eq_scaled[crisis]
        cs = spy_eq[crisis]
        print(f"\n  [2008 위기(09.01~익3월)] 전략 "
              f"{(ce.iloc[-1]/ce.iloc[0]-1)*100:+.1f}% vs SPY "
              f"{(cs.iloc[-1]/cs.iloc[0]-1)*100:+.1f}%  "
              f"(상관 {strat_r.corr(spy_r):+.2f})")
    print("\n  [SPY Buy&Hold 대비 판정 — 변동성매칭 기준]")
    print(f"   - Sharpe: {'이김' if mv['Sharpe']>bh['Sharpe'] else '짐'} "
          f"({fmt(mv['Sharpe'])} vs {fmt(bh['Sharpe'])})")
    print(f"   - MDD: {'우위(덜 깊음)' if mv['MDD']>bh['MDD'] else '열위'} "
          f"({fmt(mv['MDD'],True)} vs {fmt(bh['MDD'],True)})")

    print("\n[5/5] 차트/CSV 저장...")
    pd.DataFrame(metr).T.to_csv(BASE_DIR / "mulvaney_replica_metrics.csv",
                               encoding="utf-8-sig")
    plot_equity(curves, BASE_DIR / "mulvaney_replica_equity.png")
    plot_drawdown(curves, BASE_DIR / "mulvaney_replica_drawdown.png")
    plot_exposure({k: res[k][valid] for k in
                   ["gross", "n_long", "n_short", "n_active"]},
                  BASE_DIR / "mulvaney_replica_exposure.png")
    print(f"      저장: {BASE_DIR}")
    print("\n완료.")


if __name__ == "__main__":
    main()
