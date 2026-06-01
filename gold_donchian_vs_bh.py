# -*- coding: utf-8 -*-
"""
금(Gold) — 돈치안 추세전략 vs 단순보유(B&H) 누적수익 비교
============================================================
금 단일 자산에 추세 농법(돈치안 돌파 + 추적손절)을 적용했을 때 vs
그냥 들고만 있었을 때(Buy&Hold) 누적수익률 비교 시각화.

방식(롱온리·단일자산):
  진입: 종가 >= N일 신고가
  손절: 초기 고정스톱(신고가 − p×채널폭) → 수익 전환 후 미드라인 추적
  보유 중 = 금 100%, 청산 = 현금(0%)
파라미터: 대시보드 전략과 동일 (N=252, p=0.4)

실행: PYTHONIOENCODING=utf-8 python gold_donchian_vs_bh.py
"""
import os
import shutil
import tempfile
from pathlib import Path

try:
    import certifi
    _ca = os.path.join(tempfile.gettempdir(), "cacert_gold.pem")
    shutil.copy(certifi.where(), _ca)
    for _k in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        os.environ[_k] = _ca
except Exception:
    pass

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

BASE = Path(__file__).resolve().parent
TICKER = "GLD"          # 국제 금 가격(USD) 대표 ETF
N, P = 252, 0.4
TD = 252


def load_gold():
    df = yf.download(TICKER, start="2004-01-01", auto_adjust=True,
                     progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["High", "Low", "Close"]].dropna()


def donchian_long_only(df):
    """단일자산 롱온리: 보유=금100%, 청산=현금0%."""
    upper = df["High"].rolling(N).max().shift(1)
    dlow = df["Low"].rolling(N).min().shift(1)
    width = upper - dlow
    midline = (upper + dlow) / 2
    c = df["Close"].values
    up, lo, wd, mid = (upper.values, dlow.values, width.values,
                       midline.values)
    pos = np.zeros(len(df))
    holding, stop, entry = 0, np.nan, np.nan
    for i in range(len(df)):
        if holding == 0:
            if not np.isnan(up[i]) and c[i] >= up[i] and wd[i] > 0:
                holding = 1
                entry = c[i]
                stop = up[i] - P * wd[i]
        else:
            if c[i] > entry and not np.isnan(mid[i]):
                stop = max(stop, mid[i])
            if c[i] <= stop:
                holding = 0
        pos[i] = holding
    return pd.Series(pos, index=df.index)


def metrics(eq):
    r = eq.pct_change().fillna(0)
    n = len(eq)
    tot = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (TD / n) - 1
    vol = r.std() * np.sqrt(TD)
    sh = (r.mean() * TD) / vol if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return tot, cagr, vol, sh, mdd


def main():
    print("[1/3] 금(GLD) 데이터 로드...")
    df = load_gold()
    ret = df["Close"].pct_change()
    print(f"      {df.index[0].date()} ~ {df.index[-1].date()} "
          f"({len(df)}일)")

    print("[2/3] 돈치안 추세전략 vs B&H...")
    pos = donchian_long_only(df)
    strat_ret = pos.shift(1) * ret           # 보유중=금수익, 청산=0(현금)
    strat_eq = (1 + strat_ret.fillna(0)).cumprod()
    bh_eq = (1 + ret.fillna(0)).cumprod()
    # 채널 형성 후로 정렬
    valid = pos.index[df["High"].rolling(N).max().shift(1).notna()]
    strat_eq = strat_eq.loc[valid] / strat_eq.loc[valid].iloc[0]
    bh_eq = bh_eq.loc[valid] / bh_eq.loc[valid].iloc[0]

    ms = metrics(strat_eq)
    mb = metrics(bh_eq)
    exposure = pos.loc[valid].mean()
    print("\n      === 누적수익 비교 (금) ===")
    print(f"{'':16}{'총수익':>10}{'CAGR':>9}{'변동성':>9}"
          f"{'Sharpe':>9}{'MDD':>9}")
    print(f"{'추세전략':16}{ms[0]*100:>9.1f}%{ms[1]*100:>8.1f}%"
          f"{ms[2]*100:>8.1f}%{ms[3]:>9.2f}{ms[4]*100:>8.1f}%")
    print(f"{'단순보유(B&H)':16}{mb[0]*100:>9.1f}%{mb[1]*100:>8.1f}%"
          f"{mb[2]*100:>8.1f}%{mb[3]:>9.2f}{mb[4]*100:>8.1f}%")
    print(f"      시장노출(금 보유비율): {exposure*100:.0f}%")

    print("\n[3/3] 차트 저장...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5),
                                   gridspec_kw={"height_ratios": [3, 1.2]},
                                   sharex=True)
    ax1.plot(strat_eq.index, strat_eq, color="#D4AF37", lw=2.2,
             label=f"금 + 추세 농법 (돈치안 {N}일)")
    ax1.plot(bh_eq.index, bh_eq, color="#9ca3af", lw=1.6,
             label="금 단순보유 (Buy & Hold)")
    ax1.set_yscale("log")
    ax1.set_title("금(Gold) — 추세 농법 vs 단순보유 누적수익 비교",
                  fontsize=15, fontweight="bold")
    ax1.set_ylabel("누적 배수 (로그, 시작=1.0)")
    ax1.legend(fontsize=12, loc="upper left")
    ax1.grid(alpha=0.3)
    txt = (f"[추세전략]  총 {ms[0]*100:.0f}% · CAGR {ms[1]*100:.1f}% · "
           f"Sharpe {ms[3]:.2f} · MDD {ms[4]*100:.0f}%\n"
           f"[단순보유]  총 {mb[0]*100:.0f}% · CAGR {mb[1]*100:.1f}% · "
           f"Sharpe {mb[3]:.2f} · MDD {mb[4]*100:.0f}%")
    ax1.text(0.015, 0.97, txt, transform=ax1.transAxes, va="top",
             fontsize=10.5, color="#333",
             bbox=dict(boxstyle="round", fc="#fffbe6", ec="#D4AF37"))

    for name, eq, col in [("추세전략", strat_eq, "#D4AF37"),
                          ("단순보유", bh_eq, "#9ca3af")]:
        dd = (eq / eq.cummax() - 1) * 100
        ax2.plot(dd.index, dd, color=col, lw=1.4, label=name)
    ax2.set_title("낙폭(Drawdown) — 추세 농법이 하락장을 피하는가",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("낙폭 (%)")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = BASE / "gold_donchian_vs_bh.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"      저장: {out}")
    print("  ※ GLD=국제 금가격(USD). 청산 시 현금(0%) 가정. "
          "종가기준·무슬리피지. 투자권유 아님.")


if __name__ == "__main__":
    main()
