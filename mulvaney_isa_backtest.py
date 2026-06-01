# -*- coding: utf-8 -*-
"""
Mulvaney 복제 — 한국 ISA 백테스트 (KRW 기준, US ETF 프록시 + 환처리)
=======================================================================
ISA 투자 가능한 국내상장 ETF만으로 유니버스를 구성하되, 국내 ETF의
짧은 역사를 US ETF(장기 프록시)로 대체하고 **원화(KRW) 기준**으로 환산.

환(FX) 처리:
  - 환노출(unhedged): KRW수익 = US ETF(USD) × USD/KRW   [S&P500, 나스닥100]
  - 환헤지(H):        KRW수익 ≈ US ETF(USD)              [러셀2000H,신흥국H,금H,은H,원유H,농산물H]
  - KODEX 200:        실제 KOSPI200(KRW)
  - 미국달러선물:     USD/KRW 자체
국내 대응 ETF 없는 DBC(종합원자재)·DBB(산업금속)는 제외.

동일 Mulvaney 엔진(126일 돌파·트레일링스톱·loss-parity·피라미딩·현금담보)
사용. 벤치마크: KOSPI200 매수보유(KRW).

주의: 환헤지 비용·선물 롤·추적오차·괴리율은 미반영(프록시 한계).
      담보현금은 FF RF 사용(KRW 무위험과 근사). 실제와 차이 가능.

실행: PYTHONIOENCODING=utf-8 python mulvaney_isa_backtest.py
"""
import os
import shutil
import tempfile
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

try:
    import certifi
    _ca = os.path.join(tempfile.gettempdir(), "cacert_isa.pem")
    shutil.copy(certifi.where(), _ca)
    for _k in ("CURL_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        os.environ[_k] = _ca
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

import mulvaney_replica as M

BASE_DIR = Path(__file__).resolve().parent
TRADING_DAYS = 252
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ISA 유니버스: 라벨 → (US프록시, 환처리, 자산군)
#   환처리: 'unh'=환노출(×FX), 'hed'=환헤지(USD그대로), 'krw'=원화네이티브
ISA_DEF = {
    "S&P500(미국)":  ("SPY",   "unh", "주식"),
    "나스닥100":     ("QQQ",   "unh", "주식"),
    "러셀2000(H)":   ("IWM",   "hed", "주식"),
    "신흥국(H)":     ("EEM",   "hed", "주식"),
    "KOSPI200":      ("^KS200", "krw", "주식"),
    # 국내상장 미국채(환노출): 위기 때 채권↑+달러↑ 이중 방어
    "미국채30년":    ("TLT",   "unh", "금리"),
    "미국채10년":    ("IEF",   "unh", "금리"),
    "금(H)":         ("GLD",   "hed", "원자재"),
    "은(H)":         ("SLV",   "hed", "원자재"),
    "WTI원유(H)":    ("USO",   "hed", "원자재"),
    "농산물(H)":     ("DBA",   "hed", "원자재"),
    "미국달러선물":  ("KRW=X", "krw", "통화"),
}


def build_krw_panel():
    us = sorted({v[0] for v in ISA_DEF.values()
                 if v[0] not in ("^KS200", "KRW=X")})
    raw = yf.download(us, start="1999-01-01", auto_adjust=True,
                      progress=False)
    Hu, Lu, Cu = raw["High"], raw["Low"], raw["Close"]
    master = Cu.dropna(how="all").index

    fx = yf.download("KRW=X", start="1999-01-01", auto_adjust=True,
                     progress=False)
    ks = yf.download("^KS200", start="1999-01-01", auto_adjust=True,
                     progress=False)

    def reidx(df):
        return df.reindex(master).ffill()

    fxH, fxL, fxC = (reidx(fx["High"]), reidx(fx["Low"]), reidx(fx["Close"]))
    if isinstance(fxC, pd.DataFrame):
        fxH, fxL, fxC = fxH.iloc[:, 0], fxL.iloc[:, 0], fxC.iloc[:, 0]
    ksH, ksL, ksC = (reidx(ks["High"]), reidx(ks["Low"]), reidx(ks["Close"]))
    if isinstance(ksC, pd.DataFrame):
        ksH, ksL, ksC = ksH.iloc[:, 0], ksL.iloc[:, 0], ksC.iloc[:, 0]

    H, L, C = {}, {}, {}
    for label, (src, mode, _sec) in ISA_DEF.items():
        if mode == "krw" and src == "^KS200":
            H[label], L[label], C[label] = ksH, ksL, ksC
        elif mode == "krw" and src == "KRW=X":
            H[label], L[label], C[label] = fxH, fxL, fxC
        elif mode == "unh":
            H[label] = Hu[src] * fxC
            L[label] = Lu[src] * fxC
            C[label] = Cu[src] * fxC
        else:  # hed
            H[label], L[label], C[label] = Hu[src], Lu[src], Cu[src]

    high = pd.DataFrame(H).reindex(master)
    low = pd.DataFrame(L).reindex(master)
    close = pd.DataFrame(C).reindex(master)
    return high, low, close, ksC


def metrics(eq):
    r = eq.pct_change().fillna(0)
    n = len(eq)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (TRADING_DAYS / n) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sharpe = (r.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return {"총수익률": eq.iloc[-1] / eq.iloc[0] - 1, "CAGR": cagr,
            "변동성(연)": vol, "Sharpe": sharpe, "MDD": mdd,
            "Calmar": cagr / abs(mdd) if mdd else np.nan}


def fmt(v, pct=False):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v*100:.1f}%" if pct else f"{v:.2f}"


def main():
    print("[1/4] KRW 기준 패널 구성 (US 프록시 × 환처리)...")
    labels = list(ISA_DEF.keys())
    sectors = {}
    for lab, (_s, _m, sec) in ISA_DEF.items():
        sectors.setdefault(sec, []).append(lab)
    # 엔진 글로벌을 ISA 유니버스로 교체
    M.UNIVERSE = sectors
    M.TICKERS = labels

    high, low, close, ksC = build_krw_panel()
    cash_rate = M.load_cash_rate(close.index)
    print(f"      기간: {close.index[0].date()} ~ {close.index[-1].date()} "
          f"({len(close)}일), {len(labels)} 시장")
    print(f"      유니버스: {', '.join(labels)}")

    print("[2/4] 백테스트 (동일 Mulvaney 엔진)...")
    sig = M.precompute_signals(high, low, close)
    res = M.backtest(high, low, close, sig, cash_rate)

    valid = res["n_active"] > 0
    eq = res["equity"][valid]
    eq = eq / eq.iloc[0]

    # 벤치마크: KOSPI200 매수보유 (KRW)
    kospi = ksC.reindex(eq.index).ffill()
    kospi = (kospi / kospi.iloc[0])

    # 변동성매칭 (KOSPI200 변동성에)
    sr = eq.pct_change().fillna(0)
    kr = kospi.pct_change().fillna(0)
    scal = kr.std() / sr.std() if sr.std() > 0 else 1.0
    eq_sc = (1 + sr * scal).cumprod()

    print("[3/4] 성과")
    curves = {"ISA 복제(raw)": eq, "ISA 복제(변동성매칭)": eq_sc,
              "KOSPI200 매수보유": kospi}
    metr = {k: metrics(v) for k, v in curves.items()}
    print(f"\n      평가기간: {eq.index[0].date()} ~ {eq.index[-1].date()}")
    print(f"      변동성매칭 스케일 ×{scal:.2f} "
          f"(raw {sr.std()*np.sqrt(252)*100:.0f}% → "
          f"KOSPI200 {kr.std()*np.sqrt(252)*100:.0f}%)")
    print("=" * 76)
    cols = ["총수익률", "CAGR", "변동성(연)", "Sharpe", "MDD", "Calmar"]
    pct = {"총수익률", "CAGR", "변동성(연)", "MDD"}
    print(f"{'전략':<20}" + "".join(f"{c:>10}" for c in cols))
    print("-" * 76)
    for name, m in metr.items():
        print(f"{name:<20}" +
              "".join(f"{fmt(m[c], c in pct):>10}" for c in cols))
    print("-" * 76)
    tr = res["trades"]
    if len(tr):
        print(f"  거래 {len(tr)} | 승률 {(tr>0).mean()*100:.1f}% | "
              f"평균손익 {tr.mean()*100:.2f}% | "
              f"평균 총노출 {res['gross'][valid].mean()*100:.0f}%")
    bh, mv = metr["KOSPI200 매수보유"], metr["ISA 복제(변동성매칭)"]
    crisis = (eq.index >= "2008-09-01") & (eq.index <= "2009-03-31")
    if crisis.any():
        ce, cs = eq_sc[crisis], kospi[crisis]
        print(f"  [2008 위기] ISA복제 {(ce.iloc[-1]/ce.iloc[0]-1)*100:+.1f}% "
              f"vs KOSPI200 {(cs.iloc[-1]/cs.iloc[0]-1)*100:+.1f}%")
    print(f"  [KOSPI200 대비] Sharpe "
          f"{'이김' if mv['Sharpe']>bh['Sharpe'] else '짐'} "
          f"({fmt(mv['Sharpe'])} vs {fmt(bh['Sharpe'])}) | MDD "
          f"{'우위' if mv['MDD']>bh['MDD'] else '열위'} "
          f"({fmt(mv['MDD'],True)} vs {fmt(bh['MDD'],True)})")

    print("\n[4/4] 차트/CSV 저장...")
    pd.DataFrame(metr).T.to_csv(BASE_DIR / "mulvaney_isa_bt_metrics.csv",
                               encoding="utf-8-sig")
    _plot(curves, res, valid, BASE_DIR / "mulvaney_isa_bt.png")
    # 현재 포지션 (ISA 유니버스 기준)
    pos = res["positions"]
    if not pos.empty:
        print("\n  [현재 보유 (KRW·ISA 유니버스 기준)]")
        print(pos[["티커", "자산군", "방향", "유닛(m)", "진입일",
                   "미실현%", "R배수", "명목비중%"]].to_string(index=False))
        pos.to_csv(BASE_DIR / "mulvaney_isa_bt_positions.csv",
                   index=False, encoding="utf-8-sig")
    print(f"\n  저장: {BASE_DIR}")
    print("  ※ 프록시 한계: 환헤지비용·선물롤·추적오차·괴리율 미반영. 투자권유 아님.")


def _plot(curves, res, valid, path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                   gridspec_kw={"height_ratios": [3, 1]})
    sty = {"ISA 복제(raw)": ("#c4b5fd", 1.2),
           "ISA 복제(변동성매칭)": ("#7c3aed", 2.0),
           "KOSPI200 매수보유": ("#dc2626", 1.5)}
    for name, e in curves.items():
        c, lw = sty.get(name, ("#333", 1.2))
        ax1.plot(e.index, e / e.iloc[0], lw=lw, label=name, color=c)
    ax1.set_yscale("log")
    ax1.set_title("Mulvaney 복제(한국 ISA·KRW) vs KOSPI200 — 누적성과(로그)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("누적 배수 (시작=1.0)")
    ax1.legend(fontsize=10); ax1.grid(alpha=0.3)

    ax2.plot(res["n_long"][valid].index, res["n_long"][valid],
             color="#16a34a", lw=0.9, label="롱")
    ax2.plot(res["n_active"][valid].index, res["n_active"][valid],
             color="#9ca3af", lw=0.8, ls="--", label="활성")
    ax2.set_title("롱 시장 수 / 활성 시장 수", fontsize=11)
    ax2.set_ylabel("시장 수"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
