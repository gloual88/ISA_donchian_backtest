# -*- coding: utf-8 -*-
"""
ISA 추세전략 — 실제 KRX ETF vs US 프록시 추적오차 검증 (E)
================================================================
백테스트는 US ETF 프록시(×환율)로 KRW 수익을 근사한다. 실제 국내상장 ETF는
환헤지비용·선물롤·괴리율 때문에 다르게 움직인다. 최근 겹치는 기간에서
실제 KRX ETF 일별종가를 받아 프록시와 **추적오차/상관/누적괴리**를 측정.

KRX 데이터: pykrx_openapi (.env의 KRX_API_KEY). 캐시: krx_cache/*.json
실행: PYTHONIOENCODING=utf-8 python validate_krx_tracking.py
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from mulvaney_isa_backtest import build_krw_panel

BASE = Path(__file__).resolve().parent
CACHE = BASE / "krx_cache"
CACHE.mkdir(exist_ok=True)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
TD = 252

# KRX 코드 → 프록시 라벨(ISA_DEF) / 한글명 / 환헤지여부
MAP = {
    "069500": ("KOSPI200", "KODEX 200", "원화"),
    "379800": ("S&P500(미국)", "KODEX 미국S&P500", "환노출"),
    "133690": ("나스닥100", "TIGER 미국나스닥100", "환노출"),
    "280930": ("러셀2000(H)", "KODEX 미국러셀2000(H)", "환헤지"),
    "195980": ("신흥국(H)", "PLUS 신흥국MSCI(H)", "환헤지"),
    "261220": ("WTI원유(H)", "KODEX WTI원유선물(H)", "환헤지"),
    "261240": ("미국달러선물", "KODEX 미국달러선물", "원화"),
    "271060": ("농산물(H)", "KODEX 3대농산물선물(H)", "환헤지"),
    "132030": ("금(H)", "KODEX 골드선물(H)", "환헤지"),
    "144600": ("은(H)", "KODEX 은선물(H)", "환헤지"),
}


def _client():
    load_dotenv(Path("d:/파이선/.env"))
    key = os.getenv("KRX_API_KEY")
    if not key:
        raise RuntimeError("KRX_API_KEY 없음 (.env 확인)")
    from pykrx_openapi import KRXOpenAPI
    return KRXOpenAPI(api_key=key)


def fetch_krx_etf_closes(start, end):
    """기간 내 영업일마다 ETF일별 받아 대상 코드 종가 시리즈 구성(캐시)."""
    cli = _client()
    codes = set(MAP.keys())
    rows = {c: {} for c in codes}
    d = start
    n_fetched = 0
    while d <= end:
        if d.weekday() < 5:
            dd = d.strftime("%Y%m%d")
            cf = CACHE / f"etf_{dd}.json"
            if cf.exists():
                recs = json.loads(cf.read_text(encoding="utf-8"))
            else:
                try:
                    res = cli.get_etf_daily_trade(bas_dd=dd)
                    recs = res.get("OutBlock_1", [])
                    n_fetched += 1
                except Exception:
                    recs = []
                # 캐시 쓰기 실패가 데이터를 버리지 않도록 분리(datetime→str)
                if recs:
                    try:
                        cf.write_text(
                            json.dumps(recs, ensure_ascii=False, default=str),
                            encoding="utf-8")
                    except Exception:
                        pass
            for r in recs:
                c = r.get("ISU_CD")
                if c in codes:
                    px = pd.to_numeric(r.get("TDD_CLSPRC"), errors="coerce")
                    if pd.notna(px):
                        rows[c][pd.Timestamp(d)] = float(px)
        d += dt.timedelta(days=1)
    print(f"      KRX API 신규호출 {n_fetched}일 (나머지 캐시)")
    return {c: pd.Series(v).sort_index() for c, v in rows.items()}


def main():
    # 최근 약 1.5년 겹침 구간
    end = dt.date(2026, 5, 29)
    start = dt.date(2024, 12, 2)
    print(f"[1/3] 실제 KRX ETF 종가 수집 ({start} ~ {end})...")
    krx = fetch_krx_etf_closes(start, end)

    print("[2/3] US 프록시 패널 + 정렬...")
    high, low, close, ksC = build_krw_panel()

    print("\n[추적오차 — 실제 KRX ETF vs US 프록시]")
    print(f"{'국내ETF':<22}{'환':<6}{'상관':>7}{'추적오차(연)':>12}"
          f"{'누적괴리':>10}")
    results = []
    for code, (label, krname, hedge) in MAP.items():
        kr = krx.get(code)
        if kr is None or len(kr) < 30:
            continue
        px = close[label].reindex(kr.index).ffill()
        a = pd.DataFrame({"kr": kr, "px": px}).dropna()
        if len(a) < 30:
            continue
        rk = a["kr"].pct_change().dropna()
        rp = a["px"].pct_change().dropna()
        j = pd.DataFrame({"kr": rk, "px": rp}).dropna()
        corr = j["kr"].corr(j["px"])
        te = (j["kr"] - j["px"]).std() * np.sqrt(TD)
        gap = (a["kr"].iloc[-1] / a["kr"].iloc[0]) - \
              (a["px"].iloc[-1] / a["px"].iloc[0])
        results.append((krname, hedge, corr, te, gap, a))
        print(f"{krname:<22}{hedge:<6}{corr:>7.2f}{te*100:>11.1f}%"
              f"{gap*100:>9.1f}%")

    if not results:
        print("\n비교 가능한 KRX ETF 데이터가 없습니다 — 차트 생략.")
        return
    print("\n[3/3] 차트 (대표 3종)...")
    pick = [r for r in results if r[0] in
            ("KODEX 미국S&P500", "KODEX 200", "KODEX 골드선물(H)")]
    if not pick:
        pick = results[:3]
    fig, axes = plt.subplots(1, len(pick), figsize=(5*len(pick), 4.5))
    if len(pick) == 1:
        axes = [axes]
    for ax, (krname, hedge, corr, te, gap, a) in zip(axes, pick):
        ke = a["kr"] / a["kr"].iloc[0]
        pe = a["px"] / a["px"].iloc[0]
        ax.plot(ke.index, ke, color="#dc2626", lw=1.6, label="실제 KRX ETF")
        ax.plot(pe.index, pe, color="#7c3aed", lw=1.4, ls="--",
                label="US 프록시(×환율)")
        ax.set_title(f"{krname}\n상관 {corr:.2f} · 추적오차 {te*100:.1f}% · "
                     f"괴리 {gap*100:+.1f}%", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("실제 KRX ETF vs US 프록시 — 누적수익 비교 (정규화)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(BASE / "validate_krx_tracking.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)
    print(f"      저장: {BASE/'validate_krx_tracking.png'}")


if __name__ == "__main__":
    main()
