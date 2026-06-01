# -*- coding: utf-8 -*-
"""
Mulvaney 복제 — 한국 ISA 계좌 실행 매핑 (옵션 A)
====================================================
US 원본 엔진으로 로직을 검증(mulvaney_replica)한 뒤, 현재 무레버리지
포트폴리오를 **국내상장 ETF로 변환한 ISA 실행 매매표**를 생성.

원칙:
  - 해외상장 ETF 직접보유 불가 → 국내상장 동일물로 치환
  - 숏 불가 → 현금 (현 시점 숏 없음)
  - 레버리지 불가 → 무레버리지(총노출 100% 캡) 버전 기준
  - **국내 대응 ETF가 없으면 반영하지 않고 현금 처리** (사용자 지침)

KRX 코드는 KRX ETF 일별 시세에서 확인한 실제 코드 (2026-05-29 기준).
실행: PYTHONIOENCODING=utf-8 python mulvaney_isa_mapping.py
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mulvaney_replica as M
from mulvaney_current_portfolio import compute_nolev, CLASS_COLORS

BASE_DIR = Path(__file__).resolve().parent
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# US ETF → 국내상장 ETF 매핑 (실제 KRX 코드, 2026-05-29 확인)
# None = 국내 대응 ETF 없음 → 현금 처리
KRX_MAP = {
    "SPY": ("379800", "KODEX 미국S&P500", "고유동성"),
    "QQQ": ("133690", "TIGER 미국나스닥100", "고유동성"),
    "IWM": ("280930", "KODEX 미국러셀2000(H)", "저유동성·환헤지"),
    "EWY": ("069500", "KODEX 200", "한국 직접·초고유동성"),
    "EEM": ("195980", "PLUS 신흥국MSCI(합성 H)", "저유동성·합성"),
    "USO": ("261220", "KODEX WTI원유선물(H)", "환헤지·롤비용"),
    "UUP": ("261240", "KODEX 미국달러선물", "보통"),
    "DBA": ("271060", "KODEX 3대농산물선물(H)", "저유동성·환헤지"),
    # ── 국내 대응 없음 → 제외(현금) ──
    "DBB": (None, "산업금속 바스켓 대응 ETF 없음(구리만 존재)", "제외→현금"),
    "DBC": (None, "종합원자재 대응 ETF 없음", "제외→현금"),
    # (참고: 현 시점 미보유) GLD→132030, SLV→144600, USO만 보유
}


def main():
    print("[1/3] US 원본 엔진 실행 → 현재 무레버리지 포트폴리오...")
    high, low, close, idx = M.load_universe()
    cash_rate = M.load_cash_rate(idx)
    sig = M.precompute_signals(high, low, close)
    res = M.backtest(high, low, close, sig, cash_rate)
    asof = pd.Timestamp(res["asof"]).date()
    pos = res["positions"]
    nl, base_cash, n_short = compute_nolev(pos)

    print(f"\n{'='*72}")
    print(f"  ISA 실행 매핑  (기준일 {asof})  —  US 원본 → 국내상장 ETF")
    print(f"{'='*72}")

    rows = []
    dropped_w = 0.0
    for _, r in nl.iterrows():
        us = r["티커"]
        w = r["무레버리지비중%"]
        krx = KRX_MAP.get(us)
        if krx is None or krx[0] is None:
            name = krx[1] if krx else "대응 ETF 없음"
            rows.append({"US": us, "자산군": r["자산군"], "KRX코드": "-",
                         "국내ETF": name, "ISA비중%": 0.0,
                         "원비중%": w, "상태": "제외→현금"})
            dropped_w += w
        else:
            code, name, note = krx
            rows.append({"US": us, "자산군": r["자산군"], "KRX코드": code,
                         "국내ETF": name, "ISA비중%": w,
                         "원비중%": w, "상태": note})

    tbl = pd.DataFrame(rows)
    total_invested = tbl["ISA비중%"].sum()
    cash_total = round(100.0 - total_invested, 1)

    show = tbl[tbl["상태"] != "제외→현금"].sort_values(
        "ISA비중%", ascending=False)
    drop = tbl[tbl["상태"] == "제외→현금"]

    print("\n  [실행 — 국내상장 ETF]")
    print(show[["US", "KRX코드", "국내ETF", "자산군", "ISA비중%", "상태"]]
          .to_string(index=False))
    print(f"\n  투자 합계 {total_invested:.1f}%  +  현금 {cash_total:.1f}%  = 100%")

    if not drop.empty:
        print("\n  [제외 — 국내 대응 ETF 없음 → 현금]")
        for _, r in drop.iterrows():
            print(f"   {r['US']:4} (원비중 {r['원비중%']:.1f}%): {r['국내ETF']}")

    print("\n  [자산군 배분 (ISA 실행 기준)]")
    gv = show.groupby("자산군")["ISA비중%"].sum().sort_values(
        ascending=False)
    for k, v in gv.items():
        print(f"   {k:5}: {v:.1f}%")
    print(f"   현금 : {cash_total:.1f}%")

    out = BASE_DIR / "mulvaney_isa_portfolio.csv"
    tbl.to_csv(out, index=False, encoding="utf-8-sig")
    png = BASE_DIR / "mulvaney_isa_portfolio.png"
    _plot(show, gv, cash_total, asof, dropped_w, png)
    print(f"\n  저장: {out}\n  차트: {png}")
    print("\n  주의: US 원본으로 검증한 신호를 국내상장 ETF로 치환한 표.")
    print("  국내 ETF는 추적오차·환헤지·롤비용·괴리율이 있어 실제 성과는 다를 수 있음.")
    print("  ISA = 잦은 회전의 세금 마찰 제거(비과세 200/400만 + 9.9% 분리과세).")
    print("  투자 권유 아님.")


def _plot(show, gv, cash_total, asof, dropped_w, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7),
                                   gridspec_kw={"wspace": 0.35})
    d = show.sort_values("ISA비중%", ascending=True)
    colors = [CLASS_COLORS.get(c, "#999") for c in d["자산군"]]
    labels = [f'{n}\n({u})' for n, u in zip(d["국내ETF"], d["US"])]
    ax1.barh(labels, d["ISA비중%"], color=colors)
    for i, v in enumerate(d["ISA비중%"]):
        ax1.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
    ax1.set_title("ISA 실행 비중 (국내상장 ETF)", fontsize=12,
                  fontweight="bold")
    ax1.set_xlabel("% of 자산")
    ax1.grid(alpha=0.3, axis="x")

    parts = list(gv.items())
    if cash_total > 0.5:
        parts.append(("현금", cash_total))
    cmap = dict(CLASS_COLORS); cmap["현금"] = "#9ca3af"
    ax2.pie([v for _, v in parts],
            labels=[f"{k}\n{v:.0f}%" for k, v in parts],
            colors=[cmap.get(k, "#999") for k, _ in parts],
            startangle=90, wedgeprops=dict(width=0.42),
            textprops=dict(fontsize=10))
    ax2.set_title(f"자산군 배분 (현금 {cash_total:.0f}% — 원자재 제외분 포함)",
                  fontsize=12, fontweight="bold")
    fig.suptitle(
        f"Mulvaney 복제 — 한국 ISA 실행 포트폴리오  |  {asof}  |  "
        f"무레버리지·롱온리  |  제외(현금)분 {dropped_w:.0f}%p",
        fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
