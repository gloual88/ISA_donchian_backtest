# -*- coding: utf-8 -*-
"""
Mulvaney 복제 — 현재 시점 포트폴리오 스냅샷
==============================================
mulvaney_replica 엔진을 최신 데이터까지 돌려, 오늘 보유 중인
롱/숏 포지션·비중·진입가·현재 스톱·미실현손익을 출력.

- 명목비중은 raw(과레버리지) + 변동성매칭(실전 권장) 두 가지로 표기
- 스톱여유% = 청산 트리거까지 현재가 대비 여유 (작을수록 청산 임박)

실행: PYTHONIOENCODING=utf-8 python mulvaney_current_portfolio.py
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

BASE_DIR = Path(__file__).resolve().parent
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CLASS_COLORS = {"주식": "#3b82f6", "금리": "#f59e0b",
                "원자재": "#10b981", "통화": "#8b5cf6"}


def compute_nolev(pos):
    """
    무레버리지(현금 ETF 계좌용) 비중 산출.
    - 숏은 현금계좌서 불가 → 제외(현금으로 대체)
    - 롱 비중 합이 100% 초과면 비례 축소해 합 100%로 캡
    - 합이 100% 미만이면 잔여는 현금
    반환: (df[티커,자산군,무레버리지비중%], cash_pct, n_short_dropped)
    """
    longs = pos[pos["방향"] == "롱"].copy()
    n_short = int((pos["방향"] == "숏").sum())
    long_gross = longs["명목비중%"].sum()  # raw 명목 기준 (스케일 무관)
    if long_gross <= 0:
        return pd.DataFrame(columns=["티커", "자산군", "무레버리지비중%"]), \
            100.0, n_short
    factor = min(1.0, 100.0 / long_gross)
    longs["무레버리지비중%"] = (longs["명목비중%"] * factor).round(1)
    cash_pct = round(100.0 - longs["무레버리지비중%"].sum(), 1)
    return (longs[["티커", "자산군", "방향", "무레버리지비중%"]],
            cash_pct, n_short)


def plot_nolev(nl, cash_pct, asof, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5),
                                   gridspec_kw={"wspace": 0.25})
    d = nl.sort_values("무레버리지비중%", ascending=True)
    colors = [CLASS_COLORS.get(c, "#999") for c in d["자산군"]]
    ax1.barh(d["티커"], d["무레버리지비중%"], color=colors)
    for i, v in enumerate(d["무레버리지비중%"]):
        ax1.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10)
    ax1.set_title("무레버리지 비중 (현금계좌 실현 가능)",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("% of 자산")
    ax1.grid(alpha=0.3, axis="x")

    # 자산군 + 현금 도넛
    grp = nl.groupby("자산군")["무레버리지비중%"].sum()
    parts = list(grp.items())
    if cash_pct > 0.5:
        parts.append(("현금", cash_pct))
    cmap = dict(CLASS_COLORS); cmap["현금"] = "#9ca3af"
    ax2.pie([v for _, v in parts],
            labels=[f"{k}\n{v:.0f}%" for k, v in parts],
            colors=[cmap.get(k, "#999") for k, _ in parts],
            startangle=90, wedgeprops=dict(width=0.42),
            textprops=dict(fontsize=10))
    ax2.set_title("자산군 배분 (총 100%)", fontsize=12, fontweight="bold")
    fig.suptitle(f"Mulvaney 복제 — 무레버리지 포트폴리오  |  {asof}  |  "
                 f"총노출 100% · 현금 {cash_pct:.0f}%",
                 fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_dashboard(pos, scalar, asof, gross_raw, path):
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    p = pos.sort_values("매칭비중%", key=lambda s: s.abs(),
                        ascending=True).reset_index(drop=True)
    colors = [CLASS_COLORS.get(c, "#999") for c in p["자산군"]]
    labels = [f'{t} ({d})' for t, d in zip(p["티커"], p["방향"])]

    # (0,0) 포지션 비중 (매칭)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(labels, p["매칭비중%"], color=colors)
    for i, v in enumerate(p["매칭비중%"]):
        ax1.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    ax1.set_title("포지션별 비중 (변동성매칭 기준)", fontsize=12,
                  fontweight="bold")
    ax1.set_xlabel("명목비중 % of Equity")
    ax1.grid(alpha=0.3, axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in CLASS_COLORS.values()]
    ax1.legend(handles, CLASS_COLORS.keys(), fontsize=8,
               loc="lower right", title="자산군")

    # (0,1) 자산군별 순노출 도넛
    ax2 = fig.add_subplot(gs[0, 1])
    grp = pos.groupby("자산군")["매칭비중%"].sum()
    grp = grp[grp != 0]
    ax2.pie(grp.abs(), labels=[f"{k}\n{v:+.0f}%" for k, v in grp.items()],
            colors=[CLASS_COLORS.get(k, "#999") for k in grp.index],
            autopct="", startangle=90, wedgeprops=dict(width=0.42),
            textprops=dict(fontsize=10))
    ax2.set_title(f"자산군별 순노출  (총 {grp.sum():.0f}%)",
                  fontsize=12, fontweight="bold")

    # (1,0) 스톱 근접도 (위험)
    ax3 = fig.add_subplot(gs[1, 0])
    ps = pos.sort_values("스톱여유%", ascending=True).reset_index(drop=True)
    sc = ["#dc2626" if v < 3 else "#f59e0b" if v < 6 else "#16a34a"
          for v in ps["스톱여유%"]]
    ax3.barh(ps["티커"], ps["스톱여유%"], color=sc)
    for i, v in enumerate(ps["스톱여유%"]):
        ax3.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
    ax3.axvline(3, color="#dc2626", ls="--", lw=0.8)
    ax3.set_title("청산까지 여유 (현재가→스톱, 작을수록 위험)",
                  fontsize=12, fontweight="bold")
    ax3.set_xlabel("스톱여유 %")
    ax3.grid(alpha=0.3, axis="x")

    # (1,1) 미실현 손익 % (R배수 주석)
    ax4 = fig.add_subplot(gs[1, 1])
    pu = pos.sort_values("미실현%", ascending=True).reset_index(drop=True)
    uc = ["#16a34a" if v >= 0 else "#dc2626" for v in pu["미실현%"]]
    ax4.barh(pu["티커"], pu["미실현%"], color=uc)
    for i, (v, r) in enumerate(zip(pu["미실현%"], pu["R배수"])):
        off = 0.5 if v >= 0 else -0.5
        ha = "left" if v >= 0 else "right"
        rr = f"{v:+.1f}% (R{r:.1f})" if pd.notna(r) else f"{v:+.1f}%"
        ax4.text(v + off, i, rr, va="center", ha=ha, fontsize=9)
    ax4.axvline(0, color="#555", lw=0.8)
    ax4.set_title("포지션별 미실현 손익 (R = 초기리스크 배수)",
                  fontsize=12, fontweight="bold")
    ax4.set_xlabel("미실현 %")
    ax4.grid(alpha=0.3, axis="x")

    fig.suptitle(
        f"Mulvaney 복제 현재 포트폴리오  |  기준일 {asof}  |  "
        f"롱 {len(pos[pos.방향=='롱'])} · 숏 {len(pos[pos.방향=='숏'])}  |  "
        f"총노출 raw {gross_raw:.0f}% → 매칭 {gross_raw*scalar:.0f}%",
        fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    print("[1/3] 데이터 로드 + 엔진 실행 (최신 시점까지)...")
    high, low, close, idx = M.load_universe()
    cash_rate = M.load_cash_rate(idx)
    sig = M.precompute_signals(high, low, close)
    res = M.backtest(high, low, close, sig, cash_rate)

    asof = pd.Timestamp(res["asof"]).date()
    pos = res["positions"].copy()

    # 변동성매칭 스케일 (전략 raw vol → SPY vol)
    eq = res["equity"]
    valid = res["n_active"] > 0
    eq = eq[valid]
    strat_r = eq.pct_change().fillna(0)
    spy = close["SPY"].reindex(eq.index)
    spy_r = spy.pct_change().fillna(0)
    scalar = (spy_r.std() / strat_r.std()) if strat_r.std() > 0 else 1.0

    print(f"\n{'='*70}")
    print(f"  Mulvaney 복제 현재 포트폴리오  (기준일: {asof})")
    print(f"  파라미터: N={M.N_LOOKBACK} p={M.STOP_P} lag={M.EXEC_LAG} "
          f"cap={M.PYR_CAP} K={M.PYR_K} short={M.SHORT_WEIGHT}")
    print(f"{'='*70}")

    if pos.empty:
        print("\n  현재 보유 포지션 없음 (전 시장 관망).")
        return

    # 변동성매칭 비중 추가
    pos["매칭비중%"] = (pos["명목비중%"] * scalar).round(1)
    pos = pos.sort_values("명목비중%", key=lambda s: s.abs(),
                          ascending=False).reset_index(drop=True)

    longs = pos[pos["방향"] == "롱"]
    shorts = pos[pos["방향"] == "숏"]

    gross_raw = pos["명목비중%"].abs().sum()
    net_raw = pos["명목비중%"].sum()

    print(f"\n  보유 {len(pos)}개  (롱 {len(longs)} · 숏 {len(shorts)})  |  "
          f"활성 {int(res['n_active'].iloc[-1])} 시장")
    print(f"  총노출(raw) {gross_raw:.0f}%  순노출 {net_raw:+.0f}%  |  "
          f"변동성매칭 스케일 ×{scalar:.2f} → 총노출 {gross_raw*scalar:.0f}%")

    cols = ["티커", "자산군", "방향", "유닛(m)", "진입일", "진입가",
            "현재가", "현재스톱", "스톱여유%", "미실현%", "R배수",
            "명목비중%", "매칭비중%"]
    print("\n  [롱 포지션]")
    print(longs[cols].to_string(index=False) if not longs.empty
          else "   없음")
    print("\n  [숏 포지션]")
    print(shorts[cols].to_string(index=False) if not shorts.empty
          else "   없음")

    # 자산군별 순노출
    print("\n  [자산군별 순노출 (매칭 기준)]")
    grp = (pos.assign(s=pos["매칭비중%"])
           .groupby("자산군")["s"].sum().sort_values())
    for k, v in grp.items():
        print(f"   {k:5}: {v:+.1f}%")

    # ── 무레버리지 버전 (현금 ETF 계좌 실현 가능) ──
    nl, cash_pct, n_short_dropped = compute_nolev(pos)
    print(f"\n{'='*70}")
    print(f"  무레버리지 버전 (총노출 100% 캡)  —  현금 {cash_pct:.0f}%")
    if n_short_dropped:
        print(f"  ※ 숏 {n_short_dropped}개는 현금계좌 불가 → 현금 대체")
    print(f"{'='*70}")
    nl_sorted = nl.sort_values("무레버리지비중%", ascending=False)
    print(nl_sorted.to_string(index=False))
    print(f"  {'현금':6}{'':18}{cash_pct:>6.1f}")
    gv = nl.groupby("자산군")["무레버리지비중%"].sum().sort_values(
        ascending=False)
    print("\n  [자산군 배분]")
    for k, v in gv.items():
        print(f"   {k:5}: {v:.1f}%")
    if cash_pct > 0.5:
        print(f"   현금 : {cash_pct:.1f}%")

    out = BASE_DIR / "mulvaney_current_portfolio.csv"
    pos[cols].to_csv(out, index=False, encoding="utf-8-sig")
    nl_out = BASE_DIR / "mulvaney_current_portfolio_nolev.csv"
    nl_save = nl_sorted.copy()
    nl_save.loc[len(nl_save)] = ["현금", "현금", "-", cash_pct]
    nl_save.to_csv(nl_out, index=False, encoding="utf-8-sig")
    png = BASE_DIR / "mulvaney_current_portfolio.png"
    plot_dashboard(pos, scalar, asof, gross_raw, png)
    png2 = BASE_DIR / "mulvaney_current_portfolio_nolev.png"
    plot_nolev(nl, cash_pct, asof, png2)
    print(f"\n  저장: {out}")
    print(f"        {nl_out}")
    print(f"  차트: {png}")
    print(f"        {png2}")
    print("\n  주의: 백테스트 종가 기준 시그널. 실거래는 익일 체결·비용·"
          "유동성 반영 필요. 투자 권유 아님.")


if __name__ == "__main__":
    main()
