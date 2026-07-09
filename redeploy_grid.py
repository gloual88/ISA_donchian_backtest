# -*- coding: utf-8 -*-
"""
재투자 상한 튜닝 그리드 (ISA 무레버리지)
==========================================
redeploy_backtest의 B정책을 확장 — 종목별 상한(p_cap)과 자산군 상한(s_cap)을
격자로 비교해 최적 상한을 찾는다. 엔진은 1회만 실행(가중치 로깅) 후 정책만 반복.

정책 배분 규칙(alloc): 원시 리스크패리티 목표비중 → 합1 정규화 → 종목상한·자산군
상한 water-filling(초과분을 여유 있는 종목/자산군에 비례 재분배) 반복 수렴.

출력: mulvaney_isa_redeploy_grid.csv, mulvaney_isa_redeploy_grid.png
실행: PYTHONIOENCODING=utf-8 python redeploy_grid.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

import mulvaney_replica as M
from mulvaney_isa_backtest import ISA_DEF, build_krw_panel
from isa_signals import PARAMS, SCHEME, _metrics, _allocate

for fp in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"]:
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def configure_engine():
    M.N_LOOKBACK = PARAMS["N"]; M.STOP_P = PARAMS["p"]
    M.EXEC_LAG = PARAMS["lag"]; M.PYR_CAP = PARAMS["cap"]
    M.PYR_K = PARAMS["K"]; M.SHORT_WEIGHT = PARAMS["short"]; M.SCHEME = SCHEME
    labels = list(ISA_DEF.keys())
    sectors = {}
    for lab in labels:
        sectors.setdefault(ISA_DEF[lab][2], []).append(lab)
    M.UNIVERSE = sectors
    M.TICKERS = labels
    return labels


alloc = _allocate   # 라이브(isa_signals)와 동일한 교정 배분 로직 사용


def main():
    labels = configure_engine()
    sec_names = sorted(set(ISA_DEF[lb][2] for lb in labels))
    sec_idx = {s: i for i, s in enumerate(sec_names)}
    sec_ids = np.array([sec_idx[ISA_DEF[lb][2]] for lb in labels])
    n_sec = len(sec_names)

    print("· 데이터 로드 + 엔진(가중치 로깅) 1회...")
    high, low, close, ksC = build_krw_panel()
    cr = M.load_cash_rate(close.index)
    sig = M.precompute_signals(high, low, close)
    res = M.backtest(high, low, close, sig, cr, record_weights=True)

    W = res["weights"][labels].values
    Wsum = W.sum(axis=1)
    R = close[labels].pct_change().fillna(0.0).values
    crs = pd.Series(cr, index=close.index).values
    valid = (res["n_active"] > 0).values
    idx = close.index

    def equity(pcap, scap, current=False):
        P = np.empty_like(W)
        for t in range(len(W)):
            w = W[t]
            if current:                       # C 현행: gross>1이면 축소, else 현금
                g = Wsum[t]
                P[t] = w / g if g > 1 else w
            else:
                P[t] = alloc(w, pcap, scap, sec_ids, n_sec)
        Ps = P.sum(axis=1)
        port = np.zeros(len(W))
        port[1:] = (P[:-1] * R[1:]).sum(axis=1) \
            + np.clip(1 - Ps[:-1], 0, None) * crs[1:]
        eq = pd.Series(np.cumprod(1 + np.nan_to_num(port)), index=idx)
        eq = eq[valid]
        return eq / eq.iloc[0]

    # ── 그리드 ──
    configs = [("C 현행", None, None, True), ("A 전액(상한없음)", 1.00, 1.00, False)]
    for pc in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        configs.append((f"p{int(pc*100)}", pc, 1.00, False))
    for pc, sc in [(0.25, 0.40), (0.25, 0.50), (0.20, 0.40), (0.30, 0.50)]:
        configs.append((f"p{int(pc*100)}+자산군{int(sc*100)}", pc, sc, False))

    last1y = idx[valid][-252:]
    rows = []
    curves = {}
    for name, pc, sc, cur in configs:
        eq = equity(pc, sc, current=cur)
        curves[name] = eq
        m = _metrics(eq)
        e1 = eq.loc[last1y]; e1 = e1 / e1.iloc[0]
        rows.append(dict(
            정책=name, CAGR=round(m["CAGR"]*100, 1),
            Sharpe=round(m["Sharpe"], 3), MDD=round(m["MDD"]*100, 1),
            Calmar=round(m["Calmar"], 3),
            누적배수=round(eq.iloc[-1], 2),
            최근1y=round((e1.iloc[-1]-1)*100, 1),
            최근1y_MDD=round(_metrics(e1)["MDD"]*100, 1)))
    df = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    print("\n" + "=" * 96)
    print(f"재투자 상한 그리드 — {idx[valid][0].date()} ~ {idx[valid][-1].date()} "
          f"({valid.sum()} 거래일) · Sharpe 내림차순")
    print("=" * 96)
    print(df.to_string(index=False))
    df.to_csv("mulvaney_isa_redeploy_grid.csv", index=False,
              encoding="utf-8-sig")

    # ── 차트: Sharpe / MDD vs 종목상한(자산군상한 없는 순수 p-cap 계열) ──
    pdf = df[df["정책"].str.match(r"^p\d+$")].copy()
    pdf["pcap"] = pdf["정책"].str.extract(r"(\d+)").astype(int)
    pdf = pdf.sort_values("pcap")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(pdf["pcap"], pdf["Sharpe"], "o-", color="#1a8f5a", lw=2)
    ax[0].axhline(df[df["정책"] == "C 현행"]["Sharpe"].iloc[0], ls="--",
                  color="#8894a3", label="C 현행")
    ax[0].axhline(df[df["정책"] == "A 전액(상한없음)"]["Sharpe"].iloc[0], ls="--",
                  color="#2a78d6", label="A 전액")
    ax[0].set_xlabel("종목별 상한 %"); ax[0].set_ylabel("Sharpe")
    ax[0].set_title("Sharpe vs 종목 상한", fontweight="bold")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(pdf["pcap"], pdf["MDD"], "o-", color="#d64545", lw=2)
    ax[1].axhline(df[df["정책"] == "C 현행"]["MDD"].iloc[0], ls="--",
                  color="#8894a3", label="C 현행")
    ax[1].set_xlabel("종목별 상한 %"); ax[1].set_ylabel("MDD %")
    ax[1].set_title("MDD vs 종목 상한 (0에 가까울수록 우수)", fontweight="bold")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("mulvaney_isa_redeploy_grid.png", dpi=130)
    print("\n저장: mulvaney_isa_redeploy_grid.csv, mulvaney_isa_redeploy_grid.png")
    best = df.iloc[0]
    print(f"\n▶ Sharpe 최고: {best['정책']}  "
          f"(Sharpe {best['Sharpe']}, MDD {best['MDD']}%, Calmar {best['Calmar']})")


if __name__ == "__main__":
    main()
