# -*- coding: utf-8 -*-
"""
ISA 롱온리 추세전략 — 실거래 시그널 코어 (대시보드/precompute 공용)
=====================================================================
matplotlib 의존 없이 순수 데이터만 반환. 대시보드 서비스/precompute에서 호출.

반환:
  asof, params, cash_pct, metrics, positions[], buy_today[], stop_today[],
  near_stop[], equity(Series), kospi(Series)

파라미터: RSI 강건설정 (N=189,p=0.4,lag=2,cap=2,K=1,short=0, **HLP**)
  유니버스: 크로스애셋 12종 + 한국 업종 14종 = 26종.
  HLP(섹터 계층) 사이징으로 주식 과집중 방지(주식~33/금리~50/원자재~17).
  (rsi_portfolio_optimizer가 OOS fold 일관성으로 선택)
"""
import numpy as np
import pandas as pd

import mulvaney_replica as M
from mulvaney_isa_backtest import ISA_DEF, build_krw_panel

TRADING_DAYS = 252
# RSI 강건설정 선택값. 직전: N=252/lag1/12종 → N=189/lag2/27종(업종 확장)/HLP
PARAMS = dict(N=189, p=0.4, lag=2, cap=2, K=1, short=0.0)
# 사이징: 0=LP / 1=HLP(섹터 계층). 업종 확장으로 주식 과집중 방지 위해 HLP 채택(2026-06-08)
SCHEME = 1

KRX_NAME = {
    "S&P500(미국)": ("KODEX 미국S&P500", "379800"),
    "나스닥100": ("TIGER 미국나스닥100", "133690"),
    "러셀2000(H)": ("KODEX 미국러셀2000(H)", "280930"),
    "신흥국(H)": ("PLUS 신흥국MSCI(합성 H)", "195980"),
    "KOSPI200": ("KODEX 200", "069500"),
    "미국채30년": ("PLUS 미국채30년액티브", "464470"),
    "미국채10년": ("TIGER 미국채10년선물", "305080"),
    "금(H)": ("KODEX 골드선물(H)", "132030"),
    "은(H)": ("KODEX 은선물(H)", "144600"),
    "WTI원유(H)": ("KODEX WTI원유선물(H)", "261220"),
    "농산물(H)": ("KODEX 3대농산물선물(H)", "271060"),
    "미국달러선물": ("KODEX 미국달러선물", "261240"),
    # ── 업종 ETF (2026-06-06) — KR 코드는 yf 티커 검증 완료, 미국반도체 코드는 매매 전 확인 ──
    "한국반도체": ("KODEX 반도체", "091160"),
    "한국IT": ("TIGER 200 IT", "139260"),
    "한국2차전지": ("TIGER 2차전지테마", "305540"),
    "한국자동차": ("KODEX 자동차", "091180"),
    "한국은행": ("KODEX 은행", "091170"),
    "한국증권": ("KODEX 증권", "102970"),
    "한국헬스케어": ("KODEX 헬스케어", "266420"),
    "한국바이오": ("KODEX 바이오", "244580"),
    "한국에너지화학": ("KODEX 에너지화학", "117460"),
    "한국철강": ("KODEX 철강", "117680"),
    "한국건설": ("KODEX 건설", "117700"),
    "한국운송": ("KODEX 운송", "140710"),
    "한국필수소비재": ("TIGER 200 생활소비재", "139280"),
    "한국경기소비재": ("TIGER 200 경기소비재", "139290"),
}


def _compute_nolev(pos):
    longs = pos[pos["방향"] == "롱"].copy()
    g = longs["명목비중%"].sum()
    if g <= 0:
        return pd.Series(dtype=float), 100.0
    factor = min(1.0, 100.0 / g)
    w = (longs["명목비중%"] * factor).round(1)
    w.index = longs["티커"].values
    return w, round(100.0 - w.sum(), 1)


def _metrics(eq):
    r = eq.pct_change().fillna(0)
    n = len(eq)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (TRADING_DAYS / n) - 1
    vol = r.std() * np.sqrt(TRADING_DAYS)
    sh = (r.mean() * TRADING_DAYS) / vol if vol > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return dict(CAGR=float(cagr), vol=float(vol), Sharpe=float(sh),
                MDD=float(mdd),
                Calmar=float(cagr / abs(mdd)) if mdd else np.nan)


def get_isa_signals():
    # 엔진 파라미터/유니버스 설정 (롱온리)
    M.N_LOOKBACK = PARAMS["N"]
    M.STOP_P = PARAMS["p"]
    M.EXEC_LAG = PARAMS["lag"]
    M.PYR_CAP = PARAMS["cap"]
    M.PYR_K = PARAMS["K"]
    M.SHORT_WEIGHT = PARAMS["short"]
    M.SCHEME = SCHEME
    labels = list(ISA_DEF.keys())
    sectors = {}
    for lab in labels:
        sectors.setdefault(ISA_DEF[lab][2], []).append(lab)
    M.UNIVERSE = sectors
    M.TICKERS = labels

    high, low, close, ksC = build_krw_panel()
    cash_rate = M.load_cash_rate(close.index)
    sig = M.precompute_signals(high, low, close)
    res = M.backtest(high, low, close, sig, cash_rate)

    valid = res["n_active"] > 0
    eq = res["equity"][valid]
    eq = eq / eq.iloc[0]
    kospi = ksC.reindex(eq.index).ffill()
    # 069500(KODEX200)은 2002 상장 → 전략 시작 이전은 NaN. 첫 '유효'값으로 정규화
    # (iloc[0]가 NaN이면 전체 NaN 되는 버그 방지). 벤치마크는 데이터 있는 구간만.
    kospi = kospi / kospi.dropna().iloc[0]
    asof = pd.Timestamp(res["asof"])

    sr, kr = eq.pct_change().fillna(0), kospi.pct_change().fillna(0)
    scal = kr.std() / sr.std() if sr.std() > 0 else 1.0
    eq_sc = (1 + sr * scal).cumprod()

    # ── 벤치마크: 동일가중 바스켓 + 60/40 (KOSPI는 참고) ──
    rets = close.pct_change()
    ew = (1 + rets.mean(axis=1).fillna(0)).cumprod()
    ew = ew.reindex(eq.index).ffill()
    ew = ew / ew.iloc[0]
    eq_cols = [c for c in labels if ISA_DEF[c][2] == "주식"]
    bd_cols = [c for c in labels if ISA_DEF[c][2] == "금리"]
    sf_ret = (0.6 * rets[eq_cols].mean(axis=1)
              + 0.4 * rets[bd_cols].mean(axis=1)).fillna(0)
    sf = (1 + sf_ret).cumprod().reindex(eq.index).ffill()
    sf = sf / sf.iloc[0]

    pos = res["positions"].copy()
    nl_w, cash_pct = _compute_nolev(pos)

    positions = []
    for _, r in pos.iterrows():
        tk = r["티커"]
        nm, code = KRX_NAME.get(tk, (tk, ""))
        positions.append(dict(
            label=tk, etf=nm, code=code, sector=r["자산군"],
            units=float(r["유닛(m)"]),
            entry_date=str(r["진입일"]) if r["진입일"] else None,
            stop_room_pct=float(r["스톱여유%"]),
            unreal_pct=float(r["미실현%"]),
            r_mult=(float(r["R배수"]) if pd.notna(r["R배수"]) else None),
            entry_vs_cur_pct=round(
                float(r["진입가"]) / float(r["현재가"]) * 100 - 100, 1),
            isa_weight_pct=float(nl_w.get(tk, 0.0)),
        ))
    positions.sort(key=lambda x: -x["isa_weight_pct"])

    asof_d = asof.date()
    buy_today = [p for p in positions if p["entry_date"] == str(asof_d)]
    near_stop = sorted([p for p in positions if p["stop_room_pct"] < 5],
                       key=lambda x: x["stop_room_pct"])
    stop_today = res.get("exits_today", [])

    return dict(
        asof=str(asof_d),
        params=PARAMS,
        cash_pct=cash_pct,
        n_positions=len(positions),
        metrics=dict(strategy=_metrics(eq_sc), sixty_forty=_metrics(sf),
                     ew_basket=_metrics(ew), kospi=_metrics(kospi)),
        positions=positions,
        buy_today=buy_today,
        stop_today=stop_today,
        near_stop=near_stop,
        equity=eq_sc,
        sixty_forty=sf,
        ew_basket=ew,
        kospi=kospi,
        prices=close,   # 종목별 최근 수익률 계산용(precompute 직렬화 화이트리스트에 없어 안전)
    )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    s = get_isa_signals()
    print("asof:", s["asof"], "| 보유", s["n_positions"], "| 현금",
          s["cash_pct"], "%")
    print("전략 Sharpe", round(s["metrics"]["strategy"]["Sharpe"], 2),
          "vs KOSPI", round(s["metrics"]["kospi"]["Sharpe"], 2))
    print("오늘 매수:", [p["etf"] for p in s["buy_today"]] or "없음")
    print("오늘 손절:", [e["ticker"] for e in s["stop_today"]] or "없음")
    print("손절 임박(<5%):",
          [(p["etf"], p["stop_room_pct"]) for p in s["near_stop"]] or "없음")
