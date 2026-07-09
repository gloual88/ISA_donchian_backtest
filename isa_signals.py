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
# 손절 후 재투자 + 과집중 방지 상한 (redeploy_grid 최적: 종목 20% 단독, 2026-07-09)
#   비운 비중을 남은 보유 종목에 재분배(현금 드래그 제거)하되 단일종목 20% 상한.
#   자산군 상한은 저분산 국면에서 강제 현금 드래그로 수익을 깎아 미적용(S_CAP=1).
#   백테스트: Sharpe 0.75→0.86 / MDD -25%→-18% / 누적 유사(11.3×→10.1×).
REDEPLOY = True
P_CAP = 0.20   # 종목별 비중 상한
S_CAP = 1.00   # 자산군 상한 미적용(1.0=off)

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


def _allocate(w_raw, p_cap, s_cap, sec_ids, n_sec):
    """원시 리스크패리티 비중 → 종목상한·자산군상한을 지키는 재투자 배분.
    비운 비중을 여유 있는 기존 보유 종목에 비례 재분배하되, 두 상한을 넘지 않는다.
    상한 때문에 100%를 채울 수 없으면 나머지는 현금으로 둔다(합<1 허용)."""
    w = np.clip(w_raw, 0, None).astype(float)
    if w.sum() <= 0:
        return w
    w = w / w.sum()                       # 목표 비율(합1)
    held = w > 1e-12
    for _ in range(300):
        w0 = w.copy()
        w = np.minimum(w, p_cap)          # 종목 상한
        if s_cap < 1.0:                   # 자산군 상한
            for k in range(n_sec):
                m = sec_ids == k
                ssum = w[m].sum()
                if ssum > s_cap + 1e-12:
                    w[m] *= s_cap / ssum
        deficit = 1.0 - w.sum()           # 부족분 재분배(여유·상한·기존보유 내)
        if deficit > 1e-9:
            pos_head = np.clip(p_cap - w, 0, None)
            if s_cap < 1.0:
                used = np.array([w[sec_ids == k].sum() for k in range(n_sec)])
                sec_head = np.clip(s_cap - used, 0, None)[sec_ids]
                room = np.minimum(pos_head, sec_head)
            else:
                room = pos_head
            room[~held] = 0.0             # 새 종목 생성 금지(신규매수는 신호로만)
            if room.sum() <= 1e-12:
                break                     # 여유 없음 → 나머지는 현금
            w = w + np.minimum(room, deficit * room / room.sum())
        elif np.abs(w - w0).max() < 1e-9:
            break
    return w


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
    res = M.backtest(high, low, close, sig, cash_rate, record_weights=True)
    valid = res["n_active"] > 0

    # ── 손절 후 재투자 + 상한 배분 (종목 P_CAP / 자산군 S_CAP) ──
    # 원시 리스크패리티 목표비중을, 비운 자리를 남은 종목에 재분배하고 과집중을
    # 상한으로 제어하는 정책비중으로 변환 → 전략 수익·지표·표시비중이 모두 일치.
    order = list(M.TICKERS)
    secs = sorted({ISA_DEF[lb][2] for lb in order})
    sid = {sn: i for i, sn in enumerate(secs)}
    sec_ids = np.array([sid[ISA_DEF[lb][2]] for lb in order])
    Wm = res["weights"][order].values
    Rm = close[order].pct_change().fillna(0.0).values
    crv = np.asarray(cash_rate.values if hasattr(cash_rate, "values")
                     else cash_rate, dtype=float)
    if REDEPLOY:
        Pm = np.vstack([_allocate(Wm[t], P_CAP, S_CAP, sec_ids, len(secs))
                        for t in range(len(Wm))])
    else:                                    # 재투자 미적용(현행): 상한만 축소
        g = Wm.sum(axis=1, keepdims=True)
        Pm = np.where(g > 1, Wm / np.where(g > 0, g, 1.0), Wm)
    Psum = Pm.sum(axis=1)
    port = np.zeros(len(Wm))
    port[1:] = ((Pm[:-1] * Rm[1:]).sum(axis=1)
                + np.clip(1 - Psum[:-1], 0, None) * crv[1:])
    eq = pd.Series(np.cumprod(1 + np.nan_to_num(port)), index=close.index)
    eq = eq[valid]
    eq = eq / eq.iloc[0]
    pol_w = {order[j]: round(float(Pm[-1, j] * 100), 1)
             for j in range(len(order))}
    pol_cash = round(float((1 - Psum[-1]) * 100), 1)
    kospi = ksC.reindex(eq.index).ffill()
    # 069500(KODEX200)은 2002 상장 → 전략 시작 이전은 NaN. 첫 '유효'값으로 정규화
    # (iloc[0]가 NaN이면 전체 NaN 되는 버그 방지). 벤치마크는 데이터 있는 구간만.
    kospi = kospi / kospi.dropna().iloc[0]
    asof = pd.Timestamp(res["asof"])

    # 전략 수익률은 실제 무레버리지 포트폴리오(정책 eq) 그대로 표시한다.
    # (구 방식은 KOSPI 변동성에 맞춘 변동성매칭 eq_sc를 썼는데, 저변동 전략을
    #  ×1.4 이상 부풀려 raw 벤치마크와 불공정 비교·오해를 유발 → 폐기, 2026-07-09)

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
    cash_pct = pol_cash

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
            isa_weight_pct=pol_w.get(tk, 0.0),
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
        metrics=dict(strategy=_metrics(eq), sixty_forty=_metrics(sf),
                     ew_basket=_metrics(ew), kospi=_metrics(kospi)),
        positions=positions,
        buy_today=buy_today,
        stop_today=stop_today,
        near_stop=near_stop,
        equity=eq,
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
