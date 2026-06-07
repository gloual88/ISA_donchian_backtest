# -*- coding: utf-8 -*-
"""
ISA 추세전략 대시보드 (Mulvaney 복제 · 롱온리)
================================================
한국 ISA 계좌 실거래용 — 매일 시그널(매수/손절/보유) + 손절선 확인.

전략: 돈치안 6개월 돌파 + 추적손절(미드라인) + loss-parity + 피라미딩,
      롱온리(개인 공매도 불가), 무레버리지(현금계좌). KRW 기준.
파라미터: N=189·p=0.4·lag=2·cap=2·K=1·short=0·HLP, 27종(크로스애셋12+업종15).

실행: streamlit run app.py   (또는 run.bat)
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="헤지펀드 추세전략 대시보드", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")

CLASS_COLORS = {"주식": "#3b82f6", "금리": "#f59e0b",
                "원자재": "#10b981", "통화": "#8b5cf6"}
POS, NEG, DIM = "#22c55e", "#ef4444", "#9ca3af"


@st.cache_data(ttl=21600, show_spinner=False)
def load_signals():
    """사전계산 캐시(data/signals.json) 우선 로드 → 없으면 라이브 계산."""
    import json
    from pathlib import Path
    cache = Path(__file__).resolve().parent / "data" / "signals.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            for key in ("equity", "sixty_forty", "ew_basket", "kospi"):
                if key in d and isinstance(d[key], dict):
                    d[key] = pd.Series(
                        d[key]["values"],
                        index=pd.to_datetime(d[key]["dates"]))
            d["_cached"] = True
            return d
        except Exception:
            pass
    from isa_signals import get_isa_signals
    s = get_isa_signals()
    s["_cached"] = False
    return s


def risk_color(room):
    return NEG if room < 3 else "#f59e0b" if room < 7 else POS


def render_return_card(col, name, cum, daily, name_color):
    """수익률 카드 — 누적/당일 각각이 자기 부호에 맞는 화살표·색을 갖는다.
    (st.metric은 화살표·색이 delta=당일만 따라가, 누적이 음수인데 녹색 ▲가
     뜨는 오해를 유발해 커스텀 렌더로 교체.)"""
    c_col = POS if cum >= 0 else NEG
    c_arw = "▲" if cum >= 0 else "▼"
    d_col = POS if daily >= 0 else NEG
    d_arw = "▲" if daily >= 0 else "▼"
    col.markdown(
        f"<div style='background:#0b1220;border:1px solid #1a1f2e;"
        f"border-radius:8px;padding:10px 12px'>"
        f"<div style='font-size:0.82rem;color:{name_color};"
        f"font-weight:600'>{name}</div>"
        f"<div style='font-size:1.5rem;font-weight:700;color:{c_col};"
        f"line-height:1.6'>{c_arw} {cum*100:+.1f}%</div>"
        f"<div style='font-size:0.78rem;color:{d_col}'>"
        f"당일 {d_arw} {daily*100:+.2f}%</div>"
        f"</div>", unsafe_allow_html=True)


def render_transition():
    """전략 전환 리밸런싱(1회성) — data/transition.json 있을 때만 표시.
    '오늘의 액션'(신전략 자생 신호)과 별개로, 구→신 전환에 필요한 비중 조정."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent / "data" / "transition.json"
    if not p.exists():
        return
    t = json.loads(p.read_text(encoding="utf-8"))
    fp, tp = t["from_params"], t["to_params"]
    st.markdown("### 🔄 전략 전환 리밸런싱 (1회성)")
    st.info(f"전략 변경 {fp.get('scheme', '')} → **{tp.get('scheme', '')}** "
            f"(N={tp['N']}/lag={tp['lag']}, 기준일 {t['asof']}). 구→신 "
            "포트폴리오로 갈아타기 위해 오늘 조정할 비중입니다. "
            "※ 아래 '오늘의 액션'(신전략 자생 신호)과는 별개의 일회성 매매.")
    df = pd.DataFrame([{
        "ETF": r["etf"], "자산군": r["sector"], "구 비중%": r["old_w"],
        "신 비중%": r["new_w"], "조정%p": r["delta"], "액션": r["action"]}
        for r in t["rows"]])
    st.dataframe(
        df.style.format({"구 비중%": "{:.1f}", "신 비중%": "{:.1f}",
                         "조정%p": "{:+.1f}"})
        .background_gradient(subset=["조정%p"], cmap="RdYlGn",
                             vmin=-6, vmax=6),
        width="stretch", hide_index=True)
    st.caption("반영 완료 후 data/transition.json 삭제 시 이 섹션은 사라집니다.")
    st.markdown("---")


def main():
    st.markdown("## 📈 헤지펀드 추세전략 대시보드")
    st.caption("Mulvaney 복제 · 롱온리 · 무레버리지 · KRW — "
               "한국 ISA 실거래용 매일 시그널")

    c1, c2 = st.columns([6, 1])
    with c2:
        if st.button("🔄 새로고침", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("시그널 계산 중... (최초 15~20초, 이후 캐시)"):
        s = load_signals()

    asof = s["asof"]
    mst, mks = s["metrics"]["strategy"], s["metrics"]["kospi"]
    mbf = s["metrics"].get("sixty_forty", mks)
    p = s["params"]

    st.markdown(
        f"<div style='background:#0b1220;border:1px solid #1d4ed8;"
        f"padding:8px 14px;border-radius:8px;font-size:0.85rem;color:#d1d5db'>"
        f"<b>기준일</b> {asof} (KRW)  |  설정 N={p['N']} · 손절폭 p={p['p']} · "
        f"롱온리 · 무레버리지  |  벤치마크: 60/40·동일가중"
        f"<br><span style='font-size:0.75rem;color:#9ca3af'>※ 국내자산은 "
        f"당일 종가, 해외자산은 직전 미국 종가 반영(미국장 시차)</span></div>",
        unsafe_allow_html=True)
    gen = s.get("_generated_kst")
    if gen and s.get("_cached"):
        st.caption(f"⚡ 매일 자동 갱신 (사전계산) — 마지막 갱신: {gen}")
    st.write("")

    # ── 요약 지표 ──
    k = st.columns(6)
    k[0].metric("전략 Sharpe", f"{mst['Sharpe']:.2f}",
                f"{mst['Sharpe']-mbf['Sharpe']:+.2f} vs 60/40")
    k[1].metric("CAGR", f"{mst['CAGR']*100:.1f}%",
                f"{(mst['CAGR']-mbf['CAGR'])*100:+.1f}%p")
    k[2].metric("MDD", f"{mst['MDD']*100:.1f}%",
                f"{(mst['MDD']-mks['MDD'])*100:+.1f}%p", delta_color="normal")
    k[3].metric("변동성", f"{mst['vol']*100:.1f}%")
    k[4].metric("보유 종목", f"{s['n_positions']}개")
    k[5].metric("현금", f"{max(s['cash_pct'],0):.0f}%")

    st.markdown("---")

    # ── 전략 전환 리밸런싱 (1회성, transition.json 있을 때만) ──
    render_transition()

    # ── 오늘의 액션 ──
    st.markdown("### 🔔 오늘의 액션")
    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("**🟢 신규 매수**")
        if s["buy_today"]:
            for b in s["buy_today"]:
                st.success(f"{b['etf']} ({b['code']})")
        else:
            st.caption("없음 — 신규 진입 신호 없음")
    with a2:
        st.markdown("**🔴 손절 청산**")
        if s["stop_today"]:
            for e in s["stop_today"]:
                st.error(f"{e['ticker']}  ({e['ret_pct']:+.1f}%)")
        else:
            st.caption("없음 — 오늘 청산 종목 없음")
    with a3:
        st.markdown("**🟠 손절 임박 (<5%)**")
        if s["near_stop"]:
            for n in s["near_stop"]:
                st.warning(f"{n['etf']} — 스톱여유 {n['stop_room_pct']:.1f}%")
        else:
            st.caption("없음")

    st.markdown("---")

    # ── 보유 + 손절선 ──
    left, right = st.columns([1.1, 1])
    pos = s["positions"]
    df = pd.DataFrame([{
        "ETF": p_["etf"], "코드": p_["code"], "자산군": p_["sector"],
        "유닛": f"{p_['units']:.0f}×",
        "진입일": p_["entry_date"],
        "스톱여유%": p_["stop_room_pct"],
        "미실현%": p_["unreal_pct"],
        "ISA비중%": p_["isa_weight_pct"],
    } for p_ in pos])

    with left:
        st.markdown("### 📋 현재 보유 + 손절선")
        st.dataframe(
            df.style.format({"스톱여유%": "{:.1f}", "미실현%": "{:+.1f}",
                             "ISA비중%": "{:.1f}"})
            .background_gradient(subset=["스톱여유%"], cmap="RdYlGn",
                                 vmin=0, vmax=20)
            .background_gradient(subset=["미실현%"], cmap="RdYlGn",
                                 vmin=-20, vmax=60),
            width="stretch", height=400, hide_index=True)
        st.caption("⚠️ 실제 손절가 = 실제 ETF 현재가 × (1 − 스톱여유%). "
                   "표 가격은 프록시 합성수준이라 생략.")

    with right:
        st.markdown("### 🎯 손절매 기준 (현재가 대비)")
        ps = sorted(pos, key=lambda x: x["stop_room_pct"], reverse=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[p_["label"] for p_ in ps],
            x=[-p_["stop_room_pct"] for p_ in ps],
            orientation="h",
            marker_color=[risk_color(p_["stop_room_pct"]) for p_ in ps],
            text=[f"손절 {p_['stop_room_pct']:.1f}%↓" for p_ in ps],
            textposition="outside", textfont=dict(color="#e0e0e0", size=11),
            hovertemplate="%{y}<br>손절선까지 %{text}<extra></extra>"))
        fig.add_trace(go.Scatter(
            y=[p_["label"] for p_ in ps],
            x=[p_["entry_vs_cur_pct"] for p_ in ps],
            mode="markers", marker=dict(color="#e0e0e0", size=7,
                                        symbol="circle"),
            name="진입가", hovertemplate="진입 %{x:.1f}%<extra></extra>"))
        fig.add_vline(x=0, line_color="#e0e0e0", line_width=1.5)
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=400, showlegend=False,
            margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(title="현재가 대비 % (0=현재가, 음수=손절선)",
                       gridcolor="#1a1f2e"),
            yaxis=dict(gridcolor="#1a1f2e", autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.caption("🔴<3% 임박 · 🟠<7% · 🟢 여유 / 흰점 = 진입 시점")

    st.markdown("---")

    # ── 비중 + 누적성과 ──
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("### ⚖️ ISA 실행 비중 (자산군)")
        grp = {}
        for p_ in pos:
            grp[p_["sector"]] = grp.get(p_["sector"], 0) + p_["isa_weight_pct"]
        cash = max(s["cash_pct"], 0)
        labels = list(grp.keys()) + (["현금"] if cash > 0.5 else [])
        vals = list(grp.values()) + ([cash] if cash > 0.5 else [])
        colors = [CLASS_COLORS.get(x, "#9ca3af") for x in grp] + \
                 (["#6b7280"] if cash > 0.5 else [])
        fig2 = go.Figure(go.Pie(
            labels=labels, values=vals, hole=0.45, marker_colors=colors,
            textinfo="label+percent",
            textfont=dict(color="#fff", size=12)))
        fig2.update_layout(template="plotly_dark",
                           paper_bgcolor="rgba(0,0,0,0)", height=360,
                           showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, width="stretch")

    with g2:
        st.markdown("### 📈 누적성과 vs 벤치마크")
        eq, ks = s["equity"], s["kospi"]
        sf = s.get("sixty_forty")
        ew = s.get("ew_basket")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=eq.index, y=eq.values, name="ISA 롱온리",
                                  line=dict(color="#7c3aed", width=2.2)))
        if sf is not None:
            fig3.add_trace(go.Scatter(x=sf.index, y=sf.values, name="60/40",
                                      line=dict(color="#f59e0b", width=1.4)))
        if ew is not None:
            fig3.add_trace(go.Scatter(x=ew.index, y=ew.values,
                                      name="동일가중 바스켓",
                                      line=dict(color="#16a34a", width=1.4)))
        fig3.add_trace(go.Scatter(x=ks.index, y=ks.values,
                                  name="KOSPI200 (참고)",
                                  line=dict(color="#6b7280", width=1,
                                            dash="dot")))
        fig3.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=360,
            yaxis=dict(type="log", title="누적배수", gridcolor="#1a1f2e"),
            xaxis=dict(gridcolor="#1a1f2e"),
            legend=dict(x=0.02, y=0.98), margin=dict(t=10, b=10))
        st.plotly_chart(fig3, width="stretch")

    # ── 💹 수익률 현황 (기준일 이후) — 맨 하단 ──
    st.markdown("---")
    from return_status import compute_status, ANCHOR
    rc = {"전략": s.get("equity"), "60/40": s.get("sixty_forty"),
          "동일가중바스켓": s.get("ew_basket"), "KOSPI200(참고)": s.get("kospi")}
    rc = {k: v for k, v in rc.items() if v is not None}
    if "전략" in rc:
        rstats = {k: compute_status(v, ANCHOR) for k, v in rc.items()}
        st0 = rstats["전략"]
        st.markdown(f"### 💹 수익률 현황 — 기준일 {st0['anchor_date']} 이후 "
                    f"({st0['trading_days']}거래일)")
        cmap = {"전략": "#7c3aed", "60/40": "#f59e0b",
                "동일가중바스켓": "#16a34a", "KOSPI200(참고)": "#6b7280"}
        cc = st.columns(len(rstats))
        for col, (name, m) in zip(cc, rstats.items()):
            render_return_card(col, name, m["cum_return"], m["daily_return"],
                               cmap.get(name, "#e5e7eb"))
        fig4 = go.Figure()
        for name, v in rc.items():
            seg = v[v.index >= pd.Timestamp(rstats[name]["anchor_date"])]
            seg = (seg / seg.iloc[0] - 1) * 100
            fig4.add_trace(go.Scatter(
                x=seg.index, y=seg.values, name=name,
                line=dict(color=cmap.get(name, "#999"),
                          width=2.4 if name == "전략" else 1.4,
                          dash="dot" if "참고" in name else "solid")))
        fig4.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=340,
            yaxis=dict(title="기준일 이후 누적수익률 %", gridcolor="#1a1f2e"),
            xaxis=dict(gridcolor="#1a1f2e"),
            legend=dict(x=0.02, y=0.98), margin=dict(t=10, b=10))
        st.plotly_chart(fig4, width="stretch")
        st.caption(f"기준일 {ANCHOR} 이후 누적·당일 수익률. "
                   "전략 vs 60/40·동일가중바스켓(공정비교) · KOSPI200은 참고.")

    # ── 면책 ──
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.7rem;color:#6b7280;line-height:1.6'>"
        "⚠️ 본 자료는 투자 참고용 정보 제공 목적이며 특정 금융상품의 "
        "매수·매도를 권유하지 않습니다. 백테스트는 종가기준·무슬리피지·"
        "US ETF 프록시(환처리) 가정이며 환헤지비용·선물롤·추적오차·괴리율은 "
        "미반영입니다. 실제 성과는 다를 수 있고, 과거 수익률이 미래를 "
        "보장하지 않습니다. 모든 투자 판단과 손익 책임은 투자자 본인에게 "
        "있습니다.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
