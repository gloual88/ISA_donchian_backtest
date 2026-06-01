# -*- coding: utf-8 -*-
"""
ISA 추세전략 대시보드 (Mulvaney 복제 · 롱온리)
================================================
한국 ISA 계좌 실거래용 — 매일 시그널(매수/손절/보유) + 손절선 확인.

전략: 돈치안 6개월 돌파 + 추적손절(미드라인) + loss-parity + 피라미딩,
      롱온리(개인 공매도 불가), 무레버리지(현금계좌). KRW 기준.
파라미터: ISA 그리드서치 강건값 N=252·p=0.4·lag=1·cap=2·K=1·short=0·LP.

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
    from isa_signals import get_isa_signals
    return get_isa_signals()


def risk_color(room):
    return NEG if room < 3 else "#f59e0b" if room < 7 else POS


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
    p = s["params"]

    st.markdown(
        f"<div style='background:#0b1220;border:1px solid #1d4ed8;"
        f"padding:8px 14px;border-radius:8px;font-size:0.85rem;color:#d1d5db'>"
        f"<b>기준일</b> {asof} (KRW)  |  설정 N={p['N']} · 손절폭 p={p['p']} · "
        f"롱온리 · 무레버리지  |  벤치마크: KOSPI200 매수보유</div>",
        unsafe_allow_html=True)
    st.write("")

    # ── 요약 지표 ──
    k = st.columns(6)
    k[0].metric("전략 Sharpe", f"{mst['Sharpe']:.2f}",
                f"{mst['Sharpe']-mks['Sharpe']:+.2f} vs KOSPI")
    k[1].metric("CAGR", f"{mst['CAGR']*100:.1f}%",
                f"{(mst['CAGR']-mks['CAGR'])*100:+.1f}%p")
    k[2].metric("MDD", f"{mst['MDD']*100:.1f}%",
                f"{(mst['MDD']-mks['MDD'])*100:+.1f}%p", delta_color="normal")
    k[3].metric("변동성", f"{mst['vol']*100:.1f}%")
    k[4].metric("보유 종목", f"{s['n_positions']}개")
    k[5].metric("현금", f"{max(s['cash_pct'],0):.0f}%")

    st.markdown("---")

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
        st.markdown("### 📈 누적성과 vs KOSPI200")
        eq, ks = s["equity"], s["kospi"]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=eq.index, y=eq.values, name="ISA 롱온리",
                                  line=dict(color="#7c3aed", width=2)))
        fig3.add_trace(go.Scatter(x=ks.index, y=ks.values,
                                  name="KOSPI200 매수보유",
                                  line=dict(color="#dc2626", width=1.5)))
        fig3.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", height=360,
            yaxis=dict(type="log", title="누적배수", gridcolor="#1a1f2e"),
            xaxis=dict(gridcolor="#1a1f2e"),
            legend=dict(x=0.02, y=0.98), margin=dict(t=10, b=10))
        st.plotly_chart(fig3, width="stretch")

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
