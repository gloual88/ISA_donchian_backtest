# -*- coding: utf-8 -*-
"""
ISA 추세전략 — 일일 리포트 이메일 (수익률 현황 + 오늘의 액션)
================================================================
data/signals.json을 읽어 매일 이메일 발송:
  · 수익률 현황: 기준일(ANCHOR_DATE, 기본 2026-03-31) 이후 누적 + 당일 (전략/KOSPI)
  · 오늘의 액션: 신규매수 / 손절청산 / 손절임박

환경변수(또는 GitHub Secrets):
  EMAIL_USER  발신 Gmail / EMAIL_PASS  Gmail 앱 비밀번호 / EMAIL_TO  수신
  DASHBOARD_URL  대시보드 공개주소(선택) / ANCHOR_DATE  수익률 기준일(선택)
  NEAR_STOP_PCT  손절임박 임계 %(기본 3.0)

실행: python alert_email.py
"""
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from return_status import load_curves, compute_status, ANCHOR

NEAR = float(os.getenv("NEAR_STOP_PCT", "3.0"))
DASH = os.getenv("DASHBOARD_URL", "")


def _rows(items, fmt):
    return "".join(f"<li>{fmt(i)}</li>" for i in items) or "<li>없음</li>"


def build_report(asof, s, eq, ks):
    st = compute_status(eq, ANCHOR)
    sk = compute_status(ks, ANCHOR)
    buys = s.get("buy_today", [])
    stops = s.get("stop_today", [])
    near = [n for n in s.get("near_stop", []) if n["stop_room_pct"] < NEAR]
    n_act = len(buys) + len(stops) + len(near)

    def cell(v, good_pos=True):
        c = "#16a34a" if (v >= 0) == good_pos else "#dc2626"
        return f"<b style='color:{c}'>{v*100:+.2f}%</b>"

    subj = (f"[ISA 추세전략] 일일 리포트 {asof} — "
            f"기준일이후 {st['cum_return']*100:+.1f}% / 액션 {n_act}건")

    html = f"""
    <div style="font-family:'Malgun Gothic',sans-serif;max-width:600px">
      <h2 style="color:#7c3aed">📈 ISA 추세전략 일일 리포트</h2>
      <p style="color:#555">기준일 <b>{st['anchor_date']}</b> →
         현재 <b>{st['last_date']}</b> ({st['trading_days']} 거래일)</p>

      <h3>💹 수익률 현황</h3>
      <table cellpadding="7" style="border-collapse:collapse;font-size:0.9rem">
        <tr style="background:#f3f0ff">
          <th></th><th>당일</th><th>기준일 이후 누적</th><th>기준후 MDD</th></tr>
        <tr><td><b>전략</b></td>
          <td align=right>{cell(st['daily_return'])}</td>
          <td align=right>{cell(st['cum_return'])}</td>
          <td align=right>{st['mdd_since']*100:.1f}%</td></tr>
        <tr><td>KOSPI200</td>
          <td align=right>{cell(sk['daily_return'])}</td>
          <td align=right>{cell(sk['cum_return'])}</td>
          <td align=right>{sk['mdd_since']*100:.1f}%</td></tr>
      </table>
      <p style="font-size:0.85rem;color:#555">기준일 이후 초과수익(전략−KOSPI):
        <b>{(st['cum_return']-sk['cum_return'])*100:+.1f}%p</b> ·
        보유 {s['n_positions']}종목 · 현금 {max(s['cash_pct'],0):.0f}%</p>

      <h3>🔔 오늘의 액션 ({n_act}건)</h3>
      <p style="margin:4px 0;color:#16a34a"><b>🟢 신규 매수</b></p>
      <ul>{_rows(buys, lambda b: f"{b['etf']} ({b['code']})")}</ul>
      <p style="margin:4px 0;color:#dc2626"><b>🔴 손절 청산</b></p>
      <ul>{_rows(stops, lambda e: f"{e['ticker']} ({e['ret_pct']:+.1f}%)")}</ul>
      <p style="margin:4px 0;color:#f59e0b"><b>🟠 손절 임박 (&lt;{NEAR:.0f}%)</b></p>
      <ul>{_rows(near, lambda n: f"{n['etf']} — 스톱여유 {n['stop_room_pct']:.1f}%")}</ul>

      {f'<p><a href="{DASH}">대시보드 열기 →</a></p>' if DASH else ''}
      <hr><p style="font-size:0.72rem;color:#999">
        본 메일은 투자 정보·교육 목적의 자동 리포트이며 매매 권유가 아닙니다.
        백테스트는 종가기준·프록시 가정이라 실제와 다를 수 있습니다.
        손익 책임은 투자자 본인에게 있습니다.</p>
    </div>"""
    return subj, html


def main():
    asof, eq, ks = load_curves()
    import json
    from pathlib import Path
    s = json.loads(
        (Path(__file__).resolve().parent / "data" / "signals.json")
        .read_text(encoding="utf-8"))

    subj, html = build_report(asof, s, eq, ks)

    user = os.getenv("EMAIL_USER")
    pw = os.getenv("EMAIL_PASS")
    to = os.getenv("EMAIL_TO") or user
    if not (user and pw):
        print("EMAIL_USER/EMAIL_PASS 미설정 — 발송 생략(드라이런)")
        print("제목:", subj)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subj
    msg["From"] = user
    msg["To"] = to
    msg.attach(MIMEText(html, "html", "utf-8"))
    ctx = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587) as srv:
        srv.starttls(context=ctx)
        srv.login(user, pw)
        srv.send_message(msg)
    print(f"발송 완료 → {to} | {subj}")


if __name__ == "__main__":
    main()
