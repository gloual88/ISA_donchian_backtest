# -*- coding: utf-8 -*-
"""
ISA 추세전략 — 오늘의 액션 이메일 알림
========================================
data/signals.json을 읽어 신규매수 / 손절청산 / 손절임박이 있으면 이메일 발송.
액션이 없으면 발송 생략(스팸 방지).

환경변수(또는 GitHub Secrets):
  EMAIL_USER  발신 Gmail 주소
  EMAIL_PASS  Gmail 앱 비밀번호 (2단계인증 후 발급)
  EMAIL_TO    수신 주소 (없으면 EMAIL_USER로)
  DASHBOARD_URL  대시보드 공개주소 (선택)
  NEAR_STOP_PCT  손절임박 임계값 % (기본 3.0)

실행: python alert_email.py
"""
import os
import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

BASE = Path(__file__).resolve().parent
SIGNALS = BASE / "data" / "signals.json"
NEAR = float(os.getenv("NEAR_STOP_PCT", "3.0"))
DASH = os.getenv("DASHBOARD_URL", "")


def build_alert(s):
    """액션이 있으면 (제목, HTML) 반환, 없으면 None."""
    buys = s.get("buy_today", [])
    stops = s.get("stop_today", [])
    near = [n for n in s.get("near_stop", []) if n["stop_room_pct"] < NEAR]
    if not (buys or stops or near):
        return None

    n_total = len(buys) + len(stops) + len(near)
    subj = f"[ISA 추세전략] 오늘의 액션 {n_total}건 — {s['asof']}"

    def rows(items, fmt):
        return "".join(f"<li>{fmt(i)}</li>" for i in items) or "<li>없음</li>"

    html = f"""
    <div style="font-family:'Malgun Gothic',sans-serif;max-width:560px">
      <h2 style="color:#7c3aed">📈 ISA 추세전략 — 오늘의 액션</h2>
      <p style="color:#555">기준일 <b>{s['asof']}</b> · 보유 {s['n_positions']}종목 ·
         현금 {max(s['cash_pct'],0):.0f}%</p>
      <h3 style="color:#16a34a">🟢 신규 매수</h3>
      <ul>{rows(buys, lambda b: f"{b['etf']} ({b['code']})")}</ul>
      <h3 style="color:#dc2626">🔴 손절 청산</h3>
      <ul>{rows(stops, lambda e: f"{e['ticker']} ({e['ret_pct']:+.1f}%)")}</ul>
      <h3 style="color:#f59e0b">🟠 손절 임박 (&lt;{NEAR:.0f}%)</h3>
      <ul>{rows(near, lambda n: f"{n['etf']} — 스톱여유 {n['stop_room_pct']:.1f}%")}</ul>
      {f'<p><a href="{DASH}">대시보드 열기 →</a></p>' if DASH else ''}
      <hr>
      <p style="font-size:0.75rem;color:#999">
        본 메일은 투자 정보·교육 목적의 자동 알림이며 매매 권유가 아닙니다.
        손익 책임은 투자자 본인에게 있습니다.</p>
    </div>"""
    return subj, html


def main():
    if not SIGNALS.exists():
        print("signals.json 없음 — 알림 생략")
        return
    s = json.loads(SIGNALS.read_text(encoding="utf-8"))

    alert = build_alert(s)
    if alert is None:
        print(f"액션 없음({s['asof']}) — 이메일 생략")
        return
    subj, html = alert

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
