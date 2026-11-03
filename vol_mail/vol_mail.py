# -*- coding: utf-8 -*-
"""KOSPI 변동성·규칙 비중 일일 메일 (ISA 시그널과 독립).

kospi_close.csv에 Yahoo ^KS11 새 종가를 덧붙이고, 새 종가일이면 1통 보낸다.
같은 종가일 재실행과 휴장일은 last_sent.txt로 건너뛴다.

환경변수(GitHub Secrets):
  EMAIL_USER  발신 Gmail / EMAIL_PASS  앱 비밀번호 / VOL_MAIL_TO  수신
  EMAIL_TO는 ISA 구독자 목록이므로 대신 쓰지 않는다.
  FORCE_EMAIL=1  중복 방지 무시 / SUBJECT_PREFIX  제목 접두

실행: python vol_mail/vol_mail.py [--dry-run]
"""
import os
import smtplib
import ssl
import sys
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
HIST = ROOT / "kospi_close.csv"
STATE = ROOT / "last_sent.txt"
ANN = 246            # 2001년 이후 연평균 거래일
KST = timezone(timedelta(hours=9))
CLOSED_AT = (15, 40)  # 이 시각 전이면 당일 값은 장중으로 보고 제외


def update_history():
    """저장된 종가는 고치지 않고 새 날짜만 덧붙인다."""
    c = pd.read_csv(HIST, index_col=0, parse_dates=True)["Close"]
    y = yf.download("^KS11", start=str(c.index[-1].date()), progress=False, auto_adjust=False)["Close"]
    y = y.squeeze("columns").dropna()
    y.index = pd.to_datetime(y.index).tz_localize(None).normalize()
    now = datetime.now(KST)
    if (now.hour, now.minute) < CLOSED_AT:
        y = y[y.index < pd.Timestamp(now.date())]
    if c.index[-1] in y.index and abs(y[c.index[-1]] / c.iloc[-1] - 1) > 0.01:
        raise RuntimeError(f"직전 종가 불일치: 저장 {c.iloc[-1]:.2f}, Yahoo {y[c.index[-1]]:.2f}")
    new = y[y.index > c.index[-1]]
    if new.empty:
        return c
    out = pd.concat([c, new]).round(2)
    out.name = "Close"
    if out.pct_change().iloc[-len(new):].abs().max() > 0.30:
        raise RuntimeError("새 종가의 일간 변동이 ±30%를 넘어 중단")
    out.to_frame().to_csv(HIST, index_label="Date")
    return out


def summarize(c):
    r = c.pct_change()
    sr, sn = r.rolling(21).std(), r.rolling(1260, min_periods=252).std()
    sma = c.rolling(200).mean()
    vt = (sn / sr).clip(upper=1.0)
    ma = (c > sma).astype(float)
    sr_a = sr * np.sqrt(ANN)
    return dict(day=c.index[-1], close=c.iloc[-1], ret=r.iloc[-1], sr=sr_a.iloc[-1], sr_prev=sr_a.iloc[-2],
                sn=sn.iloc[-1] * np.sqrt(ANN), pct=(sr_a["1981":].dropna() <= sr_a.iloc[-1]).mean(),
                vt=vt.iloc[-1], vt_prev=vt.iloc[-2], sma=sma.iloc[-1], gap=c.iloc[-1] / sma.iloc[-1] - 1,
                ma=ma.iloc[-1], ma_prev=ma.iloc[-2], co=ma.iloc[-1] * vt.iloc[-1])


def compose(s):
    d = s["day"].strftime("%Y-%m-%d")
    side = "위" if s["ma"] else "아래"
    flip = "" if s["ma"] == s["ma_prev"] else (" (오늘 전환: 매수 신호)" if s["ma"] else " (오늘 전환: 매도 신호)")
    subject = (f"[KOSPI 변동성] {d} 최근 {s['sr']*100:.1f}% · 평소 {s['sn']*100:.1f}% · 비중 {s['vt']*100:.1f}%"
               f"{' · 200일선 전환' if flip else ''}")
    rows = [
        ("KOSPI 종가", f"{s['close']:,.2f} ({s['ret']*100:+.2f}%)"),
        ("최근 변동성(21일, 연율)", f"{s['sr']*100:.1f}% (전일 {s['sr_prev']*100:.1f}%)"),
        ("평소 변동성(5년, 연율)", f"{s['sn']*100:.1f}%"),
        ("최근 변동성의 역사적 위치", f"1981년 이후 상위 {(1-s['pct'])*100:.1f}%"),
        ("변동성 타기팅 비중", f"{s['vt']*100:.1f}% (전일 {s['vt_prev']*100:.1f}%)"),
        ("200일선", f"{s['sma']:,.0f}, 종가가 {abs(s['gap'])*100:.1f}% {side}{flip}"),
        ("200일선 규칙 비중", f"{s['ma']*100:.0f}%"),
        ("결합 규칙 비중", f"{s['co']*100:.1f}%"),
    ]
    note = "변동성 타기팅 비중 = min(평소 변동성 ÷ 최근 변동성, 1). Girgin(2026) 규칙의 계산값이며 투자 권유가 아닙니다."
    text = f"KOSPI 변동성 {d}\n\n" + "\n".join(f"{a}: {b}" for a, b in rows) + f"\n\n{note}\n"
    tr = "".join(f'<tr><td style="padding:6px 14px 6px 0;color:#5f5e57;border-bottom:1px solid #e8e7e1">{a}</td>'
                 f'<td style="padding:6px 0;border-bottom:1px solid #e8e7e1;font-weight:600">{b}</td></tr>' for a, b in rows)
    html = (f'<div style="font-family:Malgun Gothic,sans-serif;font-size:14px;color:#23231f">'
            f'<h3 style="margin:0 0 10px">KOSPI 변동성 {d}</h3><table style="border-collapse:collapse">{tr}</table>'
            f'<p style="font-size:12px;color:#5f5e57;margin-top:14px">{note}</p></div>')
    return subject, text, html


def send(subject, text, html):
    user = (os.getenv("EMAIL_USER") or "").strip()
    pw = (os.getenv("EMAIL_PASS") or "").replace(" ", "").strip()
    to = (os.getenv("VOL_MAIL_TO") or "").strip()
    missing = [k for k, v in (("EMAIL_USER", user), ("EMAIL_PASS", pw), ("VOL_MAIL_TO", to)) if not v]
    if missing:
        raise RuntimeError(f"Secret 미설정: {missing}")
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.attach(MIMEText(text, "plain", "utf-8"))
    msg.attach(MIMEText(html, "html", "utf-8"))
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as srv:
        srv.starttls(context=ssl.create_default_context())
        srv.login(user, pw)
        srv.send_message(msg)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    c = update_history()
    s = summarize(c)
    subject, text, html = compose(s)
    day = str(s["day"].date())
    if "--dry-run" in sys.argv:
        print(subject)
        print(text)
        return
    force = os.getenv("FORCE_EMAIL", "") == "1"
    if not force and STATE.exists() and STATE.read_text(encoding="utf-8").strip() >= day:
        print(f"이미 발송한 종가일({day}), 건너뜀")
        return
    send((os.getenv("SUBJECT_PREFIX") or "") + subject, text, html)
    STATE.write_text(day, encoding="utf-8")
    print(f"발송 완료: {day}")  # 공개 저장소이므로 수신 주소는 로그에 남기지 않는다


if __name__ == "__main__":
    main()
