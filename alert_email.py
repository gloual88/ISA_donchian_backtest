# -*- coding: utf-8 -*-
"""
ISA 추세전략 — 일일 리포트 이메일 (수익률 현황 + 오늘의 액션)
================================================================
data/signals.json을 읽어 매일 이메일 발송:
  · 수익률 현황: 기준일(ANCHOR_DATE, 기본 2026-06-02·포트폴리오 시작) 이후 누적 + 당일
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


def _transition_html():
    """전략 전환 리밸런싱(1회성) — data/transition.json 있을 때만 블록 생성."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent / "data" / "transition.json"
    if not p.exists():
        return ""
    t = json.loads(p.read_text(encoding="utf-8"))
    fp, tp = t["from_params"], t["to_params"]
    items = ""
    for r in t["rows"]:
        if r["action"] == "유지":
            continue
        c = "#16a34a" if r["delta"] > 0 else "#dc2626"
        items += (f"<li>{r['etf']} ({r['code']}) "
                  f"<b style='color:{c}'>{r['action']} {r['delta']:+.1f}%p</b> "
                  f"<span style='color:#888'>"
                  f"{r['old_w']:.1f}%→{r['new_w']:.1f}%</span></li>")
    if not items:
        return ""
    return (f"<h3>🔄 전략 전환 리밸런싱 (1회성)</h3>"
            f"<p style='font-size:0.85rem;color:#555'>전략 변경 "
            f"{fp.get('scheme', '')} → <b>{tp.get('scheme', '')}</b> "
            f"(N={tp['N']}/lag={tp['lag']}, 기준일 {t['asof']}). 구→신 전환을 "
            f"위한 일회성 비중 조정이며, 아래 '오늘의 액션'(신전략 자생 신호)과는 "
            f"별개입니다.</p><ul>{items}</ul>")


LABELS = {"strategy": "전략", "sixty_forty": "60/40",
          "ew_basket": "동일가중바스켓", "kospi": "KOSPI200(참고)"}


def build_report(asof, s, curves):
    stats = {n: compute_status(c, ANCHOR) for n, c in curves.items()}
    st = stats["strategy"]
    buys = s.get("buy_today", [])
    stops = s.get("stop_today", [])
    near = [n for n in s.get("near_stop", []) if n["stop_room_pct"] < NEAR]
    n_act = len(buys) + len(stops) + len(near)

    def cell(v):
        c = "#16a34a" if v >= 0 else "#dc2626"
        return f"<b style='color:{c}'>{v*100:+.2f}%</b>"

    def row(name):
        m = stats[name]
        nm = f"<b>{LABELS[name]}</b>" if name == "strategy" else LABELS[name]
        return (f"<tr><td>{nm}</td>"
                f"<td align=right>{cell(m['daily_return'])}</td>"
                f"<td align=right>{cell(m['cum_return'])}</td>"
                f"<td align=right>{m['mdd_since']*100:.1f}%</td></tr>")

    rows_html = "".join(
        row(n) for n in ["strategy", "sixty_forty", "ew_basket", "kospi"]
        if n in stats)
    excess = ""
    if "sixty_forty" in stats:
        d = (st['cum_return'] - stats['sixty_forty']['cum_return']) * 100
        excess = f"기준일 이후 초과수익(전략−60/40): <b>{d:+.1f}%p</b> · "

    prefix = os.getenv("SUBJECT_PREFIX", "")   # 예: "[정정] " — 정정 발송 표시
    subj = (f"{prefix}[ISA 추세전략] 일일 리포트 {asof} — "
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
        {rows_html}
      </table>
      <p style="font-size:0.85rem;color:#555">{excess}
        보유 {s['n_positions']}종목 · 현금 {max(s['cash_pct'], 0):.0f}%</p>

      {_transition_html()}
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
    asof, curves = load_curves()
    import json
    from pathlib import Path
    here = Path(__file__).resolve().parent
    s = json.loads(
        (here / "data" / "signals.json").read_text(encoding="utf-8"))

    # 중복 발송 방지: 같은 종가일(asof)로 이미 보냈으면 생략(매일 1통 보장).
    # 백업 cron이 같은 날 또 실행돼도 두 번째는 스킵. FORCE_EMAIL=1로 강제 가능.
    signals_asof = str(s.get("asof", "")).strip()
    state = here / "data" / "last_email_sent.txt"
    last_sent = state.read_text(encoding="utf-8").strip() if state.exists() else ""
    force = os.getenv("FORCE_EMAIL", "").strip().lower() not in ("", "0", "false")
    if signals_asof and signals_asof == last_sent and not force:
        print(f"이미 발송됨(종가일 {signals_asof}) — 중복 방지로 생략")
        return

    subj, html = build_report(asof, s, curves)

    import re
    user = (os.getenv("EMAIL_USER") or "").strip()
    pw = (os.getenv("EMAIL_PASS") or "").replace(" ", "").strip()
    to_raw = (os.getenv("EMAIL_TO") or user)
    # 줄바꿈/쉼표/세미콜론 모두 허용 → 정제된 수신자 리스트
    recipients = [a.strip() for a in re.split(r"[,\n;]+", to_raw) if a.strip()]
    if not (user and pw and recipients):
        print("EMAIL_USER/EMAIL_PASS 미설정 — 발송 생략(드라이런)")
        print("제목:", subj)
        return

    # 다수 구독자: 1인씩 개별 발송 → 각 수신자는 To에 자기 주소만 보이고
    # 다른 수신자의 존재/주소를 전혀 알 수 없음(상호 노출 0). 연결은 1회 재사용.
    ctx = ssl.create_default_context()
    sent, failed = 0, []
    with smtplib.SMTP("smtp.gmail.com", 587) as srv:
        srv.starttls(context=ctx)
        srv.login(user, pw)
        for rcpt in recipients:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subj
            msg["From"] = user
            msg["To"] = rcpt
            msg.attach(MIMEText(html, "html", "utf-8"))
            try:
                srv.send_message(msg)
                sent += 1
            except Exception as e:  # noqa: BLE001 — 1명 실패해도 나머지 계속
                failed.append(f"{rcpt}({e})")
    line = f"발송 완료 {sent}/{len(recipients)}명 (개별 발송) | {subj}"
    if failed:
        line += f" | 실패: {', '.join(failed)}"
    print(line)

    # 발송 성공 시 종가일 기록(다음 백업 실행이 중복 발송하지 않도록)
    if sent > 0 and signals_asof:
        state.write_text(signals_asof, encoding="utf-8")
        print(f"발송 기록 갱신: {state.name} = {signals_asof}")


if __name__ == "__main__":
    main()
