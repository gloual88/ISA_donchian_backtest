# -*- coding: utf-8 -*-
"""
track_record.json 1회 시드 — git 이력의 data/signals.json에서 asof별 '마지막 발행' 비중 복원.
(이미 파일이 있으면 기존 날짜는 덮어쓰지 않고 없는 날짜만 채운다.)

실행: python seed_track_record.py [--since 2026-05-25]
"""
import argparse
import json
import subprocess

from track_record import load_record, save_record


def _git(*a):
    return subprocess.run(["git", *a], capture_output=True).stdout.decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-05-25")
    since = ap.parse_args().since
    hashes = _git("log", "main", f"--since={since}", "--format=%h", "--",
                  "data/signals.json").split()       # 최신→과거
    found = {}
    for h in hashes:
        try:
            d = json.loads(_git("show", f"{h}:data/signals.json"))
        except Exception:
            continue
        a = d.get("asof")
        if a and a not in found and d.get("positions"):
            found[a] = {p["label"]: float(p["isa_weight_pct"])
                        for p in d["positions"]}
    rec = load_record()
    added = 0
    for a, w in found.items():
        if a not in rec:
            rec[a] = w
            added += 1
    save_record(rec)
    print(f"시드 완료: 기록 {len(rec)}일 (신규 {added}) "
          f"{min(rec)} ~ {max(rec)}")


if __name__ == "__main__":
    main()
