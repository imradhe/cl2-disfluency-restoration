#!/usr/bin/env python
"""Phase 6, step 1: audit NPTEL .rsml files -> data/nptel/manifest.csv with keep/drop reasons.

Keep rule (agreed with the project owner): English lecture AND every segment verified.
Audio is matched by file stem next to the .rsml (wav/mp3/m4a/flac); missing audio is reported.

Run:  .venv/bin/python -I scripts/nptel_manifest.py [--dir data/nptel/raw]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from nptel.rsml import parse_rsml  # noqa: E402

AUDIO_EXT = (".wav", ".mp3", ".m4a", ".flac", ".mp4", ".webm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "data/nptel/raw"))
    ap.add_argument("--out", default=str(ROOT / "data/nptel/manifest.csv"))
    a = ap.parse_args()
    root = Path(a.dir)
    rows = []
    for p in sorted(root.rglob("*.rsml")):
        if p.name.endswith(".aligned.rsml"):
            continue
        r = parse_rsml(p)
        audio = next((x for ext in AUDIO_EXT for x in [p.with_suffix(ext)] if x.exists()), None)
        if not r.segments:
            reason = "drop: no segments parsed"
        elif not r.is_english:
            reason = f"drop: not English (non-Latin script {r.script_ratio:.0%})"
        elif not r.fully_verified:
            reason = f"drop: only {r.n_verified}/{len(r.segments)} segments verified"
        else:
            reason = "keep"
        rows.append({
            "file": p.name, "segments": len(r.segments), "verified": r.n_verified,
            "fully_verified": r.fully_verified, "english": r.is_english,
            "minutes": round(r.duration / 60, 1), "audio": audio.name if audio else "MISSING",
            "aligned": p.with_name(p.stem + ".aligned.rsml").exists(),
            "decision": reason.split(":")[0], "reason": reason,
        })
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["file"])
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['file']:30s} segs={r['segments']:4d} verified={r['verified']:4d} en={str(r['english']):5s} "
              f"{r['minutes']:5.1f}min audio={r['audio']:10s} -> {r['reason']}")
    keep = [r for r in rows if r["decision"] == "keep"]
    print(f"\nkept {len(keep)}/{len(rows)}; kept with audio: {sum(r['audio'] != 'MISSING' for r in keep)}")


if __name__ == "__main__":
    main()
