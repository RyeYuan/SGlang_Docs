#!/usr/bin/env python3
"""Summarize DSV4 decode-side radix cache hits from a decode log.

Usage: python3 radix_hit_stats.py <decode.log> [--per-dp] [--since HH:MM:SS]

Requires the decode server to run with
SGLANG_DEBUG_DSV4_DECODE_RADIX_TRANSFER=1, which emits one line per request:
  DSV4 decode radix transfer stats: rid=.. origin_input_len=N
  decode_prefix_len=P transfer_tokens=T transfer_pages=.. page_size=..
P == 0 means a miss (the whole prompt is transferred); P > 0 means a hit and
only the [P, N) suffix crosses the wire.
"""
import re
import sys
from collections import defaultdict

PAT = re.compile(
    r"\[[\d-]+ (?P<ts>\d\d:\d\d:\d\d) (?P<dp>DP\d+)[^\]]*\].*?"
    r"radix transfer stats: rid=(?P<rid>\S+) origin_input_len=(?P<origin>\d+) "
    r"decode_prefix_len=(?P<prefix>\d+) transfer_tokens=(?P<xfer>\d+)"
)


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    path = args[0]
    per_dp = "--per-dp" in args
    since = None
    if "--since" in args:
        since = args[args.index("--since") + 1]

    rows = []
    for line in open(path, errors="ignore"):
        if "HEALTH_CHECK" in line:
            continue
        m = PAT.search(line)
        if not m:
            continue
        if since and m["ts"] < since:
            continue
        rows.append(
            (m["dp"], int(m["origin"]), int(m["prefix"]), int(m["xfer"]))
        )

    if not rows:
        sys.exit("no transfer-stats lines matched (is the debug env var set?)")

    n = len(rows)
    hits = [r for r in rows if r[2] > 0]
    origin = sum(r[1] for r in rows)
    xfer = sum(r[3] for r in rows)

    print(f"requests         : {n}")
    print(f"cache hits       : {len(hits)}  ({len(hits) / n * 100:.1f}%)")
    print(f"prompt tokens    : {origin}")
    print(f"transferred      : {xfer}")
    print(f"tokens saved     : {origin - xfer}  ({(origin - xfer) / origin * 100:.1f}%)")
    if hits:
        hp = sum(r[2] for r in hits) / len(hits)
        hx = sum(r[3] for r in hits) / len(hits)
        ho = sum(r[1] for r in hits) / len(hits)
        print(f"  on a hit       : prompt {ho:.0f} -> transfer {hx:.0f} "
              f"(prefix {hp:.0f}, saved {(ho - hx) / ho * 100:.1f}%)")
    miss = [r for r in rows if r[2] == 0]
    if miss:
        mo = sum(r[1] for r in miss) / len(miss)
        print(f"  on a miss      : prompt {mo:.0f} -> transfer {mo:.0f} (saved 0%)")

    if per_dp:
        print("\nper DP rank:")
        agg = defaultdict(lambda: [0, 0, 0, 0])
        for dp, o, p, x in rows:
            a = agg[dp]
            a[0] += 1
            a[1] += 1 if p > 0 else 0
            a[2] += o
            a[3] += x
        for dp in sorted(agg, key=lambda d: int(d[2:])):
            c, h, o, x = agg[dp]
            print(f"  {dp}: {c} reqs, {h} hits ({h / c * 100:.0f}%), "
                  f"{o} -> {x} tokens (saved {(o - x) / o * 100:.1f}%)")


if __name__ == "__main__":
    main()
