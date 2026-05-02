#!/usr/bin/env python3
"""
Scan output_ssv2/eval_*/log.txt JSON lines and report best test_acc1 / test_acc5 per run.

Usage:
  python scripts/report_ssv2_eval_best.py
  python scripts/report_ssv2_eval_best.py --prefix eval_shared_trunk
  python scripts/report_ssv2_eval_best.py --markdown
"""
import argparse
import json
import os
import sys
from typing import Optional, Tuple


def best_in_log(path: str) -> Optional[Tuple[float, float, int]]:
    if not os.path.isfile(path):
        return None
    best = (-1.0, -1.0, -1)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            a1 = float(d.get("test_acc1", -1))
            if a1 > best[0]:
                best = (
                    a1,
                    float(d.get("test_acc5", -1)),
                    int(d.get("epoch", -1)),
                )
    if best[0] < 0:
        return None
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=os.path.join(os.path.dirname(__file__), "..", "output_ssv2"),
        help="Directory containing eval_* folders",
    )
    ap.add_argument(
        "--prefix",
        default="",
        help="Only include eval dirs whose name starts with this (e.g. eval_shared_trunk)",
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="Print a markdown table sorted by Top-1 descending",
    )
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"No such directory: {root}", file=sys.stderr)
        sys.exit(1)

    rows: list[tuple[float, float, int, str]] = []
    for name in sorted(os.listdir(root)):
        if not name.startswith("eval_"):
            continue
        if args.prefix and not name.startswith(args.prefix.strip()):
            continue
        log_path = os.path.join(root, name, "log.txt")
        b = best_in_log(log_path)
        if b is None:
            continue
        rows.append((b[0], b[1], b[2], name))

    rows.sort(key=lambda x: -x[0])

    if args.markdown:
        print("| Eval dir | Best Top-1 (%) | Top-5 (%) | Epoch |")
        print("|---|---:|---:|---:|")
        for a1, a5, ep, name in rows:
            print(
                f"| `{name}` | {a1:.2f} | {a5:.2f} | {ep} |"
            )
    else:
        for a1, a5, ep, name in rows:
            print(f"{a1:6.2f}  {a5:6.2f}  ep{ep:2d}  {name}")


if __name__ == "__main__":
    main()
