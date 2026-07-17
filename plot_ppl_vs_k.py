#!/usr/bin/env python3
"""Aggregate baseline summary.json across k folders into a line plot with error bars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from classical_baselines import plot_final_loss_vs_k


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help="e.g. results/baselines_smoke/ppl_vs_k_T16_d16_ep300_n5",
    )
    p.add_argument("--ks", type=str, default="1,2,3,4")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    root = Path(args.root)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    points = []
    for k in ks:
        summary_path = root / f"k{k}" / "summary.json"
        if not summary_path.exists():
            print(f"[skip] missing {summary_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for m in summary.get("models", []):
            points.append(
                {
                    "k": k,
                    "model": m["model"],
                    "final_loss_mean": m["final_loss_mean"],
                    "final_loss_std": m["final_loss_std"],
                }
            )

    if not points:
        raise SystemExit(f"no summary.json found under {root}/k*/")

    out = Path(args.out) if args.out else root / "final_loss_vs_k.png"
    plot_final_loss_vs_k(points, out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
