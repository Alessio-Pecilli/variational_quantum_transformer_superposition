#!/usr/bin/env python3
"""Merge mu summary.json packs (same row schema) into one JSON for fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="summary.json paths")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--prefer", choices=("first", "last"), default="last",
                   help="On duplicate (T,d,k,model), keep first or last")
    args = p.parse_args()

    by_key: dict[tuple, dict] = {}
    sources: list[str] = []
    for path in args.inputs:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sources.append(str(path))
        for row in data.get("rows", []):
            key = (int(row["T"]), int(row["d"]), int(row["k"]), str(row["model"]))
            if key in by_key and args.prefer == "first":
                continue
            by_key[key] = row

    rows = [by_key[k] for k in sorted(by_key)]
    out = {
        "sources": sources,
        "n_cells": len(rows),
        "advantage_formula": "k^2 * log(d) / C(d+k-1,k)",
        "mu_key": "train_mu_final",
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.output} with {len(rows)} rows from {len(sources)} packs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
