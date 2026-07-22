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
        help="e.g. results/debug_convergence/T8_d8_...",
    )
    p.add_argument("--ks", type=str, default="1,2,3,4")
    p.add_argument("--out", type=str, default=None)
    p.add_argument(
        "--metric",
        choices=("val_ppl", "train_loss"),
        default="val_ppl",
        help="val_ppl = common validation perplexity; train_loss = model-specific",
    )
    p.add_argument(
        "--nl-ref",
        type=str,
        default=None,
        help="path to nl-CSA summary.json for horizontal reference (k-independent)",
    )
    args = p.parse_args()

    root = Path(args.root)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    points = []
    for k in ks:
        summary_path = root / f"k{k}" / "summary.json"
        if not summary_path.exists():
            # try flat layout (convergence debug summary)
            summary_path = root / "summary.json"
            if not summary_path.exists():
                print(f"[skip] missing summary for k={k}")
                continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "models" in summary:
            for m in summary.get("models", []):
                if args.metric == "val_ppl":
                    points.append(
                        {
                            "k": k,
                            "model": m["model"],
                            "final_val_ppl_mean": m.get("final_val_ppl", m.get("final_val_ppl_mean")),
                            "final_val_ppl_std": m.get("final_val_ppl_std", 0.0),
                        }
                    )
                else:
                    points.append(
                        {
                            "k": k,
                            "model": m["model"],
                            "final_loss_mean": m.get("final_loss", m.get("final_loss_mean")),
                            "final_loss_std": m.get("final_loss_std", 0.0),
                        }
                    )
        elif "per_k_status" in summary:
            for row in summary["per_k_status"]:
                if int(row["k"]) != k:
                    continue
                if args.metric == "val_ppl":
                    points.append(
                        {
                            "k": k,
                            "model": row["model"],
                            "final_val_ppl_mean": row["final_val_ppl"],
                            "final_val_ppl_std": 0.0,
                        }
                    )

    if not points:
        raise SystemExit(f"no data found under {root}")

    nl_ref = None
    if args.nl_ref:
        nl_ref = json.loads(Path(args.nl_ref).read_text(encoding="utf-8"))
        if "models" in nl_ref:
            nl_ref = next((m for m in nl_ref["models"] if "nl" in m["model"].lower()), nl_ref["models"][0])

    if args.metric == "val_ppl":
        out = Path(args.out) if args.out else root / "val_ppl_vs_k.png"
        plot_final_loss_vs_k(
            points,
            out,
            title="Validation perplexity vs k",
            ykey="final_val_ppl_mean",
            ylabel="validation perplexity",
            nl_ref=nl_ref,
        )
    else:
        out = Path(args.out) if args.out else root / "final_loss_vs_k.png"
        plot_final_loss_vs_k(points, out, ykey="final_loss_mean", ylabel="train loss", nl_ref=nl_ref)

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
