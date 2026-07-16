#!/usr/bin/env python3
"""Local smoke / serious check: training curves for k-QSA, k-CSA, nl-CSA."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from config import DATASET_CONFIG
from classical_baselines import (
    BaselineConfig,
    plot_training_curves,
    qsa_angle_param_count,
    train_kcsa,
    train_kqsa,
    train_nlcsa,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-sentences", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true", help="T=4 d=4 30ep 32 frasi")
    args = p.parse_args()

    if args.quick:
        args.T, args.d, args.epochs, args.max_sentences = 4, 4, 30, 32

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("baselines")

    DATASET_CONFIG["sentence_length"] = args.T
    DATASET_CONFIG["max_sentences"] = args.max_sentences

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("results") / "baselines_smoke" / stamp
    out.mkdir(parents=True, exist_ok=True)

    cfg = BaselineConfig(
        T=args.T,
        d=args.d,
        k=args.k,
        layers=args.layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_sentences=args.max_sentences,
        seed=args.seed,
        output_dir=str(out),
        run_label="serious",
    )

    budget = qsa_angle_param_count(cfg.d, cfg.layers)
    print("=" * 60)
    print("BASELINES — training curves (k-QSA / k-CSA / nl-CSA)")
    print(f"T={cfg.T} d={cfg.d} k={cfg.k} L={cfg.layers} epochs={cfg.epochs} frasi={cfg.max_sentences}")
    print(f"Target angle budget (W+V): {budget} = 2*L*log2(d)")
    print(f"Output: {out}")
    print("=" * 60)

    results = []
    print("\n--- k-QSA ---")
    results.append(train_kqsa(cfg, logger=log))
    print("\n--- k-CSA ---")
    results.append(train_kcsa(cfg, logger=log))
    print("\n--- nl-CSA ---")
    results.append(train_nlcsa(cfg, logger=log))

    plot_training_curves(results, out / "training_curves.png")
    summary = {
        "config": {
            "T": cfg.T, "d": cfg.d, "k": cfg.k, "layers": cfg.layers,
            "epochs": cfg.epochs, "max_sentences": cfg.max_sentences,
            "angle_budget": budget,
        },
        "models": [
            {
                "model": r["model"],
                "final_loss": r["final_loss"],
                "n_params": r.get("n_params_angles") or r.get("n_params_model"),
                "n_params_embedding": r.get("n_params_embedding"),
                "n_params_total": r.get("n_params_total") or r.get("n_params_angles"),
                "extra": {
                    k: r[k]
                    for k in ("obar", "ppl_mu", "final_ppl", "nl_rank", "target_angle_budget")
                    if k in r
                },
            }
            for r in results
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nPlot: {out / 'training_curves.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
