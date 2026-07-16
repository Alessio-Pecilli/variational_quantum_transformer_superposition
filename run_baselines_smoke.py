#!/usr/bin/env python3
"""Training curves: k-QSA / k-CSA / nl-CSA with shared data+init for mu-models."""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from classical_baselines import (
    BaselineConfig,
    prepare_shared_bundle,
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
    p.add_argument("--nl-learning-rate", type=float, default=5e-3)
    p.add_argument("--nl-rank", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.T, args.d, args.epochs, args.max_sentences = 4, 4, 30, 32

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("baselines")

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
        run_label="fixed",
        nl_rank=args.nl_rank,
        nl_learning_rate=args.nl_learning_rate,
    )

    budget = qsa_angle_param_count(cfg.d, cfg.layers)
    print("=" * 60)
    print("BASELINES (fixed) — shared data/init for k-QSA & k-CSA")
    print(f"T={cfg.T} d={cfg.d} k={cfg.k} L={cfg.layers} epochs={cfg.epochs} frasi={cfg.max_sentences}")
    print(f"Angle budget W+V: {budget}")
    print(f"Output: {out}")
    print("=" * 60)

    bundle = prepare_shared_bundle(cfg)
    print(f"Shared vocab={bundle['encoding'].vocabSize} sentences={len(bundle['sentences'])}")

    results = []
    print("\n--- k-QSA (classical mu path) ---")
    results.append(train_kqsa(cfg, bundle, logger=log))
    print("\n--- k-CSA (same mu, shared init) ---")
    results.append(train_kcsa(cfg, bundle, logger=log))
    print("\n--- nl-CSA ---")
    results.append(train_nlcsa(cfg, bundle, logger=log))

    # identity check
    gap = abs(results[0]["final_loss"] - results[1]["final_loss"])
    max_ep_gap = max(
        abs(a - b) for a, b in zip(results[0]["loss_history"], results[1]["loss_history"])
    )
    print(f"\n[CHECK] max |loss_QSA - loss_CSA| over epochs = {max_ep_gap:.3e}")
    print(f"[CHECK] |final_QSA - final_CSA| = {gap:.3e}")

    plot_training_curves(results, out / "training_curves.png")
    summary = {
        "config": {
            "T": cfg.T, "d": cfg.d, "k": cfg.k, "layers": cfg.layers,
            "epochs": cfg.epochs, "max_sentences": cfg.max_sentences,
            "angle_budget": budget,
            "vocab_size": int(bundle["encoding"].vocabSize),
        },
        "qsa_csa_max_epoch_gap": max_ep_gap,
        "models": [
            {
                "model": r["model"],
                "final_loss": r["final_loss"],
                "n_params_angles_or_model": r.get("n_params_angles") or r.get("n_params_model"),
                "n_params_embedding": r.get("n_params_embedding"),
                "n_params_total": r.get("n_params_total"),
                "extra": {
                    k: r[k]
                    for k in (
                        "ppl_mu", "final_ppl", "nl_rank", "nl_learning_rate",
                        "target_angle_budget", "note",
                    )
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
