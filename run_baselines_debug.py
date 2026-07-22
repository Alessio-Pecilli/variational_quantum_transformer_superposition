#!/usr/bin/env python3
"""Exploratory baseline tests (single seed) — debug phase only.

Runs k-QSA, independent k-CSA, and nl-CSA ablations with common val perplexity.
Does NOT launch expensive multi-seed / obar campaigns.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from classical_baselines import (
    NL_ABLATIONS,
    BaselineConfig,
    aggregate_seed_results,
    plot_convergence_diagnostics,
    plot_training_curves,
    prepare_data_bundle,
    qsa_angle_param_count,
    kcsa_matrix_param_count,
    train_kcsa,
    train_kqsa,
    train_nlcsa,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Single-seed baseline debug runner")
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--max-sentences", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--nl-learning-rate", type=float, default=5e-3)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--model-seed", type=int, default=1042)
    p.add_argument("--batch-size", type=int, default=64, help="0 = full batch")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--checkpoint-every", type=int, default=50)
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True)
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--loss-rel-tol", type=float, default=1e-4)
    p.add_argument("--convergence-patience", type=int, default=12)
    p.add_argument("--kernel-mode", choices=("monomial", "poly"), default="monomial")
    p.add_argument("--nl-ablations", action="store_true", help="run all 4 nl-CSA ablations")
    p.add_argument("--nl-embedding", choices=("general", "isometric"), default="isometric")
    p.add_argument("--nl-loss", choices=("ce", "renyi"), default="renyi")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.T, args.d, args.epochs, args.max_sentences = 4, 4, 80, 64
        args.batch_size = 16

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir or f"results/debug_baselines/T{args.T}_d{args.d}_k{args.k}_{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("debug_baselines")

    batch_size = None if args.batch_size <= 0 else args.batch_size
    max_epochs = args.max_epochs or args.epochs

    cfg = BaselineConfig(
        T=args.T,
        d=args.d,
        k=args.k,
        layers=args.layers,
        epochs=args.epochs,
        max_epochs=max_epochs,
        learning_rate=args.learning_rate,
        max_sentences=args.max_sentences,
        seed=args.data_seed,
        data_seed=args.data_seed,
        model_seed=args.model_seed,
        output_dir=str(out),
        nl_learning_rate=args.nl_learning_rate,
        batch_size=batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        kernel_mode=args.kernel_mode,
        early_stop=args.early_stop,
        loss_rel_tol=args.loss_rel_tol,
        convergence_patience=args.convergence_patience,
    )

    bundle = prepare_data_bundle(cfg)
    print("=" * 70)
    print("BASELINE DEBUG (single seed)")
    print(f"T={args.T} d={args.d} k={args.k} epochs<={max_epochs}")
    print(f"data_seed={args.data_seed} model_seed={args.model_seed}")
    print(f"sentences={len(bundle['sentences'])} train={bundle['train_idx'].size} val={bundle['val_idx'].size}")
    print(f"k-QSA angles={qsa_angle_param_count(args.d, args.layers)}")
    print(f"k-CSA matrices={kcsa_matrix_param_count(args.d)}")
    print(f"Output: {out}")
    print("=" * 70)

    qsa = train_kqsa(cfg, bundle, logger=log)
    csa = train_kcsa(cfg, bundle, logger=log)

    gap = max(abs(a - b) for a, b in zip(qsa["loss_history"], csa["loss_history"]))
    val_gap = abs(qsa["final_val_ppl"] - csa["final_val_ppl"])
    print(f"\n[k-QSA vs k-CSA] max train |diff|={gap:.4f}  val_ppl_diff={val_gap:.4f}")
    print("  (Models are now independent; curves should NOT match.)")

    nl_runs = []
    ablations = NL_ABLATIONS if args.nl_ablations else [(args.nl_embedding, args.nl_loss)]
    for emb_mode, loss_mode in ablations:
        nl_cfg = BaselineConfig(**{**asdict_safe(cfg), "nl_embedding_mode": emb_mode, "nl_loss_mode": loss_mode})
        print(f"\n--- nl-CSA ablation: embedding={emb_mode} loss={loss_mode} ---")
        nl_runs.append(train_nlcsa(nl_cfg, bundle, logger=log))

    all_runs = [qsa, csa] + nl_runs
    agg = [aggregate_seed_results([r]) for r in all_runs]
    plot_training_curves(agg, out / "training_curves.png")
    plot_convergence_diagnostics(all_runs, out / "convergence_diagnostics.png")

    summary = {
        "phase": "debug_single_seed",
        "config": {
            "T": args.T, "d": args.d, "k": args.k, "layers": args.layers,
            "epochs": args.epochs, "max_epochs": max_epochs,
            "data_seed": args.data_seed, "model_seed": args.model_seed,
            "batch_size": batch_size, "early_stop": args.early_stop,
            "output_dir": str(out),
        },
        "parametrization": cfg.parametrization,
        "k_qsa_vs_kcsa": {
            "train_max_abs_diff": gap,
            "val_ppl_diff": val_gap,
            "k_qsa_params": {
                "angles": qsa["n_params_angles"],
                "embedding": qsa["n_params_embedding"],
                "total": qsa["n_params_total"],
            },
            "k_csa_params": {
                "matrices": csa["n_params_matrices"],
                "embedding": csa["n_params_embedding"],
                "total": csa["n_params_total"],
            },
            "conclusion": (
                "Previously identical (shared init/angles). "
                "Now k-CSA uses independent d×d W,V with separate model_seed."
            ),
        },
        "models": [
            {
                "model": r["model"],
                "ablation": r.get("ablation"),
                "final_loss": r["final_loss"],
                "final_val_ce": r["final_val_ce"],
                "final_val_ppl": r["final_val_ppl"],
                "epochs_run": r["epochs_run"],
                "stopped_early": r.get("stopped_early", False),
                "n_params_total": r.get("n_params_total"),
            }
            for r in all_runs
        ],
        "nl_ablations": [
            {"embedding": e, "loss": l} for e, l in ablations
        ],
        "note": "Single seed debug run. Final campaigns need n_seeds>1 with error bars.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out / 'summary.json'}")
    print(f"Plots: {out / 'training_curves.png'}, {out / 'convergence_diagnostics.png'}")
    return 0


def asdict_safe(cfg: BaselineConfig) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
