#!/usr/bin/env python3
"""Convergence diagnostics for k=1,2,3,4 (single seed, debug phase).

Saves full loss curves, gradient norms, relative improvement; tests early stopping.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from classical_baselines import (
    BaselineConfig,
    plot_convergence_diagnostics,
    prepare_data_bundle,
    train_kcsa,
    train_kqsa,
)


def main() -> int:
    p = argparse.ArgumentParser(description="k-sweep convergence diagnostics (single seed)")
    p.add_argument("--T", type=int, default=8)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--ks", type=str, default="1,2,3,4")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--max-sentences", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--learning-rates", type=str, default=None, help="comma-separated LRs to try per k")
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--model-seed", type=int, default=1042)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--early-stop", action="store_true", default=True)
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    p.add_argument("--loss-rel-tol", type=float, default=1e-4)
    p.add_argument("--convergence-patience", type=int, default=15)
    p.add_argument("--models", type=str, default="both", choices=("kqsa", "kcsa", "both"))
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.T, args.d, args.epochs, args.max_epochs, args.max_sentences = 4, 4, 60, 120, 64

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir or f"results/debug_convergence/T{args.T}_d{args.d}_{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("convergence_debug")

    batch_size = None if args.batch_size <= 0 else args.batch_size
    lr_list = (
        [float(x) for x in args.learning_rates.split(",")]
        if args.learning_rates
        else [args.learning_rate]
    )

    results_by_k: dict = {}
    all_runs = []

    print("=" * 70)
    print("CONVERGENCE DEBUG — k sweep (single seed)")
    print(f"T={args.T} d={args.d} ks={ks} max_epochs={args.max_epochs}")
    print(f"data_seed={args.data_seed} model_seed={args.model_seed} LRs={lr_list}")
    print(f"early_stop={args.early_stop} patience={args.convergence_patience}")
    print(f"Output: {out}")
    print("=" * 70)

    for k in ks:
        k_dir = out / f"k{k}"
        k_dir.mkdir(parents=True, exist_ok=True)
        k_results = []

        for lr in lr_list:
            lr_tag = f"lr{lr:g}".replace(".", "p")
            run_dir = k_dir / lr_tag
            cfg = BaselineConfig(
                T=args.T,
                d=args.d,
                k=k,
                layers=args.layers,
                epochs=args.epochs,
                max_epochs=args.max_epochs,
                learning_rate=lr,
                max_sentences=args.max_sentences,
                seed=args.data_seed,
                data_seed=args.data_seed,
                model_seed=args.model_seed,
                output_dir=str(run_dir),
                batch_size=batch_size,
                checkpoint_every=50,
                resume=False,
                early_stop=args.early_stop,
                loss_rel_tol=args.loss_rel_tol,
                convergence_patience=args.convergence_patience,
            )
            bundle = prepare_data_bundle(cfg)

            runs_this = []
            if args.models in ("kqsa", "both"):
                print(f"\n--- k={k} lr={lr} k-QSA ---")
                runs_this.append(train_kqsa(cfg, bundle, logger=log))
            if args.models in ("kcsa", "both"):
                print(f"\n--- k={k} lr={lr} k-CSA ---")
                runs_this.append(train_kcsa(cfg, bundle, logger=log))

            for r in runs_this:
                rel_last5 = r.get("rel_improve_history", [])[-5:]
                entry = {
                    "k": k,
                    "lr": lr,
                    "model": r["model"],
                    "epochs_run": r["epochs_run"],
                    "stopped_early": r.get("stopped_early", False),
                    "stop_reason": r.get("stop_reason", ""),
                    "final_loss": r["final_loss"],
                    "final_val_ppl": r["final_val_ppl"],
                    "rel_improve_last5_mean": float(np.nanmean(rel_last5)) if rel_last5 else None,
                    "grad_norm_final": r.get("grad_norm_history", [None])[-1],
                    "grad_norm_mean_last10": (
                        float(np.mean(r["grad_norm_history"][-10:]))
                        if r.get("grad_norm_history") and len(r["grad_norm_history"]) >= 10
                        else None
                    ),
                }
                k_results.append(entry)
                all_runs.append(r)

            plot_convergence_diagnostics(runs_this, run_dir / "convergence_diagnostics.png")

        results_by_k[str(k)] = k_results

        # per-k loss curves
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for r in [x for x in all_runs if any(e["k"] == k and e["model"] == x["model"] for e in k_results)]:
            ep = np.arange(1, len(r["loss_history"]) + 1)
            axes[0].plot(ep, r["loss_history"], label=r["model"])
            axes[1].plot(ep, r["val_ppl_history"], label=r["model"])
        axes[0].set_title(f"k={k} train loss")
        axes[0].set_xlabel("epoch")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[1].set_title(f"k={k} val perplexity")
        axes[1].set_xlabel("epoch")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(k_dir / "curves_k.png", dpi=200)
        plt.close(fig)

    # convergence status table
    status = []
    for k, entries in results_by_k.items():
        for e in entries:
            converged = e["stopped_early"] or (
                e["rel_improve_last5_mean"] is not None and e["rel_improve_last5_mean"] < args.loss_rel_tol
            )
            needs_more = (
                not converged
                or (e["epochs_run"] >= args.max_epochs - 1)
                or (e["rel_improve_last5_mean"] or 0) > 5 * args.loss_rel_tol
            )
            status.append({**e, "converged": converged, "needs_more_epochs_or_lr_tweak": needs_more})

    summary = {
        "phase": "convergence_debug_single_seed",
        "config": {
            "T": args.T, "d": args.d, "ks": ks, "layers": args.layers,
            "max_epochs": args.max_epochs, "learning_rates": lr_list,
            "data_seed": args.data_seed, "model_seed": args.model_seed,
            "early_stop": args.early_stop,
            "loss_rel_tol": args.loss_rel_tol,
            "convergence_patience": args.convergence_patience,
            "output_dir": str(out),
        },
        "per_k_status": status,
        "recommendations": _recommendations(status, args.max_epochs),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # summary plot: final val ppl vs k
    fig, ax = plt.subplots(figsize=(8, 5))
    for model in sorted({s["model"] for s in status}):
        rows = sorted([s for s in status if s["model"] == model], key=lambda x: x["k"])
        ax.plot([r["k"] for r in rows], [r["final_val_ppl"] for r in rows], marker="o", label=model)
    ax.set_xlabel("k")
    ax.set_ylabel("final val perplexity")
    ax.set_title("Convergence debug: val ppl vs k (single seed)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "val_ppl_vs_k.png", dpi=200)
    plt.close(fig)

    print(f"\nWrote {out / 'summary.json'}")
    print(f"Plot: {out / 'val_ppl_vs_k.png'}")
    print("\nCONVERGENCE STATUS:")
    for s in status:
        flag = "NEEDS MORE" if s["needs_more_epochs_or_lr_tweak"] else "OK"
        print(
            f"  k={s['k']} {s['model']:6} ep={s['epochs_run']:3d} "
            f"loss={s['final_loss']:.4f} val_ppl={s['final_val_ppl']:.4f} "
            f"rel_impr5={s['rel_improve_last5_mean']:.2e} [{flag}]"
        )
    return 0


def _recommendations(status: list, max_epochs: int) -> list[str]:
    recs = []
    by_k = {}
    for s in status:
        by_k.setdefault(s["k"], []).append(s)
    for k, entries in sorted(by_k.items()):
        needs = [e for e in entries if e["needs_more_epochs_or_lr_tweak"]]
        if not needs:
            recs.append(f"k={k}: converged within {max_epochs} epochs at lr={entries[0]['lr']}.")
        else:
            ep_hit = any(e["epochs_run"] >= max_epochs - 1 for e in needs)
            if ep_hit:
                recs.append(f"k={k}: increase max_epochs (>={max_epochs * 2}) or try lower LR.")
            else:
                recs.append(f"k={k}: still improving at stop; extend patience or epochs.")
    recs.append("Final runs: n_seeds>=5, report mean±std on val_ppl; nl-CSA once (k-independent).")
    return recs


if __name__ == "__main__":
    raise SystemExit(main())
