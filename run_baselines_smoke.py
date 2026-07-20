#!/usr/bin/env python3
"""Training curves with multi-seed mean ± envelope (and optional minibatch).

MPI: shard seeds across ranks (srun --mpi=pmix_v3). Rank 0 aggregates + plots.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from classical_baselines import (
    BaselineConfig,
    aggregate_seed_results,
    prepare_shared_bundle,
    plot_training_curves,
    qsa_angle_param_count,
    train_kcsa,
    train_kqsa,
    train_nlcsa,
)
from mpi_runtime import barrier, gather_list, init_mpi, shard_items


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
    p.add_argument("--seed", type=int, default=42, help="base seed")
    p.add_argument("--n-seeds", type=int, default=3, help="number of seeds for mean±std")
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="minibatch size (0 = full batch, smoother curves)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="fixed output dir (required for resume across jobs; default: timestamped)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="save checkpoint every N epochs (0 disables mid-run checkpoints)",
    )
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="skip completed runs / continue from checkpoint (default)",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="ignore existing checkpoints and retrain",
    )
    p.add_argument(
        "--mpi",
        action="store_true",
        help="shard seeds across MPI ranks (use with srun)",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--kernel-mode",
        choices=("poly", "monomial"),
        default="monomial",
        help="attention kernel: monomial = s^k (default); poly = LCU (abandoned)",
    )
    args = p.parse_args()

    if args.quick:
        args.T, args.d, args.epochs, args.max_sentences = 4, 4, 40, 64
        args.n_seeds = 3
        args.batch_size = 16

    comm, rank, size = init_mpi(enabled=args.mpi)
    logging.basicConfig(level=logging.INFO, format=f"[rank{rank}] %(message)s")
    log = logging.getLogger("baselines")

    if args.output_dir:
        out = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("results") / "baselines_smoke" / stamp
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    barrier(comm)

    batch_size = None if args.batch_size <= 0 else args.batch_size
    seeds = [args.seed + i for i in range(args.n_seeds)]
    my_seeds = shard_items(seeds, rank, size) if args.mpi else seeds
    budget = qsa_angle_param_count(args.d, args.layers)

    if rank == 0:
        print("=" * 60)
        print("BASELINES — multi-seed mean ± std")
        print(
            f"T={args.T} d={args.d} k={args.k} L={args.layers} "
            f"epochs={args.epochs} frasi={args.max_sentences}"
        )
        print(f"seeds={seeds} batch_size={batch_size or 'full'} angle_budget={budget}")
        print(f"kernel_mode={args.kernel_mode}")
        print(f"resume={args.resume} checkpoint_every={args.checkpoint_every}")
        print(f"MPI: enabled={args.mpi} ranks={size}")
        print(f"Output: {out}")
        print("=" * 60)

    runs_qsa, runs_csa, runs_nl = [], [], []
    identity_gap = None

    for s in my_seeds:
        cfg = BaselineConfig(
            T=args.T,
            d=args.d,
            k=args.k,
            layers=args.layers,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_sentences=args.max_sentences,
            seed=s,
            output_dir=str(out),
            run_label=f"seed{s}",
            nl_rank=args.nl_rank,
            nl_learning_rate=args.nl_learning_rate,
            batch_size=batch_size,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            kernel_mode=args.kernel_mode,
        )
        print(f"\n===== seed {s} (rank {rank}/{size}) =====")
        bundle = prepare_shared_bundle(cfg)
        print(f"vocab={bundle['encoding'].vocabSize} sentences={len(bundle['sentences'])}")

        print("--- k-QSA ---")
        qsa = train_kqsa(cfg, bundle, logger=log)
        runs_qsa.append(qsa)

        print("--- k-CSA ---")
        csa = train_kcsa(cfg, bundle, logger=log)
        runs_csa.append(csa)

        gap = max(abs(a - b) for a, b in zip(qsa["loss_history"], csa["loss_history"]))
        print(f"[CHECK seed={s}] max |QSA-CSA| = {gap:.3e}")
        if identity_gap is None:
            identity_gap = gap

        print("--- nl-CSA ---")
        runs_nl.append(train_nlcsa(cfg, bundle, logger=log))

    barrier(comm)
    all_qsa = gather_list(comm, runs_qsa) if args.mpi else runs_qsa
    all_csa = gather_list(comm, runs_csa) if args.mpi else runs_csa
    all_nl = gather_list(comm, runs_nl) if args.mpi else runs_nl
    gaps = gather_list(comm, [identity_gap] if identity_gap is not None else []) if args.mpi else (
        [identity_gap] if identity_gap is not None else []
    )

    if rank != 0:
        barrier(comm)
        return 0

    # restore seed order for stable plots
    all_qsa = sorted(all_qsa or [], key=lambda r: int(r.get("seed", 0)))
    all_csa = sorted(all_csa or [], key=lambda r: int(r.get("seed", 0)))
    all_nl = sorted(all_nl or [], key=lambda r: int(r.get("seed", 0)))
    identity_gap = next((g for g in (gaps or []) if g is not None), None)

    agg = [
        aggregate_seed_results(all_qsa),
        aggregate_seed_results(all_csa),
        aggregate_seed_results(all_nl),
    ]
    plot_training_curves(agg, out / "training_curves.png", show_seed_traces=True)

    summary = {
        "config": {
            "T": args.T,
            "d": args.d,
            "k": args.k,
            "layers": args.layers,
            "epochs": args.epochs,
            "max_sentences": args.max_sentences,
            "batch_size": batch_size,
            "seeds": seeds,
            "n_seeds": args.n_seeds,
            "angle_budget": budget,
            "resume": args.resume,
            "checkpoint_every": args.checkpoint_every,
            "output_dir": str(out),
            "mpi_ranks": size,
        },
        "qsa_csa_max_epoch_gap_first_seed": identity_gap,
        "models": [
            {
                "model": a["model"],
                "n_seeds": a["n_seeds"],
                "final_loss_mean": a["final_loss_mean"],
                "final_loss_std": a["final_loss_std"],
                "n_params_angles": a.get("n_params_angles"),
                "n_params_model": a.get("n_params_model"),
                "nl_rank": a.get("nl_rank"),
                "batch_size": a.get("batch_size"),
            }
            for a in agg
        ],
        "plot_note": (
            "Curves show mean over seeds; shaded band is ±1 std; "
            "faint lines are individual seeds. "
            "Logged metric is full-batch eval loss each epoch "
            "(updates use minibatch when batch_size is set)."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2))
    print(f"\nPlot: {out / 'training_curves.png'}")
    barrier(comm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
