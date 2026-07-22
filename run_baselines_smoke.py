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
    prepare_data_bundle,
    plot_training_curves,
    qsa_angle_param_count,
    kcsa_matrix_param_count,
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
    p.add_argument("--nl-embedding", choices=("general", "isometric"), default="isometric")
    p.add_argument("--nl-loss", choices=("ce", "renyi"), default="renyi")
    p.add_argument("--data-seed", type=int, default=42, help="dataset / encoding seed")
    p.add_argument("--model-seed-base", type=int, default=1042, help="base model init seed")
    p.add_argument("--seed", type=int, default=None, help="legacy alias for data-seed")
    p.add_argument("--n-seeds", type=int, default=3, help="number of seeds for mean±std")
    p.add_argument("--batch-size", type=int, default=64, help="0 = full batch")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--mpi", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--kernel-mode", choices=("poly", "monomial"), default="monomial")
    p.add_argument("--skip-nl", action="store_true", help="skip nl-CSA (k-independent)")
    args = p.parse_args()

    if args.quick:
        args.T, args.d, args.epochs, args.max_sentences = 4, 4, 40, 64
        args.n_seeds = 1
        args.batch_size = 16

    data_seed = args.data_seed if args.seed is None else args.seed

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
    model_seeds = [args.model_seed_base + i for i in range(args.n_seeds)]
    my_seeds = shard_items(model_seeds, rank, size) if args.mpi else model_seeds

    if rank == 0:
        print("=" * 60)
        print("BASELINES — multi-seed mean ± std")
        print(
            f"T={args.T} d={args.d} k={args.k} L={args.layers} "
            f"epochs={args.epochs} frasi={args.max_sentences}"
        )
        print(f"data_seed={data_seed} model_seeds={model_seeds} batch_size={batch_size or 'full'}")
        print(f"k-QSA angles={qsa_angle_param_count(args.d, args.layers)}")
        print(f"k-CSA matrices={kcsa_matrix_param_count(args.d)}")
        print(f"kernel_mode={args.kernel_mode}")
        print(f"Output: {out}")
        print("=" * 60)

    runs_qsa, runs_csa, runs_nl = [], [], []
    qsa_csa_gaps = []

    for ms in my_seeds:
        cfg = BaselineConfig(
            T=args.T,
            d=args.d,
            k=args.k,
            layers=args.layers,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_sentences=args.max_sentences,
            seed=data_seed,
            data_seed=data_seed,
            model_seed=ms,
            output_dir=str(out),
            run_label=f"model_seed{ms}",
            nl_rank=args.nl_rank,
            nl_learning_rate=args.nl_learning_rate,
            nl_embedding_mode=args.nl_embedding,
            nl_loss_mode=args.nl_loss,
            batch_size=batch_size,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
            kernel_mode=args.kernel_mode,
        )
        print(f"\n===== model_seed {ms} (rank {rank}/{size}) =====")
        bundle = prepare_data_bundle(cfg)
        print(f"vocab={bundle['encoding'].vocabSize} sentences={len(bundle['sentences'])}")

        qsa = train_kqsa(cfg, bundle, logger=log)
        runs_qsa.append(qsa)
        csa = train_kcsa(cfg, bundle, logger=log)
        runs_csa.append(csa)

        gap = max(abs(a - b) for a, b in zip(qsa["loss_history"], csa["loss_history"]))
        qsa_csa_gaps.append(gap)
        print(f"[CHECK model_seed={ms}] max |QSA-CSA| train = {gap:.3e} (expect >0, models independent)")

        if not args.skip_nl:
            runs_nl.append(train_nlcsa(cfg, bundle, logger=log))

    barrier(comm)
    all_qsa = gather_list(comm, runs_qsa) if args.mpi else runs_qsa
    all_csa = gather_list(comm, runs_csa) if args.mpi else runs_csa
    all_nl = gather_list(comm, runs_nl) if args.mpi else runs_nl
    all_gaps = gather_list(comm, qsa_csa_gaps) if args.mpi else qsa_csa_gaps

    if rank != 0:
        barrier(comm)
        return 0

    all_qsa = sorted(all_qsa or [], key=lambda r: int(r.get("model_seed", 0)))
    all_csa = sorted(all_csa or [], key=lambda r: int(r.get("model_seed", 0)))
    all_nl = sorted(all_nl or [], key=lambda r: int(r.get("model_seed", 0)))

    agg = [
        aggregate_seed_results(all_qsa),
        aggregate_seed_results(all_csa),
    ]
    if all_nl:
        agg.append(aggregate_seed_results(all_nl))
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
            "data_seed": data_seed,
            "model_seeds": model_seeds,
            "n_seeds": args.n_seeds,
            "output_dir": str(out),
        },
        "parametrization": BaselineConfig(T=args.T, d=args.d).parametrization,
        "qsa_csa_independence": {
            "max_train_gap_per_seed": all_gaps,
            "note": "k-QSA and k-CSA are independent; non-zero gap expected.",
        },
        "models": [
            {
                "model": a["model"],
                "n_seeds": a["n_seeds"],
                "final_loss_mean": a["final_loss_mean"],
                "final_loss_std": a["final_loss_std"],
                "final_val_ppl_mean": a.get("final_val_ppl_mean"),
                "final_val_ppl_std": a.get("final_val_ppl_std"),
                "n_params_angles": a.get("n_params_angles"),
                "n_params_matrices": a.get("n_params_matrices"),
                "n_params_model": a.get("n_params_model"),
                "nl_rank": a.get("nl_rank"),
                "ablation": a.get("ablation"),
            }
            for a in agg
        ],
        "plot_note": (
            "Train loss is model-specific; compare architectures via val_ppl (common metric). "
            "nl-CSA is k-independent — run once per (T,d), not per k."
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
