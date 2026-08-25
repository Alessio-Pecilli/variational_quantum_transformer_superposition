#!/usr/bin/env python3
"""μ vs T / μ vs d sweeps on qsa_bench_25_08 (hybrid circuit μ + trainable emb).

Axes match final_campaign_v2 screens:
  μ vs T  (d=16, T ∈ {2,4,8,16,32}), k=2 and 5, mono+poly + advantage
  μ vs d  (T=32, d ∈ {2,4,8,16,32}), k=2 and 5, mono+poly + advantage

μ is the circuit observable from observables() (with √α), not exp(−L_B).
Advantage = k / C(d+k−1, k). Classical PTB only; jointly trained embedding.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import qsa_bench_25_08 as qb
from run_qsa_bench_2508_campaign import (
    generate_ptb_indices,
    make_model,
    train_adam_tokens,
)

MU_COLORS = {2: "#0072B2", 5: "#D55E00"}
K_MODELS = [
    ("kqsa-mono", "mono", "L_B"),
    ("kqsa-poly", "poly", "L_B"),
    ("kcsa-mono", "mono", "L_B"),
    ("kcsa-poly", "poly", "L_B"),
]


def _agg_mu(runs: list[dict[str, Any]]) -> tuple[float, float]:
    vals = np.asarray([r["train_mu"] for r in runs], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=0))


def _cell_path(out_dir: Path, T: int, d: int, k: int, name: str) -> Path:
    safe = name.replace("-", "_")
    return out_dir / "cells" / f"T{T}_d{d}_k{k}_{safe}.json"


def _load_cell(path: Path, n_seeds: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if int(data.get("n_seeds", 0)) < n_seeds:
        return None
    return data


def train_cell(
    T: int,
    d: int,
    k: int,
    name: str,
    kernel: str,
    loss: str,
    *,
    layers: int,
    train_size: int,
    test_size: int,
    n_seeds: int,
    model_seed_base: int,
    data_seed: int,
    max_epochs: int,
    lr: float,
    batch_size: int,
    min_epochs: int,
    patience: int,
    loss_rel_tol: float,
    eval_every: int,
    out_dir: Path,
) -> dict[str, Any]:
    path = _cell_path(out_dir, T, d, k, name)
    cached = _load_cell(path, n_seeds)
    if cached is not None:
        print(f"  resume T={T} d={d} k={k} {name}", flush=True)
        return cached

    # Fresh PTB split per (T,d) so vocab matches sentence length / dim.
    idx_tr, idx_te, vocab_D, _ = generate_ptb_indices(
        train_size, test_size, d, T, data_seed + 17 * T + 31 * d,
    )
    # Param-match layers for this d (QSA); CSA ignores layers for n_params.
    if name.startswith("kqsa"):
        L = layers
        if L <= 0:
            L, _ = qb.qsa_layers_matching_csa(d)
    else:
        L = max(layers, 1) if layers > 0 else qb.qsa_layers_matching_csa(d)[0]

    runs = []
    for s in range(n_seeds):
        seed = model_seed_base + s
        print(f"  >> T={T} d={d} k={k} {name} seed={seed} D={vocab_D} L={L}", flush=True)
        model = make_model(name, kernel, d, k, L, loss)
        ep = max_epochs
        if "poly" in name:
            ep = int(round(max_epochs * 1.5))
        runs.append(train_adam_tokens(
            model, idx_tr, idx_te, vocab_D,
            max_epochs=ep, lr=lr, seed=seed, batch_size=batch_size,
            min_epochs=min_epochs, patience=patience,
            loss_rel_tol=loss_rel_tol, eval_every=eval_every, verbose=True,
        ))

    mu_mean, mu_std = _agg_mu(runs)
    adv = qb.mu_advantage(d, k)
    row = {
        "T": T,
        "d": d,
        "k": k,
        "model": name,
        "kernel_mode": "monomial" if kernel == "mono" else "poly",
        "layers": L,
        "vocab_D": vocab_D,
        "n_seeds": len(runs),
        "n_params": int(runs[0]["n_params"]),
        "mu_mean": mu_mean,
        "mu_std": mu_std,
        "mu_advantage": adv,
        "margin_mean": mu_mean / adv if adv > 0 else float("nan"),
        "train_LB_excess_mean": float(np.mean([r["train_LB_excess"] for r in runs])),
        "train_L1_excess_mean": float(np.mean([r["train_L1_excess"] for r in runs])),
        "runs": runs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def plot_mu_panels(
    agg_rows: list[dict[str, Any]], out_root: Path, max_T: int, fixed_d: int = 16,
) -> None:
    mu_ks = sorted({int(r["k"]) for r in agg_rows}) or [2, 5]
    # ---- μ vs T (fixed_d) ----
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    n_labelled = 0
    for k in mu_ks:
        color = MU_COLORS.get(k, "#333333")
        pts_mono = sorted(
            [r for r in agg_rows
             if int(r["d"]) == int(fixed_d) and int(r["k"]) == k
             and r.get("kernel_mode") == "monomial"],
            key=lambda r: int(r["T"]),
        )
        pts_poly = sorted(
            [r for r in agg_rows
             if int(r["d"]) == int(fixed_d) and int(r["k"]) == k
             and r.get("kernel_mode") == "poly"],
            key=lambda r: int(r["T"]),
        )
        if pts_mono:
            Ts = [int(r["T"]) for r in pts_mono]
            ys = [float(r["mu_mean"]) for r in pts_mono]
            es = [float(r["mu_std"]) for r in pts_mono]
            ax.errorbar(Ts, ys, yerr=es, color=color, marker="o", linestyle="-",
                        linewidth=2.2, capsize=4, label=f"k={k} mono")
            adv = float(pts_mono[0]["mu_advantage"])
            ax.axhline(adv, color=color, linestyle=":", linewidth=1.8, alpha=0.95,
                       label=f"k={k} advantage")
            n_labelled += 1
        if pts_poly:
            Ts = [int(r["T"]) for r in pts_poly]
            ys = [float(r["mu_mean"]) for r in pts_poly]
            es = [float(r["mu_std"]) for r in pts_poly]
            ax.errorbar(Ts, ys, yerr=es, color=color, marker="s", linestyle="--",
                        linewidth=2.0, capsize=4, label=f"k={k} poly")
            n_labelled += 1
    ax.set_yscale("log")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\mu$  (mean $\pm$ std over seeds, log scale)")
    ax.set_title(rf"$\mu$ vs $T$  (d={fixed_d}, T\leq{max_T})")
    ax.grid(True, alpha=0.3, which="both")
    if n_labelled:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_root / "mu_vs_T.png", dpi=220)
    plt.close(fig)

    # ---- μ vs d (T=max_T) ----
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    n_labelled = 0
    for k in mu_ks:
        color = MU_COLORS.get(k, "#333333")
        pts_mono = sorted(
            [r for r in agg_rows
             if int(r["T"]) == int(max_T) and int(r["k"]) == k
             and r.get("kernel_mode") == "monomial"],
            key=lambda r: int(r["d"]),
        )
        pts_poly = sorted(
            [r for r in agg_rows
             if int(r["T"]) == int(max_T) and int(r["k"]) == k
             and r.get("kernel_mode") == "poly"],
            key=lambda r: int(r["d"]),
        )
        if pts_mono:
            ds = [int(r["d"]) for r in pts_mono]
            ys = [float(r["mu_mean"]) for r in pts_mono]
            es = [float(r["mu_std"]) for r in pts_mono]
            ax.errorbar(ds, ys, yerr=es, color=color, marker="o", linestyle="-",
                        linewidth=2.2, capsize=4, label=f"k={k} mono")
            adv = [float(r["mu_advantage"]) for r in pts_mono]
            ax.plot(ds, adv, color=color, linestyle=":", linewidth=1.8, marker="^",
                    markersize=4, label=f"k={k} advantage")
            n_labelled += 1
        if pts_poly:
            ds = [int(r["d"]) for r in pts_poly]
            ys = [float(r["mu_mean"]) for r in pts_poly]
            es = [float(r["mu_std"]) for r in pts_poly]
            ax.errorbar(ds, ys, yerr=es, color=color, marker="s", linestyle="--",
                        linewidth=2.0, capsize=4, label=f"k={k} poly")
            n_labelled += 1
    ax.set_yscale("log")
    ax.set_xlabel("d")
    ax.set_ylabel(r"$\mu$  (mean $\pm$ std over seeds, log scale)")
    ax.set_title(rf"$\mu$ vs $d$  (T={max_T})")
    ax.grid(True, alpha=0.3, which="both")
    if n_labelled:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_root / "mu_vs_d.png", dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="μ vs T / μ vs d on qsa_bench_25_08")
    p.add_argument("--Ts", type=str, default="2,4,8,16,32")
    p.add_argument("--ds", type=str, default="2,4,8,16,32")
    p.add_argument("--fixed-d", type=int, default=16, help="d for μ vs T panel")
    p.add_argument("--fixed-T", type=int, default=32, help="T for μ vs d panel")
    p.add_argument("--ks", type=str, default="2,5")
    p.add_argument("--layers", type=int, default=0, help="0 = param-match CSA")
    p.add_argument("--train-size", type=int, default=64)
    p.add_argument("--test-size", type=int, default=32)
    p.add_argument("--n-seeds", type=int, default=10)
    p.add_argument("--model-seed-base", type=int, default=42)
    p.add_argument("--data-seed", type=int, default=7)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--loss-rel-tol", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--output-dir", type=str,
                   default="results/qsa_bench_2508/mu_sweep")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--replot-only", action="store_true",
                   help="Rebuild plots from existing cells/ JSONs")
    args = p.parse_args(argv)

    if args.quick:
        args.Ts = "4,8"
        args.ds = "4,8"
        args.fixed_d = 8
        args.fixed_T = 8
        args.ks = "2"
        args.layers = 2
        args.train_size = 8
        args.test_size = 4
        args.n_seeds = 2
        args.epochs = 20
        args.min_epochs = 5
        args.patience = 4
        args.batch_size = 4
        args.eval_every = 5
        args.output_dir = str(Path(args.output_dir).parent / "mu_quick_smoke")

    Ts = [int(x) for x in args.Ts.split(",") if x.strip()]
    ds = [int(x) for x in args.ds.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not qb.HAS_JAX:
        raise SystemExit("JAX required")

    # Grid cells: all (T,d=fixed_d) for vs-T + all (T=fixed_T,d) for vs-d
    jobs: list[tuple[int, int]] = []
    for T in Ts:
        jobs.append((T, args.fixed_d))
    for d in ds:
        jobs.append((args.fixed_T, d))
    # unique
    seen = set()
    uniq_jobs = []
    for T, d in jobs:
        if (T, d) in seen:
            continue
        if d & (d - 1):
            raise ValueError(f"d must be power of two, got {d}")
        seen.add((T, d))
        uniq_jobs.append((T, d))

    agg_rows: list[dict[str, Any]] = []
    if args.replot_only:
        for path in sorted((out / "cells").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            agg_rows.append({k: v for k, v in data.items() if k != "runs"})
    else:
        for T, d in uniq_jobs:
            for k in ks:
                for name, kernel, loss in K_MODELS:
                    row = train_cell(
                        T, d, k, name, kernel, loss,
                        layers=args.layers,
                        train_size=args.train_size,
                        test_size=args.test_size,
                        n_seeds=args.n_seeds,
                        model_seed_base=args.model_seed_base,
                        data_seed=args.data_seed,
                        max_epochs=args.epochs,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        min_epochs=args.min_epochs,
                        patience=args.patience,
                        loss_rel_tol=args.loss_rel_tol,
                        eval_every=args.eval_every,
                        out_dir=out,
                    )
                    agg_rows.append({kk: vv for kk, vv in row.items() if kk != "runs"})

    plot_mu_panels(agg_rows, out, max_T=args.fixed_T, fixed_d=args.fixed_d)
    summary = {
        "config": vars(args),
        "n_cells": len(agg_rows),
        "rows": agg_rows,
        "plots": {
            "mu_vs_T": str(out / "mu_vs_T.png"),
            "mu_vs_d": str(out / "mu_vs_d.png"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'mu_vs_T.png'} and {out / 'mu_vs_d.png'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
