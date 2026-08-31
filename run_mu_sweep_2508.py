#!/usr/bin/env python3
"""μ vs T / μ vs d sweeps on qsa_bench_25_08 (hybrid circuit μ + trainable emb).

Axes match final_campaign_v2 screens:
  μ vs T  (d=16, T ∈ {2,4,8,16,32}), k ∈ {2,5,7}, mono+poly + advantage
  μ vs d  (T=32, d ∈ {2,4,8,16,32}), k ∈ {2,5,7}, mono+poly + advantage

μ is the circuit observable from observables() (with √α), not exp(−L_B).
Advantage = k² log(d) / C(d+k−1, k). Classical PTB only; jointly trained embedding.

Each (k, kernel) panel plots separate QSA and CSA curves (no double points per x).
"""
from __future__ import annotations

import argparse
import json
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

MU_COLORS = {2: "#0072B2", 5: "#D55E00", 7: "#009E73"}
K_MODELS = [
    ("kqsa-mono", "mono", "L_B"),
    ("kqsa-poly", "poly", "L_B"),
    ("kcsa-mono", "mono", "L_B"),
    ("kcsa-poly", "poly", "L_B"),
]
FAMILIES = (
    ("kqsa", "QSA", "-"),
    ("kcsa", "CSA", "--"),
)


def _mu_key(mu_at: str) -> str:
    return "train_mu_final" if mu_at == "final" else "train_mu"


def _agg_mu(runs: list[dict[str, Any]], mu_key: str) -> tuple[float, float]:
    vals = np.asarray(
        [float(r.get(mu_key, r["train_mu"])) for r in runs],
        dtype=float,
    )
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


def _row_summary(row: dict[str, Any], mu_key: str) -> dict[str, Any]:
    d, k = int(row["d"]), int(row["k"])
    adv = qb.mu_advantage(d, k)
    mu_mean = float(row.get(f"{mu_key}_mean", row["mu_mean"]))
    return {
        **{kk: vv for kk, vv in row.items() if kk != "runs"},
        "mu_mean": mu_mean,
        "mu_advantage": adv,
        "margin_mean": mu_mean / adv if adv > 0 else float("nan"),
        "mu_key": mu_key,
    }


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
    mu_at: str,
    out_dir: Path,
) -> dict[str, Any]:
    path = _cell_path(out_dir, T, d, k, name)
    mu_key = _mu_key(mu_at)
    cached = _load_cell(path, n_seeds)
    if cached is not None and cached.get("mu_key") == mu_key:
        print(f"  resume T={T} d={d} k={k} {name}", flush=True)
        return _row_summary(cached, mu_key)

    idx_tr, idx_te, vocab_D, _ = generate_ptb_indices(
        train_size, test_size, d, T, data_seed + 17 * T + 31 * d,
    )
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

    mu_mean, mu_std = _agg_mu(runs, mu_key)
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
        "mu_key": mu_key,
        "runs": runs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return _row_summary(row, mu_key)


def _family_rows(
    agg_rows: list[dict[str, Any]],
    *,
    family: str,
    k: int,
    kernel_mode: str,
    T: int | None = None,
    d: int | None = None,
    fixed_d: int | None = None,
    fixed_T: int | None = None,
) -> list[dict[str, Any]]:
    out = []
    for r in agg_rows:
        if int(r["k"]) != int(k):
            continue
        if r.get("kernel_mode") != kernel_mode:
            continue
        if not str(r.get("model", "")).startswith(family):
            continue
        if fixed_d is not None and int(r["d"]) != int(fixed_d):
            continue
        if fixed_T is not None and int(r["T"]) != int(fixed_T):
            continue
        out.append(r)
    if T is not None:
        return sorted(out, key=lambda r: int(r["T"]))
    return sorted(out, key=lambda r: int(r["d"]))


def plot_mu_panels(
    agg_rows: list[dict[str, Any]], out_root: Path, max_T: int, fixed_d: int = 16,
) -> None:
    mu_ks = sorted({int(r["k"]) for r in agg_rows}) or [2, 5, 7]

    # ---- μ vs T (fixed_d) ----
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    n_labelled = 0
    for k in mu_ks:
        color = MU_COLORS.get(k, "#333333")
        adv = qb.mu_advantage(fixed_d, k)
        ax.axhline(adv, color=color, linestyle=":", linewidth=1.6, alpha=0.9,
                   label=f"k={k} advantage")
        for family, fam_label, fam_ls in FAMILIES:
            for kernel_mode, marker, kern_ls in (
                ("monomial", "o", "-"),
                ("poly", "s", "--"),
            ):
                pts = _family_rows(
                    agg_rows, family=family, k=k, kernel_mode=kernel_mode,
                    fixed_d=fixed_d,
                )
                if not pts:
                    continue
                xs = [int(r["T"]) for r in pts]
                ys = [float(r["mu_mean"]) for r in pts]
                es = [float(r["mu_std"]) for r in pts]
                ax.errorbar(
                    xs, ys, yerr=es, color=color, marker=marker,
                    linestyle=kern_ls, linewidth=1.8, capsize=3,
                    label=f"k={k} {fam_label} {kernel_mode[:4]}",
                )
                n_labelled += 1
    ax.set_yscale("log")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\mu$ at end of training (mean $\pm$ std, log scale)")
    ax.set_title(rf"$\mu$ vs $T$  (d={fixed_d}, QSA/CSA separate)")
    ax.grid(True, alpha=0.3, which="both")
    if n_labelled:
        ax.legend(fontsize=6, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_root / "mu_vs_T.png", dpi=220)
    plt.close(fig)

    # ---- μ vs d (T=max_T) ----
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    n_labelled = 0
    for k in mu_ks:
        color = MU_COLORS.get(k, "#333333")
        for family, fam_label, _fam_ls in FAMILIES:
            for kernel_mode, marker, kern_ls in (
                ("monomial", "o", "-"),
                ("poly", "s", "--"),
            ):
                pts = _family_rows(
                    agg_rows, family=family, k=k, kernel_mode=kernel_mode,
                    fixed_T=max_T,
                )
                if not pts:
                    continue
                ds = [int(r["d"]) for r in pts]
                ys = [float(r["mu_mean"]) for r in pts]
                es = [float(r["mu_std"]) for r in pts]
                ax.errorbar(
                    ds, ys, yerr=es, color=color, marker=marker,
                    linestyle=kern_ls, linewidth=1.8, capsize=3,
                    label=f"k={k} {fam_label} {kernel_mode[:4]}",
                )
                n_labelled += 1
        ref_ds = sorted({int(r["d"]) for r in agg_rows if int(r["T"]) == int(max_T)})
        if ref_ds:
            adv = [qb.mu_advantage(int(dv), k) for dv in ref_ds]
            ax.plot(ref_ds, adv, color=color, linestyle=":", linewidth=1.6,
                    marker="^", markersize=4, label=f"k={k} advantage")
    ax.set_yscale("log")
    ax.set_xlabel("d")
    ax.set_ylabel(r"$\mu$ at end of training (mean $\pm$ std, log scale)")
    ax.set_title(rf"$\mu$ vs $d$  (T={max_T}, QSA/CSA separate)")
    ax.grid(True, alpha=0.3, which="both")
    if n_labelled:
        ax.legend(fontsize=6, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_root / "mu_vs_d.png", dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="μ vs T / μ vs d on qsa_bench_25_08")
    p.add_argument("--Ts", type=str, default="2,4,8,16,32")
    p.add_argument("--ds", type=str, default="2,4,8,16,32")
    p.add_argument("--fixed-d", type=int, default=16, help="d for μ vs T panel")
    p.add_argument("--fixed-T", type=int, default=32, help="T for μ vs d panel")
    p.add_argument("--ks", type=str, default="2,5,7")
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
    p.add_argument("--mu-at", choices=("final", "best"), default="final",
                   help="Use μ from last epoch (final) or best-checkpoint (best)")
    p.add_argument("--output-dir", type=str,
                   default="results/qsa_bench_2508/mu_sweep")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--replot-only", action="store_true",
                   help="Rebuild plots from existing cells/ JSONs")
    p.add_argument("--force-retrain", action="store_true",
                   help="Ignore cached cells (retrain all)")
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
    mu_key = _mu_key(args.mu_at)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not qb.HAS_JAX:
        raise SystemExit("JAX required")

    jobs: list[tuple[int, int]] = []
    for T in Ts:
        jobs.append((T, args.fixed_d))
    for d in ds:
        jobs.append((args.fixed_T, d))
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
            if "runs" in data:
                mu_mean, mu_std = _agg_mu(data["runs"], mu_key)
                data["mu_mean"] = mu_mean
                data["mu_std"] = mu_std
            agg_rows.append(_row_summary(data, mu_key))
    else:
        for T, d in uniq_jobs:
            for k in ks:
                for name, kernel, loss in K_MODELS:
                    if args.force_retrain:
                        cp = _cell_path(out, T, d, k, name)
                        if cp.is_file():
                            cp.unlink()
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
                        mu_at=args.mu_at,
                        out_dir=out,
                    )
                    agg_rows.append(row)

    plot_mu_panels(agg_rows, out, max_T=args.fixed_T, fixed_d=args.fixed_d)
    summary = {
        "config": vars(args),
        "mu_key": mu_key,
        "advantage_formula": "k^2 * log(d) / C(d+k-1,k)",
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
