#!/usr/bin/env python3
"""FINAL multi-seed loss campaign (Section 2 baselines).

Models / styles (professor brief):
  k-QSA L=16              blue solid
  k-CSA matrices=128      orange solid
  poly-k-QSA L=16         blue dashed
  poly-k-CSA              orange dashed
  nl-CSA iso ~288         black dashed
  nl-CSA iso ~128         black solid
  nl-CSA gen ~128         black dotted  (expect best if converged)

All metrics / histories saved under OUTPUT_DIR for aesthetic replotting:
  summary.json, aggregates/*.json, plots/*.png, per-seed metrics under model dirs.

MPI: shards (model, seed) jobs across ranks. Rank 0 aggregates + plots.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from classical_baselines import (
    BaselineConfig,
    aggregate_seed_results,
    kcsa_matrix_param_count,
    nl_model_param_count,
    nl_rank_for_budget,
    prepare_data_bundle,
    qsa_angle_param_count,
    train_kcsa,
    train_kqsa,
    train_nlcsa,
)
from mpi_runtime import barrier, gather_list, init_mpi, shard_items


# Display name -> plot style (FINAL aesthetic)
STYLES: dict[str, dict[str, Any]] = {
    "k-QSA L=16": dict(color="#0072B2", linestyle="-", marker="o", linewidth=2.4),
    "k-CSA": dict(color="#E69F00", linestyle="-", marker="s", linewidth=2.4),
    "poly-k-QSA L=16": dict(color="#0072B2", linestyle="--", marker="^", linewidth=2.2),
    "poly-k-CSA": dict(color="#E69F00", linestyle="--", marker="v", linewidth=2.2),
    "nl-CSA iso ~288": dict(color="#000000", linestyle="--", marker="D", linewidth=2.2),
    "nl-CSA iso ~128": dict(color="#000000", linestyle="-", marker="s", linewidth=2.2),
    "nl-CSA gen ~128": dict(color="#000000", linestyle=":", marker="x", linewidth=2.4),
}


def _log(rank: int) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format=f"[rank{rank}] %(message)s")
    return logging.getLogger("final_loss")


def _aligned_loss(result: dict, T: int) -> float:
    """Comparable axis: −logμ + log T for mu-models; Renyi/CE as-is for nl."""
    model = str(result.get("model", ""))
    final = float(result["final_loss"])
    if model.startswith("nl-CSA"):
        return final
    return final + math.log(T)


def _model_specs(args) -> list[dict]:
    """Build the 7 FINAL model specifications at fixed (T,d,k)."""
    d = args.d
    # ~288: default L=2, r≈√d → at d=8, r=3 → 288
    r288 = args.nl_rank_full if args.nl_rank_full is not None else max(2, int(round(d ** 0.5)))
    L288 = args.nl_layers_full
    # ~128 budget: choose rank at L=1 (or CLI)
    L128 = args.nl_layers_small
    r128 = (
        args.nl_rank_small
        if args.nl_rank_small is not None
        else nl_rank_for_budget(d, L128, args.nl_param_budget_small)
    )
    specs = [
        {
            "key": "kqsa_L16",
            "display": "k-QSA L=16",
            "family": "kqsa",
            "kernel_mode": "monomial",
            "layers": args.qsa_layers,
            "epochs": args.epochs,
            "lr": args.learning_rate,
        },
        {
            "key": "kcsa",
            "display": "k-CSA",
            "family": "kcsa",
            "kernel_mode": "monomial",
            "layers": 2,
            "epochs": args.epochs,
            "lr": args.learning_rate,
        },
        {
            "key": "poly_kqsa_L16",
            "display": "poly-k-QSA L=16",
            "family": "kqsa",
            "kernel_mode": "poly",
            "layers": args.qsa_layers,
            "epochs": args.poly_epochs,
            "lr": args.learning_rate,
        },
        {
            "key": "poly_kcsa",
            "display": "poly-k-CSA",
            "family": "kcsa",
            "kernel_mode": "poly",
            "layers": 2,
            "epochs": args.poly_epochs,
            "lr": args.learning_rate,
        },
        {
            "key": "nl_iso_288",
            "display": "nl-CSA iso ~288",
            "family": "nl",
            "nl_embedding_mode": "isometric",
            "nl_loss_mode": "renyi",
            "layers": L288,
            "nl_rank": r288,
            "epochs": args.nl_epochs,
            "lr": args.nl_learning_rate,
        },
        {
            "key": "nl_iso_128",
            "display": "nl-CSA iso ~128",
            "family": "nl",
            "nl_embedding_mode": "isometric",
            "nl_loss_mode": "renyi",
            "layers": L128,
            "nl_rank": r128,
            "epochs": args.nl_epochs,
            "lr": args.nl_learning_rate,
        },
        {
            "key": "nl_gen_128",
            "display": "nl-CSA gen ~128",
            "family": "nl",
            "nl_embedding_mode": "general",
            "nl_loss_mode": "renyi",
            "layers": L128,
            "nl_rank": r128,
            "epochs": args.nl_epochs_general,
            "lr": args.nl_learning_rate_general,
        },
    ]
    return specs


def _make_cfg(args, spec: dict, model_seed: int, out: Path, k: int) -> BaselineConfig:
    return BaselineConfig(
        T=args.T,
        d=args.d,
        k=int(k),
        layers=int(spec["layers"]),
        epochs=int(spec["epochs"]),
        learning_rate=float(spec.get("lr", args.learning_rate)),
        max_sentences=args.max_sentences,
        seed=args.data_seed,
        data_seed=args.data_seed,
        model_seed=model_seed,
        output_dir=str(out),
        run_label=f"{spec['key']}_k{k}_seed{model_seed}",
        nl_rank=spec.get("nl_rank"),
        nl_learning_rate=float(spec.get("lr", args.nl_learning_rate)) if spec["family"] == "nl" else None,
        nl_embedding_mode=spec.get("nl_embedding_mode", "isometric"),
        nl_loss_mode=spec.get("nl_loss_mode", "renyi"),
        batch_size=None if args.batch_size <= 0 else args.batch_size,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        kernel_mode=spec.get("kernel_mode", "monomial"),
        early_stop=False,
        max_epochs=int(spec["epochs"]),
    )


def _train_one(spec: dict, cfg: BaselineConfig, bundle: dict, log: logging.Logger) -> dict:
    if spec["family"] == "kqsa":
        result = train_kqsa(cfg, bundle, logger=log)
    elif spec["family"] == "kcsa":
        result = train_kcsa(cfg, bundle, logger=log)
    else:
        result = train_nlcsa(cfg, bundle, logger=log)
    return _enrich_result(result, spec, cfg)


def _enrich_result(result: dict, spec: dict, cfg: BaselineConfig) -> dict:
    # Enrich for FINAL reporting / replot
    result["display"] = spec["display"]
    result["spec_key"] = spec["key"]
    result["family"] = spec["family"]
    result["kernel_mode"] = cfg.kernel_mode
    result["aligned_loss"] = _aligned_loss(result, cfg.T)
    result["log_T"] = math.log(cfg.T)
    result["model"] = spec["display"]
    return result


def _clear_jax_memory() -> None:
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass
    import gc

    gc.collect()


def _run_dir_for(cfg: BaselineConfig) -> Path:
    return Path(cfg.output_dir) / str(cfg.run_label)


def _load_enriched_from_disk(spec: dict, cfg: BaselineConfig) -> dict | None:
    metrics = _run_dir_for(cfg) / "metrics.json"
    if not metrics.exists():
        return None
    result = json.loads(metrics.read_text(encoding="utf-8"))
    return _enrich_result(result, spec, cfg)


def _isolate_train_one(spec: dict, cfg: BaselineConfig, args: argparse.Namespace, log: logging.Logger) -> dict:
    """Train one job in a fresh Python subprocess so JAX/XLA memory is fully released."""
    existing = _load_enriched_from_disk(spec, cfg)
    if args.resume and existing is not None:
        log.info(f"[isolate] skip complete -> {_run_dir_for(cfg)}")
        return existing

    script = str(Path(__file__).resolve())
    cmd = [
        sys.executable,
        script,
        "--train-only",
        "--T",
        str(args.T),
        "--d",
        str(args.d),
        "--ks",
        str(cfg.k),
        "--k",
        str(cfg.k),
        "--qsa-layers",
        str(args.qsa_layers),
        "--epochs",
        str(args.epochs),
        "--poly-epochs",
        str(args.poly_epochs),
        "--nl-epochs",
        str(args.nl_epochs),
        "--nl-epochs-general",
        str(args.nl_epochs_general),
        "--max-sentences",
        str(args.max_sentences),
        "--n-seeds",
        "1",
        "--batch-size",
        str(args.batch_size),
        "--data-seed",
        str(args.data_seed),
        "--model-seed-base",
        str(cfg.model_seed),
        "--learning-rate",
        str(args.learning_rate),
        "--nl-learning-rate",
        str(args.nl_learning_rate),
        "--nl-learning-rate-general",
        str(args.nl_learning_rate_general),
        "--nl-param-budget-small",
        str(args.nl_param_budget_small),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--output-dir",
        str(cfg.output_dir),
        "--resume",
        "--models",
        spec["key"],
    ]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    env["MALLOC_ARENA_MAX"] = "2"
    log.info(f"[isolate] subprocess: {spec['key']} k={cfg.k} seed={cfg.model_seed}")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(
            f"isolated job failed rc={proc.returncode}: {spec['key']} k={cfg.k} seed={cfg.model_seed}"
        )
    result = _load_enriched_from_disk(spec, cfg)
    if result is None:
        raise RuntimeError(f"isolated job produced no metrics.json under {_run_dir_for(cfg)}")
    return result


def _pad_histories(runs: list[dict]) -> list[dict]:
    """Pad loss histories to common length (max epochs) for aggregation."""
    if not runs:
        return runs
    max_len = max(len(r.get("loss_history", [])) for r in runs)
    out = []
    for r in runs:
        rr = dict(r)
        h = list(r.get("loss_history", []))
        if len(h) < max_len and h:
            h = h + [h[-1]] * (max_len - len(h))
        rr["loss_history"] = h
        if "val_ppl_history" in r:
            vp = list(r["val_ppl_history"])
            if len(vp) < max_len and vp:
                vp = vp + [vp[-1]] * (max_len - len(vp))
            rr["val_ppl_history"] = vp
        if "wall_time_history" in r:
            wt = list(r["wall_time_history"])
            if len(wt) < max_len and wt:
                # extend wall time with last + small epsilon steps
                last = wt[-1]
                wt = wt + [last + 1e-6 * (i + 1) for i in range(max_len - len(wt))]
            rr["wall_time_history"] = wt
        out.append(rr)
    return out


def plot_final_aligned_bar(aggs: list[dict], T: int, out_path: Path, title_suffix: str = "") -> None:
    """Final aligned loss mean±std (bar) with FINAL styles."""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    names = [a["model"] for a in aggs]
    means = [float(a["aligned_loss_mean"]) for a in aggs]
    stds = [float(a["aligned_loss_std"]) for a in aggs]
    colors = [STYLES.get(n, {}).get("color", "0.4") for n in names]
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=4, edgecolor="0.2", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(r"aligned loss:  $-\log\mu+\log T$  or  Renyi")
    title = f"FINAL loss (aligned)  —  T={T}, mean±std over seeds"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title)
    ax.axhline(math.log(T), color="0.55", linestyle=":", linewidth=1.2, label=rf"$\log T$={math.log(T):.2f}")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_aligned_vs_k(
    mu_points: list[dict],
    nl_refs: list[dict],
    T: int,
    out_path: Path,
) -> None:
    """Aligned loss vs k: mono + poly on the same axes; nl as horizontal refs.

    Expect each mu_point: {model, k, aligned_loss_mean, aligned_loss_std}.
    nl_refs: aggregates with aligned_loss_mean (k-independent).
    """
    fig, ax = plt.subplots(figsize=(10, 5.8))
    by_model: dict[str, list[dict]] = {}
    for p in mu_points:
        by_model.setdefault(p["model"], []).append(p)
    for name, rows in by_model.items():
        rows = sorted(rows, key=lambda r: int(r["k"]))
        st = STYLES.get(name, dict(color="0.3", linestyle="-", marker="o", linewidth=2))
        ks = [int(r["k"]) for r in rows]
        means = [float(r["aligned_loss_mean"]) for r in rows]
        stds = [float(r.get("aligned_loss_std", 0.0)) for r in rows]
        ax.errorbar(
            ks,
            means,
            yerr=stds,
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st.get("marker", "o"),
            linewidth=st.get("linewidth", 2.2),
            markersize=7,
            capsize=4,
            label=name,
        )
    for ref in nl_refs:
        name = ref["model"]
        st = STYLES.get(name, dict(color="0.2", linestyle=":", linewidth=2))
        ax.axhline(
            float(ref["aligned_loss_mean"]),
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=st.get("linewidth", 2.0),
            label=f"{name} (k-indep.)",
        )
    ax.axhline(math.log(T), color="0.55", linestyle=":", linewidth=1.0, alpha=0.7, label=rf"$\log T$={math.log(T):.2f}")
    ax.set_xlabel("k")
    ax.set_ylabel(r"aligned loss:  $-\log\mu+\log T$  or  Renyi")
    ax.set_title(f"FINAL aligned loss vs k  (T={T}; mono+poly on same plot)")
    all_k = sorted({int(p["k"]) for p in mu_points})
    ax.set_xticks(all_k)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_final_curves(aggs: list[dict], out_path: Path, ykey: str = "aligned_history") -> None:
    """Training curves with FINAL line styles + error bands."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for a in aggs:
        name = a["model"]
        st = STYLES.get(name, dict(color="0.3", linestyle="-", marker=None, linewidth=2))
        ys = np.asarray(a.get(ykey, a["loss_history"]), dtype=float)
        xs = np.arange(1, len(ys) + 1)
        ax.plot(xs, ys, label=name, **{k: st[k] for k in ("color", "linestyle", "linewidth")})
        std_key = "aligned_std" if ykey.startswith("aligned") else "loss_std"
        if std_key in a and int(a.get("n_seeds", 1)) > 1:
            std = np.asarray(a[std_key], dtype=float)
            ax.fill_between(xs, ys - std, ys + std, color=st["color"], alpha=0.15, linewidth=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"aligned loss:  $-\log\mu+\log T$  or  Renyi")
    ax.set_title("FINAL training curves (aligned, mean±std)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_final_raw_curves(aggs: list[dict], out_path: Path) -> None:
    """Raw train loss (not aligned) — for diagnostics only."""
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for a in aggs:
        name = a["model"]
        st = STYLES.get(name, dict(color="0.3", linestyle="-", linewidth=2))
        ys = np.asarray(a["loss_history"], dtype=float)
        xs = np.arange(1, len(ys) + 1)
        ax.plot(xs, ys, label=name, **{k: st[k] for k in ("color", "linestyle", "linewidth")})
        if "loss_std" in a and int(a.get("n_seeds", 1)) > 1:
            std = np.asarray(a["loss_std"], dtype=float)
            ax.fill_between(xs, ys - std, ys + std, color=st["color"], alpha=0.15, linewidth=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel("raw train loss (model-specific)")
    ax.set_title("FINAL raw train loss (diagnostic; prefer aligned plot)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _check_comparable(aggs: list[dict], tol_ratio: float = 0.35) -> list[str]:
    """Warn if any model's aligned loss is far above the best (convergence check)."""
    warnings = []
    if not aggs:
        return warnings
    best = min(float(a["aligned_loss_mean"]) for a in aggs)
    for a in aggs:
        m = float(a["aligned_loss_mean"])
        if best > 0 and (m - best) / max(best, 1e-6) > tol_ratio and m > best + 0.5:
            warnings.append(
                f"{a['model']}: aligned={m:.3f} vs best={best:.3f} "
                f"(Δ={(m-best):.3f}) — check convergence / epochs"
            )
    return warnings


def _replot_from_summary(out: Path) -> int:
    summary_path = out / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    T = int(summary["config"]["T"])
    aggs_by_k = summary.get("aggregates_by_k") or {}
    nl_refs = summary.get("nl_refs") or []
    mu_points = summary.get("mu_points_vs_k") or []
    if mu_points:
        plot_aligned_vs_k(mu_points, nl_refs, T, plots / "final_aligned_loss_vs_k.png")
    # legacy single-k aggregates
    aggs = summary.get("aggregates") or []
    if aggs and not aggs_by_k:
        plot_final_aligned_bar(aggs, T, plots / "final_aligned_loss_bar.png")
        plot_final_curves(aggs, plots / "final_aligned_curves.png")
        plot_final_raw_curves(aggs, plots / "final_raw_curves.png")
    for k_str, aggs_k in aggs_by_k.items():
        plot_final_aligned_bar(
            aggs_k, T, plots / f"final_aligned_loss_bar_k{k_str}.png", title_suffix=f"k={k_str}"
        )
        plot_final_curves(aggs_k, plots / f"final_aligned_curves_k{k_str}.png")
        plot_final_raw_curves(aggs_k, plots / f"final_raw_curves_k{k_str}.png")
    print(f"replot done in {plots}")
    return 0


def _parse_ks(raw: str | None, fallback: list[int]) -> list[int]:
    if raw is None or not str(raw).strip():
        return list(fallback)
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="FINAL multi-seed loss campaign")
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--k", type=int, default=None, help="legacy single k (prefer --ks)")
    p.add_argument(
        "--ks",
        type=str,
        default="1,2,3,5,6",
        help="comma-separated k values for mu-models (nl is k-independent, run once)",
    )
    p.add_argument("--qsa-layers", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--poly-epochs", type=int, default=600)
    p.add_argument("--nl-epochs", type=int, default=500)
    p.add_argument("--nl-epochs-general", type=int, default=800)
    p.add_argument("--max-sentences", type=int, default=1000)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--nl-learning-rate", type=float, default=5e-3)
    p.add_argument("--nl-learning-rate-general", type=float, default=8e-3)
    p.add_argument("--nl-layers-full", type=int, default=2)
    p.add_argument("--nl-rank-full", type=int, default=None)
    p.add_argument("--nl-layers-small", type=int, default=1)
    p.add_argument("--nl-rank-small", type=int, default=None)
    p.add_argument("--nl-param-budget-small", type=int, default=128)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--model-seed-base", type=int, default=1042)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--checkpoint-every", type=int, default=20)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--mpi", action="store_true")
    p.add_argument("--quick", action="store_true", help="tiny local smoke")
    p.add_argument("--replot-only", type=str, default=None)
    p.add_argument("--models", type=str, default=None, help="comma-separated spec keys to run (default: all)")
    p.add_argument(
        "--no-poly",
        action="store_true",
        help="skip poly-k-QSA / poly-k-CSA (default: include on same plots)",
    )
    p.add_argument(
        "--isolate-jobs",
        action="store_true",
        help="run each (model,k,seed) in a fresh subprocess to avoid JAX CPU OOM",
    )
    p.add_argument(
        "--train-only",
        action="store_true",
        help="train assigned jobs and exit (no aggregate/plots); used by --isolate-jobs children",
    )
    args = p.parse_args()

    if args.replot_only:
        return _replot_from_summary(Path(args.replot_only))

    if args.k is not None:
        ks = [int(args.k)]
    else:
        ks = _parse_ks(args.ks, [1, 2, 3, 5, 6])
    # keep args.k as primary for any leftover single-k paths
    args.k = ks[0]

    if args.quick:
        args.T, args.d = 4, 4
        ks = [1, 2]
        args.k = 1
        args.qsa_layers = 2
        args.epochs = args.poly_epochs = args.nl_epochs = args.nl_epochs_general = 6
        args.max_sentences = 32
        args.n_seeds = 2
        args.batch_size = 16
        args.nl_layers_full = 1
        args.nl_rank_full = 2
        args.nl_layers_small = 1
        args.nl_rank_small = 1

    specs = _model_specs(args)
    if args.no_poly:
        specs = [s for s in specs if s.get("kernel_mode", "monomial") != "poly"]
    if args.models:
        want = {x.strip() for x in args.models.split(",") if x.strip()}
        specs = [s for s in specs if s["key"] in want]
        if not specs:
            raise SystemExit(f"no models matched --models={args.models}")

    mu_specs = [s for s in specs if s["family"] != "nl"]
    nl_specs = [s for s in specs if s["family"] == "nl"]

    model_seeds = [args.model_seed_base + i for i in range(args.n_seeds)]
    jobs: list[dict] = []
    for s in mu_specs:
        for k in ks:
            for ms in model_seeds:
                jobs.append({"spec": s, "seed": ms, "k": k})
    # nl is k-independent: train once (use ks[0] only for bookkeeping)
    for s in nl_specs:
        for ms in model_seeds:
            jobs.append({"spec": s, "seed": ms, "k": ks[0]})

    comm, rank, size = init_mpi(enabled=args.mpi)
    log = _log(rank)

    ks_tag = "-".join(str(k) for k in ks)
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("results") / "final_loss" / stamp
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "plots").mkdir(parents=True, exist_ok=True)
        (out / "aggregates").mkdir(parents=True, exist_ok=True)
    barrier(comm)

    my_jobs = shard_items(jobs, rank, size) if args.mpi else jobs
    if rank == 0:
        print("=" * 60)
        print("FINAL LOSS — multi-seed, multi-k (mono+poly on same plot)")
        print(f"T={args.T} d={args.d} ks={ks} n_seeds={args.n_seeds}")
        print(f"QSA L={args.qsa_layers} angles={qsa_angle_param_count(args.d, args.qsa_layers)}")
        print(f"CSA matrices={kcsa_matrix_param_count(args.d)}")
        for s in specs:
            if s["family"] == "nl":
                npar = nl_model_param_count(args.d, s["layers"], s["nl_rank"])
                print(
                    f"  {s['display']}: L={s['layers']} r={s['nl_rank']} "
                    f"params≈{npar} emb={s['nl_embedding_mode']} ep={s['epochs']} (once)"
                )
            else:
                print(f"  {s['display']}: kernel={s['kernel_mode']} ep={s['epochs']} × ks={ks}")
        print(f"jobs={len(jobs)} MPI ranks={size} (~{len(my_jobs)} on rank0)")
        print(f"Output: {out}")
        print("=" * 60)

    local_runs: list[dict] = []
    for job in my_jobs:
        spec = job["spec"]
        ms = int(job["seed"])
        k = int(job["k"])
        cfg = _make_cfg(args, spec, ms, out, k=k)
        print(f"\n===== {spec['display']} k={k} model_seed={ms} (rank {rank}/{size}) =====")
        if args.isolate_jobs:
            result = _isolate_train_one(spec, cfg, args, log)
        else:
            bundle = prepare_data_bundle(cfg)
            result = _train_one(spec, cfg, bundle, log)
            del bundle
            _clear_jax_memory()
        result["k"] = k
        print(
            f"[{spec['display']} k={k} seed={ms}] raw={result['final_loss']:.4f} "
            f"aligned={result['aligned_loss']:.4f}"
        )
        local_runs.append(result)

    if args.train_only:
        barrier(comm)
        return 0

    barrier(comm)
    all_runs = gather_list(comm, local_runs) if args.mpi else local_runs
    if rank != 0:
        barrier(comm)
        return 0

    # Prefer on-disk metrics for aggregation so a resumed/partial MPI run still
    # includes every completed (model,k,seed) even if another rank died earlier.
    disk_runs: list[dict] = []
    for job in jobs:
        spec = job["spec"]
        ms = int(job["seed"])
        k = int(job["k"])
        cfg = _make_cfg(args, spec, ms, out, k=k)
        loaded = _load_enriched_from_disk(spec, cfg)
        if loaded is None:
            print(f"[WARN] missing metrics for {cfg.run_label}")
            continue
        loaded["k"] = k
        disk_runs.append(loaded)
    if disk_runs:
        all_runs = disk_runs
        print(f"[aggregate] loaded {len(all_runs)}/{len(jobs)} runs from disk")

    # Group: nl by display; mu by (display, k)
    by_nl: dict[str, list[dict]] = {}
    by_mu: dict[tuple[str, int], list[dict]] = {}
    for r in all_runs or []:
        if str(r.get("family")) == "nl" or str(r.get("display", "")).startswith("nl-CSA"):
            by_nl.setdefault(r["display"], []).append(r)
        else:
            by_mu.setdefault((r["display"], int(r["k"])), []).append(r)

    nl_refs: list[dict] = []
    for s in nl_specs:
        name = s["display"]
        runs = sorted(by_nl.get(name, []), key=lambda r: int(r.get("model_seed", 0)))
        if not runs:
            print(f"[WARN] no runs for {name}")
            continue
        runs = _pad_histories(runs)
        for r in runs:
            r["model"] = name
        agg = aggregate_seed_results(runs)
        aligned = np.array(agg["seed_histories"], dtype=float)
        agg["aligned_history"] = aligned.mean(axis=0).tolist()
        agg["aligned_std"] = aligned.std(axis=0).tolist()
        finals = np.array([_aligned_loss(r, args.T) for r in runs], dtype=float)
        agg["aligned_loss_mean"] = float(finals.mean())
        agg["aligned_loss_std"] = float(finals.std())
        agg["aligned_loss_per_seed"] = finals.tolist()
        agg["raw_final_loss_mean"] = float(agg["final_loss_mean"])
        agg["raw_final_loss_std"] = float(agg["final_loss_std"])
        agg["spec"] = s
        agg["k"] = None
        nl_refs.append(agg)
        (out / "aggregates" / f"{s['key']}.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    mu_points: list[dict] = []
    aggregates_by_k: dict[str, list[dict]] = {str(k): [] for k in ks}
    for s in mu_specs:
        for k in ks:
            name = s["display"]
            runs = sorted(by_mu.get((name, k), []), key=lambda r: int(r.get("model_seed", 0)))
            if not runs:
                print(f"[WARN] no runs for {name} k={k}")
                continue
            runs = _pad_histories(runs)
            for r in runs:
                r["model"] = name
            agg = aggregate_seed_results(runs)
            logT = math.log(args.T)
            aligned = np.array(agg["seed_histories"], dtype=float) + logT
            agg["aligned_history"] = aligned.mean(axis=0).tolist()
            agg["aligned_std"] = aligned.std(axis=0).tolist()
            finals = np.array([_aligned_loss(r, args.T) for r in runs], dtype=float)
            agg["aligned_loss_mean"] = float(finals.mean())
            agg["aligned_loss_std"] = float(finals.std())
            agg["aligned_loss_per_seed"] = finals.tolist()
            agg["raw_final_loss_mean"] = float(agg["final_loss_mean"])
            agg["raw_final_loss_std"] = float(agg["final_loss_std"])
            agg["spec"] = s
            agg["k"] = k
            aggregates_by_k[str(k)].append(agg)
            mu_points.append(
                {
                    "model": name,
                    "k": k,
                    "aligned_loss_mean": agg["aligned_loss_mean"],
                    "aligned_loss_std": agg["aligned_loss_std"],
                    "raw_final_loss_mean": agg["raw_final_loss_mean"],
                    "n_params_angles": agg.get("n_params_angles"),
                    "n_params_matrices": agg.get("n_params_matrices"),
                    "kernel_mode": s.get("kernel_mode"),
                }
            )
            (out / "aggregates" / f"{s['key']}_k{k}.json").write_text(
                json.dumps(agg, indent=2), encoding="utf-8"
            )

    plots = out / "plots"
    plot_aligned_vs_k(mu_points, nl_refs, args.T, plots / "final_aligned_loss_vs_k.png")
    for k in ks:
        aggs_k = aggregates_by_k[str(k)] + nl_refs
        if not aggs_k:
            continue
        plot_final_aligned_bar(
            aggs_k, args.T, plots / f"final_aligned_loss_bar_k{k}.png", title_suffix=f"k={k}"
        )
        plot_final_curves(aggs_k, plots / f"final_aligned_curves_k{k}.png")
        plot_final_raw_curves(aggs_k, plots / f"final_raw_curves_k{k}.png")

    # Convergence check on all mu points + nl
    flat_for_check = [
        {"model": f"{p['model']} k={p['k']}", "aligned_loss_mean": p["aligned_loss_mean"]}
        for p in mu_points
    ] + nl_refs
    warns = _check_comparable(flat_for_check)

    # Trend diagnostic: does mono loss grow with k?
    trend_notes = []
    for name in ("k-QSA L=16", "k-CSA", "poly-k-QSA L=16", "poly-k-CSA"):
        pts = sorted([p for p in mu_points if p["model"] == name], key=lambda x: int(x["k"]))
        if len(pts) >= 2:
            delta = float(pts[-1]["aligned_loss_mean"] - pts[0]["aligned_loss_mean"])
            trend_notes.append(
                f"{name}: aligned(k={pts[0]['k']})={pts[0]['aligned_loss_mean']:.3f} -> "
                f"aligned(k={pts[-1]['k']})={pts[-1]['aligned_loss_mean']:.3f}  Δ={delta:+.3f}"
            )

    summary = {
        "config": {
            "T": args.T,
            "d": args.d,
            "ks": ks,
            "qsa_layers": args.qsa_layers,
            "n_seeds": args.n_seeds,
            "model_seeds": model_seeds,
            "data_seed": args.data_seed,
            "max_sentences": args.max_sentences,
            "batch_size": None if args.batch_size <= 0 else args.batch_size,
            "qsa_angles": qsa_angle_param_count(args.d, args.qsa_layers),
            "csa_matrices": kcsa_matrix_param_count(args.d),
            "include_poly": not args.no_poly,
        },
        "styles": STYLES,
        "alignment_note": (
            "Mu-models plotted as −logμ + log T; nl-CSA Renyi plotted as-is. "
            "poly-k-QSA/CSA use kernel_mode=poly (classical poly / LCU-truncated kernel) "
            "on the SAME vs-k plot as monomial — hoped to flatten growth of loss with k."
        ),
        "convergence_warnings": warns,
        "trend_notes": trend_notes,
        "mu_points_vs_k": mu_points,
        "nl_refs": nl_refs,
        "aggregates_by_k": aggregates_by_k,
        "per_seed": [
            {
                "display": r["display"],
                "spec_key": r["spec_key"],
                "k": r.get("k"),
                "model_seed": r.get("model_seed"),
                "final_loss": r["final_loss"],
                "aligned_loss": r["aligned_loss"],
                "n_params_angles": r.get("n_params_angles"),
                "n_params_matrices": r.get("n_params_matrices"),
                "n_params_model": r.get("n_params_model"),
                "nl_rank": r.get("nl_rank"),
                "ablation": r.get("ablation"),
                "kernel_mode": r.get("kernel_mode"),
            }
            for r in sorted(
                all_runs or [],
                key=lambda x: (x["display"], int(x.get("k", 0)), int(x.get("model_seed", 0))),
            )
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("FINAL LOSS vs k (aligned mean ± std)")
    for pnt in sorted(mu_points, key=lambda x: (x["model"], int(x["k"]))):
        print(
            f"  {pnt['model']:22s} k={pnt['k']}  "
            f"{pnt['aligned_loss_mean']:.4f} ± {pnt['aligned_loss_std']:.4f}"
        )
    for a in nl_refs:
        print(
            f"  {a['model']:22s} (k-indep)  "
            f"{a['aligned_loss_mean']:.4f} ± {a['aligned_loss_std']:.4f}"
        )
    if trend_notes:
        print("\nTrend (first→last k):")
        for t in trend_notes:
            print(f"  {t}")
    if warns:
        print("\n[WARN] loss comparability / convergence:")
        for w in warns:
            print(f"  - {w}")
    print(f"\n[FATTO] {out}")
    print(f"  replot: python run_final_loss.py --replot-only {out}")
    barrier(comm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
