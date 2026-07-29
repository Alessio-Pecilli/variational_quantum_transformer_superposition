#!/usr/bin/env python3
"""
run_study.py — unico script da lanciare per lo studio Section 2.

Cosa fa (in ordine):
  1. Self-check del circuito Section 2 (formule vs readout, pochi secondi)
  2. Sweep O_ij vs T : d=4, k=2, T in {2, 4, 8, 16}
  3. Sweep O_ij vs d : T=4 (T_lim), k=2, d in {2, 4, 8}
  4. CSV, plot PNG e RIASSUNTO.txt con tabella + interpretazione

Ogni training è lo stesso percorso di:
  python main_hpc.py --circuit-mode section2 --sentence-length T --embedding-dim d ...

Uso:
  python run_study.py                  # preset medio (default)
  python run_study.py --quick          # smoke test (~1 min)
  python run_study.py --full           # run serio
  python run_study.py --skip-self-check
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import DATASET_CONFIG, OPTIMIZATION_CONFIG
from jax_training_pipeline import run_training

ROOT = Path(__file__).resolve().parent

# Sweep fissi dello studio (limite locale: max 14 qubit)
SWEEP_T = (2, 4, 8, 16, 32, 64)  # FINAL includes T=64; local skips if > local_max_qubits
SWEEP_D = (2, 4, 8, 16, 32)      # HPC: include 16/32 for advantage regime
SWEEP_D_FIXED = 16               # default complexity d for vs-T on HPC
SWEEP_T_FIXED = 16               # default T for vs-d on HPC
# Extra panels: same k, vary the other axis
PANEL_D_ON_T = (4, 8, 16)        # obar vs T curves at these d (fixed k)
PANEL_T_ON_D = (8, 16, 32, 64)   # obar vs d curves at these T (fixed k)
T_LIM = 4  # kept for docs only; NOT drawn on FINAL plots
LOCAL_MAX_QUBITS = 15
DEFAULT_TARGET_LOSS = 2.5
FINAL_TARGET_LOSS = 3.8  # professor: ~3.8 is acceptable for FINAL obar

# Data = solid + markers; Haar/advantage refs use SAME color as the matching
# data curve, with different dash (Haar=dashed, advantage=dotted).
_DATA_MARKERS = ("o", "s", "D", "^", "v")
_DATA_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
_REF_HAAR_COLOR = "#000000"  # fallback when a single global Haar is drawn
_REF_ADV_COLOR = "#666666"


def _qubit_budget(T: int, d: int, k: int, kernel_mode: str = "monomial") -> int:
    if kernel_mode == "poly":
        from qsa_section2_circuit_polynomial import qubit_budget as qb
    else:
        from qsa_section2_circuit import qubit_budget as qb
    return int(qb(T, d, k))


def _obar_ylabel() -> str:
    return r"$\bar o = \mathrm{mean}_{i\leq j}|a_{ij}\,s_{ij}^{k}|$  (monomial diagnostic)"


def _mu_from_final_loss(final_loss: float) -> float:
    return float(math.exp(-float(final_loss)))


def _mu_advantage(d: int, k: int) -> float:
    # Requested reference: k / Binomial(d+k-1, k)
    return float(k / math.comb(int(d) + int(k) - 1, int(k)))


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("vqt_jax")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def _main_hpc_equivalent(T: int, d: int, k: int, epochs: int, max_sentences: int, lr: float, seed: int, label: str) -> str:
    return (
        f"python main_hpc.py --circuit-mode section2 "
        f"--sentence-length {T} --embedding-dim {d} --non-linear-order {k} "
        f"--epochs {epochs} --max-sentences {max_sentences} "
        f"--learning-rate {lr} --seed {seed} --run-label {label}"
    )


def _run_self_check(kernel_mode: str = "monomial") -> bool:
    print("\n" + "=" * 60)
    print(f"FASE 1/4 — Self-check circuito Section 2 (kernel={kernel_mode})")
    print("=" * 60)
    try:
        if kernel_mode == "poly":
            from qsa_section2_circuit_polynomial import self_check, set_backends

            set_backends()
            ok = True
            ok &= self_check(T=3, d=4, k=1, check_projector=True)
            ok &= self_check(T=3, d=4, k=2, check_projector=True)
            ok &= self_check(T=2, d=4, k=3, check_projector=True)
        else:
            from qsa_section2_circuit import self_check, set_backends

            set_backends()
            ok = True
            ok &= self_check(T=2, d=2, k=1, check_projector=True)
            ok &= self_check(T=4, d=4, k=2, check_projector=True)
        if ok:
            print("[SELF-CHECK] PASS (readout overlap == projector == analytic)\n")
        else:
            print("[SELF-CHECK] FAIL — readout mismatch.\n")
        return ok
    except Exception as exc:
        print(f"[SELF-CHECK] SKIP (ambiente: {exc})\n")
        return True


def _train_point(
    T: int,
    d: int,
    k: int,
    label: str,
    out_root: Path,
    epochs: int,
    max_sentences: int,
    lr: float,
    layers: int,
    seed: int,
    train_embedding: bool,
    logger: logging.Logger,
    train_opts: dict,
    local_max_qubits: int = LOCAL_MAX_QUBITS,
) -> dict:
    kernel_mode = str(train_opts.get("kernel_mode", "monomial"))
    n_q = _qubit_budget(T, d, k, kernel_mode=kernel_mode)
    if n_q > local_max_qubits:
        raise ValueError(
            f"{label}: n_qubits={n_q} > {local_max_qubits} (limite locale). "
            f"Usa HPC o --local-max-qubits."
        )

    cfg = dict(OPTIMIZATION_CONFIG)
    cfg.update(
        {
            "circuit_mode": "section2",
            "non_linear_order": k,
            "num_layers": layers,
            "epochs": epochs,
            "learning_rate": lr,
            "train_embedding": train_embedding,
            "seed": seed,
            "run_label": label,
            "embedding_dim": d,
            "output_dir": str(out_root / label),
            "local_max_qubits": local_max_qubits,
            **train_opts,
        }
    )

    DATASET_CONFIG["sentence_length"] = T
    DATASET_CONFIG["max_sentences"] = max_sentences

    equiv = _main_hpc_equivalent(T, d, k, epochs, max_sentences, lr, seed, label)
    print(f"\n--- {label} | T={T} d={d} k={k} n_qubits={n_q} epochs={epochs} ---")
    print(f"    (equiv. {equiv})")

    metrics_path = out_root / label / "summaries" / "metrics.json"
    target_loss = train_opts.get("target_loss")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        final_loss = float(metrics["final_loss"])
        ok_target = target_loss is None or final_loss <= float(target_loss)
        if ok_target:
            print(f"\n--- {label} | resume: already complete -> {metrics_path.parent.parent}")
            row = {
                "label": label,
                "T": T,
                "d": d,
                "k": k,
                "mean_O_ij": float(metrics["mean_O_ij"]),
                "obar": float(metrics.get("obar", metrics["mean_O_ij"])),
                "obar_k_root": float(metrics.get("obar_k_root", metrics["mean_O_ij"] ** (1 / max(k, 1)))),
                "final_loss": final_loss,
                "n_qubits": int(metrics["n_qubits"]),
                "reference_haar": float(metrics["reference_haar"]),
                "reference_advantage": float(metrics.get("reference_advantage", 0.0)),
                "converged": metrics.get("converged"),
                "training_phases": metrics.get("training_phases"),
                "elapsed_seconds": float(metrics.get("elapsed_seconds", 0.0)),
                "wall_seconds": 0.0,
                "run_dir": str(out_root / label),
                "main_hpc_cmd": equiv,
                "resumed": True,
            }
            print(
                f"    obar={row['obar']:.6f}  mean_O={row['mean_O_ij']:.6f}  "
                f"haar={row['reference_haar']:.6e}  adv={row['reference_advantage']:.6e}  "
                f"loss={row['final_loss']:.4f}  conv={row.get('converged')}  (skip)"
            )
            return row
        print(
            f"\n--- {label} | resume REJECTED: loss={final_loss:.4f} > target_loss={target_loss} "
            f"-> retrain"
        )
    t0 = time.perf_counter()
    exit_code = run_training(logger=logger, cfg=cfg, comm=None, rank=0, size=1)
    wall = time.perf_counter() - t0
    if exit_code != 0:
        raise RuntimeError(f"Training fallito per {label} (exit={exit_code})")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    row = {
        "label": label,
        "T": T,
        "d": d,
        "k": k,
        "mean_O_ij": float(metrics["mean_O_ij"]),
        "obar": float(metrics.get("obar", metrics["mean_O_ij"])),
        "obar_k_root": float(metrics.get("obar_k_root", metrics["mean_O_ij"] ** (1 / max(k, 1)))),
        "final_loss": float(metrics["final_loss"]),
        "n_qubits": int(metrics["n_qubits"]),
        "reference_haar": float(metrics["reference_haar"]),
        "reference_advantage": float(metrics.get("reference_advantage", 0.0)),
        "converged": metrics.get("converged"),
        "training_phases": metrics.get("training_phases"),
        "elapsed_seconds": float(metrics.get("elapsed_seconds", wall)),
        "wall_seconds": wall,
        "run_dir": str(out_root / label),
        "main_hpc_cmd": equiv,
    }
    print(
        f"    obar={row['obar']:.6f}  mean_O={row['mean_O_ij']:.6f}  "
        f"haar={row['reference_haar']:.6e}  adv={row['reference_advantage']:.6e}  "
        f"loss={row['final_loss']:.4f}  conv={row.get('converged')}  ({wall:.1f}s)"
    )
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = [k for k in rows[0] if k != "main_hpc_cmd"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_vs_T(
    series: list[tuple[str, list[dict]]],
    d_fixed: int,
    out_path: Path,
    t_lim: int | None = None,
) -> None:
    """Plot obar vs T with Haar / advantage refs matching each curve's color.

    Data = solid+markers; Haar = same color dashed; advantage = same color dotted.
    No T_lim vertical (t_lim ignored; kept for API compat). Linear y.
    """
    from qsa_section2_circuit import advantage_threshold, haar_floor

    fig, ax = plt.subplots(figsize=(8, 5))
    all_T: set[int] = set()
    drawn_refs: set[tuple[int, int]] = set()
    for si, (label, rows) in enumerate(series):
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: int(r["T"]))
        T_vals = [int(r["T"]) for r in rows]
        obar = [float(r["obar"]) for r in rows]
        all_T.update(T_vals)
        color = _DATA_COLORS[si % len(_DATA_COLORS)]
        marker = _DATA_MARKERS[si % len(_DATA_MARKERS)]
        ax.plot(
            T_vals,
            obar,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2.2,
            markersize=8,
            label=label,
            zorder=3,
        )
        k_val = int(rows[0]["k"])
        d_val = int(rows[0].get("d", d_fixed))
        key = (d_val, k_val)
        if key not in drawn_refs:
            drawn_refs.add(key)
            floor = haar_floor(d_val, k_val)
            adv = advantage_threshold(d_val, k_val)
            ax.axhline(
                floor,
                color=color,
                linestyle="--",
                linewidth=1.6,
                alpha=0.9,
                label=rf"$d^{{-(k+1)/2}}$ (k={k_val}, d={d_val})",
                zorder=1,
            )
            ax.axhline(
                adv,
                color=color,
                linestyle=":",
                linewidth=1.6,
                alpha=0.9,
                label=rf"$\sqrt{{k\,k!/d^k}}$ adv (k={k_val}, d={d_val})",
                zorder=1,
            )

    ax.set_xlabel("T (lunghezza sequenza)")
    ax.set_ylabel(_obar_ylabel())
    k_labels = ", ".join(lbl for lbl, _ in series)
    ax.set_title(f"obar vs T  (d={d_fixed}; {k_labels})")
    ax.set_xticks(sorted(all_T) if all_T else [])
    ax.set_yscale("linear")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_vs_d(
    series: list[tuple[str, list[dict]]],
    T_fixed: int,
    out_path: Path,
) -> None:
    """Plot obar vs d; Haar/advantage refs match each curve's color. Log y."""
    from qsa_section2_circuit import advantage_threshold, haar_floor

    fig, ax = plt.subplots(figsize=(8, 5))
    all_d: set[int] = set()
    drawn_k: set[int] = set()
    for si, (label, rows) in enumerate(series):
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: int(r["d"]))
        d_vals = np.array([int(r["d"]) for r in rows], dtype=float)
        obar = np.array([float(r["obar"]) for r in rows])
        all_d.update(int(d) for d in d_vals)
        color = _DATA_COLORS[si % len(_DATA_COLORS)]
        marker = _DATA_MARKERS[si % len(_DATA_MARKERS)]
        ax.plot(
            d_vals,
            obar,
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2.2,
            markersize=8,
            label=label,
            zorder=3,
        )
        k_val = int(rows[0]["k"])
        if k_val not in drawn_k:
            drawn_k.add(k_val)
            floor = np.array([haar_floor(int(d), k_val) for d in d_vals])
            adv = np.array([advantage_threshold(int(d), k_val) for d in d_vals])
            ax.plot(
                d_vals,
                floor,
                color=color,
                linestyle="--",
                linewidth=1.6,
                marker="s",
                markersize=5,
                alpha=0.9,
                label=rf"$d^{{-(k+1)/2}}$ (k={k_val})",
                zorder=1,
            )
            ax.plot(
                d_vals,
                adv,
                color=color,
                linestyle=":",
                linewidth=1.6,
                marker="^",
                markersize=5,
                alpha=0.9,
                label=rf"$\sqrt{{k\,k!/d^k}}$ adv (k={k_val})",
                zorder=1,
            )

    ax.set_xlabel("d (dimensione embedding)")
    ax.set_ylabel(_obar_ylabel())
    ax.set_title(f"obar vs d  (T={T_fixed})")
    ax.set_xticks(sorted(all_d) if all_d else [])
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_vs_T_by_d(
    rows: list[dict],
    k_fixed: int,
    out_path: Path,
    t_lim: int | None = None,
) -> None:
    """obar vs T: same k, one curve per d; Haar/adv match curve color. No T_lim."""
    from qsa_section2_circuit import advantage_threshold, haar_floor

    by_d: dict[int, list[dict]] = {}
    for r in rows:
        if int(r["k"]) != k_fixed:
            continue
        by_d.setdefault(int(r["d"]), []).append(r)
    if len(by_d) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for si, d_val in enumerate(sorted(by_d)):
        pts = sorted(by_d[d_val], key=lambda r: int(r["T"]))
        color = _DATA_COLORS[si % len(_DATA_COLORS)]
        marker = _DATA_MARKERS[si % len(_DATA_MARKERS)]
        ax.plot(
            [int(r["T"]) for r in pts],
            [float(r["obar"]) for r in pts],
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2.2,
            markersize=8,
            label=rf"trained $d={d_val}$",
            zorder=3,
        )
        floor = haar_floor(d_val, k_fixed)
        adv = advantage_threshold(d_val, k_fixed)
        ax.axhline(
            floor,
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label=rf"$d^{{-(k+1)/2}}$ ($d={d_val}$)",
            zorder=1,
        )
        ax.axhline(
            adv,
            color=color,
            linestyle=":",
            linewidth=1.4,
            alpha=0.85,
            label=rf"adv ($d={d_val}$)",
            zorder=1,
        )

    ax.set_xlabel("T (lunghezza sequenza)")
    ax.set_ylabel(_obar_ylabel())
    ax.set_title(f"obar vs T  (k={k_fixed}, curves = different d)")
    ax.set_yscale("linear")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_vs_d_by_T(
    rows: list[dict],
    k_fixed: int,
    out_path: Path,
) -> None:
    """obar vs d: same k, one curve per T; Haar/adv in black (d-only refs). Log y."""
    from qsa_section2_circuit import advantage_threshold, haar_floor

    by_T: dict[int, list[dict]] = {}
    for r in rows:
        if int(r["k"]) != k_fixed:
            continue
        by_T.setdefault(int(r["T"]), []).append(r)
    if len(by_T) < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    all_d = sorted({int(r["d"]) for pts in by_T.values() for r in pts})
    if all_d:
        floor = [haar_floor(d, k_fixed) for d in all_d]
        adv = [advantage_threshold(d, k_fixed) for d in all_d]
        ax.plot(
            all_d,
            floor,
            color=_REF_HAAR_COLOR,
            linestyle="--",
            linewidth=2.0,
            marker="s",
            markersize=5,
            label=rf"$d^{{-(k+1)/2}}$ (k={k_fixed})",
            zorder=1,
        )
        ax.plot(
            all_d,
            adv,
            color=_REF_ADV_COLOR,
            linestyle=":",
            linewidth=1.8,
            marker="^",
            markersize=5,
            label=rf"$\sqrt{{k\,k!/d^k}}$ adv (k={k_fixed})",
            zorder=1,
        )

    for si, T_val in enumerate(sorted(by_T)):
        pts = sorted(by_T[T_val], key=lambda r: int(r["d"]))
        color = _DATA_COLORS[si % len(_DATA_COLORS)]
        marker = _DATA_MARKERS[si % len(_DATA_MARKERS)]
        ax.plot(
            [int(r["d"]) for r in pts],
            [float(r["obar"]) for r in pts],
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=2.2,
            markersize=8,
            label=rf"trained $T={T_val}$",
            zorder=3,
        )

    ax.set_xlabel("d (dimensione embedding)")
    ax.set_ylabel(_obar_ylabel())
    ax.set_title(f"obar vs d  (k={k_fixed}, curves = different T)")
    ax.set_xticks(all_d)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _aggregate_final_rows_by_seed(rows: list[dict]) -> list[dict]:
    """Aggregate FINAL grid rows across model seeds by (series,T,d,k,kernel_mode)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (
            r.get("series"),
            int(r.get("T")),
            int(r.get("d")),
            int(r.get("k")),
            str(r.get("kernel_mode", "monomial")),
        )
        groups.setdefault(key, []).append(r)

    out: list[dict] = []
    for key, grp in groups.items():
        ok = [g for g in grp if g.get("error") is None and g.get("final_loss") is not None]
        if not ok:
            continue
        mu_vals = np.array([_mu_from_final_loss(float(g["final_loss"])) for g in ok], dtype=float)
        losses = np.array([float(g["final_loss"]) for g in ok], dtype=float)
        base = dict(ok[0])
        base["n_seeds"] = int(len(ok))
        base["final_loss"] = float(losses.mean())
        base["final_loss_std"] = float(losses.std())
        base["mu_mean"] = float(mu_vals.mean())
        base["mu_std"] = float(mu_vals.std())
        base["mu_values"] = mu_vals.tolist()
        base["mu_advantage"] = _mu_advantage(int(base["d"]), int(base["k"]))
        out.append(base)
    return out


def _plot_final_mu_panels(
    agg_rows: list[dict],
    out_root: Path,
    max_T: int,
    mu_ks: list[int],
) -> None:
    """Professor revision: plot mu (not obar) with multi-seed error bars."""
    # ---------- mu vs T (d=16), only k=2,5 for readability ----------
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    mu_ks_T = [k for k in (2, 5) if k in set(mu_ks)]
    for i, k in enumerate(mu_ks_T):
        color = _DATA_COLORS[i % len(_DATA_COLORS)]
        pts_mono = sorted(
            [
                r
                for r in agg_rows
                if int(r["d"]) == 16
                and int(r["k"]) == int(k)
                and str(r.get("kernel_mode", "monomial")) == "monomial"
            ],
            key=lambda r: int(r["T"]),
        )
        pts_poly = sorted(
            [
                r
                for r in agg_rows
                if int(r["d"]) == 16
                and int(r["k"]) == int(k)
                and str(r.get("kernel_mode", "monomial")) == "poly"
            ],
            key=lambda r: int(r["T"]),
        )
        if pts_mono:
            Ts = [int(r["T"]) for r in pts_mono]
            ys = [float(r["mu_mean"]) for r in pts_mono]
            es = [float(r["mu_std"]) for r in pts_mono]
            ax.errorbar(Ts, ys, yerr=es, color=color, marker="o", linestyle="-", linewidth=2.2, capsize=4, label=f"k={k} mono")
            # Advantage is T-independent
            adv = float(pts_mono[0]["mu_advantage"])
            ax.axhline(adv, color=color, linestyle=":", linewidth=1.8, alpha=0.95, label=f"k={k} advantage")
        if pts_poly:
            Ts = [int(r["T"]) for r in pts_poly]
            ys = [float(r["mu_mean"]) for r in pts_poly]
            es = [float(r["mu_std"]) for r in pts_poly]
            ax.errorbar(Ts, ys, yerr=es, color=color, marker="s", linestyle="--", linewidth=2.0, capsize=4, label=f"k={k} poly")
    ax.set_xlabel("T")
    ax.set_ylabel(r"$\mu$  (mean $\pm$ std over seeds)")
    ax.set_title(rf"$\mu$ vs $T$  (d=16, T\leq{max_T})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_root / "mu_vs_T.png", dpi=220)
    plt.close(fig)

    # ---------- mu vs d (T=max_T) for k=2,3,5 ----------
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ks_d = [k for k in (2, 3, 5) if k in set(mu_ks)]
    for i, k in enumerate(ks_d):
        color = _DATA_COLORS[i % len(_DATA_COLORS)]
        pts_mono = sorted(
            [
                r
                for r in agg_rows
                if int(r["T"]) == int(max_T)
                and int(r["k"]) == int(k)
                and str(r.get("kernel_mode", "monomial")) == "monomial"
            ],
            key=lambda r: int(r["d"]),
        )
        pts_poly = sorted(
            [
                r
                for r in agg_rows
                if int(r["T"]) == int(max_T)
                and int(r["k"]) == int(k)
                and str(r.get("kernel_mode", "monomial")) == "poly"
            ],
            key=lambda r: int(r["d"]),
        )
        if pts_mono:
            ds = [int(r["d"]) for r in pts_mono]
            ys = [float(r["mu_mean"]) for r in pts_mono]
            es = [float(r["mu_std"]) for r in pts_mono]
            ax.errorbar(ds, ys, yerr=es, color=color, marker="o", linestyle="-", linewidth=2.2, capsize=4, label=f"k={k} mono")
            adv = [float(r["mu_advantage"]) for r in pts_mono]
            ax.plot(ds, adv, color=color, linestyle=":", linewidth=1.8, marker="^", markersize=4, label=f"k={k} advantage")
        if pts_poly:
            ds = [int(r["d"]) for r in pts_poly]
            ys = [float(r["mu_mean"]) for r in pts_poly]
            es = [float(r["mu_std"]) for r in pts_poly]
            ax.errorbar(ds, ys, yerr=es, color=color, marker="s", linestyle="--", linewidth=2.0, capsize=4, label=f"k={k} poly")
    ax.set_yscale("log")
    ax.set_xlabel("d")
    ax.set_ylabel(r"$\mu$  (mean $\pm$ std over seeds, log scale)")
    ax.set_title(rf"$\mu$ vs $d$  (T={max_T})")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_root / "mu_vs_d.png", dpi=220)
    plt.close(fig)


def _write_summary(
    out_root: Path,
    preset: str,
    k: int,
    epochs: int,
    max_sentences: int,
    seed: int,
    t_rows: list[dict],
    d_rows: list[dict],
    self_check_ok: bool,
    total_wall: float,
) -> None:
    lines = [
        "RIASSUNTO STUDIO SECTION 2",
        "=" * 50,
        f"Data:        {datetime.now().isoformat(timespec='seconds')}",
        f"Preset:      {preset}",
        f"k:           {k}",
        f"Epoche:      {epochs}",
        f"Frasi:       {max_sentences}",
        f"Seed:        {seed}",
        f"Self-check:  {'PASS' if self_check_ok else 'FAIL/SKIP'}",
        f"Tempo tot.:  {total_wall:.1f} s",
        f"Output:      {out_root}",
        "",
        "METRICA: obar = mean_{i<=j} |a_ij s_ij^k|  (= absS/Ntri, NO potenza 1/k)",
        "         obar_k_root = obar^{1/k}  (solo diagnostica, non usata nei plot)",
        "         s_ij = <x_j|W|x_i>, a_ij = <x_{j+1}|V|x_i>",
        "Refs: PRIMARY bug-check = d^{-(k+1)/2} ; secondary advantage = sqrt(k * k! / d^k)",
        "      (obar always monomial mean|a s^k| even when training uses poly kernel)",
        "      (Haar floor in code == d^{-(k+1)/2}; use that as the main theory line)",
        "",
    ]

    if t_rows:
        lines += ["SWEEP vs T  (d=4, k=2)", "-" * 40]
        lines.append(f"{'T':>4}  {'obar':>12}  {'Haar':>12}  {'adv':>12}  {'n_qubits':>8}")
        for r in sorted(t_rows, key=lambda x: int(x["T"])):
            lines.append(
                f"{int(r['T']):>4}  {r['obar']:>12.6f}  "
                f"{r['reference_haar']:>12.6e}  {r['reference_advantage']:>12.6e}  "
                f"{int(r['n_qubits']):>8}"
            )
        vals = [float(r["obar"]) for r in t_rows if int(r["T"]) >= T_LIM]
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            mean_t = float(np.mean(vals))
            lines += [
                "",
                f"  T>={T_LIM}: media obar={mean_t:.4f}, spread={spread:.4f}",
                f"  Atteso: circa costante in T per T >= T_lim={T_LIM} (T=2 spesso outlier).",
            ]
        lines.append("")

    if d_rows:
        T_d = int(d_rows[0]["T"]) if d_rows else SWEEP_T_FIXED
        lines += [f"SWEEP vs d  (T={T_d}, k variabile)", "-" * 40]
        lines.append(f"{'d':>4}  {'obar':>12}  {'Haar':>12}  {'adv':>12}  {'>Haar':>8}  {'>adv':>8}")
        for r in sorted(d_rows, key=lambda x: int(x["d"])):
            beat_h = "si" if r["obar"] > r["reference_haar"] else "no"
            beat_a = "si" if r["obar"] > r["reference_advantage"] else "no"
            lines.append(
                f"{int(r['d']):>4}  {r['obar']:>12.6f}  "
                f"{r['reference_haar']:>12.6e}  {r['reference_advantage']:>12.6e}  "
                f"{beat_h:>8}  {beat_a:>8}"
            )
        lines += [
            "",
            "  Atteso: obar sopra Haar e possibilmente sopra advantage threshold.",
        ]

    lines += [
        "",
        "FILE GENERATI",
        "-" * 40,
        "  summary_vs_T.csv, summary_vs_d.csv",
        "  mean_O_vs_T.png, mean_O_vs_d.png  (obar + d^{-(k+1)/2} + adv, y linear)",
        "  mean_O_vs_T_by_d.png, mean_O_vs_d_by_T.png  (panel curves)",
        "  manifest.json",
        "  RIASSUNTO.txt  (questo file)",
        "",
        "COMANDI EQUIVALENTI (main_hpc.py per un singolo punto)",
        "-" * 40,
    ]
    for r in t_rows + d_rows:
        lines.append(f"  {r.get('main_hpc_cmd', '')}")

    text = "\n".join(lines) + "\n"
    (out_root / "RIASSUNTO.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


def _parse_int_list(raw: str | None) -> list[int]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _count_ptb_sentences(T: int) -> int:
    path = Path("ptb_sentences.txt")
    if not path.exists():
        return 0
    n = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(line.split()) == T:
                n += 1
    return n


def _build_final_obar_jobs(
    local_max_qubits: int,
    kernel_mode: str,
    max_T: int = 64,
    obar_ks: tuple[int, ...] | list[int] = (1, 2, 3, 5, 6),
    also_poly: bool = False,
) -> list[dict]:
    """Union of FINAL obar panels (deduped by T,d,k,kernel)."""
    Ts = [t for t in SWEEP_T if t <= max_T]
    if max_T not in Ts and max_T > 0:
        Ts = sorted(set(Ts) | {max_T})
    # Data-availability guard (PTB exact-length sentences).
    for T in Ts:
        n_avail = _count_ptb_sentences(T)
        if n_avail < 16:
            print(
                f"  [WARN] PTB has only {n_avail} unique sentences with T={T}. "
                f"Training will duplicate them — prefer MAX_T<=32."
            )
    panel_Ts = [t for t in PANEL_T_ON_D if t <= max_T]
    if max_T >= 64 and 64 not in panel_Ts:
        panel_Ts = sorted(set(panel_Ts) | {64})
    kernels = [kernel_mode]
    if also_poly and "poly" not in kernels:
        kernels.append("poly")
    jobs: list[dict] = []
    seen: set[tuple[int, int, int, str]] = set()

    def _add(T: int, d: int, k: int, series: str, km: str) -> None:
        key = (T, d, k, km)
        if key in seen:
            return
        n_q = _qubit_budget(T, d, k, kernel_mode=km)
        if n_q > local_max_qubits:
            print(f"  [skip final] T={T} d={d} k={k} {km}: n_qubits={n_q} > {local_max_qubits}")
            return
        seen.add(key)
        tag = "poly" if km == "poly" else "mono"
        jobs.append(
            {
                "series": series,
                "T": T,
                "d": d,
                "k": k,
                "kernel_mode": km,
                "label": f"T{T}_d{d}_k{k}_{tag}",
            }
        )

    for km in kernels:
        for k in obar_ks:
            for T in Ts:
                _add(T, 16, k, f"T_k{k}", km)
        for d in PANEL_D_ON_T:
            for T in Ts:
                _add(T, d, 3, "T_by_d", km)
        for k in (2, 3, 5):
            for d in SWEEP_D:
                _add(max_T, d, k, f"d_k{k}", km)
        for T in panel_Ts:
            for d in SWEEP_D:
                _add(T, d, 3, "d_by_T", km)
    return jobs


def _plot_final_obar_panels(
    all_rows: list[dict],
    out_root: Path,
    max_T: int = 64,
    obar_ks: tuple[int, ...] | list[int] = (1, 2, 3, 5, 6),
) -> None:
    """Emit the four FINAL figure panels from a flat list of trained points.

    If both mono and poly rows exist for the same (T,d,k), they appear on the
    same vs-T plot as separate series (poly labeled explicitly).
    """
    import math

    ok = [
        r
        for r in all_rows
        if r.get("error") is None
        and not (
            isinstance(r.get("obar"), float)
            and math.isnan(float(r.get("obar", float("nan"))))
        )
    ]
    for r in ok:
        r.setdefault("kernel_mode", "monomial")

    t_series: list[tuple[str, list[dict]]] = []
    for k in obar_ks:
        for km, tag in (("monomial", "mono"), ("poly", "poly")):
            rows = [
                r
                for r in ok
                if int(r["d"]) == 16
                and int(r["k"]) == k
                and str(r.get("kernel_mode", "monomial")) == km
            ]
            if rows:
                label = f"k={k} {tag}" if any(str(x.get("kernel_mode")) == "poly" for x in ok) else f"k={k} (trained)"
                if km == "monomial" and not any(str(x.get("kernel_mode")) == "poly" for x in ok):
                    label = f"k={k} (trained)"
                elif km == "poly":
                    label = f"k={k} poly"
                else:
                    label = f"k={k} mono"
                t_series.append((label, rows))
    if t_series:
        _write_csv(out_root / "summary_vs_T.csv", [r for _, rows in t_series for r in rows])
        _plot_vs_T(t_series, d_fixed=16, out_path=out_root / "mean_O_vs_T.png")

    # by-d / by-T panels: prefer monomial; if only poly present use that
    def _prefer_mono(rows: list[dict]) -> list[dict]:
        mono = [r for r in rows if str(r.get("kernel_mode", "monomial")) == "monomial"]
        return mono if mono else rows

    t_by_d = _prefer_mono([r for r in ok if int(r["k"]) == 3 and int(r["d"]) in PANEL_D_ON_T])
    if t_by_d:
        _write_csv(out_root / "summary_vs_T_by_d.csv", t_by_d)
        _plot_vs_T_by_d(t_by_d, k_fixed=3, out_path=out_root / "mean_O_vs_T_by_d.png")

    d_series: list[tuple[str, list[dict]]] = []
    for k in (2, 3):
        for km, tag in (("monomial", "mono"), ("poly", "poly")):
            rows = [
                r
                for r in ok
                if int(r["T"]) == max_T
                and int(r["k"]) == k
                and str(r.get("kernel_mode", "monomial")) == km
            ]
            if rows:
                has_poly = any(str(x.get("kernel_mode")) == "poly" for x in ok if int(x["T"]) == max_T)
                label = f"k={k} {tag}" if has_poly else f"k={k} (trained)"
                d_series.append((label, rows))
    if d_series:
        _write_csv(out_root / "summary_vs_d.csv", [r for _, rows in d_series for r in rows])
        _plot_vs_d(d_series, T_fixed=max_T, out_path=out_root / "mean_O_vs_d.png")

    panel_Ts = {t for t in PANEL_T_ON_D if t <= max_T}
    if max_T >= 64:
        panel_Ts.add(64)
    d_by_T = _prefer_mono([r for r in ok if int(r["k"]) == 3 and int(r["T"]) in panel_Ts])
    if d_by_T:
        _write_csv(out_root / "summary_vs_d_by_T.csv", d_by_T)
        _plot_vs_d_by_T(d_by_T, k_fixed=3, out_path=out_root / "mean_O_vs_d_by_T.png")


def _replot_study_dir(out_root: Path, args) -> int:
    """Regenerate PNGs from manifest.json / CSVs without retraining."""
    manifest_path = out_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    if man.get("final_obar_grid"):
        all_rows = list(man.get("all_rows") or [])
        for key in ("T_sweep", "T_sweep_k3", "T_by_d", "d_sweep", "d_sweep_extra", "d_by_T"):
            all_rows.extend(man.get(key) or [])
        by_label = {r["label"]: r for r in all_rows if "label" in r}
        max_T = int(man.get("max_T", 64))
        obar_ks = man.get("obar_ks") or [1, 2, 3, 5, 6]
        if man.get("metric") == "mu" or man.get("all_rows_seed_agg"):
            agg_rows = list(man.get("all_rows_seed_agg") or [])
            if not agg_rows:
                agg_rows = _aggregate_final_rows_by_seed(list(by_label.values()))
            _plot_final_mu_panels(agg_rows, out_root, max_T=max_T, mu_ks=list(obar_ks))
        else:
            _plot_final_obar_panels(
                list(by_label.values()), out_root, max_T=max_T, obar_ks=obar_ks
            )
        print(f"replot FINAL panels in {out_root}")
        return 0

    t_rows = man.get("T_sweep") or []
    t_rows_k3 = man.get("T_sweep_k3") or []
    d_rows = man.get("d_sweep") or []
    d_rows_extra = man.get("d_sweep_extra") or []
    t_by_d = man.get("T_by_d") or []
    d_by_T = man.get("d_by_T") or []
    k = int(man.get("k", args.k))
    d_fixed = int(man.get("d_fixed", SWEEP_D_FIXED))
    T_fixed = int(man.get("d_sweep_T", SWEEP_T_FIXED))
    ek = man.get("extra_k_on_d")

    t_by_k = man.get("T_sweep_by_k") or {}
    if t_rows or t_rows_k3 or t_by_k:
        if t_by_k:
            t_series = [
                (f"k={kk} (trained)", t_by_k[str(kk)])
                for kk in sorted(int(x) for x in t_by_k.keys())
            ]
        else:
            t_series = [(f"k={k} (trained)", t_rows)]
            if t_rows_k3:
                t_series.append(("k=3 (trained)", t_rows_k3))
        _plot_vs_T(t_series, d_fixed=d_fixed, out_path=out_root / "mean_O_vs_T.png")
        print(f"wrote {out_root / 'mean_O_vs_T.png'}")
    if d_rows or d_rows_extra:
        d_series = [(f"k={k} (trained)", d_rows)]
        if d_rows_extra:
            d_series.append((f"k={ek} (trained)", d_rows_extra))
        _plot_vs_d(d_series, T_fixed=T_fixed, out_path=out_root / "mean_O_vs_d.png")
        print(f"wrote {out_root / 'mean_O_vs_d.png'}")

    panel_T = t_rows + t_by_d
    if panel_T:
        _plot_vs_T_by_d(panel_T, k_fixed=k, out_path=out_root / "mean_O_vs_T_by_d.png")
        print(f"wrote {out_root / 'mean_O_vs_T_by_d.png'}")
    panel_d = d_rows + d_by_T
    if panel_d:
        _plot_vs_d_by_T(panel_d, k_fixed=k, out_path=out_root / "mean_O_vs_d_by_T.png")
        print(f"wrote {out_root / 'mean_O_vs_d_by_T.png'}")
    return 0


def _apply_preset(args) -> str:
    if args.quick:
        args.epochs = 6
        args.max_sentences = 16
        return "quick"
    if args.long:
        if args.epochs is None:
            args.epochs = 40
        if args.max_sentences is None:
            args.max_sentences = 64
        args.train_until_converged = True
        # Do NOT overwrite CLI --max-epochs (HPC passes 800).
        if args.max_epochs is None:
            args.max_epochs = 300
        if args.target_loss is None:
            args.target_loss = DEFAULT_TARGET_LOSS
        return "long"
    if args.full:
        if args.epochs is None:
            args.epochs = 40
        if args.max_sentences is None:
            args.max_sentences = 64
        return "full"
    if args.epochs is None:
        args.epochs = 20
    if args.max_sentences is None:
        args.max_sentences = 32
    if args.max_epochs is None:
        args.max_epochs = 300
    return "medium"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Studio completo Section 2: self-check + sweep T/d + plot + riassunto."
    )
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--quick", action="store_true", help="6 epoche, 16 frasi (~1 min)")
    preset.add_argument("--full", action="store_true", help="40 epoche, 64 frasi (run serio)")
    preset.add_argument(
        "--long",
        action="store_true",
        help="64 frasi, train fino a convergenza loss (max 300 epoche)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="default: 20 (preset medium)")
    parser.add_argument("--max-sentences", type=int, default=None, help="default: 32 (preset medium)")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-embedding", action="store_true")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--only", choices=("T", "d"), default=None, help="solo uno dei due sweep")
    parser.add_argument(
        "--d-sweep-T",
        type=int,
        default=SWEEP_T_FIXED,
        help=f"T fisso per sweep vs d (default {SWEEP_T_FIXED} = T_lim)",
    )
    parser.add_argument(
        "--extra-k3",
        action="store_true",
        help="aggiunge curva numerica k=3 nello sweep vs T (d=4; T=16 richiede 16 qubit)",
    )
    parser.add_argument(
        "--extra-k-on-d",
        type=int,
        default=None,
        metavar="K",
        help="seconda curva numerica vs d a T fisso (es. k=3; d=8 può superare 15 qubit)",
    )
    parser.add_argument(
        "--local-max-qubits",
        type=int,
        default=LOCAL_MAX_QUBITS,
        help=f"limite qubit locale (default {LOCAL_MAX_QUBITS}; usa 16 per T=16 k=3)",
    )
    parser.add_argument("--curriculum-k", action="store_true", help="train k=1 poi k target (warm-start params)")
    parser.add_argument("--warm-start-w", action="store_true", help="inizializza W~I (weights_w=0)")
    parser.add_argument("--train-until-converged", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None, help="cap when training to target/convergence (default 300; HPC uses 800)")
    parser.add_argument("--loss-rel-tol", type=float, default=1e-4)
    parser.add_argument("--convergence-patience", type=int, default=8)
    parser.add_argument(
        "--target-loss",
        type=float,
        default=None,
        help=f"keep training until loss<=this (default {DEFAULT_TARGET_LOSS} with --long); "
        f"resume skips only if metrics already meet the target",
    )
    parser.add_argument(
        "--kernel-mode",
        choices=("poly", "monomial"),
        default="monomial",
        help="attention kernel: monomial = legacy s^k (default); poly = LCU softmax truncation (abandoned)",
    )
    parser.add_argument(
        "--panel-d-on-T",
        type=str,
        default=",".join(str(x) for x in PANEL_D_ON_T),
        help="extra obar-vs-T panel: comma-separated d values at fixed k (empty to disable)",
    )
    parser.add_argument(
        "--panel-T-on-d",
        type=str,
        default=",".join(str(x) for x in PANEL_T_ON_D),
        help="extra obar-vs-d panel: comma-separated T values at fixed k (empty to disable)",
    )
    parser.add_argument(
        "--replot-only",
        type=str,
        default=None,
        help="only regenerate PNGs from existing manifest/CSVs in this study dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="fixed study root (skips labels with summaries/metrics.json; default: timestamped)",
    )
    parser.add_argument(
        "--mpi",
        action="store_true",
        help="shard sweep points across MPI ranks (use with srun)",
    )
    parser.add_argument(
        "--sweep-T",
        type=str,
        default=None,
        help="comma-separated T values for vs-T / panels (default: module SWEEP_T)",
    )
    parser.add_argument(
        "--sweep-D",
        type=str,
        default=None,
        help="comma-separated d values for vs-d / panels (default: module SWEEP_D)",
    )
    parser.add_argument(
        "--d-fixed",
        type=int,
        default=SWEEP_D_FIXED,
        help=f"fixed d for vs-T sweep (default {SWEEP_D_FIXED})",
    )
    parser.add_argument(
        "--extra-ks",
        type=str,
        default=None,
        help="comma-separated extra k values on vs-T (replaces --extra-k3 when set)",
    )
    parser.add_argument(
        "--final-obar-grid",
        action="store_true",
        help="run the FINAL obar campaign grid (all 4 panels; ignores --only)",
    )
    parser.add_argument(
        "--max-T",
        type=int,
        default=64,
        help="cap T for --final-obar-grid (use 32 if T=64 is too heavy)",
    )
    parser.add_argument(
        "--final-obar-ks",
        type=str,
        default="1,2,3,5,6",
        help="k values for FINAL obar vs-T panel (default 1,2,3,5,6)",
    )
    parser.add_argument(
        "--also-poly",
        action="store_true",
        help="with --final-obar-grid, also train poly kernel and overlay on same plots",
    )
    parser.add_argument(
        "--final-n-seeds",
        type=int,
        default=5,
        help="number of model seeds per FINAL point (for error bars)",
    )
    parser.add_argument(
        "--final-seed-base",
        type=int,
        default=42,
        help="base model seed for FINAL multi-seed grid",
    )
    args = parser.parse_args()

    if args.long:
        args.train_until_converged = True
    if args.target_loss is not None:
        args.train_until_converged = True
    if args.final_obar_grid and args.target_loss is None:
        args.target_loss = FINAL_TARGET_LOSS
        args.train_until_converged = True

    if args.replot_only:
        return _replot_study_dir(Path(args.replot_only), args)

    preset_name = _apply_preset(args)
    if args.max_epochs is None:
        args.max_epochs = 300
    logger = _setup_logger()

    from mpi_runtime import barrier, gather_list, init_mpi, shard_items

    comm, rank, size = init_mpi(enabled=args.mpi)

    if args.output_dir:
        out_root = Path(args.output_dir)
        stamp = Path(args.output_dir).name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = Path("results") / "study" / stamp
    if rank == 0:
        out_root.mkdir(parents=True, exist_ok=True)
    barrier(comm)

    if rank == 0:
        print("=" * 60)
        print("RUN_STUDY — studio Section 2 (overlap O_ij vs T e d)")
        print("=" * 60)
        print(f"Preset: {preset_name} | epoche={args.epochs} | frasi={args.max_sentences} | k={args.k}")
        print(f"kernel_mode={args.kernel_mode}")
        if args.train_until_converged:
            print(
                f"Convergenza: max_epochs={args.max_epochs} "
                f"rel_tol={args.loss_rel_tol} patience={args.convergence_patience}"
                + (f" target_loss={args.target_loss}" if args.target_loss is not None else "")
            )
        if args.curriculum_k:
            print("Curriculum k: 1 -> k")
        if args.warm_start_w:
            print("Warm-start: W ~ I (weights_w=0)")
        print(f"MPI: enabled={args.mpi} ranks={size}")
        print(f"Output: {out_root}")
        print(f"Ogni run = stesso percorso di main_hpc.py --circuit-mode section2")
        print("=" * 60)

    t_start = time.perf_counter()
    self_check_ok = True
    if rank == 0 and not args.skip_self_check:
        self_check_ok = _run_self_check(kernel_mode=args.kernel_mode)
        if not self_check_ok:
            print("[WARN] Self-check readout fallito; proseguo comunque (usa --skip-self-check per saltare).")
    barrier(comm)

    train_opts = {
        "warm_start_w_identity": args.warm_start_w,
        "curriculum_k": args.curriculum_k,
        "train_until_converged": args.train_until_converged,
        "max_epochs": args.max_epochs,
        "loss_rel_tol": args.loss_rel_tol,
        "convergence_patience": args.convergence_patience,
        "target_loss": args.target_loss,
        "kernel_mode": args.kernel_mode,
    }
    train_kw = dict(
        k=args.k,
        epochs=args.epochs,
        max_sentences=args.max_sentences,
        lr=args.learning_rate,
        layers=args.layers,
        seed=args.seed,
        train_embedding=not args.freeze_embedding,
        logger=logger,
        out_root=out_root,
        train_opts=train_opts,
    )

    d_sweep_T = args.d_sweep_T
    point_kw = dict(local_max_qubits=args.local_max_qubits)
    sweep_T = _parse_int_list(args.sweep_T) or list(SWEEP_T)
    sweep_D = _parse_int_list(args.sweep_D) or list(SWEEP_D)
    d_fixed = int(args.d_fixed)
    extra_ks = _parse_int_list(args.extra_ks)
    if not extra_ks and args.extra_k3:
        extra_ks = [3]

    # Build independent sweep jobs, then shard across MPI ranks.
    jobs: list[dict] = []
    final_obar_ks: list[int] = _parse_int_list(getattr(args, "final_obar_ks", None)) or [1, 2, 3, 5, 6]
    if args.final_obar_grid:
        if rank == 0:
            print(
                f"FINAL obar grid (max_T={args.max_T}, ks={final_obar_ks}, "
                f"also_poly={args.also_poly}, target_loss={args.target_loss}, "
                f"n_seeds={args.final_n_seeds})"
            )
        base_jobs = _build_final_obar_jobs(
            local_max_qubits=args.local_max_qubits,
            kernel_mode=args.kernel_mode,
            max_T=int(args.max_T),
            obar_ks=final_obar_ks,
            also_poly=bool(args.also_poly),
        )
        jobs = []
        for bj in base_jobs:
            for si in range(int(args.final_n_seeds)):
                model_seed = int(args.final_seed_base) + si
                jj = dict(bj)
                jj["model_seed"] = model_seed
                jj["base_label"] = bj["label"]
                jj["label"] = f"{bj['label']}_seed{model_seed}"
                jobs.append(jj)
    else:
        if args.only in (None, "T"):
            for T in sweep_T:
                jobs.append(
                    {
                        "series": "T",
                        "T": T,
                        "d": d_fixed,
                        "k": args.k,
                        "label": f"T{T}_d{d_fixed}_k{args.k}",
                    }
                )
            for ek in extra_ks:
                if ek == args.k:
                    continue
                for T in sweep_T:
                    n_q = _qubit_budget(T, d_fixed, ek, kernel_mode=args.kernel_mode)
                    if n_q > args.local_max_qubits:
                        if rank == 0:
                            print(f"  [skip] T={T} k={ek}: n_qubits={n_q} > {args.local_max_qubits}")
                        continue
                    jobs.append(
                        {
                            "series": f"T_k{ek}",
                            "T": T,
                            "d": d_fixed,
                            "k": ek,
                            "label": f"T{T}_d{d_fixed}_k{ek}",
                        }
                    )
        if args.only in (None, "d"):
            for d in sweep_D:
                jobs.append(
                    {
                        "series": "d",
                        "T": d_sweep_T,
                        "d": d,
                        "k": args.k,
                        "label": f"T{d_sweep_T}_d{d}_k{args.k}",
                    }
                )
            if args.extra_k_on_d is not None:
                ek = args.extra_k_on_d
                for d in sweep_D:
                    n_q = _qubit_budget(d_sweep_T, d, ek, kernel_mode=args.kernel_mode)
                    if n_q > args.local_max_qubits:
                        if rank == 0:
                            print(f"  [skip] d={d} k={ek}: n_qubits={n_q} > {args.local_max_qubits}")
                        continue
                    jobs.append(
                        {
                            "series": "d_extra",
                            "T": d_sweep_T,
                            "d": d,
                            "k": ek,
                            "label": f"T{d_sweep_T}_d{d}_k{ek}",
                        }
                    )

        scheduled = {(j["T"], j["d"], j["k"]) for j in jobs}
        panel_ds = _parse_int_list(args.panel_d_on_T)
        panel_Ts = _parse_int_list(args.panel_T_on_d)
        if args.only in (None, "T") and panel_ds:
            for d_panel in panel_ds:
                for T in sweep_T:
                    key = (T, d_panel, args.k)
                    if key in scheduled:
                        continue
                    n_q = _qubit_budget(T, d_panel, args.k, kernel_mode=args.kernel_mode)
                    if n_q > args.local_max_qubits:
                        continue
                    jobs.append(
                        {
                            "series": "T_by_d",
                            "T": T,
                            "d": d_panel,
                            "k": args.k,
                            "label": f"T{T}_d{d_panel}_k{args.k}",
                        }
                    )
                    scheduled.add(key)
        if args.only in (None, "d") and panel_Ts:
            for T_panel in panel_Ts:
                for d in sweep_D:
                    key = (T_panel, d, args.k)
                    if key in scheduled:
                        continue
                    n_q = _qubit_budget(T_panel, d, args.k, kernel_mode=args.kernel_mode)
                    if n_q > args.local_max_qubits:
                        continue
                    jobs.append(
                        {
                            "series": "d_by_T",
                            "T": T_panel,
                            "d": d,
                            "k": args.k,
                            "label": f"T{T_panel}_d{d}_k{args.k}",
                        }
                    )
                    scheduled.add(key)

    my_jobs = shard_items(jobs, rank, size) if args.mpi else jobs
    if rank == 0:
        print(f"\n[MPI] {len(jobs)} sweep points total, {size} ranks "
              f"(~{len(my_jobs)} per rank on rank0)")

    local_rows: list[dict] = []
    for job in my_jobs:
        opts = dict(train_opts)
        opts["kernel_mode"] = job.get("kernel_mode", args.kernel_mode)
        kw = {
            **train_kw,
            "k": job["k"],
            "seed": int(job.get("model_seed", train_kw["seed"])),
            "train_opts": opts,
        }
        try:
            row = _train_point(
                T=job["T"],
                d=job["d"],
                label=job["label"],
                **point_kw,
                **kw,
            )
            row["series"] = job["series"]
            row["kernel_mode"] = opts["kernel_mode"]
            row["model_seed"] = int(job.get("model_seed", kw["seed"]))
            row["base_label"] = job.get("base_label", job["label"])
            local_rows.append(row)
        except Exception as exc:
            print(f"[WARN] sweep point failed {job['label']}: {exc}")
            local_rows.append(
                {
                    "label": job["label"],
                    "series": job["series"],
                    "T": job["T"],
                    "d": job["d"],
                    "k": job["k"],
                    "kernel_mode": opts["kernel_mode"],
                    "model_seed": int(job.get("model_seed", kw["seed"])),
                    "base_label": job.get("base_label", job["label"]),
                    "error": str(exc),
                    "mean_O_ij": float("nan"),
                    "obar": float("nan"),
                    "final_loss": float("nan"),
                    "n_qubits": int(
                        _qubit_budget(
                            job["T"], job["d"], job["k"], kernel_mode=opts["kernel_mode"]
                        )
                    ),
                }
            )

    barrier(comm)
    all_rows = gather_list(comm, local_rows) if args.mpi else local_rows
    if rank != 0:
        barrier(comm)
        return 0

    print("\n" + "=" * 60)
    print("FASE 4/4 — Plot e riassunto")
    print("=" * 60)

    if args.final_obar_grid:
        agg_rows = _aggregate_final_rows_by_seed(all_rows)
        _write_csv(out_root / "summary_mu_seed_agg.csv", agg_rows)
        _plot_final_mu_panels(
            agg_rows, out_root, max_T=int(args.max_T), mu_ks=final_obar_ks
        )
        if args.target_loss is not None:
            bad = [
                r
                for r in agg_rows
                if r.get("final_loss") is not None
                and float(r["final_loss"]) > float(args.target_loss)
            ]
            if bad:
                print("\n[WARN] Punti (media seed) con loss > target_loss (da ritrenare):")
                for r in sorted(bad, key=lambda x: (int(x["T"]), int(x["d"]), int(x["k"]))):
                    print(
                        f"  {r['label']}: loss={float(r['final_loss']):.4f} "
                        f"> {args.target_loss}  (mu={float(r.get('mu_mean', float('nan'))):.4e})"
                    )
        total_wall = time.perf_counter() - t_start
        manifest = {
            "timestamp": stamp,
            "preset": preset_name,
            "final_obar_grid": True,
            "max_T": int(args.max_T),
            "obar_ks": final_obar_ks,
            "also_poly": bool(args.also_poly),
            "final_n_seeds": int(args.final_n_seeds),
            "final_seed_base": int(args.final_seed_base),
            "circuit_mode": "section2",
            "epochs": args.epochs,
            "max_sentences": args.max_sentences,
            "seed": args.seed,
            "self_check_ok": self_check_ok,
            "train_opts": train_opts,
            "total_wall_seconds": total_wall,
            "local_max_qubits": args.local_max_qubits,
            "mpi_ranks": size,
            "all_rows": all_rows,
            "all_rows_seed_agg": agg_rows,
            "metric": "mu",
        }
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _write_summary(
            out_root, preset_name, args.k, args.epochs, args.max_sentences, args.seed,
            all_rows, [], self_check_ok, total_wall,
        )
        print(f"\n[FATTO] Tutto in: {out_root}")
        barrier(comm)
        return 0

    t_rows = [r for r in all_rows if r.get("series") == "T"]
    t_rows_k3 = [r for r in all_rows if r.get("series") == "T_k3"]
    t_sweep_by_k: dict[str, list[dict]] = {}
    if t_rows:
        t_sweep_by_k[str(args.k)] = t_rows
    for r in all_rows:
        s = str(r.get("series", ""))
        if s.startswith("T_k"):
            kk = s.split("T_k", 1)[1]
            t_sweep_by_k.setdefault(kk, []).append(r)
    d_rows = [r for r in all_rows if r.get("series") == "d"]
    d_rows_extra = [r for r in all_rows if r.get("series") == "d_extra"]
    t_by_d_rows = [r for r in all_rows if r.get("series") == "T_by_d"]
    d_by_T_rows = [r for r in all_rows if r.get("series") == "d_by_T"]

    if t_rows or t_rows_k3 or len(t_sweep_by_k) > 1:
        all_t = t_rows + [r for rows in t_sweep_by_k.values() for r in rows]
        by_lab = {r["label"]: r for r in all_t}
        _write_csv(out_root / "summary_vs_T.csv", list(by_lab.values()))
        if len(t_sweep_by_k) > 1:
            t_series = [
                (f"k={kk} (trained)", t_sweep_by_k[kk])
                for kk in sorted(t_sweep_by_k.keys(), key=int)
            ]
        else:
            t_series = [(f"k={args.k} (trained)", t_rows)]
            if t_rows_k3:
                t_series.append(("k=3 (trained)", t_rows_k3))
        _plot_vs_T(t_series, d_fixed=d_fixed, out_path=out_root / "mean_O_vs_T.png")
    if d_rows or d_rows_extra:
        all_d = d_rows + d_rows_extra
        _write_csv(out_root / "summary_vs_d.csv", all_d)
        d_series: list[tuple[str, list[dict]]] = [(f"k={args.k} (trained)", d_rows)]
        if d_rows_extra:
            d_series.append((f"k={args.extra_k_on_d} (trained)", d_rows_extra))
        _plot_vs_d(d_series, T_fixed=d_sweep_T, out_path=out_root / "mean_O_vs_d.png")

    panel_T_src = t_rows + t_by_d_rows
    if panel_T_src:
        _write_csv(out_root / "summary_vs_T_by_d.csv", panel_T_src)
        _plot_vs_T_by_d(panel_T_src, k_fixed=args.k, out_path=out_root / "mean_O_vs_T_by_d.png")
    panel_d_src = d_rows + d_by_T_rows
    if panel_d_src:
        _write_csv(out_root / "summary_vs_d_by_T.csv", panel_d_src)
        _plot_vs_d_by_T(panel_d_src, k_fixed=args.k, out_path=out_root / "mean_O_vs_d_by_T.png")

    if args.target_loss is not None:
        bad = [
            r for r in (d_rows + d_rows_extra + d_by_T_rows + t_rows + t_rows_k3 + t_by_d_rows)
            if r.get("final_loss") is not None
            and float(r["final_loss"]) > float(args.target_loss)
        ]
        if bad and rank == 0:
            print("\n[WARN] Punti con loss > target_loss (da ritrenare):")
            for r in sorted(bad, key=lambda x: (int(x["T"]), int(x["d"]), int(x["k"]))):
                print(
                    f"  {r['label']}: loss={float(r['final_loss']):.4f} "
                    f"> {args.target_loss}  (obar={float(r.get('obar', float('nan'))):.4f})"
                )

    total_wall = time.perf_counter() - t_start
    manifest = {
        "timestamp": stamp,
        "preset": preset_name,
        "circuit_mode": "section2",
        "k": args.k,
        "epochs": args.epochs,
        "max_sentences": args.max_sentences,
        "seed": args.seed,
        "self_check_ok": self_check_ok,
        "train_opts": train_opts,
        "total_wall_seconds": total_wall,
        "d_sweep_T": d_sweep_T,
        "d_fixed": d_fixed,
        "extra_k3": args.extra_k3,
        "extra_ks": extra_ks,
        "extra_k_on_d": args.extra_k_on_d,
        "local_max_qubits": args.local_max_qubits,
        "mpi_ranks": size,
        "T_sweep": t_rows,
        "T_sweep_k3": t_rows_k3,
        "T_sweep_by_k": t_sweep_by_k,
        "d_sweep": d_rows,
        "d_sweep_extra": d_rows_extra,
        "T_by_d": t_by_d_rows,
        "d_by_T": d_by_T_rows,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_summary(
        out_root, preset_name, args.k, args.epochs, args.max_sentences, args.seed,
        t_rows, d_rows, self_check_ok, total_wall,
    )

    print(f"\n[FATTO] Tutto in: {out_root}")
    barrier(comm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
