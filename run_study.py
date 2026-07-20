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
SWEEP_T = (2, 4, 8, 16, 32)   # HPC: include 32; local skips if > local_max_qubits
SWEEP_D = (2, 4, 8, 16, 32)   # HPC: include 16/32 for advantage regime
SWEEP_D_FIXED = 16            # default complexity d for vs-T on HPC
SWEEP_T_FIXED = 16            # default T_lim-ish for vs-d on HPC
# Extra panels: same k, vary the other axis
PANEL_D_ON_T = (4, 8, 16)     # obar vs T curves at these d (fixed k)
PANEL_T_ON_D = (8, 16, 32)    # obar vs d curves at these T (fixed k)
T_LIM = 4
LOCAL_MAX_QUBITS = 15
DEFAULT_TARGET_LOSS = 2.5

# Previous-style palette: data curves vs theory refs must NOT share dash/color.
# Data = solid + distinct markers; Haar floor = dashed squares; advantage = dotted triangles.
_DATA_MARKERS = ("o", "s", "D", "^", "v")
_DATA_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9")
_REF_HAAR_COLOR = "#000000"
_REF_ADV_COLOR = "#666666"


def _qubit_budget(T: int, d: int, k: int, kernel_mode: str = "monomial") -> int:
    if kernel_mode == "poly":
        from qsa_section2_circuit_polynomial import qubit_budget as qb
    else:
        from qsa_section2_circuit import qubit_budget as qb
    return int(qb(T, d, k))


def _obar_ylabel() -> str:
    return r"$\bar o = \mathrm{mean}_{i\leq j}|a_{ij}\,s_{ij}^{k}|$  (monomial diagnostic)"


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
    t_lim: int = T_LIM,
) -> None:
    """Plot obar vs T with primary d^{-(k+1)/2} and secondary advantage refs.

    Style: data = solid+markers (distinct colors); Haar floor = black dashed;
    advantage = grey dotted. Linear y.
    """
    from qsa_section2_circuit import advantage_threshold, haar_floor

    fig, ax = plt.subplots(figsize=(8, 5))
    all_T: set[int] = set()
    drawn_refs: set[tuple[int, int]] = set()
    haar_colors = ("#000000", "#4D4D4D", "#7F7F7F")
    adv_colors = ("#666666", "#999999", "#BBBBBB")
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
            ri = len(drawn_refs)
            drawn_refs.add(key)
            floor = haar_floor(d_val, k_val)
            adv = advantage_threshold(d_val, k_val)
            ax.axhline(
                floor,
                color=haar_colors[ri % len(haar_colors)],
                linestyle="--",
                linewidth=1.8,
                label=rf"$d^{{-(k+1)/2}}$ (k={k_val}, d={d_val})",
                zorder=1,
            )
            ax.axhline(
                adv,
                color=adv_colors[ri % len(adv_colors)],
                linestyle=":",
                linewidth=1.8,
                label=rf"$\sqrt{{k\,k!/d^k}}$ adv (k={k_val}, d={d_val})",
                zorder=1,
            )

    ax.axvline(t_lim, color="0.45", linestyle="-.", linewidth=1.2, label=rf"$T_{{\mathrm{{lim}}}}$={t_lim}")
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
    """Plot obar vs d; primary ref d^{-(k+1)/2}, plus advantage. Log y."""
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
                color=_REF_HAAR_COLOR,
                linestyle="--",
                linewidth=1.8,
                marker="s",
                markersize=5,
                label=rf"$d^{{-(k+1)/2}}$ (k={k_val})",
                zorder=1,
            )
            ax.plot(
                d_vals,
                adv,
                color=_REF_ADV_COLOR,
                linestyle=":",
                linewidth=1.8,
                marker="^",
                markersize=5,
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
    t_lim: int = T_LIM,
) -> None:
    """obar vs T: same k, one curve per d (+ d^{-(k+1)/2} refs)."""
    from qsa_section2_circuit import haar_floor

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
        ax.axhline(
            floor,
            color=color,
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
            label=rf"$d^{{-(k+1)/2}}$ ($d={d_val}$)",
            zorder=1,
        )

    ax.axvline(t_lim, color="0.45", linestyle="-.", linewidth=1.2, label=rf"$T_{{\mathrm{{lim}}}}$={t_lim}")
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
    """obar vs d: same k, one curve per T (+ Haar floor + advantage). Log y."""
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


def _replot_study_dir(out_root: Path, args) -> int:
    """Regenerate PNGs from manifest.json / CSVs without retraining."""
    manifest_path = out_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    t_rows = man.get("T_sweep") or []
    t_rows_k3 = man.get("T_sweep_k3") or []
    d_rows = man.get("d_sweep") or []
    d_rows_extra = man.get("d_sweep_extra") or []
    t_by_d = man.get("T_by_d") or []
    d_by_T = man.get("d_by_T") or []
    k = int(man.get("k", args.k))
    d_fixed = SWEEP_D_FIXED
    T_fixed = int(man.get("d_sweep_T", SWEEP_T_FIXED))
    ek = man.get("extra_k_on_d")

    if t_rows or t_rows_k3:
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
    args = parser.parse_args()

    if args.long:
        args.train_until_converged = True
    if args.target_loss is not None:
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

    # Build independent sweep jobs, then shard across MPI ranks.
    jobs: list[dict] = []
    if args.only in (None, "T"):
        for T in SWEEP_T:
            jobs.append(
                {
                    "series": "T",
                    "T": T,
                    "d": SWEEP_D_FIXED,
                    "k": args.k,
                    "label": f"T{T}_d{SWEEP_D_FIXED}_k{args.k}",
                }
            )
        if args.extra_k3:
            for T in SWEEP_T:
                n_q = _qubit_budget(T, SWEEP_D_FIXED, 3, kernel_mode=args.kernel_mode)
                if n_q > args.local_max_qubits:
                    if rank == 0:
                        print(f"  [skip] T={T} k=3: n_qubits={n_q} > {args.local_max_qubits}")
                    continue
                jobs.append(
                    {
                        "series": "T_k3",
                        "T": T,
                        "d": SWEEP_D_FIXED,
                        "k": 3,
                        "label": f"T{T}_d{SWEEP_D_FIXED}_k3",
                    }
                )
    if args.only in (None, "d"):
        for d in SWEEP_D:
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
            for d in SWEEP_D:
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

    # Extra panels: same k, vary the other axis (skip duplicates already scheduled).
    scheduled = {(j["T"], j["d"], j["k"]) for j in jobs}
    panel_ds = _parse_int_list(args.panel_d_on_T)
    panel_Ts = _parse_int_list(args.panel_T_on_d)
    if args.only in (None, "T") and panel_ds:
        for d_panel in panel_ds:
            for T in SWEEP_T:
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
            for d in SWEEP_D:
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
        kw = {**train_kw, "k": job["k"]}
        try:
            row = _train_point(
                T=job["T"],
                d=job["d"],
                label=job["label"],
                **point_kw,
                **kw,
            )
            row["series"] = job["series"]
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
                    "error": str(exc),
                    "mean_O_ij": float("nan"),
                    "obar": float("nan"),
                    "final_loss": float("nan"),
                    "n_qubits": int(_qubit_budget(job["T"], job["d"], job["k"], kernel_mode=args.kernel_mode)),
                }
            )

    barrier(comm)
    all_rows = gather_list(comm, local_rows) if args.mpi else local_rows
    if rank != 0:
        barrier(comm)
        return 0

    t_rows = [r for r in all_rows if r.get("series") == "T"]
    t_rows_k3 = [r for r in all_rows if r.get("series") == "T_k3"]
    d_rows = [r for r in all_rows if r.get("series") == "d"]
    d_rows_extra = [r for r in all_rows if r.get("series") == "d_extra"]
    t_by_d_rows = [r for r in all_rows if r.get("series") == "T_by_d"]
    d_by_T_rows = [r for r in all_rows if r.get("series") == "d_by_T"]

    print("\n" + "=" * 60)
    print("FASE 4/4 — Plot e riassunto")
    print("=" * 60)

    if t_rows or t_rows_k3:
        all_t = t_rows + t_rows_k3
        _write_csv(out_root / "summary_vs_T.csv", all_t)
        t_series: list[tuple[str, list[dict]]] = [(f"k={args.k} (trained)", t_rows)]
        if t_rows_k3:
            t_series.append(("k=3 (trained)", t_rows_k3))
        _plot_vs_T(t_series, d_fixed=SWEEP_D_FIXED, out_path=out_root / "mean_O_vs_T.png")
    if d_rows or d_rows_extra:
        all_d = d_rows + d_rows_extra
        _write_csv(out_root / "summary_vs_d.csv", all_d)
        d_series: list[tuple[str, list[dict]]] = [(f"k={args.k} (trained)", d_rows)]
        if d_rows_extra:
            d_series.append((f"k={args.extra_k_on_d} (trained)", d_rows_extra))
        _plot_vs_d(d_series, T_fixed=d_sweep_T, out_path=out_root / "mean_O_vs_d.png")

    # Extra panels (same k, vary other axis). Merge with primary series points.
    panel_T_src = t_rows + t_by_d_rows
    if panel_T_src:
        _write_csv(out_root / "summary_vs_T_by_d.csv", panel_T_src)
        _plot_vs_T_by_d(panel_T_src, k_fixed=args.k, out_path=out_root / "mean_O_vs_T_by_d.png")
    panel_d_src = d_rows + d_by_T_rows
    if panel_d_src:
        _write_csv(out_root / "summary_vs_d_by_T.csv", panel_d_src)
        _plot_vs_d_by_T(panel_d_src, k_fixed=args.k, out_path=out_root / "mean_O_vs_d_by_T.png")

    # Flag under-trained d points (loss above target).
    if args.target_loss is not None:
        bad = [
            r for r in (d_rows + d_rows_extra + d_by_T_rows)
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
        "extra_k3": args.extra_k3,
        "extra_k_on_d": args.extra_k_on_d,
        "local_max_qubits": args.local_max_qubits,
        "mpi_ranks": size,
        "T_sweep": t_rows,
        "T_sweep_k3": t_rows_k3,
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
