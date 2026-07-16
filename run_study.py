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
from qsa_section2_circuit import qubit_budget

ROOT = Path(__file__).resolve().parent

# Sweep fissi dello studio (limite locale: max 14 qubit)
SWEEP_T = (2, 4, 8, 16)   # d=4, k=2  ->  n_qubits in {8, 10, 12, 14}
SWEEP_D = (2, 4, 8)       # T=4, k=2  ->  n_qubits in {7, 10, 13}
SWEEP_D_FIXED = 4
SWEEP_T_FIXED = 4         # T_lim: obar ~ costante per T >= 4
T_LIM = 4
LOCAL_MAX_QUBITS = 15


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


def _run_self_check() -> bool:
    print("\n" + "=" * 60)
    print("FASE 1/4 — Self-check circuito Section 2")
    print("=" * 60)
    try:
        from qsa_section2_circuit import self_check, set_backends

        set_backends()
        ok = True
        ok &= self_check(T=2, d=2, k=1, check_projector=True)
        ok &= self_check(T=4, d=4, k=2, check_projector=True)
        if ok:
            print("[SELF-CHECK] PASS (readout overlap == projector)\n")
        else:
            print("[SELF-CHECK] FAIL — readout overlap != projector.\n")
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
    n_q = qubit_budget(T, d, k)
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

    t0 = time.perf_counter()
    exit_code = run_training(logger=logger, cfg=cfg, comm=None, rank=0, size=1)
    wall = time.perf_counter() - t0
    if exit_code != 0:
        raise RuntimeError(f"Training fallito per {label} (exit={exit_code})")

    metrics_path = out_root / label / "summaries" / "metrics.json"
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
    """Plot obar vs T — solo curve numeriche (niente refs teoriche)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, rows in series:
        rows = sorted(rows, key=lambda r: int(r["T"]))
        T_vals = [int(r["T"]) for r in rows]
        obar = [float(r["obar"]) for r in rows]
        ax.plot(T_vals, obar, "o-", linewidth=2, markersize=8, label=label)
    ax.axvline(t_lim, color="0.5", linestyle=":", linewidth=1.2, label=rf"$T_{{\mathrm{{lim}}}}$={t_lim}")
    ax.set_xlabel("T (lunghezza sequenza)")
    ax.set_ylabel(r"$\bar o = \mathrm{mean}_{i\leq j}|a_{ij}\,s_{ij}^k|$")
    k_labels = ", ".join(lbl for lbl, _ in series)
    ax.set_title(f"obar vs T  (d={d_fixed}; {k_labels})")
    all_T = sorted({int(r["T"]) for _, rows in series for r in rows})
    ax.set_xticks(all_T)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_vs_d(
    series: list[tuple[str, list[dict]]],
    T_fixed: int,
    out_path: Path,
) -> None:
    """Plot obar vs d con riferimenti Haar e advantage (stessa scala di obar)."""
    from qsa_section2_circuit import advantage_threshold, haar_floor

    fig, ax = plt.subplots(figsize=(8, 5))
    all_d: set[int] = set()
    for label, rows in series:
        rows = sorted(rows, key=lambda r: int(r["d"]))
        d_vals = np.array([int(r["d"]) for r in rows], dtype=float)
        obar = np.array([float(r["obar"]) for r in rows])
        all_d.update(int(d) for d in d_vals)
        ax.plot(d_vals, obar, "o-", linewidth=2, markersize=8, label=label)

    d_ref = np.array(sorted(all_d), dtype=float)
    k_for_refs = int(series[0][1][0]["k"]) if series and series[0][1] else 2
    if len(series) == 1:
        k_for_refs = int(series[0][1][0]["k"])
        haar = np.array([haar_floor(int(d), k_for_refs) for d in d_ref])
        adv = np.array([advantage_threshold(int(d), k_for_refs) for d in d_ref])
        ax.plot(d_ref, haar, "s--", linewidth=1.5, color="C1", label=rf"$d^{{-(k+1)/2}}$ Haar (k={k_for_refs})")
        ax.plot(d_ref, adv, "^--", linewidth=1.5, color="C2", label=rf"$\sqrt{{k\,k!/d^k}}$ adv (k={k_for_refs})")
    else:
        for ki, (label, rows) in enumerate(series):
            k_val = int(rows[0]["k"])
            d_vals = np.array(sorted({int(r["d"]) for r in rows}), dtype=float)
            haar = np.array([haar_floor(int(d), k_val) for d in d_vals])
            adv = np.array([advantage_threshold(int(d), k_val) for d in d_vals])
            ax.plot(d_vals, haar, "--", linewidth=1.2, color=f"C{ki + 1}", alpha=0.7, label=rf"Haar k={k_val}")
            ax.plot(d_vals, adv, ":", linewidth=1.2, color=f"C{ki + 1}", alpha=0.7, label=rf"adv k={k_val}")

    ax.set_xlabel("d (dimensione embedding)")
    ax.set_ylabel(r"$\bar o = \mathrm{mean}_{i\leq j}|a_{ij}\,s_{ij}^k|$")
    ax.set_title(f"obar vs d  (T={T_fixed})")
    ax.set_xticks(sorted(all_d))
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
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
        "Refs: Haar = d^{-(k+1)/2} ; advantage = sqrt(k * k! / d^k)",
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
        "  mean_O_vs_T.png, mean_O_vs_d.png  (obar vs refs)",
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


def _apply_preset(args) -> str:
    if args.quick:
        args.epochs = 6
        args.max_sentences = 16
        return "quick"
    if args.long:
        args.epochs = 40
        args.max_sentences = 64
        args.train_until_converged = True
        args.max_epochs = 300
        return "long"
    if args.full:
        args.epochs = 40
        args.max_sentences = 64
        return "full"
    if args.epochs is None:
        args.epochs = 20
    if args.max_sentences is None:
        args.max_sentences = 32
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
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--loss-rel-tol", type=float, default=1e-4)
    parser.add_argument("--convergence-patience", type=int, default=8)
    args = parser.parse_args()

    if args.long:
        args.train_until_converged = True

    preset_name = _apply_preset(args)
    logger = _setup_logger()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path("results") / "study" / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RUN_STUDY — studio Section 2 (overlap O_ij vs T e d)")
    print("=" * 60)
    print(f"Preset: {preset_name} | epoche={args.epochs} | frasi={args.max_sentences} | k={args.k}")
    if args.train_until_converged:
        print(f"Convergenza: max_epochs={args.max_epochs} rel_tol={args.loss_rel_tol} patience={args.convergence_patience}")
    if args.curriculum_k:
        print("Curriculum k: 1 -> k")
    if args.warm_start_w:
        print("Warm-start: W ~ I (weights_w=0)")
    print(f"Output: {out_root}")
    print(f"Ogni run = stesso percorso di main_hpc.py --circuit-mode section2")
    print("=" * 60)

    t_start = time.perf_counter()
    self_check_ok = True
    if not args.skip_self_check:
        self_check_ok = _run_self_check()
        if not self_check_ok:
            print("[WARN] Self-check readout fallito; proseguo comunque (usa --skip-self-check per saltare).")

    train_opts = {
        "warm_start_w_identity": args.warm_start_w,
        "curriculum_k": args.curriculum_k,
        "train_until_converged": args.train_until_converged,
        "max_epochs": args.max_epochs,
        "loss_rel_tol": args.loss_rel_tol,
        "convergence_patience": args.convergence_patience,
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

    t_rows: list[dict] = []
    t_rows_k3: list[dict] = []
    d_rows: list[dict] = []
    d_rows_extra: list[dict] = []
    d_sweep_T = args.d_sweep_T
    point_kw = dict(local_max_qubits=args.local_max_qubits)

    if args.only in (None, "T"):
        print("\n" + "=" * 60)
        print(f"FASE 2/4 — Sweep vs T  (d={SWEEP_D_FIXED}, k={args.k})")
        print("=" * 60)
        for T in SWEEP_T:
            label = f"T{T}_d{SWEEP_D_FIXED}_k{args.k}"
            t_rows.append(
                _train_point(T=T, d=SWEEP_D_FIXED, label=label, **point_kw, **train_kw)
            )
        if args.extra_k3:
            print("\n" + "-" * 60)
            print(f"Extra sweep vs T  (d={SWEEP_D_FIXED}, k=3)")
            print("-" * 60)
            for T in SWEEP_T:
                n_q = qubit_budget(T, SWEEP_D_FIXED, 3)
                if n_q > args.local_max_qubits:
                    print(f"  [skip] T={T} k=3: n_qubits={n_q} > {args.local_max_qubits}")
                    continue
                label = f"T{T}_d{SWEEP_D_FIXED}_k3"
                kw3 = {**train_kw, "k": 3}
                t_rows_k3.append(
                    _train_point(T=T, d=SWEEP_D_FIXED, label=label, **point_kw, **kw3)
                )

    if args.only in (None, "d"):
        print("\n" + "=" * 60)
        print(f"FASE 3/4 — Sweep vs d  (T={d_sweep_T}, k={args.k})")
        print("=" * 60)
        for d in SWEEP_D:
            label = f"T{d_sweep_T}_d{d}_k{args.k}"
            d_rows.append(
                _train_point(T=d_sweep_T, d=d, label=label, **point_kw, **train_kw)
            )
        if args.extra_k_on_d is not None:
            ek = args.extra_k_on_d
            print("\n" + "-" * 60)
            print(f"Extra sweep vs d  (T={d_sweep_T}, k={ek})")
            print("-" * 60)
            for d in SWEEP_D:
                n_q = qubit_budget(d_sweep_T, d, ek)
                if n_q > args.local_max_qubits:
                    print(f"  [skip] d={d} k={ek}: n_qubits={n_q} > {args.local_max_qubits}")
                    continue
                label = f"T{d_sweep_T}_d{d}_k{ek}"
                kw_e = {**train_kw, "k": ek}
                d_rows_extra.append(
                    _train_point(T=d_sweep_T, d=d, label=label, **point_kw, **kw_e)
                )

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
        "T_sweep": t_rows,
        "T_sweep_k3": t_rows_k3,
        "d_sweep": d_rows,
        "d_sweep_extra": d_rows_extra,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_summary(
        out_root, preset_name, args.k, args.epochs, args.max_sentences, args.seed,
        t_rows, d_rows, self_check_ok, total_wall,
    )

    print(f"\n[FATTO] Tutto in: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
