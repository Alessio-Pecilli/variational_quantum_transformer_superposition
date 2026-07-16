#!/usr/bin/env python3
"""
run_experiments.py — menu unico di test ed esperimenti Section 2.

Dopo il fix della formula classica (s[j,i]=<x_j|W|x_i>, obar=absS/Ntri senza 1/k),
lancia le verifiche e gli studi in ordine logico.

USO RAPIDO
----------
  python run_experiments.py --list              # elenco fasi
  python run_experiments.py --phase verify      # ~30s: circuito + formula
  python run_experiments.py --phase unit        # ~2 min: test automatici
  python run_experiments.py --phase study       # ~15 min: scaling medium (post-fix)
  python run_experiments.py --phase long-d      # sweep d convergente + mitigazioni
  python run_experiments.py --phase mitigations # confronto baseline/warm/curriculum
  python run_experiments.py --phase all         # tutto in sequenza

Esempi singoli:
  python run_experiments.py --phase verify --verbose
  python run_experiments.py --phase study --only d
  python run_experiments.py --phase long-d --no-curriculum-k
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "experiments"


PHASES = {
    "verify": "Self-check circuito + diagnose per-coppia + demo formula vecchia/nuova",
    "unit": "Batteria run_all_tests.py (11-13 test automatici)",
    "study": "Sweep O_ij vs T e vs d (preset medium: 20 ep, 32 frasi)",
    "long-d": "Sweep vs d fino a convergenza loss (+ warm-start W, curriculum k)",
    "long-T": "Sweep vs T fino a convergenza loss (+ mitigazioni)",
    "mitigations": "Confronto baseline / warm-start / curriculum su T2_d4_k2",
    "all": "verify -> unit -> study -> mitigations -> long-d",
}


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log(msg: str) -> None:
    print(msg, flush=True)


def _banner(title: str) -> None:
    _log("\n" + "=" * 64)
    _log(title)
    _log("=" * 64)


def _run_py(args: list[str], label: str) -> int:
    cmd = [sys.executable] + args
    _log(f"\n>>> {label}")
    _log(f"    {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT))
    dt = time.perf_counter() - t0
    _log(f"    exit={proc.returncode}  ({dt:.1f}s)")
    return proc.returncode


# --------------------------------------------------------------------------- #
#  FASE verify — circuito e formula                                           #
# --------------------------------------------------------------------------- #
def phase_verify(verbose: bool) -> dict:
    _banner("FASE verify — circuito e formula classica")
    from qsa_section2_circuit import (
        diagnose_analytic_mismatch,
        random_instance,
        self_check,
        set_backends,
        _pair_weights,
        make_qsa_state_qnode,
        overlap_indices,
        mu_overlap,
        real_ortho_block,
    )
    import numpy as np
    import pennylane as qml
    import jax.numpy as jnp

    set_backends()
    results = {"self_checks": [], "diagnose": {}, "formula_demo": {}}

    for T, d, k in [(2, 2, 1), (4, 4, 2), (8, 4, 3)]:
        ok = self_check(T=T, d=d, k=k, check_projector=(k <= 2))
        results["self_checks"].append({"T": T, "d": d, "k": k, "pass": ok})
        if not ok:
            raise RuntimeError(f"self_check failed T={T} d={d} k={k}")

    diag = diagnose_analytic_mismatch(T=4, d=4, k=2, seed=0, tol=1e-6)
    results["diagnose"] = diag
    _log(f"\nDiagnose T=4 d=4 k=2:")
    _log(f"  mu_overlap / mu_analytic ratio = {diag['mu_ratio']:.6f}")
    _log(f"  i=j max rel err  = {diag['pairs_i_eq_j_max_rel_err']:.2e}")
    _log(f"  i!=j max rel err = {diag['pairs_i_ne_j_max_rel_err']:.2e}")
    if not diag["pairs_i_ne_j_ok"]:
        raise RuntimeError("i!=j pairs still mismatch — formula fix incomplete")

    # Demo: correct s[j,i]=x_j^T W x_i vs wrong transposed indexing
    T, d, k = 4, 4, 2
    X, Y, Wp, Vp = random_instance(T, d, k, 2, seed=0, aligned_targets=True)
    n = 2
    Wmat = np.real(
        qml.matrix(real_ortho_block, wire_order=range(n))(qml, jnp.array(Wp), range(n))
    )[:d, :d]
    Vmat = np.real(
        qml.matrix(real_ortho_block, wire_order=range(n))(qml, jnp.array(Vp), range(n))
    )[:d, :d]
    w_new = _pair_weights(X, Y, Wmat, Vmat, k)
    # wrong vectorized form that used (X @ W.T) @ X.T -> <x_i|W|x_j>
    w_wrong = (Y @ Vmat @ X.T) * ((X @ Wmat.T @ X.T) ** k)

    circ, _ = make_qsa_state_qnode(T, d, k, 2)
    st = circ(jnp.array(X), jnp.array(Y), jnp.array(Wp), jnp.array(Vp))
    ids, N = overlap_indices(T, d, k)
    mu_circ = mu_overlap(st, ids, N)

    pairs = [(j, i) for j in range(T) for i in range(j + 1)]
    err_wrong, err_new = [], []
    for idx, (j, i) in enumerate(pairs):
        amp = st[ids[idx]]
        exp_wrong = w_wrong[j, i] / N
        exp_new = w_new[j, i] / N
        if abs(exp_new) > 1e-14:
            err_new.append(abs(amp / exp_new - 1))
        if abs(exp_wrong) > 1e-14:
            err_wrong.append(abs(amp / exp_wrong - 1))

    results["formula_demo"] = {
        "max_rel_err_wrong_indexing": float(max(err_wrong) if err_wrong else 0),
        "max_rel_err_correct_s_ji": float(max(err_new) if err_new else 0),
        "mu_circuit": float(mu_circ),
    }
    _log(f"\nDemo formula (T=4,d=4,k=2):")
    _log(f"  max rel err INDEXING SBAGLIATO <x_i|W|x_j> : {results['formula_demo']['max_rel_err_wrong_indexing']:.4f}")
    _log(f"  max rel err CORRETTO <x_j|W|x_i>           : {results['formula_demo']['max_rel_err_correct_s_ji']:.2e}")
    _log(f"  mu circuito = {mu_circ:.6e}")

    if verbose:
        _log(f"\n  {diag['explanation']}")

    _log("\n[verify] OK — circuito e formula allineati.")
    return results


# --------------------------------------------------------------------------- #
#  FASE unit — test automatici                                                #
# --------------------------------------------------------------------------- #
def phase_unit(fast: bool) -> int:
    _banner("FASE unit — test automatici")
    args = [str(ROOT / "run_all_tests.py")]
    if fast:
        args.append("--fast")
    args.append("-v")
    code = _run_py(args, "run_all_tests.py")
    if code != 0:
        raise RuntimeError("run_all_tests.py failed")
    return code


# --------------------------------------------------------------------------- #
#  FASE study — scaling medium                                                #
# --------------------------------------------------------------------------- #
def phase_study(only: str | None, skip_self_check: bool) -> int:
    _banner("FASE study — scaling medium (20 ep, 32 frasi)")
    args = [str(ROOT / "run_study.py"), "--epochs", "20", "--max-sentences", "32"]
    if only:
        args.extend(["--only", only])
    if skip_self_check:
        args.append("--skip-self-check")
    code = _run_py(args, "run_study.py medium")
    if code != 0:
        raise RuntimeError("run_study.py failed")
    return code


# --------------------------------------------------------------------------- #
#  FASE long-d / long-T — convergenza                                         #
# --------------------------------------------------------------------------- #
def phase_long(
    sweep: str,
    curriculum_k: bool,
    warm_start_w: bool,
    max_epochs: int,
) -> int:
    _banner(f"FASE long-{sweep} — convergenza (max_epochs={max_epochs})")
    args = [
        str(ROOT / "run_study.py"),
        "--only", sweep,
        "--max-sentences", "64",
        "--train-until-converged",
        "--max-epochs", str(max_epochs),
        "--loss-rel-tol", "1e-4",
        "--convergence-patience", "8",
        "--skip-self-check",
    ]
    if curriculum_k:
        args.append("--curriculum-k")
    if warm_start_w:
        args.append("--warm-start-w")
    code = _run_py(args, f"run_study.py long-{sweep}")
    if code != 0:
        raise RuntimeError(f"long-{sweep} failed")
    return code


# --------------------------------------------------------------------------- #
#  FASE mitigations — confronto singolo punto                                 #
# --------------------------------------------------------------------------- #
def phase_mitigations(epochs: int = 30, max_sentences: int = 32) -> dict:
    _banner("FASE mitigations — T=2 d=4 k=2 (baseline vs warm-start vs curriculum)")
    from config import DATASET_CONFIG, OPTIMIZATION_CONFIG
    from jax_training_pipeline import run_training

    logger = logging.getLogger("vqt_jax")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)

    stamp = _stamp()
    out_root = OUT / f"mitigations_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    variants = [
        ("baseline", {}),
        ("warm_start_w", {"warm_start_w_identity": True}),
        ("curriculum_k", {"curriculum_k": True}),
        ("both", {"warm_start_w_identity": True, "curriculum_k": True}),
    ]

    rows = []
    for label, extra in variants:
        cfg = dict(OPTIMIZATION_CONFIG)
        cfg.update(
            {
                "circuit_mode": "section2",
                "embedding_dim": 4,
                "non_linear_order": 2,
                "num_layers": 2,
                "epochs": epochs,
                "learning_rate": 1e-3,
                "seed": 42,
                "run_label": label,
                "output_dir": str(out_root / label),
                **extra,
            }
        )
        DATASET_CONFIG["sentence_length"] = 2
        DATASET_CONFIG["max_sentences"] = max_sentences

        _log(f"\n--- {label} ---")
        t0 = time.perf_counter()
        code = run_training(logger=logger, cfg=cfg, comm=None, rank=0, size=1)
        wall = time.perf_counter() - t0
        if code != 0:
            raise RuntimeError(f"mitigation run {label} failed")

        metrics = json.loads((out_root / label / "summaries" / "metrics.json").read_text())
        row = {
            "label": label,
            "mean_O_ij": metrics["mean_O_ij"],
            "final_loss": metrics["final_loss"],
            "reference_haar": metrics["reference_haar"],
            "reference_sqrt_k_over_d": metrics["reference_sqrt_k_over_d"],
            "wall_seconds": wall,
            "phases": metrics.get("training_phases"),
        }
        rows.append(row)
        _log(
            f"  loss={row['final_loss']:.4f}  O_ij={row['mean_O_ij']:.4f}  "
            f"ref_sqrt(k/d)={row['reference_sqrt_k_over_d']:.4f}  ({wall:.1f}s)"
        )

    summary_path = out_root / "comparison.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _log(f"\n[mitigations] Risultati in {out_root}")
    _log(f"  comparison.json scritto.")
    return {"out_dir": str(out_root), "rows": rows}


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Menu test ed esperimenti Section 2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Fasi disponibili:\n" + "\n".join(f"  {k:12} {v}" for k, v in PHASES.items()),
    )
    parser.add_argument(
        "--phase",
        choices=list(PHASES.keys()),
        default="verify",
        help="fase da eseguire (default: verify)",
    )
    parser.add_argument("--list", action="store_true", help="elenca le fasi e esci")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true", help="unit: salta test lenti")
    parser.add_argument("--only", choices=("T", "d"), default=None, help="study/long: solo uno sweep")
    parser.add_argument("--skip-self-check", action="store_true")
    parser.add_argument("--no-curriculum-k", action="store_true")
    parser.add_argument("--no-warm-start-w", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=300)
    args = parser.parse_args()

    if args.list:
        _log("Fasi disponibili in run_experiments.py:\n")
        for k, v in PHASES.items():
            _log(f"  {k:12}  {v}")
        _log("\nEsempi:")
        _log("  python run_experiments.py --phase verify")
        _log("  python run_experiments.py --phase study")
        _log("  python run_experiments.py --phase long-d")
        _log("  python run_experiments.py --phase all")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"timestamp": _stamp(), "phase": args.phase, "steps": []}
    t_start = time.perf_counter()

    try:
        if args.phase == "verify":
            report["verify"] = phase_verify(args.verbose)

        elif args.phase == "unit":
            phase_unit(args.fast)

        elif args.phase == "study":
            phase_study(args.only, args.skip_self_check)

        elif args.phase == "long-d":
            phase_long(
                "d",
                curriculum_k=not args.no_curriculum_k,
                warm_start_w=not args.no_warm_start_w,
                max_epochs=args.max_epochs,
            )

        elif args.phase == "long-T":
            phase_long(
                "T",
                curriculum_k=not args.no_curriculum_k,
                warm_start_w=not args.no_warm_start_w,
                max_epochs=args.max_epochs,
            )

        elif args.phase == "mitigations":
            report["mitigations"] = phase_mitigations()

        elif args.phase == "all":
            report["verify"] = phase_verify(args.verbose)
            phase_unit(fast=True)
            phase_study(only=None, skip_self_check=True)
            report["mitigations"] = phase_mitigations(epochs=20, max_sentences=32)
            phase_long(
                "d",
                curriculum_k=True,
                warm_start_w=True,
                max_epochs=args.max_epochs,
            )

    except Exception as exc:
        _log(f"\n[ERRORE] {exc}")
        report["error"] = str(exc)
        report_path = OUT / f"report_{report['timestamp']}_FAIL.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        _log(f"Report parziale: {report_path}")
        return 1

    report["total_seconds"] = time.perf_counter() - t_start
    report_path = OUT / f"report_{report['timestamp']}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    _banner(f"FATTO — phase={args.phase}  ({report['total_seconds']:.1f}s)")
    _log(f"Report: {report_path}")
    _log("\nOutput utili:")
    _log("  results/experiments/          report JSON + mitigations")
    _log("  results/study/<timestamp>/    sweep + RIASSUNTO.txt + plot")
    _log("  results/test_suite/           unit test reports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
