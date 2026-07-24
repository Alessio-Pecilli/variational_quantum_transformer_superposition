#!/usr/bin/env python3
"""Preflight checks before launching FINAL HPC campaigns.

Exit 0 = safe to launch (warnings allowed).
Exit 2 = hard failure (do not launch).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _ptb_counts() -> Counter:
    c: Counter = Counter()
    path = ROOT / "ptb_sentences.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            n = len(line.split())
            if n:
                c[n] += 1
    return c


def main() -> int:
    errors: list[str] = []
    warns: list[str] = []
    oks: list[str] = []

    required = [
        "run_final_loss.py",
        "run_study.py",
        "classical_baselines.py",
        "hpc_final_loss.sh",
        "hpc_final_obar.sh",
        "hpc_env.sh",
        "mpi_runtime.py",
        "ptb_sentences.txt",
        "qsa_training.py",
        "qsa_section2_circuit.py",
        "qsa_section2_circuit_polynomial.py",
    ]
    for name in required:
        if (ROOT / name).exists():
            oks.append(f"file {name}")
        else:
            errors.append(f"MISSING {name}")

    # Imports
    try:
        import run_final_loss as fl
        import run_study as rs
        from classical_baselines import (
            kcsa_matrix_param_count,
            nl_model_param_count,
            nl_rank_for_budget,
            qsa_angle_param_count,
        )
        from qsa_section2_circuit import qubit_budget as qb
        from qsa_section2_circuit_polynomial import qubit_budget as qbp
        from mpi_runtime import init_mpi, shard_items  # noqa: F401
        oks.append("python imports")
    except Exception as exc:
        errors.append(f"import failure: {exc}")
        _report(oks, warns, errors)
        return 2

    # Param budgets
    d = 8
    if qsa_angle_param_count(d, 16) != 96:
        errors.append("QSA L=16 angles != 96 at d=8")
    else:
        oks.append("QSA L=16 → 96 angles")
    if kcsa_matrix_param_count(d) != 128:
        errors.append("CSA matrices != 128 at d=8")
    else:
        oks.append("CSA → 128 matrices")
    r = nl_rank_for_budget(d, 1, 128)
    n128 = nl_model_param_count(d, 1, r)
    if n128 > 160 or n128 < 96:
        warns.append(f"nl ~128 budget resolved to {n128} params (L=1,r={r})")
    else:
        oks.append(f"nl ~128 → {n128} params (L=1,r={r})")
    if nl_model_param_count(d, 2, 3) != 288:
        warns.append("nl full not exactly 288")
    else:
        oks.append("nl ~288 → 288 params")

    if rs.FINAL_TARGET_LOSS != 3.8:
        warns.append(f"FINAL_TARGET_LOSS={rs.FINAL_TARGET_LOSS} (expected 3.8)")
    else:
        oks.append("obar target_loss=3.8")

    # Plot style invariants
    import inspect

    if "axvline" in inspect.getsource(rs._plot_vs_T):
        errors.append("T_lim axvline still present in _plot_vs_T")
    else:
        oks.append("no T_lim vertical on vs-T")
    if "axvline" in inspect.getsource(rs._plot_vs_T_by_d):
        errors.append("T_lim axvline still present in _plot_vs_T_by_d")
    else:
        oks.append("no T_lim vertical on vs-T-by-d")

    # Job grids
    ks = (1, 2, 3, 5, 6)
    j = rs._build_final_obar_jobs(48, "monomial", 32, obar_ks=ks, also_poly=False)
    jp = rs._build_final_obar_jobs(48, "monomial", 32, obar_ks=ks, also_poly=True)
    j64 = rs._build_final_obar_jobs(48, "monomial", 64, obar_ks=ks, also_poly=False)
    if len({x["label"] for x in j}) != len(j):
        errors.append("duplicate labels in obar mono grid")
    if len({x["label"] for x in jp}) != len(jp):
        errors.append("duplicate labels in obar mono+poly grid")
    vsT_ks = sorted({x["k"] for x in j if x["d"] == 16})
    if vsT_ks != list(ks):
        errors.append(f"vs-T ks={vsT_ks} expected {list(ks)}")
    else:
        oks.append(f"obar vs-T ks={vsT_ks}")
    oks.append(f"obar jobs MAX_T=32: mono={len(j)}, +poly={len(jp)}; MAX_T=64 mono={len(j64)}")

    max_q = max(qb(x["T"], x["d"], x["k"]) for x in j64)
    max_qp = max(
        (qbp if x["kernel_mode"] == "poly" else qb)(x["T"], x["d"], x["k"])
        for x in rs._build_final_obar_jobs(48, "monomial", 64, obar_ks=ks, also_poly=True)
    )
    if max_q > 48 or max_qp > 48:
        errors.append(f"scheduled qubits exceed 48 (mono max={max_q}, poly max={max_qp})")
    else:
        oks.append(f"qubit budget OK (mono max={max_q}, poly max={max_qp} ≤ 48)")

    # Loss campaign job count
    class A:
        pass

    a = A()
    a.d = 8
    a.nl_rank_full = None
    a.nl_layers_full = 2
    a.nl_layers_small = 1
    a.nl_rank_small = None
    a.nl_param_budget_small = 128
    a.qsa_layers = 16
    a.epochs = 400
    a.poly_epochs = 600
    a.nl_epochs = 500
    a.nl_epochs_general = 800
    a.learning_rate = 1e-3
    a.nl_learning_rate = 5e-3
    a.nl_learning_rate_general = 8e-3
    specs = fl._model_specs(a)
    if len(specs) != 7:
        errors.append(f"expected 7 final-loss models, got {len(specs)}")
    else:
        oks.append("7 final-loss models (mono+poly+3 nl)")
    n_seeds = 8
    mu = sum(1 for s in specs if s["family"] != "nl")
    nl = sum(1 for s in specs if s["family"] == "nl")
    n_jobs = mu * len(ks) * n_seeds + nl * n_seeds
    oks.append(f"final_loss jobs @8 seeds / ks={list(ks)}: {n_jobs}")
    if n_jobs > 250:
        warns.append(f"final_loss is heavy ({n_jobs} jobs); 48h walltime recommended")

    # PTB data
    counts = _ptb_counts()
    for T in (4, 8, 16, 32, 64):
        n = counts.get(T, 0)
        if T <= 32 and n < 64:
            warns.append(f"PTB T={T}: only {n} unique sentences (<64)")
        elif T == 64 and n < 32:
            warns.append(
                f"PTB T=64: ONLY {n} unique sentences — results at T=64 are not reliable "
                f"(sentences get duplicated). Prefer MAX_T=32 for FINAL obar."
            )
        else:
            oks.append(f"PTB T={T}: {n} sentences")

    # Styles present
    for name in (
        "k-QSA L=16",
        "k-CSA",
        "poly-k-QSA L=16",
        "poly-k-CSA",
        "nl-CSA iso ~288",
        "nl-CSA iso ~128",
        "nl-CSA gen ~128",
    ):
        if name not in fl.STYLES:
            errors.append(f"missing style for {name}")
    else:
        oks.append("FINAL loss styles complete")

    return _report(oks, warns, errors)


def _report(oks: list[str], warns: list[str], errors: list[str]) -> int:
    print("=" * 60)
    print("PREFLIGHT FINAL CAMPAIGNS")
    print("=" * 60)
    for x in oks:
        print(f"  [OK]   {x}")
    for x in warns:
        print(f"  [WARN] {x}")
    for x in errors:
        print(f"  [FAIL] {x}")
    print("-" * 60)
    if errors:
        print("RESULT: NO-GO — fix failures before sbatch")
        return 2
    if warns:
        print("RESULT: GO WITH WARNINGS — read WARNs carefully")
        return 0
    print("RESULT: GO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
