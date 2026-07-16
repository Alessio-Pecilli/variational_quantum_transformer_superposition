#!/usr/bin/env python3
"""
run_all_tests.py — batteria completa di test per Section 2.

Esegue in sequenza:
  1. Circuito: self-check, diagnose analitico, formula classica
  2. Training: baseline, warm-start W, curriculum k, convergenza
  3. Pipeline: run_training (section2), artifacti su disco
  4. Smoke: un punto dello sweep + import run_study

Uso:
  python run_all_tests.py           # tutti i test (~2-5 min)
  python run_all_tests.py --fast    # salta test lenti (convergenza, curriculum)
  python run_all_tests.py -v        # output verboso
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "test_suite"


@dataclass
class TestResult:
    name: str
    passed: bool
    seconds: float
    detail: str = ""


@dataclass
class SuiteReport:
    results: List[TestResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, seconds: float, detail: str = "") -> None:
        self.results.append(TestResult(name, passed, seconds, detail))

    @property
    def ok(self) -> bool:
        return all(r.passed for r in self.results)

    def write(self, path: Path) -> None:
        lines = [
            "TEST SUITE REPORT",
            f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
            f"passed: {sum(r.passed for r in self.results)}/{len(self.results)}",
            "",
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"[{status}] {r.name} ({r.seconds:.2f}s)")
            if r.detail:
                lines.append(f"       {r.detail}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _logger(verbose: bool) -> logging.Logger:
    log = logging.getLogger("test_suite")
    log.setLevel(logging.DEBUG if verbose else logging.WARNING)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    return log


def _run_test(report: SuiteReport, name: str, fn: Callable[[], str], verbose: bool) -> None:
    t0 = time.perf_counter()
    detail = ""
    try:
        detail = fn() or ""
        passed = True
        mark = "PASS"
    except Exception as exc:
        passed = False
        detail = f"{type(exc).__name__}: {exc}"
        mark = "FAIL"
        if verbose:
            traceback.print_exc()
    dt = time.perf_counter() - t0
    report.add(name, passed, dt, detail)
    print(f"  [{mark}] {name} ({dt:.2f}s)" + (f" — {detail}" if detail and (verbose or not passed) else ""))


# --------------------------------------------------------------------------- #
#  1. Circuit tests                                                            #
# --------------------------------------------------------------------------- #
def _test_imports() -> str:
    from qsa_section2_circuit import (  # noqa: F401
        classical_report,
        diagnose_analytic_mismatch,
        mean_O_ij,
        qubit_budget,
        self_check,
        set_backends,
    )
    from qsa_training import QSATrainConfig, train_qsa  # noqa: F401
    from jax_training_pipeline import run_training  # noqa: F401
    return "imports ok"


def _test_classical_formula() -> str:
    from qsa_section2_circuit import classical_report, mean_O_ij, _pair_weights

    T, d, k = 3, 4, 2
    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = np.roll(X, -1, axis=0)
    Y[-1] = X[-1]
    W = np.eye(d)
    V = np.eye(d)
    w = _pair_weights(X, Y, W, V, k)
    mask = np.tril(np.ones((T, T)))
    manual = float(np.sum(np.abs(w * mask)) / (T * (T + 1) / 2))
    rep = classical_report(X, Y, W, V, k)
    got = mean_O_ij(X, Y, W, V, k)
    if abs(manual - got) > 1e-12 or abs(manual - rep["mean_O_ij"]) > 1e-12:
        raise AssertionError(f"mean_O_ij mismatch: {manual} vs {got} vs {rep['mean_O_ij']}")
    if abs(rep["obar"] - rep["mean_O_ij"]) > 1e-12:
        raise AssertionError("obar must equal mean_O_ij (raw absS/Ntri, no 1/k)")
    expected_root = rep["mean_O_ij"] ** (1 / k)
    if abs(rep["obar_k_root"] - expected_root) > 1e-12:
        raise AssertionError(f"obar_k_root mismatch: {rep['obar_k_root']} vs {expected_root}")
    if rep["mu"] <= 0:
        raise AssertionError("mu must be positive for identity W,V")
    return f"mean_O_ij={got:.4f} mu={rep['mu']:.4f}"


def _test_qubit_budget() -> str:
    from qsa_section2_circuit import qubit_budget

    cases = {
        (2, 2, 1): 4,
        (4, 4, 2): 10,
        (16, 4, 2): 14,
        (2, 8, 2): 11,
    }
    for (T, d, k), expected in cases.items():
        got = qubit_budget(T, d, k)
        if got != expected:
            raise AssertionError(f"qubit_budget({T},{d},{k})={got}, expected {expected}")
    return f"{len(cases)} budget cases ok"


def _test_self_check_readout() -> str:
    from qsa_section2_circuit import self_check, set_backends

    set_backends()
    if not self_check(T=2, d=2, k=1, check_projector=True):
        raise AssertionError("self_check T=2 d=2 k=1 failed readout")
    if not self_check(T=4, d=4, k=2, check_projector=True):
        raise AssertionError("self_check T=4 d=4 k=2 failed readout")
    return "overlap == projector on 2 configs"


def _test_diagnose_analytic() -> str:
    from qsa_section2_circuit import diagnose_analytic_mismatch, set_backends

    set_backends()
    d = diagnose_analytic_mismatch(T=4, d=4, k=2, seed=0)
    if not d["pairs_i_eq_j_ok"] or not d["pairs_i_ne_j_ok"]:
        raise AssertionError("expected all pairs to match classical formula")
    return (
        f"i=j err={d['pairs_i_eq_j_max_rel_err']:.1e}  "
        f"i!=j err={d['pairs_i_ne_j_max_rel_err']:.1e}"
    )


# --------------------------------------------------------------------------- #
#  2. Training tests                                                           #
# --------------------------------------------------------------------------- #
def _train_and_check(
    label: str,
    T: int,
    d: int,
    k: int,
    epochs: int,
    extra_cfg: dict,
    log: logging.Logger,
) -> str:
    from config import DATASET_CONFIG, OPTIMIZATION_CONFIG
    from jax_training_pipeline import run_training

    out = OUT / label
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(OPTIMIZATION_CONFIG)
    cfg.update(
        {
            "circuit_mode": "section2",
            "embedding_dim": d,
            "non_linear_order": k,
            "num_layers": 2,
            "epochs": epochs,
            "learning_rate": 1e-3,
            "max_sentences": 8,
            "seed": 42,
            "run_label": label,
            "output_dir": str(out),
            **extra_cfg,
        }
    )
    DATASET_CONFIG["sentence_length"] = T
    DATASET_CONFIG["max_sentences"] = 8

    code = run_training(logger=log, cfg=cfg, comm=None, rank=0, size=1)
    if code != 0:
        raise AssertionError(f"run_training exit={code}")

    metrics_path = out / "summaries" / "metrics.json"
    if not metrics_path.exists():
        raise AssertionError(f"missing {metrics_path}")
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    if "mean_O_ij" not in m or m["circuit_mode"] != "section2":
        raise AssertionError("bad metrics.json")
    if not (out / "matrices" / "W_matrix.npy").exists():
        raise AssertionError("missing W_matrix.npy")

    parts = [f"loss={m['final_loss']:.4f}", f"O_ij={m['mean_O_ij']:.4f}"]
    if m.get("converged") is not None:
        parts.append(f"converged={m['converged']}")
    if m.get("training_phases"):
        parts.append(f"phases={len(m['training_phases'])}")
    return " ".join(parts)


def _test_training_baseline(log: logging.Logger) -> str:
    return _train_and_check("baseline", T=2, d=2, k=1, epochs=4, extra_cfg={}, log=log)


def _test_training_warm_start(log: logging.Logger) -> str:
    return _train_and_check(
        "warm_start_w",
        T=2,
        d=2,
        k=1,
        epochs=4,
        extra_cfg={"warm_start_w_identity": True},
        log=log,
    )


def _test_training_curriculum(log: logging.Logger) -> str:
    return _train_and_check(
        "curriculum_k",
        T=2,
        d=4,
        k=2,
        epochs=3,
        extra_cfg={"curriculum_k": True},
        log=log,
    )


def _test_training_convergence(log: logging.Logger) -> str:
    return _train_and_check(
        "convergence",
        T=2,
        d=2,
        k=1,
        epochs=4,
        extra_cfg={
            "train_until_converged": True,
            "max_epochs": 80,
            "loss_rel_tol": 1e-3,
            "convergence_patience": 4,
        },
        log=log,
    )


def _test_train_qsa_direct(log: logging.Logger) -> str:
    from qsa_training import QSATrainConfig, train_qsa

    cfg = QSATrainConfig(
        T=2, d=2, k=1, epochs=3, max_sentences=8, seed=42,
        run_label="direct", output_dir=str(OUT / "direct"),
    )
    result = train_qsa(cfg, output_dir=OUT / "direct", logger=log)
    if result["mean_O_ij"] <= 0:
        raise AssertionError("mean_O_ij must be positive")
    return f"direct train_qsa O_ij={result['mean_O_ij']:.4f}"


# --------------------------------------------------------------------------- #
#  3. Integration                                                              #
# --------------------------------------------------------------------------- #
def _test_run_study_import() -> str:
    import run_study  # noqa: F401

    assert hasattr(run_study, "main")
    assert hasattr(run_study, "SWEEP_T")
    return "run_study import ok"


def _test_main_hpc_dry() -> str:
    import subprocess

    r = subprocess.run(
        [sys.executable, str(ROOT / "main_hpc.py"), "--dry-layout", "--circuit-mode", "section2",
         "--sentence-length", "4", "--embedding-dim", "4"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=60,
    )
    if r.returncode != 0:
        raise AssertionError(f"main_hpc dry-layout failed: {r.stderr[:300]}")
    if "mode=section2" not in r.stdout:
        raise AssertionError("expected section2 layout preview in stdout")
    return "main_hpc --dry-layout ok"


def _test_legacy_mode_routes(log: logging.Logger) -> str:
    """Verify legacy mode still routes (dry: just check import path, no full train)."""
    from jax_training_pipeline import run_training
    from config import OPTIMIZATION_CONFIG, DATASET_CONFIG

    cfg = dict(OPTIMIZATION_CONFIG)
    cfg["circuit_mode"] = "section2"
    cfg["epochs"] = 1
    cfg["embedding_dim"] = 2
    cfg["non_linear_order"] = 1
    cfg["run_label"] = "route_check"
    cfg["output_dir"] = str(OUT / "route_check")
    DATASET_CONFIG["sentence_length"] = 2
    DATASET_CONFIG["max_sentences"] = 4
    code = run_training(logger=log, cfg=cfg, comm=None, rank=0, size=1)
    if code != 0:
        raise AssertionError("section2 routing failed")
    return "jax_training_pipeline section2 route ok"


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Batteria completa test Section 2.")
    parser.add_argument("--fast", action="store_true", help="salta test lenti (curriculum, convergence)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report = SuiteReport()
    log = _logger(args.verbose)

    sections = [
        ("1. Imports", [("imports", _test_imports)]),
        ("2. Circuito / formule", [
            ("classical_formula", _test_classical_formula),
            ("qubit_budget", _test_qubit_budget),
            ("self_check_readout", _test_self_check_readout),
            ("diagnose_analytic_mismatch", _test_diagnose_analytic),
        ]),
        ("3. Training Section 2", [
            ("training_baseline", lambda: _test_training_baseline(log)),
            ("training_warm_start_W", lambda: _test_training_warm_start(log)),
            ("train_qsa_direct", lambda: _test_train_qsa_direct(log)),
            ("pipeline_route", lambda: _test_legacy_mode_routes(log)),
        ]),
        ("4. Training avanzato", [
            ("training_curriculum_k", lambda: _test_training_curriculum(log)),
            ("training_convergence", lambda: _test_training_convergence(log)),
        ]),
        ("5. Integrazione", [
            ("run_study_import", _test_run_study_import),
            ("main_hpc_dry_layout", _test_main_hpc_dry),
        ]),
    ]

    print("=" * 60)
    print("RUN_ALL_TESTS — batteria completa Section 2")
    print("=" * 60)
    print(f"Output: {OUT}")
    if args.fast:
        print("Modalita: FAST (salta curriculum + convergence)")
    print()

    skip_slow = {"training_curriculum_k", "training_convergence"}

    for section_name, tests in sections:
        if args.fast and section_name == "4. Training avanzato":
            print(f"--- {section_name} [SKIPPED in --fast] ---")
            continue
        print(f"--- {section_name} ---")
        for test_name, fn in tests:
            if args.fast and test_name in skip_slow:
                continue
            _run_test(report, test_name, fn, args.verbose)
        print()

    report_path = OUT / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report.write(report_path)

    n_pass = sum(r.passed for r in report.results)
    n_total = len(report.results)
    print("=" * 60)
    print(f"RISULTATO: {n_pass}/{n_total} PASS")
    print(f"Report: {report_path}")
    print("=" * 60)

    if not report.ok:
        print("\nTest falliti:")
        for r in report.results:
            if not r.passed:
                print(f"  - {r.name}: {r.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
