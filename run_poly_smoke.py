#!/usr/bin/env python3
"""Local smoke for the polynomial LCU kernel (qsa_section2_circuit_polynomial).

1) Circuit self-checks (overlap == projector == analytic)
2) Classical oracle: PR_poly vs PR_mono
3) Tiny baselines: loss vs k for k-QSA / k-CSA / nl-CSA (poly kernel)
4) Optional: rbar (poly analogue of obar) after a short train
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from classical_baselines import (
    BaselineConfig,
    aggregate_seed_results,
    plot_final_loss_vs_k,
    prepare_shared_bundle,
    train_kcsa,
    train_kqsa,
    train_nlcsa,
)
from qsa_section2_circuit_polynomial import (
    classical_report,
    lcu_coeffs,
    participation_ratio,
    run_oracle_only,
    self_check,
    set_backends,
    softmax_beta,
)


def _logger() -> logging.Logger:
    log = logging.getLogger("poly_smoke")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    return log


def check_circuit() -> bool:
    print("=" * 60)
    print("1) Circuit self-check (PennyLane + JAX)")
    print("=" * 60)
    set_backends()
    oks = [
        self_check(T=3, d=4, k=1),
        self_check(T=3, d=4, k=2),
        self_check(T=2, d=4, k=3),
    ]
    return all(oks)


def check_oracle() -> None:
    print("\n" + "=" * 60)
    print("2) Classical oracle (PR_poly vs PR_mono)")
    print("=" * 60)
    run_oracle_only()


def run_loss_vs_k(
    T: int = 8,
    d: int = 8,
    epochs: int = 60,
    max_sentences: int = 64,
    seeds: list[int] | None = None,
    ks: tuple[int, ...] = (1, 2, 3, 4),
    out_root: Path | None = None,
) -> Path:
    if seeds is None:
        seeds = [42, 43]
    out_root = out_root or Path("results") / "poly_smoke" / f"T{T}_d{d}_ep{epochs}"
    out_root.mkdir(parents=True, exist_ok=True)
    log = _logger()

    print("\n" + "=" * 60)
    print(f"3) Loss vs k  (poly kernel)  T={T} d={d} epochs={epochs} seeds={seeds}")
    print("=" * 60)

    points = []
    summary_rows = []
    for k in ks:
        out_k = out_root / f"k{k}"
        out_k.mkdir(parents=True, exist_ok=True)
        runs_qsa, runs_csa, runs_nl = [], [], []
        for s in seeds:
            cfg = BaselineConfig(
                T=T,
                d=d,
                k=k,
                epochs=epochs,
                max_sentences=max_sentences,
                seed=s,
                output_dir=str(out_k),
                run_label=f"seed{s}",
                batch_size=32,
                checkpoint_every=0,
                resume=False,
                kernel_mode="poly",
            )
            print(f"\n--- k={k} seed={s} ---")
            bundle = prepare_shared_bundle(cfg)
            qsa = train_kqsa(cfg, bundle, logger=log)
            csa = train_kcsa(cfg, bundle, logger=log)
            nl = train_nlcsa(cfg, bundle, logger=log)
            gap = max(abs(a - b) for a, b in zip(qsa["loss_history"], csa["loss_history"]))
            print(f"[CHECK] max |QSA-CSA| = {gap:.3e}")
            runs_qsa.append(qsa)
            runs_csa.append(csa)
            runs_nl.append(nl)

        for agg in (
            aggregate_seed_results(runs_qsa),
            aggregate_seed_results(runs_csa),
            aggregate_seed_results(runs_nl),
        ):
            points.append(
                {
                    "k": k,
                    "model": agg["model"],
                    "final_loss_mean": agg["final_loss_mean"],
                    "final_loss_std": agg["final_loss_std"],
                }
            )
            summary_rows.append(
                {
                    "k": k,
                    "model": agg["model"],
                    "final_loss_mean": agg["final_loss_mean"],
                    "final_loss_std": agg["final_loss_std"],
                    "n_seeds": agg["n_seeds"],
                }
            )
            print(
                f"k={k} {agg['model']:7}  loss={agg['final_loss_mean']:.4f}"
                f"±{agg['final_loss_std']:.4f}"
            )

    plot_final_loss_vs_k(
        points,
        out_root / "final_loss_vs_k_poly.png",
        title=f"Poly kernel: final loss vs k (T={T}, d={d})",
    )
    (out_root / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out_root / 'final_loss_vs_k_poly.png'}")
    return out_root


def rbar_snapshot(T: int = 8, d: int = 8, k: int = 2, seed: int = 0) -> None:
    """Quick classical rbar at random ortho W,V (no train) — poly vs monomial feel."""
    print("\n" + "=" * 60)
    print("4) rbar snapshot (untrained random W,V)")
    print("=" * 60)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((T, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    Wm, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Vm, _ = np.linalg.qr(rng.standard_normal((d, d)))
    for kk in (1, 2, 3, 4):
        beta = softmax_beta(d)
        c = lcu_coeffs(kk, beta)
        rep = classical_report(X, Y, Wm, Vm, kk, beta)
        pr_p = participation_ratio(X, Wm, c)
        pr_m = participation_ratio(X, Wm, np.eye(kk + 1)[kk])
        print(
            f"  k={kk}: rbar={rep['rbar']:.4f} mu={rep['mu']:.3e} "
            f"p_eff={rep['p_eff']} PR_poly={pr_p:.2f} PR_mono={pr_m:.2f} "
            f"adv?={rep['advantage']}"
        )


def main() -> int:
    ok = check_circuit()
    if not ok:
        print("SELF-CHECK FAILED")
        return 1
    check_oracle()
    rbar_snapshot()
    run_loss_vs_k()
    print("\n[DONE] poly smoke OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
