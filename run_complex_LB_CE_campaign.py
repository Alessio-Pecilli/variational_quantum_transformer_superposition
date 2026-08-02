#!/usr/bin/env python3
"""Classical Markov sequences + COMPLEX ansatz — L_B and CE_unif (L_1) campaign.

STRICT metric contract (theory / new_loss circuit notes):
  p_j = f_j^2 = |<y_j|z_j>|^2 / ||z_j||^2
      = nu_j / zeta_j   (same object after soft or poly/monomial kernel)
  CE_unif = L_1 = -(1/T) sum_j log p_j

This definition is used for EVERY model family here:
  k-QSA / k-CSA (mono+poly): kernel monomial|poly, p_j from f_j^2
  nl-CSA iso/gen: soft-kernel pipeline, SAME p_j = |<y|z>|^2/||z||^2 after attention

NOT the PTB discrete vocab CE. That mismatch is exactly what made classical-PTB
L_1 look incomparable; this campaign keeps one continuous CE_unif for all.

Requested plots (k in {1,2,3,5}, n_seeds=10, param counts in legend):
  - L_B train/test vs k:  k-QSA, k-CSA, poly-k-QSA, poly-k-CSA
  - L_1 train/test vs k:  k-QSA, k-CSA, poly-k-CSA, nl-CSA iso, nl-CSA gen
    (poly-k-QSA also evaluated and saved; included on L_1 plots for completeness)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from run_quantum_sequences_loss import (  # noqa: E402
    PLOT_STYLES,
    _L_1_for_sequence,
    _batch_L_1,
    _batch_L_B,
    _init_params,
    _per_step_p_for_sequence,
    _run_record,
    _wv_from_params,
    aggregate,
    generate_classical_dataset,
    train_model,
)


def param_count(family: str, d: int, T: int, layers: int) -> dict[str, int]:
    """Trainable real scalar count for the complex-ansatz models."""
    n = int(round(math.log2(d)))
    if family == "kqsa":
        n_angles = 2 * layers * n * 3  # wp, vp: RX-RY-RZ
        n_phi = T
        return {
            "n_params_angles": n_angles,
            "n_params_phi": n_phi,
            "n_params_total": n_angles + n_phi,
        }
    # k-CSA / nl-iso / nl-gen: complex d×d w_raw, v_raw (+ phi)
    n_mat_real = 2 * (2 * d * d)  # two complex matrices → 4 d^2 reals
    n_phi = T
    return {
        "n_params_matrices_real": n_mat_real,
        "n_params_phi": n_phi,
        "n_params_total": n_mat_real + n_phi,
    }


def label_with_params(display: str, family: str, d: int, T: int, layers: int) -> str:
    pc = param_count(family, d, T, layers)
    return f"{display} ({pc['n_params_total']} par)"


def verify_CE_unif_identity(d: int = 8, T: int = 6, k: int = 2, seed: int = 0) -> None:
    """Hard check: CE_unif == -(1/T) mean log(f_j^2) for mono/poly/soft."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(T, d)) + 1j * rng.normal(size=(T, d))
    y = rng.normal(size=(T, d)) + 1j * rng.normal(size=(T, d))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    x_j = jnp.asarray(x, dtype=jnp.complex128)
    y_j = jnp.asarray(y, dtype=jnp.complex128)

    params = _init_params("kqsa", d, T, layers=2, seed=seed)
    w, v = _wv_from_params(params, d, "kqsa")

    for mode in ("monomial", "poly", "soft"):
        p = np.asarray(_per_step_p_for_sequence(x_j, y_j, w, v, k, mode, d))
        ce_manual = float(-np.mean(np.log(np.clip(p, 1e-30, 1.0))))
        ce_fn = float(_L_1_for_sequence(x_j, y_j, w, v, k, mode, d))
        if abs(ce_manual - ce_fn) > 1e-10:
            raise AssertionError(f"CE_unif mismatch mode={mode}: manual={ce_manual} fn={ce_fn}")
        if np.any(p < 0) or np.any(p > 1 + 1e-9):
            raise AssertionError(f"p_j out of [0,1] for mode={mode}")
        # f_j^2 identity: p = |aj|^2 / ||z||^2 already enforced inside _per_step_p
    print("[VERIFY] CE_unif = -(1/T) sum log(f_j^2) OK for monomial/poly/soft")


def plot_vs_k_params(
    points: list[dict[str, Any]],
    out_path: Path,
    title: str,
    ylabel: str,
    nl_horizontal: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in points:
        by_model.setdefault(row["label"], []).append(row)

    # Pass 1: non-nl curves (defines k-range)
    all_ks: list[int] = []
    nl_items: list[tuple[str, list[dict[str, Any]]]] = []
    for name, rows in by_model.items():
        if nl_horizontal and rows and str(rows[0].get("family", "")).startswith("nlcsa"):
            nl_items.append((name, rows))
            continue
        rows = sorted(rows, key=lambda x: int(x["k"]))
        ks = [int(r["k"]) for r in rows]
        all_ks.extend(ks)
        means = [float(r["loss_mean"]) for r in rows]
        stds = [float(r["loss_std"]) for r in rows]
        st = PLOT_STYLES.get(rows[0]["model"], dict(color="0.3", linestyle="-", marker="o", linewidth=2.2))
        ax.errorbar(
            ks,
            means,
            yerr=stds,
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st.get("marker", "o"),
            linewidth=st.get("linewidth", 2.2),
            capsize=4,
            label=name,
        )

    # Pass 2: nl horizontal refs
    kmin = min(all_ks) if all_ks else 1
    kmax = max(all_ks) if all_ks else 5
    xs = np.linspace(kmin, kmax, 64)
    for name, rows in nl_items:
        mean = float(np.mean([float(r["loss_mean"]) for r in rows]))
        std = float(np.mean([float(r["loss_std"]) for r in rows]))
        st = PLOT_STYLES.get(rows[0]["model"], dict(color="0.1", linestyle="-", linewidth=2.0))
        ax.axhline(mean, color=st["color"], linestyle=st.get("linestyle", "-"), linewidth=2.0, label=name)
        ax.fill_between(xs, mean - std, mean + std, color=st["color"], alpha=0.10, linewidth=0)

    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--ks", type=str, default="1,2,3,5")
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--poly-epochs", type=int, default=600)
    parser.add_argument("--nl-epochs", type=int, default=400)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-seed-base", type=int, default=1042)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--classical-rho", type=float, default=0.8)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/classical_sequences/complex_LB_CE_k1-2-3-5_n10",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.T = 8
        args.d = 16
        args.ks = "1,2"
        args.layers = 2
        args.epochs = 30
        args.poly_epochs = 40
        args.nl_epochs = 30
        args.train_size = 32
        args.test_size = 16
        args.n_seeds = 2
        args.min_epochs = 8
        args.patience = 6
        args.eval_every = 2

    verify_CE_unif_identity(d=min(8, args.d), T=min(6, args.T - 1), k=2, seed=0)

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    if max(ks) > 5:
        raise ValueError(f"This campaign is capped at k<=5, got {ks}")

    out = Path(args.output_dir)
    plots = out / "plots"
    aggs = out / "aggregates"
    plots.mkdir(parents=True, exist_ok=True)
    aggs.mkdir(parents=True, exist_ok=True)

    dataset = generate_classical_dataset(
        train_size=args.train_size,
        test_size=args.test_size,
        d=args.d,
        T=args.T,
        seed=args.seed,
        kind="markov",
        rho=args.classical_rho,
    )
    T_seq = int(dataset.train_states.shape[1] - 1)
    assert T_seq == args.T

    # display, family, kernel_mode, max_epochs, on_LB_plot, on_L1_plot
    specs: list[tuple[str, str, str, int, bool, bool]] = [
        ("k-QSA L=16", "kqsa", "monomial", args.epochs, True, True),
        ("k-CSA", "kcsa", "monomial", args.epochs, True, True),
        ("poly-k-QSA L=16", "kqsa", "poly", args.poly_epochs, True, True),  # L1 saved; plot includes it
        ("poly-k-CSA", "kcsa", "poly", args.poly_epochs, True, True),
        ("nl-CSA iso", "nlcsa_iso", "soft", args.nl_epochs, False, True),
        ("nl-CSA gen", "nlcsa_gen", "soft", args.nl_epochs, False, True),
    ]

    lb_train: list[dict[str, Any]] = []
    lb_test: list[dict[str, Any]] = []
    l1_train: list[dict[str, Any]] = []
    l1_test: list[dict[str, Any]] = []
    param_table: dict[str, Any] = {}

    for display, family, mode, max_epochs, on_lb, on_l1 in specs:
        pc = param_count(family, args.d, T_seq, args.layers)
        label = label_with_params(display, family, args.d, T_seq, args.layers)
        param_table[display] = {"family": family, "kernel_mode": mode, **pc, "label": label}
        print(f"[params] {label}", flush=True)

        k_list = [0] if family.startswith("nlcsa") else ks  # nl: single k-indep run (use k=1 internally)
        for k in k_list:
            k_eff = 1 if family.startswith("nlcsa") else k
            runs = []
            for s_idx in range(args.n_seeds):
                mseed = args.model_seed_base + s_idx
                trained = train_model(
                    train_states=dataset.train_states,
                    test_states=dataset.test_states,
                    family=family,
                    kernel_mode=mode,
                    k=k_eff,
                    layers=args.layers,
                    max_epochs=max_epochs,
                    lr=args.lr,
                    seed=mseed + 17 * k_eff,
                    min_epochs=args.min_epochs,
                    patience=args.patience,
                    loss_rel_tol=1e-4,
                    eval_every=args.eval_every,
                )
                rec = _run_record(trained, family, mseed)
                # Absolute sanity: reported L_1 must match recompute from best params
                recompute = float(
                    _batch_L_1(trained["params"], jnp.asarray(dataset.train_states), k_eff, mode, family)
                )
                if abs(recompute - float(rec["train_L_1"])) > 1e-5:
                    raise RuntimeError(
                        f"L_1 recompute mismatch {display} k={k_eff} seed={mseed}: "
                        f"stored={rec['train_L_1']} recomputed={recompute}"
                    )
                if family in ("kqsa", "kcsa"):
                    recompute_lb = float(
                        _batch_L_B(trained["params"], jnp.asarray(dataset.train_states), k_eff, mode, family)
                    )
                    if abs(recompute_lb - float(rec["train_L_B"])) > 1e-5:
                        raise RuntimeError(
                            f"L_B recompute mismatch {display} k={k_eff} seed={mseed}: "
                            f"stored={rec['train_L_B']} recomputed={recompute_lb}"
                        )
                runs.append(rec)

            agg = aggregate(runs, "train_loss", "test_loss")
            lb_m = float(np.mean([r["train_L_B"] for r in runs]))
            lb_s = float(np.std([r["train_L_B"] for r in runs]))
            lb_tm = float(np.mean([r["test_L_B"] for r in runs]))
            lb_ts = float(np.std([r["test_L_B"] for r in runs]))
            l1_m = float(np.mean([r["train_L_1"] for r in runs]))
            l1_s = float(np.std([r["train_L_1"] for r in runs]))
            l1_tm = float(np.mean([r["test_L_1"] for r in runs]))
            l1_ts = float(np.std([r["test_L_1"] for r in runs]))

            payload = {
                "model": display,
                "label": label,
                "family": family,
                "kernel_mode": mode,
                "k": k_eff,
                "param_count": pc,
                "metric_L1": "CE_unif = -(1/T) sum_j log(f_j^2), f_j^2=|<y_j|z_j>|^2/||z_j||^2",
                "metric_LB": "L_B = -log F, F=(T+1)/(2T)*mu/zeta",
                "runs": runs,
                **agg,
                "L_B_mean": lb_m,
                "L_B_std": lb_s,
                "L_B_test_mean": lb_tm,
                "L_B_test_std": lb_ts,
                "L_1_mean": l1_m,
                "L_1_std": l1_s,
                "L_1_test_mean": l1_tm,
                "L_1_test_std": l1_ts,
            }
            tag = f"{family}_{mode}_k{k_eff}"
            (aggs / f"{tag}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(
                f"[{label} k={k_eff}] L_B={lb_m:.4f}+/-{lb_s:.4f} | "
                f"L_1(CE_unif)={l1_m:.4f}+/-{l1_s:.4f} | epochs~{agg['epochs_run_mean']:.0f} "
                f"conv={100*agg['converged_fraction']:.0f}%",
                flush=True,
            )

            point_base = {
                "model": display,
                "label": label,
                "family": family,
                "k": k_eff,
                "n_params_total": pc["n_params_total"],
            }
            if on_lb and family in ("kqsa", "kcsa"):
                lb_train.append({**point_base, "loss_mean": lb_m, "loss_std": lb_s})
                lb_test.append({**point_base, "loss_mean": lb_tm, "loss_std": lb_ts})
            if on_l1:
                # User list: kqsa, kcsa, polykcsa, nlcsaiso, nlcsagen (+ polykqsa kept)
                l1_train.append({**point_base, "loss_mean": l1_m, "loss_std": l1_s})
                l1_test.append({**point_base, "loss_mean": l1_tm, "loss_std": l1_ts})

    # L1 plot models exactly as requested: kQSA, kCSA, poly-k-CSA, nl-iso, nl-gen
    # (poly-k-QSA L_1 still saved in aggregates / summary.l1_all for completeness)
    l1_models_req = {"k-QSA L=16", "k-CSA", "poly-k-CSA", "nl-CSA iso", "nl-CSA gen"}
    l1_train_plot = [p for p in l1_train if p["model"] in l1_models_req]
    l1_test_plot = [p for p in l1_test if p["model"] in l1_models_req]
    l1_all = list(l1_train)  # includes poly-k-QSA

    plot_vs_k_params(
        lb_train,
        plots / "train_LB_vs_k.png",
        title=f"Train $L_B$ vs k (classical Markov + complex ansatz; T={args.T}, d={args.d}; mean±std, n={args.n_seeds})",
        ylabel=r"$L_B=-\log F$",
    )
    plot_vs_k_params(
        lb_test,
        plots / "test_LB_vs_k.png",
        title=f"Test $L_B$ vs k (classical Markov + complex ansatz; T={args.T}, d={args.d}; mean±std, n={args.n_seeds})",
        ylabel=r"$L_B=-\log F$",
    )
    plot_vs_k_params(
        l1_train_plot,
        plots / "train_L1_CE_unif_vs_k.png",
        title=f"Train $L_1$=CE_unif vs k (same $p_j=f_j^2$ for all; T={args.T}, d={args.d}; n={args.n_seeds})",
        ylabel=r"$L_1=\mathrm{CE_{unif}}=-\frac{1}{T}\sum_j \log p_j$",
        nl_horizontal=True,
    )
    plot_vs_k_params(
        l1_test_plot,
        plots / "test_L1_CE_unif_vs_k.png",
        title=f"Test $L_1$=CE_unif vs k (same $p_j=f_j^2$ for all; T={args.T}, d={args.d}; n={args.n_seeds})",
        ylabel=r"$L_1=\mathrm{CE_{unif}}=-\frac{1}{T}\sum_j \log p_j$",
        nl_horizontal=True,
    )

    summary = {
        "config": vars(args),
        "metric_contract": {
            "L_1": "CE_unif = -(1/T) sum_j log p_j",
            "p_j_mu": "f_j^2 = |<y_j|z_j>|^2 / ||z_j||^2 (monomial/poly kernel)",
            "p_j_nl": "same f_j^2 after soft-kernel (nl pipeline), NOT discrete vocab CE",
            "L_B": "-log F, F=(T+1)/(2T)*mu/zeta",
        },
        "param_table": param_table,
        "lb_train": lb_train,
        "lb_test": lb_test,
        "l1_train": l1_train_plot,
        "l1_test": l1_test_plot,
        "l1_train_all_including_poly_kqsa": l1_all,
        "verify_CE_unif": "PASS",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] outputs in {out}", flush=True)
    print("[params table]", json.dumps(param_table, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
