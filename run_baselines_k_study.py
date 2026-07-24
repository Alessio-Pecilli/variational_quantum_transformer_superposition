#!/usr/bin/env python3
"""Baselines k-study: param scaling, log-T alignment, mu/mu0, mono+poly.

Single-seed debug (not multi-seed finals / not obar).

Checks / plots:
  1) Renyi ≈ −log μ + log T  → plot mu-models as −logμ+logT alongside Renyi
  2) −log(μ/μ0) vs k   (μ0 = loss at random init)
  3) More QSA layers to approach CSA param count
  4) k=1..6, optional poly-k-QSA / poly-k-CSA (longer train)
  5) nl-CSA isometric + general Renyi as horizontal refs
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from classical_baselines import (
    BaselineConfig,
    kcsa_matrix_param_count,
    kcsa_mu_loss_sentence,
    mu_loss_sentence,
    prepare_data_bundle,
    qsa_angle_param_count,
    train_kcsa,
    train_kqsa,
    train_nlcsa,
)

jax.config.update("jax_enable_x64", True)


def _log() -> logging.Logger:
    log = logging.getLogger("k_study")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    return log


def _mean_loss(params, batch, pe, cfg, loss_fn) -> float:
    return float(jnp.mean(jax.vmap(lambda ids: loss_fn(params, ids, pe, cfg))(batch)))


def train_mu_with_mu0(name: str, cfg: BaselineConfig, bundle: dict, log: logging.Logger) -> dict:
    """Train k-QSA or k-CSA and attach −log μ0 at initialization."""
    if name == "k-QSA":
        from qsa_training import init_qsa_params

        key = jax.random.PRNGKey(cfg.model_seed)
        params0 = init_qsa_params(bundle["encoding"].vocabSize, bundle["qcfg"], key)
        loss_fn = mu_loss_sentence
        result = train_kqsa(cfg, bundle, logger=log)
    else:
        from classical_baselines import init_kcsa_params

        key = jax.random.PRNGKey(cfg.model_seed + 31)
        params0 = init_kcsa_params(bundle["encoding"].vocabSize, cfg, key)
        loss_fn = kcsa_mu_loss_sentence
        result = train_kcsa(cfg, bundle, logger=log)

    train_batch = bundle["token_batch"][bundle["train_idx"]]
    pe = bundle["pe"]
    neglog_mu0 = _mean_loss(params0, train_batch, pe, cfg, loss_fn)
    # also full-batch for reporting
    neglog_mu0_full = _mean_loss(params0, bundle["token_batch"], pe, cfg, loss_fn)
    result["neglog_mu0_train"] = neglog_mu0
    result["neglog_mu0_full"] = neglog_mu0_full
    result["final_neglog_mu"] = result["final_loss"]
    result["final_neglog_mu_over_mu0"] = result["final_loss"] - neglog_mu0
    result["final_neglog_mu_plus_logT"] = result["final_loss"] + math.log(cfg.T)
    result["n_params_angles"] = result.get("n_params_angles") or 0
    result["n_params_matrices"] = result.get("n_params_matrices") or 0
    # persist enrichment
    if cfg.output_dir:
        tag = "kqsa" if name == "k-QSA" else "kcsa"
        mpath = Path(cfg.output_dir) / f"{tag}_seed{cfg.model_seed}" / "metrics.json"
        if mpath.exists():
            data = json.loads(mpath.read_text(encoding="utf-8"))
            data.update(
                {
                    "neglog_mu0_train": neglog_mu0,
                    "neglog_mu0_full": neglog_mu0_full,
                    "final_neglog_mu_over_mu0": result["final_neglog_mu_over_mu0"],
                    "final_neglog_mu_plus_logT": result["final_neglog_mu_plus_logT"],
                }
            )
            mpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return result


def plot_loss_aligned(points, nl_refs, T: int, out_path: Path, title: str) -> None:
    """−logμ + log T for mu-models; Renyi for nl (same axis)."""
    fig, ax = plt.subplots(figsize=(9, 5.4))
    styles = {
        "k-QSA": dict(color="#0072B2", marker="o", linestyle="-"),
        "k-CSA": dict(color="#D55E00", marker="s", linestyle="--"),
        "poly-k-QSA": dict(color="#56B4E9", marker="^", linestyle="-"),
        "poly-k-CSA": dict(color="#E69F00", marker="v", linestyle="--"),
    }
    by = {}
    for p in points:
        by.setdefault(p["label"], []).append(p)
    for label, rows in by.items():
        rows = sorted(rows, key=lambda r: r["k"])
        base = "k-QSA" if "QSA" in label and "poly" not in label else (
            "k-CSA" if "CSA" in label and "poly" not in label else (
                "poly-k-QSA" if "poly" in label and "QSA" in label else "poly-k-CSA"
            )
        )
        st = styles.get(base, dict(color="0.3", marker="o", linestyle="-"))
        ys = [r["neglog_mu_plus_logT"] for r in rows]
        ax.plot(
            [r["k"] for r in rows],
            ys,
            label=label,
            linewidth=2.2,
            markersize=7,
            **{k: st[k] for k in ("color", "marker", "linestyle")},
        )
    nl_styles = {
        "isometric_renyi": dict(color="#009E73", linestyle="-."),
        "general_renyi": dict(color="#CC79A7", linestyle=":"),
    }
    for name, ref in nl_refs.items():
        st = nl_styles.get(name, dict(color="0.4", linestyle=":"))
        npar = ref.get("n_params_model")
        lbl = f"nl-CSA {name}" + (f" (params≈{npar})" if npar else "")
        ax.axhline(ref["final_loss"], linewidth=2.0, label=lbl, **st)
    ax.axhline(math.log(T), color="0.5", linestyle=":", linewidth=1, alpha=0.7, label=rf"$\log T$={math.log(T):.2f}")
    ax.set_xlabel("k")
    ax.set_ylabel(r"aligned loss:  $-\log\mu+\log T$  or  Renyi")
    ax.set_title(title)
    ax.set_xticks(sorted({p["k"] for p in points}))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_mu_over_mu0(points, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    styles = {
        "k-QSA": dict(color="#0072B2", marker="o", linestyle="-"),
        "k-CSA": dict(color="#D55E00", marker="s", linestyle="--"),
        "poly-k-QSA": dict(color="#56B4E9", marker="^", linestyle="-"),
        "poly-k-CSA": dict(color="#E69F00", marker="v", linestyle="--"),
    }
    by = {}
    for p in points:
        by.setdefault(p["label"], []).append(p)
    for label, rows in by.items():
        rows = sorted(rows, key=lambda r: r["k"])
        base = "k-QSA" if "QSA" in label and "poly" not in label else (
            "k-CSA" if "CSA" in label and "poly" not in label else (
                "poly-k-QSA" if "poly" in label and "QSA" in label else "poly-k-CSA"
            )
        )
        st = styles.get(base, dict(color="0.3", marker="o", linestyle="-"))
        ax.plot(
            [r["k"] for r in rows],
            [r["neglog_mu_over_mu0"] for r in rows],
            label=label,
            linewidth=2.2,
            markersize=7,
            **{k: st[k] for k in ("color", "marker", "linestyle")},
        )
    ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$-\log(\mu/\mu_0)=(-\log\mu)-(-\log\mu_0)$")
    ax.set_title(title)
    ax.set_xticks(sorted({p["k"] for p in points}))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_raw_and_mu0(points, out_path: Path, title: str) -> None:
    """Two panels: raw −logμ and −logμ0 vs k."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    styles = {
        "k-QSA": dict(color="#0072B2", marker="o", linestyle="-"),
        "k-CSA": dict(color="#D55E00", marker="s", linestyle="--"),
        "poly-k-QSA": dict(color="#56B4E9", marker="^", linestyle="-"),
        "poly-k-CSA": dict(color="#E69F00", marker="v", linestyle="--"),
    }
    by = {}
    for p in points:
        by.setdefault(p["label"], []).append(p)
    for ax, ykey, ylab in (
        (axes[0], "final_neglog_mu", r"final $-\log\mu$"),
        (axes[1], "neglog_mu0", r"init $-\log\mu_0$"),
    ):
        for label, rows in by.items():
            rows = sorted(rows, key=lambda r: r["k"])
            base = "k-QSA" if "QSA" in label and "poly" not in label else (
                "k-CSA" if "CSA" in label and "poly" not in label else (
                    "poly-k-QSA" if "poly" in label and "QSA" in label else "poly-k-CSA"
                )
            )
            st = styles.get(base, dict(color="0.3", marker="o", linestyle="-"))
            ax.plot(
                [r["k"] for r in rows],
                [r[ykey] for r in rows],
                label=label,
                linewidth=2.0,
                markersize=7,
                **{k: st[k] for k in ("color", "marker", "linestyle")},
            )
        ax.set_xlabel("k")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--ks", type=str, default="1,2,3,4,5,6")
    p.add_argument("--layers-list", type=str, default="2,16", help="QSA layer counts to try")
    p.add_argument("--epochs-mono", type=int, default=250)
    p.add_argument("--epochs-poly", type=int, default=500)
    p.add_argument("--max-sentences", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--model-seed", type=int, default=1042)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--nl-learning-rate", type=float, default=5e-3)
    p.add_argument("--run-poly", action="store_true", default=True)
    p.add_argument("--no-poly", dest="run_poly", action="store_false")
    p.add_argument("--poly-layers", type=int, default=2)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.T, args.d = 8, 4
        args.ks = "1,2,3"
        args.layers_list = "2,4"
        args.epochs_mono, args.epochs_poly = 40, 60
        args.max_sentences, args.batch_size = 64, 16

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    layers_list = [int(x) for x in args.layers_list.split(",") if x.strip()]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir or f"results/k_study/T{args.T}_d{args.d}_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    log = _log()
    bs = None if args.batch_size <= 0 else args.batch_size

    print("=" * 72)
    print("K-STUDY (single seed)")
    print(f"T={args.T} d={args.d} ks={ks} layers={layers_list}")
    print(f"epochs mono={args.epochs_mono} poly={args.epochs_poly} run_poly={args.run_poly}")
    print(f"log(T)={math.log(args.T):.4f}  | CSA matrices={kcsa_matrix_param_count(args.d)}")
    for L in layers_list:
        print(f"  QSA L={L}: angles={qsa_angle_param_count(args.d, L)}")
    print(f"Output: {out}")
    print("=" * 72)

    # --- nl refs once ---
    nl_refs = {}
    for emb in ("isometric", "general"):
        print(f"\n===== nl-CSA {emb}_renyi =====")
        cfg = BaselineConfig(
            T=args.T,
            d=args.d,
            k=1,
            layers=2,
            epochs=args.epochs_mono,
            max_epochs=args.epochs_mono,
            learning_rate=args.learning_rate,
            nl_learning_rate=args.nl_learning_rate,
            max_sentences=args.max_sentences,
            seed=args.data_seed,
            data_seed=args.data_seed,
            model_seed=args.model_seed,
            output_dir=str(out / "nl"),
            batch_size=bs,
            checkpoint_every=50,
            resume=False,
            early_stop=True,
            nl_embedding_mode=emb,
            nl_loss_mode="renyi",
        )
        bundle = prepare_data_bundle(cfg)
        r = train_nlcsa(cfg, bundle, logger=log)
        nl_refs[f"{emb}_renyi"] = {
            "final_loss": r["final_loss"],
            "n_params_model": r.get("n_params_model"),
            "n_params_total": r.get("n_params_total"),
        }
        # check Renyi vs −logμ+logT using a short k-QSA L=2 k=1 if available later

    points = []

    # --- monomial: each layers setting ---
    for L in layers_list:
        for k in ks:
            print(f"\n===== MONO L={L} k={k} =====")
            cfg = BaselineConfig(
                T=args.T,
                d=args.d,
                k=k,
                layers=L,
                epochs=args.epochs_mono,
                max_epochs=args.epochs_mono,
                learning_rate=args.learning_rate,
                max_sentences=args.max_sentences,
                seed=args.data_seed,
                data_seed=args.data_seed,
                model_seed=args.model_seed,
                output_dir=str(out / "mono" / f"L{L}" / f"k{k}"),
                batch_size=bs,
                checkpoint_every=50,
                resume=False,
                early_stop=True,
                kernel_mode="monomial",
            )
            # refresh qcfg layers
            bundle = prepare_data_bundle(cfg)
            bundle["qcfg"].layers = L
            qsa = train_mu_with_mu0("k-QSA", cfg, bundle, log)
            csa = train_mu_with_mu0("k-CSA", cfg, bundle, log)
            for r, model in ((qsa, "k-QSA"), (csa, "k-CSA")):
                n_ang = qsa_angle_param_count(args.d, L) if model == "k-QSA" else 0
                n_mat = kcsa_matrix_param_count(args.d) if model == "k-CSA" else 0
                label = (
                    f"{model} L={L} (angles={n_ang})"
                    if model == "k-QSA"
                    else f"{model} (matrices={n_mat})"
                )
                # CSA only once per k (same across L loops) — skip duplicates after first L
                if model == "k-CSA" and L != layers_list[0]:
                    continue
                points.append(
                    {
                        "k": k,
                        "model": model,
                        "label": label,
                        "kernel": "monomial",
                        "layers": L,
                        "final_neglog_mu": r["final_neglog_mu"],
                        "neglog_mu0": r["neglog_mu0_train"],
                        "neglog_mu_over_mu0": r["final_neglog_mu_over_mu0"],
                        "neglog_mu_plus_logT": r["final_neglog_mu_plus_logT"],
                        "n_params_angles": n_ang,
                        "n_params_matrices": n_mat,
                        "epochs_run": r["epochs_run"],
                    }
                )

    # --- poly ---
    if args.run_poly:
        L = args.poly_layers
        for k in ks:
            print(f"\n===== POLY L={L} k={k} (epochs={args.epochs_poly}) =====")
            cfg = BaselineConfig(
                T=args.T,
                d=args.d,
                k=k,
                layers=L,
                epochs=args.epochs_poly,
                max_epochs=args.epochs_poly,
                learning_rate=args.learning_rate,
                max_sentences=args.max_sentences,
                seed=args.data_seed,
                data_seed=args.data_seed,
                model_seed=args.model_seed,
                output_dir=str(out / "poly" / f"L{L}" / f"k{k}"),
                batch_size=bs,
                checkpoint_every=50,
                resume=False,
                early_stop=True,
                kernel_mode="poly",
            )
            bundle = prepare_data_bundle(cfg)
            bundle["qcfg"].layers = L
            qsa = train_mu_with_mu0("k-QSA", cfg, bundle, log)
            csa = train_mu_with_mu0("k-CSA", cfg, bundle, log)
            for r, model, tag in (
                (qsa, "k-QSA", "poly-k-QSA"),
                (csa, "k-CSA", "poly-k-CSA"),
            ):
                n_ang = qsa_angle_param_count(args.d, L) if model == "k-QSA" else 0
                n_mat = kcsa_matrix_param_count(args.d) if model == "k-CSA" else 0
                label = (
                    f"{tag} L={L} (angles={n_ang})"
                    if model == "k-QSA"
                    else f"{tag} (matrices={n_mat})"
                )
                points.append(
                    {
                        "k": k,
                        "model": tag,
                        "label": label,
                        "kernel": "poly",
                        "layers": L,
                        "final_neglog_mu": r["final_neglog_mu"],
                        "neglog_mu0": r["neglog_mu0_train"],
                        "neglog_mu_over_mu0": r["final_neglog_mu_over_mu0"],
                        "neglog_mu_plus_logT": r["final_neglog_mu_plus_logT"],
                        "n_params_angles": n_ang,
                        "n_params_matrices": n_mat,
                        "epochs_run": r["epochs_run"],
                    }
                )

    # --- Renyi ≈ −logμ + log T check ---
    check = {}
    mono_L2 = [p for p in points if p["kernel"] == "monomial" and p["model"] == "k-QSA" and p["layers"] == layers_list[0]]
    if mono_L2 and "isometric_renyi" in nl_refs:
        p1 = next((x for x in mono_L2 if x["k"] == 1), mono_L2[0])
        check = {
            "T": args.T,
            "log_T": math.log(args.T),
            "qsa_neglog_mu": p1["final_neglog_mu"],
            "qsa_neglog_mu_plus_logT": p1["neglog_mu_plus_logT"],
            "nl_isometric_renyi": nl_refs["isometric_renyi"]["final_loss"],
            "abs_diff": abs(p1["neglog_mu_plus_logT"] - nl_refs["isometric_renyi"]["final_loss"]),
            "conclusion": (
                "SUPPORTED: Renyi ≈ −logμ + log T within ~0.1"
                if abs(p1["neglog_mu_plus_logT"] - nl_refs["isometric_renyi"]["final_loss"]) < 0.5
                else "WEAK/FAIL: difference large — do not force alignment"
            ),
        }
        print("\n[CHECK]", check["conclusion"], f"diff={check['abs_diff']:.4f}")

    plot_loss_aligned(
        points,
        nl_refs,
        args.T,
        out / "loss_aligned_logT_vs_k.png",
        title=(
            f"Aligned loss vs k (T={args.T}, d={args.d}, seed {args.data_seed}/{args.model_seed})\n"
            r"mu-models: $-\log\mu+\log T$  |  nl: Renyi"
        ),
    )
    plot_mu_over_mu0(
        points,
        out / "neglog_mu_over_mu0_vs_k.png",
        title=f"Normalized improvement $-\\log(\\mu/\\mu_0)$ vs k (T={args.T}, d={args.d})",
    )
    plot_raw_and_mu0(
        points,
        out / "raw_loss_and_mu0_vs_k.png",
        title=f"Raw final $-\\log\\mu$ and init $-\\log\\mu_0$ vs k",
    )

    report = {
        "phase": "k_study_single_seed",
        "config": {
            "T": args.T,
            "d": args.d,
            "ks": ks,
            "layers_list": layers_list,
            "epochs_mono": args.epochs_mono,
            "epochs_poly": args.epochs_poly,
            "run_poly": args.run_poly,
            "data_seed": args.data_seed,
            "model_seed": args.model_seed,
            "output_dir": str(out),
        },
        "param_counts": {
            "csa_matrices": kcsa_matrix_param_count(args.d),
            "qsa_by_layers": {str(L): qsa_angle_param_count(args.d, L) for L in layers_list},
        },
        "renyi_logT_check": check,
        "nl_refs": nl_refs,
        "points": points,
        "plots": [
            str(out / "loss_aligned_logT_vs_k.png"),
            str(out / "neglog_mu_over_mu0_vs_k.png"),
            str(out / "raw_loss_and_mu0_vs_k.png"),
        ],
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDONE → {out / 'REPORT.json'}")
    print(f"Main aligned plot: {out / 'loss_aligned_logT_vs_k.png'}")
    print(f"mu/mu0 plot: {out / 'neglog_mu_over_mu0_vs_k.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
