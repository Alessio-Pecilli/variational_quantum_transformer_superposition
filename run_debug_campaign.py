#!/usr/bin/env python3
"""Full local debug campaign (single seed) before any HPC finals.

Primary paper-style deliverable:
  final_loss_vs_k.png  — −log μ vs k for k-QSA/k-CSA + isometric Rényi
                         horizontal reference (explicitly different metric).

Also: curves by k, nl ablations, convergence diagnostics, REPORT.

Does NOT run multi-seed finals or obar campaigns.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from classical_baselines import (
    NL_ABLATIONS,
    BaselineConfig,
    kcsa_matrix_param_count,
    plot_convergence_diagnostics,
    plot_final_loss_vs_k,
    prepare_data_bundle,
    qsa_angle_param_count,
    train_kcsa,
    train_kqsa,
    train_nlcsa,
)


STYLES = {
    "k-QSA": dict(color="#0072B2", marker="o", linestyle="-", linewidth=2.4),
    "k-CSA": dict(color="#D55E00", marker="s", linestyle="--", linewidth=2.4),
}
NL_STYLES = {
    "isometric_ce": dict(color="#009E73", linestyle="-", linewidth=2.0),
    "isometric_renyi": dict(color="#56B4E9", linestyle="--", linewidth=2.0),
    "general_ce": dict(color="#E69F00", linestyle="-.", linewidth=1.8),
    "general_renyi": dict(color="#CC79A7", linestyle=":", linewidth=1.8),
}


def _logger() -> logging.Logger:
    log = logging.getLogger("debug_campaign")
    log.setLevel(logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    return log


def _cfg(base: dict, **overrides) -> BaselineConfig:
    return BaselineConfig(**{**base, **overrides})


def plot_nl_ablations(nl_runs: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for r in nl_runs:
        abl = r.get("ablation", "?")
        st = NL_STYLES.get(abl, dict(color="0.3", linestyle="-"))
        ep = np.arange(1, len(r["loss_history"]) + 1)
        npar = r.get("n_params_model")
        label = f"{abl}" + (f" (model={npar})" if npar else "")
        axes[0].plot(ep, r["loss_history"], label=label, color=st["color"], linestyle=st["linestyle"], linewidth=2)
        axes[1].plot(ep, r["val_ppl_history"], label=label, color=st["color"], linestyle=st["linestyle"], linewidth=2)
    axes[0].set_title("nl-CSA train loss (CE or Renyi — not comparable to each other)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].set_title("nl-CSA next-token CE perplexity (NOT comparable to 1/μ)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val PPL = exp(CE)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_curves_by_k(mu_by_k: dict, out_path: Path, angles: int, matrices: int) -> None:
    ks = sorted(mu_by_k.keys())
    fig, axes = plt.subplots(2, len(ks), figsize=(3.6 * len(ks), 7), sharex="col")
    if len(ks) == 1:
        axes = np.array(axes).reshape(2, 1)
    for j, k in enumerate(ks):
        for r in mu_by_k[k]:
            st = STYLES[r["model"]]
            ep = np.arange(1, len(r["loss_history"]) + 1)
            if r["model"] == "k-QSA":
                label = f"k-QSA (angles={angles})"
            else:
                label = f"k-CSA (matrices={matrices})"
            axes[0, j].plot(ep, r["loss_history"], label=label, **{kk: st[kk] for kk in ("color", "linestyle", "linewidth")})
            axes[1, j].plot(
                ep,
                r["val_ppl_history"],
                label=label,
                **{kk: st[kk] for kk in ("color", "linestyle", "linewidth")},
            )
        axes[0, j].set_title(rf"k={k} train $-\log\mu$")
        axes[1, j].set_title(rf"k={k} val $1/\mu$ (NOT LM PPL)")
        axes[0, j].grid(True, alpha=0.3)
        axes[1, j].grid(True, alpha=0.3)
        axes[1, j].set_xlabel("epoch")
        if j == 0:
            axes[0, j].legend(fontsize=7)
            axes[1, j].legend(fontsize=7)
    fig.suptitle("k-QSA vs independent k-CSA (single seed)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def verify_nl_k_independence(base: dict, out_dir: Path, log: logging.Logger) -> dict:
    runs = []
    for k in (1, 4):
        cfg = _cfg(
            base,
            k=k,
            output_dir=str(out_dir / "nl_k_independence" / f"k{k}"),
            nl_embedding_mode="isometric",
            nl_loss_mode="ce",
            epochs=min(40, base["epochs"]),
            max_epochs=min(40, base["max_epochs"]),
            early_stop=False,
            resume=False,
            checkpoint_every=0,
        )
        bundle = prepare_data_bundle(cfg)
        runs.append(train_nlcsa(cfg, bundle, logger=log))
    gap = max(abs(a - b) for a, b in zip(runs[0]["loss_history"], runs[1]["loss_history"]))
    return {
        "k_values": [1, 4],
        "max_abs_train_loss_diff": gap,
        "identical": gap < 1e-12,
        "conclusion": (
            "nl-CSA is k-independent: identical curves at k=1 and k=4 "
            f"(max |Δ|={gap:.3e})."
            if gap < 1e-12
            else f"UNEXPECTED: nl-CSA differed across k (max |Δ|={gap:.3e})."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=16)
    p.add_argument("--d", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--ks", type=str, default="1,2,3,4")
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--max-epochs", type=int, default=250)
    p.add_argument("--max-sentences", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--nl-learning-rate", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--model-seed", type=int, default=1042)
    p.add_argument("--early-stop", action="store_true", default=True)
    p.add_argument("--no-early-stop", dest="early_stop", action="store_false")
    p.add_argument("--loss-rel-tol", type=float, default=1e-4)
    p.add_argument("--convergence-patience", type=int, default=15)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.T, args.d = 8, 4
        args.epochs, args.max_epochs = 40, 60
        args.max_sentences = 64
        args.batch_size = 16

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.output_dir or f"results/debug_campaign_v3/T{args.T}_d{args.d}_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    log = _logger()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    batch_size = None if args.batch_size <= 0 else args.batch_size
    angles = qsa_angle_param_count(args.d, args.layers)
    matrices = kcsa_matrix_param_count(args.d)

    base = dict(
        T=args.T,
        d=args.d,
        layers=args.layers,
        epochs=args.epochs,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        nl_learning_rate=args.nl_learning_rate,
        max_sentences=args.max_sentences,
        seed=args.data_seed,
        data_seed=args.data_seed,
        model_seed=args.model_seed,
        batch_size=batch_size,
        checkpoint_every=50,
        resume=False,
        early_stop=args.early_stop,
        loss_rel_tol=args.loss_rel_tol,
        convergence_patience=args.convergence_patience,
        kernel_mode="monomial",
        track_grad_norm=True,
    )

    print("=" * 72)
    print("DEBUG CAMPAIGN v3 — single seed (NOT multi-seed / NOT obar finals)")
    print(f"T={args.T} d={args.d} (T>d? {args.T > args.d}) ks={ks} epochs<={args.max_epochs}")
    print(f"data_seed={args.data_seed} model_seed={args.model_seed}")
    print(f"k-QSA angles={angles} | k-CSA QR-matrices={matrices}")
    print(f"Output: {out}")
    print("=" * 72)

    mu_points = []
    mu_by_k: dict[int, list] = {}
    for k in ks:
        k_dir = out / "mu_vs_k" / f"k{k}"
        cfg = _cfg(base, k=k, output_dir=str(k_dir))
        bundle = prepare_data_bundle(cfg)
        print(f"\n===== k={k} | vocab={bundle['encoding'].vocabSize} =====")
        qsa = train_kqsa(cfg, bundle, logger=log)
        csa = train_kcsa(cfg, bundle, logger=log)
        gap = max(abs(a - b) for a, b in zip(qsa["loss_history"], csa["loss_history"]))
        print(f"[INDEPENDENCE] k={k} max |QSA−CSA| train = {gap:.4f}")
        for r in (qsa, csa):
            mu_points.append(
                {
                    "k": k,
                    "model": r["model"],
                    "final_loss": r["final_loss"],
                    "final_val_metric": r["final_val_ce"],
                    "final_val_exp": r["final_val_ppl"],
                    "epochs_run": r["epochs_run"],
                    "stopped_early": r.get("stopped_early", False),
                    "rel_improve_last5": float(np.nanmean(r.get("rel_improve_history", [np.nan])[-5:])),
                    "grad_norm_final": (r.get("grad_norm_history") or [None])[-1],
                    "n_params_total": r.get("n_params_total"),
                    "n_params_angles": r.get("n_params_angles"),
                    "n_params_matrices": r.get("n_params_matrices"),
                    "train_gap_vs_other": gap,
                    "val_metric_kind": "neg_log_mu",
                }
            )
        mu_by_k[k] = [qsa, csa]
        plot_convergence_diagnostics([qsa, csa], k_dir / "convergence_diagnostics.png")

    plot_curves_by_k(mu_by_k, out / "curves_kqsa_vs_kcsa_by_k.png", angles, matrices)

    nl_runs = []
    nl_refs = {}
    for emb_mode, loss_mode in NL_ABLATIONS:
        abl = f"{emb_mode}_{loss_mode}"
        print(f"\n===== nl-CSA ablation {abl} (k unused) =====")
        cfg = _cfg(
            base,
            k=1,
            output_dir=str(out / "nl_ablations"),
            nl_embedding_mode=emb_mode,
            nl_loss_mode=loss_mode,
        )
        bundle = prepare_data_bundle(cfg)
        r = train_nlcsa(cfg, bundle, logger=log)
        nl_runs.append(r)
        nl_refs[abl] = {
            "final_val_ppl": r["final_val_ppl"],
            "final_val_ce": r["final_val_ce"],
            "final_loss": r["final_loss"],
            "epochs_run": r["epochs_run"],
            "stopped_early": r.get("stopped_early", False),
            "n_params_model": r.get("n_params_model"),
            "n_params_total": r.get("n_params_total"),
            "train_metric": r.get("train_metric"),
            "val_metric_kind": r.get("val_metric_kind"),
        }

    plot_nl_ablations(nl_runs, out / "nl_ablations_curves.png")
    plot_convergence_diagnostics(nl_runs, out / "nl_ablations_convergence.png")

    print("\n===== nl-CSA k-independence check =====")
    k_indep = verify_nl_k_independence(base, out, log)
    print(k_indep["conclusion"])

    iso_renyi = nl_refs["isometric_renyi"]
    loss_points = [
        {
            "k": p["k"],
            "model": p["model"],
            "final_loss": p["final_loss"],
            "final_loss_mean": p["final_loss"],
            "final_loss_std": 0.0,
            "n_params_angles": p.get("n_params_angles") or 0,
            "n_params_matrices": p.get("n_params_matrices") or 0,
        }
        for p in mu_points
    ]
    plot_final_loss_vs_k(
        loss_points,
        out / "final_loss_vs_k.png",
        title=(
            f"Train loss vs k (T={args.T}, d={args.d}, T>d={args.T > args.d}, "
            f"seed {args.data_seed}/{args.model_seed})\n"
            r"k-QSA/k-CSA: $-\log\mu$  |  nl: Rényi (different metric — scale not comparable)"
        ),
        ykey="final_loss_mean",
        ylabel="train loss",
        nl_ref=iso_renyi,
        nl_ref_ykey="final_loss",
        nl_ref_label="nl-CSA isometric+Rényi (k-indep.)",
    )

    status = []
    for p in mu_points:
        converged = bool(p["stopped_early"]) or (
            p["rel_improve_last5"] is not None and abs(p["rel_improve_last5"]) < args.loss_rel_tol
        )
        needs_more = (not converged) or (p["epochs_run"] >= args.max_epochs - 1)
        status.append({**p, "converged": converged, "needs_more_epochs_or_lr_tweak": needs_more})

    sample_cfg = _cfg(base, k=2)
    report = {
        "phase": "debug_campaign_v3_single_seed",
        "declared_seed": {"data_seed": args.data_seed, "model_seed": args.model_seed, "n_seeds": 1},
        "advantage_regime": {"T": args.T, "d": args.d, "T_gt_d": args.T > args.d},
        "config": {
            "T": args.T,
            "d": args.d,
            "layers": args.layers,
            "ks": ks,
            "epochs": args.epochs,
            "max_epochs": args.max_epochs,
            "learning_rate": args.learning_rate,
            "nl_learning_rate": args.nl_learning_rate,
            "max_sentences": args.max_sentences,
            "batch_size": batch_size,
            "early_stop": args.early_stop,
            "output_dir": str(out),
        },
        "parametrization": sample_cfg.parametrization,
        "param_counts": {
            "k-QSA_angles": angles,
            "k-CSA_raw_matrices": matrices,
            "formula_qsa": "2 * layers * log2(d)",
            "formula_csa": "2 * d * d (W_raw, V_raw; QR)",
        },
        "metric_policy": {
            "paper_plot": "final_loss_vs_k.png",
            "do_not_compare": "1/μ (mu-models) vs next-token CE PPL (nl-CSA)",
            "relative_improvement": "rel_impr5=(mean(loss[t-5:t-1])-loss[t])/|old|",
        },
        "k_qsa_vs_kcsa": {
            "why_csa_starts_higher": "Haar-like QR init → low μ; QSA angles less mixed.",
            "why_csa_ends_lower_parallel": "Full O(d) capacity (128) vs 12-angle ansatz.",
            "per_k_train_gaps": {
                str(p["k"]): p["train_gap_vs_other"] for p in mu_points if p["model"] == "k-QSA"
            },
        },
        "nl_architecture": (
            "embedding → causal softmax attention → residual/normalization → "
            "FFN GELU → output logits (tied to embedding)"
        ),
        "nl_ablations": nl_refs,
        "nl_paper_ref": {"name": "isometric_renyi", **iso_renyi},
        "nl_k_independence": k_indep,
        "mu_results_vs_k": mu_points,
        "convergence_status": status,
        "plots": [
            str(out / "final_loss_vs_k.png"),
            str(out / "curves_kqsa_vs_kcsa_by_k.png"),
            str(out / "nl_ablations_curves.png"),
            str(out / "nl_ablations_convergence.png"),
        ],
        "recommendations": _recommendations(status, args.max_epochs),
        "note": "Single-seed debug. Paper figure = final_loss_vs_k.png. No obar finals.",
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown_report(report, out / "REPORT.md")
    _write_notes(report, out / "NOTES_METRICS.md")

    print("\n" + "=" * 72)
    print(f"DONE → {out / 'REPORT.json'}")
    print(f"MAIN plot: {out / 'final_loss_vs_k.png'}")
    print("=" * 72)
    return 0


def _recommendations(status: list, max_epochs: int) -> list[str]:
    recs = []
    for k in sorted({s["k"] for s in status}):
        rows = [s for s in status if s["k"] == k and s["model"] == "k-QSA"]
        if not rows:
            continue
        s = rows[0]
        if s["needs_more_epochs_or_lr_tweak"]:
            recs.append(
                f"k={k} k-QSA: still improving at {s['epochs_run']} ep "
                f"(rel_impr5={s['rel_improve_last5']:.2e}); try max_epochs>={max_epochs * 2} or lr=5e-4."
            )
        else:
            recs.append(f"k={k} k-QSA: plateau OK within {max_epochs} epochs.")
    recs.append("Paper: final_loss_vs_k with isometric_renyi ref; always show param counts.")
    recs.append("Final HPC: n_seeds>=5–8 on −log μ (not cross-arch PPL); nl once per (T,d).")
    recs.append("Do not launch obar until k=3,4 show plateau.")
    return recs


def _write_markdown_report(report: dict, path: Path) -> None:
    lines = [
        "# Debug campaign report (single seed)",
        "",
        f"- data_seed={report['declared_seed']['data_seed']}, model_seed={report['declared_seed']['model_seed']}",
        f"- T={report['config']['T']}, d={report['config']['d']}, T>d={report['advantage_regime']['T_gt_d']}",
        f"- max_epochs={report['config']['max_epochs']}",
        f"- params: QSA angles={report['param_counts']['k-QSA_angles']}, "
        f"CSA matrices={report['param_counts']['k-CSA_raw_matrices']}",
        "",
        "## Paper plot",
        "`final_loss_vs_k.png` — −log μ vs k + isometric Rényi horizontal (different metric).",
        "",
        "## Convergence",
    ]
    for s in report["convergence_status"]:
        flag = "NEEDS MORE" if s["needs_more_epochs_or_lr_tweak"] else "OK"
        lines.append(
            f"- k={s['k']} {s['model']}: ep={s['epochs_run']} loss={s['final_loss']:.4f} [{flag}]"
        )
    lines += ["", "## Recommendations"] + [f"- {r}" for r in report["recommendations"]]
    lines += ["", "## Plots"] + [f"- `{p}`" for p in report["plots"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_notes(report: dict, path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Metric notes",
                "",
                f"- Paper plot: {report['metric_policy']['paper_plot']}",
                f"- Do NOT compare: {report['metric_policy']['do_not_compare']}",
                f"- rel_impr5: {report['metric_policy']['relative_improvement']}",
                f"- CSA starts higher: {report['k_qsa_vs_kcsa']['why_csa_starts_higher']}",
                f"- CSA ends lower/parallel: {report['k_qsa_vs_kcsa']['why_csa_ends_lower_parallel']}",
                f"- {report['nl_k_independence']['conclusion']}",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
