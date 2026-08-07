#!/usr/bin/env python3
"""HPC campaign on the clean-slate shared pipeline (qsa_bench).

One forward / one metrics / one chance path for every model. Models differ only in
params -> (W, V) and kernel. Trains with Adam on the shared loss (L_B for k-models,
L1 for nl-CSA); reports the common L1 axis for everyone and L_B for k-models only.

Produces, for CLASSICAL and QUANTUM:
  - train/test L1 vs k   (all models; no chance line)
  - train/test L_B vs k  (kQSA / kCSA mono+poly only)

Resume: skips aggregate cells that already have n_seeds runs.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

import qsa_bench as qb

PLOT_STYLES = {
    "k-QSA": dict(color="#0072B2", linestyle="-", marker="o", linewidth=2.4),
    "k-CSA": dict(color="#E69F00", linestyle="-", marker="s", linewidth=2.4),
    "poly-k-QSA": dict(color="#0072B2", linestyle="--", marker="^", linewidth=2.2),
    "poly-k-CSA": dict(color="#E69F00", linestyle="--", marker="v", linewidth=2.2),
    "nl-CSA iso": dict(color="#000000", linestyle="-", marker="D", linewidth=2.2),
    "nl-CSA gen": dict(color="#555555", linestyle=":", marker="x", linewidth=2.4),
}

DISPLAY = {
    "kqsa-mono": "k-QSA",
    "kqsa-poly": "poly-k-QSA",
    "kcsa-mono": "k-CSA",
    "kcsa-poly": "poly-k-CSA",
    "nlcsa-iso": "nl-CSA iso",
    "nlcsa-gen": "nl-CSA gen",
}

K_MODELS = [
    ("kqsa-mono", "mono", "L_B"),
    ("kqsa-poly", "poly", "L_B"),
    ("kcsa-mono", "mono", "L_B"),
    ("kcsa-poly", "poly", "L_B"),
]
NL_MODELS = [
    ("nlcsa-iso", "soft", "L1", True),
    ("nlcsa-gen", "soft", "L1", False),
]


# --------------------------------------------------------------------------- #
#  data                                                                        #
# --------------------------------------------------------------------------- #
def _paulis():
    i2 = np.eye(2, dtype=np.complex128)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return i2, x, z


def _kron_all(ops: list[np.ndarray]) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def build_tfim_hamiltonian(n_qubits: int, coupling_j: float, field_h: float) -> np.ndarray:
    i2, x, z = _paulis()
    dim = 2 ** n_qubits
    h = np.zeros((dim, dim), dtype=np.complex128)
    for bond in range(n_qubits - 1):
        ops = [i2] * n_qubits
        ops[bond] = z
        ops[bond + 1] = z
        h -= coupling_j * _kron_all(ops)
    for q in range(n_qubits):
        ops = [i2] * n_qubits
        ops[q] = x
        h -= field_h * _kron_all(ops)
    return h


def _unitary_from_h(h: np.ndarray, dt: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(h)
    return evecs @ np.diag(np.exp(-1j * dt * evals)) @ evecs.conj().T


def _clamp_global_phase(states: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    ref = states[..., 0]
    abs_ref = np.abs(ref)
    unit = np.where(abs_ref > eps, np.conj(ref / abs_ref), np.ones_like(ref))
    return states * unit[..., None]


def generate_quantum_xy(
    train_size: int, test_size: int, n_qubits: int, T: int, dt: float, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """TFIM trajectories with disjoint (J,h) on train/test. Returns X,Y with shape (N,T,d)."""
    dim = 2 ** n_qubits
    n_total = train_size + test_size
    rng_s = np.random.default_rng(seed)
    rng_p = np.random.default_rng(seed + 1)
    init = rng_s.standard_normal((n_total, dim)) + 1j * rng_s.standard_normal((n_total, dim))
    init /= np.linalg.norm(init, axis=1, keepdims=True)
    init = _clamp_global_phase(init[:, None, :])[:, 0, :]
    params = rng_p.uniform(0.2, 2.0, size=(n_total, 2))
    states = np.zeros((n_total, T + 1, dim), dtype=np.complex128)
    states[:, 0] = init
    for i in range(n_total):
        u = _unitary_from_h(build_tfim_hamiltonian(n_qubits, *params[i]), dt)
        cur = init[i]
        for t in range(1, T + 1):
            cur = u @ cur
            cur /= np.linalg.norm(cur)
            states[i, t] = cur
    states = _clamp_global_phase(states)
    Xtr, Ytr = states[:train_size, :-1], states[:train_size, 1:]
    Xte, Yte = states[train_size:, :-1], states[train_size:, 1:]
    return Xtr, Ytr, Xte, Yte


def generate_classical_xy(
    train_size: int, test_size: int, d: int, T: int, seed: int, rho: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, Y = qb.classical_sequences(T, d, n_seq=train_size + test_size, seed=seed, rho=rho)
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# --------------------------------------------------------------------------- #
#  shared-pipeline Adam training                                               #
# --------------------------------------------------------------------------- #
def make_model(name: str, kernel: str, d: int, k: int, layers: int, loss: str,
               v_isometric: bool = True) -> qb.Model:
    return qb.Model(name, d, kernel, k, layers=layers, loss=loss, v_isometric=v_isometric)


def _mean_metric(model: qb.Model, v: jnp.ndarray, Xs: jnp.ndarray, Ys: jnp.ndarray,
                 key: str) -> jnp.ndarray:
    W, V = model.build(v, xp=jnp)
    T = Xs.shape[1]
    mask = jnp.tril(jnp.ones((T, T)))

    def one(X, Y):
        A, w, f, lam = qb.forward(X, Y, W, V, model.kernel, model.k, model.beta, mask, xp=jnp)
        return qb.metrics(A, w, f, lam, T, None, xp=jnp)[key]

    return jnp.mean(jax.vmap(one)(Xs, Ys))


def train_adam(
    model: qb.Model,
    Xs: np.ndarray,
    Ys: np.ndarray,
    Xte: np.ndarray,
    Yte: np.ndarray,
    *,
    max_epochs: int,
    lr: float,
    seed: int,
    batch_size: int,
    min_epochs: int,
    patience: int,
    loss_rel_tol: float,
    eval_every: int,
    verbose: bool = True,
) -> dict[str, Any]:
    """Adam on the shared qsa_bench loss; report L1/L_B from the same metrics path."""
    if not qb.HAS_JAX:
        raise RuntimeError("JAX is required for the HPC campaign (analytic gradients).")

    rng = np.random.default_rng(seed)
    v = jnp.asarray(model.init(rng), dtype=jnp.float64)
    m = jnp.zeros_like(v)
    vv = jnp.zeros_like(v)
    n_seq = Xs.shape[0]
    bs = min(batch_size, n_seq)
    Xs_j, Ys_j = jnp.asarray(Xs), jnp.asarray(Ys)
    Xte_j, Yte_j = jnp.asarray(Xte), jnp.asarray(Yte)
    loss_key = model.loss

    @jax.jit
    def step(v, m, vv, Xb, Yb, t):
        loss, g = jax.value_and_grad(
            lambda p: _mean_metric(model, p, Xb, Yb, loss_key)
        )(v)
        m = 0.9 * m + 0.1 * g
        vv = 0.999 * vv + 0.001 * (g * g)
        mhat = m / (1.0 - 0.9 ** t)
        vhat = vv / (1.0 - 0.999 ** t)
        v = v - lr * mhat / (jnp.sqrt(vhat) + 1e-8)
        return v, m, vv, loss

    hist: list[float] = []
    best_loss = float("inf")
    best_v = v
    best_epoch = 0
    stagnant = 0
    stop_reason = "max_epochs"
    t0 = time.time()

    def full_train_loss(params: jnp.ndarray) -> float:
        return float(_mean_metric(model, params, Xs_j, Ys_j, loss_key))

    for ep in range(1, max_epochs + 1):
        idx = rng.choice(n_seq, size=bs, replace=False)
        Xb = Xs_j[idx]
        Yb = Ys_j[idx]
        v, m, vv, loss = step(v, m, vv, Xb, Yb, ep)
        loss_f = float(loss)
        hist.append(loss_f)

        do_eval = (ep % max(eval_every, 1) == 0) or (ep == 1) or (ep == max_epochs)
        if do_eval:
            # Best checkpoint on FULL-batch train loss (avoids minibatch noise / init freeze).
            train_full = full_train_loss(v)
            if train_full < best_loss:
                sig = (not math.isfinite(best_loss)) or (
                    (best_loss - train_full) > loss_rel_tol * max(abs(best_loss), 1.0)
                )
                best_loss = train_full
                best_v = v
                best_epoch = ep
                if sig:
                    stagnant = 0
                elif ep >= min_epochs:
                    stagnant += 1
            elif ep >= min_epochs:
                stagnant += 1
            if verbose:
                print(f"      ep {ep:4d}  batch={loss_f:.4f}  train={train_full:.4f}  "
                      f"best={best_loss:.4f}", flush=True)

            if ep >= min_epochs and stagnant >= patience:
                stop_reason = f"converged (patience={patience})"
                break

    # Final metrics on full train/test from the shared path (best checkpoint).
    W, V = model.build(np.asarray(best_v))
    train_m = qb.batch_metrics(Xs, Ys, W, V, model.kernel, model.k, model.beta)
    test_m = qb.batch_metrics(Xte, Yte, W, V, model.kernel, model.k, model.beta)
    train_m = {kk: float(vv) for kk, vv in train_m.items()}
    test_m = {kk: float(vv) for kk, vv in test_m.items()}
    qb.check_invariants(train_m, Xs.shape[1], label=f"({model.name} seed={seed})",
                        verbose=verbose)

    if verbose:
        print(
            f"    [{model.name:>10}] {loss_key} {hist[0]:8.4f} -> {train_m[loss_key]:8.4f}  "
            f"L1={train_m['L1']:.4f}/{test_m['L1']:.4f}  "
            f"({model.n_params()} params, ep={best_epoch}/{len(hist)}, "
            f"{time.time() - t0:.1f}s, {stop_reason})",
            flush=True,
        )
    return {
        "seed": int(seed),
        "n_params": int(model.n_params()),
        "epochs_run": int(len(hist)),
        "best_epoch": int(best_epoch),
        "best_batch_loss": float(best_loss),
        "converged": stop_reason.startswith("converged"),
        "stop_reason": stop_reason,
        "loss_history": [float(x) for x in hist[:: max(1, len(hist) // 200)]],
        "train_L1": train_m["L1"],
        "test_L1": test_m["L1"],
        "train_LB": train_m["L_B"],
        "test_LB": test_m["L_B"],
        "train_L_half": train_m["L_half"],
        "test_L_half": test_m["L_half"],
        "train_mu": train_m["mu"],
        "test_mu": test_m["mu"],
    }


# --------------------------------------------------------------------------- #
#  aggregates / plots / resume                                                 #
# --------------------------------------------------------------------------- #
def _agg_stats(runs: list[dict[str, Any]], key: str) -> tuple[float, float]:
    vals = np.asarray([r[key] for r in runs], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=0))


def aggregate_runs(runs: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta)
    out["n_seeds"] = len(runs)
    for key in ("train_L1", "test_L1", "train_LB", "test_LB", "train_L_half", "test_L_half"):
        mean, std = _agg_stats(runs, key)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    out["epochs_run_mean"] = float(np.mean([r["epochs_run"] for r in runs]))
    out["converged_fraction"] = float(np.mean([1.0 if r["converged"] else 0.0 for r in runs]))
    out["n_params"] = int(runs[0]["n_params"])
    out["runs"] = runs
    return out


def _load_complete(path: Path, n_seeds: int, n_params: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if int(data.get("n_seeds", 0)) < n_seeds:
        return None
    if int(data.get("n_params", -1)) != int(n_params):
        return None
    # Truncate to requested n_seeds if a longer pack exists.
    if len(data.get("runs", [])) > n_seeds:
        data = aggregate_runs(data["runs"][:n_seeds], {k: v for k, v in data.items()
                                                       if k != "runs"})
    return data


def plot_vs_k(
    points: list[dict[str, Any]],
    out_path: Path,
    title: str,
    y_mean: str,
    y_std: str,
    ylabel: str,
    nl_refs: list[dict[str, Any]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in points:
        by_model.setdefault(row["display"], []).append(row)
    all_ks: list[int] = []
    for name, rows in by_model.items():
        rows = sorted(rows, key=lambda r: int(r["k"]))
        ks = [int(r["k"]) for r in rows]
        all_ks.extend(ks)
        means = [float(r[y_mean]) for r in rows]
        stds = [float(r[y_std]) for r in rows]
        st = PLOT_STYLES.get(name, dict(color="0.3", linestyle="-", marker="o", linewidth=2.2))
        n_params = rows[0].get("n_params")
        label = f"{name} (n={n_params})" if n_params is not None else name
        ax.errorbar(
            ks, means, yerr=stds,
            color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
            linewidth=st["linewidth"], capsize=4, label=label,
        )
    if nl_refs and all_ks:
        xmin, xmax = min(all_ks), max(all_ks)
        xs = np.linspace(xmin, xmax, 64)
        for ref in nl_refs:
            name = str(ref["display"])
            mean = float(ref[y_mean])
            std = float(ref[y_std])
            st = PLOT_STYLES.get(name, dict(color="0.1", linestyle="-", linewidth=2.0))
            n_params = ref.get("n_params")
            label = f"{name} (k-indep., n={n_params})" if n_params is not None else f"{name} (k-indep.)"
            ax.axhline(mean, color=st["color"], linestyle=st["linestyle"],
                       linewidth=st["linewidth"], label=label)
            ax.fill_between(xs, mean - std, mean + std, color=st["color"], alpha=0.10, linewidth=0)
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  one world (classical or quantum)                                            #
# --------------------------------------------------------------------------- #
def run_world(
    world: str,
    Xs: np.ndarray,
    Ys: np.ndarray,
    Xte: np.ndarray,
    Yte: np.ndarray,
    *,
    ks: list[int],
    layers: int,
    n_seeds: int,
    model_seed_base: int,
    epochs: int,
    poly_epochs: int,
    nl_epochs: int,
    lr: float,
    batch_size: int,
    min_epochs: int,
    patience: int,
    loss_rel_tol: float,
    eval_every: int,
    out_dir: Path,
) -> dict[str, Any]:
    d = Xs.shape[-1]
    aggs_dir = out_dir / "aggregates"
    plots_dir = out_dir / "plots"
    aggs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88, flush=True)
    print(f"{world.upper()}  T={Xs.shape[1]}  d={d}  n_train={Xs.shape[0]}  "
          f"n_test={Xte.shape[0]}  ks={ks}  n_seeds={n_seeds}  layers={layers}",
          flush=True)
    print("=" * 88, flush=True)

    l1_train_pts: list[dict[str, Any]] = []
    l1_test_pts: list[dict[str, Any]] = []
    lb_train_pts: list[dict[str, Any]] = []
    lb_test_pts: list[dict[str, Any]] = []
    nl_refs_train: list[dict[str, Any]] = []
    nl_refs_test: list[dict[str, Any]] = []
    param_counts: dict[str, int] = {}

    # --- k-models ---
    for name, kernel, loss in K_MODELS:
        display = DISPLAY[name]
        max_ep = poly_epochs if "poly" in name else epochs
        for k in ks:
            model0 = make_model(name, kernel, d, k, layers, loss)
            n_params = model0.n_params()
            param_counts[display] = n_params
            agg_path = aggs_dir / f"{name}_k{k}.json"
            loaded = _load_complete(agg_path, n_seeds, n_params)
            if loaded is not None:
                print(f"  [resume] {name} k={k} ({n_seeds} seeds)", flush=True)
                agg = loaded
            else:
                runs = []
                for i in range(n_seeds):
                    seed = model_seed_base + i
                    print(f"  [{world}] {name} k={k} seed={seed} ({i+1}/{n_seeds})", flush=True)
                    model = make_model(name, kernel, d, k, layers, loss)
                    runs.append(train_adam(
                        model, Xs, Ys, Xte, Yte,
                        max_epochs=max_ep, lr=lr, seed=seed, batch_size=batch_size,
                        min_epochs=min_epochs, patience=patience,
                        loss_rel_tol=loss_rel_tol, eval_every=eval_every,
                    ))
                agg = aggregate_runs(runs, {
                    "world": world, "model": name, "display": display,
                    "kernel": kernel, "k": k, "loss": loss,
                })
                agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")

            pt = {
                "model": name, "display": display, "k": k,
                "n_params": agg["n_params"],
                "train_L1_mean": agg["train_L1_mean"], "train_L1_std": agg["train_L1_std"],
                "test_L1_mean": agg["test_L1_mean"], "test_L1_std": agg["test_L1_std"],
                "train_LB_mean": agg["train_LB_mean"], "train_LB_std": agg["train_LB_std"],
                "test_LB_mean": agg["test_LB_mean"], "test_LB_std": agg["test_LB_std"],
            }
            l1_train_pts.append(pt)
            l1_test_pts.append(pt)
            lb_train_pts.append(pt)
            lb_test_pts.append(pt)

    # --- nl-models (k-independent; train once) ---
    for name, kernel, loss, v_iso in NL_MODELS:
        display = DISPLAY[name]
        # soft ignores k; use k=1 placeholder
        model0 = make_model(name, kernel, d, 1, layers, loss, v_isometric=v_iso)
        n_params = model0.n_params()
        param_counts[display] = n_params
        agg_path = aggs_dir / f"{name}.json"
        loaded = _load_complete(agg_path, n_seeds, n_params)
        if loaded is not None:
            print(f"  [resume] {name} ({n_seeds} seeds)", flush=True)
            agg = loaded
        else:
            runs = []
            for i in range(n_seeds):
                seed = model_seed_base + i
                print(f"  [{world}] {name} seed={seed} ({i+1}/{n_seeds})", flush=True)
                model = make_model(name, kernel, d, 1, layers, loss, v_isometric=v_iso)
                runs.append(train_adam(
                    model, Xs, Ys, Xte, Yte,
                    max_epochs=nl_epochs, lr=lr, seed=seed, batch_size=batch_size,
                    min_epochs=min_epochs, patience=patience,
                    loss_rel_tol=loss_rel_tol, eval_every=eval_every,
                ))
            agg = aggregate_runs(runs, {
                "world": world, "model": name, "display": display,
                "kernel": kernel, "k": None, "loss": loss,
            })
            agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")

        ref = {
            "model": name, "display": display, "n_params": agg["n_params"],
            "train_L1_mean": agg["train_L1_mean"], "train_L1_std": agg["train_L1_std"],
            "test_L1_mean": agg["test_L1_mean"], "test_L1_std": agg["test_L1_std"],
        }
        nl_refs_train.append(ref)
        nl_refs_test.append(ref)

    # --- plots (NO chance) ---
    plot_vs_k(
        l1_train_pts, plots_dir / "train_L1_vs_k.png",
        f"{world}: train $L_1$ vs $k$ (shared pipeline)",
        "train_L1_mean", "train_L1_std", r"train $L_1$",
        nl_refs=nl_refs_train,
    )
    plot_vs_k(
        l1_test_pts, plots_dir / "test_L1_vs_k.png",
        f"{world}: test $L_1$ vs $k$ (shared pipeline)",
        "test_L1_mean", "test_L1_std", r"test $L_1$",
        nl_refs=nl_refs_test,
    )
    plot_vs_k(
        lb_train_pts, plots_dir / "train_LB_vs_k.png",
        f"{world}: train $\\mathcal{{L}}_B$ vs $k$ (k-models only)",
        "train_LB_mean", "train_LB_std", r"train $\mathcal{L}_B$",
        nl_refs=None,
    )
    plot_vs_k(
        lb_test_pts, plots_dir / "test_LB_vs_k.png",
        f"{world}: test $\\mathcal{{L}}_B$ vs $k$ (k-models only)",
        "test_LB_mean", "test_LB_std", r"test $\mathcal{L}_B$",
        nl_refs=None,
    )

    summary = {
        "world": world,
        "T": int(Xs.shape[1]),
        "d": int(d),
        "n_train": int(Xs.shape[0]),
        "n_test": int(Xte.shape[0]),
        "ks": ks,
        "n_seeds": n_seeds,
        "layers": layers,
        "param_counts": param_counts,
        "l1_train_points": l1_train_pts,
        "l1_test_points": l1_test_pts,
        "lb_train_points": lb_train_pts,
        "lb_test_points": lb_test_pts,
        "nl_refs_train": nl_refs_train,
        "nl_refs_test": nl_refs_test,
        "plots": {
            "train_L1": str(plots_dir / "train_L1_vs_k.png"),
            "test_L1": str(plots_dir / "test_L1_vs_k.png"),
            "train_LB": str(plots_dir / "train_LB_vs_k.png"),
            "test_LB": str(plots_dir / "test_LB_vs_k.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir / 'summary.json'}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Clean-slate qsa_bench HPC campaign (classical + quantum, multi-seed, k<=6)."
    )
    p.add_argument("--data-mode", choices=["both", "classical", "quantum"], default="both")
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--dt", type=float, default=0.35)
    p.add_argument("--classical-rho", type=float, default=0.8)
    p.add_argument("--ks", type=str, default="1,2,3,4,5,6")
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--poly-epochs", type=int, default=600)
    p.add_argument("--nl-epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--train-size", type=int, default=256)
    p.add_argument("--test-size", type=int, default=128)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--seed", type=int, default=7, help="Data seed")
    p.add_argument("--model-seed-base", type=int, default=1042)
    p.add_argument("--min-epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--loss-rel-tol", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--output-dir", type=str,
                   default="results/qsa_bench_campaign/LB_L1_shared")
    p.add_argument("--quick", action="store_true",
                   help="Tiny smoke: T=8 d=8, k=1,2, 2 seeds, few epochs.")
    args = p.parse_args(argv)

    if args.quick:
        args.T = 8
        args.d = 8
        args.n_qubits = 3
        args.ks = "1,2"
        args.layers = 4
        args.epochs = 40
        args.poly_epochs = 50
        args.nl_epochs = 40
        args.train_size = 16
        args.test_size = 8
        args.n_seeds = 2
        args.min_epochs = 10
        args.patience = 8
        args.batch_size = 8
        args.eval_every = 5
        args.output_dir = str(Path(args.output_dir).parent / "quick_smoke")

    if args.d & (args.d - 1):
        raise ValueError(f"d must be a power of two for the complex ansatz, got {args.d}")
    if args.data_mode in ("both", "quantum"):
        if 2 ** args.n_qubits != args.d:
            args.n_qubits = int(round(math.log2(args.d)))
        if 2 ** args.n_qubits != args.d:
            raise ValueError(f"quantum requires d=2^n_qubits, got d={args.d}, n_qubits={args.n_qubits}")

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"HAS_JAX={qb.HAS_JAX}  output={out}", flush=True)
    if not qb.HAS_JAX:
        raise SystemExit("JAX required. Activate the Leonardo venv / install jax.")

    # Self-check once before the long campaign.
    if not args.quick:
        ok = qb.selftest()
        if not ok:
            raise SystemExit("qsa_bench selftest FAILED — aborting campaign.")
        print(flush=True)

    common = dict(
        ks=ks, layers=args.layers, n_seeds=args.n_seeds,
        model_seed_base=args.model_seed_base, epochs=args.epochs,
        poly_epochs=args.poly_epochs, nl_epochs=args.nl_epochs, lr=args.lr,
        batch_size=args.batch_size, min_epochs=args.min_epochs,
        patience=args.patience, loss_rel_tol=args.loss_rel_tol,
        eval_every=args.eval_every,
    )
    summaries: dict[str, Any] = {"config": vars(args), "HAS_JAX": qb.HAS_JAX}

    if args.data_mode in ("both", "classical"):
        Xtr, Ytr, Xte, Yte = generate_classical_xy(
            args.train_size, args.test_size, args.d, args.T, args.seed, args.classical_rho,
        )
        summaries["classical"] = run_world(
            "classical", Xtr, Ytr, Xte, Yte, out_dir=out / "classical", **common,
        )

    if args.data_mode in ("both", "quantum"):
        Xtr, Ytr, Xte, Yte = generate_quantum_xy(
            args.train_size, args.test_size, args.n_qubits, args.T, args.dt, args.seed + 100,
        )
        summaries["quantum"] = run_world(
            "quantum", Xtr, Ytr, Xte, Yte, out_dir=out / "quantum", **common,
        )

    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nDone. Top-level summary: {out / 'summary.json'}", flush=True)
    print("Plots:", flush=True)
    for world in ("classical", "quantum"):
        if world in summaries:
            for k, v in summaries[world]["plots"].items():
                print(f"  {world}/{k}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
