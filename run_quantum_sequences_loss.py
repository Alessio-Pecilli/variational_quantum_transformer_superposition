#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

PLOT_STYLES = {
    "k-QSA L=16": dict(color="#0072B2", linestyle="-", marker="o", linewidth=2.4),
    "k-CSA": dict(color="#E69F00", linestyle="-", marker="s", linewidth=2.4),
    "poly-k-QSA L=16": dict(color="#0072B2", linestyle="--", marker="^", linewidth=2.2),
    "poly-k-CSA": dict(color="#E69F00", linestyle="--", marker="v", linewidth=2.2),
}


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
    dim = 2**n_qubits
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


def _unitary_from_hamiltonian(h: np.ndarray, dt: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(h)
    phase = np.exp(-1j * dt * evals)
    return evecs @ np.diag(phase) @ evecs.conj().T


def _clamp_global_phase(states: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    ref = states[..., 0]
    abs_ref = np.abs(ref)
    unit = np.where(abs_ref > eps, np.conj(ref / abs_ref), np.ones_like(ref))
    return states * unit[..., None]


def _sample_haar_states(rng: np.random.Generator, n_samples: int, dim: int) -> np.ndarray:
    real = rng.standard_normal((n_samples, dim))
    imag = rng.standard_normal((n_samples, dim))
    x = real + 1j * imag
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return _clamp_global_phase(x)


@dataclass
class QuantumDataset:
    train_states: np.ndarray
    test_states: np.ndarray
    train_params: np.ndarray
    test_params: np.ndarray


def generate_quantum_dataset(
    train_size: int,
    test_size: int,
    n_qubits: int,
    T: int,
    dt: float,
    seed: int,
) -> QuantumDataset:
    dim = 2**n_qubits
    n_total = train_size + test_size
    rng_s = np.random.default_rng(seed)
    rng_p = np.random.default_rng(seed + 1)
    init_states = _sample_haar_states(rng_s, n_total, dim)
    params = rng_p.uniform(0.2, 2.0, size=(n_total, 2))
    states = np.zeros((n_total, T + 1, dim), dtype=np.complex128)
    states[:, 0, :] = init_states
    for idx in range(n_total):
        j_val, h_val = float(params[idx, 0]), float(params[idx, 1])
        h = build_tfim_hamiltonian(n_qubits=n_qubits, coupling_j=j_val, field_h=h_val)
        u = _unitary_from_hamiltonian(h, dt)
        cur = init_states[idx]
        for t in range(1, T + 1):
            cur = u @ cur
            cur = cur / np.linalg.norm(cur)
            states[idx, t, :] = cur
    states = _clamp_global_phase(states)
    return QuantumDataset(
        train_states=states[:train_size],
        test_states=states[train_size:],
        train_params=params[:train_size],
        test_params=params[train_size:],
    )


def _complex_ansatz_matrix(params: jnp.ndarray) -> jnp.ndarray:
    layers, n, _ = params.shape
    dim = 2**n

    def rx(t):
        return jnp.array(
            [[jnp.cos(t / 2), -1j * jnp.sin(t / 2)], [-1j * jnp.sin(t / 2), jnp.cos(t / 2)]],
            dtype=jnp.complex128,
        )

    def ry(t):
        return jnp.array(
            [[jnp.cos(t / 2), -jnp.sin(t / 2)], [jnp.sin(t / 2), jnp.cos(t / 2)]], dtype=jnp.complex128
        )

    def rz(t):
        return jnp.array([[jnp.exp(-1j * t / 2), 0], [0, jnp.exp(1j * t / 2)]], dtype=jnp.complex128)

    def cnot(a, b, nq):
        out = np.zeros((2**nq, 2**nq), dtype=np.complex128)
        for x in range(2**nq):
            bits = [(x >> (nq - 1 - w)) & 1 for w in range(nq)]
            if bits[a] == 1:
                bits[b] ^= 1
            y = sum(bt << (nq - 1 - w) for w, bt in enumerate(bits))
            out[y, x] = 1.0
        return jnp.asarray(out)

    m = jnp.eye(dim, dtype=jnp.complex128)
    for l in range(layers):
        kron_gate = jnp.array([1.0 + 0.0j])
        for q in range(n):
            sq = rz(params[l, q, 2]) @ ry(params[l, q, 1]) @ rx(params[l, q, 0])
            kron_gate = jnp.kron(kron_gate, sq)
        m = kron_gate @ m
        if n >= 2:
            for a, b in zip(range(n), list(range(1, n)) + [0]):
                m = cnot(a, b, n) @ m
    return m


def _unitary_from_raw(raw: jnp.ndarray) -> jnp.ndarray:
    q, r = jnp.linalg.qr(raw)
    d = jnp.diag(r)
    ph = d / jnp.where(jnp.abs(d) > 1e-12, jnp.abs(d), 1.0)
    return q * jnp.conj(ph)[None, :]


def _lcu_coeffs(k: int, d: int) -> jnp.ndarray:
    beta = jnp.sqrt(float(d))
    vals = [beta**p / math.factorial(p) for p in range(k + 1)]
    return jnp.asarray(vals, dtype=jnp.float64)


def _kernel_matrix(s: jnp.ndarray, k: int, mode: str, d: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    if mode == "monomial":
        return s**k, jnp.asarray(1.0, dtype=jnp.float64)
    c = _lcu_coeffs(k, d)
    g = jnp.zeros_like(s)
    s_pow = jnp.ones_like(s)
    for p in range(k + 1):
        if p > 0:
            s_pow = s_pow * s
        g = g + c[p] * s_pow
    return g, jnp.sum(c)


def _wv_from_params(params: dict[str, jnp.ndarray], d: int, family: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    if family == "kqsa":
        w = _complex_ansatz_matrix(params["wp"])[:d, :d]
        v = _complex_ansatz_matrix(params["vp"])[:d, :d]
    else:
        w = _unitary_from_raw(params["w_raw"])
        v = _unitary_from_raw(params["v_raw"])
    return w, v


def _coherent_mu_for_sequence(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    v: jnp.ndarray,
    phi: jnp.ndarray,
    k: int,
    mode: str,
) -> jnp.ndarray:
    t, d = x.shape
    s = jnp.conj(x) @ (w @ x.T)
    a = jnp.conj(y) @ (v @ x.T)
    kern, lam = _kernel_matrix(s, k, mode, d)
    tri = jnp.tril(jnp.ones((t, t), dtype=jnp.float64))
    aj = jnp.sum(a * kern * tri, axis=1)
    amp = jnp.sum(jnp.exp(1j * phi[:t]) * aj)
    ntri = t * (t + 1) / 2.0
    mu = jnp.abs(amp / (lam * ntri)) ** 2
    return jnp.real(mu)


def _batch_mu_loss(
    params: dict[str, jnp.ndarray],
    states: jnp.ndarray,
    k: int,
    mode: str,
    family: str,
) -> jnp.ndarray:
    d = states.shape[-1]
    w, v = _wv_from_params(params, d, family)
    phi = params["phi"]
    y_batch = states[:, 1:, :]
    x_batch = states[:, :-1, :]
    mu_vals = jax.vmap(lambda xs, ys: _coherent_mu_for_sequence(xs, ys, w, v, phi, k, mode))(x_batch, y_batch)
    return jnp.mean(-jnp.log(jnp.maximum(mu_vals, 1e-12)))


@jax.jit(static_argnames=("k", "mode", "family"))
def _adam_step(
    params,
    moments,
    x_batch,
    k: int,
    mode: str,
    family: str,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
):
    loss, grads = jax.value_and_grad(_batch_mu_loss)(params, x_batch, k, mode, family)
    m, v = moments
    m = jax.tree_util.tree_map(lambda mm, g: beta1 * mm + (1.0 - beta1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda vv, g: beta2 * vv + (1.0 - beta2) * (jnp.abs(g) ** 2), v, grads)
    b1_corr = 1.0 - beta1**step
    b2_corr = 1.0 - beta2**step
    params = jax.tree_util.tree_map(
        lambda p, mm, vv: p - lr * (mm / b1_corr) / (jnp.sqrt(vv / b2_corr) + eps), params, m, v
    )
    return params, (m, v), loss


def _copy_params(params: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    return jax.tree_util.tree_map(lambda p: jnp.array(p), params)


def _init_params(family: str, d: int, T: int, layers: int, seed: int) -> dict[str, jnp.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(round(math.log2(d)))
    if family == "kqsa":
        return {
            "wp": jnp.asarray(rng.standard_normal((layers, n, 3)), dtype=jnp.float64),
            "vp": jnp.asarray(rng.standard_normal((layers, n, 3)), dtype=jnp.float64),
            "phi": jnp.zeros((T,), dtype=jnp.float64),
        }
    w_raw = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    v_raw = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    return {
        "w_raw": jnp.asarray(w_raw, dtype=jnp.complex128),
        "v_raw": jnp.asarray(v_raw, dtype=jnp.complex128),
        "phi": jnp.zeros((T,), dtype=jnp.float64),
    }


def train_model(
    train_states: np.ndarray,
    test_states: np.ndarray,
    family: str,
    kernel_mode: str,
    k: int,
    layers: int,
    max_epochs: int,
    lr: float,
    seed: int,
    min_epochs: int = 40,
    patience: int = 30,
    loss_rel_tol: float = 1e-4,
    eval_every: int = 5,
) -> dict[str, Any]:
    _, T_plus, d = train_states.shape
    T = T_plus - 1
    params = _init_params(family, d, T, layers, seed)
    m0 = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
    v0 = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
    train_batch = jnp.asarray(train_states, dtype=jnp.complex128)
    test_batch = jnp.asarray(test_states, dtype=jnp.complex128)

    losses: list[float] = []
    test_losses: list[float] = []
    test_epochs: list[int] = []
    moments = (m0, v0)
    best_loss = float("inf")
    best_params = _copy_params(params)
    stagnant = 0
    stop_reason = "max_epochs"

    for ep in range(1, max_epochs + 1):
        params, moments, loss = _adam_step(
            params, moments, train_batch, k, kernel_mode, family, lr, 0.9, 0.999, 1e-8, ep
        )
        loss_f = float(loss)
        losses.append(loss_f)

        if ep % eval_every == 0 or ep == max_epochs:
            test_loss = float(_batch_mu_loss(params, test_batch, k, kernel_mode, family))
            test_losses.append(test_loss)
            test_epochs.append(ep)

        if loss_f + loss_rel_tol * max(abs(best_loss), 1.0) < best_loss:
            best_loss = loss_f
            best_params = _copy_params(params)
            stagnant = 0
        elif ep >= min_epochs:
            stagnant += 1

        if ep >= min_epochs and stagnant >= patience:
            stop_reason = f"converged (patience={patience}, rel_tol={loss_rel_tol})"
            break

    final_train = float(_batch_mu_loss(best_params, train_batch, k, kernel_mode, family))
    final_test_mu = float(_batch_mu_loss(best_params, test_batch, k, kernel_mode, family))
    aligned_history = [float(x + math.log(T)) for x in losses]
    aligned_test_history = [float(x + math.log(T)) for x in test_losses]

    return {
        "params": best_params,
        "loss_history": losses,
        "aligned_history": aligned_history,
        "test_loss_history": test_losses,
        "test_loss_epochs": test_epochs,
        "aligned_test_history": aligned_test_history,
        "final_loss": final_train,
        "final_test_mu_loss": final_test_mu,
        "epochs_run": len(losses),
        "best_epoch_train_loss": float(best_loss),
        "converged": stop_reason.startswith("converged"),
        "stop_reason": stop_reason,
    }


def _predict_last_fidelity(
    states: np.ndarray, params: dict[str, jnp.ndarray], family: str, mode: str, k: int
) -> float:
    x = jnp.asarray(states[:, :-1, :], dtype=jnp.complex128)
    y_true = jnp.asarray(states[:, -1, :], dtype=jnp.complex128)
    d = x.shape[-1]
    w, v = _wv_from_params(params, d, family)

    def one(seq, yt):
        s_row = jnp.conj(seq[-1:]) @ (w @ seq.T)
        kern, _ = _kernel_matrix(s_row, k, mode, d)
        vals = seq @ v.T
        z = jnp.sum(kern[0][:, None] * vals, axis=0)
        z = z / jnp.maximum(jnp.linalg.norm(z), 1e-12)
        yt = yt / jnp.maximum(jnp.linalg.norm(yt), 1e-12)
        return jnp.abs(jnp.vdot(yt, z)) ** 2

    fids = jax.vmap(one)(x, y_true)
    return float(jnp.mean(fids))


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    hist = np.array([r["aligned_history"] for r in runs], dtype=float)
    finals = np.array([r["aligned_loss"] for r in runs], dtype=float)
    test_fid = np.array([r["test_neglog_fid"] for r in runs], dtype=float)
    test_mu = np.array([r["aligned_test_mu_loss"] for r in runs], dtype=float)
    epochs = np.array([r["epochs_run"] for r in runs], dtype=float)
    return {
        "aligned_history": hist.mean(axis=0).tolist(),
        "aligned_std": hist.std(axis=0).tolist(),
        "aligned_loss_mean": float(finals.mean()),
        "aligned_loss_std": float(finals.std()),
        "aligned_test_fidelity_loss_mean": float(test_fid.mean()),
        "aligned_test_fidelity_loss_std": float(test_fid.std()),
        "aligned_test_mu_loss_mean": float(test_mu.mean()),
        "aligned_test_mu_loss_std": float(test_mu.std()),
        "epochs_run_mean": float(epochs.mean()),
        "epochs_run_std": float(epochs.std()),
        "converged_fraction": float(np.mean([1.0 if r["converged"] else 0.0 for r in runs])),
    }


def plot_vs_k(
    mu_points: list[dict[str, Any]],
    out_path: Path,
    title: str,
    ykey_mean: str,
    ykey_std: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in mu_points:
        by_model.setdefault(row["model"], []).append(row)
    for name, rows in by_model.items():
        rows = sorted(rows, key=lambda x: int(x["k"]))
        ks = [int(r["k"]) for r in rows]
        means = [float(r[ykey_mean]) for r in rows]
        stds = [float(r[ykey_std]) for r in rows]
        st = PLOT_STYLES.get(name, dict(color="0.3", linestyle="-", marker="o", linewidth=2.2))
        ax.errorbar(
            ks,
            means,
            yerr=stds,
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            linewidth=st["linewidth"],
            capsize=4,
            label=name,
        )
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_convergence_curves(aggs: list[dict[str, Any]], out_path: Path, T: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=False)
    for agg in aggs:
        name = agg["model"]
        st = PLOT_STYLES.get(name, dict(color="0.3", linestyle="-", linewidth=2.0))
        train = np.asarray(agg["aligned_history"], dtype=float)
        xs = np.arange(1, len(train) + 1)
        axes[0].plot(xs, train, color=st["color"], linestyle=st["linestyle"], linewidth=2.0, label=name)
        if "aligned_std" in agg:
            std = np.asarray(agg["aligned_std"], dtype=float)
            if len(std) == len(train):
                axes[0].fill_between(xs, train - std, train + std, color=st["color"], alpha=0.12, linewidth=0)
        if agg.get("test_loss_epochs") and agg.get("aligned_test_history"):
            tx = np.asarray(agg["test_loss_epochs"], dtype=int)
            ty = np.asarray(agg["aligned_test_history"], dtype=float)
            axes[1].plot(tx, ty, color=st["color"], linestyle=st["linestyle"], linewidth=2.0, label=name)
    axes[0].set_title(f"Train aligned loss (-log mu + log T), T={T}")
    axes[0].set_xlabel("epoch")
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Test aligned mu-loss (same metric, held-out TFIM)")
    axes[1].set_xlabel("epoch")
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantum-sequences k sweep (mono+poly)")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--ks", type=str, default="1,2,3,5,6")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--poly-epochs", type=int, default=600)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--loss-rel-tol", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-seed-base", type=int, default=1042)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results_from_hpc/final_campaign_quantum_sequences/loss")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.train_size = 64
        args.test_size = 32
        args.T = 8
        args.d = 16
        args.n_qubits = 4
        args.ks = "1,2"
        args.epochs = 40
        args.poly_epochs = 50
        args.min_epochs = 10
        args.patience = 8
        args.n_seeds = 2
        args.layers = 2
        args.eval_every = 2

    if args.d != 16:
        raise ValueError("This quantum-sequences campaign is configured for d=16.")
    if args.n_qubits != 4:
        raise ValueError("d=16 requires n_qubits=4.")

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    out = Path(args.output_dir)
    plots = out / "plots"
    aggs_dir = out / "aggregates"
    plots.mkdir(parents=True, exist_ok=True)
    aggs_dir.mkdir(parents=True, exist_ok=True)

    dataset = generate_quantum_dataset(
        train_size=args.train_size,
        test_size=args.test_size,
        n_qubits=args.n_qubits,
        T=args.T,
        dt=args.dt,
        seed=args.seed,
    )
    model_specs = [
        ("k-QSA L=16", "kqsa", "monomial", args.epochs),
        ("k-CSA", "kcsa", "monomial", args.epochs),
        ("poly-k-QSA L=16", "kqsa", "poly", args.poly_epochs),
        ("poly-k-CSA", "kcsa", "poly", args.poly_epochs),
    ]
    mu_points_train = []
    mu_points_test_fid = []
    mu_points_test_mu = []
    convergence_aggs: list[dict[str, Any]] = []

    for display, family, mode, max_epochs in model_specs:
        for k in ks:
            runs = []
            for s_idx in range(args.n_seeds):
                mseed = args.model_seed_base + s_idx
                trained = train_model(
                    train_states=dataset.train_states,
                    test_states=dataset.test_states,
                    family=family,
                    kernel_mode=mode,
                    k=k,
                    layers=args.layers,
                    max_epochs=max_epochs,
                    lr=args.lr,
                    seed=mseed + 17 * k,
                    min_epochs=args.min_epochs,
                    patience=args.patience,
                    loss_rel_tol=args.loss_rel_tol,
                    eval_every=args.eval_every,
                )
                test_fid = _predict_last_fidelity(
                    dataset.test_states, trained["params"], family=family, mode=mode, k=k
                )
                runs.append(
                    {
                        "seed": mseed,
                        "loss_history": trained["loss_history"],
                        "aligned_history": trained["aligned_history"],
                        "test_loss_history": trained["test_loss_history"],
                        "test_loss_epochs": trained["test_loss_epochs"],
                        "aligned_test_history": trained["aligned_test_history"],
                        "aligned_loss": float(trained["final_loss"] + math.log(args.T)),
                        "aligned_test_mu_loss": float(trained["final_test_mu_loss"] + math.log(args.T)),
                        "test_neglog_fid": float(-math.log(max(test_fid, 1e-12))),
                        "epochs_run": trained["epochs_run"],
                        "converged": trained["converged"],
                        "stop_reason": trained["stop_reason"],
                    }
                )
            agg = aggregate(runs)
            payload = {"model": display, "family": family, "kernel_mode": mode, "k": k, "runs": runs, **agg}
            (aggs_dir / f"{family}_{mode}_k{k}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            mu_points_train.append(
                {
                    "model": display,
                    "k": k,
                    "aligned_loss_mean": agg["aligned_loss_mean"],
                    "aligned_loss_std": agg["aligned_loss_std"],
                }
            )
            mu_points_test_fid.append(
                {
                    "model": display,
                    "k": k,
                    "aligned_test_fidelity_loss_mean": agg["aligned_test_fidelity_loss_mean"],
                    "aligned_test_fidelity_loss_std": agg["aligned_test_fidelity_loss_std"],
                }
            )
            mu_points_test_mu.append(
                {
                    "model": display,
                    "k": k,
                    "aligned_test_mu_loss_mean": agg["aligned_test_mu_loss_mean"],
                    "aligned_test_mu_loss_std": agg["aligned_test_mu_loss_std"],
                }
            )
            if k == ks[len(ks) // 2]:
                convergence_aggs.append(
                    {
                        "model": display,
                        "aligned_history": agg["aligned_history"],
                        "aligned_std": agg["aligned_std"],
                        "test_loss_epochs": runs[0]["test_loss_epochs"],
                        "aligned_test_history": runs[0]["aligned_test_history"],
                    }
                )
            print(
                f"[{display} k={k}] train={agg['aligned_loss_mean']:.4f}±{agg['aligned_loss_std']:.4f} | "
                f"test_mu={agg['aligned_test_mu_loss_mean']:.4f}±{agg['aligned_test_mu_loss_std']:.4f} | "
                f"test_-logF={agg['aligned_test_fidelity_loss_mean']:.4f}±{agg['aligned_test_fidelity_loss_std']:.4f} | "
                f"epochs~{agg['epochs_run_mean']:.0f} conv={agg['converged_fraction']:.0%}"
            )

    plot_vs_k(
        mu_points_train,
        plots / "final_aligned_loss_vs_k.png",
        f"FINAL aligned loss vs k (T={args.T}, d={args.d}; train; mono+poly)",
        "aligned_loss_mean",
        "aligned_loss_std",
        r"aligned loss: $-\log\mu+\log T$",
    )
    plot_vs_k(
        mu_points_test_mu,
        plots / "final_aligned_loss_vs_k_test_mu.png",
        f"FINAL aligned loss vs k (T={args.T}, d={args.d}; held-out TFIM; test mu-loss)",
        "aligned_test_mu_loss_mean",
        "aligned_test_mu_loss_std",
        r"test aligned loss: $-\log\mu+\log T$",
    )
    plot_vs_k(
        mu_points_test_fid,
        plots / "final_aligned_loss_vs_k_test.png",
        f"FINAL aligned loss vs k (T={args.T}, d={args.d}; held-out TFIM; y=-log prediction fidelity)",
        "aligned_test_fidelity_loss_mean",
        "aligned_test_fidelity_loss_std",
        r"test loss: $-\log(\mathrm{prediction\ fidelity})$",
    )
    if convergence_aggs:
        plot_convergence_curves(
            convergence_aggs,
            plots / "training_convergence_train_vs_test.png",
            T=args.T,
        )

    summary = {
        "config": vars(args),
        "mu_points_vs_k": mu_points_train,
        "mu_points_vs_k_test_mu": mu_points_test_mu,
        "mu_points_vs_k_test_fidelity": mu_points_test_fid,
        "dataset": {
            "train_shape": list(dataset.train_states.shape),
            "test_shape": list(dataset.test_states.shape),
            "note": "Hamiltonian parameters (J,h) vary per trajectory and are never given as model inputs.",
        },
        "training_note": (
            "Early stopping on train -log(mu) with patience; best checkpoint used for test metrics. "
            "Test mu-loss uses the same coherent readout as train; test fidelity is separate single-step T+1 metric."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
