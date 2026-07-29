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


def _kernel_matrix(s: jnp.ndarray, k: int, mode: str, d: int) -> tuple[jnp.ndarray, float]:
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


def _train_loss(params: dict[str, jnp.ndarray], x_batch: jnp.ndarray, k: int, mode: str, family: str) -> jnp.ndarray:
    if family == "kqsa":
        w = _complex_ansatz_matrix(params["wp"])[: x_batch.shape[-1], : x_batch.shape[-1]]
        v = _complex_ansatz_matrix(params["vp"])[: x_batch.shape[-1], : x_batch.shape[-1]]
    else:
        w = _unitary_from_raw(params["w_raw"])
        v = _unitary_from_raw(params["v_raw"])
    phi = params["phi"]
    y_batch = x_batch[:, 1:, :]
    in_batch = x_batch[:, :-1, :]
    mu_vals = jax.vmap(lambda xs, ys: _coherent_mu_for_sequence(xs, ys, w, v, phi, k, mode))(in_batch, y_batch)
    return jnp.mean(-jnp.log(jnp.maximum(mu_vals, 1e-12)))


@jax.jit(static_argnames=("k", "mode", "family"))
def _adam_step(params, moments, x_batch, k: int, mode: str, family: str, lr: float, beta1: float, beta2: float, eps: float, step: int):
    loss, grads = jax.value_and_grad(_train_loss)(params, x_batch, k, mode, family)
    m, v = moments
    m = jax.tree_util.tree_map(lambda mm, g: beta1 * mm + (1.0 - beta1) * g, m, grads)
    v = jax.tree_util.tree_map(lambda vv, g: beta2 * vv + (1.0 - beta2) * (jnp.abs(g) ** 2), v, grads)
    b1_corr = 1.0 - beta1**step
    b2_corr = 1.0 - beta2**step
    params = jax.tree_util.tree_map(
        lambda p, mm, vv: p - lr * (mm / b1_corr) / (jnp.sqrt(vv / b2_corr) + eps), params, m, v
    )
    return params, (m, v), loss


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
    family: str,
    kernel_mode: str,
    k: int,
    layers: int,
    epochs: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    _, T_plus, d = train_states.shape
    T = T_plus - 1
    params = _init_params(family, d, T, layers, seed)
    m0 = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
    v0 = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params)
    x_batch = jnp.asarray(train_states, dtype=jnp.complex128)
    losses = []
    moments = (m0, v0)
    for ep in range(1, epochs + 1):
        params, moments, loss = _adam_step(
            params, moments, x_batch, k, kernel_mode, family, lr, 0.9, 0.999, 1e-8, ep
        )
        losses.append(float(loss))
    aligned = [float(x + math.log(T)) for x in losses]
    return {"params": params, "loss_history": losses, "aligned_history": aligned, "final_loss": losses[-1]}


def _predict_last_fidelity(states: np.ndarray, params: dict[str, jnp.ndarray], family: str, mode: str, k: int) -> float:
    x = jnp.asarray(states[:, :-1, :], dtype=jnp.complex128)
    y_true = jnp.asarray(states[:, -1, :], dtype=jnp.complex128)
    d = x.shape[-1]
    if family == "kqsa":
        w = _complex_ansatz_matrix(params["wp"])[:d, :d]
        v = _complex_ansatz_matrix(params["vp"])[:d, :d]
    else:
        w = _unitary_from_raw(params["w_raw"])
        v = _unitary_from_raw(params["v_raw"])

    def one(seq, yt):
        s_row = jnp.conj(seq[-1:]) @ (w @ seq.T)
        kern, _ = _kernel_matrix(s_row, k, mode, d)
        vals = (seq @ v.T)
        z = jnp.sum(kern[0][:, None] * vals, axis=0)
        z = z / jnp.maximum(jnp.linalg.norm(z), 1e-12)
        yt = yt / jnp.maximum(jnp.linalg.norm(yt), 1e-12)
        return jnp.abs(jnp.vdot(yt, z)) ** 2

    fids = jax.vmap(one)(x, y_true)
    return float(jnp.mean(fids))


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    hist = np.array([r["aligned_history"] for r in runs], dtype=float)
    finals = np.array([r["aligned_loss"] for r in runs], dtype=float)
    tests = np.array([r["test_neglog_fid"] for r in runs], dtype=float)
    return {
        "aligned_history": hist.mean(axis=0).tolist(),
        "aligned_std": hist.std(axis=0).tolist(),
        "aligned_loss_mean": float(finals.mean()),
        "aligned_loss_std": float(finals.std()),
        "aligned_test_loss_mean": float(tests.mean()),
        "aligned_test_loss_std": float(tests.std()),
    }


def plot_vs_k(mu_points: list[dict[str, Any]], out_path: Path, title: str, ykey_mean: str, ykey_std: str) -> None:
    styles = {
        "k-QSA L=16": dict(color="#0072B2", linestyle="-", marker="o", linewidth=2.4),
        "k-CSA": dict(color="#E69F00", linestyle="-", marker="s", linewidth=2.4),
        "poly-k-QSA L=16": dict(color="#0072B2", linestyle="--", marker="^", linewidth=2.2),
        "poly-k-CSA": dict(color="#E69F00", linestyle="--", marker="v", linewidth=2.2),
    }
    fig, ax = plt.subplots(figsize=(10.5, 6))
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in mu_points:
        by_model.setdefault(row["model"], []).append(row)
    for name, rows in by_model.items():
        rows = sorted(rows, key=lambda x: int(x["k"]))
        ks = [int(r["k"]) for r in rows]
        means = [float(r[ykey_mean]) for r in rows]
        stds = [float(r[ykey_std]) for r in rows]
        st = styles.get(name, dict(color="0.3", linestyle="-", marker="o", linewidth=2.2))
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
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
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
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--poly-epochs", type=int, default=320)
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
        args.epochs = 20
        args.poly_epochs = 25
        args.n_seeds = 2
        args.layers = 2

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
    mu_points = []
    mu_points_test = []
    for display, family, mode, epochs in model_specs:
        for k in ks:
            runs = []
            for s_idx in range(args.n_seeds):
                mseed = args.model_seed_base + s_idx
                trained = train_model(
                    train_states=dataset.train_states,
                    family=family,
                    kernel_mode=mode,
                    k=k,
                    layers=args.layers,
                    epochs=epochs,
                    lr=args.lr,
                    seed=mseed + 17 * k,
                )
                test_fid = _predict_last_fidelity(
                    dataset.test_states, trained["params"], family=family, mode=mode, k=k
                )
                runs.append(
                    {
                        "seed": mseed,
                        "loss_history": trained["loss_history"],
                        "aligned_history": trained["aligned_history"],
                        "aligned_loss": float(trained["final_loss"] + math.log(args.T)),
                        "test_neglog_fid": float(-math.log(max(test_fid, 1e-12))),
                    }
                )
            agg = aggregate(runs)
            payload = {
                "model": display,
                "family": family,
                "kernel_mode": mode,
                "k": k,
                "runs": runs,
                **agg,
            }
            (aggs_dir / f"{family}_{mode}_k{k}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            mu_points.append(
                {
                    "model": display,
                    "k": k,
                    "aligned_loss_mean": agg["aligned_loss_mean"],
                    "aligned_loss_std": agg["aligned_loss_std"],
                }
            )
            mu_points_test.append(
                {
                    "model": display,
                    "k": k,
                    "aligned_test_loss_mean": agg["aligned_test_loss_mean"],
                    "aligned_test_loss_std": agg["aligned_test_loss_std"],
                }
            )
            print(
                f"[{display} k={k}] train aligned={agg['aligned_loss_mean']:.4f}±{agg['aligned_loss_std']:.4f} | "
                f"test -logF={agg['aligned_test_loss_mean']:.4f}±{agg['aligned_test_loss_std']:.4f}"
            )

    plot_vs_k(
        mu_points,
        plots / "final_aligned_loss_vs_k.png",
        f"FINAL aligned loss vs k (T={args.T}, d={args.d}; train; mono+poly)",
        "aligned_loss_mean",
        "aligned_loss_std",
    )
    plot_vs_k(
        mu_points_test,
        plots / "final_aligned_loss_vs_k_test.png",
        f"FINAL aligned loss vs k (T={args.T}, d={args.d}; held-out TFIM params; y=-log prediction fidelity)",
        "aligned_test_loss_mean",
        "aligned_test_loss_std",
    )
    summary = {
        "config": vars(args),
        "mu_points_vs_k": mu_points,
        "mu_points_vs_k_test": mu_points_test,
        "dataset": {
            "train_shape": list(dataset.train_states.shape),
            "test_shape": list(dataset.test_states.shape),
            "note": "Hamiltonian parameters (J,h) vary per trajectory and are never given as model inputs.",
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] outputs in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
