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
    "nl-CSA iso": dict(color="#000000", linestyle="-", marker="s", linewidth=2.2),
    "nl-CSA gen": dict(color="#000000", linestyle=":", marker="x", linewidth=2.4),
    "nl-CSA iso CE": dict(color="#555555", linestyle="-", marker="D", linewidth=2.2),
    "nl-CSA gen CE": dict(color="#555555", linestyle=":", marker="x", linewidth=2.4),
    "k-QSA L=16 (L_B)": dict(color="#0072B2", linestyle="-", marker="o", linewidth=2.4),
    "k-QSA L=16 (L_half)": dict(color="#0072B2", linestyle=":", marker="o", linewidth=2.0),
    "k-CSA (L_B)": dict(color="#E69F00", linestyle="-", marker="s", linewidth=2.4),
    "k-CSA (L_half)": dict(color="#E69F00", linestyle=":", marker="s", linewidth=2.0),
    "poly-k-QSA L=16 (L_B)": dict(color="#0072B2", linestyle="-", marker="^", linewidth=2.2),
    "poly-k-QSA L=16 (L_half)": dict(color="#0072B2", linestyle=":", marker="^", linewidth=2.0),
    "poly-k-CSA (L_B)": dict(color="#E69F00", linestyle="-", marker="v", linewidth=2.2),
    "poly-k-CSA (L_half)": dict(color="#E69F00", linestyle=":", marker="v", linewidth=2.0),
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
    if mode == "soft":
        # Causal fidelity-softmax; k-independent (nl-CSA).
        beta = jnp.sqrt(float(d))
        g = jnp.abs(s) ** 2
        t_q, t_k = s.shape
        if t_q == t_k:
            tri = jnp.tril(jnp.ones((t_q, t_k), dtype=jnp.float64))
            e = jnp.exp(beta * g) * tri
        else:
            # Single query row (e.g. T+1 fidelity): columns are already the causal prefix.
            e = jnp.exp(beta * g)
        z = jnp.sum(e, axis=1, keepdims=True)
        z = jnp.where(z < 1e-12, 1.0, z)
        return e / z, jnp.asarray(1.0, dtype=jnp.float64)
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
    elif family in ("nlcsa_gen", "nlcsa_gen_ce"):
        # Free (non-unitary) complex maps: quantum-sequence analogue of nl-CSA "general".
        w = params["w_raw"]
        v = params["v_raw"]
    else:
        # k-CSA and nl-CSA iso / iso CE: free unitary W,V (isometric value map).
        w = _unitary_from_raw(params["w_raw"])
        v = _unitary_from_raw(params["v_raw"])
    return w, v


def _prefix_aj_zj(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    v: jnp.ndarray,
    k: int,
    mode: str,
    d: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-step <y_j|z_j> and ||z_j||^2 from causal kernel attention."""
    t = x.shape[0]
    s = jnp.conj(x) @ (w @ x.T)
    a = jnp.conj(y) @ (v @ x.T)
    kern, _ = _kernel_matrix(s, k, mode, d)
    tri = jnp.tril(jnp.ones((t, t), dtype=jnp.float64))
    kern = kern * tri
    aj = jnp.sum(a * kern, axis=1)
    zj = kern @ (x @ v.T)
    wj2 = jnp.sum(jnp.abs(zj) ** 2, axis=1)
    return aj, wj2, kern


def _mu_zeta_nu_for_sequence(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    v: jnp.ndarray,
    phi: jnp.ndarray,
    k: int,
    mode: str,
    d: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    t = x.shape[0]
    aj, wj2, kern = _prefix_aj_zj(x, y, w, v, k, mode, d)
    _, lam = _kernel_matrix(jnp.conj(x) @ (w @ x.T), k, mode, d)
    ntri = t * (t + 1) / 2.0
    mu = jnp.abs(jnp.sum(jnp.exp(1j * phi[:t]) * aj)) ** 2 / (lam**2 * ntri**2)
    zeta = jnp.sum(wj2) / (lam**2 * t * ntri)
    nu = jnp.sum(jnp.abs(aj) ** 2) / (lam**2 * t * ntri)
    return jnp.real(mu), jnp.real(zeta), jnp.real(nu)


def _L_B_for_sequence(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    v: jnp.ndarray,
    phi: jnp.ndarray,
    k: int,
    mode: str,
    d: int,
) -> jnp.ndarray:
    t = x.shape[0]
    mu, zeta, _ = _mu_zeta_nu_for_sequence(x, y, w, v, phi, k, mode, d)
    pref = (t + 1) / (2.0 * t)
    f_val = pref * mu / jnp.maximum(zeta, 1e-30)
    return -jnp.log(jnp.maximum(f_val, 1e-30))


def _per_step_p_for_sequence(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    v: jnp.ndarray,
    k: int,
    mode: str,
    d: int,
) -> jnp.ndarray:
    aj, wj2, _ = _prefix_aj_zj(x, y, w, v, k, mode, d)
    w_norm = jnp.sqrt(jnp.maximum(wj2, 1e-30))
    return jnp.clip((jnp.abs(aj) / jnp.maximum(w_norm, 1e-30)) ** 2, 1e-30, 1.0)


def _L_half_uniform_for_sequence(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    v: jnp.ndarray,
    k: int,
    mode: str,
    d: int,
) -> jnp.ndarray:
    p = _per_step_p_for_sequence(x, y, w, v, k, mode, d)
    return -2.0 * jnp.log(jnp.maximum(jnp.mean(jnp.sqrt(p)), 1e-30))


def _L_1_for_sequence(
    x: jnp.ndarray,
    y: jnp.ndarray,
    w: jnp.ndarray,
    v: jnp.ndarray,
    k: int,
    mode: str,
    d: int,
) -> jnp.ndarray:
    """L_1 = CE_uniform = -(1/T) sum_j log p_j (Shannon cross-entropy)."""
    p = _per_step_p_for_sequence(x, y, w, v, k, mode, d)
    return -jnp.mean(jnp.log(p))


def _batch_L_B(
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
    losses = jax.vmap(lambda xs, ys: _L_B_for_sequence(xs, ys, w, v, phi, k, mode, d))(x_batch, y_batch)
    return jnp.mean(losses)


def _batch_L_half_uniform(
    params: dict[str, jnp.ndarray],
    states: jnp.ndarray,
    k: int,
    mode: str,
    family: str,
) -> jnp.ndarray:
    d = states.shape[-1]
    w, v = _wv_from_params(params, d, family)
    y_batch = states[:, 1:, :]
    x_batch = states[:, :-1, :]
    losses = jax.vmap(lambda xs, ys: _L_half_uniform_for_sequence(xs, ys, w, v, k, mode, d))(x_batch, y_batch)
    return jnp.mean(losses)


def _batch_L_1(
    params: dict[str, jnp.ndarray],
    states: jnp.ndarray,
    k: int,
    mode: str,
    family: str,
) -> jnp.ndarray:
    d = states.shape[-1]
    w, v = _wv_from_params(params, d, family)
    y_batch = states[:, 1:, :]
    x_batch = states[:, :-1, :]
    losses = jax.vmap(lambda xs, ys: _L_1_for_sequence(xs, ys, w, v, k, mode, d))(x_batch, y_batch)
    return jnp.mean(losses)


def _batch_train_loss(
    params: dict[str, jnp.ndarray],
    states: jnp.ndarray,
    k: int,
    mode: str,
    family: str,
) -> jnp.ndarray:
    # nl-CSA CE variants train with Shannon L_1; soft Renyi variants with L_half;
    # k-QSA / k-CSA always train with L_B.
    if family.endswith("_ce"):
        return _batch_L_1(params, states, k, mode, family)
    if family.startswith("nlcsa"):
        return _batch_L_half_uniform(params, states, k, mode, family)
    return _batch_L_B(params, states, k, mode, family)


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
    loss, grads = jax.value_and_grad(_batch_train_loss)(params, x_batch, k, mode, family)
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
    scale = 1.0 / math.sqrt(d) if family in ("nlcsa_gen", "nlcsa_gen_ce") else 1.0
    w_raw = scale * (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
    v_raw = scale * (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
    return {
        "w_raw": jnp.asarray(w_raw, dtype=jnp.complex128),
        "v_raw": jnp.asarray(v_raw, dtype=jnp.complex128),
        # Shared learned phases across sequences; reused as-is at test (never phi*).
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
            test_loss = float(_batch_train_loss(params, test_batch, k, kernel_mode, family))
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

    final_train = float(_batch_train_loss(best_params, train_batch, k, kernel_mode, family))
    final_test = float(_batch_train_loss(best_params, test_batch, k, kernel_mode, family))
    final_train_L_B = float(_batch_L_B(best_params, train_batch, k, kernel_mode, family))
    final_train_L_half = float(_batch_L_half_uniform(best_params, train_batch, k, kernel_mode, family))
    final_train_L_1 = float(_batch_L_1(best_params, train_batch, k, kernel_mode, family))
    final_test_L_B = float(_batch_L_B(best_params, test_batch, k, kernel_mode, family))
    final_test_L_half = float(_batch_L_half_uniform(best_params, test_batch, k, kernel_mode, family))
    final_test_L_1 = float(_batch_L_1(best_params, test_batch, k, kernel_mode, family))

    return {
        "params": best_params,
        "loss_history": losses,
        "test_loss_history": test_losses,
        "test_loss_epochs": test_epochs,
        "final_loss": final_train,
        "final_test_loss": final_test,
        "final_train_L_B": final_train_L_B,
        "final_train_L_half_uniform": final_train_L_half,
        "final_train_L_1": final_train_L_1,
        "final_test_L_B": final_test_L_B,
        "final_test_L_half_uniform": final_test_L_half,
        "final_test_L_1": final_test_L_1,
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


def aggregate(runs: list[dict[str, Any]], train_key: str, test_key: str) -> dict[str, Any]:
    train_vals = np.array([r[train_key] for r in runs], dtype=float)
    test_vals = np.array([r[test_key] for r in runs], dtype=float)
    epochs = np.array([r["epochs_run"] for r in runs], dtype=float)
    return {
        "train_loss_mean": float(train_vals.mean()),
        "train_loss_std": float(train_vals.std()),
        "test_loss_mean": float(test_vals.mean()),
        "test_loss_std": float(test_vals.std()),
        "epochs_run_mean": float(epochs.mean()),
        "epochs_run_std": float(epochs.std()),
        "converged_fraction": float(np.mean([1.0 if r["converged"] else 0.0 for r in runs])),
    }


def plot_vs_k(
    points: list[dict[str, Any]],
    out_path: Path,
    title: str,
    ykey_mean: str,
    ykey_std: str,
    ylabel: str,
    nl_refs: list[dict[str, Any]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in points:
        by_model.setdefault(row["model"], []).append(row)
    all_ks: list[int] = []
    for name, rows in by_model.items():
        rows = sorted(rows, key=lambda x: int(x["k"]))
        ks = [int(r["k"]) for r in rows]
        all_ks.extend(ks)
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
    if nl_refs and all_ks:
        xmin, xmax = min(all_ks), max(all_ks)
        xs = np.linspace(xmin, xmax, 64)
        for ref in nl_refs:
            name = str(ref["model"])
            mean = float(ref[ykey_mean])
            std = float(ref[ykey_std])
            st = PLOT_STYLES.get(name, dict(color="0.1", linestyle="-", linewidth=2.0))
            ax.axhline(
                mean,
                color=st["color"],
                linestyle=st["linestyle"],
                linewidth=st["linewidth"],
                label=f"{name} (k-indep.)",
            )
            ax.fill_between(xs, mean - std, mean + std, color=st["color"], alpha=0.10, linewidth=0)
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_loss_pair_comparison(
    points: list[dict[str, Any]],
    out_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Plot L_B and L_half_uniform (train, final params) for mu-models."""
    fig, ax = plt.subplots(figsize=(10.5, 6))
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in points:
        by_model.setdefault(row["model"], []).append(row)
    for name, rows in by_model.items():
        rows = sorted(rows, key=lambda x: int(x["k"]))
        ks = [int(r["k"]) for r in rows]
        lb = [float(r["L_B_mean"]) for r in rows]
        lb_std = [float(r["L_B_std"]) for r in rows]
        lh = [float(r["L_half_mean"]) for r in rows]
        lh_std = [float(r["L_half_std"]) for r in rows]
        st_b = PLOT_STYLES.get(f"{name} (L_B)", PLOT_STYLES.get(name, dict(color="0.3", linestyle="-", marker="o")))
        st_h = PLOT_STYLES.get(f"{name} (L_half)", dict(color=st_b["color"], linestyle=":", marker=st_b.get("marker", "o")))
        ax.errorbar(ks, lb, yerr=lb_std, color=st_b["color"], linestyle=st_b["linestyle"], marker=st_b["marker"],
                    linewidth=2.2, capsize=4, label=f"{name} ($L_B$)")
        ax.errorbar(ks, lh, yerr=lh_std, color=st_h["color"], linestyle=st_h["linestyle"], marker=st_h["marker"],
                    linewidth=2.0, capsize=4, label=f"{name} ($L_{{1/2}}$)")
    ax.set_xlabel("k")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_training_curves_LB(
    curves: list[dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    for row in curves:
        name = str(row["model"])
        hist = np.asarray(row["history_mean"], dtype=float)
        std = np.asarray(row.get("history_std", np.zeros_like(hist)), dtype=float)
        xs = np.arange(1, len(hist) + 1)
        st = PLOT_STYLES.get(name, dict(color="0.3", linestyle="-", linewidth=2.0))
        ax.plot(xs, hist, color=st["color"], linestyle=st["linestyle"], linewidth=2.2, label=name)
        if len(std) == len(hist):
            ax.fill_between(xs, hist - std, hist + std, color=st["color"], alpha=0.12, linewidth=0)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$L_B$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _run_record(
    trained: dict[str, Any],
    family: str,
    mseed: int,
) -> dict[str, Any]:
    if family.endswith("_ce"):
        train_metric = "L_1"
    elif family.startswith("nlcsa"):
        train_metric = "L_half_uniform"
    else:
        train_metric = "L_B"
    return {
        "seed": mseed,
        "loss_history": trained["loss_history"],
        "test_loss_history": trained["test_loss_history"],
        "test_loss_epochs": trained["test_loss_epochs"],
        "train_loss": float(trained["final_loss"]),
        "test_loss": float(trained["final_test_loss"]),
        "train_L_B": float(trained["final_train_L_B"]),
        "train_L_half_uniform": float(trained["final_train_L_half_uniform"]),
        "train_L_1": float(trained["final_train_L_1"]),
        "test_L_B": float(trained["final_test_L_B"]),
        "test_L_half_uniform": float(trained["final_test_L_half_uniform"]),
        "test_L_1": float(trained["final_test_L_1"]),
        "epochs_run": trained["epochs_run"],
        "converged": trained["converged"],
        "stop_reason": trained["stop_reason"],
        "train_metric": train_metric,
        "test_metric": train_metric,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantum-sequences k sweep (L_B / L_half_uniform)")
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--test-size", type=int, default=128)
    parser.add_argument("--T", type=int, default=32)
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--ks", type=str, default="1,2,3,5,6")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--poly-epochs", type=int, default=600)
    parser.add_argument("--nl-epochs", type=int, default=400)
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
    parser.add_argument("--skip-nl", action="store_true", help="Skip nl-CSA iso/gen horizontal refs.")
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
        args.nl_epochs = 40
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
    train_points: list[dict[str, Any]] = []
    test_points: list[dict[str, Any]] = []
    mono_compare_points: list[dict[str, Any]] = []
    poly_compare_points: list[dict[str, Any]] = []
    lhalf_train_points: list[dict[str, Any]] = []
    lhalf_test_points: list[dict[str, Any]] = []
    l1_train_points: list[dict[str, Any]] = []
    l1_test_points: list[dict[str, Any]] = []
    nl_refs_train: list[dict[str, Any]] = []
    nl_refs_test: list[dict[str, Any]] = []
    nl_refs_lhalf_train: list[dict[str, Any]] = []
    nl_refs_lhalf_test: list[dict[str, Any]] = []
    nl_refs_l1_train: list[dict[str, Any]] = []
    nl_refs_l1_test: list[dict[str, Any]] = []
    lb_curves_k3: list[dict[str, Any]] = []
    curves_k = 3 if 3 in ks else ks[len(ks) // 2]

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
                runs.append(_run_record(trained, family, mseed))

            agg_train = aggregate(runs, "train_loss", "test_loss")
            agg_lb = {
                "L_B_mean": float(np.mean([r["train_L_B"] for r in runs])),
                "L_B_std": float(np.std([r["train_L_B"] for r in runs])),
                "L_half_mean": float(np.mean([r["train_L_half_uniform"] for r in runs])),
                "L_half_std": float(np.std([r["train_L_half_uniform"] for r in runs])),
                "L_1_mean": float(np.mean([r["train_L_1"] for r in runs])),
                "L_1_std": float(np.std([r["train_L_1"] for r in runs])),
                "L_half_test_mean": float(np.mean([r["test_L_half_uniform"] for r in runs])),
                "L_half_test_std": float(np.std([r["test_L_half_uniform"] for r in runs])),
                "L_1_test_mean": float(np.mean([r["test_L_1"] for r in runs])),
                "L_1_test_std": float(np.std([r["test_L_1"] for r in runs])),
            }
            payload = {"model": display, "family": family, "kernel_mode": mode, "k": k, "runs": runs, **agg_train, **agg_lb}
            (aggs_dir / f"{family}_{mode}_k{k}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

            train_points.append(
                {"model": display, "k": k, "loss_mean": agg_train["train_loss_mean"], "loss_std": agg_train["train_loss_std"]}
            )
            test_points.append(
                {"model": display, "k": k, "loss_mean": agg_train["test_loss_mean"], "loss_std": agg_train["test_loss_std"]}
            )
            lhalf_train_points.append(
                {"model": display, "k": k, "loss_mean": agg_lb["L_half_mean"], "loss_std": agg_lb["L_half_std"]}
            )
            lhalf_test_points.append(
                {"model": display, "k": k, "loss_mean": agg_lb["L_half_test_mean"], "loss_std": agg_lb["L_half_test_std"]}
            )
            l1_train_points.append(
                {"model": display, "k": k, "loss_mean": agg_lb["L_1_mean"], "loss_std": agg_lb["L_1_std"]}
            )
            l1_test_points.append(
                {"model": display, "k": k, "loss_mean": agg_lb["L_1_test_mean"], "loss_std": agg_lb["L_1_test_std"]}
            )
            compare_row = {"model": display, "k": k, **agg_lb}
            if mode == "monomial":
                mono_compare_points.append(compare_row)
            else:
                poly_compare_points.append(compare_row)

            if k == curves_k:
                # Pad histories to common length for mean±std curves.
                max_len = max(len(r["loss_history"]) for r in runs)
                padded = np.full((len(runs), max_len), np.nan, dtype=float)
                for i, r in enumerate(runs):
                    h = np.asarray(r["loss_history"], dtype=float)
                    padded[i, : len(h)] = h
                    if len(h) < max_len:
                        padded[i, len(h) :] = h[-1]
                lb_curves_k3.append(
                    {
                        "model": display,
                        "history_mean": np.nanmean(padded, axis=0).tolist(),
                        "history_std": np.nanstd(padded, axis=0).tolist(),
                    }
                )

            print(
                f"[{display} k={k}] train_L_B={agg_lb['L_B_mean']:.4f}+/-{agg_lb['L_B_std']:.4f} | "
                f"L_half={agg_lb['L_half_mean']:.4f}+/-{agg_lb['L_half_std']:.4f} | "
                f"L_1={agg_lb['L_1_mean']:.4f}+/-{agg_lb['L_1_std']:.4f} | "
                f"epochs~{agg_train['epochs_run_mean']:.0f} conv={agg_train['converged_fraction']:.0%}"
            )

    if not args.skip_nl:
        nl_specs = [
            ("nl-CSA iso", "nlcsa_iso", "soft", args.nl_epochs, "half"),
            ("nl-CSA gen", "nlcsa_gen", "soft", args.nl_epochs, "half"),
            ("nl-CSA iso CE", "nlcsa_iso_ce", "soft", args.nl_epochs, "ce"),
            ("nl-CSA gen CE", "nlcsa_gen_ce", "soft", args.nl_epochs, "ce"),
        ]
        bookkeeping_k = ks[0]
        for display, family, mode, max_epochs, nl_kind in nl_specs:
            runs = []
            for s_idx in range(args.n_seeds):
                mseed = args.model_seed_base + s_idx
                trained = train_model(
                    train_states=dataset.train_states,
                    test_states=dataset.test_states,
                    family=family,
                    kernel_mode=mode,
                    k=bookkeeping_k,
                    layers=args.layers,
                    max_epochs=max_epochs,
                    lr=args.lr,
                    seed=mseed + (911 if nl_kind == "half" else 1911),
                    min_epochs=args.min_epochs,
                    patience=args.patience,
                    loss_rel_tol=args.loss_rel_tol,
                    eval_every=args.eval_every,
                )
                runs.append(_run_record(trained, family, mseed))
            agg_train = aggregate(runs, "train_loss", "test_loss")
            payload = {
                "model": display,
                "family": family,
                "kernel_mode": mode,
                "k": "independent",
                "nl_kind": nl_kind,
                "runs": runs,
                **agg_train,
                "L_half_mean": float(np.mean([r["train_L_half_uniform"] for r in runs])),
                "L_half_std": float(np.std([r["train_L_half_uniform"] for r in runs])),
                "L_1_mean": float(np.mean([r["train_L_1"] for r in runs])),
                "L_1_std": float(np.std([r["train_L_1"] for r in runs])),
                "L_half_test_mean": float(np.mean([r["test_L_half_uniform"] for r in runs])),
                "L_half_test_std": float(np.std([r["test_L_half_uniform"] for r in runs])),
                "L_1_test_mean": float(np.mean([r["test_L_1"] for r in runs])),
                "L_1_test_std": float(np.std([r["test_L_1"] for r in runs])),
            }
            (aggs_dir / f"{family}_soft.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            ref = {"model": display, "loss_mean": agg_train["train_loss_mean"], "loss_std": agg_train["train_loss_std"]}
            ref_t = {"model": display, "loss_mean": agg_train["test_loss_mean"], "loss_std": agg_train["test_loss_std"]}
            if nl_kind == "half":
                nl_refs_train.append(ref)
                nl_refs_test.append(ref_t)
                nl_refs_lhalf_train.append(
                    {"model": display, "loss_mean": payload["L_half_mean"], "loss_std": payload["L_half_std"]}
                )
                nl_refs_lhalf_test.append(
                    {"model": display, "loss_mean": payload["L_half_test_mean"], "loss_std": payload["L_half_test_std"]}
                )
            else:
                nl_refs_l1_train.append(
                    {"model": display, "loss_mean": payload["L_1_mean"], "loss_std": payload["L_1_std"]}
                )
                nl_refs_l1_test.append(
                    {"model": display, "loss_mean": payload["L_1_test_mean"], "loss_std": payload["L_1_test_std"]}
                )
            print(
                f"[{display} k-indep] train={agg_train['train_loss_mean']:.4f}+/-{agg_train['train_loss_std']:.4f} | "
                f"test={agg_train['test_loss_mean']:.4f}+/-{agg_train['test_loss_std']:.4f} | "
                f"epochs~{agg_train['epochs_run_mean']:.0f} conv={agg_train['converged_fraction']:.0%}"
            )

    plot_vs_k(
        train_points,
        plots / "train_loss_L_B_vs_k.png",
        f"Train loss vs k (T={args.T}, d={args.d}; kQSA/kCSA: $L_B$, nl-CSA: $L_{{1/2}}$)",
        "loss_mean",
        "loss_std",
        r"train loss",
        nl_refs=nl_refs_train,
    )
    plot_vs_k(
        test_points,
        plots / "test_loss_L_B_vs_k.png",
        f"Test loss vs k (T={args.T}, d={args.d}; held-out TFIM; kQSA/kCSA: $L_B$, nl-CSA: $L_{{1/2}}$)",
        "loss_mean",
        "loss_std",
        r"test loss",
        nl_refs=nl_refs_test,
    )
    plot_loss_pair_comparison(
        mono_compare_points,
        plots / "train_mono_L_B_vs_L_half_uniform.png",
        f"Mono kQSA/kCSA: $L_B$ vs $L_{{1/2}}$ on train (final params, T={args.T}, d={args.d})",
        r"train loss (final checkpoint)",
    )
    plot_loss_pair_comparison(
        poly_compare_points,
        plots / "train_poly_L_B_vs_L_half_uniform.png",
        f"Poly kQSA/kCSA: $L_B$ vs $L_{{1/2}}$ on train (final params, T={args.T}, d={args.d})",
        r"train loss (final checkpoint)",
    )
    plot_vs_k(
        lhalf_train_points,
        plots / "train_Lhalf_vs_k.png",
        f"Train $L_{{1/2}}$ vs k (T={args.T}, d={args.d}; all models, final params)",
        "loss_mean",
        "loss_std",
        r"$L_{1/2}$ (train)",
        nl_refs=nl_refs_lhalf_train,
    )
    plot_vs_k(
        lhalf_test_points,
        plots / "test_Lhalf_vs_k.png",
        f"Test $L_{{1/2}}$ vs k (T={args.T}, d={args.d}; held-out TFIM)",
        "loss_mean",
        "loss_std",
        r"$L_{1/2}$ (test)",
        nl_refs=nl_refs_lhalf_test,
    )
    plot_vs_k(
        l1_train_points,
        plots / "train_L1_vs_k.png",
        f"Train $L_1$ (Shannon CE) vs k (T={args.T}, d={args.d}; all models, final params)",
        "loss_mean",
        "loss_std",
        r"$L_1$ Shannon CE (train)",
        nl_refs=nl_refs_l1_train,
    )
    plot_vs_k(
        l1_test_points,
        plots / "test_L1_vs_k.png",
        f"Test $L_1$ (Shannon CE) vs k (T={args.T}, d={args.d}; held-out TFIM)",
        "loss_mean",
        "loss_std",
        r"$L_1$ Shannon CE (test)",
        nl_refs=nl_refs_l1_test,
    )
    if lb_curves_k3:
        plot_training_curves_LB(
            lb_curves_k3,
            plots / f"train_LB_curves_k{curves_k}.png",
            f"Training curves $L_B$ at k={curves_k} (T={args.T}, d={args.d}; mean±std over seeds)",
        )

    summary = {
        "config": vars(args),
        "loss_protocol": {
            "kQSA_kCSA_train": "L_B = -log F, F = (T+1)/(2T) * mu/zeta (complex ansatz, trainable phi)",
            "nlCSA_renyi_train": "L_half_uniform (soft kernel, k-independent)",
            "nlCSA_ce_train": "L_1 = Shannon CE_uniform (soft kernel, k-independent)",
            "eval_all": "L_half and L_1 evaluated on train/test for all models at final params",
        },
        "train_points": train_points,
        "test_points": test_points,
        "lhalf_train_points": lhalf_train_points,
        "lhalf_test_points": lhalf_test_points,
        "l1_train_points": l1_train_points,
        "l1_test_points": l1_test_points,
        "mono_compare_points": mono_compare_points,
        "poly_compare_points": poly_compare_points,
        "nl_refs_train": nl_refs_train,
        "nl_refs_test": nl_refs_test,
        "nl_refs_lhalf_train": nl_refs_lhalf_train,
        "nl_refs_lhalf_test": nl_refs_lhalf_test,
        "nl_refs_l1_train": nl_refs_l1_train,
        "nl_refs_l1_test": nl_refs_l1_test,
        "lb_curves_k": curves_k,
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
