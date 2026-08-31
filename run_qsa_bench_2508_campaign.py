#!/usr/bin/env python3
"""HPC campaign on qsa_bench_25_08 (hybrid readout + trainable embedding).

Classical (PTB): jointly train real embedding + W,V; norms enter loss via alpha_t.
Quantum (TFIM): unit-norm states, alpha=1 (no embedding).

Train L_B (k-models) / L1 (nl-CSA). Report capacity-normalized excess:
  L1_excess, LB_excess  on train/test vs k.

Param-match: --layers 0 → QSA layers nearest to CSA 4 d^2.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

import qsa_bench_25_08 as qb
from encoding import Encoding

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
    """TFIM trajectories. Returns unit-norm X,Y with shape (N,T,d); alpha ≡ 1."""
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


def _load_ptb_sentences(word_len: int, max_sentences: int, data_seed: int) -> list[str]:
    """Exact-length PTB lines (word_len words each)."""
    rng = random.Random(data_seed)
    path = Path("ptb_sentences.txt")
    with path.open("r", encoding="utf-8") as f:
        valid = [line.strip() for line in f if line.strip() and len(line.split()) == word_len]
    if not valid:
        raise RuntimeError(f"No PTB sentences with length={word_len} in {path}")
    if len(valid) <= max_sentences:
        sentences = list(valid)
    else:
        sentences = rng.sample(valid, max_sentences)
    if len(sentences) < max_sentences:
        expanded = list(sentences)
        while len(expanded) < max_sentences:
            expanded.extend(sentences)
        sentences = expanded[:max_sentences]
    return sentences


def generate_ptb_indices(
    train_size: int,
    test_size: int,
    d: int,
    T: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int, Encoding]:
    """PTB token ids for hybrid next-token: sentence length T+1 → T prediction steps.

    Returns idx_train, idx_test shaped (N, T+1), vocab size D, and Encoding (vocab only).
    """
    word_len = T + 1
    n_total = train_size + test_size
    sentences = _load_ptb_sentences(word_len, n_total, seed)
    enc = Encoding(sentences, embeddingDim=d, embeddingSeed=seed)
    ids = np.stack([np.asarray(enc.encode_tokens(s), dtype=np.int32) for s in sentences])
    assert ids.shape[1] == word_len, ids.shape
    return ids[:train_size], ids[train_size:], int(enc.vocabSize), enc


# --------------------------------------------------------------------------- #
#  Adam training                                                               #
# --------------------------------------------------------------------------- #
def make_model(name: str, kernel: str, d: int, k: int, layers: int, loss: str,
               v_isometric: bool = True) -> qb.Model:
    return qb.Model(name, d, kernel, k, layers=layers, loss=loss, v_isometric=v_isometric)


def _mean_metric_states(
    model: qb.Model, v: jnp.ndarray, Xs: jnp.ndarray, Ys: jnp.ndarray,
    key: str, alpha_t: jnp.ndarray | None,
) -> jnp.ndarray:
    W, V = model.build(v, xp=jnp)
    T = Xs.shape[1]
    mask = jnp.tril(jnp.ones((T, T)))

    def one(X, Y, at):
        A, w, f, lam = qb.forward(
            X, Y, W, V, model.kernel, model.k, model.beta, mask, xp=jnp, alpha_t=at,
        )
        return qb.metrics(A, w, f, lam, T, None, xp=jnp, alpha_t=at)[key]

    if alpha_t is None:
        at = jnp.ones((Xs.shape[0], T), dtype=jnp.float64)
    else:
        at = alpha_t
    return jnp.mean(jax.vmap(one)(Xs, Ys, at))


def _mean_metric_tokens(
    model: qb.Model, circuit_v: jnp.ndarray, Xvoc: jnp.ndarray,
    idx: jnp.ndarray, key: str,
) -> jnp.ndarray:
    Xi, Yi, at = qb.sequences_from_vocab(Xvoc, idx, xp=jnp)
    return _mean_metric_states(model, circuit_v, Xi, Yi, key, at)


def _full_metrics_states(
    model: qb.Model, v: np.ndarray, Xs: np.ndarray, Ys: np.ndarray,
    alpha_t: np.ndarray | None,
) -> dict[str, float]:
    W, V = model.build(v)
    m = qb.batch_metrics(Xs, Ys, W, V, model.kernel, model.k, model.beta, alpha_t=alpha_t)
    return {kk: float(vv) for kk, vv in m.items()}


def _full_metrics_tokens(
    model: qb.Model, circuit_v: np.ndarray, Xvoc: np.ndarray, idx: np.ndarray,
) -> dict[str, float]:
    Xi, Yi, at = qb.sequences_from_vocab(Xvoc, idx, xp=np)
    return _full_metrics_states(model, circuit_v, Xi, Yi, at)


def train_adam_states(
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
    alpha_t: np.ndarray | None = None,
    alpha_te: np.ndarray | None = None,
) -> dict[str, Any]:
    """Adam on prebuilt (normalized) sequences — quantum path (α=1)."""
    if not qb.HAS_JAX:
        raise RuntimeError("JAX required")

    rng = np.random.default_rng(seed)
    v = jnp.asarray(model.init(rng), dtype=jnp.float64)
    m = jnp.zeros_like(v)
    vv = jnp.zeros_like(v)
    n_seq = Xs.shape[0]
    bs = min(batch_size, n_seq)
    Xs_j, Ys_j = jnp.asarray(Xs), jnp.asarray(Ys)
    if alpha_t is None:
        at_j = jnp.ones((n_seq, Xs.shape[1]), dtype=jnp.float64)
    else:
        at_j = jnp.asarray(alpha_t)
    loss_key = model.loss

    @jax.jit
    def step(v, m, vv, Xb, Yb, atb, t):
        loss, g = jax.value_and_grad(
            lambda p: _mean_metric_states(model, p, Xb, Yb, loss_key, atb)
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
        return float(_mean_metric_states(model, params, Xs_j, Ys_j, loss_key, at_j))

    for ep in range(1, max_epochs + 1):
        ii = rng.choice(n_seq, size=bs, replace=False)
        v, m, vv, loss = step(v, m, vv, Xs_j[ii], Ys_j[ii], at_j[ii], ep)
        loss_f = float(loss)
        hist.append(loss_f)
        do_eval = (ep % max(eval_every, 1) == 0) or (ep == 1) or (ep == max_epochs)
        if do_eval:
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

    train_m = _full_metrics_states(model, np.asarray(best_v), Xs, Ys, alpha_t)
    test_m = _full_metrics_states(model, np.asarray(best_v), Xte, Yte, alpha_te)
    qb.check_invariants(
        train_m, Xs.shape[1], label=f"({model.name} seed={seed})",
        verbose=verbose, alpha_mean=float(train_m.get("alpha_ar", 1.0)),
    )
    if verbose:
        print(
            f"    [{model.name:>10}] {loss_key} {hist[0]:8.4f} -> {train_m[loss_key]:8.4f}  "
            f"L1x={train_m['L1_excess']:.4f}/{test_m['L1_excess']:.4f}  "
            f"LBx={train_m['LB_excess']:.4f}/{test_m['LB_excess']:.4f}  "
            f"({model.n_params()} params, ep={best_epoch}/{len(hist)}, "
            f"{time.time() - t0:.1f}s, {stop_reason})",
            flush=True,
        )
    return _run_result(seed, model, hist, best_epoch, best_loss, stop_reason, train_m, test_m)


def train_adam_tokens(
    model: qb.Model,
    idx_tr: np.ndarray,
    idx_te: np.ndarray,
    D: int,
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
    emb_lr_scale: float = 1.0,
) -> dict[str, Any]:
    """Adam on concat(Xvoc, circuit_v) — classical PTB path."""
    if not qb.HAS_JAX:
        raise RuntimeError("JAX required")

    d = model.d
    rng = np.random.default_rng(seed)
    X0 = qb.init_embedding(D, d, seed=seed)
    c0 = model.init(rng)
    v = jnp.asarray(qb.pack_params(X0, c0), dtype=jnp.float64)
    n_emb = D * d
    n_c = model.n_par
    m = jnp.zeros_like(v)
    vv = jnp.zeros_like(v)
    n_seq = idx_tr.shape[0]
    bs = min(batch_size, n_seq)
    idx_j = jnp.asarray(idx_tr)
    idx_te_j = jnp.asarray(idx_te)
    loss_key = model.loss

    # Separate effective LR for embedding vs circuit via gradient scaling mask.
    lr_mask = jnp.concatenate([
        jnp.full((n_emb,), float(emb_lr_scale), dtype=jnp.float64),
        jnp.ones((n_c,), dtype=jnp.float64),
    ])

    @jax.jit
    def step(v, m, vv, idb, t):
        def loss_fn(p):
            Xvoc = p[:n_emb].reshape(D, d)
            circ = p[n_emb:]
            return _mean_metric_tokens(model, circ, Xvoc, idb, loss_key)

        loss, g = jax.value_and_grad(loss_fn)(v)
        g = g * lr_mask
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
        Xvoc = params[:n_emb].reshape(D, d)
        circ = params[n_emb:]
        return float(_mean_metric_tokens(model, circ, Xvoc, idx_j, loss_key))

    for ep in range(1, max_epochs + 1):
        ii = rng.choice(n_seq, size=bs, replace=False)
        v, m, vv, loss = step(v, m, vv, idx_j[ii], ep)
        loss_f = float(loss)
        hist.append(loss_f)
        do_eval = (ep % max(eval_every, 1) == 0) or (ep == 1) or (ep == max_epochs)
        if do_eval:
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

    Xvoc_b, circ_b = qb.unpack_params(np.asarray(best_v), D, d, n_c)
    train_m = _full_metrics_tokens(model, circ_b, Xvoc_b, idx_tr)
    test_m = _full_metrics_tokens(model, circ_b, Xvoc_b, idx_te)
    Xvoc_f, circ_f = qb.unpack_params(np.asarray(v), D, d, n_c)
    final_train_m = _full_metrics_tokens(model, circ_f, Xvoc_f, idx_tr)
    final_test_m = _full_metrics_tokens(model, circ_f, Xvoc_f, idx_te)
    qb.check_invariants(
        train_m, idx_tr.shape[1] - 1, label=f"({model.name} seed={seed})",
        verbose=verbose, alpha_mean=float(train_m.get("alpha_ar", 1.0)),
    )
    if verbose:
        print(
            f"    [{model.name:>10}] {loss_key} {hist[0]:8.4f} -> {train_m[loss_key]:8.4f}  "
            f"L1x={train_m['L1_excess']:.4f}/{test_m['L1_excess']:.4f}  "
            f"LBx={train_m['LB_excess']:.4f}/{test_m['LB_excess']:.4f}  "
            f"(circuit={model.n_params()} + emb={n_emb}, ep={best_epoch}/{len(hist)}, "
            f"{time.time() - t0:.1f}s, {stop_reason})",
            flush=True,
        )
    out = _run_result(seed, model, hist, best_epoch, best_loss, stop_reason,
                      train_m, test_m, final_train_m, final_test_m)
    out["n_emb"] = int(n_emb)
    out["alpha_ar_train"] = float(train_m.get("alpha_ar", 1.0))
    return out


def _run_result(seed, model, hist, best_epoch, best_loss, stop_reason,
                train_m, test_m, final_train_m=None, final_test_m=None):
    if final_train_m is None:
        final_train_m = train_m
    if final_test_m is None:
        final_test_m = test_m
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
        "train_L1_excess": train_m["L1_excess"],
        "test_L1_excess": test_m["L1_excess"],
        "train_LB_excess": train_m["LB_excess"],
        "test_LB_excess": test_m["LB_excess"],
        "train_mu": train_m["mu"],
        "test_mu": test_m["mu"],
        "train_mu_final": final_train_m["mu"],
        "test_mu_final": final_test_m["mu"],
        "final_epoch": int(len(hist)),
        "train_rho_mean": train_m.get("rho_mean", float("nan")),
        "test_rho_mean": test_m.get("rho_mean", float("nan")),
    }


# --------------------------------------------------------------------------- #
#  aggregates / plots                                                          #
# --------------------------------------------------------------------------- #
def _agg_stats(runs: list[dict[str, Any]], key: str) -> tuple[float, float]:
    vals = np.asarray([r[key] for r in runs], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=0))


METRIC_KEYS = (
    "train_L1", "test_L1", "train_LB", "test_LB",
    "train_L1_excess", "test_L1_excess", "train_LB_excess", "test_LB_excess",
    "train_mu", "test_mu",
)


def aggregate_runs(runs: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    out = dict(meta)
    out["n_seeds"] = len(runs)
    for key in METRIC_KEYS:
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
#  one world                                                                   #
# --------------------------------------------------------------------------- #
def run_world_classical(
    idx_tr: np.ndarray,
    idx_te: np.ndarray,
    d: int,
    vocab_D: int,
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
    aggs_dir = out_dir / "aggregates"
    plots_dir = out_dir / "plots"
    aggs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88, flush=True)
    print(
        f"CLASSICAL PTB (trainable emb)  T={idx_tr.shape[1]-1}  d={d}  D={vocab_D}  "
        f"n_train={idx_tr.shape[0]}  n_test={idx_te.shape[0]}  ks={ks}  "
        f"n_seeds={n_seeds}  layers={layers}",
        flush=True,
    )
    print("=" * 88, flush=True)

    k_points: list[dict[str, Any]] = []
    nl_refs: list[dict[str, Any]] = []

    train_kw = dict(
        lr=lr, batch_size=batch_size, min_epochs=min_epochs, patience=patience,
        loss_rel_tol=loss_rel_tol, eval_every=eval_every, verbose=True,
    )

    for name, kernel, loss in K_MODELS:
        for k in ks:
            model0 = make_model(name, kernel, d, k, layers, loss)
            agg_path = aggs_dir / f"{name.replace('-', '_')}_k{k}.json"
            cached = _load_complete(agg_path, n_seeds, model0.n_params())
            if cached is not None:
                print(f"  resume {name} k={k} ({cached['n_seeds']} seeds)", flush=True)
                agg = cached
            else:
                runs = []
                max_ep = poly_epochs if "poly" in name else epochs
                for s in range(n_seeds):
                    seed = model_seed_base + s
                    print(f"  >> {name} k={k} seed={seed}", flush=True)
                    model = make_model(name, kernel, d, k, layers, loss)
                    runs.append(train_adam_tokens(
                        model, idx_tr, idx_te, vocab_D,
                        max_epochs=max_ep, seed=seed, **train_kw,
                    ))
                agg = aggregate_runs(runs, {
                    "world": "classical", "model": name, "display": DISPLAY[name],
                    "kernel": kernel, "k": k, "loss": loss, "layers": layers,
                    "vocab_D": vocab_D,
                })
                agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
            row = {**agg, "display": DISPLAY[name], "k": k}
            k_points.append(row)

    for name, kernel, loss, v_iso in NL_MODELS:
        model0 = make_model(name, kernel, d, 1, layers, loss, v_isometric=v_iso)
        agg_path = aggs_dir / f"{name.replace('-', '_')}.json"
        cached = _load_complete(agg_path, n_seeds, model0.n_params())
        if cached is not None:
            print(f"  resume {name} ({cached['n_seeds']} seeds)", flush=True)
            agg = cached
        else:
            runs = []
            for s in range(n_seeds):
                seed = model_seed_base + 1000 + s
                print(f"  >> {name} seed={seed}", flush=True)
                model = make_model(name, kernel, d, 1, layers, loss, v_isometric=v_iso)
                runs.append(train_adam_tokens(
                    model, idx_tr, idx_te, vocab_D,
                    max_epochs=nl_epochs, seed=seed, **train_kw,
                ))
            agg = aggregate_runs(runs, {
                "world": "classical", "model": name, "display": DISPLAY[name],
                "kernel": kernel, "k": None, "loss": loss, "layers": layers,
                "vocab_D": vocab_D,
            })
            agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        nl_refs.append({**agg, "display": DISPLAY[name]})

    return _finish_plots(out_dir, plots_dir, "classical PTB", k_points, nl_refs, d, layers)


def run_world_quantum(
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
    print(
        f"QUANTUM TFIM (alpha=1)  T={Xs.shape[1]}  d={d}  "
        f"n_train={Xs.shape[0]}  n_test={Xte.shape[0]}  ks={ks}  "
        f"n_seeds={n_seeds}  layers={layers}",
        flush=True,
    )
    print("=" * 88, flush=True)

    k_points: list[dict[str, Any]] = []
    nl_refs: list[dict[str, Any]] = []
    train_kw = dict(
        lr=lr, batch_size=batch_size, min_epochs=min_epochs, patience=patience,
        loss_rel_tol=loss_rel_tol, eval_every=eval_every, verbose=True,
    )

    for name, kernel, loss in K_MODELS:
        for k in ks:
            model0 = make_model(name, kernel, d, k, layers, loss)
            agg_path = aggs_dir / f"{name.replace('-', '_')}_k{k}.json"
            cached = _load_complete(agg_path, n_seeds, model0.n_params())
            if cached is not None:
                print(f"  resume {name} k={k} ({cached['n_seeds']} seeds)", flush=True)
                agg = cached
            else:
                runs = []
                max_ep = poly_epochs if "poly" in name else epochs
                for s in range(n_seeds):
                    seed = model_seed_base + s
                    print(f"  >> {name} k={k} seed={seed}", flush=True)
                    model = make_model(name, kernel, d, k, layers, loss)
                    runs.append(train_adam_states(
                        model, Xs, Ys, Xte, Yte,
                        max_epochs=max_ep, seed=seed, **train_kw,
                    ))
                agg = aggregate_runs(runs, {
                    "world": "quantum", "model": name, "display": DISPLAY[name],
                    "kernel": kernel, "k": k, "loss": loss, "layers": layers,
                })
                agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
            k_points.append({**agg, "display": DISPLAY[name], "k": k})

    for name, kernel, loss, v_iso in NL_MODELS:
        model0 = make_model(name, kernel, d, 1, layers, loss, v_isometric=v_iso)
        agg_path = aggs_dir / f"{name.replace('-', '_')}.json"
        cached = _load_complete(agg_path, n_seeds, model0.n_params())
        if cached is not None:
            print(f"  resume {name} ({cached['n_seeds']} seeds)", flush=True)
            agg = cached
        else:
            runs = []
            for s in range(n_seeds):
                seed = model_seed_base + 1000 + s
                print(f"  >> {name} seed={seed}", flush=True)
                model = make_model(name, kernel, d, 1, layers, loss, v_isometric=v_iso)
                runs.append(train_adam_states(
                    model, Xs, Ys, Xte, Yte,
                    max_epochs=nl_epochs, seed=seed, **train_kw,
                ))
            agg = aggregate_runs(runs, {
                "world": "quantum", "model": name, "display": DISPLAY[name],
                "kernel": kernel, "k": None, "loss": loss, "layers": layers,
            })
            agg_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
        nl_refs.append({**agg, "display": DISPLAY[name]})

    return _finish_plots(out_dir, plots_dir, "quantum TFIM", k_points, nl_refs, d, layers)


def _finish_plots(out_dir, plots_dir, title_world, k_points, nl_refs, d, layers):
    plot_vs_k(
        k_points, plots_dir / "train_L1_excess_vs_k.png",
        f"{title_world}: train $L_1^{{\\mathrm{{excess}}}}$ vs k (d={d}, L={layers})",
        "train_L1_excess_mean", "train_L1_excess_std",
        r"$L_1^{\mathrm{excess}}$ (train)", nl_refs=nl_refs,
    )
    plot_vs_k(
        k_points, plots_dir / "test_L1_excess_vs_k.png",
        f"{title_world}: test $L_1^{{\\mathrm{{excess}}}}$ vs k (d={d}, L={layers})",
        "test_L1_excess_mean", "test_L1_excess_std",
        r"$L_1^{\mathrm{excess}}$ (test)", nl_refs=nl_refs,
    )
    k_only = [r for r in k_points if r["display"] in ("k-QSA", "k-CSA", "poly-k-QSA", "poly-k-CSA")]
    plot_vs_k(
        k_only, plots_dir / "train_LB_excess_vs_k.png",
        f"{title_world}: train $L_B^{{\\mathrm{{excess}}}}$ vs k (d={d}, L={layers})",
        "train_LB_excess_mean", "train_LB_excess_std",
        r"$L_B^{\mathrm{excess}}$ (train)", nl_refs=None,
    )
    plot_vs_k(
        k_only, plots_dir / "test_LB_excess_vs_k.png",
        f"{title_world}: test $L_B^{{\\mathrm{{excess}}}}$ vs k (d={d}, L={layers})",
        "test_LB_excess_mean", "test_LB_excess_std",
        r"$L_B^{\mathrm{excess}}$ (test)", nl_refs=None,
    )
    summary = {
        "k_points": [{k: v for k, v in r.items() if k != "runs"} for r in k_points],
        "nl_refs": [{k: v for k, v in r.items() if k != "runs"} for r in nl_refs],
        "plots": {
            "train_L1_excess": str(plots_dir / "train_L1_excess_vs_k.png"),
            "test_L1_excess": str(plots_dir / "test_L1_excess_vs_k.png"),
            "train_LB_excess": str(plots_dir / "train_LB_excess_vs_k.png"),
            "test_LB_excess": str(plots_dir / "test_LB_excess_vs_k.png"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir / 'summary.json'}", flush=True)
    return summary


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="qsa_bench_25_08 campaign: trainable emb + excess L1/LB vs k."
    )
    p.add_argument("--data-mode", choices=["both", "classical", "quantum"], default="both")
    p.add_argument("--T", type=int, default=32)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--dt", type=float, default=0.35)
    p.add_argument("--ks", type=str, default="1,2,3,4,5,6")
    p.add_argument(
        "--layers", type=int, default=0,
        help="kQSA layers. 0 = match CSA param count (d=16 → L=43).",
    )
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
    p.add_argument(
        "--output-dir", type=str,
        default="results/qsa_bench_2508/LB_excess_shared",
    )
    p.add_argument("--quick", action="store_true",
                   help="Tiny smoke: T=8 d=8, k=1,2, 2 seeds, few epochs.")
    args = p.parse_args(argv)

    if args.quick:
        args.T = 8
        args.d = 8
        args.n_qubits = 3
        args.ks = "1,2"
        args.layers = 4
        args.epochs = 30
        args.poly_epochs = 40
        args.nl_epochs = 30
        args.train_size = 16
        args.test_size = 8
        args.n_seeds = 2
        args.min_epochs = 8
        args.patience = 6
        args.batch_size = 8
        args.eval_every = 5
        args.output_dir = str(Path(args.output_dir).parent / "quick_smoke")

    if args.d & (args.d - 1):
        raise ValueError(f"d must be a power of two, got {args.d}")
    if args.data_mode in ("both", "quantum"):
        if 2 ** args.n_qubits != args.d:
            args.n_qubits = int(round(math.log2(args.d)))
        if 2 ** args.n_qubits != args.d:
            raise ValueError(f"quantum requires d=2^n, got d={args.d}")

    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
    if int(args.layers) <= 0:
        args.layers, qsa_np = qb.qsa_layers_matching_csa(args.d)
    else:
        qsa_np = qb.qsa_n_params(args.d, args.layers)
    csa_np = qb.csa_n_params(args.d)
    print(
        f"param match: CSA/nl={csa_np}  QSA={qsa_np} (L={args.layers}, "
        f"n={max(1, math.ceil(math.log2(args.d)))})",
        flush=True,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"HAS_JAX={qb.HAS_JAX}  output={out}", flush=True)
    if not qb.HAS_JAX:
        raise SystemExit("JAX required.")

    if not args.quick:
        ok = qb.selftest()
        if not ok:
            raise SystemExit("qsa_bench_25_08 selftest FAILED")
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
        idx_tr, idx_te, vocab_D, _enc = generate_ptb_indices(
            args.train_size, args.test_size, args.d, args.T, args.seed,
        )
        print(f"PTB vocab D={vocab_D}  idx_tr={idx_tr.shape}  idx_te={idx_te.shape}", flush=True)
        summaries["classical"] = run_world_classical(
            idx_tr, idx_te, args.d, vocab_D, out_dir=out / "classical", **common,
        )

    if args.data_mode in ("both", "quantum"):
        Xtr, Ytr, Xte, Yte = generate_quantum_xy(
            args.train_size, args.test_size, args.n_qubits, args.T, args.dt, args.seed + 100,
        )
        summaries["quantum"] = run_world_quantum(
            Xtr, Ytr, Xte, Yte, out_dir=out / "quantum", **common,
        )

    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nDone. Top-level summary: {out / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
