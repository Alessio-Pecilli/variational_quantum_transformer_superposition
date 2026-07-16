"""
Ottimizzazione qiskit-free basata su gradienti JAX per input legacy (psi, U, Z).
"""

from __future__ import annotations

import time
from itertools import zip_longest

import jax
import jax.numpy as jnp
import numpy as np

from pennylane_jax_vqt import (
    apply_adam_step,
    create_layout,
    flatten_quantum_parameters,
    get_qnode,
    initialize_quantum_parameters,
    split_quantum_parameters,
    zero_feature_state,
    zero_system_state,
    zeros_like_tree,
)
from visualization import save_loss_plot, save_loss_values_to_file, save_parameters


def _recover_state_from_unitary(unitary_or_state) -> jnp.ndarray:
    array = jnp.asarray(unitary_or_state, dtype=jnp.complex128)
    if array.ndim == 1:
        return array
    return array[:, 0]


def _prepare_legacy_inputs(psi, U, Z, embedding_dim: int, num_layers: int):
    sentence_length = max(len(U) + 1, 2)
    layout = create_layout(
        sequence_length=sentence_length,
        feature_dimension=embedding_dim,
        num_layers=num_layers,
        non_linear_order=2,
    )

    active_steps = len(U)
    zero_feature = zero_feature_state(embedding_dim)
    zero_system = zero_system_state(embedding_dim, 2)

    x = jnp.tile(zero_feature, (layout.padded_sequence_length, 1))
    tilde_x = jnp.tile(zero_feature, (layout.padded_sequence_length + 1, 1))
    prep = jnp.tile(zero_system, (layout.padded_sequence_length, 1))

    if active_steps:
        x = x.at[:active_steps].set(jnp.stack([_recover_state_from_unitary(item) for item in Z]))
        tilde_x = tilde_x.at[1:active_steps + 1].set(
            jnp.stack([_recover_state_from_unitary(item.conj().T) for item in U])
        )
        prep = prep.at[:active_steps].set(jnp.stack([_recover_state_from_unitary(item) for item in psi]))

    return layout, x, tilde_x, prep


def _normalize_initial_params(best_params, layout):
    n_v = int(np.prod(layout.weights_v_shape))
    n_w = int(np.prod(layout.weights_w_shape))
    n_c = int(np.prod(layout.weights_c_shape))

    if best_params is None:
        params = initialize_quantum_parameters(jax.random.PRNGKey(0), layout)
        return flatten_quantum_parameters(
            params["weights_v"],
            params["weights_w"],
            params["weights_c"],
        )

    best_params = np.asarray(best_params, dtype=np.float64).reshape(-1)
    if best_params.size == n_v + n_w + n_c:
        return jnp.asarray(best_params, dtype=jnp.float64)
    if best_params.size == n_v + n_w:
        padded = np.concatenate([best_params, np.zeros((n_c,), dtype=np.float64)])
        return jnp.asarray(padded, dtype=jnp.float64)

    raise ValueError(
        f"Unexpected parameter vector size {best_params.size}. "
        f"Expected {n_v + n_w} (legacy) or {n_v + n_w + n_c} (jax backend)."
    )


def _build_loss_function(psi, U, Z, num_layers, dim):
    layout, x, tilde_x, prep = _prepare_legacy_inputs(psi, U, Z, dim, num_layers)
    qnode = get_qnode(layout)

    def loss_fn(flat_params):
        weights_v, weights_w, weights_c = split_quantum_parameters(flat_params, layout)
        overlap = qnode(x, tilde_x, prep, weights_v, weights_w, weights_c)
        overlap = jnp.clip(jnp.real(overlap), 1e-12, 1.0)
        return -jnp.log(overlap)

    return layout, jax.jit(loss_fn), jax.jit(jax.value_and_grad(loss_fn))


def compute_loss_variant(args):
    params, shift_value, states_calculated, U, Z, num_layers, embedding_dim = args
    layout, loss_fn, _ = _build_loss_function(
        states_calculated,
        U,
        Z,
        num_layers=num_layers,
        dim=embedding_dim,
    )
    del layout
    params = _normalize_initial_params(params, create_layout(
        sequence_length=max(len(U) + 1, 2),
        feature_dimension=embedding_dim,
        num_layers=num_layers,
        non_linear_order=2,
    ))
    params = params + shift_value
    return float(loss_fn(params))


def adam_update(params, gradients, m, v, lr, beta1, beta2, epsilon, t):
    m = beta1 * m + (1 - beta1) * gradients
    v = beta2 * v + (1 - beta2) * (gradients ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    params = params - lr * m_hat / (np.sqrt(v_hat) + epsilon)
    return np.mod(params + np.pi, 2 * np.pi) - np.pi, m, v


def aggregate_losses(loss_lists):
    cols = list(zip_longest(*loss_lists, fillvalue=None))
    media, best, worst = [], [], []
    for col in cols:
        vals = [v for v in col if v is not None]
        if vals:
            media.append(sum(vals) / len(vals))
            best.append(min(vals))
            worst.append(max(vals))
    return media, best, worst


def create_loss_function(circuit_function, psi, U, Z, num_layers, n_qubit, dim, param_shape, n_params):
    del circuit_function, n_qubit, param_shape, n_params
    _, loss_fn, _ = _build_loss_function(psi, U, Z, num_layers=num_layers, dim=dim)

    def wrapped_loss(params_all):
        params_all = jnp.asarray(params_all, dtype=jnp.float64)
        return float(loss_fn(params_all))

    return wrapped_loss


def optimize_parameters(
    max_hours,
    num_iterations,
    num_layers,
    psi,
    U,
    Z,
    n_qubit,
    best_params=None,
    dim=16,
    opt_maxiter=40,
    opt_maxfev=60,
):
    del max_hours, num_iterations, n_qubit, opt_maxfev

    layout, _, loss_and_grad = _build_loss_function(psi, U, Z, num_layers=num_layers, dim=dim)
    params_flat = _normalize_initial_params(best_params, layout)
    params = {"flat": params_flat}
    first_moment = zeros_like_tree(params)
    second_moment = zeros_like_tree(params)

    epochs = int(opt_maxiter)
    loss_history = []
    best_loss = float("inf")
    best_flat = params_flat
    learning_rate = 1e-3
    start = time.time()

    @jax.jit
    def optimizer_step(params_tree, grads_tree, m_tree, v_tree, step):
        return apply_adam_step(
            params=params_tree,
            grads=grads_tree,
            first_moment=m_tree,
            second_moment=v_tree,
            step=step,
            learning_rate=learning_rate,
        )

    for epoch in range(1, epochs + 1):
        loss, grad = loss_and_grad(params["flat"])
        grads = {"flat": grad}
        params, first_moment, second_moment = optimizer_step(
            params,
            grads,
            first_moment,
            second_moment,
            epoch,
        )

        loss_value = float(np.asarray(jax.device_get(loss)))
        loss_history.append(loss_value)

        if loss_value < best_loss:
            best_loss = loss_value
            best_flat = np.asarray(jax.device_get(params["flat"]))

        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            elapsed = time.time() - start
            print(f"[JAX-OPT] epoch={epoch:04d}/{epochs:04d} loss={loss_value:.8f} elapsed={elapsed:.1f}s")

    try:
        save_parameters(best_flat)
        avg_losses, best_losses, worst_losses = aggregate_losses([loss_history])
        save_loss_plot(avg_losses, best_losses, worst_losses, num_layers)
        save_loss_values_to_file(avg_losses, best_losses, worst_losses, "loss_results.txt")
    except Exception:
        pass

    return best_flat


def optimize_experimental_parameters(
    max_hours,
    num_iterations,
    num_layers,
    psi,
    U,
    Z,
    best_params=None,
    dim=16,
    opt_maxiter=40,
    opt_maxfev=60,
):
    return optimize_parameters(
        max_hours=max_hours,
        num_iterations=num_iterations,
        num_layers=num_layers,
        psi=psi,
        U=U,
        Z=Z,
        n_qubit=int(np.log2(dim)),
        best_params=best_params,
        dim=dim,
        opt_maxiter=opt_maxiter,
        opt_maxfev=opt_maxfev,
    )


def optimize_parameters_parallel(
    params,
    shift,
    states_calculated,
    U,
    Z,
    num_layers,
    embedding_dim,
    num_qubits,
    sentence_length,
    num_workers=None,
    opt_maxiter=150,
    opt_maxfev=50,
):
    del shift, num_qubits, sentence_length, num_workers
    return optimize_parameters(
        max_hours=1,
        num_iterations=opt_maxiter,
        num_layers=num_layers,
        psi=states_calculated,
        U=U,
        Z=Z,
        n_qubit=int(np.log2(embedding_dim)),
        best_params=params,
        dim=embedding_dim,
        opt_maxiter=opt_maxiter,
        opt_maxfev=opt_maxfev,
    )
