from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import pennylane as qml


jax.config.update("jax_enable_x64", True)


EPSILON = 1e-12


@dataclass(frozen=True)
class QuantumParameterLayout:
    sequence_length: int
    active_branch_count: int
    padded_sequence_length: int
    control_qubits: int
    feature_dimension: int
    feature_qubits: int
    non_linear_order: int
    prune_inactive_branches: bool
    num_layers: int
    blocks_per_ansatz: int
    weights_v_shape: Tuple[int, int]
    weights_w_shape: Tuple[int, int]
    weights_c_shape: Tuple[int, ...]
    total_parameters: int


def _system_dimension(feature_dimension: int, non_linear_order: int) -> int:
    return feature_dimension ** (non_linear_order + 1)


def _safe_log2_power(value: int) -> int:
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"Expected a power of two, got {value}.")
    return int(round(math.log2(value)))


def normalize_state(state: jnp.ndarray, eps: float = EPSILON) -> jnp.ndarray:
    state = jnp.nan_to_num(
        jnp.asarray(state),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    norm = jnp.linalg.norm(state)
    fallback = jnp.zeros_like(state).at[0].set(1.0 + 0.0j)
    safe_state = state / jnp.maximum(norm, eps)
    return jnp.where(jnp.isfinite(norm) & (norm > eps), safe_state, fallback)


def normalize_state_batch(states: jnp.ndarray, eps: float = EPSILON) -> jnp.ndarray:
    states = jnp.nan_to_num(
        jnp.asarray(states),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    norms = jnp.linalg.norm(states, axis=-1, keepdims=True)
    safe_states = states / jnp.maximum(norms, eps)
    fallback = jnp.zeros_like(states)
    fallback = fallback.at[..., 0].set(1.0 + 0.0j)
    valid = jnp.isfinite(norms) & (norms > eps)
    return jnp.where(valid, safe_states, fallback)


def kronecker_power(state: jnp.ndarray, power: int) -> jnp.ndarray:
    result = state
    for _ in range(power - 1):
        result = jnp.kron(result, state)
    return result


def count_blocks(n_qubits: int, num_layers: int) -> int:
    if n_qubits < 2:
        return 0
    even_blocks = len(list(range(0, n_qubits - 1, 2)))
    odd_blocks = len(list(range(1, n_qubits - 1, 2)))
    return num_layers * (even_blocks + odd_blocks)


def create_layout(
    sequence_length: int,
    feature_dimension: int,
    num_layers: int,
    non_linear_order: int = 1,
    prune_inactive_branches: bool = False,
    active_branch_count: Optional[int] = None,
) -> QuantumParameterLayout:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")
    feature_qubits = _safe_log2_power(feature_dimension)
    if active_branch_count is None:
        active_branch_count = max(sequence_length - 1, 0)
    active_branch_count = max(int(active_branch_count), 0)
    branch_register_length = (
        active_branch_count if prune_inactive_branches else max(sequence_length, active_branch_count)
    )
    padded_sequence_length = 1 << int(math.ceil(math.log2(max(branch_register_length, 2))))
    active_branch_count = min(active_branch_count, padded_sequence_length)
    control_qubits = int(math.log2(padded_sequence_length))
    blocks_per_ansatz = count_blocks(feature_qubits, num_layers)
    weights_v_shape = (blocks_per_ansatz, 12)
    weights_w_shape = (blocks_per_ansatz, 12)
    weights_c_shape = (3 * control_qubits,)
    total_parameters = (
        int(jnp.prod(jnp.asarray(weights_v_shape)))
        + int(jnp.prod(jnp.asarray(weights_w_shape)))
        + weights_c_shape[0]
    )
    return QuantumParameterLayout(
        sequence_length=sequence_length,
        active_branch_count=active_branch_count,
        padded_sequence_length=padded_sequence_length,
        control_qubits=control_qubits,
        feature_dimension=feature_dimension,
        feature_qubits=feature_qubits,
        non_linear_order=non_linear_order,
        prune_inactive_branches=prune_inactive_branches,
        num_layers=num_layers,
        blocks_per_ansatz=blocks_per_ansatz,
        weights_v_shape=weights_v_shape,
        weights_w_shape=weights_w_shape,
        weights_c_shape=weights_c_shape,
        total_parameters=total_parameters,
    )


def zero_feature_state(feature_dimension: int, dtype=jnp.complex128) -> jnp.ndarray:
    state = jnp.zeros((feature_dimension,), dtype=dtype)
    return state.at[0].set(1.0 + 0.0j)


def zero_system_state(
    feature_dimension: int,
    non_linear_order: int,
    dtype=jnp.complex128,
) -> jnp.ndarray:
    return kronecker_power(zero_feature_state(feature_dimension, dtype=dtype), non_linear_order + 1)


def sequence_to_circuit_inputs(
    current_states: jnp.ndarray,
    next_targets: jnp.ndarray,
    padded_sequence_length: int,
    non_linear_order: int = 1,
    prune_inactive_branches: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    current_states = normalize_state_batch(jnp.asarray(current_states, dtype=jnp.complex128))
    next_targets = normalize_state_batch(jnp.asarray(next_targets, dtype=jnp.complex128))

    active_steps = int(current_states.shape[0])
    feature_dimension = int(current_states.shape[-1]) if active_steps else int(next_targets.shape[-1])
    padded_x = jnp.tile(
        zero_feature_state(feature_dimension, dtype=current_states.dtype),
        (padded_sequence_length, 1),
    )
    padded_targets = jnp.tile(
        zero_feature_state(feature_dimension, dtype=next_targets.dtype),
        (padded_sequence_length + 1, 1),
    )
    system_zero = zero_system_state(
        feature_dimension=feature_dimension,
        non_linear_order=non_linear_order,
        dtype=current_states.dtype,
    )

    if active_steps:
        padded_x = padded_x.at[:active_steps].set(current_states)
        padded_targets = padded_targets.at[1:active_steps + 1].set(next_targets)

    branch_states = []
    for branch_idx in range(padded_sequence_length):
        if branch_idx < active_steps:
            branch_vec = jnp.zeros_like(system_zero)
            for source_idx in range(branch_idx + 1):
                branch_vec = branch_vec + kronecker_power(
                    current_states[source_idx], non_linear_order + 1
                )
            branch_states.append(normalize_state(branch_vec))
        else:
            if prune_inactive_branches:
                branch_states.append(jnp.zeros_like(system_zero))
            else:
                branch_states.append(system_zero)

    return padded_x, padded_targets, jnp.stack(branch_states)


def batch_sequences_to_circuit_inputs(
    current_batch: jnp.ndarray,
    next_batch: jnp.ndarray,
    padded_sequence_length: int,
    non_linear_order: int = 1,
    prune_inactive_branches: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    transform = partial(
        sequence_to_circuit_inputs,
        padded_sequence_length=padded_sequence_length,
        non_linear_order=non_linear_order,
        prune_inactive_branches=prune_inactive_branches,
    )
    return jax.vmap(transform)(current_batch, next_batch)


def isometrize_matrix(matrix: jnp.ndarray) -> jnp.ndarray:
    matrix = jnp.nan_to_num(
        jnp.asarray(matrix, dtype=jnp.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    q, r = jnp.linalg.qr(matrix, mode="reduced")
    q = jnp.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    r = jnp.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    diag = jnp.sign(jnp.real(jnp.diag(r)))
    diag = jnp.where(diag == 0, 1.0, diag)
    return q * diag


def materialize_text_matrices(
    raw_embedding: jnp.ndarray,
    raw_rotation: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    embedding = isometrize_matrix(raw_embedding)
    rotation = isometrize_matrix(raw_rotation)
    output = embedding @ rotation
    return embedding, rotation, output


def prepare_text_batch(
    token_batch: jnp.ndarray,
    positional_encoding: jnp.ndarray,
    raw_embedding: jnp.ndarray,
    raw_rotation: jnp.ndarray,
    layout: QuantumParameterLayout,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    embedding, rotation, output = materialize_text_matrices(raw_embedding, raw_rotation)
    token_batch = jnp.asarray(token_batch, dtype=jnp.int32)
    positional_encoding = jnp.asarray(positional_encoding, dtype=jnp.float64)
    current_states = embedding[token_batch[:, :-1]] + positional_encoding[:-1]
    next_targets = output[token_batch[:, 1:]]
    x_batch, tilde_batch, prep_batch = batch_sequences_to_circuit_inputs(
        current_states,
        next_targets,
        padded_sequence_length=layout.padded_sequence_length,
        non_linear_order=layout.non_linear_order,
        prune_inactive_branches=layout.prune_inactive_branches,
    )
    return x_batch, tilde_batch, prep_batch, embedding, rotation, output


def project_quantum_states(
    state_batch: jnp.ndarray,
    raw_projection: Optional[jnp.ndarray],
) -> jnp.ndarray:
    state_batch = jnp.asarray(state_batch, dtype=jnp.complex128)
    if raw_projection is None:
        return normalize_state_batch(state_batch)
    projection = jnp.asarray(raw_projection, dtype=jnp.complex128)
    projected = jnp.einsum("ij,bsj->bsi", projection, state_batch)
    return normalize_state_batch(projected)


def prepare_quantum_batch(
    state_batch: jnp.ndarray,
    layout: QuantumParameterLayout,
    raw_projection: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    projected = project_quantum_states(state_batch, raw_projection)
    current_states = projected[:, :-1]
    next_targets = projected[:, 1:]
    return batch_sequences_to_circuit_inputs(
        current_states,
        next_targets,
        padded_sequence_length=layout.padded_sequence_length,
        non_linear_order=layout.non_linear_order,
        prune_inactive_branches=layout.prune_inactive_branches,
    )


def combined_targets(
    x: jnp.ndarray,
    tilde_x: jnp.ndarray,
    branch_count: int,
    non_linear_order: int,
) -> jnp.ndarray:
    branch_count = max(int(branch_count), 0)
    x = jnp.asarray(x, dtype=jnp.complex128)
    tilde_x = jnp.asarray(tilde_x, dtype=jnp.complex128)
    targets = tilde_x[1:branch_count + 1]
    if branch_count == 0:
        return jnp.zeros(
            (0, _system_dimension(int(x.shape[-1]), non_linear_order)),
            dtype=jnp.complex128,
        )
    x_prefix = x[:branch_count]
    for _ in range(non_linear_order):
        targets = jax.vmap(jnp.kron)(targets, x_prefix)
    return normalize_state_batch(targets)


def circuit_g_block(q1: int, q2: int, block_weights: jnp.ndarray) -> None:
    qml.RX(block_weights[0], wires=q1)
    qml.RY(block_weights[1], wires=q1)
    qml.RZ(block_weights[2], wires=q1)
    qml.RX(block_weights[3], wires=q2)
    qml.RY(block_weights[4], wires=q2)
    qml.RZ(block_weights[5], wires=q2)
    qml.CNOT(wires=[q1, q2])
    qml.RX(block_weights[6], wires=q1)
    qml.RY(block_weights[7], wires=q1)
    qml.RZ(block_weights[8], wires=q1)
    qml.RX(block_weights[9], wires=q2)
    qml.RY(block_weights[10], wires=q2)
    qml.RZ(block_weights[11], wires=q2)


def apply_sim_ansatz(wires: Tuple[int, ...], ansatz_weights: jnp.ndarray, num_layers: int) -> None:
    n_wires = len(wires)
    if n_wires < 2:
        return
    block_idx = 0
    for _ in range(num_layers):
        for i in range(0, n_wires - 1, 2):
            circuit_g_block(wires[i], wires[i + 1], ansatz_weights[block_idx])
            block_idx += 1
        for i in range(1, n_wires - 1, 2):
            circuit_g_block(wires[i], wires[i + 1], ansatz_weights[block_idx])
            block_idx += 1


def _branch_limit(layout: QuantumParameterLayout) -> int:
    return layout.active_branch_count if layout.prune_inactive_branches else layout.padded_sequence_length


def _resolve_readout_mode(layout: QuantumParameterLayout, readout_mode: str) -> str:
    if readout_mode == "auto":
        return "uniform_active" if layout.prune_inactive_branches else "hadamard"
    if readout_mode not in {"hadamard", "uniform_active"}:
        raise ValueError(
            f"Unsupported readout_mode '{readout_mode}'. Expected 'auto', 'hadamard', or 'uniform_active'."
        )
    return readout_mode


def _uniform_readout_state(layout: QuantumParameterLayout) -> jnp.ndarray:
    state = jnp.zeros((layout.padded_sequence_length,), dtype=jnp.complex128)
    branch_count = _branch_limit(layout)
    if branch_count <= 0:
        return state.at[0].set(1.0 + 0.0j)
    amplitude = 1.0 / math.sqrt(branch_count)
    return state.at[:branch_count].set(amplitude + 0.0j)


def _rot_matrix(phi: jnp.ndarray, theta: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
    """qml.Rot(phi, theta, omega) as a 2x2 unitary matrix."""
    half_phi = phi / 2.0
    half_theta = theta / 2.0
    half_omega = omega / 2.0
    exp_minus = jnp.exp(-0.5j * omega)
    exp_plus = jnp.exp(0.5j * omega)
    exp_phi_minus = jnp.exp(-0.5j * phi)
    exp_phi_plus = jnp.exp(0.5j * phi)
    c = jnp.cos(half_theta)
    s = jnp.sin(half_theta)
    return jnp.array(
        [
            [exp_minus * exp_phi_minus * c, -exp_minus * exp_phi_plus * s],
            [exp_plus * exp_phi_minus * s, exp_plus * exp_phi_plus * c],
        ],
        dtype=jnp.complex128,
    )


def _control_readout_row(
    layout: QuantumParameterLayout,
    weights_c: jnp.ndarray,
    readout_mode: str,
    uniform_state: jnp.ndarray,
) -> jnp.ndarray:
    readout_mode = _resolve_readout_mode(layout, readout_mode)
    control_matrix = jnp.array([[1.0 + 0.0j]], dtype=jnp.complex128)
    for wire_idx in range(layout.control_qubits):
        base = 3 * wire_idx
        control_matrix = jnp.kron(
            control_matrix,
            _rot_matrix(weights_c[base], weights_c[base + 1], weights_c[base + 2]),
        )
    if readout_mode == "hadamard":
        hadamard = (1.0 / math.sqrt(2.0)) * jnp.array(
            [[1.0, 1.0], [1.0, -1.0]],
            dtype=jnp.complex128,
        )
        hadamard_layer = jnp.array([[1.0 + 0.0j]], dtype=jnp.complex128)
        for _ in range(layout.control_qubits):
            hadamard_layer = jnp.kron(hadamard_layer, hadamard)
        measurement_row = hadamard_layer[0]
    else:
        measurement_row = uniform_state.conj()
    return measurement_row @ control_matrix


@lru_cache(maxsize=None)
def _build_circuit_context(
    layout: QuantumParameterLayout,
    device_name: str = "lightning.qubit",
    interface: str = "jax",
    diff_method: str = "adjoint",
    readout_mode: str = "auto",
) -> Dict[str, object]:
    wires_c = tuple(range(layout.control_qubits))
    start_a = layout.control_qubits
    wires_a = tuple(range(start_a, start_a + layout.feature_qubits))
    start_b = start_a + layout.feature_qubits
    wires_b = tuple(
        tuple(
            range(
                start_b + register_idx * layout.feature_qubits,
                start_b + (register_idx + 1) * layout.feature_qubits,
            )
        )
        for register_idx in range(layout.non_linear_order)
    )
    all_system_wires = wires_a + tuple(wire for register in wires_b for wire in register)
    total_wires = wires_c + all_system_wires
    device = qml.device(device_name, wires=total_wires)
    branch_limit = _branch_limit(layout)
    readout_mode = _resolve_readout_mode(layout, readout_mode)
    uniform_readout_state = _uniform_readout_state(layout)

    @qml.qnode(device, interface=interface, diff_method=diff_method)
    def qnode(
        x: jnp.ndarray,
        tilde_x: jnp.ndarray,
        precomputed_states: jnp.ndarray,
        weights_v: jnp.ndarray,
        weights_w: jnp.ndarray,
        weights_c: jnp.ndarray,
    ) -> jnp.ndarray:
        global_state = jnp.reshape(precomputed_states, (-1,))
        global_state = normalize_state(global_state)
        qml.StatePrep(global_state, wires=total_wires)

        if layout.blocks_per_ansatz > 0:
            apply_sim_ansatz(wires_a, weights_v, layout.num_layers)
            for register in wires_b:
                apply_sim_ansatz(register, weights_w, layout.num_layers)

        for branch_idx in range(branch_limit):
            control_values = tuple(
                int(bit)
                for bit in format(branch_idx, f"0{layout.control_qubits}b")
            )
            combined_target = tilde_x[branch_idx + 1]
            for _ in range(layout.non_linear_order):
                combined_target = jnp.kron(combined_target, x[branch_idx])
            combined_target = normalize_state(combined_target)
            qml.ctrl(
                qml.adjoint(qml.StatePrep),
                control=wires_c,
                control_values=control_values,
            )(combined_target, wires=all_system_wires)

        idx_c = 0
        for wire in wires_c:
            qml.Rot(
                weights_c[idx_c],
                weights_c[idx_c + 1],
                weights_c[idx_c + 2],
                wires=wire,
            )
            idx_c += 3

        if readout_mode == "uniform_active":
            qml.adjoint(qml.StatePrep(uniform_readout_state, wires=wires_c))
        else:
            for wire in wires_c:
                qml.Hadamard(wires=wire)

        return qml.expval(qml.Projector([0] * len(total_wires), wires=total_wires))

    @qml.qnode(device, interface=interface, diff_method=diff_method)
    def state_qnode(
        precomputed_states: jnp.ndarray,
        weights_v: jnp.ndarray,
        weights_w: jnp.ndarray,
    ) -> jnp.ndarray:
        global_state = jnp.reshape(precomputed_states, (-1,))
        global_state = normalize_state(global_state)
        qml.StatePrep(global_state, wires=total_wires)

        if layout.blocks_per_ansatz > 0:
            apply_sim_ansatz(wires_a, weights_v, layout.num_layers)
            for register in wires_b:
                apply_sim_ansatz(register, weights_w, layout.num_layers)

        return qml.state()

    def overlap_probability(
        state: jnp.ndarray,
        x: jnp.ndarray,
        tilde_x: jnp.ndarray,
        weights_c: jnp.ndarray,
    ) -> jnp.ndarray:
        system_dimension = _system_dimension(layout.feature_dimension, layout.non_linear_order)
        psi = jnp.reshape(state, (layout.padded_sequence_length, system_dimension))[:branch_limit]
        targets = combined_targets(x, tilde_x, branch_limit, layout.non_linear_order)
        branch_overlaps = jnp.einsum("jd,jd->j", targets.conj(), psi)
        readout_row = _control_readout_row(layout, weights_c, readout_mode, uniform_readout_state)
        amplitude = readout_row[:branch_limit] @ branch_overlaps
        return jnp.abs(amplitude) ** 2

    vmapped_qnode = jax.vmap(qnode, in_axes=(0, 0, 0, None, None, None))
    vmapped_state_qnode = jax.vmap(state_qnode, in_axes=(0, None, None))
    vmapped_overlap_probability = jax.vmap(overlap_probability, in_axes=(0, 0, 0, None))
    return {
        "qnode": qnode,
        "vmapped_qnode": vmapped_qnode,
        "state_qnode": state_qnode,
        "vmapped_state_qnode": vmapped_state_qnode,
        "overlap_probability": overlap_probability,
        "vmapped_overlap_probability": vmapped_overlap_probability,
        "wires_c": wires_c,
        "wires_a": wires_a,
        "wires_b": wires_b,
        "all_system_wires": all_system_wires,
        "total_wires": total_wires,
        "device_name": device_name,
        "interface": interface,
        "diff_method": diff_method,
        "readout_mode": readout_mode,
        "branch_limit": branch_limit,
    }


def get_qnode(
    layout: QuantumParameterLayout,
    device_name: str = "lightning.qubit",
    interface: str = "jax",
    diff_method: str = "adjoint",
    readout_mode: str = "auto",
):
    return _build_circuit_context(layout, device_name, interface, diff_method, readout_mode)["qnode"]


def get_vmapped_qnode(
    layout: QuantumParameterLayout,
    device_name: str = "lightning.qubit",
    interface: str = "jax",
    diff_method: str = "adjoint",
    readout_mode: str = "auto",
):
    return _build_circuit_context(layout, device_name, interface, diff_method, readout_mode)["vmapped_qnode"]


def get_state_qnode(
    layout: QuantumParameterLayout,
    device_name: str = "lightning.qubit",
    interface: str = "jax",
    diff_method: str = "adjoint",
    readout_mode: str = "auto",
):
    return _build_circuit_context(layout, device_name, interface, diff_method, readout_mode)["state_qnode"]


def get_vmapped_state_qnode(
    layout: QuantumParameterLayout,
    device_name: str = "lightning.qubit",
    interface: str = "jax",
    diff_method: str = "adjoint",
    readout_mode: str = "auto",
):
    return _build_circuit_context(layout, device_name, interface, diff_method, readout_mode)["vmapped_state_qnode"]


def get_overlap_probability_fn(
    layout: QuantumParameterLayout,
    device_name: str = "lightning.qubit",
    interface: str = "jax",
    diff_method: str = "adjoint",
    readout_mode: str = "auto",
):
    return _build_circuit_context(layout, device_name, interface, diff_method, readout_mode)["overlap_probability"]


def get_vmapped_overlap_probability_fn(
    layout: QuantumParameterLayout,
    device_name: str = "lightning.qubit",
    interface: str = "jax",
    diff_method: str = "adjoint",
    readout_mode: str = "auto",
):
    return _build_circuit_context(layout, device_name, interface, diff_method, readout_mode)[
        "vmapped_overlap_probability"
    ]


def flatten_quantum_parameters(
    weights_v: jnp.ndarray,
    weights_w: jnp.ndarray,
    weights_c: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.concatenate(
        [
            jnp.ravel(jnp.asarray(weights_v, dtype=jnp.float64)),
            jnp.ravel(jnp.asarray(weights_w, dtype=jnp.float64)),
            jnp.ravel(jnp.asarray(weights_c, dtype=jnp.float64)),
        ]
    )


def split_quantum_parameters(
    params_quantum: jnp.ndarray,
    layout: QuantumParameterLayout,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    params_quantum = jnp.asarray(params_quantum, dtype=jnp.float64)
    n_v = int(jnp.prod(jnp.asarray(layout.weights_v_shape)))
    n_w = int(jnp.prod(jnp.asarray(layout.weights_w_shape)))
    n_c = layout.weights_c_shape[0]
    expected = n_v + n_w + n_c
    if params_quantum.size != expected:
        raise ValueError(
            f"Quantum parameter vector mismatch: expected {expected}, got {params_quantum.size}."
        )
    weights_v = params_quantum[:n_v].reshape(layout.weights_v_shape)
    weights_w = params_quantum[n_v:n_v + n_w].reshape(layout.weights_w_shape)
    weights_c = params_quantum[n_v + n_w:n_v + n_w + n_c].reshape(layout.weights_c_shape)
    return weights_v, weights_w, weights_c


def initialize_quantum_parameters(
    key: jax.Array,
    layout: QuantumParameterLayout,
) -> Dict[str, jnp.ndarray]:
    key_v, key_w, key_c = jax.random.split(key, 3)
    return {
        "weights_v": jax.random.uniform(
            key_v,
            layout.weights_v_shape,
            minval=0.0,
            maxval=2.0 * jnp.pi,
            dtype=jnp.float64,
        ),
        "weights_w": jax.random.uniform(
            key_w,
            layout.weights_w_shape,
            minval=0.0,
            maxval=2.0 * jnp.pi,
            dtype=jnp.float64,
        ),
        "weights_c": jax.random.uniform(
            key_c,
            layout.weights_c_shape,
            minval=0.0,
            maxval=2.0 * jnp.pi,
            dtype=jnp.float64,
        ),
    }


def apply_gradient_step(
    params: Dict[str, jnp.ndarray],
    grads: Dict[str, jnp.ndarray],
    learning_rate: float,
) -> Dict[str, jnp.ndarray]:
    return jax.tree_util.tree_map(
        lambda value, grad: value - learning_rate * grad,
        params,
        grads,
    )


def apply_adam_step(
    params: Dict[str, jnp.ndarray],
    grads: Dict[str, jnp.ndarray],
    first_moment: Dict[str, jnp.ndarray],
    second_moment: Dict[str, jnp.ndarray],
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    grads = jax.tree_util.tree_map(
        lambda g: jnp.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0),
        grads,
    )
    first_moment = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1.0 - beta1) * g,
        first_moment,
        grads,
    )
    second_moment = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1.0 - beta2) * (g ** 2),
        second_moment,
        grads,
    )
    corrected_m = jax.tree_util.tree_map(
        lambda m: m / (1.0 - beta1 ** step),
        first_moment,
    )
    corrected_v = jax.tree_util.tree_map(
        lambda v: v / (1.0 - beta2 ** step),
        second_moment,
    )
    updated_params = jax.tree_util.tree_map(
        lambda p, m, v: p - learning_rate * m / (jnp.sqrt(v) + epsilon),
        params,
        corrected_m,
        corrected_v,
    )
    updated_params = jax.tree_util.tree_map(
        lambda p: jnp.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0),
        updated_params,
    )
    return updated_params, first_moment, second_moment


def zeros_like_tree(params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    return jax.tree_util.tree_map(jnp.zeros_like, params)


def compute_overlap_loss(overlaps: jnp.ndarray, eps: float = EPSILON) -> jnp.ndarray:
    overlaps = jnp.clip(jnp.real(overlaps), eps, 1.0)
    return -jnp.mean(jnp.log(overlaps))


def ansatz_matrix(weights: jnp.ndarray, feature_qubits: int, num_layers: int) -> jnp.ndarray:
    wires = tuple(range(feature_qubits))

    def ansatz_template(local_weights):
        apply_sim_ansatz(wires, local_weights, num_layers)

    return jnp.asarray(
        qml.matrix(ansatz_template, wire_order=wires)(jnp.asarray(weights, dtype=jnp.float64))
    )
