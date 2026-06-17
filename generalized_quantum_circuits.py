from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import jax.numpy as jnp
import numpy as np

from pennylane_jax_vqt import (
    create_layout,
    get_qnode,
    split_quantum_parameters,
    zero_feature_state,
    zero_system_state,
)
from quantum_utils import get_unitary_from_tk


def _recover_state_from_unitary(unitary_or_state) -> jnp.ndarray:
    array = jnp.asarray(unitary_or_state)
    if array.ndim == 1:
        return array
    return array[:, 0]


def process_sentence_states(states, targets=None):
    states = [np.asarray(state) for state in states]
    if targets is not None:
        targets = [np.asarray(target) for target in targets]
        if len(targets) != len(states):
            raise ValueError("targets length must match states length.")
    target_states = targets if targets is not None else states

    psi_states = []
    next_targets = []
    current_states = []

    for idx in range(1, len(states)):
        psi = None
        for source in states[:idx]:
            source = source / np.linalg.norm(source)
            kron = np.kron(source, source)
            psi = kron if psi is None else psi + kron
        psi = psi / np.linalg.norm(psi)
        psi_states.append(get_unitary_from_tk(psi))
        next_targets.append(get_unitary_from_tk(target_states[idx]).conj().T)
        current_states.append(get_unitary_from_tk(states[idx - 1]).conj().T)

    return psi_states, next_targets, current_states


@dataclass
class GeneralizedQuantumCircuitBuilder:
    embedding_dim: int
    sentence_length: int
    non_linear_order: int = 1

    def __post_init__(self):
        self.n_control_qubits = int(math.ceil(math.log2(max(self.sentence_length, 2))))
        self.n_target_qubits = int(math.log2(self.embedding_dim)) * (self.non_linear_order + 1)
        self.n_total_qubits = self.n_control_qubits + self.n_target_qubits

    def create_generalized_circuit(
        self,
        psi,
        U,
        Z,
        params_v,
        params_k,
        num_layers,
        params_c: Optional[np.ndarray] = None,
    ) -> float:
        layout = create_layout(
            sequence_length=self.sentence_length,
            feature_dimension=self.embedding_dim,
            num_layers=num_layers,
            non_linear_order=self.non_linear_order,
        )
        qnode = get_qnode(layout)

        active_steps = len(U)
        zero_feature = zero_feature_state(self.embedding_dim)
        zero_system = zero_system_state(self.embedding_dim, self.non_linear_order)

        x = jnp.tile(zero_feature, (layout.padded_sequence_length, 1))
        tilde_x = jnp.tile(zero_feature, (layout.padded_sequence_length + 1, 1))
        prep = jnp.tile(zero_system, (layout.padded_sequence_length, 1))

        if active_steps:
            x = x.at[:active_steps].set(
                jnp.stack([_recover_state_from_unitary(item) for item in Z])
            )
            tilde_x = tilde_x.at[1:active_steps + 1].set(
                jnp.stack([_recover_state_from_unitary(item.conj().T) for item in U])
            )
            prep = prep.at[:active_steps].set(
                jnp.stack([_recover_state_from_unitary(item) for item in psi])
            )

        weights_v = jnp.asarray(params_v, dtype=jnp.float64)
        weights_w = jnp.asarray(params_k, dtype=jnp.float64)
        if params_c is None:
            weights_c = jnp.zeros(layout.weights_c_shape, dtype=jnp.float64)
        else:
            weights_c = jnp.asarray(params_c, dtype=jnp.float64)

        overlap = qnode(x, tilde_x, prep, weights_v, weights_w, weights_c)
        overlap = jnp.clip(jnp.real(overlap), 1e-12, 1.0)
        return float(-jnp.log(overlap))

    def get_circuit_info(self) -> Dict[str, int]:
        layout = create_layout(
            sequence_length=self.sentence_length,
            feature_dimension=self.embedding_dim,
            num_layers=1,
            non_linear_order=self.non_linear_order,
        )
        return {
            "embedding_dim": self.embedding_dim,
            "sentence_length": self.sentence_length,
            "n_target_qubits": self.n_target_qubits,
            "n_control_qubits": self.n_control_qubits,
            "n_total_qubits": self.n_total_qubits,
            "padded_sequence_length": layout.padded_sequence_length,
        }


class AdaptiveQuantumCircuitFactory:
    @staticmethod
    def create_circuit_builder(embedding_dim: int, sentence_length: int) -> GeneralizedQuantumCircuitBuilder:
        return GeneralizedQuantumCircuitBuilder(
            embedding_dim=embedding_dim,
            sentence_length=sentence_length,
        )

    @staticmethod
    def get_optimal_embedding_dim(vocab_size: int) -> int:
        if vocab_size <= 4:
            return 4
        if vocab_size <= 16:
            return 16
        if vocab_size <= 64:
            return 64
        return 256

    @staticmethod
    def estimate_circuit_complexity(embedding_dim: int, sentence_length: int) -> Dict[str, int | bool]:
        feature_qubits = int(math.log2(embedding_dim))
        control_qubits = int(math.ceil(math.log2(max(sentence_length, 2))))
        total_qubits = control_qubits + 2 * feature_qubits
        padded_sequence_length = 1 << control_qubits
        estimated_branches = padded_sequence_length
        is_feasible = total_qubits <= 24
        return {
            "embedding_dim": embedding_dim,
            "sentence_length": sentence_length,
            "feature_qubits": feature_qubits,
            "control_qubits": control_qubits,
            "total_qubits": total_qubits,
            "estimated_branches": estimated_branches,
            "is_feasible": is_feasible,
        }
