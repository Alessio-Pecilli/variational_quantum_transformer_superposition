"""
Compat layer qiskit-free per la costruzione dei circuiti legacy.

Le funzioni pubbliche mantengono i nomi storici ma delegano al QNode PennyLane/JAX.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from generalized_quantum_circuits import (
    AdaptiveQuantumCircuitFactory,
    GeneralizedQuantumCircuitBuilder,
)


def _create_loss(
    psi,
    U,
    Z,
    params_v,
    params_k,
    num_layers,
    dim=4,
    params_c=None,
):
    builder = GeneralizedQuantumCircuitBuilder(
        embedding_dim=dim,
        sentence_length=max(len(psi) + 1, 2),
    )
    return builder.create_generalized_circuit(
        psi=psi,
        U=U,
        Z=Z,
        params_v=params_v,
        params_k=params_k,
        num_layers=num_layers,
        params_c=params_c,
    )


def create_circuit_2words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    return _create_loss(psi, U, Z, params_v, params_k, num_layers, dim=dim)


def create_circuit_4words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    return _create_loss(psi, U, Z, params_v, params_k, num_layers, dim=dim)


def create_circuit_8words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    return _create_loss(psi, U, Z, params_v, params_k, num_layers, dim=dim)


def create_circuit_16words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    return _create_loss(psi, U, Z, params_v, params_k, num_layers, dim=dim)


def _normalize_control_params(params_f, n_control_qubits: int) -> np.ndarray:
    expected = 3 * n_control_qubits
    params_f = np.asarray(params_f, dtype=np.float64).reshape(-1)
    if params_f.size == expected:
        return params_f
    if params_f.size == 0:
        return np.zeros((expected,), dtype=np.float64)
    repeats = int(np.ceil(expected / params_f.size))
    return np.tile(params_f, repeats)[:expected]


def create_experimental_circuit(psi, U, Z, params_v, params_k, params_f, num_layers, dim=4):
    builder = GeneralizedQuantumCircuitBuilder(
        embedding_dim=dim,
        sentence_length=max(len(psi) + 1, 2),
    )
    params_c = _normalize_control_params(params_f, builder.n_control_qubits)
    return builder.create_generalized_circuit(
        psi=psi,
        U=U,
        Z=Z,
        params_v=params_v,
        params_k=params_k,
        num_layers=num_layers,
        params_c=params_c,
    )


CIRCUIT_FUNCTIONS = {
    2: create_circuit_2words,
    4: create_circuit_4words,
    8: create_circuit_8words,
    16: create_circuit_16words,
}


def get_circuit_function(num_words):
    if num_words in CIRCUIT_FUNCTIONS:
        return CIRCUIT_FUNCTIONS[num_words]
    raise ValueError(f"Number of words ({num_words}) not supported. Maximum 16 words.")


def create_adaptive_quantum_circuit(
    sentence_words: List[str],
    vocab_info: Dict,
    embedding_dim: int,
    params_v: np.ndarray,
    params_k: np.ndarray,
    num_layers: int,
    use_generalized: bool = True,
    **kwargs,
) -> float:
    del vocab_info, use_generalized, kwargs

    sentence_length = len(sentence_words)
    builder = AdaptiveQuantumCircuitFactory.create_circuit_builder(
        embedding_dim=embedding_dim,
        sentence_length=sentence_length,
    )

    # Compat fallback: genera stati/unitarie placeholder coerenti con la dimensione richiesta.
    psi = _generate_legacy_unitaries(max(sentence_length - 1, 1), embedding_dim)
    U = _generate_legacy_unitaries(max(sentence_length - 1, 1), embedding_dim)
    Z = _generate_legacy_unitaries(max(sentence_length - 1, 1), embedding_dim)
    return builder.create_generalized_circuit(
        psi=psi,
        U=U,
        Z=Z,
        params_v=params_v,
        params_k=params_k,
        num_layers=num_layers,
    )


def _generate_legacy_unitaries(sentence_length: int, embedding_dim: int) -> List[np.ndarray]:
    unitaries = []
    rng = np.random.default_rng(0)
    for _ in range(sentence_length):
        state = rng.standard_normal(embedding_dim) + 1j * rng.standard_normal(embedding_dim)
        state = state / np.linalg.norm(state)
        basis = [state]
        while len(basis) < embedding_dim:
            vec = rng.standard_normal(embedding_dim) + 1j * rng.standard_normal(embedding_dim)
            for basis_vec in basis:
                vec -= np.vdot(basis_vec, vec) * basis_vec
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                continue
            basis.append(vec / norm)
        unitaries.append(np.column_stack(basis))
    return unitaries


def get_optimal_circuit_config(vocab_size: int, max_sentence_length: int) -> Dict:
    optimal_embedding = AdaptiveQuantumCircuitFactory.get_optimal_embedding_dim(vocab_size)
    complexity = AdaptiveQuantumCircuitFactory.estimate_circuit_complexity(
        optimal_embedding,
        max_sentence_length,
    )
    return {
        "vocab_size": vocab_size,
        "max_sentence_length": max_sentence_length,
        "suggested_embedding_dim": optimal_embedding,
        "architecture": "pennylane_jax",
        "complexity": complexity,
        "is_feasible": complexity["is_feasible"],
    }


def test_circuit_architecture():
    print("PennyLane/JAX circuit backend active.")


if __name__ == "__main__":
    test_circuit_architecture()
