"""
Ansatz legacy qiskit-free basato sulla stessa struttura usata dal backend PennyLane.
"""

from __future__ import annotations

import numpy as np

from pennylane_jax_vqt import ansatz_matrix, count_blocks


def _legacy_params_to_blocks(params, num_qubits: int, num_layers: int) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    if params.ndim == 5:
        return params.reshape(-1, 12)
    if params.ndim == 4:
        return params.reshape(-1, 12)
    if params.ndim == 2 and params.shape[-1] == 12:
        return params

    blocks = count_blocks(num_qubits, num_layers)
    expected = blocks * 12
    if params.size != expected:
        raise ValueError(f"Expected {expected} ansatz parameters, got {params.size}.")
    return params.reshape(blocks, 12)


class AnsatzBuilder:
    """
    Compat wrapper: espone una API simile alla vecchia classe ma produce matrici unitarie.
    """

    def __init__(self, num_qubits, params, num_layers):
        self._num_qubits = int(num_qubits)
        self._num_layers = int(num_layers)
        self._params = _legacy_params_to_blocks(params, self._num_qubits, self._num_layers)

    @staticmethod
    def get_param_resolver(num_qubits, num_layers):
        num_angles = 12 * num_qubits * num_layers
        angles = np.pi * (2.0 * np.random.rand(num_angles) - 1.0)
        return {f"theta_{idx}": angle for idx, angle in enumerate(angles)}

    @staticmethod
    def get_params_shape(param_list, num_qubits, num_layers):
        param_values = np.asarray(list(param_list.values()), dtype=np.float64)
        x = param_values.reshape(num_layers, 2, num_qubits // 2, 12)
        return x.reshape(num_layers, 2, num_qubits // 2, 4, 3)

    @staticmethod
    def get_params(num_qubits, num_layers):
        return AnsatzBuilder.get_params_shape(
            AnsatzBuilder.get_param_resolver(num_qubits, num_layers),
            num_qubits,
            num_layers,
        )

    def get_ansatz(self):
        return self.matrix()

    def add_layer(self, params, shifted_params):
        del params, shifted_params
        raise NotImplementedError(
            "AnsatzBuilder qiskit-free is immutable; initialize it with the full parameter tensor."
        )

    def num_angles_required_for_layer(self):
        return 12 * self._num_qubits

    def matrix(self) -> np.ndarray:
        return np.asarray(ansatz_matrix(self._params, self._num_qubits, self._num_layers))

    def get_unitary(self, circuit_name=None):
        del circuit_name
        return self.matrix()


def generate_vqe_ansatz(qc, num_qubits, params):
    del qc
    builder = AnsatzBuilder(num_qubits=num_qubits, params=params, num_layers=1)
    return builder.matrix()
