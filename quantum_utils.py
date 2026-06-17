"""
Utility numeriche qiskit-free per il backend PennyLane/JAX.
"""

from __future__ import annotations

import gc
from typing import Dict

import numpy as np


def clear_memory():
    """Forza una pulizia conservativa della memoria del processo."""
    import os

    try:
        import psutil
    except Exception:
        psutil = None

    mem_before = -1.0
    if psutil is not None and hasattr(os, "getpid"):
        try:
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024**2
        except Exception:
            mem_before = -1.0

    for _ in range(3):
        gc.collect()

    try:
        import ctypes

        for libc_name in ("libc.so.6", "libc.dylib"):
            try:
                libc = ctypes.CDLL(libc_name)
                libc.malloc_trim(0)
                break
            except Exception:
                continue
    except Exception:
        pass

    if psutil is not None and mem_before >= 0:
        try:
            process = psutil.Process(os.getpid())
            mem_after = process.memory_info().rss / 1024**2
            freed_mb = mem_before - mem_after
            if freed_mb > 1.0:
                print(f"[MEMORY] Freed {freed_mb:.1f} MB ({mem_before:.1f} -> {mem_after:.1f} MB)")
        except Exception:
            pass


def check_memory_usage(threshold_gb: float = 1.5, rank: int = 0) -> bool:
    """Controlla l'RSS del processo e avvisa se supera la soglia."""
    try:
        import os
        import psutil

        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024**3
        if mem_gb > threshold_gb:
            print(
                f"[MEMORY WARNING] Rank {rank}: {mem_gb:.2f} GB used "
                f"(threshold: {threshold_gb:.1f} GB)"
            )
            return False
    except Exception:
        return True

    return True


def get_unitary_from_tk(psi) -> np.ndarray:
    """
    Costruisce una matrice unitaria con `psi` come prima colonna tramite Gram-Schmidt.
    """
    psi = np.asarray(psi, dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    dim = len(psi)
    base = [psi]
    rng = np.random.default_rng(0)

    while len(base) < dim:
        vec = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        for basis_vec in base:
            vec -= np.vdot(basis_vec, vec) * basis_vec

        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            continue

        base.append(vec / norm)

    return np.column_stack(base)


def get_param_resolver(num_qubits: int, num_layers: int) -> Dict[str, float]:
    """
    Restituisce un dizionario di angoli casuali, compatibile con l'API legacy ma senza Qiskit.
    """
    num_angles = 12 * num_qubits * num_layers
    angles = np.pi * (2.0 * np.random.rand(num_angles) - 1.0)
    return {f"theta_{idx}": angle for idx, angle in enumerate(angles)}


def get_params_shape(param_list: Dict[str, float], num_qubits: int, num_layers: int) -> np.ndarray:
    """Rimodella il dizionario legacy nel tensore atteso dagli ansatz storici."""
    param_values = np.asarray(list(param_list.values()), dtype=np.float64)
    x = param_values.reshape(num_layers, 2, num_qubits // 2, 12)
    return x.reshape(num_layers, 2, num_qubits // 2, 4, 3)


def get_params(num_qubits: int, num_layers: int) -> np.ndarray:
    """Genera direttamente il tensore parametri dell'ansatz legacy."""
    return get_params_shape(get_param_resolver(num_qubits, num_layers), num_qubits, num_layers)


def get_circuit_ux_dagger_from_tk(t_k) -> np.ndarray:
    """Compat layer legacy: restituisce direttamente `U^dagger` come matrice."""
    t_k = np.asarray(t_k, dtype=np.complex128)
    t_k = t_k / np.linalg.norm(t_k)
    return get_unitary_from_tk(t_k).conj().T


def build_controlled_unitary(U, controls, targets, label=None, activate_on=0) -> np.ndarray:
    """
    Costruisce la matrice del controlled-unitary attivata sullo stato `activate_on`.
    """
    del label, targets

    U = np.asarray(U, dtype=np.complex128)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be a square matrix.")

    n_controls = len(controls)
    if isinstance(activate_on, str):
        ctrl_state = int(activate_on, 2)
    else:
        ctrl_state = int(activate_on)

    control_dim = 2 ** n_controls
    target_dim = U.shape[0]
    full_dim = control_dim * target_dim
    controlled = np.eye(full_dim, dtype=np.complex128)
    start = ctrl_state * target_dim
    controlled[start:start + target_dim, start:start + target_dim] = U
    return controlled


def safe_controlled_unitary(U, control_indices, target_indices, label=None) -> np.ndarray:
    """Alias qiskit-free della costruzione controlled-unitary."""
    return build_controlled_unitary(
        U=U,
        controls=control_indices,
        targets=target_indices,
        label=label,
        activate_on=(2 ** len(control_indices)) - 1,
    )


def wrap_angles(theta):
    """Riporta gli angoli in `[-pi, pi]`."""
    theta = np.asarray(theta, dtype=np.float64)
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


def calculate_loss_from_statevector(statevector_or_probability, eps: float = 1e-12) -> float:
    """
    Calcola `-log(p_0)` dato:
    - una probabilità scalare,
    - uno statevector 1D,
    - una density matrix 2D.
    """
    value = np.asarray(statevector_or_probability)

    if value.ndim == 0:
        probability = float(np.real(value))
    elif value.ndim == 1:
        probability = float(np.abs(value[0]) ** 2)
    elif value.ndim == 2:
        probability = float(np.real(value[0, 0]))
    else:
        raise ValueError("Unsupported input shape for loss computation.")

    probability = float(np.clip(probability, eps, 1.0))
    return float(-np.log(probability))
