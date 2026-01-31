"""
Quantum State Projection Layer (P matrix).

Implements a trainable linear projection layer that maps quantum states from
a high-dimensional Hilbert space (2^D) to a lower-dimensional space (2^d) 
compatible with the QSA circuit.

Pipeline (quantum_states=True):
    Stati(2^D) -> P -> Embedding(2^d) -> QSA

Example:
    D = 10 (source qubits) -> 2^10 = 1024 dimensional states
    d = 4 (target qubits)  -> 2^4 = 16 dimensional embedding
    P matrix shape: (16, 1024) - trainable

Formula Loss: L = -2 ln(1/T * sum(sqrt(p(y_t|x))))
Formula PPL: exp(L)
"""

import numpy as np
from typing import Optional, Tuple


class QuantumStateProjector:
    """
    Projects high-dimensional quantum states to lower-dimensional embeddings.
    
    The projection matrix P is trainable and maps:
        |ψ⟩ ∈ C^{2^D} -> embedding ∈ C^{2^d}
    
    where:
        - D = source_qubits (large, e.g., 10)
        - d = target_qubits (small, e.g., 4)
    
    The embedding is then treated like classical word embeddings in the QSA.
    """
    
    def __init__(self, source_dim: int, target_dim: int, seed: Optional[int] = None):
        """
        Initialize the projection layer.
        
        Args:
            source_dim: Dimension of input states (2^D, e.g., 1024 for D=10)
            target_dim: Dimension of output embedding (2^d, e.g., 16 for d=4)
            seed: Random seed for reproducibility
        """
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.n_params = target_dim * source_dim  # P is (target_dim, source_dim)
        
        # Initialize P matrix (will be set from optimization params)
        self._P_matrix = None
        
        # Store initialization seed
        self._seed = seed
        
        print(f"[QuantumStateProjector] Initialized:")
        print(f"    Source dim: {source_dim} (2^{int(np.log2(source_dim))} qubits)")
        print(f"    Target dim: {target_dim} (2^{int(np.log2(target_dim))} qubits)")
        print(f"    P matrix shape: ({target_dim}, {source_dim})")
        print(f"    Trainable parameters: {self.n_params}")
    
    @property
    def P(self) -> np.ndarray:
        """Get the projection matrix P."""
        if self._P_matrix is None:
            raise ValueError("P matrix not set. Call set_params() first.")
        return self._P_matrix
    
    def get_shape(self) -> Tuple[int, int]:
        """Return shape of P matrix (target_dim, source_dim)."""
        return (self.target_dim, self.source_dim)
    
    def get_n_params(self) -> int:
        """Return number of trainable parameters."""
        return self.n_params
    
    def set_params(self, params: np.ndarray):
        """
        Set the P matrix from a flat parameter vector.
        
        Args:
            params: Flat array of shape (target_dim * source_dim,)
                   Will be reshaped to (target_dim, source_dim)
        """
        if params.size != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {params.size}"
            )
        
        # Reshape to matrix and store (REAL values for now, can extend to complex)
        self._P_matrix = params.reshape(self.target_dim, self.source_dim)
    
    def project(self, quantum_state: np.ndarray) -> np.ndarray:
        """
        Project a single quantum state to lower-dimensional embedding.
        
        Args:
            quantum_state: Complex vector of shape (source_dim,)
        
        Returns:
            embedding: Complex vector of shape (target_dim,)
        """
        if self._P_matrix is None:
            raise ValueError("P matrix not set. Call set_params() first.")
        
        if quantum_state.shape[0] != self.source_dim:
            raise ValueError(
                f"Expected state of dim {self.source_dim}, got {quantum_state.shape[0]}"
            )
        
        # Project: embedding = P @ state (matrix-vector multiplication)
        embedding = self._P_matrix @ quantum_state
        
        # Normalize to unit norm (preserve quantum state interpretation)
        norm = np.linalg.norm(embedding)
        if norm > 1e-10:
            embedding = embedding / norm
        
        return embedding
    
    def project_sequence(self, quantum_states: np.ndarray) -> np.ndarray:
        """
        Project a sequence of quantum states.
        
        Args:
            quantum_states: Array of shape (seq_len, source_dim)
        
        Returns:
            embeddings: Array of shape (seq_len, target_dim)
        """
        if self._P_matrix is None:
            raise ValueError("P matrix not set. Call set_params() first.")
        
        seq_len = quantum_states.shape[0]
        embeddings = np.zeros((seq_len, self.target_dim), dtype=complex)
        
        for i in range(seq_len):
            embeddings[i] = self.project(quantum_states[i])
        
        return embeddings
    
    def get_initial_params(self, scale: float = 0.1) -> np.ndarray:
        """
        Generate initial random parameters for P matrix.
        
        Uses Xavier-like initialization scaled for quantum states.
        
        Args:
            scale: Scaling factor for initialization
        
        Returns:
            params: Flat array of initial parameters
        """
        rng = np.random.default_rng(self._seed)
        
        # Xavier-like initialization
        std = scale * np.sqrt(2.0 / (self.source_dim + self.target_dim))
        
        params = rng.normal(0, std, self.n_params)
        
        return params


def create_projector_from_config(cfg: dict) -> Optional[QuantumStateProjector]:
    """
    Create a QuantumStateProjector from configuration dict.
    
    Only creates projector if:
        - use_quantum_states is True
        - use_projection is True
    
    Args:
        cfg: Configuration dict (QUANTUM_STATES_CONFIG)
    
    Returns:
        QuantumStateProjector or None if not needed
    """
    use_quantum_states = cfg.get('use_quantum_states', False)
    use_projection = cfg.get('use_projection', True)
    
    if not use_quantum_states or not use_projection:
        return None
    
    source_qubits = cfg.get('source_qubits', 10)  # D = large qubits (e.g., 10)
    target_qubits = cfg.get('target_qubits', 4)   # d = QSA qubits (e.g., 4)
    
    source_dim = 2 ** source_qubits  # 1024 for D=10
    target_dim = 2 ** target_qubits  # 16 for d=4
    
    seed = cfg.get('seed', None)
    
    projector = QuantumStateProjector(source_dim, target_dim, seed=seed)
    
    return projector


def get_projection_shape(cfg: dict) -> Optional[Tuple[int, int]]:
    """
    Get the shape of the projection matrix P if needed.
    
    Args:
        cfg: Configuration dict (QUANTUM_STATES_CONFIG)
    
    Returns:
        (target_dim, source_dim) or None if projection not needed
    """
    use_quantum_states = cfg.get('use_quantum_states', False)
    use_projection = cfg.get('use_projection', True)
    
    if not use_quantum_states or not use_projection:
        return None
    
    source_qubits = cfg.get('source_qubits', 10)
    target_qubits = cfg.get('target_qubits', 4)
    
    source_dim = 2 ** source_qubits
    target_dim = 2 ** target_qubits
    
    return (target_dim, source_dim)


def get_projection_param_count(cfg: dict) -> int:
    """
    Get the number of trainable parameters for P matrix.
    
    Args:
        cfg: Configuration dict (QUANTUM_STATES_CONFIG)
    
    Returns:
        Number of parameters (0 if projection not needed)
    """
    shape = get_projection_shape(cfg)
    if shape is None:
        return 0
    return shape[0] * shape[1]


# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("QuantumStateProjector Test")
    print("=" * 70)
    
    # Simulate D=10 source qubits, d=4 target qubits
    source_qubits = 10
    target_qubits = 4
    source_dim = 2 ** source_qubits  # 1024
    target_dim = 2 ** target_qubits  # 16
    
    print(f"\nConfiguration:")
    print(f"  Source qubits D = {source_qubits} -> dim = {source_dim}")
    print(f"  Target qubits d = {target_qubits} -> dim = {target_dim}")
    
    # Create projector
    projector = QuantumStateProjector(source_dim, target_dim, seed=42)
    
    # Initialize with random params
    initial_params = projector.get_initial_params()
    print(f"\nInitial params shape: {initial_params.shape}")
    print(f"Initial params range: [{initial_params.min():.4f}, {initial_params.max():.4f}]")
    
    # Set params
    projector.set_params(initial_params)
    
    # Create a random quantum state (normalized)
    rng = np.random.default_rng(42)
    psi = rng.standard_normal(source_dim) + 1j * rng.standard_normal(source_dim)
    psi = psi / np.linalg.norm(psi)
    
    print(f"\nInput state |ψ⟩:")
    print(f"  Shape: {psi.shape}")
    print(f"  Norm: {np.linalg.norm(psi):.6f}")
    
    # Project
    embedding = projector.project(psi)
    
    print(f"\nOutput embedding:")
    print(f"  Shape: {embedding.shape}")
    print(f"  Norm: {np.linalg.norm(embedding):.6f}")
    print(f"  First 5 components: {embedding[:5]}")
    
    # Test sequence projection
    seq_len = 5  # 5 "words" in the sentence
    sequence = np.zeros((seq_len, source_dim), dtype=complex)
    for i in range(seq_len):
        state = rng.standard_normal(source_dim) + 1j * rng.standard_normal(source_dim)
        sequence[i] = state / np.linalg.norm(state)
    
    print(f"\nSequence of {seq_len} states:")
    embeddings = projector.project_sequence(sequence)
    print(f"  Output shape: {embeddings.shape}")
    
    for i in range(seq_len):
        print(f"  State {i}: norm={np.linalg.norm(sequence[i]):.4f} -> embedding norm={np.linalg.norm(embeddings[i]):.4f}")
    
    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
