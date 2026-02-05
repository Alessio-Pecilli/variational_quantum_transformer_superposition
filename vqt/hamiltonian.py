import numpy as np
from scipy.linalg import eigh

class HamiltonianEvolution:
    def __init__(self, H=None, dim=None):
        """
        Initialize with a Hamiltonian matrix.
        If H is None, a random Hamiltonian will be generated for the specified dimension.
        
        Args:
            H: Hamiltonian matrix (optional)
            dim: Dimension of the Hilbert space (optional, defaults to 2)
        """
        self.dim = dim if dim is not None else 2
        if H is None:
            if self.dim == 2:
                self.H = self._generate_random_pauli_hamiltonian()
            else:
                self.H = self._generate_random_hermitian_hamiltonian(self.dim)
        else:
            self.H = H
    
    def _pauli_matrices(self):
        """Return the three Pauli matrices"""
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        identity = np.array([[1, 0], [0, 1]], dtype=complex)
        return sigma_x, sigma_y, sigma_z, identity
    
    def _generate_random_pauli_hamiltonian(self):
        """Generate a random Hamiltonian as linear combination of Pauli matrices"""
        sigma_x, sigma_y, sigma_z, identity = self._pauli_matrices()
        
        # Random coefficients
        a0 = np.random.uniform(-1, 1)
        a1 = np.random.uniform(-1, 1)
        a2 = np.random.uniform(-1, 1)
        a3 = np.random.uniform(-1, 1)
        
        H = a0 * identity + a1 * sigma_x + a2 * sigma_y + a3 * sigma_z
        return H
    
    def _generate_random_hermitian_hamiltonian(self, dim):
        """Generate a random Hermitian Hamiltonian of specified dimension"""
        # Create random matrix
        A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        # Make it Hermitian
        H = (A + A.conj().T) / 2
        return H
    
    def evolve(self, psi0, times):
        """
        Compute time evolution and return array of states
        
        Args:
            psi0: Initial state
            times: Array of time points
            
        Returns:
            Array of evolved states
        """
        # Diagonalize Hamiltonian
        eigenvalues, eigenvectors = eigh(self.H)
        
        # Calculate coefficients ak = <Ek|psi0>
        ak = np.conj(eigenvectors.T) @ psi0
        
        # Time evolution
        psi_t = np.zeros((len(times), len(psi0)), dtype=complex)
        
        for i, t in enumerate(times):
            psi_t[i] = np.sum(ak[:, np.newaxis] * 
                             np.exp(-1j * eigenvalues[:, np.newaxis] * t) * 
                             eigenvectors, axis=0)
        
        return psi_t
    
    def generate_sequential_states(self, num_states, max_time=10.0, state_dim=4):
        """
        Generate sequential quantum states using Hamiltonian evolution.
        
        Args:
            num_states (int): Number of sequential states to generate
            max_time (float): Maximum evolution time
            state_dim (int): Dimension of the state space
            
        Returns:
            numpy.ndarray: Array of sequential states, shape (num_states, state_dim)
        """
        # Ensure Hamiltonian has correct dimension (generate only once)
        if self.H.shape[0] != state_dim:
            print(f"Generating new {state_dim}x{state_dim} Hamiltonian...")
            # Use fixed seed for reproducibility
            np.random.seed(42)  
            self.H = self._generate_random_hermitian_hamiltonian(state_dim)
            np.random.seed()  # Reset seed
        
        # Generate initial state (fixed for consistency)
        np.random.seed(123)
        psi0 = np.random.randn(state_dim) + 1j * np.random.randn(state_dim)
        np.random.seed()  # Reset seed
        psi0 = psi0 / np.linalg.norm(psi0)
        
        # Create time points for sequential evolution
        times = np.linspace(0, max_time, num_states)
        
        # Evolve the state
        evolved_states = self.evolve(psi0, times)
        
        # Normalize each state
        for i in range(len(evolved_states)):
            evolved_states[i] = evolved_states[i] / np.linalg.norm(evolved_states[i])
        
        return evolved_states

# Example usage
if __name__ == "__main__":
    # Instantiate the class
    evolution = HamiltonianEvolution()
    
    # Generate sequential states (e.g., for 5 time steps)
    num_states = 5
    sequential_states = evolution.generate_sequential_states(num_states)
    
    print(f"Generated {num_states} sequential states")
    print(f"Shape of states array: {sequential_states.shape}")
    print(f"Each state is normalized: {[np.abs(np.linalg.norm(state) - 1) < 1e-10 for state in sequential_states]}")
    
    # Example of individual evolution
    psi0 = np.array([1, 0], dtype=complex)  # |0⟩ state
    times = np.linspace(0, 5, 10)
    evolved = evolution.evolve(psi0, times)
    print(f"\nEvolution from |0⟩ state over {len(times)} time points")
    print(f"Final state: {evolved[-1]}")
    print(f"Probability amplitudes: |ψ(t)|² = {np.abs(evolved[-1])**2}")
