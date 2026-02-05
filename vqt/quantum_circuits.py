"""
Quantum circuit creation functions for different word configurations.
Updated with Generalized Architecture Support.
"""
import numpy as np
import datetime
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector
from typing import List, Optional, Dict

from .layer import AnsatzBuilder
from .quantum_utils import build_controlled_unitary, calculate_loss_from_statevector
from . import visualization

# Import new generalized architecture
try:
    from .generalized_quantum_circuits import (
        AdaptiveQuantumCircuitFactory, 
        GeneralizedQuantumCircuitBuilder
    )
    GENERALIZED_AVAILABLE = True
    print("✅ Generalized Quantum Circuits available")
except ImportError as e:
    GENERALIZED_AVAILABLE = False
    print(f"⚠️ Generalized Quantum Circuits not available: {e}")
    print("   Falling back to original implementation")


def create_circuit_2words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    """
    Create quantum circuit for 2 words (1 transition).
    
    Args:
        psi (list): List of unitaries for initial states
        U (list): List of unitaries for next word prediction
        Z (list): List of unitaries for current word
        params_v (array): Parameters for V ansatz
        params_k (array): Parameters for K ansatz
        num_layers (int): Number of ansatz layers
        dim (int): Dimension parameter
        
    Returns:
        float: Loss value
    """
    n_qubits = 4
    qtarget = n_qubits
    n_qubits += 1  # 1 control qubit for 2 words
    a = 2
    
    ansatz_v = AnsatzBuilder(a, params_v, num_layers)
    ansatz_k = AnsatzBuilder(a, params_k, num_layers)
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize control qubits in superposition
    for i in range(qtarget, n_qubits):
        qc.h(i)

    control_indices = [4]
    target_indices = list(range(0, 4))

    # Apply controlled unitaries for different states
    qc.compose(
        build_controlled_unitary(psi[0], control_indices, target_indices, "Ψ_0", activate_on='0'),
        qubits=control_indices + target_indices,
        inplace=True
    )

    qc.compose(
        build_controlled_unitary(psi[1], control_indices, target_indices, "ψ_1", activate_on='1'),
        qubits=control_indices + target_indices,
        inplace=True
    )

    # Apply variational ansatz
    qc.compose(ansatz_v.get_unitary("V"), list(range(0, 2)), inplace=True)
    qc.compose(ansatz_k.get_unitary("W"), list(range(2, 4)), inplace=True)

    # Apply controlled prediction unitaries
    qc.compose(
        build_controlled_unitary(U[0], control_indices, list(range(0, 2)), "next_0", activate_on='0'),
        qubits=control_indices + list(range(0, 2)),
        inplace=True
    )
    qc.compose(
        build_controlled_unitary(Z[0], control_indices, list(range(2, 4)), "current_0", activate_on='0'),
        qubits=control_indices + list(range(2, 4)),
        inplace=True
    )

    qc.compose(
        build_controlled_unitary(U[1], control_indices, list(range(0, 2)), "next_1", activate_on='1'),
        qubits=control_indices + list(range(0, 2)),
        inplace=True
    )
    qc.compose(
        build_controlled_unitary(Z[1], control_indices, list(range(2, 4)), "current_1", activate_on='1'),
        qubits=control_indices + list(range(2, 4)),
        inplace=True
    )

    # Final Hadamard on control qubits
    for i in range(qtarget, n_qubits):
        qc.h(i)
    
    loss = calculate_loss_from_statevector(qc)
    print(f"Loss: {loss:.6f} at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    return loss


def create_circuit_4words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    """
    Create quantum circuit for 4 words.
    
    Args:
        psi (list): List of unitaries for initial states
        U (list): List of unitaries for next word prediction
        Z (list): List of unitaries for current word
        params_v (array): Parameters for V ansatz
        params_k (array): Parameters for K ansatz
        num_layers (int): Number of ansatz layers
        dim (int): Dimension parameter
        
    Returns:
        float: Loss value
    """
    n_qubits = 4
    qtarget = n_qubits
    n_qubits += 2  # 2 control qubits for 4 words
    a = 2

    ansatz_v = AnsatzBuilder(a, params_v, num_layers)
    ansatz_k = AnsatzBuilder(a, params_k, num_layers)
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize control qubits in superposition
    for i in range(qtarget, n_qubits):
        qc.h(i)

    control_indices = [4, 5]
    target_indices = list(range(0, 4))

    # Apply controlled unitaries for 4 different combinations
    states = ['00', '10', '01', '11']
    for i, state in enumerate(states):
        qc.compose(
            build_controlled_unitary(psi[i], control_indices, target_indices, f"ψ_{state}", activate_on=state),
            qubits=control_indices + target_indices,
            inplace=True
        )

    # Apply variational ansatz
    qc.compose(ansatz_v.get_unitary("V"), list(range(0, 2)), inplace=True)
    qc.compose(ansatz_k.get_unitary("W"), list(range(2, 4)), inplace=True)

    # Apply controlled prediction unitaries
    for i, state in enumerate(states):
        qc.compose(
            build_controlled_unitary(U[i], control_indices, list(range(0, 2)), f"next_{state}", activate_on=state),
            qubits=control_indices + list(range(0, 2)),
            inplace=True
        )
        qc.compose(
            build_controlled_unitary(Z[i], control_indices, list(range(2, 4)), f"current_{state}", activate_on=state),
            qubits=control_indices + list(range(2, 4)),
            inplace=True
        )

    # Final Hadamard on control qubits
    for i in range(qtarget, n_qubits):
        qc.h(i)
    
    loss = calculate_loss_from_statevector(qc)
    print(f"Loss: {loss:.6f} at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    return loss


def create_circuit_8words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    """
    Create quantum circuit for 8 words.
    
    Args:
        psi (list): List of unitaries for initial states
        U (list): List of unitaries for next word prediction
        Z (list): List of unitaries for current word
        params_v (array): Parameters for V ansatz
        params_k (array): Parameters for K ansatz
        num_layers (int): Number of ansatz layers
        dim (int): Dimension parameter
        
    Returns:
        float: Loss value
    """
    n_qubits = 4
    qtarget = n_qubits
    n_qubits += 3  # 3 control qubits for 8 words
    a = 2

    ansatz_v = AnsatzBuilder(a, params_v, num_layers)
    ansatz_k = AnsatzBuilder(a, params_k, num_layers)
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize control qubits in superposition
    for i in range(qtarget, n_qubits):
        qc.h(i)

    control_indices = [4, 5, 6]
    target_indices = list(range(0, 4))

    # Apply controlled unitaries for 8 different combinations
    states = ['000', '001', '010', '011', '100', '101', '110', '111']
    for i, state in enumerate(states):
        qc.compose(
            build_controlled_unitary(psi[i], control_indices, target_indices, f"ψ_{state}", activate_on=state),
            qubits=control_indices + target_indices,
            inplace=True
        )

    # Apply variational ansatz
    qc.compose(ansatz_v.get_unitary("V"), list(range(0, 2)), inplace=True)
    qc.compose(ansatz_k.get_unitary("W"), list(range(2, 4)), inplace=True)

    # Apply controlled prediction unitaries
    for i, state in enumerate(states):
        qc.compose(
            build_controlled_unitary(U[i], control_indices, list(range(0, 2)), f"next_{state}", activate_on=state),
            qubits=control_indices + list(range(0, 2)),
            inplace=True
        )
        qc.compose(
            build_controlled_unitary(Z[i], control_indices, list(range(2, 4)), f"current_{state}", activate_on=state),
            qubits=control_indices + list(range(2, 4)),
            inplace=True
        )

    # Final Hadamard on control qubits
    for i in range(qtarget, n_qubits):
        qc.h(i)

    loss = calculate_loss_from_statevector(qc)
    print(f"Loss: {loss:.6f} at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    return loss


def create_circuit_16words(psi, U, Z, params_v, params_k, num_layers, dim=4):
    """
    Create quantum circuit for 16 words.
    
    Args:
        psi (list): List of unitaries for initial states
        U (list): List of unitaries for next word prediction
        Z (list): List of unitaries for current word
        params_v (array): Parameters for V ansatz
        params_k (array): Parameters for K ansatz
        num_layers (int): Number of ansatz layers
        dim (int): Dimension parameter
        
    Returns:
        float: Loss value
    """
    n_qubits = 4
    qtarget = n_qubits
    n_qubits += 4  # 4 control qubits for 16 words
    a = 2

    ansatz_v = AnsatzBuilder(a, params_v, num_layers)
    ansatz_k = AnsatzBuilder(a, params_k, num_layers)
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize control qubits in superposition
    for i in range(qtarget, n_qubits):
        qc.h(i)

    control_indices = [4, 5, 6, 7]
    target_indices = list(range(0, 4))

    # Generate all 16 binary combinations
    activate_states = [f'{i:04b}' for i in range(16)]

    # Apply controlled unitaries for 16 different combinations
    for i, state in enumerate(activate_states):
        qc.compose(
            build_controlled_unitary(psi[i], control_indices, target_indices, f"ψ_{state}", activate_on=state),
            qubits=control_indices + target_indices,
            inplace=True
        )

    # Apply variational ansatz
    qc.compose(ansatz_v.get_unitary("V"), list(range(0, 2)), inplace=True)
    qc.compose(ansatz_k.get_unitary("W"), list(range(2, 4)), inplace=True)

    # Apply controlled prediction unitaries
    for i, state in enumerate(activate_states):
        qc.compose(
            build_controlled_unitary(U[i], control_indices, list(range(0, 2)), f"next_{state}", activate_on=state),
            qubits=control_indices + list(range(0, 2)),
            inplace=True
        )
        qc.compose(
            build_controlled_unitary(Z[i], control_indices, list(range(2, 4)), f"current_{state}", activate_on=state),
            qubits=control_indices + list(range(2, 4)),
            inplace=True
        )

    # Final Hadamard on control qubits
    for i in range(qtarget, n_qubits):
        qc.h(i)

    loss = calculate_loss_from_statevector(qc)
    print(f"Loss: {loss:.6f} at {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    return loss


def create_experimental_circuit(psi, U, Z, params_v, params_k, params_f, num_layers, dim=4):
    """
    Create experimental quantum circuit with additional phase parameters.
    
    Args:
        psi (list): List of unitaries for initial states
        U (list): List of unitaries for next word prediction
        Z (list): List of unitaries for current word
        params_v (array): Parameters for V ansatz
        params_k (array): Parameters for K ansatz
        params_f (array): Additional phase parameters
        num_layers (int): Number of ansatz layers
        dim (int): Dimension parameter
        
    Returns:
        float: Loss value
    """
    n_qubits = 4
    qtarget = n_qubits
    n_qubits += 1
    a = 2
    
    ansatz_v = AnsatzBuilder(a, params_v, num_layers)
    ansatz_k = AnsatzBuilder(a, params_k, num_layers)

    # Create diagonal phase matrix
    print("paramsF:", np.diag(params_f))
    diag_matrix = np.diag(np.exp(1j * np.array(params_f)))
    ansatz_f = UnitaryGate(diag_matrix, label="φ")
    
    qc = QuantumCircuit(n_qubits, n_qubits)
    
    # Initialize control qubits in superposition
    for i in range(qtarget, n_qubits):
        qc.h(i)

    control_indices = [4]
    target_indices = list(range(0, 4))

    # Apply controlled unitaries
    qc.compose(
        build_controlled_unitary(psi[0], control_indices, target_indices, "ψ_0", activate_on='0'),
        qubits=control_indices + target_indices,
        inplace=True
    )

    qc.compose(
        build_controlled_unitary(psi[1], control_indices, target_indices, "ψ_1", activate_on='1'),
        qubits=control_indices + target_indices,
        inplace=True
    )

    # Apply variational ansatz
    qc.compose(ansatz_v.get_unitary("V"), list(range(0, 2)), inplace=True)
    qc.compose(ansatz_k.get_unitary("W"), list(range(2, 4)), inplace=True)

    # Apply controlled prediction unitaries
    qc.compose(
        build_controlled_unitary(U[0], control_indices, list(range(0, 2)), "next_0", activate_on='0'),
        qubits=control_indices + list(range(0, 2)),
        inplace=True
    )
    qc.compose(
        build_controlled_unitary(Z[0], control_indices, list(range(2, 4)), "current_0", activate_on='0'),
        qubits=control_indices + list(range(2, 4)),
        inplace=True
    )

    qc.compose(
        build_controlled_unitary(U[1], control_indices, list(range(0, 2)), "next_1", activate_on='1'),
        qubits=control_indices + list(range(0, 2)),
        inplace=True
    )
    qc.compose(
        build_controlled_unitary(Z[1], control_indices, list(range(2, 4)), "current_1", activate_on='1'),
        qubits=control_indices + list(range(2, 4)),
        inplace=True
    )

    # Apply experimental phase gate
    qc.compose(ansatz_f, list(range(qtarget, n_qubits)), inplace=True)
    
    # Final Hadamard on control qubits
    for i in range(qtarget, n_qubits):
        qc.h(i)

    # Calculate probabilities for analysis
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="matrix_product_state")
    result = sim.run(qc).result()
    state = result.get_statevector(qc)
    probs = state.probabilities_dict()
    
    # Calculate ancilla probabilities
    prob_ancilla0 = sum(p for state, p in probs.items() if state[-1] == '0')
    prob_ancilla1 = sum(p for state, p in probs.items() if state[-1] == '1')
    
    loss = calculate_loss_from_statevector(qc)
    print(f"Loss: {loss:.6f}")
    
    return loss


# Circuit function mapping (Legacy)
CIRCUIT_FUNCTIONS = {
    2: create_circuit_2words,
    4: create_circuit_4words,
    8: create_circuit_8words,
    16: create_circuit_16words
}


def get_circuit_function(num_words):
    """
    Get the appropriate circuit function based on number of words.
    
    Args:
        num_words (int): Number of words to handle
        
    Returns:
        function: Circuit creation function
        
    Raises:
        ValueError: If number of words is not supported
    """
    if num_words in CIRCUIT_FUNCTIONS:
        return CIRCUIT_FUNCTIONS[num_words]
    else:
        raise ValueError(f"Number of words ({num_words}) not supported. Maximum 16 words.")


def create_adaptive_quantum_circuit(sentence_words: List[str], vocab_info: Dict,
                                  embedding_dim: int, params_v: np.ndarray, 
                                  params_k: np.ndarray, num_layers: int,
                                  use_generalized: bool = True,
                                  **kwargs) -> float:
    """
    Crea circuito quantico adattivo che sceglie automaticamente tra implementazioni.
    
    Args:
        sentence_words: Lista parole della frase
        vocab_info: Informazioni vocabolario
        embedding_dim: Dimensione embedding (4, 16, 64, 256, etc.)
        params_v: Parametri ansatz V
        params_k: Parametri ansatz K 
        num_layers: Numero layer ansatz
        use_generalized: Se usare architettura generalizzata (default: True)
        **kwargs: Argomenti aggiuntivi
        
    Returns:
        Loss calcolata dal circuito
    """
    
    sentence_length = len(sentence_words)
    
    # Usa architettura generalizzata se disponibile e richiesta
    if GENERALIZED_AVAILABLE and use_generalized:
        try:
            print(f"🚀 Using Generalized Architecture: {embedding_dim}D embedding, {sentence_length} words")
            
            """return create_quantum_circuit_for_sentence(
                sentence_words=sentence_words,
                vocab_info=vocab_info,
                embedding_dim=embedding_dim,
                params_v=params_v,
                params_k=params_k,
                num_layers=num_layers,
                **kwargs
            )"""
            
        except Exception as e:
            print(f"⚠️ Generalized architecture failed: {e}")
            print("🔄 Falling back to legacy implementation")
    
    # Fallback a implementazione legacy
    return _create_legacy_circuit(sentence_words, vocab_info, embedding_dim, 
                                 params_v, params_k, num_layers, **kwargs)


def _create_legacy_circuit(sentence_words: List[str], vocab_info: Dict,
                          embedding_dim: int, params_v: np.ndarray, 
                          params_k: np.ndarray, num_layers: int,
                          **kwargs) -> float:
    """
    Implementazione legacy per compatibilità con codice esistente.
    Supporta solo embedding_dim=4 e sentence_length limitati.
    """
    
    sentence_length = len(sentence_words)
    
    # Verifica compatibilità legacy
    if embedding_dim != 4:
        raise ValueError(f"Legacy implementation supports only embedding_dim=4, got {embedding_dim}")
    
    if sentence_length not in CIRCUIT_FUNCTIONS:
        raise ValueError(f"Legacy implementation supports only {list(CIRCUIT_FUNCTIONS.keys())} words, got {sentence_length}")
    
    print(f"⚙️ Using Legacy Architecture: {embedding_dim}D embedding, {sentence_length} words")
    
    # Genera unitarie default per legacy
    psi = _generate_legacy_unitaries(sentence_length)
    U = _generate_legacy_unitaries(sentence_length)
    Z = _generate_legacy_unitaries(sentence_length)
    
    # Chiama funzione legacy appropriata
    circuit_function = CIRCUIT_FUNCTIONS[sentence_length]
    
    return circuit_function(
        psi=psi, U=U, Z=Z,
        params_v=params_v,
        params_k=params_k,
        num_layers=num_layers,
        dim=embedding_dim
    )


def _generate_legacy_unitaries(sentence_length: int) -> List[np.ndarray]:
    """Genera unitarie 4x4 per implementazione legacy."""
    
    unitaries = []
    
    for i in range(sentence_length):
        # Genera unitaria 4x4 random
        angles = np.random.random(4) * 2 * np.pi
        
        # Costruisci unitaria 2x2
        u2x2 = np.array([
            [np.cos(angles[0]) * np.exp(1j * angles[2]), 
             np.sin(angles[0]) * np.exp(1j * (angles[1] + angles[2]))],
            [-np.sin(angles[0]) * np.exp(1j * (angles[1] - angles[2])), 
             np.cos(angles[0]) * np.exp(-1j * angles[2])]
        ])
        
        # Estendi a 4x4 con prodotto tensoriale
        I = np.eye(2)
        if i % 2 == 0:
            unitary_4x4 = np.kron(u2x2, I)
        else:
            unitary_4x4 = np.kron(I, u2x2)
        
        unitaries.append(unitary_4x4)
    
    return unitaries


def get_optimal_circuit_config(vocab_size: int, max_sentence_length: int) -> Dict:
    """
    Suggerisce configurazione ottimale per vocabolario e lunghezza frase.
    
    Args:
        vocab_size: Dimensione vocabolario
        max_sentence_length: Lunghezza massima frasi
        
    Returns:
        Dict con configurazione suggerita
    """
    
    if not GENERALIZED_AVAILABLE:
        return {
            "embedding_dim": 4,
            "max_supported_length": 16,
            "architecture": "legacy",
            "recommendation": "Install generalized architecture for better scaling"
        }
    
    # Usa factory per suggerimenti
    optimal_embedding = AdaptiveQuantumCircuitFactory.get_optimal_embedding_dim(vocab_size)
    complexity = AdaptiveQuantumCircuitFactory.estimate_circuit_complexity(optimal_embedding, max_sentence_length)
    
    config = {
        "vocab_size": vocab_size,
        "max_sentence_length": max_sentence_length,
        "suggested_embedding_dim": optimal_embedding,
        "architecture": "generalized" if complexity["is_feasible"] else "legacy",
        "complexity": complexity,
        "is_feasible": complexity["is_feasible"]
    }
    
    if not complexity["is_feasible"]:
        # Suggerisci configurazione più piccola
        for test_embedding in [64, 16, 4]:
            test_complexity = AdaptiveQuantumCircuitFactory.estimate_circuit_complexity(test_embedding, max_sentence_length)
            if test_complexity["is_feasible"]:
                config["suggested_embedding_dim"] = test_embedding
                config["complexity"] = test_complexity
                config["is_feasible"] = True
                config["note"] = f"Reduced from {optimal_embedding} for feasibility"
                break
    
    return config


def test_circuit_architecture():
    """Test delle diverse architetture disponibili."""
    
    print("🧪 TESTING QUANTUM CIRCUIT ARCHITECTURES")
    print("="*50)
    
    # Test configurations
    test_cases = [
        # (sentence_length, embedding_dim, expected_architecture)
        (3, 4, "legacy"),
        (5, 16, "generalized"),
        (9, 64, "generalized"), 
        (17, 256, "generalized")
    ]
    
    for sentence_length, embedding_dim, expected_arch in test_cases:
        print(f"\n🔬 Test: {sentence_length} words, {embedding_dim}D embedding")
        
        try:
            # Genera dati di test
            sentence_words = [f"word{i}" for i in range(sentence_length)]
            vocab_info = {"word_to_idx": {}, "idx_to_word": {}}
            
            # Parametri ansatz appropriati
            if embedding_dim == 4:
                params_v = np.random.random(6) * 0.1  # 2x2 ansatz
                params_k = np.random.random(6) * 0.1
            else:
                n_qubits = int(np.log2(embedding_dim))
                ansatz_dim = max(1, n_qubits // 2)
                params_v = np.random.random(ansatz_dim * 3) * 0.1
                params_k = np.random.random(ansatz_dim * 3) * 0.1
            
            # Test adattivo
            loss = create_adaptive_quantum_circuit(
                sentence_words=sentence_words,
                vocab_info=vocab_info,
                embedding_dim=embedding_dim,
                params_v=params_v,
                params_k=params_k,
                num_layers=2
            )
            
            print(f"   ✅ Loss: {loss:.6f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Test configurazione ottimale
    print(f"\n📊 OPTIMAL CONFIGURATIONS:")
    
    test_vocabs = [1000, 5000, 50000]
    test_lengths = [5, 17, 32]
    
    for vocab_size in test_vocabs:
        for max_length in test_lengths:
            config = get_optimal_circuit_config(vocab_size, max_length)
            print(f"   Vocab {vocab_size:,}, MaxLen {max_length}: "
                  f"{config['suggested_embedding_dim']}D embedding "
                  f"({config['architecture']} architecture)")


if __name__ == "__main__":
    print("main quantum_circuits.py")
