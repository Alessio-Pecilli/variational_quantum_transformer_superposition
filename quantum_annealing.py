"""
Quantum Annealing Hamiltonian for Variational Quantum Transformer.

Implementa l'Hamiltoniana dipendente dal tempo per Quantum Annealing:
    H(t) = -B(t) * Σᵢ σᵢˣ + Σ⟨i,j⟩ Jᵢⱼ σᵢᶻ σⱼᶻ

Dove:
    - B(t): funzione di scheduling (da 1 a 0)
    - σˣ: operatore di Pauli X (driver/tunneling)
    - σᶻ: operatore di Pauli Z (problema/interazione)
    - Jᵢⱼ: coefficienti di accoppiamento tra parole i e j
"""

import numpy as np
from scipy.linalg import expm
from typing import Callable, Optional, Union, List


class PauliOperators:
    """Operatori di Pauli e utility per sistemi multi-qubit."""
    
    # Matrici di Pauli singolo qubit
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    @staticmethod
    def tensor_product(*matrices):
        """Prodotto tensoriale di più matrici."""
        result = matrices[0]
        for m in matrices[1:]:
            result = np.kron(result, m)
        return result
    
    @staticmethod
    def single_qubit_op(op: np.ndarray, qubit_idx: int, num_qubits: int) -> np.ndarray:
        """
        Costruisce operatore single-qubit sul sistema completo.
        
        Args:
            op: Operatore 2x2 (es. X, Y, Z)
            qubit_idx: Indice del qubit (0-indexed)
            num_qubits: Numero totale di qubit
            
        Returns:
            Operatore 2^N x 2^N
        """
        ops = [PauliOperators.I] * num_qubits
        ops[qubit_idx] = op
        return PauliOperators.tensor_product(*ops)
    
    @staticmethod
    def two_qubit_op(op1: np.ndarray, op2: np.ndarray, 
                     idx1: int, idx2: int, num_qubits: int) -> np.ndarray:
        """
        Costruisce operatore two-qubit sul sistema completo.
        
        Args:
            op1, op2: Operatori 2x2
            idx1, idx2: Indici dei qubit
            num_qubits: Numero totale di qubit
            
        Returns:
            Operatore 2^N x 2^N
        """
        ops = [PauliOperators.I] * num_qubits
        ops[idx1] = op1
        ops[idx2] = op2
        return PauliOperators.tensor_product(*ops)


class TFIMHamiltonian:
    """
    Transverse Field Ising Model (TFIM) Hamiltonian.
    
    Hamiltoniana statica (no annealing):
        H = Σᵢ Xᵢ + Σ⟨i,j⟩ Jᵢⱼ ZᵢZⱼ
    
    Caratteristiche:
        - Entrambi i termini sempre attivi (no scheduling)
        - Coefficiente campo trasverso = 1 (fisso)
        - Coefficiente interazione Ising = 1 (fisso)
        - Jᵢⱼ ~ Uniform[0,1] (generati una volta all'init)
    
    Uso:
        tfim = TFIMHamiltonian(num_qubits=3)
        H = tfim.H  # Hamiltoniana completa
        states = tfim.generate_sequential_states(5)  # 5 stati evoluti
    """
    
    def __init__(self, num_qubits: int, 
                 J_matrix: Optional[np.ndarray] = None,
                 seed: Optional[int] = None):
        """
        Inizializza l'Hamiltoniana TFIM.
        
        Args:
            num_qubits: Numero di qubit
            J_matrix: Matrice di accoppiamento Jᵢⱼ (simmetrica).
                     Se None, viene generata casualmente da Uniform[0,1].
            seed: Seed per riproducibilità dei Jᵢⱼ random
        """
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits  # Dimensione spazio di Hilbert
        
        # Coefficienti fissi (come da specifica)
        self.transverse_field_coeff = 1.0
        self.ising_coeff = 1.0
        
        # Genera matrice di accoppiamento J (una sola volta)
        if J_matrix is not None:
            assert J_matrix.shape == (num_qubits, num_qubits), \
                f"J_matrix deve essere {num_qubits}x{num_qubits}"
            self.J = J_matrix.copy()
        else:
            self.J = self._generate_random_coupling(seed)
        
        # Pre-calcola operatori (costosi da costruire)
        self._build_operators()
        
        # Costruisce Hamiltoniana completa (statica, no scheduling)
        self._H = self.H_transverse + self.H_ising
        
        print(f"[TFIM] Inizializzata Hamiltoniana Transverse Field Ising Model:")
        print(f"       - Qubit: {self.num_qubits}")
        print(f"       - Dimensione Hilbert: {self.dim}")
        print(f"       - Coefficiente campo trasverso: {self.transverse_field_coeff}")
        print(f"       - Coefficiente interazione Ising: {self.ising_coeff}")
        print(f"       - Jᵢⱼ ∈ [0, 1] (random, fissi)")
    
    def _generate_random_coupling(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Genera matrice di accoppiamento casuale simmetrica.
        Jᵢⱼ ~ Uniform[0, 1]
        
        Args:
            seed: Seed opzionale per riproducibilità
            
        Returns:
            Matrice J simmetrica con Jᵢⱼ ∈ [0, 1], diagonale = 0
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        # Genera matrice triangolare superiore
        J = np.zeros((self.num_qubits, self.num_qubits))
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                J[i, j] = rng.uniform(0, 1)
                J[j, i] = J[i, j]  # Simmetrica
        
        return J
    
    def _build_operators(self):
        """
        Pre-calcola i termini dell'Hamiltoniana.
        
        H_transverse = Σᵢ Xᵢ          (campo trasverso)
        H_ising = Σ⟨i,j⟩ Jᵢⱼ ZᵢZⱼ    (interazione Ising)
        """
        N = self.num_qubits
        
        # Termine Campo Trasverso: Σᵢ Xᵢ (coefficiente = 1)
        self.H_transverse = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(N):
            self.H_transverse += self.transverse_field_coeff * \
                PauliOperators.single_qubit_op(PauliOperators.X, i, N)
        
        # Termine Interazione Ising: Σ⟨i,j⟩ Jᵢⱼ ZᵢZⱼ (coefficiente = 1)
        self.H_ising = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(N):
            for j in range(i + 1, N):
                if self.J[i, j] != 0:
                    ZZ = PauliOperators.two_qubit_op(
                        PauliOperators.Z, PauliOperators.Z, i, j, N
                    )
                    self.H_ising += self.ising_coeff * self.J[i, j] * ZZ
    
    @property
    def H(self) -> np.ndarray:
        """
        Hamiltoniana completa (statica).
        
        H = Σᵢ Xᵢ + Σ⟨i,j⟩ Jᵢⱼ ZᵢZⱼ
        
        Returns:
            Matrice Hamiltoniana 2^N x 2^N
        """
        return self._H
    
    def ground_state(self) -> tuple:
        """
        Calcola lo stato fondamentale.
        
        Returns:
            (energia, stato): Energia e autovettore dello stato fondamentale
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self._H)
        idx = np.argmin(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]
    
    def compute_gap(self) -> float:
        """
        Calcola il gap energetico tra ground state e primo eccitato.
        
        Returns:
            Gap energetico Δ = E₁ - E₀
        """
        eigenvalues = np.linalg.eigvalsh(self._H)
        return eigenvalues[1] - eigenvalues[0]
    
    def generate_sequential_states(self, num_states: int, 
                                   max_time: float = 10.0,
                                   initial_state: Optional[np.ndarray] = None) -> tuple:
        """
        Genera N stati sequenziali usando evoluzione unitaria e^{-iHt}.
        
        Args:
            num_states: Numero di stati da generare
            max_time: Tempo massimo di evoluzione
            initial_state: Stato iniziale (default: |000...0⟩)
            
        Returns:
            (states, times): Array di stati shape (num_states, dim) e tempi
        """
        # Stato iniziale
        if initial_state is not None:
            psi0 = initial_state.copy()
        else:
            # Default: |000...0⟩ 
            psi0 = np.zeros(self.dim, dtype=complex)
            psi0[0] = 1.0
        
        # Normalizza
        psi0 = psi0 / np.linalg.norm(psi0)
        
        # Tempi per ogni stato
        times = np.linspace(0, max_time, num_states)
        
        # Diagonalizza H per evoluzione efficiente
        eigenvalues, eigenvectors = np.linalg.eigh(self._H)
        
        # Coefficienti: c_k = ⟨E_k|ψ₀⟩
        coeffs = np.conj(eigenvectors.T) @ psi0
        
        # Genera stati evoluti
        states = np.zeros((num_states, self.dim), dtype=complex)
        
        for i, t in enumerate(times):
            # |ψ(t)⟩ = Σ_k c_k e^{-iE_k t} |E_k⟩
            phase_factors = np.exp(-1j * eigenvalues * t)
            states[i] = eigenvectors @ (coeffs * phase_factors)
            # Normalizza per stabilità
            states[i] = states[i] / np.linalg.norm(states[i])
        
        return states, times
    
    def generate_sentences(self, num_sentences: int, max_time: Optional[int] = None, 
                          seed: Optional[int] = None) -> np.ndarray:
        """
        Genera N "frasi" quantistiche usando stati base casuali SENZA rimpiazzo.
        Cambia Hamiltoniana (matrice J) ogni 10 stati totali.
        
        Ogni "frase" è una sequenza di M stati ottenuti per evoluzione temporale
        da un diverso stato base computazionale |i⟩ scelto casualmente.
        
        Procedura:
        1. Sceglie N stati base diversi dai 2^D possibili (senza rimpiazzo)
        2. Per ogni stato base |i⟩, genera M stati usando evoluzione temporale
        3. Cambia matrice J ogni 10 stati totali (non ogni N/10 frasi)
        4. Risultato: array (N, M, 2^D) = (frasi, parole, dim_Hilbert)
        
        Args:
            num_sentences: N = numero di "frasi" (stati base diversi)
            max_time: M = numero di "parole" per frase (se None, usa config.py)  
            seed: Seed per riproducibilità della selezione casuale
            
        Returns:
            np.ndarray: Stati quantistici shape (N, M, 2^D)
                       - N = numero di frasi (stati base diversi)
                       - M = numero di parole per frase (tempi)
                       - 2^D = dimensione spazio di Hilbert
        """
        # Importa config per max_time
        if max_time is None:
            try:
                from config import QUANTUM_STATES_CONFIG
                max_time = QUANTUM_STATES_CONFIG.get('max_time', 5)
                print(f"[FixedH] 📋 max_time da config.py: {max_time}")
            except ImportError:
                max_time = 5
                print(f"[FixedH] ⚠️  config.py non trovato, uso max_time = 5")
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        total_basis_states = self.dim  # 2^D
        
        # Gestisci il caso N > 2^D con replacement ciclico
        if num_sentences <= total_basis_states:
            # Caso normale: senza replacement
            selected_indices = rng.choice(total_basis_states, size=num_sentences, replace=False)
            actual_sentences = num_sentences
            use_replacement = False
        else:
            # Caso speciale: usa tutti gli stati + replacement ciclico
            selected_indices = np.tile(np.arange(total_basis_states), 
                                     (num_sentences // total_basis_states) + 1)[:num_sentences]
            rng.shuffle(selected_indices)
            actual_sentences = num_sentences
            use_replacement = True
        
        # Stampa informazioni con bella formattazione
        print(f"\n[FixedH] Generazione sequenze con input variabile:")
        print(f"         - Possibili basis states: {total_basis_states} (2^{self.num_qubits})")
        print(f"         - N richiesti: {num_sentences}  ")
        print(f"         - N generati: {actual_sentences} ✓")
        
        if use_replacement:
            print(f"         - ⚠️  N > 2^D: usato replacement ciclico")
        else:
            print(f"         - ✅ N ≤ 2^D: nessun replacement")
        
        # Mostra gli stati base selezionati in notazione |xxx⟩
        print(f"[FixedH] 📊 BASIS STATES SELEZIONATI:")
        basis_display = []
        for idx in selected_indices:
            # Converti indice in stringa binaria
            binary_str = format(idx, f'0{self.num_qubits}b')
            basis_display.append(f"|{binary_str}⟩")
        
        # Stampa fino a 8 stati per riga per leggibilità
        line = "         "
        for i, state in enumerate(basis_display):
            line += state + " "
            if (i + 1) % 8 == 0 or i == len(basis_display) - 1:
                print(line)
                line = "         "
        
        # Calcola STATI totali e intervallo per cambiare J (ogni 10 STATI)
        total_states = actual_sentences * max_time
        change_j_every_states = 10  # Fisso: ogni 10 stati
        
        print(f"[FixedH] 🔄 Cambio J ogni {change_j_every_states} STATI (totale: {total_states} stati)")
        print(f"[FixedH] 📊 {actual_sentences} frasi × {max_time} parole = {total_states} stati")
        
        # Genera le sequenze per ogni stato base
        sentences = np.zeros((actual_sentences, max_time, self.dim), dtype=complex)
        current_j_group = -1  # Per tracciare quando cambiare J
        
        for i, basis_idx in enumerate(selected_indices):
            # Calcola range stati per questa frase
            first_state_idx = i * max_time  # Primo stato di questa frase
            last_state_idx = (i + 1) * max_time - 1  # Ultimo stato di questa frase
            
            # Verifica se è il momento di rigenerare J (controllo all'inizio della frase)
            j_group = first_state_idx // change_j_every_states
            if j_group != current_j_group:
                current_j_group = j_group
                
                # Rigenera matrice J con nuovo seed
                j_seed = (seed if seed is not None else 42) + j_group * 1000
                print(f"\n[FixedH] 🎲 GRUPPO J-{j_group + 1}: Rigenerando J")
                print(f"[FixedH] 📍 Frase {i+1}: stati globali {first_state_idx+1}-{last_state_idx+1}")
                
                # Rigenera J
                old_J = self.J.copy() if hasattr(self, 'J') else None
                self.J = self._generate_random_coupling(seed=j_seed)
                self._build_hamiltonian()  # Ricostruisce H con nuova J
                
                # Stampa confronto J
                print(f"[FixedH] 📊 MATRICE J GRUPPO {j_group + 1}:")
                print(f"         J shape: {self.J.shape}")
                print(f"         J range: [{np.min(self.J):.4f}, {np.max(self.J):.4f}]")
                print(f"         J media: {np.mean(self.J):.4f}")
                
                # Mostra matrice J (se piccola)
                if self.num_qubits <= 6:  # Solo per matrici piccole
                    print(f"         J matrix:")
                    for row_idx, row in enumerate(self.J):
                        print(f"           [{' '.join(f'{val:6.3f}' for val in row)}]")
                else:
                    print(f"         J matrix troppo grande ({self.J.shape}) - mostro solo statistiche")
                
                # Verifica che sia diversa dalla precedente
                if old_J is not None:
                    is_different = not np.allclose(old_J, self.J, atol=1e-10)
                    print(f"         Diversa da precedente: {'✅ SÌ' if is_different else '❌ NO'}")
            
            # Crea stato base |i⟩ 
            initial_state = np.zeros(self.dim, dtype=complex)
            initial_state[basis_idx] = 1.0
            
            # Genera sequenza temporale da questo stato base (usa J corrente)
            states, _ = self.generate_sequential_states(
                num_states=max_time,
                max_time=10.0,  # Tempo totale fisso
                initial_state=initial_state
            )
            
            sentences[i] = states
        
        print(f"\n[FixedH] ✅ Generate {actual_sentences} frasi × {max_time} parole = {actual_sentences * max_time} stati totali")
        print(f"[FixedH] 🎲 Utilizzate {current_j_group + 1} matrici J diverse")
        return sentences
    
    @classmethod
    def create_from_config(cls, config: dict) -> 'TFIMHamiltonian':
        """
        Crea istanza TFIM da configurazione (compatibile con config.py).
        
        Args:
            config: Dizionario con 'num_qubits' e opzionalmente 'seed'
            
        Returns:
            TFIMHamiltonian configurata
        """
        num_qubits = config.get('num_qubits', 2)
        seed = config.get('seed', None)
        return cls(num_qubits=num_qubits, seed=seed)


class SchedulingFunctions:
    """Funzioni di scheduling B(t) per quantum annealing."""
    
    @staticmethod
    def constant(t: float, T: float) -> float:
        """
        Scheduling costante: B(t) = 1 (sempre tunneling, per test)
        """
        return 1.0
    
    @staticmethod
    def linear(t: float, T: float) -> float:
        """
        Scheduling lineare: B(t) = 1 - t/T
        
        Args:
            t: Tempo corrente
            T: Tempo totale di annealing
        """
        return max(0.0, 1.0 - t / T)
    
    @staticmethod
    def quadratic(t: float, T: float) -> float:
        """
        Scheduling quadratico: B(t) = (1 - t/T)²
        Più lento all'inizio, più veloce alla fine.
        """
        s = t / T
        return max(0.0, (1.0 - s) ** 2)
    
    @staticmethod
    def sqrt(t: float, T: float) -> float:
        """
        Scheduling radice quadrata: B(t) = √(1 - t/T)
        Più veloce all'inizio, più lento alla fine.
        """
        s = t / T
        return max(0.0, np.sqrt(max(0.0, 1.0 - s)))
    
    @staticmethod
    def sinusoidal(t: float, T: float) -> float:
        """
        Scheduling sinusoidale: B(t) = cos²(πt/2T)
        Transizione smooth.
        """
        return np.cos(np.pi * t / (2 * T)) ** 2
    
    @staticmethod
    def exponential(t: float, T: float, tau: float = None) -> float:
        """
        Scheduling esponenziale: B(t) = exp(-t/τ)
        
        Args:
            tau: Costante di tempo (default: T/3)
        """
        if tau is None:
            tau = T / 3
        return np.exp(-t / tau)


class QuantumAnnealingHamiltonian:
    """
    Hamiltoniana per Quantum Annealing / Modello di Ising Trasverso.
    
    H(t) = -B(t) * Σᵢ σᵢˣ + Σ⟨i,j⟩ Jᵢⱼ σᵢᶻ σⱼᶻ
    
    Il primo termine (driver) favorisce il tunneling quantistico.
    Il secondo termine (problema) codifica le interazioni tra parole.
    """
    
    def __init__(self, num_words: int, 
                 J_matrix: Optional[np.ndarray] = None,
                 scheduling: str = 'linear',
                 total_time: float = 10.0):
        """
        Inizializza l'Hamiltoniana di Quantum Annealing.
        
        Args:
            num_words: Numero di parole (= numero di qubit)
            J_matrix: Matrice di accoppiamento Jᵢⱼ (simmetrica).
                     Se None, viene generata casualmente.
            scheduling: Tipo di scheduling ('linear', 'quadratic', 'sqrt', 
                       'sinusoidal', 'exponential')
            total_time: Tempo totale di annealing T
        """
        self.num_words = num_words
        self.num_qubits = num_words
        self.dim = 2 ** num_words  # Dimensione spazio di Hilbert
        self.total_time = total_time
        
        # Imposta matrice di accoppiamento
        if J_matrix is not None:
            assert J_matrix.shape == (num_words, num_words), \
                f"J_matrix deve essere {num_words}x{num_words}"
            self.J = J_matrix
        else:
            self.J = self._generate_random_coupling()
        
        # Imposta funzione di scheduling
        self._set_scheduling(scheduling)
        
        # Pre-calcola operatori (costosi da costruire)
        self._build_operators()
        
        print(f"[QA] Inizializzata Hamiltoniana Quantum Annealing:")
        print(f"     - Qubit/Parole: {self.num_words}")
        print(f"     - Dimensione Hilbert: {self.dim}")
        print(f"     - Scheduling: {scheduling}")
        print(f"     - Tempo totale: {total_time}")
    
    def _generate_random_coupling(self) -> np.ndarray:
        """Genera matrice di accoppiamento casuale simmetrica."""
        # Genera matrice triangolare superiore
        J = np.random.uniform(-1, 1, (self.num_words, self.num_words))
        # Rendi simmetrica
        J = (J + J.T) / 2
        # Zero sulla diagonale (no auto-interazione)
        np.fill_diagonal(J, 0)
        return J
    
    def _generate_nearest_neighbor_coupling(self, value: float = 1.0) -> np.ndarray:
        """
        Genera matrice di accoppiamento a PRIMI VICINI (tridiagonale).
        Solo interazioni tra qubit adiacenti: J_{i,i+1} = value
        
        Struttura (esempio N=4):
            [0, 1, 0, 0]
            [1, 0, 1, 0]
            [0, 1, 0, 1]
            [0, 0, 1, 0]
        """
        J = np.zeros((self.num_words, self.num_words))
        for i in range(self.num_words - 1):
            J[i, i + 1] = value
            J[i + 1, i] = value  # Simmetrica
        return J
    
    def _generate_uniform_coupling(self, value: float = 1.0) -> np.ndarray:
        """Genera matrice di accoppiamento uniforme (per test)."""
        J = np.full((self.num_words, self.num_words), value)
        np.fill_diagonal(J, 0)
        return J
    
    @classmethod
    def create_test_instance(cls, num_words: int, total_time: float = 10.0, coupling_type: str = 'nearest_neighbor'):
        """
        Crea un'istanza per test con B(t)=1 costante e J configurabile.
        
        Args:
            num_words: Numero di parole/qubit
            total_time: Tempo totale
            coupling_type: 'nearest_neighbor' (primi vicini) o 'uniform' (tutti con tutti)
            
        Returns:
            QuantumAnnealingHamiltonian configurata per test
        """
        if coupling_type == 'nearest_neighbor':
            # J a primi vicini (tridiagonale)
            J_matrix = np.zeros((num_words, num_words))
            for i in range(num_words - 1):
                J_matrix[i, i + 1] = 1.0
                J_matrix[i + 1, i] = 1.0
        else:  # uniform
            # J uniforme (tutti con tutti)
            J_matrix = np.ones((num_words, num_words))
            np.fill_diagonal(J_matrix, 0)
        
        return cls(
            num_words=num_words,
            J_matrix=J_matrix,
            scheduling='constant',  # B(t) = 1 sempre
            total_time=total_time
        )
    
    def _set_scheduling(self, scheduling: str):
        """Imposta la funzione di scheduling."""
        schedules = {
            'constant': SchedulingFunctions.constant,  # Per test: B(t) = 1 sempre
            'linear': SchedulingFunctions.linear,
            'quadratic': SchedulingFunctions.quadratic,
            'sqrt': SchedulingFunctions.sqrt,
            'sinusoidal': SchedulingFunctions.sinusoidal,
            'exponential': SchedulingFunctions.exponential
        }
        if scheduling not in schedules:
            raise ValueError(f"Scheduling '{scheduling}' non supportato. "
                           f"Opzioni: {list(schedules.keys())}")
        self._scheduling_func = schedules[scheduling]
    
    def _build_operators(self):
        """Pre-calcola gli operatori del driver e del problema."""
        N = self.num_qubits
        
        # Termine Driver: -Σᵢ σᵢˣ
        self.H_driver = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(N):
            self.H_driver -= PauliOperators.single_qubit_op(
                PauliOperators.X, i, N
            )
        
        # Termine Problema: Σ⟨i,j⟩ Jᵢⱼ σᵢᶻ σⱼᶻ
        self.H_problem = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(N):
            for j in range(i + 1, N):
                if self.J[i, j] != 0:
                    ZZ = PauliOperators.two_qubit_op(
                        PauliOperators.Z, PauliOperators.Z, i, j, N
                    )
                    self.H_problem += self.J[i, j] * ZZ
    
    def B(self, t: float) -> float:
        """
        Funzione di scheduling B(t).
        
        Args:
            t: Tempo corrente
            
        Returns:
            Valore di B(t) ∈ [0, 1]
        """
        return self._scheduling_func(t, self.total_time)
    
    def H(self, t: float) -> np.ndarray:
        """
        Hamiltoniana completa al tempo t.
        
        H(t) = B(t) * H_driver + (1 - B(t)) * H_problem
        
        Nota: usiamo la formulazione standard dove:
        - A t=0: H ≈ H_driver (tunneling dominante)
        - A t=T: H ≈ H_problem (stato classico)
        
        Args:
            t: Tempo corrente
            
        Returns:
            Matrice Hamiltoniana 2^N x 2^N
        """
        b = self.B(t)
        return b * self.H_driver + (1 - b) * self.H_problem
    
    def ground_state(self, t: float) -> tuple:
        """
        Calcola lo stato fondamentale di H(t).
        
        Args:
            t: Tempo corrente
            
        Returns:
            (energia, stato): Energia e autovettore dello stato fondamentale
        """
        H_t = self.H(t)
        eigenvalues, eigenvectors = np.linalg.eigh(H_t)
        idx = np.argmin(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]
    
    def evolve_adiabatic(self, num_steps: int = 100, 
                         dt: Optional[float] = None) -> np.ndarray:
        """
        Evoluzione adiabatica del sistema.
        
        Inizia dallo stato fondamentale del driver e evolve
        seguendo l'Hamiltoniana dipendente dal tempo.
        
        Args:
            num_steps: Numero di step temporali
            dt: Passo temporale (default: T/num_steps)
            
        Returns:
            Array di stati evoluti, shape (num_steps+1, dim)
        """
        if dt is None:
            dt = self.total_time / num_steps
        
        # Stato iniziale: ground state del driver (t=0)
        # Per H_driver = -Σ σˣ, il ground state è |+⟩^⊗N
        psi = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        
        states = [psi.copy()]
        times = [0.0]
        
        for step in range(num_steps):
            t = step * dt
            H_t = self.H(t)
            
            # Evoluzione unitaria: |ψ(t+dt)⟩ = exp(-i H(t) dt) |ψ(t)⟩
            U = expm(-1j * H_t * dt)
            psi = U @ psi
            
            # Normalizza (per stabilità numerica)
            psi = psi / np.linalg.norm(psi)
            
            states.append(psi.copy())
            times.append((step + 1) * dt)
        
        return np.array(states), np.array(times)
    
    def compute_gap(self, t: float) -> float:
        """
        Calcola il gap energetico tra ground state e primo eccitato.
        
        Args:
            t: Tempo corrente
            
        Returns:
            Gap energetico Δ = E₁ - E₀
        """
        H_t = self.H(t)
        eigenvalues = np.linalg.eigvalsh(H_t)
        return eigenvalues[1] - eigenvalues[0]
    
    def find_minimum_gap(self, num_points: int = 100) -> tuple:
        """
        Trova il gap minimo durante l'annealing.
        
        Returns:
            (t_min, gap_min): Tempo e valore del gap minimo
        """
        times = np.linspace(0, self.total_time, num_points)
        gaps = [self.compute_gap(t) for t in times]
        idx = np.argmin(gaps)
        return times[idx], gaps[idx]
    
    def set_coupling_from_attention(self, attention_matrix: np.ndarray):
        """
        Imposta la matrice di accoppiamento J dai pesi di attenzione.
        
        Questo permette di codificare le relazioni tra parole
        (dal transformer) nell'Hamiltoniana di Ising.
        
        Args:
            attention_matrix: Matrice di attenzione NxN dal transformer
        """
        assert attention_matrix.shape == (self.num_words, self.num_words), \
            f"Attention matrix deve essere {self.num_words}x{self.num_words}"
        
        # Normalizza e rendi simmetrica
        J = (attention_matrix + attention_matrix.T) / 2
        
        # Scala a [-1, 1]
        if np.max(np.abs(J)) > 0:
            J = J / np.max(np.abs(J))
        
        np.fill_diagonal(J, 0)
        self.J = J
        
        # Ricostruisci operatore problema
        self._build_operators()
        
        print(f"[QA] Matrice J aggiornata da attention weights")


    def generate_sequential_states(self, num_states: int, max_time: float = None, 
                                     initial_state: np.ndarray = None) -> np.ndarray:
        """
        Genera N stati sequenziali usando evoluzione unitaria e^{-iHt}.
        Ogni stato rappresenta una "parola" nel contesto del transformer.
        
        Segue la stessa logica di hamiltonian.py:
        - Stato iniziale (fisso o random)
        - Evoluzione temporale con H costante (modalità test)
        - Ogni tempo t corrisponde a una parola
        
        Args:
            num_states: Numero di stati da generare (= numero di parole)
            max_time: Tempo massimo di evoluzione (default: total_time)
            initial_state: Stato iniziale (default: |+⟩^⊗N)
            
        Returns:
            np.ndarray: Array di stati, shape (num_states, dim)
        """
        if max_time is None:
            max_time = self.total_time
        
        # Stato iniziale
        if initial_state is not None:
            psi0 = initial_state.copy()
        else:
            # Default: stato |+⟩^⊗N (ground state del driver)
            psi0 = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)
        
        # Normalizza
        psi0 = psi0 / np.linalg.norm(psi0)
        
        # Tempi per ogni "parola"
        times = np.linspace(0, max_time, num_states)
        
        # Usa H(0) per l'evoluzione (in modalità test B(t)=1, è sempre H_driver)
        H = self.H(0)
        
        # Diagonalizza H per evoluzione efficiente
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        # Coefficienti: c_k = ⟨E_k|ψ₀⟩
        coeffs = np.conj(eigenvectors.T) @ psi0
        
        # Genera stati evoluti
        states = np.zeros((num_states, self.dim), dtype=complex)
        
        for i, t in enumerate(times):
            # |ψ(t)⟩ = Σ_k c_k e^{-iE_k t} |E_k⟩
            phase_factors = np.exp(-1j * eigenvalues * t)
            states[i] = eigenvectors @ (coeffs * phase_factors)
            # Normalizza per stabilità
            states[i] = states[i] / np.linalg.norm(states[i])
        
        return states, times


def generate_quantum_states(num_states: int, num_qubits: int = 2, 
                            max_time: float = 10.0,
                            initial_state: np.ndarray = None,
                            use_test_mode: bool = True,
                            seed: Optional[int] = None,
                            use_random_basis: bool = False,
                            num_time_steps: int = 5) -> np.ndarray:
    """
    Funzione principale per generare N stati quantistici.
    
    Due modalità:
    1. SEQUENTIAL (use_random_basis=False): 
       Evoluzione temporale da stato iniziale fisso
       Returns: shape (num_states, 2^num_qubits)
       
    2. RANDOM BASIS (use_random_basis=True):
       N stati base casuali senza rimpiazzo, ognuno evoluto M volte
       Returns: shape (num_states, num_time_steps, 2^num_qubits)
    
    USA SEMPRE TFIMHamiltonian con:
        H = Σᵢ Xᵢ + Σ⟨i,j⟩ Jᵢⱼ ZᵢZⱼ
    
    Caratteristiche:
        - Entrambi i termini sempre attivi (NO scheduling)
        - Coefficiente campo trasverso = +1 (fisso, segno positivo)
        - Coefficiente interazione Ising = +1 (fisso, segno positivo)
        - Jᵢⱼ ~ Uniform[0,1]
    
    Args:
        num_states: Numero di stati/frasi da generare
        num_qubits: Numero di qubit (default: 2, → dim=4)
        max_time: Tempo massimo di evoluzione (default: 10.0)
        initial_state: Stato iniziale custom (solo per sequential)
        use_test_mode: IGNORATO - mantenuto per retrocompatibilità
        seed: Seed per riproducibilità
        use_random_basis: True = modalità frasi (random basis), False = sequential
        num_time_steps: Numero di step temporali per ogni frase (solo random basis)
        
    Returns:
        np.ndarray: Stati quantistici
                   - Sequential: shape (num_states, 2^num_qubits)
                   - Random basis: shape (num_states, num_time_steps, 2^num_qubits)
    """
    # USA SEMPRE TFIMHamiltonian (ignora use_test_mode)
    hamiltonian = TFIMHamiltonian(num_qubits=num_qubits, seed=seed)
    
    if use_random_basis:
        # Modalità RANDOM BASIS: genera frasi con stati base diversi
        return hamiltonian.generate_sentences(
            num_sentences=num_states,
            max_time=num_time_steps,
            seed=seed
        )
    else:
        # Modalità SEQUENTIAL: evoluzione da stato iniziale fisso
        if initial_state is None:
            initial_state = np.zeros(2**num_qubits, dtype=complex)
            initial_state[0] = 1.0
        
        # Genera stati
        states, times = hamiltonian.generate_sequential_states(
            num_states, max_time=max_time, initial_state=initial_state
        )
        
        return states


def generate_tfim_states(num_states: int, num_qubits: int = 2,
                         max_time: float = 10.0,
                         initial_state: np.ndarray = None,
                         seed: Optional[int] = None) -> np.ndarray:
    """
    Genera N stati quantistici sequenziali usando l'Hamiltoniana TFIM.
    
    Hamiltoniana: H = Σᵢ Xᵢ + Σ⟨i,j⟩ Jᵢⱼ ZᵢZⱼ
    
    Caratteristiche:
        - Entrambi i termini sempre attivi (NO scheduling)
        - Coefficiente campo trasverso = +1 (fisso, segno positivo)
        - Coefficiente interazione Ising = +1 (fisso, segno positivo)
        - Jᵢⱼ ~ Uniform[0,1]
    
    Uso semplice:
        states = generate_tfim_states(N)
    
    Args:
        num_states: Numero di stati da generare (= numero di "parole")
        num_qubits: Numero di qubit (default: 2, → dim=4)
        max_time: Tempo massimo di evoluzione (default: 10.0)
        initial_state: Stato iniziale custom (default: |000...0⟩)
        seed: Seed per riproducibilità dei Jᵢⱼ random
        
    Returns:
        np.ndarray: Array di stati quantistici, shape (num_states, 2^num_qubits)
    """
    # Delega a generate_quantum_states (stessa implementazione)
    return generate_quantum_states(
        num_states=num_states,
        num_qubits=num_qubits,
        max_time=max_time,
        initial_state=initial_state,
        seed=seed
    )


class AnnealingSimulator:
    """
    Simulatore completo per Quantum Annealing su frasi.
    """
    
    def __init__(self, sentence_length: int, **kwargs):
        """
        Args:
            sentence_length: Numero di parole nella frase
            **kwargs: Argomenti passati a QuantumAnnealingHamiltonian
        """
        self.hamiltonian = QuantumAnnealingHamiltonian(sentence_length, **kwargs)
    
    def run(self, num_steps: int = 100) -> dict:
        """
        Esegue la simulazione completa.
        
        Returns:
            dict con: states, times, final_state, ground_energy, success_prob
        """
        states, times = self.hamiltonian.evolve_adiabatic(num_steps)
        
        # Stato finale e ground state target
        final_state = states[-1]
        target_energy, target_state = self.hamiltonian.ground_state(self.hamiltonian.total_time)
        
        # Probabilità di successo: |⟨target|final⟩|²
        success_prob = np.abs(np.vdot(target_state, final_state)) ** 2
        
        # Gap minimo
        t_min, gap_min = self.hamiltonian.find_minimum_gap()
        
        return {
            'states': states,
            'times': times,
            'final_state': final_state,
            'target_state': target_state,
            'ground_energy': target_energy,
            'success_probability': success_prob,
            'minimum_gap': gap_min,
            'minimum_gap_time': t_min
        }


# ============================================================================
# ESEMPIO DI UTILIZZO E TEST
# ============================================================================
if __name__ == "__main__":
    from scipy.linalg import eigh
    
    print("=" * 70)
    print("QUANTUM ANNEALING HAMILTONIAN - Test Suite")
    print("=" * 70)
    
    # =========================================================================
    # TEST 1: Istanza di test con B(t)=1 e J=1 (uniforme)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Modalità Test (B(t)=1 costante, J=1 uniforme)")
    print("=" * 70)
    
    num_words = 9  # 3 qubit = 8 dimensioni
    qa_test = QuantumAnnealingHamiltonian.create_test_instance(num_words, total_time=5.0)
    
    print(f"\n[TEST] Matrice di accoppiamento J (uniforme):")
    print(qa_test.J)
    
    print(f"\n[TEST] Verifica B(t) = 1 costante:")
    for t in [0, 1.0, 2.5, 5.0]:
        b_val = qa_test.B(t)
        print(f"  B({t}) = {b_val:.4f} {'✓' if b_val == 1.0 else '✗'}")
    
    print(f"\n[TEST] Hamiltoniana Driver H_driver (solo termine -Σσˣ):")
    print(f"  Shape: {qa_test.H_driver.shape}")
    print(f"  È Hermitiana: {np.allclose(qa_test.H_driver, qa_test.H_driver.conj().T)}")
    print(f"  Traccia: {np.trace(qa_test.H_driver):.4f}")
    
    print(f"\n[TEST] Hamiltoniana Problema H_problem (solo termine Σ Jᵢⱼ σᶻσᶻ):")
    print(f"  Shape: {qa_test.H_problem.shape}")
    print(f"  È Hermitiana: {np.allclose(qa_test.H_problem, qa_test.H_problem.conj().T)}")
    print(f"  Traccia: {np.trace(qa_test.H_problem):.4f}")
    
    # Con B(t)=1, H(t) = H_driver sempre
    H_at_0 = qa_test.H(0)
    H_at_5 = qa_test.H(5.0)
    print(f"\n[TEST] Con B(t)=1: H(0) == H(5) == H_driver?")
    print(f"  H(0) == H_driver: {np.allclose(H_at_0, qa_test.H_driver)}")
    print(f"  H(5) == H_driver: {np.allclose(H_at_5, qa_test.H_driver)}")
    
    # Autovalori
    eigenvalues_driver = np.linalg.eigvalsh(qa_test.H_driver)
    print(f"\n[TEST] Autovalori H_driver:")
    for i, ev in enumerate(eigenvalues_driver):
        print(f"  E_{i} = {ev:.4f}")
    
    # Ground state
    E0, psi0 = qa_test.ground_state(0)
    print(f"\n[TEST] Ground state:")
    print(f"  Energia: E₀ = {E0:.4f}")
    print(f"  Stato |ψ₀⟩: {psi0}")
    print(f"  Normalizzato: {np.abs(np.linalg.norm(psi0) - 1) < 1e-10}")
    
    # =========================================================================
    # TEST 2: Evoluzione temporale (stile hamiltonian.py)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Evoluzione Temporale (come hamiltonian.py)")
    print("=" * 70)
    
    num_states = 5
    max_time = 5.0
    
    print(f"\n[EVOLVE] Generazione {num_states} stati sequenziali...")
    states, times = qa_test.evolve_adiabatic(num_steps=num_states)
    
    print(f"[EVOLVE] Shape array stati: {states.shape}")
    print(f"[EVOLVE] Tempi: {times}")
    
    # Verifica normalizzazione
    norms = [np.linalg.norm(s) for s in states]
    print(f"[EVOLVE] Ogni stato è normalizzato: {[np.abs(n - 1) < 1e-10 for n in norms]}")
    
    # Mostra evoluzione probabilità
    print(f"\n[EVOLVE] Evoluzione probabilità |ψ(t)|²:")
    for i, (t, psi) in enumerate(zip(times, states)):
        probs = np.abs(psi) ** 2
        print(f"  t={t:.2f}: P(|000⟩)={probs[0]:.4f}, P(|001⟩)={probs[1]:.4f}, "
              f"P(|010⟩)={probs[2]:.4f}, P(|011⟩)={probs[3]:.4f}")
    
    # =========================================================================
    # TEST 3: Confronto scheduling diversi
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Confronto Scheduling B(t)")
    print("=" * 70)
    
    schedules = ['constant', 'linear', 'quadratic', 'sinusoidal']
    T = 10.0
    test_times = [0, 2.5, 5.0, 7.5, 10.0]
    
    print(f"\n{'t':<8}", end="")
    for s in schedules:
        print(f"{s:<12}", end="")
    print()
    print("-" * 56)
    
    for t in test_times:
        print(f"{t:<8.1f}", end="")
        for s in schedules:
            if s == 'constant':
                val = SchedulingFunctions.constant(t, T)
            elif s == 'linear':
                val = SchedulingFunctions.linear(t, T)
            elif s == 'quadratic':
                val = SchedulingFunctions.quadratic(t, T)
            elif s == 'sinusoidal':
                val = SchedulingFunctions.sinusoidal(t, T)
            print(f"{val:<12.4f}", end="")
        print()
    
    # =========================================================================
    # TEST 4: Operatori di Pauli
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Verifica Operatori di Pauli")
    print("=" * 70)
    
    print("\n[PAULI] Matrici base:")
    print(f"  σˣ:\n{PauliOperators.X}")
    print(f"  σʸ:\n{PauliOperators.Y}")
    print(f"  σᶻ:\n{PauliOperators.Z}")
    
    # Verifica proprietà
    print(f"\n[PAULI] Proprietà:")
    print(f"  σˣ² = I: {np.allclose(PauliOperators.X @ PauliOperators.X, PauliOperators.I)}")
    print(f"  σʸ² = I: {np.allclose(PauliOperators.Y @ PauliOperators.Y, PauliOperators.I)}")
    print(f"  σᶻ² = I: {np.allclose(PauliOperators.Z @ PauliOperators.Z, PauliOperators.I)}")
    print(f"  [σˣ,σʸ] = 2iσᶻ: {np.allclose(PauliOperators.X @ PauliOperators.Y - PauliOperators.Y @ PauliOperators.X, 2j * PauliOperators.Z)}")
    
    # Test operatore su sistema multi-qubit
    print(f"\n[PAULI] Operatore σˣ₀ su 2 qubit (agisce sul primo):")
    sigma_x_0 = PauliOperators.single_qubit_op(PauliOperators.X, 0, 2)
    print(sigma_x_0.real)
    
    print(f"\n[PAULI] Operatore σᶻ₀σᶻ₁ su 2 qubit:")
    sigma_zz = PauliOperators.two_qubit_op(PauliOperators.Z, PauliOperators.Z, 0, 1, 2)
    print(sigma_zz.real)
    
    # =========================================================================
    # TEST 5: Gap energetico
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 5: Gap Energetico")
    print("=" * 70)
    
    # Usa scheduling lineare per vedere il gap cambiare
    qa_gap = QuantumAnnealingHamiltonian(
        num_words=3,
        scheduling='linear',
        total_time=10.0
    )
    
    print(f"\n[GAP] Evoluzione gap Δ(t) = E₁ - E₀:")
    for t in [0, 2, 4, 6, 8, 10]:
        gap = qa_gap.compute_gap(t)
        b = qa_gap.B(t)
        print(f"  t={t:>4.1f}: B(t)={b:.3f}, Δ={gap:.6f}")
    
    t_min, gap_min = qa_gap.find_minimum_gap()
    print(f"\n[GAP] Gap MINIMO: Δ_min = {gap_min:.6f} a t = {t_min:.2f}")
    
    # =========================================================================
    # TEST 6: Simulazione completa (stile hamiltonian.py)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 6: Simulazione Completa")
    print("=" * 70)
    
    sim = AnnealingSimulator(sentence_length=3, scheduling='linear', total_time=10.0)
    results = sim.run(num_steps=100)
    
    print(f"\n[SIM] Risultati:")
    print(f"  Probabilità successo: {results['success_probability']:.4f}")
    print(f"  Energia ground state: {results['ground_energy']:.4f}")
    print(f"  Gap minimo: {results['minimum_gap']:.6f} a t={results['minimum_gap_time']:.2f}")
    print(f"  Shape stati: {results['states'].shape}")
    print(f"  Stato finale normalizzato: {np.abs(np.linalg.norm(results['final_state']) - 1) < 1e-10}")
    
    # =========================================================================
    # TEST 7: Test con J da attention matrix (simulata)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 7: J da Attention Matrix (simulata)")
    print("=" * 70)
    
    # Simula una matrice di attenzione
    np.random.seed(42)
    attention = np.random.rand(3, 3)
    attention = (attention + attention.T) / 2  # Rendi simmetrica
    
    print(f"\n[ATTN] Matrice di attenzione simulata:")
    print(attention)
    
    qa_attn = QuantumAnnealingHamiltonian(num_words=3, scheduling='linear')
    qa_attn.set_coupling_from_attention(attention)
    
    print(f"\n[ATTN] Matrice J risultante (normalizzata):")
    print(qa_attn.J)
    
    print("\n" + "=" * 70)
    print("TUTTI I TEST COMPLETATI ✓")
    print("=" * 70)
    
    # =========================================================================
    # TEST 8: Generazione Stati Sequenziali (come hamiltonian.py)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 8: Generazione Stati Sequenziali (e^{-iHt})")
    print("=" * 70)
    
    num_words_test = 3
    num_states_test = 5
    
    print(f"\n[STATES] Configurazione:")
    print(f"  Numero qubit (parole): {num_words_test}")
    print(f"  Dimensione Hilbert: 2^{num_words_test} = {2**num_words_test}")
    print(f"  Numero stati da generare: {num_states_test}")
    
    # Usa istanza test con B(t)=1, J=1
    qa_states = QuantumAnnealingHamiltonian.create_test_instance(num_words_test, total_time=10.0)
    
    print(f"\n[STATES] Hamiltoniana usata: H = H_driver (B(t)=1)")
    print(f"  Shape H: {qa_states.H(0).shape}")
    print(f"  H è Hermitiana: {np.allclose(qa_states.H(0), qa_states.H(0).conj().T)}")
    
    # Genera stati
    states, times = qa_states.generate_sequential_states(num_states_test, max_time=10.0)
    
    print(f"\n[STATES] Stati generati:")
    print(f"  Shape array: {states.shape}")
    print(f"  Tempi: {times}")
    
    # Verifica normalizzazione
    print(f"\n[STATES] Verifica normalizzazione:")
    for i, (t, psi) in enumerate(zip(times, states)):
        norm = np.linalg.norm(psi)
        print(f"  Stato {i} (t={t:.2f}): ||ψ|| = {norm:.10f} {'✓' if np.abs(norm-1) < 1e-10 else '✗'}")
    
    # Mostra ampiezze di probabilità
    print(f"\n[STATES] Ampiezze di probabilità |⟨basis|ψ(t)⟩|²:")
    basis_labels = [f"|{bin(i)[2:].zfill(num_words_test)}⟩" for i in range(2**num_words_test)]
    
    # Header
    print(f"  {'t':<8}", end="")
    for label in basis_labels:
        print(f"{label:<10}", end="")
    print()
    print("  " + "-" * (8 + 10 * len(basis_labels)))
    
    # Probabilità per ogni stato
    for i, (t, psi) in enumerate(zip(times, states)):
        probs = np.abs(psi) ** 2
        print(f"  {t:<8.2f}", end="")
        for p in probs:
            print(f"{p:<10.4f}", end="")
        print()
    
    # Verifica che la somma delle probabilità sia 1
    print(f"\n[STATES] Verifica Σ|ψᵢ|² = 1:")
    for i, (t, psi) in enumerate(zip(times, states)):
        prob_sum = np.sum(np.abs(psi) ** 2)
        print(f"  Stato {i} (t={t:.2f}): Σ = {prob_sum:.10f} {'✓' if np.abs(prob_sum-1) < 1e-10 else '✗'}")
    
    # =========================================================================
    # TEST 9: Overlap tra stati consecutivi
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 9: Overlap tra Stati Consecutivi")
    print("=" * 70)
    
    print(f"\n[OVERLAP] |⟨ψ(tᵢ)|ψ(tᵢ₊₁)⟩|² (quanto sono simili stati consecutivi):")
    for i in range(len(states) - 1):
        overlap = np.abs(np.vdot(states[i], states[i+1])) ** 2
        print(f"  ⟨ψ({times[i]:.2f})|ψ({times[i+1]:.2f})⟩|² = {overlap:.6f}")
    
    # Overlap con stato iniziale
    print(f"\n[OVERLAP] |⟨ψ(0)|ψ(t)⟩|² (overlap con stato iniziale):")
    psi_initial = states[0]
    for i, (t, psi) in enumerate(zip(times, states)):
        overlap = np.abs(np.vdot(psi_initial, psi)) ** 2
        print(f"  |⟨ψ(0)|ψ({t:.2f})⟩|² = {overlap:.6f}")
    
    # =========================================================================
    # TEST 10: Evoluzione fasi (parte reale e immaginaria)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 10: Evoluzione Fasi degli Stati")
    print("=" * 70)
    
    print(f"\n[PHASE] Componente |000⟩ (primo elemento):")
    print(f"  {'t':<8} {'Re(ψ₀)':<15} {'Im(ψ₀)':<15} {'|ψ₀|':<15} {'arg(ψ₀)':<15}")
    print("  " + "-" * 68)
    for t, psi in zip(times, states):
        re_part = psi[0].real
        im_part = psi[0].imag
        magnitude = np.abs(psi[0])
        phase = np.angle(psi[0])
        print(f"  {t:<8.2f} {re_part:<15.6f} {im_part:<15.6f} {magnitude:<15.6f} {phase:<15.6f}")
    
    # =========================================================================
    # TEST 11: Confronto con hamiltonian.py style
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 11: Confronto con HamiltonianEvolution (hamiltonian.py style)")
    print("=" * 70)
    
    # Genera più stati per vedere l'evoluzione
    num_states_many = 10
    states_many, times_many = qa_states.generate_sequential_states(num_states_many, max_time=2*np.pi)
    
    print(f"\n[COMPARE] Generati {num_states_many} stati su intervallo [0, 2π]")
    print(f"  Shape: {states_many.shape}")
    print(f"  Questo simula {num_states_many} 'parole' in una frase")
    
    # Calcola energia media per ogni stato
    H = qa_states.H(0)
    print(f"\n[COMPARE] Energia ⟨ψ(t)|H|ψ(t)⟩ per ogni stato:")
    for i, (t, psi) in enumerate(zip(times_many, states_many)):
        energy = np.real(np.vdot(psi, H @ psi))
        print(f"  Parola {i} (t={t:.4f}): E = {energy:.6f}")
    
    # =========================================================================
    # TEST 12: Stati con stato iniziale custom
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 12: Stati con Stato Iniziale Custom")
    print("=" * 70)
    
    # Stato iniziale |000⟩
    psi0_custom = np.zeros(2**num_words_test, dtype=complex)
    psi0_custom[0] = 1.0
    
    print(f"\n[CUSTOM] Stato iniziale: |000⟩")
    print(f"  ψ₀ = {psi0_custom}")
    
    states_custom, times_custom = qa_states.generate_sequential_states(
        5, max_time=5.0, initial_state=psi0_custom
    )
    
    print(f"\n[CUSTOM] Evoluzione probabilità da |000⟩:")
    print(f"  {'t':<8}", end="")
    for label in basis_labels:
        print(f"{label:<10}", end="")
    print()
    print("  " + "-" * (8 + 10 * len(basis_labels)))
    
    for t, psi in zip(times_custom, states_custom):
        probs = np.abs(psi) ** 2
        print(f"  {t:<8.2f}", end="")
        for p in probs:
            print(f"{p:<10.4f}", end="")
        print()
    
    # =========================================================================
    # TEST 13: Verifica unitarietà evoluzione
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 13: Verifica Unitarietà dell'Evoluzione")
    print("=" * 70)
    
    # U = e^{-iHt} deve essere unitaria: U†U = I
    from scipy.linalg import expm
    
    H = qa_states.H(0)
    t_test = 1.0
    U = expm(-1j * H * t_test)
    
    print(f"\n[UNITARY] Operatore U = e^{{-iHt}} con t={t_test}")
    print(f"  Shape U: {U.shape}")
    print(f"  U†U ≈ I: {np.allclose(U.conj().T @ U, np.eye(U.shape[0]))}")
    print(f"  UU† ≈ I: {np.allclose(U @ U.conj().T, np.eye(U.shape[0]))}")
    print(f"  |det(U)| ≈ 1: {np.abs(np.abs(np.linalg.det(U)) - 1) < 1e-10}")
    
    print("\n" + "=" * 70)
    print("TUTTI I TEST COMPLETATI ✓✓")
    print("=" * 70)
    
    # =========================================================================
    # TEST 14: Hamiltoniana Statica H = X₁ + X₂ + Z₁Z₂ (2 qubit)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 14: Hamiltoniana Statica H = X₁ + X₂ + Z₁Z₂")
    print("=" * 70)
    
    print("\n[STATIC] Costruzione Hamiltoniana per 2 qubit:")
    print("         H = σˣ₁ ⊗ I + I ⊗ σˣ₂ + σᶻ₁ ⊗ σᶻ₂")
    print("           = X₁ + X₂ + Z₁Z₂")
    
    # Costruisci manualmente
    I = PauliOperators.I
    X = PauliOperators.X
    Z = PauliOperators.Z
    
    # X₁ = X ⊗ I
    X1 = np.kron(X, I)
    # X₂ = I ⊗ X  
    X2 = np.kron(I, X)
    # Z₁Z₂ = Z ⊗ Z
    Z1Z2 = np.kron(Z, Z)
    
    # H totale
    H_static = X1 + X2 + Z1Z2
    
    print(f"\n[STATIC] Matrici componenti:")
    print(f"  X₁ (X ⊗ I):\n{X1.real}")
    print(f"\n  X₂ (I ⊗ X):\n{X2.real}")
    print(f"\n  Z₁Z₂ (Z ⊗ Z):\n{Z1Z2.real}")
    
    print(f"\n[STATIC] Hamiltoniana H = X₁ + X₂ + Z₁Z₂:")
    print(H_static.real)
    
    print(f"\n[STATIC] Proprietà:")
    print(f"  Shape: {H_static.shape}")
    print(f"  È Hermitiana: {np.allclose(H_static, H_static.conj().T)}")
    print(f"  Traccia: {np.trace(H_static).real:.4f}")
    
    # Autovalori e autovettori
    eigenvalues, eigenvectors = np.linalg.eigh(H_static)
    
    print(f"\n[STATIC] AUTOVALORI:")
    for i, ev in enumerate(eigenvalues):
        print(f"  E_{i} = {ev:.6f}")
    
    print(f"\n[STATIC] AUTOVETTORI (colonne):")
    print(f"  Base computazionale: |00⟩, |01⟩, |10⟩, |11⟩")
    print(f"\n  Autovettori:")
    for i in range(4):
        vec = eigenvectors[:, i]
        print(f"  |E_{i}⟩ = {vec.real.round(6)}")
        # Mostra in notazione ket
        labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
        components = []
        for j, (amp, label) in enumerate(zip(vec, labels)):
            if np.abs(amp) > 1e-10:
                if np.abs(amp.imag) < 1e-10:
                    components.append(f"{amp.real:+.4f}{label}")
                else:
                    components.append(f"({amp.real:+.4f}{amp.imag:+.4f}i){label}")
        print(f"        = {' '.join(components)}")
    
    # Verifica: H|E_i⟩ = E_i|E_i⟩
    print(f"\n[STATIC] Verifica H|Eᵢ⟩ = Eᵢ|Eᵢ⟩:")
    for i in range(4):
        Hv = H_static @ eigenvectors[:, i]
        Ev = eigenvalues[i] * eigenvectors[:, i]
        is_correct = np.allclose(Hv, Ev)
        print(f"  E_{i} = {eigenvalues[i]:+.4f}: {is_correct} ✓" if is_correct else f"  E_{i}: ✗")
    
    # Ground state
    print(f"\n[STATIC] GROUND STATE:")
    gs_idx = np.argmin(eigenvalues)
    gs_energy = eigenvalues[gs_idx]
    gs_vector = eigenvectors[:, gs_idx]
    print(f"  Energia: E₀ = {gs_energy:.6f}")
    print(f"  Vettore: |ψ₀⟩ = {gs_vector.real.round(6)}")
    print(f"  Probabilità:")
    for j, (amp, label) in enumerate(zip(gs_vector, ['|00⟩', '|01⟩', '|10⟩', '|11⟩'])):
        prob = np.abs(amp)**2
        print(f"    P({label}) = {prob:.6f}")
    
    # =========================================================================
    # TEST 15: J a Primi Vicini (tridiagonale)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 15: Matrice J a Primi Vicini")
    print("=" * 70)
    
    for n in [2, 3, 4, 5]:
        print(f"\n[NN] J per N={n} qubit (primi vicini):")
        qa_nn = QuantumAnnealingHamiltonian.create_test_instance(n, coupling_type='nearest_neighbor')
        print(qa_nn.J.astype(int))
    
    # Confronto con uniform
    print(f"\n[NN] Confronto N=3:")
    print(f"  Primi vicini:")
    qa_nn3 = QuantumAnnealingHamiltonian.create_test_instance(3, coupling_type='nearest_neighbor')
    print(qa_nn3.J.astype(int))
    print(f"\n  Uniforme (tutti con tutti):")
    qa_un3 = QuantumAnnealingHamiltonian.create_test_instance(3, coupling_type='uniform')
    print(qa_un3.J.astype(int))
    
    # =========================================================================
    # ESEMPIO FINALE: Uso semplice della funzione principale
    # =========================================================================
    print("\n" + "=" * 70)
    print("ESEMPIO FINALE: Uso Semplice")
    print("=" * 70)
    
    print("\n[SIMPLE] Genera 5 stati quantistici con una sola chiamata:")
    print("         states = generate_quantum_states(5)")
    
    states = generate_quantum_states(5)
    
    print(f"\n[SIMPLE] Risultato:")
    print(f"  Shape: {states.shape}")
    print(f"  Tipo: {states.dtype}")
    print(f"  Ogni stato è normalizzato: {all(np.abs(np.linalg.norm(s) - 1) < 1e-10 for s in states)}")
    
    print(f"\n[SIMPLE] I 5 stati (probabilità):")
    for i, s in enumerate(states):
        probs = np.abs(s)**2
        print(f"  Stato {i}: {probs.round(4)}")

