
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from scipy.linalg import expm

from typing import Optional, Tuple, List



# ============================================================================

# CONFIGURAZIONE STANDALONE (non dipende da config.py)

# ============================================================================

QSAS_CONFIG = {

    # Dimensioni

    'source_dim': 1024,           # 2^D = dimensione stati sorgente (D=6 -> 64)

    'embed_dim': 4,            # Embedding dimension (increased)

    'd_model': 4,              # Dimensione modello transformer

    'n_head': 1,                # Numero teste attention

    'num_layers': 5,            # Numero layer transformer
    # Training

    'learning_rate': 0.001,

    'epochs': 100,

    'dropout': 0,

   

    # Dataset

    'num_sequences': 100,       # N = numero "frasi quantistiche"

    'sequence_length': 5,       # M = "parole" per sequenza (evoluzioni temporali)

    'source_qubits': 10,         # D = qubit Hamiltoniana

    'num_classes': 16,          # Numero classi per target discretization

   

    # Device

    'device': 'cuda' if torch.cuda.is_available() else 'cpu'

}



# ============================================================================

# 1. GENERAZIONE STATI QUANTISTICI (TFIM Hamiltonian)

# ============================================================================



class PauliOperators:

    """Operatori di Pauli per costruzione Hamiltoniana."""

    I = np.array([[1, 0], [0, 1]], dtype=complex)

    X = np.array([[0, 1], [1, 0]], dtype=complex)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)

   

    @staticmethod

    def tensor_product(*matrices):

        result = matrices[0]

        for m in matrices[1:]:

            result = np.kron(result, m)

        return result

   

    @staticmethod

    def single_qubit_op(op, qubit_idx, num_qubits):

        ops = [PauliOperators.I] * num_qubits

        ops[qubit_idx] = op

        return PauliOperators.tensor_product(*ops)

   

    @staticmethod

    def two_qubit_op(op1, op2, idx1, idx2, num_qubits):

        ops = [PauliOperators.I] * num_qubits

        ops[idx1] = op1

        ops[idx2] = op2

        return PauliOperators.tensor_product(*ops)





class TFIMHamiltonian:

    """

    Transverse Field Ising Model Hamiltonian.

    H = Σᵢ Xᵢ + Σ⟨i,j⟩ Jᵢⱼ ZᵢZⱼ

    """

   

    def __init__(self, num_qubits: int, seed: Optional[int] = None):

        self.num_qubits = num_qubits

        self.dim = 2 ** num_qubits

       

        # Genera matrice di accoppiamento J ~ Uniform[0,1]

        rng = np.random.default_rng(seed)

        self.J = np.zeros((num_qubits, num_qubits))

        for i in range(num_qubits):

            for j in range(i + 1, num_qubits):

                self.J[i, j] = rng.uniform(0, 1)

                self.J[j, i] = self.J[i, j]

       

        # Costruisci Hamiltoniana

        self._build_hamiltonian()

   

    def _build_hamiltonian(self):

        N = self.num_qubits

       

        # Termine campo trasverso: Σᵢ Xᵢ

        H_transverse = np.zeros((self.dim, self.dim), dtype=complex)

        for i in range(N):

            H_transverse += PauliOperators.single_qubit_op(PauliOperators.X, i, N)

       

        # Termine Ising: Σ⟨i,j⟩ Jᵢⱼ ZᵢZⱼ

        H_ising = np.zeros((self.dim, self.dim), dtype=complex)

        for i in range(N):

            for j in range(i + 1, N):

                if self.J[i, j] != 0:

                    ZZ = PauliOperators.two_qubit_op(

                        PauliOperators.Z, PauliOperators.Z, i, j, N

                    )

                    H_ising += self.J[i, j] * ZZ

       

        self._H = H_transverse + H_ising

   

    def generate_sequence(self, num_states: int, max_time: float = 10.0,

                          initial_state: Optional[np.ndarray] = None) -> np.ndarray:

        """

        Genera sequenza di stati via evoluzione unitaria e^{-iHt}.

       

        Args:

            num_states: Numero di stati ("parole") da generare

            max_time: Tempo massimo di evoluzione

            initial_state: Stato iniziale (default: random)

           

        Returns:

            states: Array (num_states, dim) di stati quantistici

        """

        # Stato iniziale

        if initial_state is None:

            # Random normalizzato

            psi0 = np.random.randn(self.dim) + 1j * np.random.randn(self.dim)

            psi0 = psi0 / np.linalg.norm(psi0)

        else:

            psi0 = initial_state / np.linalg.norm(initial_state)

       

        # Tempi

        times = np.linspace(0, max_time, num_states)

       

        # Diagonalizza per evoluzione efficiente

        eigenvalues, eigenvectors = np.linalg.eigh(self._H)

        coeffs = np.conj(eigenvectors.T) @ psi0

       

        # Genera stati

        states = np.zeros((num_states, self.dim), dtype=complex)

        for i, t in enumerate(times):

            phase_factors = np.exp(-1j * eigenvalues * t)

            states[i] = eigenvectors @ (coeffs * phase_factors)

            states[i] = states[i] / np.linalg.norm(states[i])

       

        return states





def generate_quantum_dataset(
    num_sequences: int,
    sequence_length: int,
    source_qubits: int,
    seed: int = 42
) -> np.ndarray:
    """
    Genera dataset di sequenze quantistiche con Hamiltoniana FISSA e stati base senza rimpiazzo.
    
    Vincoli rigorosi:
    - Hamiltoniana fissa: una sola istanza TFIMHamiltonian per tutte le sequenze
    - Campionamento senza rimpiazzo: num_sequences indici UNICI da 0 a 2^source_qubits - 1
    - Stati iniziali: vettori one-hot |k⟩ (stati base computazionali)
    - Logging: stampa indice e notazione binaria
   
    Args:
        num_sequences: Numero di "frasi" quantistiche
        sequence_length: Numero di "parole" per frase
        source_qubits: Numero di qubit Hamiltoniana
        seed: Seed per riproducibilità
       
    Returns:
        dataset: Array (num_sequences, sequence_length, 2^source_qubits)
    """
    rng = np.random.default_rng(seed)
    dim = 2 ** source_qubits
    
    print(f"\n[QSAS-DATASET] Generazione dataset quantistico:")
    print(f"               - Sequenze: {num_sequences}")
    print(f"               - Lunghezza: {sequence_length}")
    print(f"               - Qubit: {source_qubits} (D = 2^{source_qubits} = {dim})")
    
    # VINCOLO 1: Hamiltoniana FISSA (una sola istanza)
    print(f"[QSAS-DATASET] Creazione Hamiltoniana FISSA...")
    tfim = TFIMHamiltonian(source_qubits, seed=seed)
    print(f"               ✓ Hamiltoniana creata con J e B fissi per tutte le sequenze")
    
    # VINCOLO 2: Campionamento SENZA RIMPIAZZO
    if num_sequences > dim:
        raise ValueError(f"Impossibile selezionare {num_sequences} stati unici da spazio {dim}-dimensionale")
    
    # Seleziona num_sequences indici UNICI senza rimpiazzo
    selected_indices = rng.choice(dim, size=num_sequences, replace=False)
    print(f"[QSAS-DATASET] Selezionati {num_sequences} indici unici senza rimpiazzo")
    
    # Genera sequenze
    all_sequences = []
    for i, state_index in enumerate(selected_indices):
        # VINCOLO 3: Stato iniziale ONE-HOT |k⟩
        initial_state = np.zeros(dim, dtype=complex)
        initial_state[state_index] = 1.0
        
        # Converti indice in notazione binaria per logging
        binary_str = format(state_index, f'0{source_qubits}b')
        
        # VINCOLO 4: Logging con indice e notazione binaria
        print(f"[QSAS-DATASET] Sequenza {i+1:3d}/{num_sequences}: "
              f"Evoluzione stato |{binary_str}⟩ (indice {state_index})")
        
        # Genera sequenza temporale da questo stato base
        sequence = tfim.generate_sequence(
            num_states=sequence_length,
            max_time=float(sequence_length),
            initial_state=initial_state
        )
        all_sequences.append(sequence)
    
    print(f"[QSAS-DATASET] ✓ Dataset completato: {num_sequences} sequenze × {sequence_length} stati")
    return np.array(all_sequences)





# ============================================================================

# 2. LAYER DI PROIEZIONE (2^D -> embed_dim)

# ============================================================================



class QuantumProjectionLayer(nn.Module):

    """

    Proietta stati quantistici da 2^D a embed_dim.

    Equivalente alla matrice P nel codice quantum.

    """

   

    def __init__(self, source_dim: int, target_dim: int):

        super().__init__()

        self.source_dim = source_dim

        self.target_dim = target_dim

       

        # Proiezione lineare trainabile (parte reale e immaginaria separate)

        # P: (target_dim, source_dim) - ma lavoriamo con Re+Im concatenati

        self.proj_real = nn.Linear(source_dim, target_dim, bias=False)

        self.proj_imag = nn.Linear(source_dim, target_dim, bias=False)

       

        # Inizializzazione Xavier

        nn.init.xavier_uniform_(self.proj_real.weight)

        nn.init.xavier_uniform_(self.proj_imag.weight)

   

    def forward(self, x_complex: torch.Tensor) -> torch.Tensor:

        """

        Args:

            x_complex: (batch, seq_len, source_dim * 2) - Re e Im concatenati

           

        Returns:

            embedding: (batch, seq_len, target_dim)

        """

        # Split in parte reale e immaginaria

        x_real = x_complex[..., :self.source_dim]

        x_imag = x_complex[..., self.source_dim:]

       

        # Proiezione (P @ state per Re e Im)

        out_real = self.proj_real(x_real) - self.proj_imag(x_imag)  # Re(P @ z)

        out_imag = self.proj_real(x_imag) + self.proj_imag(x_real)  # Im(P @ z)

       

        # Calcola modulo per ogni componente

        out_magnitude = torch.sqrt(out_real**2 + out_imag**2 + 1e-10)

       

        # Normalizza per avere embedding di norma unitaria

        norm = torch.norm(out_magnitude, dim=-1, keepdim=True) + 1e-10

        return out_magnitude / norm





# ============================================================================

# 3. METRICHE QUANTUM (CON TARGET SUPERVISIONATO - COME CSAS!)

# ============================================================================



def calculate_quantum_metrics(logits: torch.Tensor, targets: torch.Tensor):

    """

    Calcola Loss e PPL seguendo la Perplexity di Rényi (fedeltà media).

    """

    probs = F.softmax(logits, dim=-1)

    p_true = probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)

   

    # Lunghezza sequenza T

    T = p_true.size(1)

   

    # 1. Calcoliamo la fedeltà media (F_bar): media delle radici quadrate delle probabilità

    # F_bar = (1/T) * sum(sqrt(p))

    f_bar = torch.mean(torch.sqrt(p_true + 1e-10), dim=1)

   

    # 2. PPL = F_bar^-2

    ppl_quantum = torch.pow(f_bar, -2)

   

    # 3. Loss = log(PPL) = -2 * log(f_bar)

    loss_quantum = torch.log(ppl_quantum + 1e-10).mean()

   

    # PPL media per il log

    true_geometric_ppl = torch.exp(loss_quantum).item()

   

    return loss_quantum, true_geometric_ppl





# ============================================================================

# 4. ARCHITETTURA TRANSFORMER

# ============================================================================



# --- 4.1 Attention Senza Softmax ---

class NoSoftmaxAttention(nn.Module):

    """

    MultiHeadAttention SENZA Softmax sui coefficienti.

    Permette interferenza negativa (distruttiva).

    """

   

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):

        super().__init__()

        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        self.scale = self.head_dim ** -0.5

       

        assert self.head_dim * num_heads == embed_dim

       

        self.q_proj = nn.Linear(embed_dim, embed_dim)

        self.k_proj = nn.Linear(embed_dim, embed_dim)

        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

   

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape

       

        q = self.q_proj(x)

        k = self.k_proj(x)

        v = self.v_proj(x)

       

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

       

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

       

        if mask is not None:

            attn_scores = attn_scores.masked_fill(mask == float('-inf'), 0.0)

       

        # NIENTE SOFTMAX!

        attn_probs = self.dropout(attn_scores)

       

        out = attn_probs @ v

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

       

        return self.out_proj(out)





# --- 4.2 Layer Transformer (Con Softmax) ---

class QuantumStatesLayer(nn.Module):

    """Layer transformer standard con softmax."""

   

    def __init__(self, embed_dim: int, nhead: int, dropout: float = 0.1):

        super().__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True, dropout=dropout)

       

        self.ffn = nn.Sequential(

            nn.Linear(embed_dim, 4 * embed_dim),

            nn.GELU(),

            nn.Linear(4 * embed_dim, embed_dim),

            nn.Dropout(dropout)

        )

       

        self.norm1 = nn.LayerNorm(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

   

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        x_norm = self.norm1(x)

        attn_output, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm, attn_mask=mask)

        x = x + self.dropout(attn_output)

       

        x_norm = self.norm2(x)

        ffn_output = self.ffn(x_norm)

        x = x + ffn_output

       

        return x





# --- 4.3 Layer Transformer (Senza Softmax) ---

class QuantumStatesLayerNoSoftmax(nn.Module):

    """Layer transformer SENZA softmax nell'attention."""

   

    def __init__(self, embed_dim: int, nhead: int, dropout: float = 0.1):

        super().__init__()

        self.self_attn = NoSoftmaxAttention(embed_dim, nhead, dropout=dropout)

       

        self.ffn = nn.Sequential(

            nn.Linear(embed_dim, 4 * embed_dim),

            nn.GELU(),

            nn.Linear(4 * embed_dim, embed_dim),

            nn.Dropout(dropout)

        )

       

        self.norm1 = nn.LayerNorm(embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

   

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        x_norm = self.norm1(x)

        attn_output = self.self_attn(x_norm, mask=mask)

        x = x + self.dropout(attn_output)

       

        x_norm = self.norm2(x)

        ffn_output = self.ffn(x_norm)

        x = x + ffn_output

       

        return x





# --- 4.4 Modello Completo (Con Softmax) ---

class QuantumStatesTransformer(nn.Module):

    """

    Transformer per stati quantistici CON softmax nell'attention.

   

    Pipeline:

        Stati(2^D) -> Proiezione P -> Transformer -> Output(num_classes)

    """

   

    def __init__(

        self,

        source_dim: int,

        embed_dim: int,

        num_classes: int,

        nhead: int = 1,

        num_layers: int = 5,

        context_length: int = 100,

        dropout: float = 0.1

    ):

        super().__init__()

       

        self.source_dim = source_dim

        self.embed_dim = embed_dim

       

        # Proiezione da 2^D a embed_dim

        self.projection = QuantumProjectionLayer(source_dim, embed_dim)

       

        # Positional Encoding

        self.pos_encoder = nn.Parameter(torch.randn(1, context_length, embed_dim))

        self.dropout_emb = nn.Dropout(dropout)

       

        # Stack di layer transformer

        self.layers = nn.ModuleList([

            QuantumStatesLayer(embed_dim, nhead, dropout=dropout) for _ in range(num_layers)

        ])

       

        # Output

        self.norm_f = nn.LayerNorm(embed_dim)

        self.fc_out = nn.Linear(embed_dim, num_classes)

   

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """

        Args:

            x: (batch, seq_len, source_dim * 2) - Stati quantistici (Re+Im concatenati)

           

        Returns:

            logits: (batch, seq_len, num_classes)

        """

        seq_len = x.size(1)

       

        # Proiezione: 2^D -> embed_dim

        x = self.projection(x)

       

        # Positional encoding

        x = x + self.pos_encoder[:, :seq_len, :]

        x = self.dropout_emb(x)

       

        # Causal mask

        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)

       

        # Transformer layers

        for layer in self.layers:

            x = layer(x, mask)

       

        x = self.norm_f(x)

        return self.fc_out(x)





# --- 4.5 Modello Completo (Senza Softmax) ---

class QuantumStatesTransformerNoSoftmax(nn.Module):

    """

    Transformer per stati quantistici SENZA softmax nell'attention.

    Permette interferenza negativa (distruttiva).

    """

   

    def __init__(

        self,

        source_dim: int,

        embed_dim: int,

        num_classes: int,

        nhead: int = 1,

        num_layers: int = 5,

        context_length: int = 100,

        dropout: float = 0.1

    ):

        super().__init__()

       

        self.source_dim = source_dim

        self.embed_dim = embed_dim

       

        # Proiezione da 2^D a embed_dim

        self.projection = QuantumProjectionLayer(source_dim, embed_dim)

       

        # Positional Encoding

        self.pos_encoder = nn.Parameter(torch.randn(1, context_length, embed_dim))

        self.dropout_emb = nn.Dropout(dropout)

       

        # Stack di layer transformer SENZA SOFTMAX

        self.layers = nn.ModuleList([

            QuantumStatesLayerNoSoftmax(embed_dim, nhead, dropout=dropout) for _ in range(num_layers)

        ])

       

        # Output

        self.norm_f = nn.LayerNorm(embed_dim)

        self.fc_out = nn.Linear(embed_dim, num_classes)

   

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """

        Args:

            x: (batch, seq_len, source_dim * 2) - Stati quantistici (Re+Im concatenati)

           

        Returns:

            logits: (batch, seq_len, num_classes)

        """

        seq_len = x.size(1)

       

        # Proiezione: 2^D -> embed_dim

        x = self.projection(x)

       

        # Positional encoding

        x = x + self.pos_encoder[:, :seq_len, :]

        x = self.dropout_emb(x)

       

        # Causal mask

        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)

       

        # Transformer layers

        for layer in self.layers:

            x = layer(x, mask)

       

        x = self.norm_f(x)

        return self.fc_out(x)





# ============================================================================

# 5. FUNZIONI DI TRAINING E VALUTAZIONE

# ============================================================================



def prepare_quantum_data(

    quantum_states: np.ndarray,

    num_classes: int,

    device: torch.device

) -> Tuple[torch.Tensor, torch.Tensor]:

    """

    Prepara dati quantistici per training SUPERVISIONATO.

   

    Usa energy-based binning per discretizzazione target:

    - Calcola "energia" come somma pesata di |ψ|²

    - Usa fase della componente dominante

    - Combina per creare classi non-triviali

   

    Args:

        quantum_states: (num_seq, seq_len, dim) array complesso

        num_classes: Numero di classi per classificazione

        device: Torch device

       

    Returns:

        x_data: (num_seq, seq_len-1, dim*2) - Input (Re+Im concatenati)

        y_data: (num_seq, seq_len-1) - Target (indice classe)

    """

    num_seq, seq_len, dim = quantum_states.shape

   

    # Compute probability distribution for each state

    probs = np.abs(quantum_states) ** 2  # |ψ|²

   

    # Energy-like feature: weighted position (like expectation value)

    positions = np.arange(dim)

    energy = np.sum(probs * positions, axis=-1)  # (num_seq, seq_len)

   

    # Add phase information for more diversity

    # Use the phase of the dominant component

    dominant_idx = np.argmax(np.abs(quantum_states), axis=-1)

    phases = np.zeros((num_seq, seq_len))

    for i in range(num_seq):

        for j in range(seq_len):

            phases[i, j] = np.angle(quantum_states[i, j, dominant_idx[i, j]])

   

    # Combine energy and phase into a single feature

    # Normalize energy to [0, 1] range

    energy_norm = (energy - energy.min()) / (energy.max() - energy.min() + 1e-10)

    # Normalize phase to [0, 1] range

    phase_norm = (phases + np.pi) / (2 * np.pi)

   

    # Combined feature with more weight on energy

    combined = 0.7 * energy_norm + 0.3 * phase_norm

   

    # Bin into classes

    bins = np.linspace(0, 1, num_classes + 1)

    classes = np.digitize(combined, bins) - 1

    classes = np.clip(classes, 0, num_classes - 1)

   

    # Input: tutti tranne l'ultimo (come nel language modeling)

    x_input = quantum_states[:, :-1, :]

   

    # Target: classi shiftate di 1 (predici il prossimo stato)

    y_target = classes[:, 1:]

   

    # Converti in tensori (Re+Im concatenati per input)

    x_real = np.real(x_input)

    x_imag = np.imag(x_input)

    x_concat = np.concatenate([x_real, x_imag], axis=-1)

   

    x_data = torch.tensor(x_concat, dtype=torch.float32).to(device)

    y_data = torch.tensor(y_target, dtype=torch.long).to(device)

   

    # Print class distribution for debugging

    unique, counts = np.unique(y_target, return_counts=True)

    print(f"    Distribuzione classi: {dict(zip(unique, counts))}")

   

    return x_data, y_data





def train_epoch(

    model: nn.Module,

    optimizer: optim.Optimizer,

    x_data: torch.Tensor,

    y_data: torch.Tensor

) -> Tuple[float, float]:

    """Training di una singola epoca (SUPERVISIONATO)."""

    model.train()

    optimizer.zero_grad()

   

    logits = model(x_data)

    loss, ppl = calculate_quantum_metrics(logits, y_data)

   

    loss.backward()

    optimizer.step()

   

    return loss.item(), ppl





def evaluate_fold(

    model: nn.Module,

    x_data: torch.Tensor,

    y_data: torch.Tensor,

    test_idx: np.ndarray

) -> Tuple[float, float]:

    """Valutazione su un fold (PPL quantum + Error Rate)."""

    model.eval()

    with torch.no_grad():

        x_test = x_data[test_idx]

        y_test = y_data[test_idx]

       

        logits = model(x_test)

       

        # PPL (quantum)

        _, ppl = calculate_quantum_metrics(logits, y_test)

       

        # Error rate (classico)

        pred = torch.argmax(logits, dim=-1)

        correct = (pred == y_test).float().sum()

        total = y_test.numel()

        err = 1.0 - (correct / total).item()

       

        return ppl, err





# ============================================================================

# 6. MAIN - TRAINING COMPLETO

# ============================================================================



def main():

    print("="*70)

    print("QSAS - Quantum States Attention Sequence")

    print("="*70)

   

    cfg = QSAS_CONFIG

    device = torch.device(cfg['device'])

    print(f"Device: {device}")

   

    # --- 1. GENERA DATASET QUANTISTICO ---

    print(f"\n[1] Generazione dataset quantistico...")

    print(f"    - Sequenze: {cfg['num_sequences']}")

    print(f"    - Lunghezza: {cfg['sequence_length']}")

    print(f"    - Qubit sorgente: {cfg['source_qubits']} (dim = {2**cfg['source_qubits']})")

   

    quantum_states = generate_quantum_dataset(

        num_sequences=cfg['num_sequences'],

        sequence_length=cfg['sequence_length'],

        source_qubits=cfg['source_qubits'],

        seed=42

    )

    print(f"    ✓ Dataset generato: shape = {quantum_states.shape}")

   

    # --- 2. PREPARA DATI ---

    NUM_CLASSES = cfg.get('num_classes', 16)  # Use config num_classes

    x_data, y_data = prepare_quantum_data(quantum_states, NUM_CLASSES, device)

    print(f"\n[2] Dati preparati:")

    print(f"    - x_data: {x_data.shape}")

    print(f"    - y_data: {y_data.shape}")

    print(f"    - Classi: {NUM_CLASSES}")

   

    # --- 3. CREA MODELLI ---

    source_dim = 2 ** cfg['source_qubits']

   

    model_softmax = QuantumStatesTransformer(

        source_dim=source_dim,

        embed_dim=cfg['embed_dim'],

        num_classes=NUM_CLASSES,

        nhead=cfg['n_head'],

        num_layers=cfg['num_layers'],

        context_length=cfg['sequence_length'],

        dropout=cfg['dropout']

    ).to(device)

   

    model_no_softmax = QuantumStatesTransformerNoSoftmax(

        source_dim=source_dim,

        embed_dim=cfg['embed_dim'],

        num_classes=NUM_CLASSES,

        nhead=cfg['n_head'],

        num_layers=cfg['num_layers'],

        context_length=cfg['sequence_length'],

        dropout=cfg['dropout']

    ).to(device)

   

    n_params = sum(p.numel() for p in model_softmax.parameters())

    print(f"\n[3] Modelli creati:")

    print(f"    - Parametri: {n_params:,}")

   

    # =========================================================================

    # TRAINING CON SOFTMAX

    # =========================================================================

    print("\n" + "="*70)

    print("TRAINING: QuantumStatesTransformer (CON SOFTMAX)")

    print("="*70)

   

    optimizer = optim.Adam(model_softmax.parameters(), lr=cfg['learning_rate'])

   

    loss_history_softmax = []

    ppl_history_softmax = []

   

    for epoch in range(cfg['epochs']):

        loss, ppl = train_epoch(model_softmax, optimizer, x_data, y_data)

        loss_history_softmax.append(loss)

        ppl_history_softmax.append(ppl)

       

        if (epoch + 1) % 10 == 0 or epoch == 0:

            print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | PPL: {ppl:.4f}")

   

    # --- 3-Fold CV ---

    print("\n--- VALUTAZIONE 3-FOLD (CON SOFTMAX) ---")

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    fold_ppl_softmax = []

    fold_err_softmax = []

   

    for fold_idx, (_, test_idx) in enumerate(kf.split(x_data)):

        ppl, err = evaluate_fold(model_softmax, x_data, y_data, test_idx)

        fold_ppl_softmax.append(ppl)

        fold_err_softmax.append(err)

        print(f"[Fold {fold_idx+1}] PPL: {ppl:.4f} | Error: {err:.4f} ({err*100:.2f}%)")

   

    print(f"\n[FINALE CON SOFTMAX] PPL Media: {np.mean(fold_ppl_softmax):.4f}")

    print(f"[FINALE CON SOFTMAX] Error Medio: {np.mean(fold_err_softmax):.4f}")

   

    # =========================================================================

    # TRAINING SENZA SOFTMAX

    # =========================================================================

    print("\n" + "="*70)

    print("TRAINING: QuantumStatesTransformerNoSoftmax (SENZA SOFTMAX)")

    print("="*70)

   

    optimizer = optim.Adam(model_no_softmax.parameters(), lr=cfg['learning_rate'])

   

    loss_history_no_softmax = []

    ppl_history_no_softmax = []

   

    for epoch in range(cfg['epochs']):

        loss, ppl = train_epoch(model_no_softmax, optimizer, x_data, y_data)

        loss_history_no_softmax.append(loss)

        ppl_history_no_softmax.append(ppl)

       

        if (epoch + 1) % 10 == 0 or epoch == 0:

            print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | PPL: {ppl:.4f}")

   

    # --- 3-Fold CV ---

    print("\n--- VALUTAZIONE 3-FOLD (SENZA SOFTMAX) ---")

    fold_ppl_no_softmax = []

    fold_err_no_softmax = []

   

    for fold_idx, (_, test_idx) in enumerate(kf.split(x_data)):

        ppl, err = evaluate_fold(model_no_softmax, x_data, y_data, test_idx)

        fold_ppl_no_softmax.append(ppl)

        fold_err_no_softmax.append(err)

        print(f"[Fold {fold_idx+1}] PPL: {ppl:.4f} | Error: {err:.4f} ({err*100:.2f}%)")

   

    print(f"\n[FINALE SENZA SOFTMAX] PPL Media: {np.mean(fold_ppl_no_softmax):.4f}")

    print(f"[FINALE SENZA SOFTMAX] Error Medio: {np.mean(fold_err_no_softmax):.4f}")

   

    # =========================================================================

    # CONFRONTO E PLOT

    # =========================================================================

    print("\n" + "="*70)

    print("CONFRONTO FINALE")

    print("="*70)

    print(f"{'Modello':<30} | {'PPL Media':>12} | {'Error Medio':>12}")

    print("-"*60)

    print(f"{'Con Softmax':<30} | {np.mean(fold_ppl_softmax):>12.4f} | {np.mean(fold_err_softmax):>12.4f}")

    print(f"{'Senza Softmax':<30} | {np.mean(fold_ppl_no_softmax):>12.4f} | {np.mean(fold_err_no_softmax):>12.4f}")

   

    # --- Plot ---

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

   

    # Loss

    axes[0, 0].plot(loss_history_softmax, label='Con Softmax', alpha=0.8)

    axes[0, 0].plot(loss_history_no_softmax, label='Senza Softmax', alpha=0.8)

    axes[0, 0].set_title('Training Loss')

    axes[0, 0].set_xlabel('Epoch')

    axes[0, 0].set_ylabel('Loss')

    axes[0, 0].legend()

    axes[0, 0].grid(True, alpha=0.3)

   

    # PPL

    axes[0, 1].plot(ppl_history_softmax, label='Con Softmax', alpha=0.8)

    axes[0, 1].plot(ppl_history_no_softmax, label='Senza Softmax', alpha=0.8)

    axes[0, 1].set_title('Training PPL')

    axes[0, 1].set_xlabel('Epoch')

    axes[0, 1].set_ylabel('PPL')

    axes[0, 1].legend()

    axes[0, 1].grid(True, alpha=0.3)

   

    # PPL per Fold

    x_folds = [1, 2, 3]

    width = 0.35

    axes[1, 0].bar([x - width/2 for x in x_folds], fold_ppl_softmax, width, label='Con Softmax')

    axes[1, 0].bar([x + width/2 for x in x_folds], fold_ppl_no_softmax, width, label='Senza Softmax')

    axes[1, 0].set_title('PPL per Fold (3-Fold CV)')

    axes[1, 0].set_xlabel('Fold')

    axes[1, 0].set_ylabel('PPL')

    axes[1, 0].set_xticks(x_folds)

    axes[1, 0].legend()

    axes[1, 0].grid(True, alpha=0.3)

   

    # Error per Fold

    axes[1, 1].bar([x - width/2 for x in x_folds], fold_err_softmax, width, label='Con Softmax')

    axes[1, 1].bar([x + width/2 for x in x_folds], fold_err_no_softmax, width, label='Senza Softmax')

    axes[1, 1].set_title('Error Rate per Fold (3-Fold CV)')

    axes[1, 1].set_xlabel('Fold')

    axes[1, 1].set_ylabel('Error Rate')

    axes[1, 1].set_xticks(x_folds)

    axes[1, 1].legend()

    axes[1, 1].grid(True, alpha=0.3)

   

    plt.tight_layout()

    plt.savefig('qsas_results.png', dpi=150)

    plt.show()

   

    print("\n✓ Plot salvato in: qsas_results.png")

    print("\n" + "="*70)

    print("QSAS COMPLETATO")

    print("="*70)





if __name__ == "__main__":

    main()