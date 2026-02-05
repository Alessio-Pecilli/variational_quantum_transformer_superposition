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
    'num_sequences': 300,       # N = numero "frasi quantistiche"
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
                    ZZ_op = PauliOperators.two_qubit_op(
                        PauliOperators.Z, PauliOperators.Z, i, j, N
                    )
                    H_ising += self.J[i, j] * ZZ_op
       
        self._H = H_transverse + H_ising
        
        # OTTIMIZZAZIONE MEMORIA: Pre-diagonalizza una volta sola
        print(f"[HAMILTONIAN] Diagonalizzazione {self.dim}×{self.dim} in corso...")
        self._eigenvalues, self._eigenvectors = np.linalg.eigh(self._H)
        print(f"[HAMILTONIAN] ✓ Diagonalizzazione completata e cached")
        
        # Libera la matrice originale per risparmiare memoria
        del self._H
        # Forza garbage collection
        import gc
        gc.collect()
   
    def generate_sequence(
        self,
        num_states: int,
        max_time: float = 1.0,
        initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Genera sequenza di stati via evoluzione unitaria e^{-iHt} OTTIMIZZATA.
        
        Args:
            num_states: Numero di stati ("parole") da generare
            max_time: Tempo massimo di evoluzione
            initial_state: Stato iniziale (default: random)
           
        Returns:
            states: Array (num_states, dim) di PROBABILITÀ |ψ(t)|²
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
       
        # Usa diagonalizzazione PRE-CALCOLATA (cached)
        coeffs = np.conj(self._eigenvectors.T) @ psi0
       
        # Genera stati
        states = np.zeros((num_states, self.dim), dtype=complex)
        for i, t in enumerate(times):
            phase_factors = np.exp(-1j * self._eigenvalues * t)
            states[i] = self._eigenvectors @ (coeffs * phase_factors)
            states[i] = states[i] / np.linalg.norm(states[i])
       
        # Ritorna PROBABILITÀ come in QSAS
        return np.abs(states) ** 2

def generate_quantum_dataset(
    num_sequences: int,
    sequence_length: int,
    source_qubits: int,
    seed: int = 42
) -> np.ndarray:
    """
    Genera dataset di sequenze quantistiche con Hamiltoniana FISSA e stati base senza rimpiazzo.
    """
    rng = np.random.default_rng(seed)
    dim = 2 ** source_qubits
    
    print(f"\n[QSASISO-DATASET] Generazione dataset quantistico:")
    print(f"               - Sequenze: {num_sequences}")
    print(f"               - Lunghezza: {sequence_length}")
    print(f"               - Qubit: {source_qubits} (D = 2^{source_qubits} = {dim})")
    
    # VINCOLO 1: Hamiltoniana FISSA (una sola istanza)
    print(f"[QSASISO-DATASET] Creazione Hamiltoniana FISSA...")
    tfim = TFIMHamiltonian(source_qubits, seed=seed)
    print(f"               ✓ Hamiltoniana creata con J e B fissi per tutte le sequenze")
    
    # VINCOLO 2: Campionamento SENZA RIMPIAZZO
    if num_sequences > dim:
        raise ValueError(f"Impossibile selezionare {num_sequences} stati unici da spazio {dim}-dimensionale")
    
    # Seleziona num_sequences indici UNICI senza rimpiazzo
    selected_indices = rng.choice(dim, size=num_sequences, replace=False)
    print(f"[QSASISO-DATASET] Selezionati {num_sequences} indici unici senza rimpiazzo")
    
    # Genera sequenze
    all_sequences = []
    for i, state_index in enumerate(selected_indices):
        # VINCOLO 3: Stato iniziale ONE-HOT |k⟩
        initial_state = np.zeros(dim, dtype=complex)
        initial_state[state_index] = 1.0
        
        # Logging solo per prime 5 sequenze per performance
        if i < 5:
            binary_str = format(state_index, f'0{source_qubits}b')
            print(f"[QSASISO-DATASET] Sequenza {i+1:3d}/{num_sequences}: "
                  f"Evoluzione stato |{binary_str}⟩ (indice {state_index})")
        elif i == 5:
            print(f"[QSASISO-DATASET] ... (logging silenzioso per performance)")
        
        # Genera sequenza temporale da questo stato base
        sequence = tfim.generate_sequence(
            num_states=sequence_length,
            max_time=float(sequence_length),
            initial_state=initial_state
        )
        all_sequences.append(sequence)
    
    print(f"[QSASISO-DATASET] ✓ Dataset completato: {num_sequences} sequenze × {sequence_length} stati")
    
    # Libera memoria e forza garbage collection
    import gc
    gc.collect()
    
    return np.array(all_sequences)

# ============================================================================
# 2. METRICHE GEOMETRICHE (da CSASISO)
# ============================================================================

def calculate_geometric_overlap(logits: torch.Tensor, targets: torch.Tensor):
    """
    Calcola Loss e PPL seguendo l'approccio geometrico con cosine similarity.
    """
    # Protezione numerica (epsilon)
    eps = 1e-10
    batch_size, seq_len, vocab_size = logits.shape
    
    # Cosine Similarity
    logits_norm = F.normalize(logits, p=2, dim=-1)
    targets_onehot = F.one_hot(targets, num_classes=vocab_size).float()
    targets_norm = F.normalize(targets_onehot, p=2, dim=-1)
    
    cosine_sim = (logits_norm * targets_norm).sum(dim=-1)
    overlap = (1 + cosine_sim) / 2
    f_bar = torch.mean(overlap, dim=1)
    
    # Calcolo Perplexity Geometrica
    ppl_geometric = torch.pow(f_bar + eps, -2)
    loss_geometric = torch.log(ppl_geometric + eps).mean()
    
    return loss_geometric, ppl_geometric.mean().item()

# ============================================================================
# 3. LAYER DI PROIEZIONE QUANTISTICA ISOMETRICA (da QSAS + Isometrie da CSASISO)
# ============================================================================

class QuantumProjectionLayer(nn.Module):
    """
    Proietta stati quantistici da 2^D a embed_dim con condizioni isometriche.
    Equivalente alla matrice P nel codice quantum.
    """
   
    def __init__(self, source_dim: int, target_dim: int):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
       
        # CONDIZIONE ISOMETRICA: Inizializzazione ortogonale della proiezione
        self.projection = nn.Linear(source_dim, target_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, source_dim)
        return: (batch_size, seq_len, target_dim)
        """
        return self.projection(x)

# ============================================================================
# 4. TRANSFORMER COMPONENTS CON CONDIZIONI ISOMETRICHE
# ============================================================================

class StandardLayer(nn.Module):
    """Layer transformer standard con embedding isometrico."""
    def __init__(self, embed_dim, nhead, dropout=0.1):
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

    def forward(self, x, mask):
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(query=x_norm, key=x_norm, value=x_norm, attn_mask=mask)
        x = x + self.dropout(attn_output)
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x

# ============================================================================
# 5. QUANTUM STATES TRANSFORMER ISOMETRICO
# ============================================================================

class QuantumStatesTransformerISO(nn.Module):
    """
    Transformer per stati quantistici con condizioni di range e isometrie per l'embedding.
    Usa la loss geometrica invece di quella quantum.
    
    Pipeline:
        Stati(2^D) -> Proiezione P (Isometrica) -> Embedding + Rotation (Ortogonale) -> Transformer -> Output(num_classes)
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
        self.num_classes = num_classes
       
        # 1. Proiezione quantistica con condizioni isometriche
        self.quantum_projection = QuantumProjectionLayer(source_dim, embed_dim)
       
        # 2. CONDIZIONI ISOMETRICHE: Embedding ortogonale + Rotation matrix
        self.embedding = nn.Embedding(embed_dim, embed_dim)  # Per compatibilità con output
        self.rotation = nn.Parameter(torch.empty(embed_dim, embed_dim))
        
        # Inizializzazione ortogonale sia per embedding che per rotation
        nn.init.orthogonal_(self.embedding.weight)
        nn.init.orthogonal_(self.rotation)
       
        # 3. Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, context_length, embed_dim))
        self.dropout_emb = nn.Dropout(dropout)
       
        # 4. Layer transformer
        self.layers = nn.ModuleList([
            StandardLayer(embed_dim, nhead, dropout=dropout) for _ in range(num_layers)
        ])
        self.norm_f = nn.LayerNorm(embed_dim)
       
        # 5. Output head per classificazione
        self.output_head = nn.Linear(embed_dim, num_classes, bias=False)
        # Inizializzazione ortogonale anche per l'output
        nn.init.orthogonal_(self.output_head.weight)
   
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, source_dim) - Stati quantistici
        return: (batch_size, seq_len, num_classes) - Logits per classificazione
        """
        batch_size, seq_len, _ = x.shape
       
        # 1. Proiezione quantistica (2^D -> embed_dim)
        x_projected = self.quantum_projection(x)  # (batch_size, seq_len, embed_dim)
       
        # 2. Aggiungi positional encoding
        x_embedded = x_projected + self.pos_encoder[:, :seq_len, :]
        x_embedded = self.dropout_emb(x_embedded)
       
        # 3. Causal mask per autoregressive training
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
       
        # 4. Passa attraverso layer transformer
        for layer in self.layers:
            x_embedded = layer(x_embedded, mask)
       
        # 5. Normalizzazione finale
        x_out = self.norm_f(x_embedded)
       
        # 6. CONDIZIONI ISOMETRICHE per output: usa embedding + rotation matrix
        F_matrix = self.embedding.weight @ self.rotation
        
        # Normalizzazione per cosine similarity
        x_norm = F.normalize(x_out, p=2, dim=-1)
        F_norm = F.normalize(F_matrix, p=2, dim=-1)
        
        # Output con proiezione coseno
        output_logits = F.linear(x_norm, F_norm)  # (batch_size, seq_len, embed_dim)
        
        # Mappa finale ai num_classes
        return self.output_head(output_logits)  # (batch_size, seq_len, num_classes)

# ============================================================================
# 6. PREPARAZIONE DATI PER TRAINING
# ============================================================================

def prepare_quantum_data(
    quantum_states: np.ndarray,
    num_classes: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepara dati quantistici per training autoregressive.
    
    Args:
        quantum_states: (num_sequences, seq_len, source_dim)
        num_classes: Numero classi per discretizzazione target
        device: Device PyTorch
    
    Returns:
        x_data: Input sequences (num_sequences, seq_len-1, source_dim)
        y_data: Target sequences (num_sequences, seq_len-1) - class indices
    """
    num_sequences, seq_len, source_dim = quantum_states.shape
    
    # Input: tutti i timestep tranne l'ultimo
    x_data = torch.from_numpy(quantum_states[:, :-1, :]).float().to(device)
    
    # Target: discretizza gli stati quantistici successivi
    next_states = quantum_states[:, 1:, :]  # (num_sequences, seq_len-1, source_dim)
    
    # Discretizzazione: argmax per trovare lo stato di probabilità massima
    y_indices = np.argmax(next_states, axis=-1)  # (num_sequences, seq_len-1)
    
    # Mappa gli indici alle classi desiderate
    y_classes = y_indices % num_classes
    y_data = torch.from_numpy(y_classes).long().to(device)
    
    return x_data, y_data

# ============================================================================
# 7. TRAINING FUNCTIONS
# ============================================================================

def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    x_data: torch.Tensor,
    y_data: torch.Tensor
) -> Tuple[float, float]:
    """Training di una singola epoca con LOSS GEOMETRICA."""
    model.train()
    optimizer.zero_grad()
   
    logits = model(x_data)
    # USA LA LOSS GEOMETRICA invece di quella quantum
    loss, ppl = calculate_geometric_overlap(logits, y_data)
   
    loss.backward()
    optimizer.step()
   
    return loss.item(), ppl

def evaluate_fold(
    model: nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    test_idx: np.ndarray
) -> Tuple[float, float]:
    """Valutazione su un fold con metriche geometriche."""
    model.eval()
    with torch.no_grad():
        x_test = x_data[test_idx]
        y_test = y_data[test_idx]
       
        logits = model(x_test)
        # USA LA LOSS GEOMETRICA invece di quella quantum
        loss, ppl = calculate_geometric_overlap(logits, y_test)
        
        # Error rate
        pred_classes = logits.argmax(dim=-1)
        error_rate = (pred_classes != y_test).float().mean().item()
       
        return ppl, error_rate

# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    """Main function per training QSAS Isometrico."""
    device = torch.device(QSAS_CONFIG['device'])
    cfg = QSAS_CONFIG
    
    print("="*70)
    print("QSAS ISOMETRICO: Quantum States Transformer con Condizioni Isometriche")
    print("="*70)
    print(f"Device: {device}")
    print(f"Configurazione: {cfg}")
    
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
    NUM_CLASSES = cfg.get('num_classes', 16)
    x_data, y_data = prepare_quantum_data(quantum_states, NUM_CLASSES, device)
    print(f"\n[2] Dati preparati:")
    print(f"    - x_data: {x_data.shape}")
    print(f"    - y_data: {y_data.shape}")
    print(f"    - Classi: {NUM_CLASSES}")
   
    # --- 3. CREA MODELLO ---
    source_dim = 2 ** cfg['source_qubits']
   
    model = QuantumStatesTransformerISO(
        source_dim=source_dim,
        embed_dim=cfg['embed_dim'],
        num_classes=NUM_CLASSES,
        nhead=cfg['n_head'],
        num_layers=cfg['num_layers'],
        context_length=cfg['sequence_length'],
        dropout=cfg['dropout']
    ).to(device)
   
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[3] Modello creato:")
    print(f"    - Parametri: {n_params:,}")
    print(f"    - Embedding isometrico: ✓")
    print(f"    - Loss geometrica: ✓")
   
    # --- 4. TRAINING ---
    print("\n" + "="*70)
    print("TRAINING: QuantumStatesTransformerISO (CON CONDIZIONI ISOMETRICHE)")
    print("="*70)
   
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
   
    loss_history = []
    ppl_history = []
   
    for epoch in range(cfg['epochs']):
        loss, ppl = train_epoch(model, optimizer, x_data, y_data)
        loss_history.append(loss)
        ppl_history.append(ppl)
       
        print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | PPL: {ppl:.4f}")
   
    # --- 5. VALUTAZIONE 3-FOLD ---
    print("\n--- VALUTAZIONE 3-FOLD (ISOMETRICO) ---")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_ppl = []
    fold_err = []
   
    for fold_idx, (_, test_idx) in enumerate(kf.split(x_data)):
        ppl, err = evaluate_fold(model, x_data, y_data, test_idx)
        fold_ppl.append(ppl)
        fold_err.append(err)
        print(f"[Fold {fold_idx+1}] PPL: {ppl:.4f} | Error: {err:.4f} ({err*100:.2f}%)")
   
    print(f"\n[FINALE ISOMETRICO] PPL Media: {np.mean(fold_ppl):.4f}")
    print(f"[FINALE ISOMETRICO] Error Medio: {np.mean(fold_err):.4f}")
   
    # --- 6. PLOT RISULTATI ---
    plt.figure(figsize=(15, 5))
   
    plt.subplot(1, 3, 1)
    plt.plot(loss_history, 'b-', linewidth=2, label='Loss Geometrica')
    plt.title('Training Loss (Geometrica)', fontsize=12)
    plt.xlabel('Epoca')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
   
    plt.subplot(1, 3, 2)
    plt.plot(ppl_history, 'r-', linewidth=2, label='PPL Geometrica')
    plt.title('Training Perplexity (Geometrica)', fontsize=12)
    plt.xlabel('Epoca')
    plt.ylabel('PPL')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
   
    plt.subplot(1, 3, 3)
    fold_labels = [f'Fold {i+1}' for i in range(len(fold_ppl))]
    plt.bar(fold_labels, fold_ppl, color='green', alpha=0.7)
    plt.title('Test PPL per Fold', fontsize=12)
    plt.ylabel('PPL')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
   
    plt.tight_layout()
    plt.show()
   
    print("\n" + "="*70)
    print("QSAS ISOMETRICO COMPLETATO!")
    print("="*70)
    
    return {
        'model': model,
        'loss_history': loss_history,
        'ppl_history': ppl_history,
        'fold_results': {
            'ppl_mean': np.mean(fold_ppl),
            'ppl_std': np.std(fold_ppl),
            'error_mean': np.mean(fold_err),
            'error_std': np.std(fold_err)
        }
    }

if __name__ == "__main__":
    results = main()