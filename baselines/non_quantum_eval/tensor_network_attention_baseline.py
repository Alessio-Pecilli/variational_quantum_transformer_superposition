import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import KFold
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PTB_FILE = REPO_ROOT / "data" / "ptb_sentences.txt"

# --- 1. CONFIGURAZIONE ---
EMBED_DIM = 4
D_MODEL = 4
N_HEAD = 1
NUM_TRANSFORMER_LAYERS = 3
LEARNING_RATE = 0.001
EPOCHS = 100
SENTENCE_LENGTH = 5
NUM_SENTENCES = 100
TOTAL_NEEDED = NUM_SENTENCES * 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. DATA PREPARATION ---
raw_sentences = []
try:
    with open(PTB_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            if len(words) >= SENTENCE_LENGTH:
                raw_sentences.append(" ".join(words[:SENTENCE_LENGTH]))
except FileNotFoundError:
    raw_sentences = ["word " * SENTENCE_LENGTH] * TOTAL_NEEDED

selected_sentences = random.sample(raw_sentences, min(len(raw_sentences), TOTAL_NEEDED))
vocab = Counter()
for s in selected_sentences: vocab.update(s.split())
word2id = {w: i+1 for i, (w, c) in enumerate(vocab.items())}
word2id['<pad>'] = 0
VOCAB_SIZE = len(word2id)

def to_tensor(sentences_list):
    ins, tgts = [], []
    for s in sentences_list:
        ids = [word2id[w] for w in s.split()]
        ins.append(ids[:-1]); tgts.append(ids[1:])
    return torch.tensor(ins).to(device), torch.tensor(tgts).to(device)

x_data, y_data = to_tensor(selected_sentences[:NUM_SENTENCES])
x_val_final, y_val_final = to_tensor(selected_sentences[NUM_SENTENCES:])

# --- 2.5 GENERAZIONE STATI QUANTISTICI ---
def generate_quantum_states(token_ids, embedding_layer):
    """
    Converte sequenze di token ID in stati quantistici.
    
    Args:
        token_ids: (batch, seq_len) tensor di ID token
        embedding_layer: Layer di embedding per convertire token in vettori
    
    Returns:
        quantum_states: (batch, seq_len, embed_dim * 2) stati quantistici complessi
                        formato: [parte_reale | parte_immaginaria]
    """
    with torch.no_grad():
        # Ottieni embedding dei token
        embeddings = embedding_layer(token_ids)  # (batch, seq_len, embed_dim)
        
        # Normalizza per avere norma unitaria (caratteristica quantistica)
        norm = torch.norm(embeddings, dim=-1, keepdim=True) + 1e-10
        embeddings_normalized = embeddings / norm
        
        # Genera parte immaginaria usando rotazione ortogonale
        # Questo simula una fase quantistica
        phase = torch.randn_like(embeddings_normalized) * 0.1
        
        # Costruisci stato quantistico complesso: |ψ⟩ = Re|ψ⟩ + i·Im|ψ⟩
        real_part = embeddings_normalized
        imag_part = phase
        
        # Concatena Re e Im: [Re | Im]
        quantum_state = torch.cat([real_part, imag_part], dim=-1)
        
        return quantum_state

# --- 3. LOGICA QUANTISTICA (RÉNYI) ---
def calculate_quantum_metrics(logits, targets):
    probs = F.softmax(logits, dim=-1)
    p_true = probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    # Fedeltà media (Quantum Perplexity base)
    f_bar = torch.mean(torch.sqrt(p_true + 1e-10), dim=1)
    ppl_quantum = torch.pow(f_bar, -2)
    loss_quantum = torch.log(ppl_quantum + 1e-10).mean()
    return loss_quantum, ppl_quantum.mean().item()

def orthogonal_penalty(matrix):
    dim = matrix.size(-1)
    identity = torch.eye(dim, device=matrix.device)
    res = torch.mm(matrix.t(), matrix)
    return torch.norm(res - identity)

# --- 4. COMPONENTI TENSOR NETWORK & ATTENTION ---
class NoSoftmaxAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = (embed_dim // num_heads) ** -0.5

    def forward(self, x, mask=None):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == float('-inf'), 0.0)
        # Niente Softmax: Interferenza lineare
        out = torch.matmul(attn_scores, v)
        return self.out_proj(out)

class SeriousLayerNoSoftmax(nn.Module):
    def __init__(self, embed_dim, nhead):
        super().__init__()
        self.self_attn = NoSoftmaxAttention(embed_dim, nhead)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), nn.GELU(), nn.Linear(4*embed_dim, embed_dim))
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        x = x + self.self_attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

# --- 4.5 LAYER DI PROIEZIONE QUANTISTICA (PER STATI QUANTISTICI) ---
class QuantumProjectionLayer(nn.Module):
    """
    Proietta stati quantistici da source_dim a embed_dim.
    Gestisce parte reale e immaginaria separate.
    """
    def __init__(self, source_dim, target_dim):
        super().__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        
        # Proiezione lineare trainabile (parte reale e immaginaria separate)
        self.proj_real = nn.Linear(source_dim, target_dim, bias=False)
        self.proj_imag = nn.Linear(source_dim, target_dim, bias=False)
        
        # Inizializzazione Xavier
        nn.init.xavier_uniform_(self.proj_real.weight)
        nn.init.xavier_uniform_(self.proj_imag.weight)
    
    def forward(self, x_complex):
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


# --- 5. MODELLO TENSOR NETWORK ISOMETRICO ---
class SeriousTransformerTensorNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, source_dim=None):
        """
        Args:
            vocab_size: Dimensione vocabolario per frasi embeddade
            embed_dim: Dimensione embedding
            nhead: Numero di teste attention
            num_layers: Numero di layer transformer
            source_dim: Dimensione stati quantistici (opzionale, se None usa solo embedding)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.source_dim = source_dim
        
        # Embedding per frasi (token IDs) - come CSAS
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.orthogonal_(self.embedding.weight)
        
        # Proiezione per stati quantistici - come QSAS
        if source_dim is not None:
            self.projection = QuantumProjectionLayer(source_dim, embed_dim)
        else:
            self.projection = None

        # Tensor Network Core: Scomposizione della rotazione V in due nodi (MPO-like)
        self.tn_rank = max(2, embed_dim // 2)
        self.tn_core1 = nn.Parameter(torch.empty(embed_dim, self.tn_rank))
        self.tn_core2 = nn.Parameter(torch.empty(self.tn_rank, embed_dim))
        nn.init.orthogonal_(self.tn_core1)
        nn.init.orthogonal_(self.tn_core2)

        self.pos_encoder = nn.Parameter(torch.randn(1, SENTENCE_LENGTH, embed_dim))
        self.layers = nn.ModuleList([SeriousLayerNoSoftmax(embed_dim, nhead) for _ in range(num_layers)])
        self.norm_f = nn.LayerNorm(embed_dim)
        self.fc_out_bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        """
        Args:
            x: Input può essere:
               - (batch, seq_len) per frasi embeddade (token IDs) - come CSAS
               - (batch, seq_len, source_dim * 2) per stati quantistici - come QSAS
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        seq_len = x.size(1)
        
        # Determina il tipo di input basandosi sulla dimensionalità
        if x.dim() == 2:
            # Input: frasi embeddade (token IDs) - usa Embedding come CSAS
            x = self.embedding(x)
        elif x.dim() == 3:
            # Input: stati quantistici - usa Proiezione come QSAS
            if self.projection is None:
                raise ValueError("Modello non configurato per stati quantistici. Impostare source_dim in __init__")
            x = self.projection(x)
        else:
            raise ValueError(f"Input shape non supportato: {x.shape}")
        
        # Aggiungi positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]
        
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(device)
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm_f(x)

        # Contrazione Tensor Network: V = Core1 ⊗ Core2
        # Rappresenta la rotazione isometrica nello spazio di Hilbert
        V_tn = torch.matmul(self.tn_core1, self.tn_core2)
        F_eff = torch.matmul(self.embedding.weight, V_tn)

        return F.linear(x, F_eff, bias=self.fc_out_bias)

# --- 6. TRAINING LOOP ---
# Configurato per accettare stati quantistici come input
model = SeriousTransformerTensorNetwork(
    VOCAB_SIZE, EMBED_DIM, N_HEAD, NUM_TRANSFORMER_LAYERS, 
    source_dim=EMBED_DIM  # Abilita layer di proiezione quantistica
).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Converti dati in stati quantistici
print("Generazione stati quantistici da embedding...")
x_quantum = generate_quantum_states(x_data, model.embedding)
x_val_quantum = generate_quantum_states(x_val_final, model.embedding)

loss_hist, ppl_hist, error_hist = [], [], []

print("Training Tensor Network Model con Stati Quantistici...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    # Input: stati quantistici (batch, seq_len, embed_dim * 2)
    logits = model(x_quantum)
    loss_q, ppl_val = calculate_quantum_metrics(logits, y_data)
    
    # Calcola errore di predizione (come numero 0-1)
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)
        error_rate = (1 - (preds == y_data).float().mean()).item()
    
    # Vincoli di Isometria sui core della Tensor Network
    reg = orthogonal_penalty(model.embedding.weight) + \
          orthogonal_penalty(model.tn_core1) + \
          orthogonal_penalty(model.tn_core2)
    
    total_loss = loss_q + 0.1 * reg
    total_loss.backward()
    optimizer.step()
    
    loss_hist.append(total_loss.item())
    ppl_hist.append(ppl_val)
    error_hist.append(error_rate)
    print(f"Epoch {epoch+1:03d} | Loss: {total_loss.item():.4f} | PPL: {ppl_val:.4f}")

# --- 7. FINAL EVALUATION ---
model.eval()
with torch.no_grad():
    # Check Isometria Finale
    V_final = torch.matmul(model.tn_core1, model.tn_core2)
    is_iso = torch.allclose(V_final.t() @ V_final, torch.eye(EMBED_DIM).to(device), atol=1e-1)
    print(f"\nTensor Network Isometry Check: {'PASSED' if is_iso else 'FAILED'}")

    # Calcola statistiche PPL training
    ppl_mean = np.mean(ppl_hist)
    ppl_std = np.std(ppl_hist)
    
    # Calcola statistiche Error training
    error_mean = np.mean(error_hist)
    error_std = np.std(error_hist)
    
    # Test con stati quantistici
    test_logits = model(x_val_quantum)
    _, final_ppl = calculate_quantum_metrics(test_logits, y_val_final)
    preds = torch.argmax(test_logits, dim=-1)
    acc = (preds == y_val_final).float().mean()
    final_error = (1 - acc).item()
    
    print(f"\n=== TRAINING STATISTICS ===")
    print(f"PPL medio: {ppl_mean:.4f} +/- {ppl_std:.4f}")
    print(f"Error medio: {error_mean:.4f} +/- {error_std:.4f}")
    print(f"\n=== TEST RESULTS ===")
    print(f"Final Test PPL: {final_ppl:.4f}")
    print(f"Final Test Error: {final_error:.4f}")

# --- 8. PLOTS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(loss_hist)
ax1.set_title("TN Training Loss (Quantum States)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(error_hist, color='red')
ax2.set_title("TN Training Error Rate (Quantum States)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Error (0-1)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
