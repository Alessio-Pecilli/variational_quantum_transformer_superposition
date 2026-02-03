import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import KFold
import random

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
    with open('ptb_sentences.txt', 'r', encoding='utf-8') as f:
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

# --- 5. MODELLO TENSOR NETWORK ISOMETRICO ---
class SeriousTransformerTensorNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.orthogonal_(self.embedding.weight)

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
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        
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
model = SeriousTransformerTensorNetwork(VOCAB_SIZE, EMBED_DIM, N_HEAD, NUM_TRANSFORMER_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_hist, ppl_hist = [], []

print("Training Tensor Network Model...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    logits = model(x_data)
    loss_q, ppl_val = calculate_quantum_metrics(logits, y_data)
    
    # Vincoli di Isometria sui core della Tensor Network
    reg = orthogonal_penalty(model.embedding.weight) + \
          orthogonal_penalty(model.tn_core1) + \
          orthogonal_penalty(model.tn_core2)
    
    total_loss = loss_q + 0.1 * reg
    total_loss.backward()
    optimizer.step()
    
    loss_hist.append(total_loss.item())
    ppl_hist.append(ppl_val)
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {total_loss.item():.4f} | PPL: {ppl_val:.4f}")

# --- 7. FINAL EVALUATION ---
model.eval()
with torch.no_grad():
    # Check Isometria Finale
    V_final = torch.matmul(model.tn_core1, model.tn_core2)
    is_iso = torch.allclose(V_final.t() @ V_final, torch.eye(EMBED_DIM).to(device), atol=1e-1)
    print(f"\nTensor Network Isometry Check: {'PASSED' if is_iso else 'FAILED'}")

    test_logits = model(x_val_final)
    _, final_ppl = calculate_quantum_metrics(test_logits, y_val_final)
    preds = torch.argmax(test_logits, dim=-1)
    acc = (preds == y_val_final).float().mean()
    
    print(f"Final Test PPL: {final_ppl:.4f}")
    print(f"Final Test Error: {(1-acc)*100:.2f}%")

plt.plot(loss_hist); plt.title("TN Training Loss"); plt.show()