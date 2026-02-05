# -*- coding: utf-8 -*-
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
NUM_TRANSFORMER_LAYERS = 5
LEARNING_RATE = 0.001
EPOCHS = 100
SENTENCE_LENGTH = 5
NUM_SENTENCES = 100 
TOTAL_NEEDED = NUM_SENTENCES * 2 # 100 per train, 100 per test k-fold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. PREPARAZIONE DATI ---
raw_sentences = []
try:
    with open(PTB_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            if len(words) >= SENTENCE_LENGTH:
                raw_sentences.append(" ".join(words[:SENTENCE_LENGTH]))
except FileNotFoundError:
    print("File ptb_sentences.txt non trovato, uso dati dummy.")
    raw_sentences = ["word " * SENTENCE_LENGTH] * TOTAL_NEEDED

# Campionamento unico per evitare KeyError nel test
if len(raw_sentences) >= TOTAL_NEEDED:
    selected_sentences = random.sample(raw_sentences, TOTAL_NEEDED)
else:
    selected_sentences = raw_sentences
    NUM_SENTENCES = len(selected_sentences) // 2

# Vocabolario costruito su TUTTE le frasi selezionate (Train + Test)
vocab = Counter()
for s in selected_sentences: vocab.update(s.split())
word2id = {w: i+1 for i, (w, c) in enumerate(vocab.items())}
word2id['<pad>'] = 0
VOCAB_SIZE = len(word2id)
print(f"Vocabolario: {VOCAB_SIZE} token.")

def to_tensor(sentences_list):
    ins, tgts = [], []
    for s in sentences_list:
        ids = [word2id[w] for w in s.split()]
        ins.append(ids[:-1])
        tgts.append(ids[1:])
    return torch.tensor(ins).to(device), torch.tensor(tgts).to(device)

# Split netto: 100 per training "normale", 100 per test k-fold
train_subset = selected_sentences[:NUM_SENTENCES]
test_subset = selected_sentences[NUM_SENTENCES:NUM_SENTENCES+100]

x_train, y_train = to_tensor(train_subset)
x_test_pool, y_test_pool = to_tensor(test_subset)

# --- 3. METRICHE ---
def calculate_geometric_overlap(logits: torch.Tensor, targets: torch.Tensor):
    batch_size, seq_len, vocab_size = logits.shape
    logits_norm = F.normalize(logits, p=2, dim=-1)
    targets_onehot = F.one_hot(targets, num_classes=vocab_size).float()
    targets_norm = F.normalize(targets_onehot, p=2, dim=-1)
    
    cosine_sim = (logits_norm * targets_norm).sum(dim=-1)
    overlap = (1 + cosine_sim) / 2
    f_bar = torch.mean(overlap, dim=1)
    
    ppl_geometric = torch.pow(f_bar + 1e-10, -2)
    loss_geometric = torch.log(ppl_geometric + 1e-10).mean()
    return loss_geometric, torch.exp(loss_geometric).item()

# --- 4. MODELLO ---
class StandardLayer(nn.Module):
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

class CosineOutputTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, context_length=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, context_length, embed_dim))
        self.layers = nn.ModuleList([StandardLayer(embed_dim, nhead) for _ in range(num_layers)])
        self.norm_f = nn.LayerNorm(embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        for layer in self.layers:
            x = layer(x, mask)
        x_norm = F.normalize(self.norm_f(x), p=2, dim=-1)
        embed_norm = F.normalize(self.embedding.weight, p=2, dim=-1)
        return F.linear(x_norm, embed_norm)

# --- 5. TRAINING NORMALE ---
model = CosineOutputTransformer(VOCAB_SIZE, EMBED_DIM, N_HEAD, NUM_TRANSFORMER_LAYERS, SENTENCE_LENGTH).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss_history = []
train_ppl_history = []

print(f"Inizio Training Normale su {NUM_SENTENCES} frasi...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    logits = model(x_train)
    loss, ppl = calculate_geometric_overlap(logits, y_train)
    loss.backward()
    optimizer.step()
    
    train_loss_history.append(loss.item())
    train_ppl_history.append(ppl)

    print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | PPL: {ppl:.2f}")

# --- 6. TEST K-FOLD (3-FOLD) SU 100 FRASI NUOVE ---
print(f"\n--- TEST K-FOLD (3-Splits) su 100 frasi di test ---")
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold_ppls = []

model.eval()
with torch.no_grad():
    for fold, (_, test_idx) in enumerate(kf.split(x_test_pool)):
        x_fold = x_test_pool[test_idx]
        y_fold = y_test_pool[test_idx]
        
        _, ppl_fold = calculate_geometric_overlap(model(x_fold), y_fold)
        fold_ppls.append(ppl_fold)
        print(f"Fold {fold+1} Test PPL: {ppl_fold:.4f}")

# --- 7. PLOT ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.title('Training Loss (Normale)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_ppl_history, color='orange', label='Train PPL')
plt.title('Training Perplexity')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"\nMedia PPL sui 3 Test Folds: {np.mean(fold_ppls):.4f}")
