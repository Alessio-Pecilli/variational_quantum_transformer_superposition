# Note su metriche e gap k-QSA vs k-CSA

## Relative improvement (nl_ablations_convergence / diagnostics)
`
old = mean(loss[t-5 : t-1])
new = loss[t]
rel_impr5 = (old - new) / |old|
`
Positivo = la loss sta ancora scendendo. Usato solo come indicatore di plateau
(insieme a early-stop su |Δ|/|loss| < 1e-4 per 15 epoche consecutive).

## BUG / malinteso sulla PPL (CRITICO)
Nei plot precedenti al_ppl **non era la stessa metrica**:

| Modello | Cosa chiamavamo val_ppl | Significato reale |
|---------|-------------------------|-------------------|
| k-QSA / k-CSA | exp(val) con val = −log μ | **1/μ** (non è LM perplexity) |
| nl-CSA | exp(CE) next-token | **vera perplexity** sul vocabolario (~855) |

Uniform baseline LM: PPL ≈ vocab ≈ 855.
nl-CSA isometrico arriva a ~620 → solo leggermente meglio del random.
I valori ~12–80 di k-QSA/k-CSA sembravano "molto meglio" solo perché 1/μ ≠ PPL.
**Non confrontare queste due quantità.** Il calcolo CE di nl-CSA è coerente;
il problema era presentarli sullo stesso asse come "common metric".

## Perché k-CSA parte più in alto
- Init k-CSA: W_raw, V_raw Gaussiani → QR ≈ ortogonale Haar-like → μ vicino al floor → −log μ alto.
- Init k-QSA: pochi angoli RY (layers×log2(d)=12) → W,V più "strutturati" / meno misti → μ iniziale più alto → −log μ più basso.
Quindi lo start gap è un artefatto di inizializzazione + parametrizzazione, non un bug di training.

## Perché k-CSA scende più in fretta e finisce ≈ kQSA − costante
- **Capacità**: k-CSA ha 2·d² = 128 parametri grezzi (QR → O(d)); k-QSA ha solo 12 angoli RY (sottomanifold di O(d)).
- Stessa loss −log μ e stesso data → landscape simile in forma (da qui il "parallelo" vs k), ma CSA può raggiungere minimi più profondi.
- La discesa rapida: più gradi di libertà + gradiente su matrici piene.
- Il gap sistematico **non** implica che QSA sia "sbagliato": confronta ansatz ristretto (hardware-like) vs ortogonale classico pieno. Va riportato sempre il n° parametri.

## Parametri (T=8, d=8, L=2)
- k-QSA: 12 angoli + embedding
- k-CSA: 128 matrici raw + embedding
- nl-CSA: ~6·L·d·r ≈ 288 (r≈√d) + embedding

## Plot consigliato per il paper (fase attuale)
inal_loss_vs_k.png: −log μ vs k per QSA/CSA, linea orizzontale isometric_renyi
con caption esplicita che Rényi ≠ −log μ (solo riferimento di andamento/scala diversa).

## Setup successivo
Preferibile T > d (advantage). Es. T=16, d=8, ancora single seed in debug.
