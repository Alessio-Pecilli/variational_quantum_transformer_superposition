# Debug campaign report (single seed)

- data_seed=42, model_seed=1042
- T=16, d=8, T>d=True
- max_epochs=250
- params: QSA angles=12, CSA matrices=128

## Paper plot
`final_loss_vs_k.png` — −log μ vs k + isometric Rényi horizontal (different metric).

## Convergence
- k=1 k-QSA: ep=250 loss=3.3222 [NEEDS MORE]
- k=1 k-CSA: ep=250 loss=1.8212 [NEEDS MORE]
- k=2 k-QSA: ep=250 loss=4.3587 [NEEDS MORE]
- k=2 k-CSA: ep=250 loss=2.5608 [NEEDS MORE]
- k=3 k-QSA: ep=250 loss=4.8501 [NEEDS MORE]
- k=3 k-CSA: ep=250 loss=3.0969 [NEEDS MORE]
- k=4 k-QSA: ep=250 loss=5.3605 [NEEDS MORE]
- k=4 k-CSA: ep=250 loss=3.5627 [NEEDS MORE]

## Recommendations
- k=1 k-QSA: still improving at 250 ep (rel_impr5=3.00e-03); try max_epochs>=500 or lr=5e-4.
- k=2 k-QSA: still improving at 250 ep (rel_impr5=3.50e-03); try max_epochs>=500 or lr=5e-4.
- k=3 k-QSA: still improving at 250 ep (rel_impr5=3.68e-03); try max_epochs>=500 or lr=5e-4.
- k=4 k-QSA: still improving at 250 ep (rel_impr5=3.91e-03); try max_epochs>=500 or lr=5e-4.
- Paper: final_loss_vs_k with isometric_renyi ref; always show param counts.
- Final HPC: n_seeds>=5–8 on −log μ (not cross-arch PPL); nl once per (T,d).
- Do not launch obar until k=3,4 show plateau.

## Plots
- `results\debug_campaign_v3\T16_d8_seed42\final_loss_vs_k.png`
- `results\debug_campaign_v3\T16_d8_seed42\curves_kqsa_vs_kcsa_by_k.png`
- `results\debug_campaign_v3\T16_d8_seed42\nl_ablations_curves.png`
- `results\debug_campaign_v3\T16_d8_seed42\nl_ablations_convergence.png`
