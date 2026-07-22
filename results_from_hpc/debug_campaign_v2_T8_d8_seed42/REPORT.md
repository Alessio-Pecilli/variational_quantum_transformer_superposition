# Debug campaign report (single seed)

- data_seed=42, model_seed=1042
- T=8, d=8, ks=[1, 2, 3, 4]
- max_epochs=200, early_stop=True

## Parametrization
- k-QSA: isometric embedding (vocab×d) + quantum-angle ortho blocks (weights_w, weights_v: layers×log2(d) RY angles each); angles=12
- k-CSA: isometric embedding (vocab×d) + classical QR-orthogonal W, V (learnable W_raw, V_raw: d×d each, orthogonalized via QR; no quantum gate angles); raw params=128
- nl-CSA: embedding_mode=isometric, loss_mode=renyi; arch: embedding → causal softmax attn → residual norm → FFN GELU → tied logits (k-independent)

## nl-CSA architecture
embedding → causal softmax attention → residual/normalization → FFN GELU → output logits (tied to embedding)

## k-independence
nl-CSA is k-independent: same data/model seeds yield identical curves at k=1 and k=4 (max |Δ|=0.000e+00).

## Convergence status
- k=1 k-QSA: ep=200 loss=2.5093 val_ppl=12.2594 [NEEDS MORE]
- k=1 k-CSA: ep=200 loss=1.8751 val_ppl=6.4973 [NEEDS MORE]
- k=2 k-QSA: ep=200 loss=3.3695 val_ppl=28.9436 [NEEDS MORE]
- k=2 k-CSA: ep=200 loss=2.4723 val_ppl=11.8155 [NEEDS MORE]
- k=3 k-QSA: ep=200 loss=3.9225 val_ppl=50.3848 [NEEDS MORE]
- k=3 k-CSA: ep=200 loss=2.9595 val_ppl=19.2525 [NEEDS MORE]
- k=4 k-QSA: ep=200 loss=4.3658 val_ppl=78.6198 [NEEDS MORE]
- k=4 k-CSA: ep=200 loss=3.3733 val_ppl=29.1471 [NEEDS MORE]

## Recommendations
- k=1 k-QSA: still improving at 200 ep (rel_impr5=2.83e-03); use max_epochs>=400 or try lr=5e-4.
- k=2 k-QSA: still improving at 200 ep (rel_impr5=3.52e-03); use max_epochs>=400 or try lr=5e-4.
- k=3 k-QSA: still improving at 200 ep (rel_impr5=4.02e-03); use max_epochs>=400 or try lr=5e-4.
- k=4 k-QSA: still improving at 200 ep (rel_impr5=4.31e-03); use max_epochs>=400 or try lr=5e-4.
- nl-CSA: use ablation 'isometric_renyi' as horizontal reference; run once per (T,d).
- Final baselines: n_seeds=8, report mean±std of val_ppl; skip repeating nl per k.
- Do not launch obar/complexity HPC until baselines converge for k=3,4.

## Plots
- `results\debug_campaign_v2\T8_d8_seed42\val_ppl_vs_k_multiline.png`
- `results\debug_campaign_v2\T8_d8_seed42\val_ppl_vs_k_with_nl_ref.png`
- `results\debug_campaign_v2\T8_d8_seed42\curves_kqsa_vs_kcsa_by_k.png`
- `results\debug_campaign_v2\T8_d8_seed42\nl_ablations_curves.png`
- `results\debug_campaign_v2\T8_d8_seed42\nl_ablations_convergence.png`
