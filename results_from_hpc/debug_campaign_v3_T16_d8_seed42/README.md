# Debug campaign v3 (T=16 > d=8, single seed)

Advantage-regime baselines (single seed, not multi-seed finals, no obar).

- T=16, d=8, k=1..4, epochs<=250
- data_seed=42, model_seed=1042
- k-QSA: 12 RY angles; k-CSA: 128 QR matrices
- Paper figure: [final_loss_vs_k.png](final_loss_vs_k.png)
- Notes: [NOTES_METRICS.md](NOTES_METRICS.md) / [REPORT.md](REPORT.md)

**Do not compare** `1/\mu` of mu-models with nl-CSA next-token CE perplexity.
