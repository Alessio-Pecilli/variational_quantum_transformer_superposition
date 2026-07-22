# Debug campaign v2 (single seed)

Exploratory baselines before HPC finals. **Not** multi-seed; **no** obar campaign.

- T=8, d=8, k=1..4
- data_seed=42, model_seed=1042, n_seeds=1
- k-QSA independent from k-CSA (QR classical W/V)
- nl-CSA: 4 ablations, run once (k-independent)
- Common metric: validation perplexity

## Main figures
- [baselines_summary_3panel.png](baselines_summary_3panel.png)
- [curves_kqsa_vs_kcsa_by_k.png](curves_kqsa_vs_kcsa_by_k.png)
- [val_ppl_vs_k_with_nl_ref.png](val_ppl_vs_k_with_nl_ref.png)
- [nl_ablations_curves.png](nl_ablations_curves.png)

Full report: [REPORT.md](REPORT.md) / [REPORT.json](REPORT.json)
