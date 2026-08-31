# qsa_bench_25_08 — results pack

Hybrid readout (`xi` on device, `alpha=||x||^2` in measurement), **trainable PTB embedding**, train on **L_B**, report **L1_excess / LB_excess** and **circuit mu**.

## Campaigns

| Campaign | Job | Status | Output |
|----------|-----|--------|--------|
| Excess L1/LB vs k (classical+quantum) | 54131638 | COMPLETED | `LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/` |
| mu vs T / mu vs d | 54577147 | COMPLETED (72 cells) | `mu_T32_d16_ks2-5_n10/` |

Param match: CSA/nl **1024**, QSA **1032** (L=43).

## Plots

### Excess vs k
- [classical train L1_excess](LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/train_L1_excess_vs_k.png)
- [classical test L1_excess](LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/test_L1_excess_vs_k.png)
- [classical train LB_excess](LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/train_LB_excess_vs_k.png)
- [classical test LB_excess](LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/classical/plots/test_LB_excess_vs_k.png)
- [quantum train L1_excess](LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/train_L1_excess_vs_k.png)
- [quantum test L1_excess](LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/test_L1_excess_vs_k.png)
- [quantum train LB_excess](LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/train_LB_excess_vs_k.png)
- [quantum test LB_excess](LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/quantum/plots/test_LB_excess_vs_k.png)

### mu (circuit observable, k=2,5, mono+poly)
- [mu vs T (d=16)](mu_T32_d16_ks2-5_n10/mu_vs_T.png)
- [mu vs d (T=32)](mu_T32_d16_ks2-5_n10/mu_vs_d.png)

Email draft: [EMAIL_PROF.md](EMAIL_PROF.md)
