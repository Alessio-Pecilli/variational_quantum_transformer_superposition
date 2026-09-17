# qsa_bench_25_08 — results pack

Hybrid readout (`xi` on device, `alpha=||x||^2` in measurement), **trainable PTB embedding**, train on **L_B**, report **L1_excess / LB_excess** and **circuit mu**.

## Campaigns

| Campaign | Jobs | Status | Output |
|----------|------|--------|--------|
| Excess L1/LB vs k (classical+quantum) | 54131638 | COMPLETED | `LB_excess_T32_d16_ks1-2-3-4-5-6_LmatchCSA_n8/` |
| mu vs T / mu vs d (v1, old adv / mixed QSA+CSA curves) | 54577147 | COMPLETED (72 cells) | `mu_T32_d16_ks2-5_n10/` |
| **mu v2** (new adv, QSA/CSA separate, μ final, k=2,5,7) | 55399591 + 56761069 + **57807829** | **COMPLETED 108/108** | `mu_T32_d16_ks2-5-7_n10_v2/` |

Param match: CSA/nl **1024**, QSA **1032** (L=43).  
Advantage v2: \(k^2 \log d / C(d+k-1,k)\).

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

### mu v2 (circuit observable, k=2,5,7, QSA/CSA separate)
- [mu vs T (d=16)](mu_T32_d16_ks2-5-7_n10_v2/mu_vs_T.png)
- [mu vs d (T=32)](mu_T32_d16_ks2-5-7_n10_v2/mu_vs_d.png)

Email drafts:
- full campaign: [EMAIL_PROF.md](EMAIL_PROF.md)
- **mu redo (send this):** [EMAIL_PROF_mu_v2.md](EMAIL_PROF_mu_v2.md)
