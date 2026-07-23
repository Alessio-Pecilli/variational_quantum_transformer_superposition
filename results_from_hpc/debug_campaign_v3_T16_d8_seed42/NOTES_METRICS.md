# Metric notes

- Paper plot: final_loss_vs_k.png
- Do NOT compare: 1/μ (mu-models) vs next-token CE PPL (nl-CSA)
- rel_impr5: rel_impr5=(mean(loss[t-5:t-1])-loss[t])/|old|
- CSA starts higher: Haar-like QR init → low μ; QSA angles less mixed.
- CSA ends lower/parallel: Full O(d) capacity (128) vs 12-angle ansatz.
- nl-CSA is k-independent: identical curves at k=1 and k=4 (max |Δ|=0.000e+00).
