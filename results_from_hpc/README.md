# HPC results — poly LCU kernel (Leonardo, Jul 2026)

Branch: `PennyLaneG` · Job campaign Section-2 with `kernel_mode=poly`.

## Layout

```
study/definitive_complexity_poly_T32_modeboth/
  mean_O_vs_T.png          # ō vs T (d=16, k=2 and k=3) + d^{-(k+1)/2} + advantage
  mean_O_vs_d.png          # ō vs d (T=32, k=2 and k=3)
  mean_O_vs_T_by_d.png     # same k=2, curves for d∈{4,8,16}
  mean_O_vs_d_by_T.png     # same k=2, curves for T∈{8,16,32}
  summary_*.csv, RIASSUNTO.txt

baselines_smoke/definitive_poly_T16_d16_k2_ep300_n5/
  training_curves.png, summary.json   # k=2, 5 seeds

baselines_smoke/ppl_vs_k_poly_T16_d16_ep300_n5/
  final_loss_vs_k.png                 # loss vs k for k-QSA / k-CSA / nl-CSA
  k*_summary.json
```

Training target: loss ≤ 3 when possible (cap 800 epochs on complexity).
Diagnostic ō is always the monomial `mean|a s^k|` (Haar / advantage on that scale);
the poly LCU kernel is used only for the training loss.
