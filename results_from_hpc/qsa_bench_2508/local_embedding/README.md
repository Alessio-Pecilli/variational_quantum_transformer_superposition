# Local embedding / hybrid-readout checks (qsa_bench_25_08)

Path verified:

- unnormalized `Xvoc` → `α = ||x||²`, `ξ = x/√α` with `||ξ||=1`
- `α_t` matches `||Y_unnorm||²` (max abs diff ~1e-16)
- with vs without `α` on same random W,V: raw L1 rises by the leverage floor; `L1 - L1_floor = L1_excess`
- short PTB Adam train (T=8,d=8,k=2): QSA≈CSA on excess; mean ρ ~0.33–0.38
- TFIM unit states (`α=1`): `L_B ≡ LB_excess`

These checks confirm norms enter the loss automatically via the hybrid R_t factor when the embedding is trained jointly.
