"""
qsa_section2_circuit.py
=======================
Two-control-register overlap-interference quantum attention with a TRAINABLE-DEGREE
polynomial kernel (LCU over copy-number), read out by the FAST OVERLAP method.

WHAT CHANGED vs the previous version
------------------------------------
1. KERNEL.  The bare monomial s^k is a SHARPENER, not a softmax approximation: its
   participation ratio collapses (8.0 -> 1.3 as k: 1->4 at T=d=16), destroying value
   blending -- this is why the loss GREW with k. Replaced by
        g(s) = sum_{p=0..k} c_p s^p ,  c_p = beta^p / p!   (truncation of exp(beta*s))
   realized as an LCU over the NUMBER of participating key copies, via a k-qubit
   unary control register P.
2. TEMPERATURE.  beta is NOT optional. Standard attention is softmax(q.k/sqrt(d_k));
   for unit-norm tokens that is exp(beta*s) with beta = sqrt(d). The reduced
   temperature tau = beta/sqrt(d) governs everything, since the degree-p contribution
   ~ tau^p/p! .  beta=1 (plain 1/p!) gives tau<<1 -> near-uniform attention, p_eff=1,
   NO advantage.  We fix beta = sqrt(d) by default.
3. READOUT.  Fast overlap: we do NOT apply PREP^dag (on P) nor P^dag (on C1,C2), and
   never build the all-zeros projector. Instead we read N_tri*(k+1) amplitudes and
   combine them with known weights (verified exact):
        A = sum_{i<=j} sum_p (1/N) sqrt(c_p/lam) * state[idx(j,i,p)] ,   mu = |A|^2
   The j-dependent TARGET uncomputes (Step 3b) are kept -- they define s_ij and a_ij
   and cannot be replaced by a fixed basis overlap (|x_j> is in superposition over C1).

PREP PARAMETERS: FIXED, NOT TRAINED (recommended, and the default here).
   c_p = beta^p/p! with beta = sqrt(d) are the softmax-surrogate coefficients: a
   principled choice, not a hyperparameter. Fixing them means (i) the ONLY thing
   varying across a k-sweep is k, so the trend is attributable to k alone;
   (ii) no extra gradients / no parameter-shift rules through PREP; (iii) no risk of
   the optimizer collapsing to the degenerate tau<<1 regime (p_eff=1). Only W, V train.

Registers:  C1 (ceil log2 T) | C2 (ceil log2 T) | A (n=log2 d) | B_1..B_k (n each) | P (k, unary)
            n_tot = 2*ceil(log2 T) + (k+1)*log2 d + k

Measured:   mu = (1/(lam^2 N^4)) | sum_{j} sum_{i<=j} a_ij g(s_ij) |^2 ,  N^2 = T(T+1)/2
            s_ij = <x_j|W|x_i>,  a_ij = <y_j|V|x_i>,  lam = sum_p |c_p|

Run:  python qsa_section2_circuit.py            # self-check (needs PennyLane+JAX)
      python qsa_section2_circuit.py --oracle   # classical-only (numpy)
      python qsa_section2_circuit.py --leonardo
"""
from __future__ import annotations
import sys
from math import comb, ceil, log2, factorial
import numpy as np

# --------------------------------------------------------------------------- #
#  0.  helpers                                                                 #
# --------------------------------------------------------------------------- #
def bits_msb(x: int, width: int):
    return [(x >> (width - 1 - b)) & 1 for b in range(width)]

def qubit_budget(T, d, k):
    """n_tot = 2*ceil(log2 T) + (k+1)*log2 d + k   (the +k is the unary LCU register P)."""
    cT = max(1, ceil(log2(max(T, 2)))); n = max(1, ceil(log2(d)))
    return 2 * cT + (k + 1) * n + k

def triangular_state(T, cT):
    dimC = 2 ** cT
    v = np.zeros(dimC * dimC, dtype=float)
    N = np.sqrt(T * (T + 1) / 2)
    for j in range(T):
        for i in range(j + 1):
            v[j * dimC + i] = 1.0 / N
    return v

# --------------------------------------------------------------------------- #
#  1.  LCU coefficients and PREP angles  (FIXED, not trained)                  #
# --------------------------------------------------------------------------- #
def softmax_beta(d):
    """Temperature reproducing standard attention for unit-norm tokens: beta = sqrt(d)."""
    return np.sqrt(d)

def lcu_coeffs(k, beta):
    """c_p = beta^p/p!  ->  g(s) = exp(beta*s) truncated at degree k."""
    return np.array([beta ** p / factorial(p) for p in range(k + 1)], dtype=float)

def lam_of(c):
    """LCU subnormalization lam = sum_p |c_p|  (shots scale as lam^2)."""
    return float(np.sum(np.abs(c)))

def reduced_temperature(beta, d):
    """tau = beta/sqrt(d). Degree-p contribution ~ tau^p/p!.  tau<<1 -> p_eff=1 (no advantage)."""
    return beta / np.sqrt(d)

def prep_angles(c):
    """Unary cascade angles: theta_p = 2 arcsin sqrt(t_{p+1}/t_p), t_p = sum_{q>=p} c_q.
    Gives amplitude sqrt(c_p/lam) on |1^p 0^{k-p}>. Requires c_p >= 0 (true for beta>0)."""
    c = np.abs(np.asarray(c, dtype=float)); k = len(c) - 1
    t = np.array([c[p:].sum() for p in range(k + 2)])   # t[k+1] = 0
    th = []
    for p in range(k):
        r = t[p + 1] / t[p] if t[p] > 0 else 0.0
        th.append(2 * np.arcsin(np.sqrt(np.clip(r, 0.0, 1.0))))
    return np.array(th)

# --------------------------------------------------------------------------- #
#  2.  fast-readout index/weight table                                         #
# --------------------------------------------------------------------------- #
def overlap_table(T, d, k, c):
    """Flat indices and weights for the fast readout.
         A = sum_{i<=j} sum_p  weight(p) * state[idx(j,i,p)] ,   mu = |A|^2
    Wire order [C1(cT) | C2(cT) | A(n) | B_1..B_k(n) | P(k)], MSB = wire 0.
    A,B all-zeros; P in unary |1^p 0^{k-p}>  ->  P-value = 2^k - 2^(k-p)."""
    cT = max(1, ceil(log2(max(T, 2)))); n = max(1, ceil(log2(d)))
    ntot = 2 * cT + (k + 1) * n + k
    lam = lam_of(c); N = np.sqrt(T * (T + 1) / 2)
    idx, wts = [], []
    for j in range(T):
        for i in range(j + 1):
            base = j * 2 ** (ntot - cT) + i * 2 ** (ntot - 2 * cT)
            for p in range(k + 1):
                idx.append(base + (2 ** k - 2 ** (k - p)))
                wts.append((1.0 / N) * np.sqrt(c[p] / lam))
    return np.array(idx, dtype=np.int64), np.array(wts, dtype=float)

def mu_overlap(state, idx, wts):
    """mu from the fast overlap readout (no PREP^dag, no P^dag, no projector)."""
    jnp = _BK["jnp"]
    A = jnp.sum(jnp.asarray(wts) * jnp.take(state, jnp.asarray(idx)))
    return float(jnp.abs(A) ** 2)

# --------------------------------------------------------------------------- #
#  3.  classical path (cheap; the reference and the diagnostics)               #
# --------------------------------------------------------------------------- #
def g_poly(s, c):
    return sum(c[p] * s ** p for p in range(len(c)))

def classical_report(X, Y, Wmat, Vmat, k, beta=None):
    """Analytic mu plus the diagnostics that decide the advantage claim.
    Criterion (exponent-free):  mu >= k/m_k,  m_k = C(d+k-1,k)."""
    T, d = X.shape
    if beta is None: beta = softmax_beta(d)
    c = lcu_coeffs(k, beta); lam = lam_of(c)
    S = X @ (Wmat @ X.T); A = Y @ (Vmat @ X.T)
    w, sv = [], []
    for j in range(T):
        for i in range(j + 1):
            w.append(A[j, i] * g_poly(S[j, i], c)); sv.append(S[j, i])
    w = np.array(w); sv = np.array(sv); Ntri = T * (T + 1) // 2
    mu = (w.sum() / (lam * Ntri)) ** 2
    # rbar = mean |a g(s)| / lam  (same lam normalisation as the quantum mu readout)
    rbar = float(np.abs(w).sum() / (lam * Ntri)) if Ntri > 0 else 0.0
    # effective degree: which degrees actually carry g(s) on THIS overlap distribution
    contrib = np.array([np.abs(c[p] * sv ** p).mean() for p in range(k + 1)])
    frac = contrib / contrib.sum()
    p_eff = max([p for p in range(k + 1) if frac[p] > 0.05], default=0)
    mk = comb(d + k - 1, k)
    return dict(
        mu=mu, rbar=rbar,
        alignment=(abs(w.sum()) / np.abs(w).sum() if np.abs(w).sum() > 0 else 0.0),
        beta=beta, tau=reduced_temperature(beta, d), lam=lam, lam2=lam ** 2,
        degree_fracs=frac, p_eff=p_eff,
        mu_threshold=k / mk,                       # advantage iff mu >= k/m_k
        advantage=(mu >= k / mk),
        fair_classical=T * d * (comb(d + p_eff - 1, p_eff) if p_eff >= 1 else 1),
        claimed_classical=T * d * mk,
        n_qubits=qubit_budget(T, d, k),
    )

def participation_ratio(X, Wmat, c):
    """Effective # attended tokens: 1 = argmax, T = uniform. Softmax target ~1.45 at T=d=16."""
    T = X.shape[0]; S = X @ (Wmat @ X.T); out = []
    for j in range(T):
        a = np.abs(np.array([g_poly(S[j, i], c) for i in range(j + 1)]))
        if a.sum() <= 0: continue
        a = a / a.sum(); out.append(1.0 / np.sum(a ** 2))
    return float(np.mean(out))

# --------------------------------------------------------------------------- #
#  4.  backends + variational block                                            #
# --------------------------------------------------------------------------- #
_BK = {}
def set_backends():
    import pennylane as qml, jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _BK.update(qml=qml, jax=jax, jnp=jnp)
    return qml, jax, jnp

def real_ortho_block(qml, params, wires):
    """RY layers + CNOT ring -> real orthogonal. (Entangler skipped for a single wire:
    zip(w, w[1:]+w[:1]) would emit CNOT(0,0) at n=1, which PennyLane rejects.)"""
    L = params.shape[0]; w = list(wires)
    for l in range(L):
        for q, wire in enumerate(w):
            qml.RY(params[l, q], wires=wire)
        if len(w) >= 2:
            for a, b in zip(w, w[1:] + w[:1]):
                qml.CNOT(wires=[a, b])

# --------------------------------------------------------------------------- #
#  5.  the circuit (LCU + fast overlap readout)                                #
# --------------------------------------------------------------------------- #
def _layout(T, d, k):
    cT = max(1, ceil(log2(max(T, 2)))); n = max(1, ceil(log2(d)))
    C1 = list(range(cT)); C2 = list(range(cT, 2 * cT))
    A = list(range(2 * cT, 2 * cT + n))
    B = [list(range(2 * cT + n + c * n, 2 * cT + n + (c + 1) * n)) for c in range(k)]
    P = list(range(2 * cT + (k + 1) * n, 2 * cT + (k + 1) * n + k))
    return cT, n, C1, C2, A, B, P, 2 * cT + (k + 1) * n + k

def make_qsa_state_qnode(T, d, k, c, layers=2, dev_name="default.qubit",
                         c_dtype=None, mpi=False):
    """FAST path: runs Steps 1-3 only and returns the statevector.
    Combine with overlap_table()/mu_overlap() to get mu. c = FIXED LCU coefficients."""
    qml = _BK["qml"]
    cT, n, C1, C2, A, B, P, ntot = _layout(T, d, k)
    tri = triangular_state(T, cT); th = prep_angles(c)

    dk = dict(wires=ntot)
    if c_dtype is not None: dk["c_dtype"] = c_dtype
    if mpi: dk["mpi"] = True
    dev = qml.device(dev_name, **dk)

    @qml.qnode(dev, interface="jax", diff_method=None)
    def circ(X, Y, Wp, Vp):
        qml.StatePrep(tri, wires=C1 + C2)                      # Step 1
        qml.RY(th[0], wires=P[0])                              # PREP (unary cascade, fixed)
        for p in range(1, k):
            qml.ctrl(qml.RY, control=P[p - 1])(th[p], wires=P[p])
        for i in range(T):                                     # Step 2: loads
            cv = bits_msb(i, cT)
            qml.ctrl(lambda v: qml.StatePrep(v, wires=A),
                     control=C2, control_values=cv)(X[i])      #   A: always
            for cc in range(k):                                #   B_c: gated on P[c]
                qml.ctrl(lambda v, cc=cc: qml.StatePrep(v, wires=B[cc]),
                         control=C2 + [P[cc]], control_values=cv + [1])(X[i])
        real_ortho_block(qml, Vp, A)                           # Step 3a: V on A
        for cc in range(k):                                    #   W on B_c, gated on P[c]
            qml.ctrl(lambda cc=cc: real_ortho_block(qml, Wp, B[cc]), control=P[cc])()
        for j in range(T):                                     # Step 3b: target uncompute
            cv = bits_msb(j, cT)
            qml.ctrl(lambda v: qml.adjoint(qml.StatePrep)(v, wires=A),
                     control=C1, control_values=cv)(Y[j])
            for cc in range(k):
                qml.ctrl(lambda v, cc=cc: qml.adjoint(qml.StatePrep)(v, wires=B[cc]),
                         control=C1 + [P[cc]], control_values=cv + [1])(X[j])
        return qml.state()      # NO PREP^dag, NO P^dag, NO projector -- read amplitudes
    return circ, ntot

def make_qsa_projector_qnode(T, d, k, c, layers=2, dev_name="default.qubit", c_dtype=None):
    """SLOW literal path (PREP^dag, P^dag, all-zeros projector). Validation reference only."""
    qml = _BK["qml"]
    cT, n, C1, C2, A, B, P, ntot = _layout(T, d, k)
    tri = triangular_state(T, cT); th = prep_angles(c)
    dk = dict(wires=ntot)
    if c_dtype is not None: dk["c_dtype"] = c_dtype
    dev = qml.device(dev_name, **dk)

    def PREP():
        qml.RY(th[0], wires=P[0])
        for p in range(1, k):
            qml.ctrl(qml.RY, control=P[p - 1])(th[p], wires=P[p])

    @qml.qnode(dev, interface="jax", diff_method=None)
    def circ(X, Y, Wp, Vp):
        qml.StatePrep(tri, wires=C1 + C2); PREP()
        for i in range(T):
            cv = bits_msb(i, cT)
            qml.ctrl(lambda v: qml.StatePrep(v, wires=A), control=C2, control_values=cv)(X[i])
            for cc in range(k):
                qml.ctrl(lambda v, cc=cc: qml.StatePrep(v, wires=B[cc]),
                         control=C2 + [P[cc]], control_values=cv + [1])(X[i])
        real_ortho_block(qml, Vp, A)
        for cc in range(k):
            qml.ctrl(lambda cc=cc: real_ortho_block(qml, Wp, B[cc]), control=P[cc])()
        for j in range(T):
            cv = bits_msb(j, cT)
            qml.ctrl(lambda v: qml.adjoint(qml.StatePrep)(v, wires=A),
                     control=C1, control_values=cv)(Y[j])
            for cc in range(k):
                qml.ctrl(lambda v, cc=cc: qml.adjoint(qml.StatePrep)(v, wires=B[cc]),
                         control=C1 + [P[cc]], control_values=cv + [1])(X[j])
        qml.adjoint(PREP)()
        qml.adjoint(qml.StatePrep)(tri, wires=C1 + C2)
        return qml.expval(qml.Projector(np.zeros(ntot, dtype=int), wires=range(ntot)))
    return circ, ntot

# --------------------------------------------------------------------------- #
#  6.  self-check                                                              #
# --------------------------------------------------------------------------- #
def random_instance(T, d, k, layers, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    Y = rng.standard_normal((T, d)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    n = max(1, ceil(log2(d)))
    return X, Y, rng.standard_normal((layers, n)), rng.standard_normal((layers, n))

def self_check(T=3, d=4, k=2, layers=2, seed=0, tol=1e-6, check_projector=True):
    qml, jax, jnp = _BK["qml"], _BK["jax"], _BK["jnp"]
    beta = softmax_beta(d); c = lcu_coeffs(k, beta)
    nb = qubit_budget(T, d, k)
    if nb > 15:
        print(f"[guard] n_qubits={nb} > 15 (local limit). Use HPC (--leonardo).")
    X, Y, Wp, Vp = random_instance(T, d, k, layers, seed)
    circ, ntot = make_qsa_state_qnode(T, d, k, c, layers)
    st = circ(jnp.array(X), jnp.array(Y), jnp.array(Wp), jnp.array(Vp))
    idx, wts = overlap_table(T, d, k, c)
    mu_ov = mu_overlap(st, idx, wts)
    n = max(1, ceil(log2(d)))
    Wm = np.real(qml.matrix(real_ortho_block, wire_order=range(n))(qml, jnp.array(Wp), range(n)))[:d, :d]
    Vm = np.real(qml.matrix(real_ortho_block, wire_order=range(n))(qml, jnp.array(Vp), range(n)))[:d, :d]
    rep = classical_report(X, Y, Wm, Vm, k, beta)
    ok = abs(mu_ov - rep["mu"]) < tol
    line = (f"[self-check] T={T} d={d} k={k} n_q={ntot} beta=sqrt(d)={beta:.2f} tau={rep['tau']:.2f}: "
            f"mu_overlap={mu_ov:.6e} mu_analytic={rep['mu']:.6e}")
    if check_projector:
        cp_, _ = make_qsa_projector_qnode(T, d, k, c, layers)
        mu_p = float(cp_(jnp.array(X), jnp.array(Y), jnp.array(Wp), jnp.array(Vp)))
        ok = ok and abs(mu_ov - mu_p) < tol
        line += f" mu_projector={mu_p:.6e}"
    print(line + f"  {'PASS' if ok else 'FAIL'}")
    print(f"             lam^2={rep['lam2']:.1f}  phi={np.round(rep['degree_fracs'],3)}  "
          f"p_eff={rep['p_eff']}  PR={participation_ratio(X,Wm,c):.2f}  "
          f"mu>=k/m_k? {rep['advantage']}")
    return ok

def run_oracle_only():
    print("=== classical-only (numpy); beta = sqrt(d) fixed ===")
    for (T, d, k) in [(8, 16, 1), (8, 16, 2), (8, 16, 3), (8, 16, 4)]:
        X, Y, _, _ = random_instance(T, d, k, 2, seed=0)
        rng = np.random.default_rng(1)
        Wm, _ = np.linalg.qr(rng.standard_normal((d, d)))
        Vm, _ = np.linalg.qr(rng.standard_normal((d, d)))
        beta = softmax_beta(d); c = lcu_coeffs(k, beta)
        rep = classical_report(X, Y, Wm, Vm, k, beta)
        pr_m = participation_ratio(X, Wm, np.eye(k + 1)[k])      # bare monomial s^k
        print(f"  T={T} d={d} k={k}: n_q={rep['n_qubits']:2d} tau={rep['tau']:.2f} "
              f"lam^2={rep['lam2']:7.1f} p_eff={rep['p_eff']} "
              f"PR_poly={participation_ratio(X,Wm,c):5.2f} PR_mono={pr_m:5.2f} "
              f"mu={rep['mu']:.2e} thr={rep['mu_threshold']:.2e}")
    print("\n  PR_poly stays soft while PR_mono collapses -> this is the fix for 'loss grows with k'.")

def main():
    if "--oracle" in sys.argv:
        run_oracle_only(); return
    try:
        set_backends()
    except Exception as e:
        print(f"[env] PennyLane/JAX not importable ({e}). Classical-only.\n"); run_oracle_only(); return
    print("=== LCU circuit self-check: fast overlap == projector == analytic ===")
    self_check(T=3, d=4, k=1)     # 2*2 + 2*2 + 1 = 9 q
    self_check(T=3, d=4, k=2)     # 2*2 + 3*2 + 2 = 12 q
    self_check(T=2, d=4, k=3)     # 2*1 + 4*2 + 3 = 13 q
    print("\nFor a k-sweep use the classical_report path (cheap) and reserve the circuit for "
          "readout validation / HPC in-regime runs. PREP coefficients are FIXED (beta=sqrt d); "
          "only W,V train.")

LEONARDO_GUIDE = r"""
Leonardo Booster: 4x A100 64GB / node, 512 GB RAM. Statevector mem = 16*2^q (fp64), 8*2^q (fp32).
Reachable q: 1x A100 ~30 (fp64 fwd) / ~32 (fp32); 1 node (4 GPU, cuStateVec MPI) ~33;
             node CPU ~35; multi-node +1 qubit per 2x GPUs.

n_tot = 2*ceil(log2 T) + (k+1)*log2 d + k     <-- NOTE the +k (unary LCU register P)

Advantage regime is mu >= k/m_k with m_k = C(d+k-1,k); the classical baseline saturates
once T >= m_k (below that it is O(T^2 d)). Suggested k-sweep at fixed d, T >= m_k:
  d=8  k=1 T=64  -> 12+6+1 = 19 q
  d=8  k=2 T=64  -> 12+9+2 = 23 q
  d=8  k=3 T=128 -> 14+12+3 = 29 q      (m_3=120)
  d=8  k=4 T=256 -> 16+15+4 = 35 q      -> 4 nodes
  d=16 k=2 T=256 -> 16+12+2 = 30 q      (m_2=136)  1x A100
  d=16 k=3 T=1024-> 20+16+3 = 39 q      (m_3=816)  multi-node

CODE CHANGES for HPC:
  dev_name="lightning.gpu", c_dtype=np.complex64 (+1 qubit, ~2x; watch tiny-mu underflow)
  mpi=True + srun (1 rank/GPU) for multi-GPU. CAVEAT: with a DISTRIBUTED statevector the
  N_tri*(k+1) amplitudes of the fast readout live on different ranks -> gather them (small)
  before mu_overlap, or stay single-GPU up to ~30 q.
  The fast readout (make_qsa_state_qnode + mu_overlap) avoids PREP^dag, P^dag and the
  projector -- keep it; the projector path costs ~T^2 more work and is validation-only.
  Training: mu_overlap uses qml.state() (diff_method=None). For device gradients use the
  projector qnode with diff_method="adjoint", or differentiate the overlap expression.
  k-sweep note: PREP coefficients are FIXED (c_p = beta^p/p!, beta=sqrt d), so no gradients
  flow through P and the only trainable objects are W and V, identically across k.
"""

if __name__ == "__main__":
    if "--leonardo" in sys.argv:
        print(LEONARDO_GUIDE)
    else:
        main()
