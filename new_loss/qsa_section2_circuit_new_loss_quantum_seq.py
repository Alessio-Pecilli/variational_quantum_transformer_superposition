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
#  2b.  FULL fast readout:  mu, zeta, nu  from ONE statevector                 #
# --------------------------------------------------------------------------- #
#  Derivation (see renyi_exact_note.tex Secs. 3-4). With the A-uncompute applied
#  (as make_qsa_state_qnode does), the amplitude at (C1=j, C2=i, A=m, B=0, P=1^p0^{k-p}) is
#      psi[j,i,m,p] = Ntri^{-1/2} e^{i phi_j} sqrt(c_p/lam) <e_m|U_{y_j}^dag V|x_i> s_ij^p .
#  Define the per-(j,m) weighted partial sums
#      alpha_{j,m} = sum_{i<=j} sum_p Ntri^{-1/2} sqrt(c_p/lam) psi[j,i,m,p]
#                  = e^{i phi_j} (lam Ntri)^{-1} <e_m|U_{y_j}^dag|z_j> .
#  Then, EXACTLY:
#      mu   = |sum_j alpha_{j,0}|^2                        (coherent over j: needs the phases)
#      nu   = (Ntri/T) sum_j |alpha_{j,0}|^2               (phase-insensitive)
#      zeta = (Ntri/T) sum_j sum_m |alpha_{j,m}|^2         (phase-insensitive; unitary A-uncompute
#                                                           preserves the A-norm, so summing over
#                                                           the full A basis gives ||z_j||^2)
#  Verified against the classical reference to 10 digits.
#  NOTE the different prefactors: mu carries (lam*Ntri)^{-2}, zeta/nu carry (lam^2 T Ntri)^{-1}.
# --------------------------------------------------------------------------- #

def overlap_table_full(T, d, k, c):
    """Padded index/weight tables for the full (mu, zeta, nu) fast readout.

    Returns (idx, wts) each of shape (T, d, T, k+1):
        idx[j, m, i, p] = flat statevector index of (C1=j, C2=i, A=m, B=0, P=1^p 0^{k-p})
        wts[j, m, i, p] = Ntri^{-1/2} sqrt(c_p/lam)   for i <= j, else 0  (padding)
    so that   alpha[j, m] = sum_{i,p} wts[j,m,i,p] * state[idx[j,m,i,p]].
    Wire order [C1(cT) | C2(cT) | A(n) | B_1..B_k(n) | P(k)], MSB = wire 0."""
    cT = max(1, ceil(log2(max(T, 2)))); n = max(1, ceil(log2(d)))
    ntot = 2 * cT + (k + 1) * n + k
    lam = lam_of(c); Ntri = T * (T + 1) // 2; N = np.sqrt(Ntri)
    strideC1 = 2 ** (ntot - cT)
    strideC2 = 2 ** (ntot - 2 * cT)
    strideA  = 2 ** (k * n + k)                     # qubits after A: B_1..B_k (k*n) then P (k)
    pval = np.array([2 ** k - 2 ** (k - p) for p in range(k + 1)], dtype=np.int64)  # unary
    wp   = np.sqrt(np.asarray(c, dtype=float) / lam) / N

    idx = np.zeros((T, d, T, k + 1), dtype=np.int64)
    wts = np.zeros((T, d, T, k + 1), dtype=float)
    for j in range(T):
        for m in range(d):
            for i in range(T):
                base = j * strideC1 + i * strideC2 + m * strideA
                idx[j, m, i, :] = base + pval
                if i <= j:
                    wts[j, m, i, :] = wp                # i > j stays 0 (padding)
    return idx, wts

def readout_alphas(state, idx, wts):
    """alpha[j,m] = sum_{i<=j} sum_p wts[j,m,i,p] * state[idx[j,m,i,p]]
                  = e^{i phi_j} (lam Ntri)^{-1} <e_m|U_{y_j}^dag|z_j>.
    Shape (T,d). Everything below is a function of these numbers alone."""
    jnp = _BK.get("jnp", np)
    amps = jnp.take(jnp.asarray(state), jnp.asarray(idx.reshape(-1))).reshape(idx.shape)
    return jnp.sum(jnp.asarray(wts) * amps, axis=(2, 3))            # (T,d)

def readout_mu_zeta_nu(state, idx, wts, T, d, k):
    """mu, zeta, nu from ONE statevector via the fast overlap readout (no un-preps,
    no projector). Returns a dict with the three observables and the derived quantities."""
    jnp = _BK.get("jnp", np)
    Ntri = T * (T + 1) // 2
    alpha = readout_alphas(state, idx, wts)
    a0 = alpha[:, 0]
    mu   = jnp.abs(jnp.sum(a0)) ** 2
    nu   = (Ntri / T) * jnp.sum(jnp.abs(a0) ** 2)
    zeta = (Ntri / T) * jnp.sum(jnp.abs(alpha) ** 2)
    return derived_quantities(float(mu), float(zeta), float(nu), T)

def readout_per_step(state, idx, wts, T, d, k):
    """EXACT per-step normalized fidelities and the UNIFORM cross-entropy.

         p_j = f_j^2 = |alpha_{j,0}|^2 / sum_m |alpha_{j,m}|^2 = nu_j / zeta_j
         CE_unif      = -(1/T) sum_j log p_j                       (Shannon, uniform weights)
         L_half_unif  = -2 log( (1/T) sum_j sqrt(p_j) )            (Renyi-1/2, uniform)
    with L_half_unif <= CE_unif by Jensen. Both are UNIFORMLY weighted -- the quantities
    the O(1)-setting objectives can only bound.

    All prefactors (lam, Ntri, T, the frame factor d/D) cancel in p_j = nu_j/zeta_j, so this
    is convention-independent.

    COST CAVEAT: exact here because the amplitude table is resolved in j, which a simulator
    gives for free. On hardware, resolving C1=j is the T-SETTING streaming readout (the
    per-j pairs (nu_j, zeta_j) of the obstruction proposition), NOT an O(1)-setting
    observable. Use this as the end-of-training EVALUATION metric / rigor anchor; train on
    objective A or B."""
    jnp = _BK.get("jnp", np)
    Ntri = T * (T + 1) // 2
    alpha = readout_alphas(state, idx, wts)
    nu_j   = jnp.abs(alpha[:, 0]) ** 2                    # propto |<y_j|z_j>|^2
    zeta_j = jnp.sum(jnp.abs(alpha) ** 2, axis=1)         # propto ||z_j||^2
    p = np.clip(np.asarray(nu_j) / np.maximum(np.asarray(zeta_j), 1e-300), 1e-300, 1.0)
    ce = float(-np.mean(np.log(p)))
    lh = float(-2 * np.log(max(np.mean(np.sqrt(p)), 1e-300)))
    return dict(p_j=p, CE_uniform=ce, L_half_uniform=lh,
                jensen_ok=(lh <= ce + 1e-9),
                nu_j=(Ntri / T) * np.asarray(nu_j), zeta_j=(Ntri / T) * np.asarray(zeta_j),
                participation=float(1.0 / np.sum((p / p.sum()) ** 2)))

def derived_quantities(mu, zeta, nu, T):
    """F, D_{1/2}, the two objectives, and the exactness check.
         F        = (T+1)/(2T) * mu/zeta          (pure-state fidelity to the target)
         D_{1/2}  = -log[(T+1)/(2T) * mu/nu]      (Renyi-1/2 rel. entropy, uniform || success)
         L_A      = -log(nu/zeta)   <= WCE
         L_B      = -log F          = L_A + D_{1/2}  <= WCE + D_1
    """
    out = dict(mu=mu, zeta=zeta, nu=nu, **_derived(mu, zeta, nu, T))
    out["identity_residual"] = abs(out["loss_B"] - (out["loss_A"] + out["D_half"]))
    return out

# --------------------------------------------------------------------------- #
#  3.  classical path (cheap; the reference and the diagnostics)               #
# --------------------------------------------------------------------------- #
def _derived(mu, zeta, nu, T):
    pref = (T + 1) / (2 * T)
    F = pref * mu / zeta if zeta > 0 else 0.0
    return dict(F=F,
                loss_A=-np.log(max(nu / zeta, 1e-300)) if zeta > 0 else np.inf,
                loss_B=-np.log(max(F, 1e-300)),
                D_half=-np.log(max(pref * mu / nu, 1e-300)) if nu > 0 else np.inf,
                weighted_fidelity=nu / zeta if zeta > 0 else 0.0)

def g_poly(s, c):
    return sum(c[p] * s ** p for p in range(len(c)))

def classical_report(X, Y, Wmat, Vmat, k, beta=None, phi=None):
    """Analytic mu plus the diagnostics that decide the advantage claim.
    Criterion (exponent-free):  mu >= k/m_k,  m_k = C(d+k-1,k).
    Complex-aware: s_ij = <x_j|W|x_i> = conj(x_j).(W x_i), a_ij = <y_j|V|x_i> (Hermitian
    inner products). phi: length-T trainable C1 phases, w_ij *= e^{i phi_j}."""
    T, d = X.shape
    if beta is None: beta = softmax_beta(d)
    if phi is None: phi = np.zeros(T)
    c = lcu_coeffs(k, beta); lam = lam_of(c)
    S = np.conj(X) @ (Wmat @ X.T); A = np.conj(Y) @ (Vmat @ X.T)   # S[j,i]=<x_j|W|x_i>
    w, sv = [], []
    for j in range(T):
        for i in range(j + 1):
            w.append(np.exp(1j * phi[j]) * A[j, i] * g_poly(S[j, i], c)); sv.append(S[j, i])
    w = np.array(w); sv = np.array(sv); Ntri = T * (T + 1) // 2
    mu = np.abs(w.sum() / (lam * Ntri)) ** 2                       # |.|^2 (complex)
    # --- reference values of the three observables (see renyi_exact_note.tex) ---
    S_ = np.conj(X) @ (Wmat @ X.T); A_ = np.conj(Y) @ (Vmat @ X.T)
    Kk = g_poly(S_, c) * np.tril(np.ones((T, T)))
    Aj = (A_ * Kk).sum(1)                                          # <y_j|z_j>, complex
    Zj = Kk @ (X @ Vmat.T)                                         # z_j rows (unnormalized)
    wj = np.linalg.norm(Zj, axis=1)
    zeta_ref = np.sum(wj ** 2) / (lam ** 2 * T * Ntri)
    nu_ref = np.sum(np.abs(Aj) ** 2) / (lam ** 2 * T * Ntri)
    # effective degree: which degrees actually carry g(s) on THIS overlap distribution
    contrib = np.array([np.abs(c[p] * sv ** p).mean() for p in range(k + 1)])
    frac = contrib / contrib.sum()
    p_eff = max([p for p in range(k + 1) if frac[p] > 0.05], default=0)
    mk = comb(d + k - 1, k)
    return dict(
        mu=mu, rbar=np.abs(w).sum() / Ntri,
        alignment=(abs(w.sum()) / np.abs(w).sum() if np.abs(w).sum() > 0 else 0.0),
        beta=beta, tau=reduced_temperature(beta, d), lam=lam, lam2=lam ** 2,
        degree_fracs=frac, p_eff=p_eff,
        zeta=zeta_ref, nu=nu_ref,
        **{key: val for key, val in
           _derived(mu, zeta_ref, nu_ref, T).items() if key in
           ('F', 'loss_A', 'loss_B', 'D_half', 'weighted_fidelity')},
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
    """REAL-orthogonal block (classical data). params (layers, n): RY + CNOT ring.
    (Entangler skipped for a single wire: zip(w,w[1:]+w[:1]) -> CNOT(0,0), which PennyLane rejects.)"""
    L = params.shape[0]; w = list(wires)
    for l in range(L):
        for q, wire in enumerate(w):
            qml.RY(params[l, q], wires=wire)
        if len(w) >= 2:
            for a, b in zip(w, w[1:] + w[:1]):
                qml.CNOT(wires=[a, b])

def unitary_block(qml, params, wires):
    """COMPLEX unitary block (quantum data). params (layers, n, 3): RX,RY,RZ per qubit
    (a general single-qubit rotation) + CNOT ring. Spans U(2^n) up to global phase.
    With complex W,V the overlaps s_ij, a_ij become genuinely complex -- the phases are
    no longer data-determined signs, which is why the trainable C1 phases below are needed."""
    L = params.shape[0]; w = list(wires)
    for l in range(L):
        for q, wire in enumerate(w):
            qml.RX(params[l, q, 0], wires=wire)
            qml.RY(params[l, q, 1], wires=wire)
            qml.RZ(params[l, q, 2], wires=wire)
        if len(w) >= 2:
            for a, b in zip(w, w[1:] + w[:1]):
                qml.CNOT(wires=[a, b])

def c1_phase(qml, phi, C1, cT, T):
    """Independently trainable phase e^{i phi_j} on the prediction-step register C1=|j>.
    Applied AFTER the triangular prep and NOT un-done: it modulates the coherent sum,
    A = (1/(lam N^2)) sum_j e^{i phi_j} sum_{i<=j} a_ij g(s_ij), giving the model a per-step
    knob to align the complex contributions for constructive interference. Because the
    reference in the fast readout is phase-free (P|0> x PREP|0>), e^{i phi_j} survives in
    the statevector and the overlap WEIGHTS stay real -- verified against analytic mu."""
    jnp = _BK["jnp"]
    diag = jnp.ones(2 ** cT, dtype=complex)
    diag = diag.at[jnp.arange(T)].set(jnp.exp(1j * phi[:T]))   # e^{i phi_j} on |j<T>, 1 elsewhere
    qml.DiagonalQubitUnitary(diag, wires=C1)

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
                         c_dtype=None, mpi=False, complex_gates=True):
    """FAST path: runs Steps 1-3 (+ trainable C1 phases) and returns the statevector.
    Combine with overlap_table()/mu_overlap() to get mu.
      complex_gates=True  -> V,W are complex unitary blocks (RX,RY,RZ); QUANTUM data.
                             params Wp,Vp shape (layers, n, 3); tokens X,Y complex.
      complex_gates=False -> real-orthogonal blocks (RY); classical data; Wp,Vp (layers,n).
    The QNode signature gains `phi` (length-T trainable phases on C1). c = FIXED LCU coeffs.
    For complex data pass c_dtype=complex128 on GPU (fp32 will lose the small mu)."""
    qml = _BK["qml"]
    cT, n, C1, C2, A, B, P, ntot = _layout(T, d, k)
    tri = triangular_state(T, cT); th = prep_angles(c)
    block = unitary_block if complex_gates else real_ortho_block

    dk = dict(wires=ntot)
    if c_dtype is not None: dk["c_dtype"] = c_dtype
    if mpi: dk["mpi"] = True
    dev = qml.device(dev_name, **dk)

    @qml.qnode(dev, interface="jax", diff_method=None)
    def circ(X, Y, Wp, Vp, phi):
        qml.StatePrep(tri, wires=C1 + C2)                      # Step 1
        c1_phase(qml, phi, C1, cT, T)                          # trainable phases e^{i phi_j} on C1
        qml.RY(th[0], wires=P[0])                              # PREP (unary cascade, FIXED)
        for p in range(1, k):
            qml.ctrl(qml.RY, control=P[p - 1])(th[p], wires=P[p])
        for i in range(T):                                     # Step 2: loads
            cv = bits_msb(i, cT)
            qml.ctrl(lambda v: qml.StatePrep(v, wires=A),
                     control=C2, control_values=cv)(X[i])      #   A: always
            for cc in range(k):                                #   B_c: gated on P[c]
                qml.ctrl(lambda v, cc=cc: qml.StatePrep(v, wires=B[cc]),
                         control=C2 + [P[cc]], control_values=cv + [1])(X[i])
        block(qml, Vp, A)                                      # Step 3a: V on A
        for cc in range(k):                                    #   W on B_c, gated on P[c]
            qml.ctrl(lambda cc=cc: block(qml, Wp, B[cc]), control=P[cc])()
        for j in range(T):                                     # Step 3b: target uncompute
            cv = bits_msb(j, cT)
            qml.ctrl(lambda v: qml.adjoint(qml.StatePrep)(v, wires=A),
                     control=C1, control_values=cv)(Y[j])
            for cc in range(k):
                qml.ctrl(lambda v, cc=cc: qml.adjoint(qml.StatePrep)(v, wires=B[cc]),
                         control=C1 + [P[cc]], control_values=cv + [1])(X[j])
        return qml.state()      # NO PREP^dag, NO P^dag, NO projector -- read amplitudes
    return circ, ntot

def make_qsa_projector_qnode(T, d, k, c, layers=2, dev_name="default.qubit", c_dtype=None,
                             complex_gates=True):
    """SLOW literal path (PREP^dag, P^dag, all-zeros projector). Validation reference only.
    NOTE: the C1 phases are applied but the reference is P|0> x PREP|0> (phase-free), so the
    un-preps are PREP^dag and P^dag ONLY -- the phase D is deliberately NOT un-applied."""
    qml = _BK["qml"]
    cT, n, C1, C2, A, B, P, ntot = _layout(T, d, k)
    tri = triangular_state(T, cT); th = prep_angles(c)
    block = unitary_block if complex_gates else real_ortho_block
    dk = dict(wires=ntot)
    if c_dtype is not None: dk["c_dtype"] = c_dtype
    dev = qml.device(dev_name, **dk)

    def PREP():
        qml.RY(th[0], wires=P[0])
        for p in range(1, k):
            qml.ctrl(qml.RY, control=P[p - 1])(th[p], wires=P[p])

    @qml.qnode(dev, interface="jax", diff_method=None)
    def circ(X, Y, Wp, Vp, phi):
        qml.StatePrep(tri, wires=C1 + C2)
        c1_phase(qml, phi, C1, cT, T)
        PREP()
        for i in range(T):
            cv = bits_msb(i, cT)
            qml.ctrl(lambda v: qml.StatePrep(v, wires=A), control=C2, control_values=cv)(X[i])
            for cc in range(k):
                qml.ctrl(lambda v, cc=cc: qml.StatePrep(v, wires=B[cc]),
                         control=C2 + [P[cc]], control_values=cv + [1])(X[i])
        block(qml, Vp, A)
        for cc in range(k):
            qml.ctrl(lambda cc=cc: block(qml, Wp, B[cc]), control=P[cc])()
        for j in range(T):
            cv = bits_msb(j, cT)
            qml.ctrl(lambda v: qml.adjoint(qml.StatePrep)(v, wires=A),
                     control=C1, control_values=cv)(Y[j])
            for cc in range(k):
                qml.ctrl(lambda v, cc=cc: qml.adjoint(qml.StatePrep)(v, wires=B[cc]),
                         control=C1 + [P[cc]], control_values=cv + [1])(X[j])
        qml.adjoint(PREP)()                                    # un-PREP the LCU register
        qml.adjoint(qml.StatePrep)(tri, wires=C1 + C2)         # un-prep controls (NOT the phase)
        return qml.expval(qml.Projector(np.zeros(ntot, dtype=int), wires=range(ntot)))
    return circ, ntot

# --------------------------------------------------------------------------- #
#  6.  self-check                                                              #
# --------------------------------------------------------------------------- #
def random_instance(T, d, k, layers, seed=0, complex_gates=True):
    """Returns X, Y, Wp, Vp, phi.
      complex_gates=True : complex unit tokens; Wp,Vp shape (layers,n,3) (RX,RY,RZ).
      complex_gates=False: real  unit tokens; Wp,Vp shape (layers,n)   (RY only).
    phi: length-T trainable C1 phases (initialized to 0 -> no phase)."""
    rng = np.random.default_rng(seed)
    n = max(1, ceil(log2(d)))
    if complex_gates:
        X = rng.standard_normal((T, d)) + 1j * rng.standard_normal((T, d))
        Y = rng.standard_normal((T, d)) + 1j * rng.standard_normal((T, d))
        Wp = rng.standard_normal((layers, n, 3)); Vp = rng.standard_normal((layers, n, 3))
    else:
        X = rng.standard_normal((T, d)); Y = rng.standard_normal((T, d))
        Wp = rng.standard_normal((layers, n)); Vp = rng.standard_normal((layers, n))
    X /= np.linalg.norm(X, axis=1, keepdims=True); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    phi = np.zeros(T)                                  # init phases to 0 (identity)
    return X, Y, Wp, Vp, phi

def self_check(T=3, d=4, k=2, layers=2, seed=0, tol=1e-6, check_projector=True,
               complex_gates=True):
    qml, jax, jnp = _BK["qml"], _BK["jax"], _BK["jnp"]
    beta = softmax_beta(d); c = lcu_coeffs(k, beta)
    nb = qubit_budget(T, d, k)
    if nb > 15:
        print(f"[guard] n_qubits={nb} > 15 (local limit). Use HPC (--leonardo).")
    X, Y, Wp, Vp, phi = random_instance(T, d, k, layers, seed, complex_gates=complex_gates)
    phi = _BK["jnp"].array(np.random.default_rng(seed + 9).uniform(0, 2 * np.pi, T))  # nonzero phases
    block = unitary_block if complex_gates else real_ortho_block
    circ, ntot = make_qsa_state_qnode(T, d, k, c, layers, complex_gates=complex_gates)
    st = circ(jnp.array(X), jnp.array(Y), jnp.array(Wp), jnp.array(Vp), phi)
    idx, wts = overlap_table(T, d, k, c)
    mu_ov = mu_overlap(st, idx, wts)
    # FULL readout: mu, zeta, nu from the SAME statevector
    idxF, wtsF = overlap_table_full(T, d, k, c)
    RO = readout_mu_zeta_nu(st, idxF, wtsF, T, d, k)
    n = max(1, ceil(log2(d)))
    Wm = np.array(qml.matrix(block, wire_order=range(n))(qml, jnp.array(Wp), range(n)))[:d, :d]
    Vm = np.array(qml.matrix(block, wire_order=range(n))(qml, jnp.array(Vp), range(n)))[:d, :d]
    if not complex_gates: Wm, Vm = Wm.real, Vm.real
    rep = classical_report(X, Y, Wm, Vm, k, beta, phi=np.array(phi))
    ok = abs(mu_ov - rep["mu"]) < tol
    gates = "complex(RX,RY,RZ)" if complex_gates else "real(RY)"
    line = (f"[self-check {gates}] T={T} d={d} k={k} n_q={ntot} tau={rep['tau']:.2f}: "
            f"mu_overlap={mu_ov:.6e} mu_analytic={rep['mu']:.6e}")
    if check_projector:
        cp_, _ = make_qsa_projector_qnode(T, d, k, c, layers, complex_gates=complex_gates)
        mu_p = float(cp_(jnp.array(X), jnp.array(Y), jnp.array(Wp), jnp.array(Vp), phi))
        ok = ok and abs(mu_ov - mu_p) < tol
        line += f" mu_projector={mu_p:.6e}"
    okz = abs(RO['zeta'] - rep['zeta']) < tol * max(1, rep['zeta'])
    okn = abs(RO['nu'] - rep['nu']) < tol * max(1, rep['nu'])
    okm = abs(RO['mu'] - rep['mu']) < tol
    ok = ok and okz and okn and okm
    print(line + f"  {'PASS' if ok else 'FAIL'}")
    print(f"             lam^2={rep['lam2']:.1f}  p_eff={rep['p_eff']}  "
          f"alignment={rep['alignment']:.3f}  mu>=k/m_k? {rep['advantage']}")
    print(f"             FULL readout: mu={RO['mu']:.6e}({'ok' if okm else 'BAD'})  "
          f"zeta={RO['zeta']:.6e}({'ok' if okz else 'BAD'})  nu={RO['nu']:.6e}({'ok' if okn else 'BAD'})")
    print(f"             F={RO['F']:.4f}  L_A={RO['loss_A']:.4f}  L_B={RO['loss_B']:.4f}  "
          f"D_1/2={RO['D_half']:.4f}  identity residual={RO['identity_residual']:.2e}")
    PS = readout_per_step(st, idxF, wtsF, T, d, k)
    print(f"             UNIFORM (eval only, T settings on hardware): CE={PS['CE_uniform']:.4f}  "
          f"L_1/2={PS['L_half_uniform']:.4f}  Jensen ok={PS['jensen_ok']}")
    return ok

def run_oracle_only():
    print("=== classical-only (numpy); beta = sqrt(d) fixed ===")
    for (T, d, k) in [(8, 16, 1), (8, 16, 2), (8, 16, 3), (8, 16, 4)]:
        X, Y, _, _, _ = random_instance(T, d, k, 2, seed=0, complex_gates=False)
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
    print("--- COMPLEX gates + trainable C1 phases (quantum data) ---")
    self_check(T=3, d=4, k=1, complex_gates=True)     # 9 q
    self_check(T=3, d=4, k=2, complex_gates=True)     # 12 q
    self_check(T=2, d=4, k=3, complex_gates=True)     # 13 q
    print("--- REAL gates (classical data), for contrast ---")
    self_check(T=3, d=4, k=2, complex_gates=False)
    print("\nComplex V,W (RX,RY,RZ) + per-step phases e^{i phi_j} on C1 for quantum sequences. "
          "The phases are trainable and let the model align the complex contributions "
          "(constructive interference). PREP coefficients FIXED (beta=sqrt d); W,V,phi train.")

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
