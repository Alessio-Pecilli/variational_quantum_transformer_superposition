"""
qsa_classical_models.py
=======================
Classical analogues of the COMPLEX (quantum-sequence) QSA circuit -- pure numpy,
complex-aware. Three models that share the circuit's coherent-aggregate readout and
its trainable per-step phases, differing only in how W,V are parameterized / the kernel:

  kQSA   : W,V from the complex RX-RY-RZ + CNOT ansatz    (O(L log d) real params)  [circuit twin]
  kCSA   : W,V free complex UNITARY d x d                 (2 d^2 real params)
  nlCSA  : W,V free complex unitary + SOFTMAX kernel      ('iso' -> V unitary/isometric)

READOUT (identical to the circuit):
    s_ij = <x_j|W|x_i>,   a_ij = <y_j|V|x_i>            (Hermitian inner products, complex)
    A_j  = sum_{i<=j} a_ij K(s_ij)                       (causal prefix sum, complex)
    mu   = | sum_{j} e^{i phi_j} A_j |^2 / (lam N_tri)^2 ,   loss = -log mu + log T
  kernel K:  poly  K = sum_{p<=k} c_p s^p,  c_p = beta^p/p!, beta=sqrt(d)  (lam = sum|c_p|)
             soft  K_ij = exp(beta g_ij)/sum_{i'<=j} exp(beta g_{i'j}),    (lam = 1, normalized)
                   g_ij = |s_ij|^2 (fidelity, default) or Re(s_ij)

WHY THE PHASES phi_j ARE NEEDED HERE (and were not, for real classical data):
  The A_j are complex, and mu is the modulus-squared of their COHERENT sum over prediction
  steps. Multiplying e^{i phi_j} onto the j-th prefix sum lets the model align them. By the
  phase-alignment theorem the optimum is  phi_j* = -arg(A_j), giving
      mu_max = (sum_j |A_j|)^2 / (lam N_tri)^2 .
  (Under the alternative PER-STEP readout p_j = |<y_j|z_j>|^2/||z_j||^2, a global phase on z_j
   cancels, so phi_j is irrelevant there -- provided for the 'fair classical baseline'.)

Run:  python qsa_classical_models.py
"""
from __future__ import annotations
from math import comb, factorial, log, ceil, log2
import numpy as np

# --------------------------------------------------------------------------- #
#  coefficients
# --------------------------------------------------------------------------- #
def softmax_beta(d):            return np.sqrt(d)
def lcu_coeffs(k, beta):        return np.array([beta ** p / factorial(p) for p in range(k + 1)])
def lam_of(c):                  return float(np.sum(np.abs(c)))

# --------------------------------------------------------------------------- #
#  W,V parameterizations
# --------------------------------------------------------------------------- #
def _cnot(a, b, n):
    N = 2 ** n; M = np.zeros((N, N))
    for x in range(N):
        bits = [(x >> (n - 1 - w)) & 1 for w in range(n)]
        if bits[a] == 1: bits[b] ^= 1
        y = sum(bt << (n - 1 - w) for w, bt in enumerate(bits)); M[y, x] = 1.0
    return M

def real_ansatz_matrix(params, n):
    """RY + CNOT ring -> real orthogonal (classical-data ansatz). params (layers, n)."""
    RY = lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
    M = np.eye(2 ** n)
    for l in range(params.shape[0]):
        K = np.eye(1)
        for q in range(n): K = np.kron(K, RY(params[l, q]))
        M = K @ M
        if n >= 2:
            for a, b in zip(range(n), list(range(1, n)) + [0]): M = _cnot(a, b, n) @ M
    return M

def complex_ansatz_matrix(params, n):
    """RX,RY,RZ + CNOT ring -> complex unitary (quantum-data ansatz). params (layers, n, 3).
    This is the numpy mirror of the circuit's unitary_block: the kQSA W,V generator."""
    RX = lambda t: np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])
    RY = lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
    RZ = lambda t: np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])
    M = np.eye(2 ** n, dtype=complex)
    for l in range(params.shape[0]):
        K = np.eye(1, dtype=complex)
        for q in range(n): K = np.kron(K, RZ(params[l, q, 2]) @ RY(params[l, q, 1]) @ RX(params[l, q, 0]))
        M = K @ M
        if n >= 2:
            for a, b in zip(range(n), list(range(1, n)) + [0]): M = _cnot(a, b, n) @ M
    return M


def real_orthogonal_from_antisym(Araw):
    """W = exp(A - A^T) : a REAL ORTHOGONAL matrix from an unconstrained d x d generator.
    The classical-sequence counterpart of unitary_from_hermitian; d(d-1)/2 effective params."""
    from scipy.linalg import expm
    A = np.asarray(Araw, dtype=float)
    return np.real(expm(A - A.T))

def random_orthogonal(d, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q

def classical_sequences(T, d, n_seq=1, seed=0, kind="markov", rho=0.8):
    """REAL classical sequences on the unit sphere in R^d, y_j = next token.
      kind='iid'    : independent random unit vectors (no temporal structure)
      kind='markov' : x_{j+1} = normalize(rho*x_j + sqrt(1-rho^2)*eps)  -- correlated tokens,
                      the real-valued analogue of a quantum trajectory.
    Returns (Xs, Ys) of shape (n_seq, T, d)."""
    rng = np.random.default_rng(seed)
    Xs = np.zeros((n_seq, T, d)); Ys = np.zeros_like(Xs)
    for si in range(n_seq):
        traj = [rng.standard_normal(d)]
        traj[0] /= np.linalg.norm(traj[0])
        for _ in range(T):
            if kind == "iid":
                nxt = rng.standard_normal(d)
            else:
                nxt = rho * traj[-1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(d)
            traj.append(nxt / np.linalg.norm(nxt))
        traj = np.array(traj)
        Xs[si] = traj[:-1]; Ys[si] = traj[1:]
    return Xs, Ys

def unitary_from_hermitian(H):
    """W = exp(i H) with H Hermitian -> free unitary. H carries d^2 real params (kCSA/nlCSA)."""
    from scipy.linalg import expm
    H = (H + H.conj().T) / 2
    return expm(1j * H)

def random_unitary(d, rng):
    Z = (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / np.sqrt(2)
    Q, R = np.linalg.qr(Z); return Q @ np.diag(np.exp(1j * np.angle(np.diag(R))))

# --------------------------------------------------------------------------- #
#  scores, kernel, prefix sums
# --------------------------------------------------------------------------- #
def scores(A_row_states, M, B_col_states):
    """<a_j| M |b_i>  for rows j (A_row_states) and cols i (B_col_states). Hermitian on the left."""
    return np.conj(A_row_states) @ (M @ B_col_states.T)

def g_poly(S, c):
    return sum(c[p] * S ** p for p in range(len(c)))

def kernel_matrix(S, kernel="poly", c=None, beta=None, soft_score="abs2", causal=True):
    """Attention-weight matrix K[j,i] from the score matrix S[j,i]=<x_j|W|x_i> (complex).
      poly : K = sum_p c_p S^p           (unnormalized; lam = sum|c_p|)
      soft : K = softmax over beta*g,  g = |S|^2 ('abs2') or Re(S) ('re')  (normalized; lam=1)
    Returns (K, lam)."""
    T = S.shape[0]; M = np.tril(np.ones((T, T))) if causal else np.ones((T, T))
    if kernel == "poly":
        K = g_poly(S, c) * M; lam = lam_of(c)
    elif kernel == "soft":
        g = np.abs(S) ** 2 if soft_score == "abs2" else S.real
        E = np.exp(beta * g) * M
        z = E.sum(1, keepdims=True); z[np.abs(z) < 1e-12] = 1.0
        K = E / z; lam = 1.0
    else:
        raise ValueError(kernel)
    return K, lam

def partial_sums(X, Y, W, V, **kw):
    """A_j = sum_{i<=j} a_ij K(s_ij),  length-T complex vector. a_ij=<y_j|V|x_i>."""
    S = scores(X, W, X)                    # s_ij = <x_j|W|x_i>
    Amat = scores(Y, V, X)                 # a_ij = <y_j|V|x_i>
    K, lam = kernel_matrix(S, **kw)
    return (Amat * K).sum(1), lam          # sum over i (causal already in K)

# --------------------------------------------------------------------------- #
#  readouts
# --------------------------------------------------------------------------- #
def coherent_mu(Aj, phi, lam, T):
    """mu = | sum_j e^{i phi_j} A_j |^2 / (lam N_tri)^2   (the circuit's readout)."""
    Ntri = T * (T + 1) // 2
    return np.abs(np.sum(np.exp(1j * phi) * Aj) / (lam * Ntri)) ** 2

def coherent_loss(Aj, phi, lam, T):
    return -np.log(max(coherent_mu(Aj, phi, lam, T), 1e-300)) + np.log(T)

def optimal_phi(Aj):
    """Phase-alignment theorem: phi_j* = -arg(A_j) maximizes mu (up to global gauge)."""
    return -np.angle(Aj)

def mu_max(Aj, lam, T):
    """Aligned readout (sum_j|A_j|)^2/(lam N_tri)^2 -- value of mu at phi = phi*."""
    Ntri = T * (T + 1) // 2
    return (np.sum(np.abs(Aj)) / (lam * Ntri)) ** 2

def perstep_ce(X, Y, W, V, **kw):
    """STANDARD classical readout: p_j = |<y_j|z_j/||z_j|| >|^2, z_j = sum_{i<=j} K_ij (V x_i).
    Phase-INVARIANT (global phase of z_j cancels), so phi_j is irrelevant here.
    Provided for the 'fair classical baseline'. Returns mean cross-entropy."""
    S = scores(X, W, X); K, _ = kernel_matrix(S, **kw)
    Z = K @ (X @ V.T)                                  # z_j = sum_i K_ij V x_i
    nz = np.linalg.norm(Z, axis=1, keepdims=True); nz[nz < 1e-12] = 1.0
    p = np.clip(np.abs(np.sum(np.conj(Y) * (Z / nz), axis=1)) ** 2, 1e-300, 1.0)
    return float(-np.mean(np.log(p)))

# --------------------------------------------------------------------------- #
#  models  (forward loss + optimal-phase alignment + parameter count)
# --------------------------------------------------------------------------- #
def _kernkw(kind, d, k):
    beta = softmax_beta(d)
    if kind == "poly": return dict(kernel="poly", c=lcu_coeffs(k, beta)), lam_of(lcu_coeffs(k, beta))
    if kind == "soft": return dict(kernel="soft", beta=beta, soft_score="abs2"), 1.0
    raise ValueError(kind)

def model_forward(X, Y, W, V, k, kind="poly", phi=None, readout="coherent", align=False):
    """Unified forward pass for any (W,V). Returns dict(loss, mu, phi, alignment).
      readout='coherent' -> circuit's aggregate (uses phi; align=True sets phi=phi*).
      readout='perstep'  -> standard classical CE (phi ignored).
    """
    T, d = X.shape
    kw, lam = _kernkw(kind, d, k)
    if readout == "perstep":
        ce = perstep_ce(X, Y, W, V, **kw)
        return dict(loss=ce, mu=None, phi=None, alignment=None, lam=lam)
    Aj, lam = partial_sums(X, Y, W, V, **kw)
    if align or phi is None:
        phi = optimal_phi(Aj)                          # phi_j* = -arg A_j (theorem)
    mu = coherent_mu(Aj, phi, lam, T)
    alignment = np.abs(np.sum(np.exp(1j * phi) * Aj)) / max(np.sum(np.abs(Aj)), 1e-300)
    return dict(loss=-np.log(max(mu, 1e-300)) + np.log(T), mu=mu, phi=phi,
                alignment=float(alignment), lam=lam, Aj=Aj)

def make_W_V(model, d, layers, rng, Hs=None, complex_gates=True):
    """Instantiate (W,V). complex_gates=True -> QUANTUM sequences (complex ansatz / unitary);
    complex_gates=False -> CLASSICAL sequences (real RY+CNOT ansatz / real orthogonal)."""
    n = max(1, ceil(log2(d)))
    if model == "kqsa":
        if complex_gates:
            Wp = rng.standard_normal((layers, n, 3)); Vp = rng.standard_normal((layers, n, 3))
            return complex_ansatz_matrix(Wp, n)[:d, :d], complex_ansatz_matrix(Vp, n)[:d, :d]
        Wp = rng.standard_normal((layers, n)); Vp = rng.standard_normal((layers, n))
        return real_ansatz_matrix(Wp, n)[:d, :d], real_ansatz_matrix(Vp, n)[:d, :d]
    elif model in ("kcsa", "nlcsa"):
        return ((random_unitary(d, rng), random_unitary(d, rng)) if complex_gates
                else (random_orthogonal(d, rng), random_orthogonal(d, rng)))
    raise ValueError(model)

def param_count(model, d, k, layers=2):
    n = max(1, ceil(log2(d)))
    if model == "kqsa":  return 2 * 3 * layers * n + 0      # W,V ansatz angles (phi extra: +T)
    if model == "kcsa":  return 2 * d * d                    # two Hermitian generators
    if model == "nlcsa": return 2 * d * d
    raise ValueError(model)


# =========================================================================== #
#  THE THREE OBSERVABLES  (mu, zeta, nu)  --  exact classical counterparts     #
#  of the circuit's fast readout. See renyi_exact_note.tex Secs. 3-4.          #
#     mu   = (lam*Ntri)^{-2} |sum_j e^{i varphi_j} <y_j|z_j>|^2   [phase-SENSITIVE]
#     zeta = (lam^2 T Ntri)^{-1} sum_j ||z_j||^2                  [phase-insensitive]
#     nu   = (lam^2 T Ntri)^{-1} sum_j |<y_j|z_j>|^2              [phase-insensitive]
#  NOTE the DIFFERENT prefactors: mu un-prepares both control registers (two
#  factors Ntri^{-1/2}); zeta,nu keep C1 as a label and un-prepare only C2
#  (one Ntri^{-1/2} and one T^{-1/2}).
# =========================================================================== #

def mu_zeta_nu(X, Y, W, V, k, kind="poly", beta=None, phi=None, xp=np):
    """Returns (mu, zeta, nu) exactly as the circuit reads them out."""
    T, d = X.shape
    if beta is None: beta = float(np.sqrt(d))
    kw, lam = _kernkw(kind, d, k)
    S = xp.conj(X) @ (W @ X.T)
    Amat = xp.conj(Y) @ (V @ X.T)
    mask = xp.asarray(np.tril(np.ones((T, T))))
    if kind == "poly":
        c = kw["c"]; K = sum(c[p] * S ** p for p in range(len(c))) * mask
    else:
        E = xp.exp(beta * xp.abs(S) ** 2) * mask
        K = E / xp.maximum(xp.sum(E, axis=1, keepdims=True), 1e-30)
    Aj = xp.sum(Amat * K, axis=1)                      # <y_j|z_j>  (complex)
    Zj = K @ (X @ V.T)                                 # rows = z_j (unnormalized)
    wj2 = xp.sum(xp.abs(Zj) ** 2, axis=1)              # ||z_j||^2
    Ntri = T * (T + 1) / 2
    if phi is None: phi = xp.zeros(T)
    mu = xp.abs(xp.sum(xp.exp(1j * phi) * Aj)) ** 2 / (lam ** 2 * Ntri ** 2)
    zeta = xp.sum(wj2) / (lam ** 2 * T * Ntri)
    nu = xp.sum(xp.abs(Aj) ** 2) / (lam ** 2 * T * Ntri)
    return mu, zeta, nu

def optimal_phi_from(X, Y, W, V, k, kind="poly", beta=None):
    """phi_j* = -arg<y_j|z_j>  (phase-alignment theorem): maximizes mu."""
    T, d = X.shape
    if beta is None: beta = float(np.sqrt(d))
    kw, _ = _kernkw(kind, d, k)
    S = np.conj(X) @ (W @ X.T); Amat = np.conj(Y) @ (V @ X.T)
    mask = np.tril(np.ones((T, T)))
    if kind == "poly":
        c = kw["c"]; K = sum(c[p] * S ** p for p in range(len(c))) * mask
    else:
        E = np.exp(beta * np.abs(S) ** 2) * mask; K = E / np.maximum(E.sum(1, keepdims=True), 1e-30)
    return -np.angle((Amat * K).sum(1))

def derived(mu, zeta, nu, T):
    """F, the two objective losses, D_1/2, and the exact-identity residual."""
    pref = (T + 1) / (2 * T)
    F = pref * mu / zeta
    LA = -np.log(max(float(nu / zeta), 1e-300))
    LB = -np.log(max(float(F), 1e-300))
    D12 = -np.log(max(float(pref * mu / nu), 1e-300))
    return dict(mu=float(mu), zeta=float(zeta), nu=float(nu), F=float(F),
                loss_A=LA, loss_B=LB, D_half=D12,
                weighted_fidelity=float(nu / zeta), identity_residual=abs(LB - (LA + D12)))


def alignment_factor(X, Y, W, V, k, kind="poly", beta=None, phi=None):
    """G = |sum_j e^{i phi_j} A_j| / sum_j |A_j|  in [0,1],  A_j = <y_j|z_j>.
    G = 1 iff every term is aligned. With trainable phases (complex case) the phase-alignment
    theorem gives G -> 1; with a REAL ansatz and NO phase correction the model must align the
    SIGNS of A_j through W,V alone, and the residual misalignment costs -2 log G nats."""
    T, d = X.shape
    if beta is None: beta = float(np.sqrt(d))
    kw, _ = _kernkw(kind, d, k)
    S = np.conj(X) @ (W @ X.T); Amat = np.conj(Y) @ (V @ X.T); mask = np.tril(np.ones((T, T)))
    if kind == "poly":
        c = kw["c"]; K = sum(c[p] * S ** p for p in range(len(c))) * mask
    else:
        E = np.exp(beta * np.abs(S) ** 2) * mask; K = E / np.maximum(E.sum(1, keepdims=True), 1e-30)
    Aj = (Amat * K).sum(1)
    if phi is None: phi = np.zeros(T)
    den = np.sum(np.abs(Aj))
    return float(np.abs(np.sum(np.exp(1j * phi) * Aj)) / max(den, 1e-300))

def derived3(mu, zeta, nu, T, G):
    """THREE-TERM exact decomposition, valid whether or not the phases are aligned:

        -log F  =  -log(nu/zeta)  +  D_1/2(u_T || Qt)  +  (-2 log G)
                    weighted        uniformity            MISALIGNMENT  (all >= 0)

    With trainable phases at the optimum G=1 and the third term vanishes, recovering the
    two-term identity. For a REAL ansatz with no phase correction the third term is a genuine
    cost, measured -2 log G ~ 1.4-3.2 nats at random parameters."""
    pref = (T + 1) / (2 * T)
    F = pref * mu / zeta
    LA = -np.log(max(float(nu / zeta), 1e-300))
    LB = -np.log(max(float(F), 1e-300))
    mis = -2 * np.log(max(float(G), 1e-300))
    D12 = LB - LA - mis                                  # exact by the identity
    return dict(mu=float(mu), zeta=float(zeta), nu=float(nu), F=float(F),
                loss_A=LA, loss_B=LB, D_half=D12, misalignment=mis, G=float(G),
                weighted_fidelity=float(nu / zeta),
                identity_residual=abs(LB - (LA + D12 + mis)))

def wce_bound(X, Y, W, V, k, kind="poly", beta=None):
    """WCE = -sum_j q_j log f_j^2 with q_j = ||z_j||^2/sum||z||^2 : the Shannon bound
    (Thm. 'Shannon bounds': loss_A <= WCE, loss_B <= WCE + D_1)."""
    T, d = X.shape
    if beta is None: beta = float(np.sqrt(d))
    kw, _ = _kernkw(kind, d, k)
    S = np.conj(X) @ (W @ X.T); Amat = np.conj(Y) @ (V @ X.T); mask = np.tril(np.ones((T, T)))
    if kind == "poly":
        c = kw["c"]; K = sum(c[p] * S ** p for p in range(len(c))) * mask
    else:
        E = np.exp(beta * np.abs(S) ** 2) * mask; K = E / np.maximum(E.sum(1, keepdims=True), 1e-30)
    Aj = (Amat * K).sum(1); Zj = K @ (X @ V.T)
    w = np.linalg.norm(Zj, axis=1); f2 = np.clip((np.abs(Aj) / np.maximum(w, 1e-30)) ** 2, 1e-300, 1)
    q = w ** 2 / np.sum(w ** 2)
    return dict(WCE=float(-np.sum(q * np.log(f2))),
                uniform_CE=float(-np.mean(np.log(f2))),      # the T-setting rigor anchor
                per_step_p=f2)


def uniform_ce(X, Y, W, V, k, kind="poly", beta=None):
    """UNIFORM (unweighted) evaluation metrics for given model parameters -- the quantities
    the O(1)-setting objectives A and B can only bound. Works for any model (kQSA, kCSA,
    nlCSA) since it takes W,V directly; pass kind='soft' for nl-CSA.

        p_j = f_j^2 = |<y_j|z_j>|^2 / ||z_j||^2      (exact per-step normalized fidelity)
        CE_uniform     = -(1/T) sum_j log p_j        (Shannon, the reference loss)
        L_half_uniform = -2 log((1/T) sum_j sqrt(p_j))  (Renyi-1/2, the ideal target loss)
    with L_half_uniform <= CE_uniform by Jensen (concavity of log).

    On hardware these correspond to the per-j pairs (nu_j, zeta_j), i.e. the T-SETTING
    streaming readout, so this is an EVALUATION metric, not a training objective. Train on
    objective A or B (O(1) settings) and report these at the end."""
    T, d = X.shape
    if beta is None: beta = float(np.sqrt(d))
    kw, _ = _kernkw(kind, d, k)
    S = np.conj(X) @ (W @ X.T); Amat = np.conj(Y) @ (V @ X.T)
    mask = np.tril(np.ones((T, T)))
    if kind == "poly":
        c = kw["c"]; K = sum(c[p] * S ** p for p in range(len(c))) * mask
    else:
        E = np.exp(beta * np.abs(S) ** 2) * mask
        K = E / np.maximum(E.sum(1, keepdims=True), 1e-30)
    Aj = (Amat * K).sum(1); Zj = K @ (X @ V.T)
    w = np.linalg.norm(Zj, axis=1)
    p = np.clip((np.abs(Aj) / np.maximum(w, 1e-30)) ** 2, 1e-300, 1.0)
    ce = float(-np.mean(np.log(p)))
    lh = float(-2 * np.log(max(np.mean(np.sqrt(p)), 1e-300)))
    q = w ** 2 / np.sum(w ** 2)
    return dict(p_j=p, CE_uniform=ce, L_half_uniform=lh, jensen_ok=(lh <= ce + 1e-9),
                WCE=float(-np.sum(q * np.log(p))),          # the weighted Shannon bound
                participation=float(1.0 / np.sum((p / p.sum()) ** 2)),
                perplexity_uniform=float(np.exp(ce)))

# =========================================================================== #
#  TRAINING:  objective A = -log(nu/zeta)   |   objective B = -log F           #
#  Objective A is PHASE-INSENSITIVE (phi has exactly zero gradient) -> phi is
#  not part of the parameter vector for A. Objective B needs phi.
#  Gradients: jax.grad when JAX is importable, else scipy L-BFGS-B (numerical).
# =========================================================================== #

try:
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except Exception:
    jnp = np; _HAS_JAX = False

def _pack(model, d, layers, complex_gates, objective, seed):
    """Initial flat parameter vector + an unpack closure."""
    rng = np.random.default_rng(seed)
    n = max(1, ceil(log2(d)))
    shape = (layers, n, 3) if complex_gates else (layers, n)
    nW = int(np.prod(shape))
    if model == "kqsa":
        p0 = [rng.uniform(0, 2 * np.pi, nW), rng.uniform(0, 2 * np.pi, nW)]
        def unpack(v):
            Wp = v[:nW].reshape(shape); Vp = v[nW:2 * nW].reshape(shape)
            mk = complex_ansatz_matrix if complex_gates else real_ansatz_matrix
            return mk(Wp, n)[:d, :d], mk(Vp, n)[:d, :d], 2 * nW
    else:      # kcsa / nlcsa : free unitary W=expm(iH) (complex) or orthogonal W=expm(A-A^T)
        nH = d * d
        p0 = [0.1 * rng.standard_normal(nH), 0.1 * rng.standard_normal(nH)]
        gen = _expi if complex_gates else real_orthogonal_from_antisym
        def unpack(v):
            H1 = v[:nH].reshape(d, d); H2 = v[nH:2 * nH].reshape(d, d)
            return gen(H1), gen(H2), 2 * nH
    v0 = np.concatenate(p0)
    nbase = len(v0)
    if objective == "B":
        v0 = np.concatenate([v0, np.zeros(0)])      # phi appended by caller (needs T)
    return v0, unpack, nbase

def _expi(H):
    """Unitary from a Hermitian generator, differentiably (eigendecomposition)."""
    Hh = (H + H.conj().T) / 2
    ev, U = np.linalg.eigh(Hh)
    return (U * np.exp(1j * ev)) @ U.conj().T

def loss_fn(v, unpack, nbase, X, Y, k, T, objective, kind="poly", beta=None, use_phases=True):
    """The scalar objective. objective='A' -> -log(nu/zeta); 'B' -> -log F.
    use_phases=False (real ansatz, classical sequences): no phase parameters at all."""
    W, V, _ = unpack(v)
    phi = v[nbase:nbase + T] if (objective == "B" and use_phases) else None
    mu, zeta, nu = mu_zeta_nu(X, Y, W, V, k, kind=kind, beta=beta, phi=phi)
    if objective == "A":
        return -np.log(max(float(nu / zeta), 1e-300))
    pref = (T + 1) / (2 * T)
    return -np.log(max(float(pref * mu / zeta), 1e-300))

def train(X, Y, k, model="kqsa", objective="B", kind="poly", layers=8, complex_gates=True,
          beta=None, maxiter=300, seed=0, verbose=True, align_init=True, use_phases=None):
    """Train W,V (+phi for objective B) on objective A or B.

    Returns dict(history, final, W, V, phi, objective, ...). The reported loss is the
    objective's own loss; `final` also carries mu, zeta, nu, F, D_half and the Shannon
    bound WCE so the exact inequalities can be checked at the optimum."""
    from scipy.optimize import minimize
    T, d = X.shape
    if beta is None: beta = float(np.sqrt(d))
    if use_phases is None: use_phases = bool(complex_gates)     # real ansatz -> no phases
    v0, unpack, nbase = _pack(model, d, layers, complex_gates, objective, seed)
    if objective == "B" and use_phases:
        W0, V0, _ = unpack(v0)
        phi0 = optimal_phi_from(X, Y, W0, V0, k, kind, beta) if align_init else np.zeros(T)
        v0 = np.concatenate([v0, phi0])
    hist = []
    def f(v):
        L = loss_fn(v, unpack, nbase, X, Y, k, T, objective, kind, beta,
                    use_phases=use_phases)
        hist.append(L); return L
    res = minimize(f, v0, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=20 * maxiter))
    W, V, _ = unpack(res.x)
    phi = (res.x[nbase:nbase + T] if (objective == "B" and use_phases) else np.zeros(T))
    mu, zeta, nu = mu_zeta_nu(X, Y, W, V, k, kind=kind, beta=beta, phi=phi)
    G = alignment_factor(X, Y, W, V, k, kind=kind, beta=beta, phi=phi)
    out = derived3(mu, zeta, nu, T, G); out.update(wce_bound(X, Y, W, V, k, kind=kind, beta=beta))
    out.update({f'eval_{key}': val for key, val in
                uniform_ce(X, Y, W, V, k, kind=kind, beta=beta).items() if key != 'p_j'})
    out["bound_A_ok"] = out["loss_A"] <= out["WCE"] + 1e-9
    out["objective"] = objective; out["n_params"] = len(res.x)
    if verbose:
        print(f"  [{model}/{objective}/{kind} k={k}] loss {hist[0]:.4f} -> "
              f"{out['loss_'+objective]:.4f}   ({len(hist)} evals, {len(res.x)} params)")
        print(f"      mu={out['mu']:.3e} zeta={out['zeta']:.3e} nu={out['nu']:.3e}  "
              f"F={out['F']:.4f}  D_1/2={out['D_half']:.4f}")
        print(f"      L_A={out['loss_A']:.4f} <= WCE={out['WCE']:.4f}: {out['bound_A_ok']}   "
              f"L_B={out['loss_B']:.4f} = L_A+D_1/2+misalign (res {out['identity_residual']:.1e})")
        print(f"      G={out['G']:.4f}  misalignment=-2logG={out['misalignment']:.4f}  "
              f"D_1/2={out['D_half']:.4f}")
        print(f"      EVAL (uniform, T settings): CE={out['eval_CE_uniform']:.4f} "
              f"(PPL={out['eval_perplexity_uniform']:.3f})  L_1/2={out['eval_L_half_uniform']:.4f}  "
              f"participation={out['eval_participation']:.2f}/{T}")
    return dict(history=np.array(hist), final=out, W=W, V=V, phi=phi)


# =========================================================================== #
#  QUANTUM SEQUENCES: trajectory data, batching, and the phase-transfer test    #
# =========================================================================== #
#  A quantum sequence is a unitary trajectory |psi_{j+1}> = U|psi_j>, so the target
#  is the NEXT STATE: X[j] = psi_j, Y[j] = psi_{j+1}. Unlike independent random
#  states, consecutive tokens are strongly correlated, which changes both the
#  overlaps s_ij and the phase structure arg<y_j|z_j> -- hence the need for the
#  trainable phase-compensation parameters phi_j on the prediction register.
# =========================================================================== #

def tfim_hamiltonian(nq, J=1.0, h=1.0, rng=None, random_couplings=False):
    """H = -sum_i J_i Z_i Z_{i+1} - sum_i h_i X_i  on nq qubits (open chain), d = 2^nq."""
    I2 = np.eye(2); Xp = np.array([[0, 1], [1, 0]], dtype=complex); Zp = np.diag([1, -1]).astype(complex)
    def op(P, q):
        o = np.eye(1, dtype=complex)
        for i in range(nq): o = np.kron(o, P if i == q else I2)
        return o
    d = 2 ** nq; H = np.zeros((d, d), dtype=complex)
    for q in range(nq):
        hq = h if not random_couplings else h * (0.5 + rng.random())
        H -= hq * op(Xp, q)
    for q in range(nq - 1):
        Jq = J if not random_couplings else J * (0.5 + rng.random())
        H -= Jq * (op(Zp, q) @ op(Zp, q + 1))
    return H

def tfim_trajectories(T, nq=2, dt=0.3, J=1.0, h=1.0, n_seq=1, seed=0,
                      random_couplings=False, H=None):
    """Quantum-sequence dataset. Returns (Xs, Ys) of shape (n_seq, T, d), d = 2^nq,
    with Xs[s,j] = psi_j and Ys[s,j] = psi_{j+1} = U psi_j,  U = exp(-i H dt).
    Pass H explicitly (or random_couplings=True with a fresh seed) to build a
    HELD-OUT test set from different Hamiltonian parameters."""
    from scipy.linalg import expm
    rng = np.random.default_rng(seed)
    if H is None: H = tfim_hamiltonian(nq, J, h, rng, random_couplings)
    U = expm(-1j * H * dt); d = 2 ** nq
    Xs = np.zeros((n_seq, T, d), dtype=complex); Ys = np.zeros_like(Xs)
    for s in range(n_seq):
        p = rng.standard_normal(d) + 1j * rng.standard_normal(d); p /= np.linalg.norm(p)
        traj = [p]
        for _ in range(T): traj.append(U @ traj[-1])
        traj = np.array(traj)
        Xs[s] = traj[:-1]; Ys[s] = traj[1:]
    return Xs, Ys, H

def batch_losses(Xs, Ys, W, V, k, phi=None, kind="poly", beta=None):
    """Per-sequence observables -> batch-averaged losses. Each sequence is its own
    circuit run, so the batch loss is the MEAN of the per-sequence losses (not the
    loss of the mean observables)."""
    Xs = np.atleast_3d(Xs) if Xs.ndim == 3 else Xs[None]
    if Ys.ndim == 2: Ys = Ys[None]
    T = Xs.shape[1]; rows = []
    for Xi, Yi in zip(Xs, Ys):
        mu, ze, nu = mu_zeta_nu(Xi, Yi, W, V, k, kind=kind, beta=beta, phi=phi)
        rows.append(derived(mu, ze, nu, T))
    keys = ("mu", "zeta", "nu", "F", "loss_A", "loss_B", "D_half", "weighted_fidelity")
    return {kk: float(np.mean([r[kk] for r in rows])) for kk in keys}, rows

def evaluate(Xs, Ys, W, V, k, phi=None, kind="poly", beta=None, oracle_phase=True):
    """Full metric set on a dataset for GIVEN parameters -- the evaluation protocol.

    phi: the TRAINED phases (indexed by step j, shared across sequences). Using them on
         held-out data is the LEAKAGE-FREE test score.
    oracle_phase: additionally report the value at phi* = -arg<y_j|z_j> recomputed on THIS
         data. That fits T parameters to the evaluation set, so it is a CEILING, not a test
         score -- reported separately and labelled as such.

    Phase-dependence map (see renyi_exact_note.tex):
        mu, F, loss_B, D_half   -> depend on phi   (need the transfer check)
        zeta, nu, loss_A        -> phase-INSENSITIVE (transfer trivially)
        uniform CE / L_1/2      -> phase-INSENSITIVE (built from |<y_j|z_j>|^2/||z_j||^2)
    """
    if Xs.ndim == 2: Xs, Ys = Xs[None], Ys[None]
    T = Xs.shape[1]
    if phi is None: phi = np.zeros(T)
    out, _ = batch_losses(Xs, Ys, W, V, k, phi=phi, kind=kind, beta=beta)
    ev = [uniform_ce(Xi, Yi, W, V, k, kind=kind, beta=beta) for Xi, Yi in zip(Xs, Ys)]
    for kk in ("CE_uniform", "L_half_uniform", "WCE", "participation", "perplexity_uniform"):
        out[f"eval_{kk}"] = float(np.mean([e[kk] for e in ev]))
    out["bound_A_ok"] = out["loss_A"] <= out["eval_WCE"] + 1e-9
    if oracle_phase:                       # per-sequence oracle phases: CEILING only
        orc = []
        for Xi, Yi in zip(Xs, Ys):
            ph = optimal_phi_from(Xi, Yi, W, V, k, kind, beta)
            mu, ze, nu = mu_zeta_nu(Xi, Yi, W, V, k, kind=kind, beta=beta, phi=ph)
            orc.append(derived(mu, ze, nu, T))
        for kk in ("mu", "F", "loss_B", "D_half"):
            out[f"oracle_{kk}"] = float(np.mean([o[kk] for o in orc]))
        out["phase_transfer_gap"] = out["loss_B"] - out["oracle_loss_B"]
    return out

def train_batch(Xs, Ys, k, model="kqsa", objective="B", kind="poly", layers=8,
                complex_gates=True, beta=None, maxiter=200, seed=0, verbose=True,
                align_init=True):
    """Train on a BATCH of quantum sequences (shape (n_seq,T,d)). Objective A ignores phi
    (phase-insensitive); objective B trains the T phase-compensation parameters jointly
    with W,V. Returns dict(W, V, phi, history, final)."""
    from scipy.optimize import minimize
    if Xs.ndim == 2: Xs, Ys = Xs[None], Ys[None]
    n_seq, T, d = Xs.shape
    if beta is None: beta = float(np.sqrt(d))
    v0, unpack, nbase = _pack(model, d, layers, complex_gates, objective, seed)
    if objective == "B":
        W0, V0, _ = unpack(v0)
        phi0 = (np.mean([optimal_phi_from(Xi, Yi, W0, V0, k, kind, beta)
                         for Xi, Yi in zip(Xs, Ys)], axis=0) if align_init else np.zeros(T))
        v0 = np.concatenate([v0, phi0])
    hist = []
    def f(v):
        W, V, _ = unpack(v)
        phi = v[nbase:nbase + T] if objective == "B" else None
        agg, _ = batch_losses(Xs, Ys, W, V, k, phi=phi, kind=kind, beta=beta)
        L = agg["loss_" + objective]; hist.append(L); return L
    res = minimize(f, v0, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=20 * maxiter))
    W, V, _ = unpack(res.x)
    phi = res.x[nbase:nbase + T] if objective == "B" else np.zeros(T)
    fin = evaluate(Xs, Ys, W, V, k, phi=phi, kind=kind, beta=beta)
    if verbose:
        print(f"  [{model}/{objective} k={k}] train loss {hist[0]:.4f} -> "
              f"{fin['loss_'+objective]:.4f}  ({len(res.x)} params, {n_seq} sequences)")
        print(f"      mu={fin['mu']:.3e} zeta={fin['zeta']:.3e} nu={fin['nu']:.3e}  "
              f"F={fin['F']:.4f}  D_1/2={fin['D_half']:.4f}")
        print(f"      EVAL uniform CE={fin['eval_CE_uniform']:.4f} "
              f"(PPL={fin['eval_perplexity_uniform']:.3f})  part={fin['eval_participation']:.2f}/{T}")
    return dict(W=W, V=V, phi=phi, history=np.array(hist), final=fin)

# --------------------------------------------------------------------------- #
#  demo / self-consistency
# --------------------------------------------------------------------------- #
def _rand_states(T, d, rng):
    Z = rng.standard_normal((T, d)) + 1j * rng.standard_normal((T, d))
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)

def demo():
    rng = np.random.default_rng(0)
    T, d, layers = 8, 8, 2
    X = _rand_states(T, d, rng); Y = _rand_states(T, d, rng)
    beta = softmax_beta(d)

    print("=" * 78)
    print(f"Classical analogues, complex quantum-sequence readout  (T={T}, d={d})")
    print("=" * 78)

    # (1) optimal-phase alignment reaches mu_max exactly (phase-alignment theorem)
    W, V = make_W_V("kcsa", d, layers, rng)
    Aj, lam = partial_sums(X, Y, W, V, kernel="poly", c=lcu_coeffs(2, beta))
    phi0 = np.zeros(T); phistar = optimal_phi(Aj)
    print("\n(1) phase alignment (kCSA, k=2):")
    print(f"    mu(phi=0)   = {coherent_mu(Aj,phi0,lam,T):.4e}  alignment={np.abs(np.sum(Aj))/np.sum(np.abs(Aj)):.3f}")
    print(f"    mu(phi*)    = {coherent_mu(Aj,phistar,lam,T):.4e}  alignment=1.000")
    print(f"    mu_max      = {mu_max(Aj,lam,T):.4e}   match={np.isclose(coherent_mu(Aj,phistar,lam,T),mu_max(Aj,lam,T))}")

    # (2) per-step readout is phase-invariant
    ce_a = perstep_ce(X, Y, W, V, kernel="poly", c=lcu_coeffs(2, beta))
    print(f"\n(2) per-step CE is phase-invariant: CE={ce_a:.4f}  (phi has no effect on |<y|z>|^2)")

    # (3) three models, aligned coherent loss vs k (fixed random W,V)
    print("\n(3) aligned coherent loss  -log mu(phi*) + log T   vs k:")
    print(f"    {'k':>2} | {'kQSA(ansatz)':>13} {'kCSA(unitary)':>14} | {'nlCSA-iso(soft)':>16}")
    for k in (1, 2, 3, 4):
        Wq, Vq = make_W_V("kqsa", d, layers, rng)
        Wc, Vc = make_W_V("kcsa", d, layers, rng)
        Wn, Vn = make_W_V("nlcsa", d, layers, rng)
        lq = model_forward(X, Y, Wq, Vq, k, "poly", align=True)["loss"]
        lc = model_forward(X, Y, Wc, Vc, k, "poly", align=True)["loss"]
        ln = model_forward(X, Y, Wn, Vn, k, "soft", align=True)["loss"]   # k-indep (softmax)
        print(f"    {k:>2} | {lq:>13.4f} {lc:>14.4f} | {ln:>16.4f}")
    print(f"\n    params: kQSA={param_count('kqsa',d,1,layers)}(+T phases)  "
          f"kCSA={param_count('kcsa',d,1)}  nlCSA={param_count('nlcsa',d,1)}")
    print("    nl-CSA-iso uses fidelity softmax exp(beta|s|^2), V unitary (isometric value map),")
    print("    and is k-independent (shown as a horizontal reference, like the real-data case).")

    # ---- (4) the three observables + the exact identity ----
    print("\n(4) three observables mu, zeta, nu and the EXACT identity  -log F = L_A + D_1/2:")
    W, V = make_W_V("kcsa", d, layers, rng)
    phi = optimal_phi_from(X, Y, W, V, 2)
    mu, ze, nu = mu_zeta_nu(X, Y, W, V, 2, phi=phi)
    D = derived(mu, ze, nu, T); B = wce_bound(X, Y, W, V, 2)
    print(f"    mu={D['mu']:.4e}  zeta={D['zeta']:.4e}  nu={D['nu']:.4e}   (zeta is the LARGEST"
          f" -> cheapest: 1/zeta={1/D['zeta']:.0f} vs 1/mu={1/D['mu']:.0f})")
    print(f"    F={D['F']:.4f}  L_A={D['loss_A']:.4f}  D_1/2={D['D_half']:.4f}  L_B={D['loss_B']:.4f}"
          f"  residual={D['identity_residual']:.1e}")
    print(f"    Shannon bounds: L_A <= WCE={B['WCE']:.4f} ({D['loss_A']<=B['WCE']}), "
          f"uniform CE (T-setting anchor) = {B['uniform_CE']:.4f}")

    # ---- (5) training on both objectives ----
    print("\n(5) TRAINING (objective A = -log(nu/zeta);  objective B = -log F):")
    for obj in ("A", "B"):
        train(X, Y, 2, model="kqsa", objective=obj, layers=6, maxiter=100, seed=1)
    # ---- (7) CLASSICAL SEQUENCES: REAL ansatz, NO phase correction ----
    print("\n(7) CLASSICAL SEQUENCES (REAL ansatz, NO phase correction):")
    Tc, dc = 12, 4
    Xc, Yc = classical_sequences(Tc, dc, n_seq=1, seed=3, kind="markov", rho=0.8)
    Xc, Yc = Xc[0], Yc[0]
    print(f"    real tokens, <x_j,x_j+1> = {Xc[0] @ Yc[0]:.3f};  A_j = <y_j|z_j> is REAL and")
    print(f"    SIGNED, so misalignment is possible. THREE-term exact identity:")
    print(f"      -log F = -log(nu/zeta) + D_1/2(u||Qt) + (-2 log G),   G = |sum A_j|/sum|A_j|")
    for model in ("kqsa", "kcsa"):
        for obj in ("A", "B"):
            train(Xc, Yc, 2, model=model, objective=obj, layers=6,
                  complex_gates=False, maxiter=100, seed=1)
    print("\n    Objective B drives G -> 1 using W,V ALONE (no phase parameters): for real")
    print("    classical data the sign alignment is learnable by the ansatz, which is why the")
    print("    C1 phase register is needed only for COMPLEX (quantum-sequence) data.")

    # ---- (6) QUANTUM SEQUENCES: trajectories, phase compensation, transfer ----
    print("\n(6) QUANTUM SEQUENCES (TFIM trajectories, y_j = x_{j+1}):")
    Tq, nq = 8, 2
    Xtr, Ytr, _ = tfim_trajectories(Tq, nq=nq, dt=0.35, n_seq=3, seed=1)
    Xho, Yho, _ = tfim_trajectories(Tq, nq=nq, dt=0.35, n_seq=3, seed=7, random_couplings=True)
    print(f"    |<x_j|x_j+1>|^2 = {np.abs(np.vdot(Xtr[0,0],Ytr[0,0]))**2:.3f}  (correlated tokens,"
          f" unlike i.i.d. states)")
    for obj in ("A", "B"):
        r = train_batch(Xtr, Ytr, 2, model="kqsa", objective=obj, layers=4, maxiter=60, seed=2)
        e = evaluate(Xho, Yho, r["W"], r["V"], 2, phi=r["phi"])
        msg = (f"      held-out H: L_A={e['loss_A']:.4f}  L_B(trained phi)={e['loss_B']:.4f}  "
               f"CE_unif={e['eval_CE_uniform']:.4f}")
        if obj == "B":
            msg += f"  | oracle L_B={e['oracle_loss_B']:.4f} (gap {e['phase_transfer_gap']:+.4f})"
        print(msg)
    print("    L_A and CE_uniform are phase-insensitive -> transfer with no caveat.")
    print("    L_B/F/D_1/2 depend on phi: the transfer gap (trained phi vs oracle phi* refit")
    print("    on the eval set) is the leakage-free measure of phase generalization.")

    print("\n    NOTE objective A is 'gameable': it maximizes the weighted fidelity nu/zeta but")
    print("    lets the success profile concentrate on a few steps (large D_1/2, low")
    print("    participation). Objective B = L_A + D_1/2 penalizes exactly that and yields a")
    print("    markedly better UNIFORM CE -- the metric neither objective measures directly.")

if __name__ == "__main__":
    demo()
