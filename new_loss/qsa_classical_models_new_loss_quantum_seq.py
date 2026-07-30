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

def make_W_V(model, d, layers, rng, Hs=None):
    """Instantiate (W,V) for a model. kQSA: complex ansatz; kCSA/nlCSA: free unitary."""
    n = max(1, ceil(log2(d)))
    if model == "kqsa":
        Wp = rng.standard_normal((layers, n, 3)); Vp = rng.standard_normal((layers, n, 3))
        return complex_ansatz_matrix(Wp, n)[:d, :d], complex_ansatz_matrix(Vp, n)[:d, :d]
    elif model in ("kcsa", "nlcsa"):
        return random_unitary(d, rng), random_unitary(d, rng)     # free unitary (iso value map)
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
    else:                                   # kcsa / nlcsa : free unitary via W=expm(iH)
        nH = d * d
        p0 = [0.1 * rng.standard_normal(nH), 0.1 * rng.standard_normal(nH)]
        def unpack(v):
            H1 = v[:nH].reshape(d, d); H2 = v[nH:2 * nH].reshape(d, d)
            return _expi(H1), _expi(H2), 2 * nH
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

def loss_fn(v, unpack, nbase, X, Y, k, T, objective, kind="poly", beta=None):
    """The scalar objective. objective='A' -> -log(nu/zeta); 'B' -> -log F."""
    W, V, _ = unpack(v)
    phi = v[nbase:nbase + T] if objective == "B" else None
    mu, zeta, nu = mu_zeta_nu(X, Y, W, V, k, kind=kind, beta=beta, phi=phi)
    if objective == "A":
        return -np.log(max(float(nu / zeta), 1e-300))
    pref = (T + 1) / (2 * T)
    return -np.log(max(float(pref * mu / zeta), 1e-300))

def train(X, Y, k, model="kqsa", objective="B", kind="poly", layers=8, complex_gates=True,
          beta=None, maxiter=300, seed=0, verbose=True, align_init=True):
    """Train W,V (+phi for objective B) on objective A or B.

    Returns dict(history, final, W, V, phi, objective, ...). The reported loss is the
    objective's own loss; `final` also carries mu, zeta, nu, F, D_half and the Shannon
    bound WCE so the exact inequalities can be checked at the optimum."""
    from scipy.optimize import minimize
    T, d = X.shape
    if beta is None: beta = float(np.sqrt(d))
    v0, unpack, nbase = _pack(model, d, layers, complex_gates, objective, seed)
    if objective == "B":
        W0, V0, _ = unpack(v0)
        phi0 = optimal_phi_from(X, Y, W0, V0, k, kind, beta) if align_init else np.zeros(T)
        v0 = np.concatenate([v0, phi0])
    hist = []
    def f(v):
        L = loss_fn(v, unpack, nbase, X, Y, k, T, objective, kind, beta)
        hist.append(L); return L
    res = minimize(f, v0, method="L-BFGS-B",
                   options=dict(maxiter=maxiter, maxfun=20 * maxiter))
    W, V, _ = unpack(res.x)
    phi = res.x[nbase:nbase + T] if objective == "B" else np.zeros(T)
    mu, zeta, nu = mu_zeta_nu(X, Y, W, V, k, kind=kind, beta=beta, phi=phi)
    out = derived(mu, zeta, nu, T); out.update(wce_bound(X, Y, W, V, k, kind=kind, beta=beta))
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
              f"L_B={out['loss_B']:.4f} = L_A+D_1/2 (res {out['identity_residual']:.1e})")
        print(f"      EVAL (uniform, T settings): CE={out['eval_CE_uniform']:.4f} "
              f"(PPL={out['eval_perplexity_uniform']:.3f})  L_1/2={out['eval_L_half_uniform']:.4f}  "
              f"participation={out['eval_participation']:.2f}/{T}")
    return dict(history=np.array(hist), final=out, W=W, V=V, phi=phi)

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
    print("\n    NOTE objective A is 'gameable': it maximizes the weighted fidelity nu/zeta but")
    print("    lets the success profile concentrate on a few steps (large D_1/2, low")
    print("    participation). Objective B = L_A + D_1/2 penalizes exactly that and yields a")
    print("    markedly better UNIFORM CE -- the metric neither objective measures directly.")

if __name__ == "__main__":
    demo()
