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

if __name__ == "__main__":
    demo()
