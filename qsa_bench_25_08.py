"""
qsa_bench.py -- clean-slate, audit-friendly benchmark for overlap-interference
                quantum attention and its classical analogues.

DESIGN PRINCIPLE (why the numbers are comparable beyond reasonable doubt)
------------------------------------------------------------------------
There is exactly ONE forward pass (`forward`), ONE observable routine (`observables`)
and ONE metric routine (`metrics`), used by EVERY model. A model is nothing but

        (parameter vector)  --model.build-->  (W, V)      +      a kernel name.

Nothing else differs. In particular the reported L1 (uniform Shannon cross-entropy) is
computed by the same three lines for kQSA, kCSA and nlCSA, so cross-model comparison
cannot be contaminated by a convention mismatch. The chance level is obtained by running
the SAME code path with randomised parameters.

MODELS
------
  kqsa-mono / kqsa-poly   W,V from the complex RX-RY-RZ+CNOT ansatz (O(L log d) angles)
                          trained on L_B ; L1 reported
  kcsa-mono / kcsa-poly   W,V free unitary, W=exp(iH)  (O(d^2) params)
                          trained on L_B ; L1 reported
  nlcsa-iso               softmax kernel, V unitary (isometric value map), trained on L1
  nlcsa-gen               softmax kernel, V a general complex matrix,      trained on L1

Complex ansatz throughout, for classical AND quantum sequences: by the phase-alignment
theorem the trainable phases on the prediction register can always be set to their
closed-form optimum phi_j* = -arg<y_j|z_j>, so a complex ansatz never costs anything.
`phase_mode='analytic'` (default) substitutes phi*; `phase_mode='free'` optimises the T
phases explicitly and is provided to verify the theorem numerically.

READOUT (fast / overlap route, no uncomputation)
------------------------------------------------
With z_j = sum_{i<=j} g(s_ij) V x_i , A_j = <y_j|z_j> , w_j = ||z_j|| , f_j = |A_j|/w_j:

    mu   = |sum_j e^{i phi_j} A_j|^2 / (lam Ntri)^2      [phase-sensitive]
    zeta =  sum_j w_j^2             / (lam^2 T Ntri)     [phase-insensitive]
    nu   =  sum_j |A_j|^2           / (lam^2 T Ntri)     [phase-insensitive]

    F      = (T+1)/(2T) * mu/zeta   in [0,1]      (state fidelity)
    L_B    = -log F                              (training objective; 2 observables)
    L_A    = -log(nu/zeta)                       (alternative objective)
    D_half = L_B - L_A = D_{1/2}(u_T || Qt)      (exact identity)
    L1     = -(1/T) sum_j log f_j^2              (uniform Shannon CE; COMMON AXIS)
    L_half = -2 log( mean_j f_j )                (uniform Renyi-1/2 CE; nl-CSA's own loss)

Note the different prefactors: mu carries (lam Ntri)^-2 (both control registers are
un-prepared), zeta and nu carry (lam^2 T Ntri)^-1 (only C2 is un-prepared).

Run:  python qsa_bench.py            # full benchmark, classical + quantum sequences
      python qsa_bench.py --selftest # invariants and identities only
"""
from __future__ import annotations

import sys
from math import ceil, comb, factorial, log2

import numpy as np
from scipy.special import digamma

# --------------------------------------------------------------------------- #
#  0. optional JAX backend (analytic gradients). Falls back to scipy + finite  #
#     differences, which is correct but slower and budget-sensitive.           #
# --------------------------------------------------------------------------- #
try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except Exception:  # pragma: no cover
    jnp = np
    HAS_JAX = False


# --------------------------------------------------------------------------- #
#  1. data                                                                     #
# --------------------------------------------------------------------------- #
def classical_sequences(T, d, n_seq=1, seed=0, rho=0.8, kind="markov"):
    """REAL classical token sequences on the unit sphere, y_j = x_{j+1}.
    kind='markov': x_{j+1} ~ normalize(rho x_j + sqrt(1-rho^2) eps)  (correlated)
    kind='iid'   : independent uniform directions (no temporal structure).
    Returned as complex dtype so that one code path serves both data types."""
    rng = np.random.default_rng(seed)
    Xs = np.zeros((n_seq, T, d)), np.zeros((n_seq, T, d))
    X, Y = Xs
    for s in range(n_seq):
        v = rng.standard_normal(d)
        traj = [v / np.linalg.norm(v)]
        for _ in range(T):
            nxt = (rng.standard_normal(d) if kind == "iid"
                   else rho * traj[-1] + np.sqrt(1 - rho ** 2) * rng.standard_normal(d))
            traj.append(nxt / np.linalg.norm(nxt))
        traj = np.array(traj)
        X[s], Y[s] = traj[:-1], traj[1:]
    return X.astype(complex), Y.astype(complex)


def tfim_hamiltonian(nq, J=1.0, h=1.0, rng=None, random_couplings=False):
    """H = -sum_q h_q X_q - sum_q J_q Z_q Z_{q+1}  on nq qubits (open chain)."""
    I2 = np.eye(2)
    Xp = np.array([[0, 1], [1, 0]], dtype=complex)
    Zp = np.diag([1, -1]).astype(complex)

    def op(P, q):
        o = np.eye(1, dtype=complex)
        for i in range(nq):
            o = np.kron(o, P if i == q else I2)
        return o

    H = np.zeros((2 ** nq, 2 ** nq), dtype=complex)
    for q in range(nq):
        H -= (h * (0.5 + rng.random()) if random_couplings else h) * op(Xp, q)
    for q in range(nq - 1):
        H -= (J * (0.5 + rng.random()) if random_couplings else J) * (op(Zp, q) @ op(Zp, q + 1))
    return H


def quantum_sequences(T, nq=3, dt=0.3, n_seq=1, seed=0, random_couplings=False, H=None):
    """Unitary (TFIM) trajectories: x_j = psi_j, y_j = psi_{j+1} = U psi_j, d = 2^nq.
    Pass H (or random_couplings=True with a new seed) to build a held-out test set."""
    from scipy.linalg import expm

    rng = np.random.default_rng(seed)
    if H is None:
        H = tfim_hamiltonian(nq, rng=rng, random_couplings=random_couplings)
    U = expm(-1j * H * dt)
    d = 2 ** nq
    X = np.zeros((n_seq, T, d), dtype=complex)
    Y = np.zeros_like(X)
    for s in range(n_seq):
        p = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        p /= np.linalg.norm(p)
        traj = [p]
        for _ in range(T):
            traj.append(U @ traj[-1])
        traj = np.array(traj)
        X[s], Y[s] = traj[:-1], traj[1:]
    return X, Y, H






# =========================================================================== #
#  1a. DATA INGESTION -- externally supplied UNNORMALIZED tokens               #
# =========================================================================== #
#  This is the entry point for real data. You supply unnormalized tokens; the code
#  derives everything the hybrid readout needs:
#
#      alpha_l = ||x^(l)||^2          leverage score  (must be <= 1)
#      xi_l    = x^(l)/sqrt(alpha_l)  amplitude encoding actually loaded on the device
#      alpha_t                        target leverage per step -> the single R_t control
#
#  Two equivalent ways in:
#    (A) dataset_from_tokens(Xseq, Yseq)      -- token SEQUENCES, shape (n_seq, T, d)
#    (B) load_vocabulary(Xvoc) + dataset_from_indices(vocab, idx)   -- vocabulary + labels
#
#  (A) is the minimum needed to train. (B) additionally lets us check the Parseval
#  condition sum_l |x><x| = I_d, which is what makes sum_l p^(l) = 1 and guarantees
#  alpha_l <= 1; without it the R_t amplitude sqrt(alpha) may exceed 1 and be
#  unphysical, and the "probability" is not normalized over the vocabulary.
# =========================================================================== #

def load_vocabulary(Xvoc, whiten=False, tol=1e-6, verbose=True):
    """Ingest an externally supplied UNNORMALIZED vocabulary embedding.

    Xvoc : array (D, d), row l = the unnormalized classical token x^(l) (real or complex).
    whiten : if True, apply x -> S^{-1/2} x (S = sum_l |x><x|) to enforce Parseval exactly.
             This CHANGES the tokens; off by default so that what you supply is what is used.

    Returns a dict with Xvoc, Xi (normalized), alpha, and frame diagnostics."""
    X = np.asarray(Xvoc)
    if X.ndim != 2:
        raise ValueError(f"Xvoc must be (D,d); got {X.shape}")
    D, d = X.shape
    S = X.conj().T @ X
    dev = float(np.linalg.norm(S - np.eye(d)))
    if whiten:
        ev, U = np.linalg.eigh((S + S.conj().T) / 2)
        if ev.min() <= 0:
            raise ValueError("frame operator is singular; tokens do not span C^d")
        X = X @ (U * ev ** -0.5) @ U.conj().T
        S = X.conj().T @ X
        dev = float(np.linalg.norm(S - np.eye(d)))
    alpha = np.sum(np.abs(X) ** 2, axis=1).real
    if np.any(alpha <= 0):
        raise ValueError("some tokens have zero norm; alpha must be > 0 on the used support")
    Xi = X / np.sqrt(alpha)[:, None]
    ok_parseval = dev < tol
    ok_cap = bool(alpha.max() <= 1 + 1e-9)
    if verbose:
        print(f"  vocabulary: D={D}, d={d}   ||sum_l |x><x| - I_d|| = {dev:.3e}"
              f"  {'OK' if ok_parseval else '<-- NOT Parseval'}")
        print(f"    alpha in [{alpha.min():.4e}, {alpha.max():.4e}]  sum={alpha.sum():.4f}"
              f" (Parseval would give d={d})  mean={alpha.mean():.4e}")
        if not ok_cap:
            print("    !! alpha_max > 1: sqrt(alpha) is not a valid R_t amplitude.")
            print("       Rescale the embedding globally, or call load_vocabulary(..., whiten=True).")
        if not ok_parseval:
            print("    !! sum_l p^(l) != 1: the vocabulary readout is not normalized.")
            print("       Use whiten=True (changes tokens) or supply a Parseval frame.")
    return dict(Xvoc=X, Xi=Xi, alpha=alpha, S=S, parseval_dev=dev,
                ok_parseval=ok_parseval, ok_cap=ok_cap, D=D, d=d)


def dataset_from_tokens(Xseq, Yseq=None):
    """Build a dataset directly from UNNORMALIZED token sequences.

    Xseq : (n_seq, T, d) unnormalized context tokens  (or (T, d) for a single sequence)
    Yseq : (n_seq, T, d) unnormalized target tokens. If None, the targets are taken to be
           the next tokens of Xseq, i.e. Xseq must have T+1 steps and is split into
           X = Xseq[:, :-1], Y = Xseq[:, 1:].

    Returns (X, Y, alpha_t) with X, Y NORMALIZED (what the device loads) and alpha_t the
    target leverage scores (what the R_t control applies)."""
    X = np.asarray(Xseq, dtype=complex)
    if X.ndim == 2:
        X = X[None]
    if Yseq is None:
        if X.shape[1] < 2:
            raise ValueError("need at least T+1=2 steps to form next-token targets")
        X, Y = X[:, :-1], X[:, 1:]
    else:
        Y = np.asarray(Yseq, dtype=complex)
        if Y.ndim == 2:
            Y = Y[None]
    if X.shape != Y.shape:
        raise ValueError(f"X {X.shape} and Y {Y.shape} must match")
    nx = np.linalg.norm(X, axis=2, keepdims=True)
    ny = np.linalg.norm(Y, axis=2, keepdims=True)
    if np.any(nx == 0) or np.any(ny == 0):
        raise ValueError("zero-norm token in the sequences")
    alpha_t = (ny[..., 0] ** 2).real                       # (n_seq, T)
    if alpha_t.max() > 1 + 1e-9:
        print(f"  [warn] max target alpha = {alpha_t.max():.4f} > 1: sqrt(alpha) is not a "
              f"valid R_t amplitude. Rescale the embedding globally.")
    return X / nx, Y / ny, alpha_t


def dataset_from_indices(vocab, idx):
    """Build a dataset from a loaded vocabulary and integer token labels.
    idx : (n_seq, T+1) integer array of vocabulary labels.
    Returns (X, Y, alpha_t) as in dataset_from_tokens."""
    idx = np.atleast_2d(np.asarray(idx, dtype=int))
    Xi, alpha = vocab["Xi"], vocab["alpha"]
    X = Xi[idx[:, :-1]]
    Y = Xi[idx[:, 1:]]
    alpha_t = alpha[idx[:, 1:]]
    return X, Y, alpha_t


def run_from_tokens(Xseq, Yseq=None, Xseq_test=None, Yseq_test=None, Xvoc=None,
                    k=2, layers=8, maxiter=300, seed=0, n_chance=80,
                    title="USER TOKENS", mu_csv=None, ks=None):
    """One-call entry point for externally supplied UNNORMALIZED tokens.

        rows = run_from_tokens(Xseq, Yseq, Xseq_test, Yseq_test, Xvoc=my_embedding)

    Xseq/Yseq      : (n_seq, T, d) unnormalized tokens (or Xseq with T+1 steps and Yseq=None)
    Xvoc           : optional (D, d) unnormalized vocabulary, used only for the Parseval /
                     leverage-cap diagnostics; training does not need it.
    ks             : if given, also run mu_sweep over these k (for the mu plots).
    """
    vocab = load_vocabulary(Xvoc) if Xvoc is not None else None
    X, Y, at = dataset_from_tokens(Xseq, Yseq)
    Xte = Yte = None
    if Xseq_test is not None:
        Xte, Yte, _ = dataset_from_tokens(Xseq_test, Yseq_test)
    rows = run_benchmark(X, Y, k=k, layers=layers, maxiter=maxiter, seed=seed,
                         n_chance=n_chance, title=title, complex_tokens=True,
                         Xte=Xte, Yte=Yte, alpha_t=at,
                         alpha=(vocab["alpha"] if vocab else None),
                         D=(vocab["D"] if vocab else None))
    if ks:
        print("\n  mu sweep (for the mu plots):")
        mu_sweep(X, Y, ks=ks, layers=layers, maxiter=maxiter, seed=seed,
                 alpha_t=at, csv_path=mu_csv)
    return rows

# --------------------------------------------------------------------------- #
#  1b. SYNTHETIC generators (SELF-TEST ONLY -- real runs use Sec. 1a)                    #
# --------------------------------------------------------------------------- #
#  The classical embedded tokens are generally UNNORMALIZED:
#      |x^(l)> = A^dag |w^(l)>,   alpha_l = || x^(l) ||^2,   |xi^(l)> = |x^(l)>/sqrt(alpha_l)
#  with A: C^d -> C^D an isometry (A^dag A = I_d). Parseval then gives, for free,
#      sum_l |x^(l)><x^(l)| = A^dag A = I_d          (a POVM: no tight-frame assumption)
#      0 <= alpha_l <= 1,   sum_l alpha_l = d        (so the MEAN alpha is d/D)
#  A quantum register can only hold the normalized xi^(l). The classical norms are
#  restored inside the circuit by three one-qubit filters (appendix Sec. 2):
#      R_i  (controlled by C2,P):  |0> -> alpha_i^((p+1)/2) |0> + ... |1>   [value + p keys]
#      R_j  (controlled by C1,P):  |0> -> alpha_j^(p/2)     |0> + ... |1>   [p query bras]
#      R_t  (controlled by C1  ):  |0> -> alpha_t^(1/2)     |0> + ... |1>   [known target]
#  On the joint good branch R_i=R_j=R_t=0 these cancel the alpha's introduced by the
#  normalized loads and reproduce EXACTLY the classical unnormalized quantities
#      s_ij = <x_j|W|x_i>,   a_ij = <x_{j+1}|V|x_i>,   z_j = sum_i g(s_ij) V x_i .
#  Consequently the shared forward pass below needs no modification: feeding it
#  unnormalized tokens computes precisely what the filtered circuit measures. The filters
#  are not transparent for the SHOT BUDGET, however -- see flag_success_probability().
# --------------------------------------------------------------------------- #

def anti_embedding_tokens(D, d, seed=0, concentration=0.0, complex_tokens=True):
    """Unnormalized classical tokens from an isometric anti-embedding A (A^dag A = I_d).

    Returns (Xvoc, alpha) with Xvoc of shape (D, d) (rows = x^(l)) and alpha = ||x^(l)||^2.
    Guarantees sum_l |x><x| = I_d exactly and sum_l alpha_l = d.

    concentration >= 0 tilts the norm budget towards low-index tokens (a crude stand-in for
    an embedding that allocates more norm to frequent words); concentration=0 gives the
    generic isometry, for which alpha_l concentrates around d/D."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((D, d))
    if complex_tokens:
        M = M + 1j * rng.standard_normal((D, d))
    if concentration > 0:                      # reweight rows before orthonormalising
        wgt = np.exp(-concentration * np.arange(D) / D)
        M = M * wgt[:, None]
    Q, _ = np.linalg.qr(M)                     # Q: D x d with orthonormal COLUMNS
    Xvoc = Q.conj()                            # rows x^(l); sum_l |x><x| = Q^dag Q = I_d
    alpha = np.sum(np.abs(Xvoc) ** 2, axis=1)
    return Xvoc, alpha


def vocabulary_sequences(T, D, d, n_seq=1, seed=0, concentration=0.0,
                         freq_exponent=0.0, complex_tokens=True):
    """Token-index sequences drawn from a Zipf(freq_exponent) frequency law over the
    vocabulary, mapped to UNNORMALIZED embedded tokens. Returns (X, Y, alpha, idx) with
    X[s,j] = x^(l_j) and Y[s,j] = x^(l_{j+1}) (next-token targets)."""
    Xvoc, alpha = anti_embedding_tokens(D, d, seed=seed, concentration=concentration,
                                        complex_tokens=complex_tokens)
    rng = np.random.default_rng(seed + 1)
    ranks = np.arange(1, D + 1)
    f = ranks ** (-freq_exponent) if freq_exponent > 0 else np.ones(D)
    f = f / f.sum()
    X = np.zeros((n_seq, T, d), dtype=complex)
    Y = np.zeros_like(X)
    idx = np.zeros((n_seq, T + 1), dtype=int)
    for s_ in range(n_seq):
        ii = rng.choice(D, size=T + 1, p=f)
        idx[s_] = ii
        X[s_] = Xvoc[ii[:-1]]
        Y[s_] = Xvoc[ii[1:]]
    return X, Y, alpha, idx


def filter_amplitudes(alpha_i, alpha_j, alpha_t, p):
    """Good-branch amplitudes of the three normalization filters for a degree-p term.
       R_i -> alpha_i^((p+1)/2)   R_j -> alpha_j^(p/2)   R_t -> alpha_t^(1/2)
    All are valid amplitudes because 0 <= alpha <= 1."""
    return alpha_i ** ((p + 1) / 2), alpha_j ** (p / 2), np.sqrt(alpha_t)


def branch_amplitude_filtered(xi_i, xi_j, xi_t, alpha_i, alpha_j, alpha_t, W, V, p):
    """Explicit simulation of ONE circuit branch (i,j,p) using ONLY normalized states and
    the filter amplitudes -- i.e. what the hardware actually does. Must equal the classical
    a_ij * s_ij^p built from the unnormalized tokens."""
    Ri, Rj, Rt = filter_amplitudes(alpha_i, alpha_j, alpha_t, p)
    key = np.vdot(xi_j, W @ xi_i) ** p          # p key copies, normalized states
    tgt = np.vdot(xi_t, V @ xi_i)               # target uncomputation, normalized states
    return Ri * Rj * Rt * key * tgt


def verify_filters(d=8, D=64, seed=0, tol=1e-10, verbose=True):
    """Self-test: the filtered normalized-state circuit reproduces the classical
    unnormalized quantities EXACTLY, branch by branch."""
    rng = np.random.default_rng(seed)
    Xvoc, alpha = anti_embedding_tokens(D, d, seed=seed)
    S = Xvoc.conj().T @ Xvoc
    povm_err = np.linalg.norm(S - np.eye(d))
    W = unitary_from_hermitian(0.4 * rng.standard_normal(2 * d * d), d)
    V = unitary_from_hermitian(0.4 * rng.standard_normal(2 * d * d), d)
    worst = 0.0
    for _ in range(60):
        i, j, t = rng.integers(0, D, 3)
        pp = int(rng.integers(0, 5))
        xi = lambda l: Xvoc[l] / np.sqrt(alpha[l])
        got = branch_amplitude_filtered(xi(i), xi(j), xi(t), alpha[i], alpha[j], alpha[t],
                                        W, V, pp)
        s_ij = np.vdot(Xvoc[j], W @ Xvoc[i])          # unnormalized
        a_ij = np.vdot(Xvoc[t], V @ Xvoc[i])
        want = a_ij * s_ij ** pp
        worst = max(worst, abs(got - want) / max(abs(want), 1e-300))
    ok = (povm_err < tol) and (worst < 1e-8)
    if verbose:
        print(f"  filters (d={d},D={D}): ||sum_l|x><x| - I|| = {povm_err:.2e}  "
              f"max rel. branch error = {worst:.2e}  {'OK' if ok else 'FAIL'}")
        print(f"    alpha in [{alpha.min():.4f},{alpha.max():.4f}], "
              f"sum={alpha.sum():.4f} (=d={d}), mean={alpha.mean():.4f} (=d/D={d/D:.4f})")
    return ok


def flag_success_probability(alpha, idx=None, k=2):
    """Probability that all three filters land in their good branch, per degree p<=k.
       P_p = alpha_i^(p+1) alpha_j^p alpha_t  ~  <alpha>^(2p+2)
    If idx (the realised token indices) is given, the average is taken over the tokens that
    ACTUALLY OCCUR -- which is the relevant quantity, not the flat vocabulary mean."""
    a = alpha if idx is None else alpha[np.asarray(idx).ravel()]
    am = float(np.mean(a))
    return {p: am ** (2 * p + 2) for p in range(k + 1)}, am

# --------------------------------------------------------------------------- #
#  2. parameterizations  (params -> matrix).  All complex.                     #
# --------------------------------------------------------------------------- #
def _cnot(a, b, n):
    N = 2 ** n
    M = np.zeros((N, N))
    for x in range(N):
        bits = [(x >> (n - 1 - w)) & 1 for w in range(n)]
        if bits[a] == 1:
            bits[b] ^= 1
        M[sum(bt << (n - 1 - w) for w, bt in enumerate(bits)), x] = 1.0
    return M


def _cnot_ring(n):
    """Precomputed CNOT-ring permutation matrix (constant, no parameters)."""
    M = np.eye(2 ** n)
    if n >= 2:
        for a, b in zip(range(n), list(range(1, n)) + [0]):
            M = _cnot(a, b, n) @ M
    return M


def ansatz_unitary(params, n, ring, xp=np):
    """Complex hardware-efficient ansatz: RZ RY RX per qubit per layer + CNOT ring.
    params: (L, n, 3). Returns a (2^n, 2^n) unitary. Spans U(2^n) up to global phase
    as L grows (dynamical Lie algebra = u(2^n))."""
    L = params.shape[0]
    M = xp.eye(2 ** n, dtype=complex)
    for l in range(L):
        K = xp.eye(1, dtype=complex)
        for q in range(n):
            tx, ty, tz = params[l, q, 0], params[l, q, 1], params[l, q, 2]
            cx, sx = xp.cos(tx / 2), xp.sin(tx / 2)
            cy, sy = xp.cos(ty / 2), xp.sin(ty / 2)
            RX = xp.array([[cx, -1j * sx], [-1j * sx, cx]])
            RY = xp.array([[cy + 0j, -sy + 0j], [sy + 0j, cy + 0j]])
            RZ = xp.array([[xp.exp(-1j * tz / 2), 0], [0, xp.exp(1j * tz / 2)]])
            K = xp.kron(K, RZ @ RY @ RX)
        M = xp.asarray(ring, dtype=complex) @ (K @ M)
    return M


def unitary_from_hermitian(v, d, xp=np):
    """W = exp(i H), H Hermitian built from 2 d^2 reals -> a free unitary."""
    A = v[: d * d].reshape(d, d)
    B = v[d * d: 2 * d * d].reshape(d, d)
    H = A + 1j * B
    H = (H + H.conj().T) / 2
    ev, U = xp.linalg.eigh(H)
    return (U * xp.exp(1j * ev)) @ U.conj().T


def general_matrix(v, d, xp=np):
    """An arbitrary complex d x d matrix from 2 d^2 reals (NOT isometric)."""
    return v[: d * d].reshape(d, d) + 1j * v[d * d: 2 * d * d].reshape(d, d)


# --------------------------------------------------------------------------- #
#  3. kernels                                                                  #
# --------------------------------------------------------------------------- #
def lcu_coeffs(k, beta):
    return np.array([beta ** p / factorial(p) for p in range(k + 1)], dtype=float)


def kernel_and_lambda(S, kind, k, beta, mask, xp=np):
    """K[j,i] = attention weight from the score matrix S[j,i] = <x_j|W|x_i> (complex).
      'mono' : K = S^k                      , lam = 1
      'poly' : K = sum_p (beta^p/p!) S^p    , lam = sum_p beta^p/p!   (LCU subnormalization)
      'soft' : K = softmax_i( beta |S|^2 )  , lam = 1                 (ROW-NORMALIZED)
    The causal mask is applied in all cases."""
    if kind == "mono":
        return (S ** k) * mask, 1.0
    if kind == "poly":
        c = lcu_coeffs(k, beta)
        K = sum(c[p] * S ** p for p in range(k + 1)) * mask
        return K, float(np.sum(np.abs(c)))
    if kind == "soft":
        E = xp.exp(beta * xp.abs(S) ** 2) * mask
        z = xp.sum(E, axis=1, keepdims=True)
        return E / xp.maximum(z, 1e-30), 1.0
    raise ValueError(f"unknown kernel {kind!r}")


# --------------------------------------------------------------------------- #
#  4. THE single forward pass, shared by every model                           #
# --------------------------------------------------------------------------- #
def forward(X, Y, W, V, kind, k, beta, mask, xp=np, alpha_t=None):
    """HYBRID readout (appendix Sec. 2). Attention is built from the NORMALIZED encodings
    xi_i (which is all a quantum register can hold); only the PREDICTION uses the
    unnormalized Parseval-frame target x_{j+1} = sqrt(alpha_{j+1}) xi_{j+1}:

        s_ij = <xi_j|W|xi_i>,        z_j = sum_{i<=j} K_ij V xi_i,     w_j = ||z_j||
        A_j  = <x_{j+1}|G z_j> = sqrt(alpha_{j+1}) <xi_{j+1}|G z_j>
        f_j  = |A_j| / w_j ,          p_j = f_j^2 = model prob. of the correct token

    X, Y are the NORMALIZED tokens; alpha_t (length T) carries the target leverage scores
    alpha_{j+1}. On the circuit this single factor is the one extra control R_t:
        |0>_{R_t} -> sqrt(alpha_{j+1})|0> + sqrt(1-alpha_{j+1})|1>,
    controlled on C_1 = |j>, i.e. the coherent implementation of the single POVM effect
    M_{j+1} = |x_{j+1}><x_{j+1}|. G is absorbed into V (G unitary), as in the appendix.

    Note the resulting CAP p_j <= alpha_{j+1}: a token of small leverage score can never be
    assigned high probability, so both L1 and L_B acquire a floor (see chance/floor helpers)."""
    S = xp.conj(X) @ (W @ X.T)
    K, lam = kernel_and_lambda(S, kind, k, beta, mask, xp)
    Z = K @ (X @ V.T)                       # rows are z_j  (normalized-token values)
    A = xp.sum(xp.conj(Y) * Z, axis=1)      # <xi_{j+1}|G z_j>
    if alpha_t is not None:                 # the R_t control
        A = A * xp.sqrt(xp.asarray(alpha_t))
    w = xp.sqrt(xp.maximum(xp.sum(xp.abs(Z) ** 2, axis=1), 1e-300))
    f = xp.clip(xp.abs(A) / w, 1e-12, 1.0)
    return A, w, f, lam


# --------------------------------------------------------------------------- #
#  5. observables and metrics, shared by every model                           #
# --------------------------------------------------------------------------- #
def observables(A, w, lam, T, phi=None, xp=np):
    """mu, zeta, nu with the circuit's exact prefactors."""
    Ntri = T * (T + 1) / 2
    if phi is None:                          # analytic phase alignment: phi_j = -arg A_j
        coh = xp.sum(xp.abs(A))
    else:
        coh = xp.abs(xp.sum(xp.exp(1j * phi) * A))
    mu = (coh / (lam * Ntri)) ** 2
    zeta = xp.sum(w ** 2) / (lam ** 2 * T * Ntri)
    nu = xp.sum(xp.abs(A) ** 2) / (lam ** 2 * T * Ntri)
    return mu, zeta, nu


def metrics(A, w, f, lam, T, phi=None, xp=np, alpha_t=None):
    """Every reported quantity, from one forward pass. Keys:
       L_B, L_A, D_half, F           -- objective side (needs mu, zeta[, nu])
       L1, L_half                    -- COMMON AXIS (uniform, phase-insensitive)
       mu, zeta, nu, G               -- raw observables and the alignment factor
    """
    mu, zeta, nu = observables(A, w, lam, T, phi, xp)
    F = (T + 1) / (2 * T) * mu / zeta
    L_B = -xp.log(xp.maximum(F, 1e-300))
    L_A = -xp.log(xp.maximum(nu / zeta, 1e-300))
    L1 = -xp.mean(xp.log(f ** 2))
    L_half = -2 * xp.log(xp.maximum(xp.mean(f), 1e-300))
    denom = xp.maximum(xp.sum(xp.abs(A)), 1e-300)
    G = (xp.abs(xp.sum(A)) / denom if phi is None
         else xp.abs(xp.sum(xp.exp(1j * phi) * A)) / denom)
    out = dict(mu=mu, zeta=zeta, nu=nu, F=F, L_B=L_B, L_A=L_A,
               D_half=L_B - L_A, L1=L1, L_half=L_half, G=G)
    out.update(excess_losses(w, f, T, alpha_t, xp))
    return out


def excess_losses(w, f, T, alpha_t=None, xp=np):
    """CAPACITY-NORMALIZED EXCESS LOSSES (appendix Sec. 'Capacity-normalized excess losses').

    The hybrid decoder has a nonzero floor because p_j <= alpha_j^tar. Separate the
    architectural capacity from what better prediction can still fix. With

        alpha_j^tar = alpha_{j+1},   rho_j = p_j/alpha_j^tar = |<xi_{j+1}|G z_j>|^2/w_j^2
        alpha_ar = mean_j alpha_j^tar,   alpha_geo = exp(mean_j log alpha_j^tar)
        a_j = alpha_j^tar/(T alpha_ar),  q_j = w_j^2/sum_l w_l^2,
        q^(alpha)_j = q_j alpha_j^tar / sum_l q_l alpha_l^tar

    the definitions are
        L_B^excess = L_B + log alpha_ar   = -log(F/alpha_ar) = -2 log sum_j sqrt(a_j q_j rho_j)
        L_1^excess = L_1  + log alpha_geo = -(1/T) sum_j log rho_j = CE_1(u_T || rho)

    with the exact decomposition
        L_B^excess = D_{1/2}(a||q) + CE_{1/2}(q^(alpha) || rho)   >= 0,
        L_B^excess = 0  iff  q = a  AND  rho_j = 1 for all j,
        L_1^excess = 0  iff  rho_j = 1 for all j,
    so L_B^excess remains the stricter objective. Also
        Delta_alpha = log(alpha_ar/alpha_geo) = D_1(u_T||a) >= 0
    is the leverage-profile heterogeneity separating the two floors.

    TRAINING still uses L_B; these are post-hoc reporting quantities (alpha_t is classical
    data, so the shifts are parameter-independent for a fixed embedding)."""
    p_ = f ** 2
    q = w ** 2 / xp.maximum(xp.sum(w ** 2), 1e-300)
    if alpha_t is None:                      # no leverage cap: rho = p, floors vanish
        at = xp.ones(T)
    else:
        at = xp.clip(xp.asarray(alpha_t), 1e-300, 1.0)
    a_ar = xp.mean(at)
    a_geo = xp.exp(xp.mean(xp.log(at)))
    a = at / (T * a_ar)                                    # capacity profile, sums to 1
    rho = xp.clip(p_ / at, 1e-300, 1.0)                    # directional success in [0,1]
    LB_x = -2 * xp.log(xp.maximum(xp.sum(xp.sqrt(a * q * rho)), 1e-300))
    L1_x = -xp.mean(xp.log(rho))
    D12_aq = -2 * xp.log(xp.maximum(xp.sum(xp.sqrt(a * q)), 1e-300))
    u = xp.ones(T) / T
    D1_ua = xp.sum(u * xp.log(u / xp.maximum(a, 1e-300)))
    D1_uq = xp.sum(u * xp.log(u / xp.maximum(q, 1e-300)))
    return dict(LB_excess=LB_x, L1_excess=L1_x,
                D_half_aq=D12_aq, CE_half_qa_rho=LB_x - D12_aq,
                Delta_alpha=xp.log(a_ar / a_geo), D1_ua=D1_ua, D1_uq=D1_uq,
                alpha_ar=a_ar, alpha_geo=a_geo, rho_mean=xp.mean(rho))


def batch_metrics(Xs, Ys, W, V, kind, k, beta, phi=None, xp=np, alpha_t=None):
    """Mean over sequences of the per-sequence metrics (each sequence = one circuit run).
    alpha_t: None, a length-T vector (shared), or an (n_seq, T) array."""
    T = Xs.shape[1]
    mask = xp.asarray(np.tril(np.ones((T, T))))
    at = alpha_t
    if at is not None:
        at = np.asarray(at)
        if at.ndim == 1:
            at = np.tile(at, (Xs.shape[0], 1))
    out = None
    for si, (Xi, Yi) in enumerate(zip(Xs, Ys)):
        A, w, f, lam = forward(Xi, Yi, W, V, kind, k, beta, mask, xp,
                               alpha_t=(None if at is None else at[si]))
        m = metrics(A, w, f, lam, T, phi, xp,
                    alpha_t=(None if at is None else at[si]))
        out = m if out is None else {kk: out[kk] + m[kk] for kk in out}
    return {kk: v / Xs.shape[0] for kk, v in out.items()}


# --------------------------------------------------------------------------- #
#  6. chance levels                                                            #
# --------------------------------------------------------------------------- #
def chance_L1_analytic(d, complex_tokens=True, alpha_t=None):
    """Model predicting a random direction. With the hybrid readout
       p = alpha_t |<xi_t, zhat>|^2, so the chance level SHIFTS by <-log alpha_t>:
       complex: E[-log p] = <-log alpha> + psi(d) - psi(1)
       real   : E[-log p] = <-log alpha> + psi(d/2) - psi(1/2)"""
    base = float(digamma(d) - digamma(1)) if complex_tokens \
        else float(digamma(d / 2) - digamma(0.5))
    if alpha_t is None:
        return base
    return base + float(np.mean(-np.log(np.maximum(np.asarray(alpha_t), 1e-300))))


def loss_floors(alpha_t):
    """Structural floors induced by the leverage cap p_j <= alpha_{j+1}:
         L1  >= <-log alpha_t>            (per-step cap, Jensen-free)
         L_B >= -log <alpha_t>            (since F <= <alpha_t>, saturated iff
                                           f_j = sqrt(alpha_{j+1}) and w_j ∝ sqrt(alpha_{j+1}))
    Both are 0 only when every target has full leverage alpha = 1."""
    a = np.asarray(alpha_t, dtype=float).ravel()
    return dict(L1_floor=float(np.mean(-np.log(np.maximum(a, 1e-300)))),
                LB_floor=float(-np.log(max(a.mean(), 1e-300))),
                alpha_mean=float(a.mean()))


def chance_level(model, Xs, Ys, n_draws=200, seed=12345, alpha_t=None):
    """Empirical chance: the SAME code path with parameters drawn from the model's own
    initialisation distribution. This is the only chance level that is directly
    comparable to a trained number."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_draws):
        W, V = model.build(model.init(rng))
        rows.append(batch_metrics(Xs, Ys, W, V, model.kernel, model.k, model.beta,
                                  alpha_t=alpha_t))
    return {kk: float(np.median([r[kk] for r in rows])) for kk in rows[0]}


# --------------------------------------------------------------------------- #
#  6b. parameter-count helpers (param-matched QSA ≈ CSA)                       #
# --------------------------------------------------------------------------- #
def csa_n_params(d: int) -> int:
    """Free unitary W=exp(iH) and V: 2 real matrices of size d×d each, times two operators."""
    return 4 * int(d) * int(d)


def qsa_n_params(d: int, layers: int) -> int:
    """Complex RX·RY·RZ + CNOT ansatz for W and V: 2 × L × n_qubits × 3 angles."""
    n = max(1, ceil(log2(d)))
    return 2 * int(layers) * n * 3


def qsa_layers_matching_csa(d: int) -> tuple[int, int]:
    """Choose L so QSA param count is as close as possible to CSA's 4 d^2.

    Exact equality is impossible: QSA count is always a multiple of 6 n_qubits.
    For d=16 (n=4) CSA=1024, nearest is L=43 → 1032.
    """
    n = max(1, ceil(log2(d)))
    target = csa_n_params(d)
    step = 6 * n
    layers = max(1, int(round(target / step)))
    return layers, qsa_n_params(d, layers)


def mu_advantage(d: int, k: int) -> float:
    """Reference threshold k / C(d+k-1, k) used on the μ vs T / μ vs d panels."""
    return float(k / comb(int(d) + int(k) - 1, int(k)))


# --------------------------------------------------------------------------- #
#  6c. trainable embedding helpers (classical discrete tokens)                 #
# --------------------------------------------------------------------------- #
def init_embedding(D: int, d: int, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """Real (D, d) embedding. QR-isometric columns, then row-rescale so max α ≤ 1.

    Starts Parseval-like (sum_l |x><x| ≈ I after QR on columns) with mean α ≈ d/D;
    training is free to move norms (they enter the hybrid loss via α).
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((D, d))
    Q, _ = np.linalg.qr(M)  # orthonormal columns → row norms^2 sum to d
    Xvoc = scale * Q.astype(np.float64)
    return cap_embedding(Xvoc)


def cap_embedding(Xvoc, xp=np):
    """Global rescale so max_l ||x_l||^2 ≤ 1 (valid √α for the R_t control)."""
    norms = xp.sqrt(xp.maximum(xp.sum(xp.abs(Xvoc) ** 2, axis=-1), 0.0))
    m = xp.maximum(xp.max(norms), 1.0)
    return Xvoc / m


def sequences_from_vocab(Xvoc, idx, xp=np):
    """(Xvoc, idx) → normalized (Xi, Yi) and target leverage alpha_t.

    idx : (n_seq, T+1) integer token ids
    Returns Xi, Yi complex (n_seq, T, d) and alpha_t (n_seq, T) with α ≤ 1.
    """
    Xc = cap_embedding(xp.asarray(Xvoc), xp)
    # Real emb → complex tokens (one code path for classical + quantum).
    Xc = Xc.astype(getattr(xp, "complex128", complex))
    ids = xp.asarray(idx)
    Xu = Xc[ids[:, :-1]]
    Yu = Xc[ids[:, 1:]]
    nx = xp.linalg.norm(Xu, axis=-1, keepdims=True)
    ny = xp.linalg.norm(Yu, axis=-1, keepdims=True)
    Xi = Xu / xp.maximum(nx, 1e-12)
    Yi = Yu / xp.maximum(ny, 1e-12)
    alpha_t = xp.real(ny[..., 0] ** 2)
    return Xi, Yi, alpha_t


def pack_params(Xvoc, circuit_v):
    """Flatten trainable emb + circuit into one real vector."""
    return np.concatenate([np.asarray(Xvoc, dtype=np.float64).ravel(),
                           np.asarray(circuit_v, dtype=np.float64).ravel()])


def unpack_params(v, D: int, d: int, n_circuit: int):
    """Split flat vector → (Xvoc (D,d), circuit_v)."""
    n_emb = D * d
    Xvoc = np.asarray(v[:n_emb], dtype=np.float64).reshape(D, d)
    circuit = np.asarray(v[n_emb:n_emb + n_circuit], dtype=np.float64)
    return Xvoc, circuit


# --------------------------------------------------------------------------- #
#  7. models: the ONLY thing that differs between rows of the benchmark        #
# --------------------------------------------------------------------------- #
class Model:
    """A model = (parameter initialiser, parameter->(W,V) map, kernel, training loss)."""

    def __init__(self, name, d, kernel, k, layers=8, loss="L_B", v_isometric=True):
        self.name, self.d, self.kernel, self.k = name, d, kernel, k
        self.loss, self.layers, self.v_isometric = loss, layers, v_isometric
        self.beta = float(np.sqrt(d))
        self.n = max(1, ceil(log2(d)))
        self.ring = _cnot_ring(self.n)
        self.ansatz = name.startswith("kqsa")
        if self.ansatz:
            self.shape = (layers, self.n, 3)
            self.n_par = 2 * int(np.prod(self.shape))
        else:
            self.n_par = 2 * (2 * d * d)

    def init(self, rng):
        if self.ansatz:
            return rng.uniform(0, 2 * np.pi, self.n_par)
        return 0.3 * rng.standard_normal(self.n_par)

    def build(self, v, xp=np):
        d, half = self.d, self.n_par // 2
        if self.ansatz:
            Wp = v[:half].reshape(self.shape)
            Vp = v[half:].reshape(self.shape)
            W = ansatz_unitary(Wp, self.n, self.ring, xp)[:d, :d]
            V = ansatz_unitary(Vp, self.n, self.ring, xp)[:d, :d]
            return W, V
        W = unitary_from_hermitian(v[:half], d, xp)
        V = (unitary_from_hermitian(v[half:], d, xp) if self.v_isometric
             else general_matrix(v[half:], d, xp))
        return W, V

    def n_params(self):
        return self.n_par


def build_model_suite(d, k, layers=8):
    """The six rows of the benchmark."""
    return [
        Model("kqsa-mono", d, "mono", k, layers, loss="L_B"),
        Model("kqsa-poly", d, "poly", k, layers, loss="L_B"),
        Model("kcsa-mono", d, "mono", k, layers, loss="L_B"),
        Model("kcsa-poly", d, "poly", k, layers, loss="L_B"),
        Model("nlcsa-iso", d, "soft", k, layers, loss="L1", v_isometric=True),
        Model("nlcsa-gen", d, "soft", k, layers, loss="L1", v_isometric=False),
    ]


# --------------------------------------------------------------------------- #
#  8. training                                                                 #
# --------------------------------------------------------------------------- #
def train(model, Xs, Ys, maxiter=400, seed=0, n_restarts=1, verbose=True,
          phase_mode="analytic", tol=1e-12, alpha_t=None):
    """Minimise model.loss. Uses JAX analytic gradients when available, else
    scipy L-BFGS-B with finite differences and a budget scaled to the parameter count.

    phase_mode='analytic' substitutes the closed-form optimum phi_j* = -arg A_j
    (phase-alignment theorem); 'free' optimises the T phases explicitly."""
    from scipy.optimize import minimize

    T = Xs.shape[1]
    n_extra = T if (phase_mode == "free" and model.loss == "L_B") else 0

    def loss_np(v):
        W, V = model.build(v[: model.n_par])
        phi = v[model.n_par:] if n_extra else None
        return float(batch_metrics(Xs, Ys, W, V, model.kernel, model.k,
                                   model.beta, phi, alpha_t=alpha_t)[model.loss])

    best = None
    for r in range(n_restarts):
        rng = np.random.default_rng(seed + 1000 * r)
        v0 = model.init(rng)
        if n_extra:
            v0 = np.concatenate([v0, np.zeros(T)])
        hist = []

        if HAS_JAX:
            def loss_jax(v):
                W, V = model.build(v[: model.n_par], xp=jnp)
                phi = v[model.n_par:] if n_extra else None
                return batch_metrics(jnp.asarray(Xs), jnp.asarray(Ys), W, V,
                                     model.kernel, model.k, model.beta, phi,
                                     xp=jnp, alpha_t=alpha_t)[model.loss]
            gfun = jax.jit(jax.grad(loss_jax))
            fun = lambda v: (hist.append(loss_np(v)), loss_np(v))[1]
            res = minimize(fun, v0, jac=lambda v: np.asarray(gfun(jnp.asarray(v))),
                           method="L-BFGS-B",
                           options=dict(maxiter=maxiter, ftol=tol, gtol=tol))
        else:
            nb = len(v0)
            maxfun = max(4000, 40 * nb)           # ~>=40 finite-difference gradient steps
            if nb > 600:
                print(f"    [warn] {nb} params with finite differences: install JAX for "
                      f"analytic gradients, or expect an under-trained fit.")
            fun = lambda v: (hist.append(loss_np(v)), hist[-1])[1]
            res = minimize(fun, v0, method="L-BFGS-B",
                           options=dict(maxiter=100000, maxfun=maxfun, ftol=tol, gtol=tol))

        if best is None or res.fun < best[0]:
            best = (res.fun, res.x, np.array(hist))

    fval, vbest, hist = best
    W, V = model.build(vbest[: model.n_par])
    phi = vbest[model.n_par:] if n_extra else None
    fin = batch_metrics(Xs, Ys, W, V, model.kernel, model.k, model.beta, phi,
                        alpha_t=alpha_t)
    fin = {kk: float(vv) for kk, vv in fin.items()}
    if verbose:
        grad_steps = (len(hist) if HAS_JAX else len(hist) // max(len(vbest), 1))
        print(f"    [{model.name:>10}] {model.loss} {hist[0]:8.4f} -> {fval:8.4f}   "
              f"({model.n_params()} params, ~{grad_steps} grad steps"
              f"{', JAX' if HAS_JAX else ', finite-diff'})")
    return dict(W=W, V=V, phi=phi, history=hist, final=fin, model=model)


# --------------------------------------------------------------------------- #
#  9. invariants                                                               #
# --------------------------------------------------------------------------- #
def check_invariants(m, T, label="", tol=1e-8, verbose=True, alpha_mean=1.0):
    """Structural identities that must hold for ANY parameters. A failure here means a
    bug in the observables, not a modelling result."""
    ok = {}
    ok["F in [0, <alpha>]"] = (-tol <= m["F"] <= alpha_mean * (1 + 1e-6) + tol)
    ok["nu <= zeta"] = (m["nu"] <= m["zeta"] * (1 + tol))
    ok["mu <= 2T/(T+1) zeta"] = (m["mu"] <= (2 * T / (T + 1)) * m["zeta"] * (1 + tol))
    ok["L_B = L_A + D_half"] = (abs(m["L_B"] - (m["L_A"] + m["D_half"])) < 1e-9)
    ok["L_A >= 0"] = (m["L_A"] >= -tol)
    ok["D_half >= 0"] = (m["D_half"] >= -1e-9)
    ok["L_half <= L1"] = (m["L_half"] <= m["L1"] + 1e-9)   # CE_alpha(u||.) monotone in alpha
    ok["G in [0,1]"] = (-tol <= m["G"] <= 1 + tol)
    bad = [kk for kk, vv in ok.items() if not vv]
    if verbose and bad:
        print(f"    !! INVARIANT FAILURE {label}: {bad}")
    elif verbose:
        print(f"    invariants OK {label}")
    return len(bad) == 0


# --------------------------------------------------------------------------- #
# 10. benchmark driver                                                         #
# --------------------------------------------------------------------------- #
def run_benchmark(Xs, Ys, k=2, layers=8, maxiter=300, seed=0, n_restarts=1,
                  n_chance=80, title="", complex_tokens=True, Xte=None, Yte=None,
                  alpha=None, idx=None, D=None, alpha_t=None):
    n_seq, T, d = Xs.shape
    print("=" * 92)
    print(f"{title}   T={T}  d={d}  k={k}  n_seq={n_seq}  layers={layers}")
    print("=" * 92)
    ch_an = chance_L1_analytic(d, complex_tokens, alpha_t)
    fl = loss_floors(alpha_t) if alpha_t is not None else None
    print(f"  analytic chance for L1 (random direction, "
          f"{'complex' if complex_tokens else 'real'} d={d}) = {ch_an:.4f}")
    mk = comb(d + k - 1, k)
    print(f"  advantage threshold  k/min(T,m_k) = {k}/min({T},{mk}) = {k/min(T,mk):.5f}")
    if fl is not None:
        print(f"  HYBRID readout: p_j <= alpha_(j+1)  =>  L1 floor = {fl['L1_floor']:.4f},"
              f"  L_B floor = -log<alpha> = {fl['LB_floor']:.4f}   (<alpha>={fl['alpha_mean']:.5f})")
        print(f"  single R_t control: mu carries ONE factor alpha (target only), not alpha^(2k+2)")
    if alpha is not None:
        Pflag, am_occ = flag_success_probability(alpha, idx, k)
        am_all = float(np.mean(alpha))
        Deff = float(1.0 / np.sum((alpha / alpha.sum()) ** 2))
        print(f"  UNNORMALIZED tokens: alpha in [{alpha.min():.4f},{alpha.max():.4f}], "
              f"sum={alpha.sum():.3f} (=d), mean={am_all:.5f}"
              + (f" (=d/D={d/D:.5f})" if D else ""))
        print(f"  mean alpha over OCCURRING tokens = {am_occ:.5f}   "
              f"effective vocabulary D_eff = {Deff:.1f}"
              + (f" of D={D}" if D else ""))
        print(f"  filter success prob. P_p ~ <alpha>^(2p+2): " +
              "  ".join(f"p={p}: {v:.2e}" for p, v in Pflag.items()))
        print(f"  -> the flags cost a factor ~1/P_k = {1/max(Pflag[k],1e-300):.2e} in shots; "
              f"this is already inside mu (mu is the JOINT all-zero probability).\n")
    else:
        print()

    rows = []
    for model in build_model_suite(d, k, layers):
        res = train(model, Xs, Ys, maxiter=maxiter, seed=seed,
                    n_restarts=n_restarts, verbose=True, alpha_t=alpha_t)
        fin = res["final"]
        check_invariants(fin, T, label=f"({model.name})",
                         alpha_mean=(fl["alpha_mean"] if fl else 1.0))
        ch = chance_level(model, Xs, Ys, n_draws=n_chance, alpha_t=alpha_t)
        row = dict(name=model.name, params=model.n_params(),
                   L1=fin["L1"], L1_chance=ch["L1"], gain=ch["L1"] - fin["L1"],
                   L_half=fin["L_half"], L_B=fin["L_B"], L_A=fin["L_A"],
                   D_half=fin["D_half"], mu=fin["mu"], G=fin["G"],
                   LB_excess=fin["LB_excess"], L1_excess=fin["L1_excess"],
                   D_half_aq=fin["D_half_aq"], Delta_alpha=fin["Delta_alpha"],
                   rho_mean=fin["rho_mean"])
        if Xte is not None:
            te = batch_metrics(Xte, Yte, res["W"], res["V"], model.kernel,
                               model.k, model.beta, res["phi"], alpha_t=alpha_t)
            row["L1_test"] = float(te["L1"])
        rows.append(row)

    print("\n  " + "-" * 88)
    hdr = (f"  {'model':>10} {'par':>5} | {'L1':>7} {'chance':>7} {'gain':>7} {'learns?':>8} | "
           f"{'L_1/2':>7} {'L_B':>7} {'D_1/2':>7} {'mu':>9}")
    print(hdr)
    print("  " + "-" * 88)
    for r in rows:
        verdict = "YES" if r["gain"] > 0.10 else ("marginal" if r["gain"] > 0.02 else "NO")
        te = f" test {r['L1_test']:.3f}" if "L1_test" in r else ""
        print(f"  {r['name']:>10} {r['params']:>5} | {r['L1']:>7.3f} {r['L1_chance']:>7.3f} "
              f"{r['gain']:>+7.3f} {verdict:>8} | {r['L_half']:>7.3f} {r['L_B']:>7.3f} "
              f"{r['D_half']:>7.3f} {r['mu']:>9.2e}{te}")
    print("  " + "-" * 88)
    print(f"  {'model':>10} | {'LB_exc':>8} {'L1_exc':>8} | {'D12(a||q)':>10} "
          f"{'CE12(qa||rho)':>14} {'Delta_alpha':>12} {'<rho>':>7}")
    print("  " + "-" * 88)
    for r in rows:
        print(f"  {r['name']:>10} | {r['LB_excess']:>8.4f} {r['L1_excess']:>8.4f} | "
              f"{r['D_half_aq']:>10.4f} {r['LB_excess']-r['D_half_aq']:>14.4f} "
              f"{r['Delta_alpha']:>12.4f} {r['rho_mean']:>7.4f}")
    print("  " + "-" * 88)
    print("  EXCESS losses subtract the architectural capacity floor:")
    print("    L_B^exc = L_B + log alpha_ar = D_1/2(a||q) + CE_1/2(q^(a)||rho)  (0 iff q=a and rho=1)")
    print("    L_1^exc = L_1 + log alpha_geo = CE_1(u||rho)                     (0 iff rho=1)")
    print("  L1 is the COMMON axis (uniform Shannon CE), identical code for all models.")
    print("  gain = own-chance - L1. Any row with gain ~ 0 has NOT learned; ignore its")
    print("  position relative to the others until that is fixed.")
    thr = k / min(T, mk)
    for r in rows:
        if r["name"].startswith(("kqsa", "kcsa")):
            print(f"    {r['name']:>10}: mu={r['mu']:.3e} vs k/min(T,m_k)={thr:.3e} "
                  f"-> margin {r['mu']/thr:.2f}x  {'PASS' if r['mu']>=thr else 'fail'}")
    return rows




# --------------------------------------------------------------------------- #
# 10b. mu sweep -- for re-plotting mu after the readout change                  #
# --------------------------------------------------------------------------- #
def mu_sweep(Xs, Ys, ks=(1, 2, 3), models=("kqsa-mono", "kqsa-poly", "kcsa-mono", "kcsa-poly"),
             layers=8, maxiter=200, seed=0, alpha_t=None, verbose=True, csv_path=None):
    """Train each model at each k and return the trained mu (and the advantage threshold),
    ready for plotting. mu has CHANGED relative to the fully-normalized readout: it now
    carries the single target-leverage factor alpha_{j+1} from the R_t control."""
    n_seq, T, d = Xs.shape
    rows = []
    for k in ks:
        thr = k / min(T, comb(d + k - 1, k))
        suite = {m.name: m for m in build_model_suite(d, k, layers)}
        for name in models:
            m = suite[name]
            r = train(m, Xs, Ys, maxiter=maxiter, seed=seed, verbose=False, alpha_t=alpha_t)
            f = r["final"]
            rows.append(dict(model=name, k=k, mu=f["mu"], zeta=f["zeta"], nu=f["nu"],
                             F=f["F"], L_B=f["L_B"], L1=f["L1"],
                             LB_excess=f["LB_excess"], L1_excess=f["L1_excess"],
                             D_half_aq=f["D_half_aq"], Delta_alpha=f["Delta_alpha"],
                             rho_mean=f["rho_mean"],
                             threshold=thr, margin=f["mu"] / thr))
            if verbose:
                print(f"    k={k} {name:>10}: mu={f['mu']:.4e}  margin={f['mu']/thr:6.2f}x  "
                      f"L_B={f['L_B']:.4f} (exc {f['LB_excess']:.4f})  "
                      f"L1={f['L1']:.4f} (exc {f['L1_excess']:.4f})")
    if csv_path:
        import csv
        with open(csv_path, "w", newline="") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows)
        if verbose:
            print(f"    -> wrote {csv_path}")
    return rows

# --------------------------------------------------------------------------- #
# 11. self-test                                                                #
# --------------------------------------------------------------------------- #
def selftest():
    print("SELF-TEST: identities and invariants on random parameters\n")
    rng = np.random.default_rng(0)
    allok = True
    for (T, d, k, kind) in [(10, 4, 1, "mono"), (12, 8, 2, "poly"),
                            (8, 8, 3, "mono"), (10, 16, 2, "poly"), (9, 8, 2, "soft")]:
        X, Y, _ = quantum_sequences(T, nq=int(log2(d)), n_seq=2, seed=1)
        m = Model(f"kcsa-{kind}" if kind != "soft" else "nlcsa-iso", d, kind, k)
        W, V = m.build(m.init(rng))
        mm = batch_metrics(X, Y, W, V, kind, k, m.beta)
        mm = {kk: float(v) for kk, v in mm.items()}
        ok = check_invariants(mm, T, label=f"(T={T} d={d} k={k} {kind})", verbose=False)
        # analytic-vs-free phase: phi* must beat any random phi
        A, w, f, lam = forward(X[0], Y[0], W, V, kind, k, m.beta,
                               np.tril(np.ones((T, T))))
        mu_star = float(observables(A, w, lam, T, None)[0])
        mu_rand = max(float(observables(A, w, lam, T,
                                        rng.uniform(0, 2 * np.pi, T))[0])
                      for _ in range(40))
        ok_phase = mu_star >= mu_rand - 1e-12
        allok &= ok and ok_phase
        print(f"  T={T:>3} d={d:>3} k={k} {kind:>5}: invariants={'OK' if ok else 'FAIL'}  "
              f"phase-alignment optimal={'OK' if ok_phase else 'FAIL'}  "
              f"(mu*={mu_star:.3e} >= max mu_rand={mu_rand:.3e})")
    print("\n  HYBRID readout (z from normalized xi, prediction via POVM on unnormalized x):")
    for (dd, DD, TT) in ((4, 32, 8), (8, 64, 10), (8, 512, 10)):
        Xv, al = anti_embedding_tokens(DD, dd, seed=0)
        r_ = np.random.default_rng(1); ii = r_.integers(0, DD, TT + 1)
        Xi = Xv / np.sqrt(al)[:, None]
        Xh, Yh, at = Xi[ii[:-1]][None], Xi[ii[1:]][None], al[ii[1:]]
        mm = Model("kcsa-mono", dd, "mono", 2, layers=4, loss="L_B")
        Wq, Vq = mm.build(mm.init(np.random.default_rng(0)))
        rr = {kk: float(v) for kk, v in
              batch_metrics(Xh, Yh, Wq, Vq, "mono", 2, mm.beta, alpha_t=at).items()}
        flr = loss_floors(at)
        # Parseval: the full next-token distribution must sum to 1
        S_ = Xv.conj().T @ Xv
        okp = np.linalg.norm(S_ - np.eye(dd)) < 1e-10
        okF = rr["F"] <= flr["alpha_mean"] * (1 + 1e-6)
        okL = (rr["L1"] >= flr["L1_floor"] - 1e-9) and (rr["L_B"] >= flr["LB_floor"] - 1e-9)
        allok &= (okp and okF and okL)
        print(f"    d={dd:>3} D={DD:>4}: Parseval={'OK' if okp else 'FAIL'}  "
              f"F={rr['F']:.4f}<=<alpha>={flr['alpha_mean']:.4f} {'OK' if okF else 'FAIL'}  "
              f"floors L1>={flr['L1_floor']:.3f}, L_B>={flr['LB_floor']:.3f} "
              f"{'OK' if okL else 'FAIL'}")
    print(f"\n  ALL SELF-TESTS {'PASSED' if allok else 'FAILED'}")
    return allok


# --------------------------------------------------------------------------- #
def main():
    """The synthetic runs below are DEMOS of the API. For real data use, in your own script:

        import numpy as np, qsa_bench as qb

        Xvoc  = np.load("embedding.npy")        # (D, d) UNNORMALIZED vocabulary  [optional]
        Xtr   = np.load("train_tokens.npy")     # (n_seq, T+1, d) UNNORMALIZED token sequences
        Xte   = np.load("test_tokens.npy")

        rows = qb.run_from_tokens(Xtr, Xseq_test=Xte, Xvoc=Xvoc,
                                  k=2, layers=16, maxiter=500,
                                  ks=(1, 2, 3), mu_csv="mu_sweep.csv")

    or, if you have explicit (context, target) pairs rather than next-token sequences,

        X, Y, alpha_t = qb.dataset_from_tokens(Xctx, Ytgt)      # both (n_seq, T, d)
        qb.run_benchmark(X, Y, k=2, alpha_t=alpha_t, ...)

    Requirements on the tokens: alpha_l = ||x^(l)||^2 <= 1 (so that sqrt(alpha) is a valid
    R_t amplitude) and, for the vocabulary readout to be normalized, sum_l |x><x| = I_d.
    load_vocabulary() checks both and offers whiten=True to enforce the second.
    """
    if "--selftest" in sys.argv:
        selftest()
        return
    selftest()
    print("\n" + "=" * 92)
    print("DEMO on synthetic data. For real tokens see the docstring of main() /")
    print("run_from_tokens(); nothing below is needed for a real run.")
    print("=" * 92)
    k, layers = 2, 6

    Dv, dv, Tv = 64, 4, 10
    Xvoc, alv = anti_embedding_tokens(Dv, dv, seed=5)
    rv = np.random.default_rng(5)
    iv = rv.integers(0, Dv, (2, Tv + 1))
    it = rv.integers(0, Dv, (2, Tv + 1))
    vocab = load_vocabulary(Xvoc)
    Xd, Yd, atd = dataset_from_indices(vocab, iv)
    Xdt, Ydt, _ = dataset_from_indices(vocab, it)
    run_benchmark(Xd, Yd, k=k, layers=layers, maxiter=200,
                  title="DEMO: hybrid readout (xi for z, x for prediction)",
                  complex_tokens=True, Xte=Xdt, Yte=Ydt, alpha=alv, idx=iv, D=Dv,
                  alpha_t=atd)
    print("\n  mu sweep (for the mu plots):")
    mu_sweep(Xd, Yd, ks=(1, 2, 3), layers=layers, maxiter=120, alpha_t=atd,
             csv_path="/mnt/user-data/outputs/mu_sweep.csv")


if __name__ == "__main__":
    main()
