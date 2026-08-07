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
def forward(X, Y, W, V, kind, k, beta, mask, xp=np):
    """Returns (A_j, w_j, f_j) for one sequence.
        s_ij = <x_j|W|x_i>,  z_j = sum_{i<=j} K_ij V x_i,
        A_j  = <y_j|z_j>,    w_j = ||z_j||,   f_j = |A_j| / w_j  in [0,1].
    Hermitian inner products throughout (conj on the bra)."""
    S = xp.conj(X) @ (W @ X.T)
    K, lam = kernel_and_lambda(S, kind, k, beta, mask, xp)
    Z = K @ (X @ V.T)                       # rows are z_j
    A = xp.sum(xp.conj(Y) * Z, axis=1)      # <y_j|z_j>
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


def metrics(A, w, f, lam, T, phi=None, xp=np):
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
    return dict(mu=mu, zeta=zeta, nu=nu, F=F, L_B=L_B, L_A=L_A,
                D_half=L_B - L_A, L1=L1, L_half=L_half, G=G)


def batch_metrics(Xs, Ys, W, V, kind, k, beta, phi=None, xp=np):
    """Mean over sequences of the per-sequence metrics (each sequence = one circuit run)."""
    T = Xs.shape[1]
    mask = xp.asarray(np.tril(np.ones((T, T))))
    out = None
    for Xi, Yi in zip(Xs, Ys):
        A, w, f, lam = forward(Xi, Yi, W, V, kind, k, beta, mask, xp)
        m = metrics(A, w, f, lam, T, phi, xp)
        out = m if out is None else {kk: out[kk] + m[kk] for kk in out}
    return {kk: v / Xs.shape[0] for kk, v in out.items()}


# --------------------------------------------------------------------------- #
#  6. chance levels                                                            #
# --------------------------------------------------------------------------- #
def chance_L1_analytic(d, complex_tokens=True):
    """Model predicting a random direction:
       complex: |<y,z>|^2 ~ Beta(1,d-1)      -> E[-log p] = psi(d) - psi(1)
       real   : |<y,z>|^2 ~ Beta(1/2,(d-1)/2)-> E[-log p] = psi(d/2) - psi(1/2)"""
    return float(digamma(d) - digamma(1)) if complex_tokens \
        else float(digamma(d / 2) - digamma(0.5))


def chance_level(model, Xs, Ys, n_draws=200, seed=12345):
    """Empirical chance: the SAME code path with parameters drawn from the model's own
    initialisation distribution. This is the only chance level that is directly
    comparable to a trained number."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_draws):
        W, V = model.build(model.init(rng))
        rows.append(batch_metrics(Xs, Ys, W, V, model.kernel, model.k, model.beta))
    return {kk: float(np.median([r[kk] for r in rows])) for kk in rows[0]}


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
          phase_mode="analytic", tol=1e-12):
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
                                   model.beta, phi)[model.loss])

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
                                     xp=jnp)[model.loss]
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
    fin = batch_metrics(Xs, Ys, W, V, model.kernel, model.k, model.beta, phi)
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
def check_invariants(m, T, label="", tol=1e-8, verbose=True):
    """Structural identities that must hold for ANY parameters. A failure here means a
    bug in the observables, not a modelling result."""
    ok = {}
    ok["F in [0,1]"] = (-tol <= m["F"] <= 1 + tol)
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
                  n_chance=80, title="", complex_tokens=True, Xte=None, Yte=None):
    n_seq, T, d = Xs.shape
    print("=" * 92)
    print(f"{title}   T={T}  d={d}  k={k}  n_seq={n_seq}  layers={layers}")
    print("=" * 92)
    ch_an = chance_L1_analytic(d, complex_tokens)
    print(f"  analytic chance for L1 (random direction, "
          f"{'complex' if complex_tokens else 'real'} d={d}) = {ch_an:.4f}")
    mk = comb(d + k - 1, k)
    print(f"  advantage threshold  k/min(T,m_k) = {k}/min({T},{mk}) = {k/min(T,mk):.5f}\n")

    rows = []
    for model in build_model_suite(d, k, layers):
        res = train(model, Xs, Ys, maxiter=maxiter, seed=seed,
                    n_restarts=n_restarts, verbose=True)
        fin = res["final"]
        check_invariants(fin, T, label=f"({model.name})")
        ch = chance_level(model, Xs, Ys, n_draws=n_chance)
        row = dict(name=model.name, params=model.n_params(),
                   L1=fin["L1"], L1_chance=ch["L1"], gain=ch["L1"] - fin["L1"],
                   L_half=fin["L_half"], L_B=fin["L_B"], L_A=fin["L_A"],
                   D_half=fin["D_half"], mu=fin["mu"], G=fin["G"])
        if Xte is not None:
            te = batch_metrics(Xte, Yte, res["W"], res["V"], model.kernel,
                               model.k, model.beta, res["phi"])
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
    print(f"\n  ALL SELF-TESTS {'PASSED' if allok else 'FAILED'}")
    return allok


# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        description="Clean-slate shared-pipeline benchmark (kQSA / kCSA / nlCSA)."
    )
    p.add_argument("--selftest", action="store_true",
                   help="Run invariants + phase-alignment checks only.")
    p.add_argument("--skip-selftest", action="store_true",
                   help="Skip the self-test before the full benchmark.")
    p.add_argument("--T", type=int, default=16, help="Sequence length (default 16).")
    p.add_argument("--d", type=int, default=8,
                   help="Token dimension (classical). For quantum, use --nq.")
    p.add_argument("--nq", type=int, default=3,
                   help="Number of qubits for TFIM trajectories (d=2^nq).")
    p.add_argument("--k", type=int, default=2, help="Kernel degree (default 2).")
    p.add_argument("--layers", type=int, default=8, help="Ansatz layers for kQSA.")
    p.add_argument("--maxiter", type=int, default=300, help="L-BFGS max iterations.")
    p.add_argument("--n-seq", type=int, default=4, help="Train sequences.")
    p.add_argument("--n-test", type=int, default=4, help="Held-out test sequences.")
    p.add_argument("--n-restarts", type=int, default=1, help="Random restarts per model.")
    p.add_argument("--n-chance", type=int, default=80, help="Draws for empirical chance.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quantum-only", action="store_true")
    p.add_argument("--classical-only", action="store_true")
    p.add_argument("--demo", action="store_true",
                   help="Tiny demo sizes (T=10,d=4) as in the original Claude draft.")
    p.add_argument("--out", type=str, default="",
                   help="Optional JSON path to dump all rows.")
    return p.parse_args(argv)


def main(argv=None):
    import json
    from pathlib import Path

    args = _parse_args(argv)
    if args.selftest:
        selftest()
        return

    if not args.skip_selftest:
        selftest()
        print()

    if args.demo:
        T, d, nq, layers, maxiter, n_seq = 10, 4, 2, 6, 200, 2
    else:
        T, d, nq = args.T, args.d, args.nq
        layers, maxiter, n_seq = args.layers, args.maxiter, args.n_seq
    k = args.k
    all_rows = {}

    if not args.classical_only:
        Xq, Yq, H = quantum_sequences(T=T, nq=nq, dt=0.35, n_seq=n_seq, seed=1)
        Xqt, Yqt, _ = quantum_sequences(T=T, nq=nq, dt=0.35, n_seq=args.n_test,
                                        seed=77, H=H)
        rows_q = run_benchmark(
            Xq, Yq, k=k, layers=layers, maxiter=maxiter, seed=args.seed,
            n_restarts=args.n_restarts, n_chance=args.n_chance,
            title="QUANTUM SEQUENCES (TFIM trajectories)",
            complex_tokens=True, Xte=Xqt, Yte=Yqt,
        )
        all_rows["quantum"] = rows_q

    if not args.quantum_only:
        print()
        d_c = d if not args.demo else 4
        Xc, Yc = classical_sequences(T=T, d=d_c, n_seq=n_seq, seed=2, rho=0.8)
        Xct, Yct = classical_sequences(T=T, d=d_c, n_seq=args.n_test, seed=88, rho=0.8)
        rows_c = run_benchmark(
            Xc, Yc, k=k, layers=layers, maxiter=maxiter, seed=args.seed,
            n_restarts=args.n_restarts, n_chance=args.n_chance,
            title="CLASSICAL SEQUENCES (correlated real tokens, complex ansatz)",
            complex_tokens=False, Xte=Xct, Yte=Yct,
        )
        all_rows["classical"] = rows_c

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "T": T, "d": d, "nq": nq, "k": k, "layers": layers,
                "maxiter": maxiter, "n_seq": n_seq, "n_test": args.n_test,
                "n_restarts": args.n_restarts, "seed": args.seed,
                "HAS_JAX": HAS_JAX, "demo": bool(args.demo),
            },
            "rows": all_rows,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
