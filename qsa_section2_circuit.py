"""
qsa_section2_circuit.py
=======================
Full two-control-register overlap-interference quantum-attention circuit
(qsa_supplement, Section 2), PennyLane + JAX -- with the fast OVERLAP readout.

Readout twist (the optimization): the loss is
    mu = |A|^2,   A = (1/N^2) sum_{i<=j} a_ij s_ij^k ,   N^2 = T(T+1)/2 .
Instead of applying P^dagger on the controls and measuring the all-zeros projector
(Step 4), run only through Step 3 and read A directly as the overlap of the resulting
state with the reference  P|0>_C (x) |0>_AB.  On a statevector that is just
    A = (1/N) * sum_{i<=j} state[ index(j,i, AB=0) ] ,
i.e. read T^2 specific amplitudes -- no inverse state-prep, no projector expval.
(Validated: overlap readout == projector readout == classical mu, to machine precision.
 Classical weight on branch (j,i): a_ij * s_ij^k with
   s_ij = <x_j|W|x_i> = x_j^T W x_i   (x_i as ket, W not transposed)
   a_ij = <y_j|V|x_i> = y_j^T V x_i .)

Paths, fastest first:
  * classical_report(...)  : O(T^2 d) from W,V matrices. USE THIS for o-bar / mu scans.
  * mu_overlap(state,...)  : the circuit with the fast overlap readout (Steps 1-3 + dot).
                             Use for on-device readout validation / finite-shot & noise
                             studies at T>>d (HPC). Same mu, ~T^2 cheaper than the projector.
  * projector qnode        : the literal Step-1..4 circuit. Reference/validation only.

Local: n_tot = 2*ceil(log2 T) + (k+1)*log2 d <= 15.
Run:  python qsa_section2_circuit.py          # self-check (needs PennyLane+JAX)
      python qsa_section2_circuit.py --oracle  # classical-only (numpy)
      python qsa_section2_circuit.py --leonardo
Assumes d is a power of 2 (n = log2 d exactly): d in {2,4,8,16,32,...}.
"""
from __future__ import annotations
import sys
from math import comb, ceil, log2
import numpy as np

# --------------------------------------------------------------------------- #
#  0.  helpers                                                                 #
# --------------------------------------------------------------------------- #
def bits_msb(x, width):
    return [(x >> (width - 1 - b)) & 1 for b in range(width)]

def qubit_budget(T, d, k):
    cT = max(1, ceil(log2(max(T, 2)))); n = max(1, ceil(log2(d)))
    return 2 * cT + (k + 1) * n

def _layout(T, d, k):
    cT = max(1, ceil(log2(max(T, 2)))); n = max(1, ceil(log2(d)))
    C1 = list(range(0, cT)); C2 = list(range(cT, 2 * cT))
    A = list(range(2 * cT, 2 * cT + n))
    B = [list(range(2 * cT + n + c * n, 2 * cT + n + (c + 1) * n)) for c in range(k)]
    return cT, n, C1, C2, A, B, 2 * cT + (k + 1) * n

def triangular_state(T, cT):
    dimC = 2 ** cT; v = np.zeros(dimC * dimC, dtype=float); N = np.sqrt(T * (T + 1) / 2)
    for j in range(T):
        for i in range(j + 1):
            v[j * dimC + i] = 1.0 / N
    return v

def overlap_indices(T, d, k):
    """Flat indices of state[j,i,AB=0] for i<=j, plus N=sqrt(Ntri). Wire order
    [C1(cT),C2(cT),A(n),B_1..B_k(n)], MSB=wire 0."""
    cT = max(1, ceil(log2(max(T, 2)))); n = max(1, ceil(log2(d)))
    block = 2 ** ((k + 1) * n); cdim = 2 ** cT
    ids = np.array([block * (j * cdim + i) for j in range(T) for i in range(j + 1)], dtype=np.int64)
    return ids, np.sqrt(T * (T + 1) / 2)

# --------------------------------------------------------------------------- #
#  backends                                                                    #
# --------------------------------------------------------------------------- #
_BK = {}
def set_backends():
    import pennylane as qml, jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _BK.update(qml=qml, jax=jax, jnp=jnp)
    return qml, jax, jnp

# --------------------------------------------------------------------------- #
#  classical path (fastest; the default for o-bar / mu scans)                  #
# --------------------------------------------------------------------------- #
def _pair_weights(X, Y, Wmat, Vmat, k):
    """Branch (j,i): w[j,i] = a_ij * s_ij^k.

    Convention (matches circuit + numpy emulator; W orthogonal, not necessarily
    symmetric; ket x_i on the RIGHT):
      s[j,i] = <x_j|W|x_i> = x_j^T W x_i   = (X @ W @ X.T)[j,i]
      a[j,i] = <y_j|V|x_i> = y_j^T V x_i   = (Y @ V @ X.T)[j,i]
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Wmat = np.asarray(Wmat, dtype=np.float64)
    Vmat = np.asarray(Vmat, dtype=np.float64)
    s = X @ Wmat @ X.T
    a = Y @ Vmat @ X.T
    return a * (s ** k)


def mean_O_ij(X, Y, Wmat, Vmat, k):
    """Mean of O_ij = |(<x_j|W|x_i>)^k * <y_j|V|x_i>| over i <= j (y_j = x_{j+1})."""
    w = _pair_weights(X, Y, Wmat, Vmat, k)
    T = w.shape[0]
    mask = np.tril(np.ones((T, T), dtype=np.float64))
    denom = T * (T + 1) / 2
    return float(np.sum(np.abs(w) * mask) / denom)


def haar_floor(d: int, k: int) -> float:
    """Haar reference for obar: d^{-(k+1)/2}."""
    return float(d ** (-(k + 1) / 2))


def advantage_threshold(d: int, k: int) -> float:
    """Advantage threshold for obar: sqrt(k * k! / d^k)."""
    from math import factorial

    return float(np.sqrt(k * factorial(k) / (d ** k)))


def classical_report(X, Y, Wmat, Vmat, k):
    T, d = X.shape
    w = _pair_weights(X, Y, Wmat, Vmat, k)
    mask = np.tril(np.ones((T, T), dtype=np.float64))
    w_tri = w * mask
    S = float(np.sum(w_tri))
    absS = float(np.sum(np.abs(w_tri)))
    Ntri = T * (T + 1) / 2
    # obar = mean_{i<=j} |a_ij s_ij^k| = absS/Ntri (same scale as Haar / advantage refs)
    mean_o = float(absS / Ntri) if Ntri > 0 else 0.0
    obar = mean_o
    obar_k_root = mean_o ** (1 / k) if (Ntri > 0 and k > 0) else mean_o
    haar = haar_floor(d, k)
    thr = advantage_threshold(d, k)
    return dict(
        mu=S ** 2 / Ntri ** 2,
        obar=obar,
        obar_k_root=obar_k_root,
        mean_O_ij=mean_o,
        alignment=(abs(S) / absS if absS > 0 else 0.0),
        threshold=thr,
        haar_floor=haar,
        lift_over_haar=obar / haar if haar > 0 else 0.0,
        n_qubits=qubit_budget(T, d, k),
    )


def extract_ortho_matrix(qml, params, d):
    """Real d x d block of the variational orthogonal gate."""
    n = max(1, ceil(log2(d)))
    full = np.real(
        qml.matrix(real_ortho_block, wire_order=range(n))(
            qml,
            np.asarray(params, dtype=np.float64),
            range(n),
        )
    )
    return full[:d, :d]

# --------------------------------------------------------------------------- #
#  real-orthogonal variational block                                          #
# --------------------------------------------------------------------------- #
def real_ortho_block(qml, params, wires):
    L = params.shape[0]
    w = list(wires)
    for l in range(L):
        for q, wire in enumerate(w):
            qml.RY(params[l, q], wires=wire)
        if len(w) >= 2:
            # n=2: single CNOT; n>2: ring. Skip entirely for n=1 (d=2).
            ring = [(w[0], w[1])] if len(w) == 2 else list(zip(w, w[1:] + w[:1]))
            for a, b in ring:
                qml.CNOT(wires=[a, b])

# --------------------------------------------------------------------------- #
#  shared circuit body: Steps 1-3                                             #
# --------------------------------------------------------------------------- #
def _steps_123(qml, T, k, cT, C1, C2, A, B, tri, X, Y, Wp, Vp):
    qml.StatePrep(tri, wires=C1 + C2)                              # Step 1
    for i in range(T):                                            # Step 2: load x_i^{(k+1)}
        def load(vec):
            qml.StatePrep(vec, wires=A)
            for c in range(k):
                qml.StatePrep(vec, wires=B[c])
        qml.ctrl(load, control=C2, control_values=bits_msb(i, cT))(X[i])
    real_ortho_block(qml, Vp, A)                                  # Step 3a: V,W
    for c in range(k):
        real_ortho_block(qml, Wp, B[c])
    for j in range(T):                                           # Step 3b: target uncompute
        def uncompute(yv, xv):
            qml.adjoint(qml.StatePrep)(yv, wires=A)
            for c in range(k):
                qml.adjoint(qml.StatePrep)(xv, wires=B[c])
        qml.ctrl(uncompute, control=C1, control_values=bits_msb(j, cT))(Y[j], X[j])
    # NOTE: swap StatePrep -> qml.MottonenStatePreparation above if ctrl(StatePrep) errors.

# --------------------------------------------------------------------------- #
#  FAST overlap readout: Steps 1-3 -> statevector; A read from T^2 amplitudes  #
# --------------------------------------------------------------------------- #
def make_qsa_state_qnode(T, d, k, layers=2, dev_name="default.qubit", c_dtype=None, mpi=False):
    qml = _BK["qml"]
    cT, n, C1, C2, A, B, ntot = _layout(T, d, k); tri = triangular_state(T, cT)
    dk = dict(wires=ntot)
    if c_dtype is not None: dk["c_dtype"] = c_dtype
    if mpi: dk["mpi"] = True
    dev = qml.device(dev_name, **dk)

    @qml.qnode(dev, interface="jax", diff_method=None)
    def circ(X, Y, Wp, Vp):
        _steps_123(qml, T, k, cT, C1, C2, A, B, tri, X, Y, Wp, Vp)
        return qml.state()                     # NO adjoint-prep, NO projector
    return circ, ntot

def mu_overlap(state, ids, N):
    """mu = |(1/N) sum_{i<=j} state[index(j,i,0_AB)]|^2 . `ids`,`N` from overlap_indices()."""
    jnp = _BK["jnp"]
    A = jnp.sum(jnp.take(state, jnp.asarray(ids))) / N
    return float(jnp.abs(A) ** 2)

# --------------------------------------------------------------------------- #
#  literal projector circuit (Steps 1-4) -- reference/validation only          #
# --------------------------------------------------------------------------- #
def make_qsa_projector_qnode(T, d, k, layers=2, dev_name="default.qubit",
                             c_dtype=None, diff_method="best", shots=None, mpi=False):
    qml = _BK["qml"]
    cT, n, C1, C2, A, B, ntot = _layout(T, d, k); tri = triangular_state(T, cT)
    dk = dict(wires=ntot, shots=shots)
    if c_dtype is not None: dk["c_dtype"] = c_dtype
    if mpi: dk["mpi"] = True
    dev = qml.device(dev_name, **dk)

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def circ(X, Y, Wp, Vp):
        _steps_123(qml, T, k, cT, C1, C2, A, B, tri, X, Y, Wp, Vp)
        qml.adjoint(qml.StatePrep)(tri, wires=C1 + C2)            # Step 4: P^dag
        return qml.expval(qml.Projector(np.zeros(ntot, dtype=int), wires=range(ntot)))
    return circ, ntot

# --------------------------------------------------------------------------- #
#  single-pair on-device overlap (same trick): s_ij = <x_j|W|x_i>              #
# --------------------------------------------------------------------------- #
def make_pair_qnode(d, dev_name="default.qubit"):
    """QNode returning statevector of W|x_i> on one n-qubit register (d=2^n)."""
    qml = _BK["qml"]; n = max(1, ceil(log2(d)))
    dev = qml.device(dev_name, wires=n)
    @qml.qnode(dev, interface="jax", diff_method=None)
    def circ(xi, Wp):
        qml.StatePrep(xi, wires=range(n))
        real_ortho_block(qml, Wp, range(n))
        return qml.state()                     # snapshot, then dot with |x_j> below
    return circ, n

def single_pair_overlap(circ_pair, xi, xj, Wp):
    """s_ij = <x_j|W|x_i> by snapshot + dot (no U_{x_j}^dag). d must be a power of 2."""
    jnp = _BK["jnp"]
    st = circ_pair(jnp.asarray(xi), jnp.asarray(Wp))
    return complex(jnp.vdot(jnp.asarray(xj), st))

# --------------------------------------------------------------------------- #
#  self-check + instances                                                     #
# --------------------------------------------------------------------------- #
def diagnose_analytic_mismatch(T=4, d=4, k=2, layers=2, seed=0, tol=1e-6):
    """Per-pair check: circuit amplitude vs classical w[j,i]/N on i=j and i!=j."""
    qml, _, jnp = _BK["qml"], _BK["jax"], _BK["jnp"]
    X, Y, Wp, Vp = random_instance(T, d, k, layers, seed, aligned_targets=True)
    circ_state, _ = make_qsa_state_qnode(T, d, k, layers)
    ids, N = overlap_indices(T, d, k)
    st = circ_state(jnp.array(X), jnp.array(Y), jnp.array(Wp), jnp.array(Vp))
    mu_ov = mu_overlap(st, ids, N)
    n = max(1, ceil(log2(d)))
    Wmat = np.real(qml.matrix(real_ortho_block, wire_order=range(n))(qml, jnp.array(Wp), range(n)))[:d, :d]
    Vmat = np.real(qml.matrix(real_ortho_block, wire_order=range(n))(qml, jnp.array(Vp), range(n)))[:d, :d]
    rep = classical_report(X, Y, Wmat, Vmat, k)
    w = _pair_weights(X, Y, Wmat, Vmat, k)
    amps = np.array([st[i] for i in ids])
    pairs = [(j, i) for j in range(T) for i in range(j + 1)]
    diag_err, off_err = [], []
    for idx, (j, i) in enumerate(pairs):
        expected = w[j, i] / N
        if abs(expected) < 1e-14:
            continue
        rel = abs(amps[idx] / expected - 1.0)
        (diag_err if i == j else off_err).append(rel)
    return {
        "mu_overlap": float(mu_ov),
        "mu_analytic": float(rep["mu"]),
        "mu_ratio": float(mu_ov / max(rep["mu"], 1e-30)),
        "pairs_i_eq_j_max_rel_err": float(max(diag_err) if diag_err else 0.0),
        "pairs_i_ne_j_max_rel_err": float(max(off_err) if off_err else 0.0),
        "pairs_i_eq_j_ok": bool(diag_err and max(diag_err) < tol),
        "pairs_i_ne_j_ok": bool(off_err and max(off_err) < tol),
        "explanation": (
            "Classical weight on branch (j,i): a_ij * s_ij^k with "
            "s_ij = <x_j|W|x_i> = x_j^T W x_i (ket x_i on the right) and "
            "a_ij = <y_j|V|x_i>. Matches circuit / numpy emulator."
        ),
    }


def random_instance(T, d, k, layers, seed=0, aligned_targets=False):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, d)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    if aligned_targets:
        Y = np.roll(X, shift=-1, axis=0)
        Y[-1] = X[-1]
    else:
        Y = rng.standard_normal((T, d)); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    n = max(1, ceil(log2(d)))
    return X, Y, rng.standard_normal((layers, n)), rng.standard_normal((layers, n))

def self_check(T=4, d=4, k=2, layers=2, seed=0, tol=1e-6, check_projector=True):
    qml, jax, jnp = _BK["qml"], _BK["jax"], _BK["jnp"]
    nb = qubit_budget(T, d, k)
    if nb > 15:
        print(f"[guard] n_qubits={nb} > 15 (local limit); use HPC (see --leonardo).")
    X, Y, Wp, Vp = random_instance(T, d, k, layers, seed, aligned_targets=True)
    circ_state, ntot = make_qsa_state_qnode(T, d, k, layers)
    ids, N = overlap_indices(T, d, k)
    st = circ_state(jnp.array(X), jnp.array(Y), jnp.array(Wp), jnp.array(Vp))
    mu_ov = mu_overlap(st, ids, N)
    n = max(1, ceil(log2(d)))
    Wmat = np.real(qml.matrix(real_ortho_block, wire_order=range(n))(qml, jnp.array(Wp), range(n)))[:d, :d]
    Vmat = np.real(qml.matrix(real_ortho_block, wire_order=range(n))(qml, jnp.array(Vp), range(n)))[:d, :d]
    rep = classical_report(X, Y, Wmat, Vmat, k)
    line = f"[self-check] T={T} d={d} k={k} n_q={ntot}: mu_overlap={mu_ov:.6e}  mu_analytic={rep['mu']:.6e}"
    ok_readout = True
    if check_projector:
        circ_p, _ = make_qsa_projector_qnode(T, d, k, layers)
        mu_p = float(circ_p(jnp.array(X), jnp.array(Y), jnp.array(Wp), jnp.array(Vp)))
        ok_readout = abs(mu_ov - mu_p) < tol
        line += f"  mu_projector={mu_p:.6e}"
    ok_analytic = abs(mu_ov - rep["mu"]) < tol
    if ok_readout and ok_analytic:
        status = "PASS"
    elif ok_readout:
        status = "PASS (readout) - analytic mismatch"
    else:
        status = "FAIL"
    print(line + f"  {status}")
    print(f"             obar={rep['obar']:.3f}  adv={rep['threshold']:.3e}  "
          f"haar={rep['haar_floor']:.3e}  lift={rep['lift_over_haar']:.2f}  align={rep['alignment']:.3f}")
    if not ok_analytic:
        diag = diagnose_analytic_mismatch(T, d, k, layers, seed, tol=tol)
        print(f"             [diag] i=j max rel err={diag['pairs_i_eq_j_max_rel_err']:.2e}  "
              f"i!=j max rel err={diag['pairs_i_ne_j_max_rel_err']:.2e}")
    return ok_readout and ok_analytic

def run_oracle_only():
    print("=== classical-only (numpy) ===")
    for (T, d, k) in [(4, 4, 2), (4, 8, 2), (4, 16, 1), (8, 4, 3)]:
        X, Y, Wp, Vp = random_instance(T, d, k, 2, seed=0)
        rng = np.random.default_rng(1)
        Wmat, _ = np.linalg.qr(rng.standard_normal((d, d))); Vmat, _ = np.linalg.qr(rng.standard_normal((d, d)))
        rep = classical_report(X, Y, Wmat, Vmat, k)
        print(f"  T={T} d={d} k={k}: n_q={rep['n_qubits']:2d} obar={rep['obar']:.3f} "
              f"thr={rep['threshold']:.3e} haar={rep['haar_floor']:.3e} lift={rep['lift_over_haar']:.2f}")

def main():
    if "--oracle" in sys.argv:
        run_oracle_only(); return
    try:
        set_backends()
    except Exception as e:
        print(f"[env] PennyLane/JAX not importable ({e}); classical-only.\n"); run_oracle_only(); return
    print("=== Section-2 circuit self-check (fast overlap readout == projector == analytic) ===")
    self_check(T=2, d=2, k=1)
    self_check(T=4, d=4, k=2)
    self_check(T=8, d=4, k=3)     # 14 qubits, local-OK
    print("\nFor o-bar(d)/mu scans use classical_report (O(T^2 d)). The circuit + mu_overlap is "
          "for on-device readout validation and, on HPC, finite-shot/noise studies at T>>d.")

LEONARDO_GUIDE = r"""
Leonardo Booster node: 1x Xeon 8358 (32c), 512 GB, 4x A100 64 GB, 2x HDR-100 IB. ~3456 nodes.
Statevector mem = (16 or 8)*2^q bytes (complex128/64). Leave ~2-3 qubits headroom.

Reachable q:  single A100 ~30-31 (fp64 fwd) / ~32 (fp32);  1 node (4 GPU, cuStateVec MPI) ~33;
              node CPU (512 GB) ~35;  multi-node MPI +1 qubit per 2x GPUs (64 nodes ~39).
n_tot = 2*ceil(log2 T) + (k+1)*log2 d.
  local (<=15): d=8,k=2,T=4 (13)  -- below advantage regime
  1x A100 (~30): d=16,k=3,T=128 (30)  <-- T=128 >> kd=48 : IN the regime
  1 node (~33):  d=16,k=3,T=256 (32)
  multi-node:    d=32,k=4,T=128 (39)

CODE CHANGES:
  device:  "default.qubit" -> "lightning.gpu"  (make_qsa_state_qnode(..., dev_name=..., c_dtype=np.complex64))
  readout: use mu_overlap (Steps 1-3 + amplitude dot). On single-GPU qml.state() is local, so the
           overlap read is cheap; the projector path (adjoint-prep + Projector) is ~T^2 more work -- avoid.
  MPI:     mpi=True + srun (1 rank/GPU). CAVEAT: with a DISTRIBUTED statevector, the T^2 amplitudes of
           mu_overlap live on different ranks -> gather them via MPI (small, T^2 values) OR prefer
           single-GPU up to ~30 q (already reaches d=16,k=3,T=128). Keep multi-node for pure forward mu.
  grads:   mu_overlap uses qml.state() (diff_method=None). For DEVICE TRAINING use the projector qnode
           with diff_method="adjoint", or differentiate the overlap expression explicitly.
  precision: c_dtype=complex64 -> +~1 qubit, ~2x; watch mu~d^-k underflow at large k.
  env:  module load cuda openmpi ; venv with pennylane-lightning-gpu (built vs cuQuantum), mpi4py, jax.
  split: o-bar(d) scaling is classical -> CPU. GPUs for readout validation + T>>d forward-mu runs.
"""

if __name__ == "__main__":
    if "--leonardo" in sys.argv:
        print(LEONARDO_GUIDE)
    else:
        main()
