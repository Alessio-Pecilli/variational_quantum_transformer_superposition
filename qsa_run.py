"""
qsa_run.py  --  HPC runner for the Section-2 circuit (call from SLURM on Leonardo).

Uses the FAST overlap readout: run Steps 1-3 -> statevector, read mu from T^2 amplitudes
(no adjoint state-prep, no projector). Evaluates mu at (T,d,k) on lightning.gpu and prints
wall-time + the classical o-bar cross-check.

Examples:
  python qsa_run.py --T 128 --d 16 --k 3 --device lightning.gpu --dtype complex128
  srun python qsa_run.py --T 256 --d 16 --k 3 --device lightning.gpu --mpi
"""
import argparse, time, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--device", default="lightning.gpu")
    ap.add_argument("--dtype", default="complex128", choices=["complex64", "complex128"])
    ap.add_argument("--mpi", action="store_true")
    ap.add_argument("--projector", action="store_true", help="use the slow literal circuit (validation)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import qsa_section2_circuit as M
    qml, jax, jnp = M.set_backends()
    jax.config.update("jax_enable_x64", args.dtype == "complex128")

    rank = 0
    if args.mpi:
        try:
            from mpi4py import MPI; rank = MPI.COMM_WORLD.Get_rank()
        except Exception:
            pass

    cdt = np.complex128 if args.dtype == "complex128" else np.complex64
    nb = M.qubit_budget(args.T, args.d, args.k)
    if rank == 0:
        print(f"[cfg] T={args.T} d={args.d} k={args.k} n_qubits={nb} device={args.device} "
              f"dtype={args.dtype} mpi={args.mpi} readout={'projector' if args.projector else 'overlap'} "
              f"(T>kd ? {args.T > args.k*args.d})")

    X, Y, Wp, Vp = M.random_instance(args.T, args.d, args.k, args.layers, args.seed)
    t0 = time.time()
    if args.projector:
        circ, ntot = M.make_qsa_projector_qnode(args.T, args.d, args.k, args.layers,
                                                dev_name=args.device, c_dtype=cdt,
                                                diff_method=None, mpi=args.mpi)
        mu = float(circ(jnp.array(X), jnp.array(Y), jnp.array(Wp), jnp.array(Vp)))
    else:
        circ, ntot = M.make_qsa_state_qnode(args.T, args.d, args.k, args.layers,
                                            dev_name=args.device, c_dtype=cdt, mpi=args.mpi)
        st = circ(jnp.array(X), jnp.array(Y), jnp.array(Wp), jnp.array(Vp))
        ids, N = M.overlap_indices(args.T, args.d, args.k)
        # NOTE (MPI): with a distributed statevector these T^2 amplitudes live on different
        # ranks; gather them (small) before mu_overlap, or run single-GPU up to ~30 qubits.
        mu = M.mu_overlap(st, ids, N)
    dt = time.time() - t0

    if rank == 0:
        n = max(1, int(np.ceil(np.log2(args.d))))
        Wmat = np.real(qml.matrix(M.real_ortho_block, wire_order=range(n))(qml, jnp.array(Wp), range(n)))[:args.d, :args.d]
        Vmat = np.real(qml.matrix(M.real_ortho_block, wire_order=range(n))(qml, jnp.array(Vp), range(n)))[:args.d, :args.d]
        rep = M.classical_report(X, Y, Wmat, Vmat, args.k)
        print(f"[run] mu={mu:.6e}  wall={dt:.2f}s  |  o-bar={rep['obar']:.3f} "
              f"thr adv={rep['threshold']:.3e} lift={rep['lift_over_haar']:.2f} "
              f"shots~1/mu={1/max(mu,1e-300):.1e}  (classical mu={rep['mu']:.6e})")

if __name__ == "__main__":
    main()
