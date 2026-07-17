"""
Training pipeline for the QSA Section-2 circuit.

Integrated into jax_training_pipeline.run_training when circuit_mode='section2'.
Uses the classical overlap-interference loss (mu) differentiated through W, V
and the trainable isometric embedding.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from math import log2
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

from config import DATASET_CONFIG, get_training_sentences
from encoding import Encoding
from pennylane_jax_vqt import isometrize_matrix
from qsa_section2_circuit import (
    advantage_threshold,
    extract_ortho_matrix,
    haar_floor,
    qubit_budget,
    real_ortho_block,
    set_backends,
)

jax.config.update("jax_enable_x64", True)


@dataclass
class QSATrainConfig:
    T: int
    d: int
    k: int = 2
    layers: int = 2
    epochs: int = 40
    learning_rate: float = 1e-3
    max_sentences: int = 64
    seed: int = 42
    train_embedding: bool = True
    run_label: Optional[str] = None
    epsilon: float = 1e-12
    output_dir: Optional[str] = None
    warm_start_w_identity: bool = False
    curriculum_k: bool = False
    train_until_converged: bool = False
    max_epochs: int = 300
    loss_rel_tol: float = 1e-4
    convergence_patience: int = 8
    # If set, keep training until loss <= target_loss (or max_epochs). Relative
    # early-stop alone is not enough for large d (can "converge" at loss >> 3).
    target_loss: Optional[float] = None
    # "monomial" = a * s^k (legacy); "poly" = a * sum_p c_p s^p with c_p=beta^p/p!, beta=sqrt(d)
    kernel_mode: str = "poly"
    local_max_qubits: int = 15


def qsa_config_from_dict(cfg: dict) -> QSATrainConfig:
    """Build QSA config from the shared OPTIMIZATION_CONFIG / CLI overrides."""
    raw_target = cfg.get("target_loss", None)
    return QSATrainConfig(
        T=int(DATASET_CONFIG.get("sentence_length", cfg.get("sentence_length", 8))),
        d=int(cfg.get("embedding_dim", 4)),
        k=int(cfg.get("non_linear_order", 2)),
        layers=int(cfg.get("num_layers", 2)),
        epochs=int(cfg.get("epochs", 40)),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        max_sentences=int(DATASET_CONFIG.get("max_sentences", cfg.get("max_sentences", 64))),
        seed=int(cfg.get("seed", 42)),
        train_embedding=bool(cfg.get("train_embedding", True)),
        run_label=cfg.get("run_label"),
        epsilon=float(cfg.get("numerical_epsilon", 1e-12)),
        output_dir=cfg.get("output_dir"),
        warm_start_w_identity=bool(cfg.get("warm_start_w_identity", False)),
        curriculum_k=bool(cfg.get("curriculum_k", False)),
        train_until_converged=bool(cfg.get("train_until_converged", False)),
        max_epochs=int(cfg.get("max_epochs", 300)),
        loss_rel_tol=float(cfg.get("loss_rel_tol", 1e-4)),
        convergence_patience=int(cfg.get("convergence_patience", 8)),
        target_loss=(None if raw_target is None else float(raw_target)),
        kernel_mode=str(cfg.get("kernel_mode", "poly")),
        local_max_qubits=int(cfg.get("local_max_qubits", 15)),
    )


def _feature_qubits(d: int) -> int:
    if d <= 0 or (d & (d - 1)) != 0:
        raise ValueError(f"d must be a power of two, got {d}.")
    return int(round(log2(d)))


def _normalize_rows(vectors: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    norms = jnp.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / jnp.maximum(norms, eps)


def sequence_xy_from_tokens(
    token_ids: jnp.ndarray,
    embedding: jnp.ndarray,
    positional_encoding: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build (X, Y) for Section 2: X[t]=x_t, Y[j]=x_{j+1} (last row duplicated)."""
    x_rows = embedding[token_ids] + positional_encoding
    x_rows = _normalize_rows(x_rows)
    y_rows = jnp.roll(x_rows, shift=-1, axis=0)
    y_rows = y_rows.at[-1].set(x_rows[-1])
    return x_rows, y_rows


def init_qsa_params(
    vocab_size: int,
    cfg: QSATrainConfig,
    key: jax.Array,
) -> Dict[str, jnp.ndarray]:
    n = _feature_qubits(cfg.d)
    key_e, key_w, key_v = jax.random.split(key, 3)
    embedding = jax.random.normal(key_e, (vocab_size, cfg.d), dtype=jnp.float64)
    if cfg.warm_start_w_identity:
        wp = jnp.zeros((cfg.layers, n), dtype=jnp.float64)
    else:
        wp = jax.random.normal(key_w, (cfg.layers, n), dtype=jnp.float64)
    vp = jax.random.normal(key_v, (cfg.layers, n), dtype=jnp.float64)
    embedding = isometrize_matrix(embedding)
    return {
        "embedding": embedding,
        "weights_w": wp,
        "weights_v": vp,
    }


def ortho_matrix_jax(params: jnp.ndarray, d: int) -> jnp.ndarray:
    n = _feature_qubits(d)
    wires = tuple(range(n))

    def template(local_params):
        real_ortho_block(qml, local_params, wires)

    full = qml.matrix(template, wire_order=wires)(jnp.asarray(params, dtype=jnp.float64))
    return jnp.real(full)[:d, :d]


def classical_mu_jax(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    W: jnp.ndarray,
    V: jnp.ndarray,
    k: int,
    kernel_mode: str = "poly",
    beta: Optional[float] = None,
) -> jnp.ndarray:
    """mu = |sum_{i<=j} a_ij * kernel(s_ij)|^2 / norm^2.

    kernel_mode:
      - "monomial": kernel = s^k  (legacy sharpener)
      - "poly":     kernel = sum_{p=0..k} c_p s^p with c_p = beta^p/p!,
                    beta = sqrt(d) by default, divided by lam = sum c_p
                    (matches LCU / softmax truncation in qsa_section2_circuit_polynomial)
    """
    from math import factorial

    # s[j,i] = <x_j|W|x_i>, a[j,i] = <y_j|V|x_i>  (ket on the right; W not transposed)
    s = X @ W @ X.T
    a = Y @ V @ X.T
    mask = jnp.tril(jnp.ones((X.shape[0], X.shape[0]), dtype=jnp.float64))
    ntri = X.shape[0] * (X.shape[0] + 1) / 2.0
    d = X.shape[1]
    if kernel_mode == "monomial":
        w = a * (s ** k)
        S = jnp.sum(w * mask)
        return (S ** 2) / (ntri ** 2)

    # polynomial LCU kernel
    if beta is None:
        beta = jnp.sqrt(jnp.asarray(d, dtype=jnp.float64))
    g = jnp.zeros_like(s)
    s_pow = jnp.ones_like(s)
    lam = jnp.asarray(0.0, dtype=jnp.float64)
    for p in range(k + 1):
        c_p = (beta ** p) / float(factorial(p))
        if p > 0:
            s_pow = s_pow * s
        g = g + c_p * s_pow
        lam = lam + c_p
    w = a * g
    S = jnp.sum(w * mask)
    return (S / (lam * ntri)) ** 2


def _prepare_dataset(cfg: QSATrainConfig) -> Tuple[Encoding, List[str], jnp.ndarray, jnp.ndarray]:
    sentences = get_training_sentences()
    if len(sentences) < cfg.max_sentences:
        expanded = list(sentences)
        while len(expanded) < cfg.max_sentences:
            expanded.extend(sentences)
        sentences = expanded[: cfg.max_sentences]

    encoding = Encoding(sentences, embeddingDim=cfg.d, embeddingSeed=cfg.seed)
    token_batch = jnp.stack([encoding.encode_tokens(s) for s in sentences])
    pe = encoding._positionalEncoding(cfg.T)
    return encoding, sentences, token_batch, pe


def _loss_for_sentence(
    params: Dict[str, jnp.ndarray],
    token_ids: jnp.ndarray,
    positional_encoding: jnp.ndarray,
    cfg: QSATrainConfig,
) -> jnp.ndarray:
    embedding_raw = params["embedding"]
    if not cfg.train_embedding:
        embedding_raw = jax.lax.stop_gradient(embedding_raw)
    embedding = isometrize_matrix(embedding_raw)
    X, Y = sequence_xy_from_tokens(token_ids, embedding, positional_encoding)
    W = ortho_matrix_jax(params["weights_w"], cfg.d)
    V = ortho_matrix_jax(params["weights_v"], cfg.d)
    mu = classical_mu_jax(X, Y, W, V, cfg.k, kernel_mode=cfg.kernel_mode)
    return -jnp.log(jnp.maximum(mu, cfg.epsilon))


def build_train_step(cfg: QSATrainConfig):
    def loss_batch(params, token_batch, positional_encoding):
        per_sentence = jax.vmap(
            lambda ids: _loss_for_sentence(params, ids, positional_encoding, cfg),
            in_axes=(0,),
        )(token_batch)
        return jnp.mean(per_sentence)

    @jax.jit
    def train_step(params, token_batch, positional_encoding):
        loss, grads = jax.value_and_grad(loss_batch)(params, token_batch, positional_encoding)
        params = jax.tree_util.tree_map(
            lambda p, g: p - cfg.learning_rate * g,
            params,
            grads,
        )
        return params, loss

    return train_step, loss_batch


def _k_schedule(cfg: QSATrainConfig) -> List[int]:
    if cfg.curriculum_k and cfg.k > 1:
        return list(range(1, cfg.k + 1))
    return [cfg.k]


def _run_training_phase(
    params: Dict[str, jnp.ndarray],
    token_batch: jnp.ndarray,
    positional_encoding: jnp.ndarray,
    cfg: QSATrainConfig,
    log: logging.Logger,
) -> Tuple[Dict[str, jnp.ndarray], List[float], int, bool]:
    train_step, _ = build_train_step(cfg)
    losses: List[float] = []
    stagnant = 0
    max_epochs = cfg.max_epochs if cfg.train_until_converged else cfg.epochs
    log_frequency = max(1, max_epochs // 10)

    for epoch in range(1, max_epochs + 1):
        params, loss = train_step(params, token_batch, positional_encoding)
        losses.append(float(loss))
        if epoch == 1 or epoch % log_frequency == 0 or epoch == max_epochs:
            log.info(f"[QSA] k={cfg.k} epoch={epoch:03d}/{max_epochs} loss={float(loss):.6f}")

        hit_target = cfg.target_loss is not None and losses[-1] <= cfg.target_loss
        if hit_target:
            log.info(
                f"[QSA] target loss reached at epoch={epoch}: "
                f"{losses[-1]:.6f} <= {cfg.target_loss}"
            )
            return params, losses, epoch, True

        if cfg.train_until_converged and epoch > 1:
            # Do not early-stop on flat loss while still above target_loss.
            if cfg.target_loss is not None and losses[-1] > cfg.target_loss:
                stagnant = 0
                continue
            rel_change = abs(losses[-2] - losses[-1]) / max(abs(losses[-2]), 1e-12)
            if rel_change < cfg.loss_rel_tol:
                stagnant += 1
            else:
                stagnant = 0
            if stagnant >= cfg.convergence_patience:
                log.info(
                    f"[QSA] converged at epoch={epoch} "
                    f"(rel_change<{cfg.loss_rel_tol} for {cfg.convergence_patience} steps)"
                )
                return params, losses, epoch, True

    converged = (
        (cfg.target_loss is not None and losses[-1] <= cfg.target_loss)
        or (cfg.train_until_converged and stagnant >= cfg.convergence_patience)
    )
    return params, losses, max_epochs, converged


def evaluate_mean_O(
    params: Dict[str, jnp.ndarray],
    token_batch: jnp.ndarray,
    positional_encoding: jnp.ndarray,
    cfg: QSATrainConfig,
) -> Tuple[float, float, float]:
    """Return (mean_O_ij, obar/rbar, root) averaged over the training batch."""
    embedding = np.asarray(isometrize_matrix(params["embedding"]))
    qml_backend, _, _ = set_backends()
    W = np.asarray(extract_ortho_matrix(qml_backend, params["weights_w"], cfg.d))
    V = np.asarray(extract_ortho_matrix(qml_backend, params["weights_v"], cfg.d))

    mean_os, obars, obar_roots = [], [], []
    for ids in np.asarray(token_batch):
        X, Y = sequence_xy_from_tokens(
            jnp.asarray(ids), jnp.asarray(embedding), positional_encoding
        )
        X_np = np.asarray(X)
        Y_np = np.asarray(Y)
        if cfg.kernel_mode == "poly":
            from qsa_section2_circuit_polynomial import classical_report as poly_report

            rep = poly_report(X_np, Y_np, W, V, cfg.k)
            # rbar is the poly analogue of mean |a g(s)| (signed-sum mu is separate)
            mean_os.append(float(rep["rbar"]))
            obars.append(float(rep["rbar"]))
            obar_roots.append(float(rep["rbar"]))
        else:
            from qsa_section2_circuit import classical_report

            rep = classical_report(X_np, Y_np, W, V, cfg.k)
            mean_os.append(rep["mean_O_ij"])
            obars.append(rep["obar"])
            obar_roots.append(rep["obar_k_root"])
    return (
        float(np.mean(mean_os)) if mean_os else 0.0,
        float(np.mean(obars)) if obars else 0.0,
        float(np.mean(obar_roots)) if obar_roots else 0.0,
    )


def _prepare_output_dirs(timestamp: str, seed: int, run_label: Optional[str], override: Optional[Path] = None):
    if override is not None:
        run_dir = Path(override)
    else:
        label = run_label or "unlabeled"
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label))
        run_dir = Path("results") / f"run_{timestamp}_seed{seed}_{safe_label}"
    matrices_dir = run_dir / "matrices"
    parameters_dir = run_dir / "parameters"
    summaries_dir = run_dir / "summaries"
    plots_dir = run_dir / "plots"
    for directory in (run_dir, matrices_dir, parameters_dir, summaries_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return run_dir, matrices_dir, parameters_dir, summaries_dir, plots_dir


def _save_qsa_artifacts(
    params: Dict[str, jnp.ndarray],
    cfg: QSATrainConfig,
    result: dict,
    sentences: List[str],
    run_dir: Path,
    matrices_dir: Path,
    parameters_dir: Path,
    summaries_dir: Path,
) -> None:
    qml_backend, _, _ = set_backends()
    W = extract_ortho_matrix(qml_backend, params["weights_w"], cfg.d)
    V = extract_ortho_matrix(qml_backend, params["weights_v"], cfg.d)
    embedding = np.asarray(isometrize_matrix(params["embedding"]))

    np.save(matrices_dir / "W_matrix.npy", W)
    np.save(matrices_dir / "V_matrix.npy", V)
    np.save(matrices_dir / "E_matrix.npy", embedding)
    np.save(matrices_dir / "weights_w.npy", np.asarray(params["weights_w"]))
    np.save(matrices_dir / "weights_v.npy", np.asarray(params["weights_v"]))
    np.save(parameters_dir / "theta_finali_native.npy", np.asarray(params["weights_w"]))

    metadata = {
        "circuit_mode": "section2",
        "run_id": run_dir.name,
        "timestamp": datetime.now().isoformat(),
        "T": cfg.T,
        "d": cfg.d,
        "k": cfg.k,
        "n_qubits": result["n_qubits"],
        "mean_O_ij": result["mean_O_ij"],
        "obar": result["obar"],
        "reference_haar": result["reference_haar"],
        "reference_advantage": result["reference_advantage"],
        "num_sentences": result["num_sentences"],
        "sentences": sentences,
    }
    np.save(matrices_dir / "metadata.npy", metadata)
    (summaries_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (summaries_dir / "optimization_loss_history.txt").write_text(
        "\n".join(f"{v:.8f}" for v in result["loss_history"]),
        encoding="utf-8",
    )


def train_qsa(
    cfg: QSATrainConfig,
    output_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, object]:
    log = logger or logging.getLogger("vqt_jax")

    if qubit_budget(cfg.T, cfg.d, cfg.k) > cfg.local_max_qubits:
        raise ValueError(
            f"n_qubits={qubit_budget(cfg.T, cfg.d, cfg.k)} exceeds local limit {cfg.local_max_qubits} "
            f"for T={cfg.T}, d={cfg.d}, k={cfg.k}."
        )

    set_backends()
    encoding, sentences, token_batch, positional_encoding = _prepare_dataset(cfg)
    key = jax.random.PRNGKey(cfg.seed)
    params = init_qsa_params(encoding.vocabSize, cfg, key)

    all_losses: List[float] = []
    phase_summaries: List[dict] = []
    start = time.perf_counter()
    k_values = _k_schedule(cfg)

    log.info(
        f"[QSA] Section-2 training | T={cfg.T} d={cfg.d} k={cfg.k} "
        f"n_qubits={qubit_budget(cfg.T, cfg.d, cfg.k)} "
        f"epochs={cfg.epochs if not cfg.train_until_converged else f'<= {cfg.max_epochs} (converge)'} "
        f"target_loss={cfg.target_loss} "
        f"curriculum_k={cfg.curriculum_k} warm_start_W={cfg.warm_start_w_identity}"
    )

    for phase_k in k_values:
        phase_cfg = QSATrainConfig(**{**asdict(cfg), "k": phase_k})
        if qubit_budget(phase_cfg.T, phase_cfg.d, phase_cfg.k) > cfg.local_max_qubits:
            raise ValueError(
                f"n_qubits={qubit_budget(phase_cfg.T, phase_cfg.d, phase_cfg.k)} exceeds local limit "
                f"{cfg.local_max_qubits} for T={phase_cfg.T}, d={phase_cfg.d}, k={phase_cfg.k}."
            )
        if phase_k != cfg.k:
            log.info(f"[QSA] curriculum: starting phase k={phase_k} (target k={cfg.k})")
        params, phase_losses, epochs_run, converged = _run_training_phase(
            params, token_batch, positional_encoding, phase_cfg, log
        )
        all_losses.extend(phase_losses)
        phase_summaries.append(
            {"k": phase_k, "epochs_run": epochs_run, "final_loss": phase_losses[-1], "converged": converged}
        )

    elapsed = time.perf_counter() - start
    mean_o, mean_obar, mean_obar_k = evaluate_mean_O(params, token_batch, positional_encoding, cfg)

    result = {
        "config": asdict(cfg),
        "circuit_mode": "section2",
        "mean_O_ij": mean_o,
        "obar": mean_obar,
        "obar_k_root": mean_obar_k,
        "final_loss": all_losses[-1] if all_losses else float("nan"),
        "loss_history": all_losses,
        "training_phases": phase_summaries,
        "converged": all(s.get("converged", False) for s in phase_summaries) if cfg.train_until_converged else None,
        "elapsed_seconds": elapsed,
        "n_qubits": qubit_budget(cfg.T, cfg.d, cfg.k),
        "num_sentences": int(token_batch.shape[0]),
        "reference_haar": haar_floor(cfg.d, cfg.k),
        "reference_advantage": advantage_threshold(cfg.d, cfg.k),
        "reference_sqrt_k_over_d": float(np.sqrt(cfg.k / cfg.d)),
        "run_dir": str(output_dir) if output_dir is not None else None,
    }

    if output_dir is not None:
        run_dir = Path(output_dir)
        _, matrices_dir, parameters_dir, summaries_dir, _ = _prepare_output_dirs(
            timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            seed=cfg.seed,
            run_label=cfg.run_label,
            override=run_dir,
        )
        _save_qsa_artifacts(params, cfg, result, sentences, run_dir, matrices_dir, parameters_dir, summaries_dir)

    log.info(
        f"[QSA] mean_O_ij={mean_o:.6f} obar={mean_obar:.6f} | "
        f"haar=d^(-(k+1)/2)={result['reference_haar']:.6f} "
        f"| adv=sqrt(k*k!/d^k)={result['reference_advantage']:.6f}"
    )

    return {"params": params, "token_batch": token_batch, "positional_encoding": positional_encoding, **result}


def run_qsa_training(logger, cfg: dict, comm=None, rank: int = 0, size: int = 1) -> int:
    """Entry point called from jax_training_pipeline.run_training."""
    if rank != 0:
        if comm is not None and hasattr(comm, "Barrier"):
            try:
                comm.Barrier()
            except Exception:
                pass
        return 0

    try:
        qsa_cfg = qsa_config_from_dict(cfg)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = qsa_cfg.run_label or "section2"
        safe_label = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label))
        if qsa_cfg.output_dir:
            run_dir = Path(qsa_cfg.output_dir)
        else:
            run_dir = Path("results") / f"run_{timestamp}_seed{qsa_cfg.seed}_{safe_label}"

        train_qsa(qsa_cfg, output_dir=run_dir, logger=logger)
        logger.info(f"[QSA] Risultati salvati in {run_dir}")
        return 0
    except Exception as exc:
        logger.error(f"[QSA] Errore training Section-2: {exc}")
        raise
    finally:
        if comm is not None and hasattr(comm, "Barrier"):
            try:
                comm.Barrier()
            except Exception:
                pass
